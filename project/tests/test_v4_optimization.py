import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import simu_every_para_up as simulation
from src import AgentV4, FieldV4


class V4OptimizationTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())

    def make_field(self, **kwargs):
        return FieldV4(initial_agents=0, initial_food_prob=0, step_food_prob=0, **kwargs)

    def test_local_candidates_match_full_scan_without_duplicates(self):
        for size in (1, 2, 3, 10):
            with self.subTest(size=size):
                positions = [(x, y) for x in range(size) for y in range(size)]
                positions.extend([(0, 0), (size - 1, size - 1)])
                grid = [[[] for _ in range(size)] for _ in range(size)]
                for index, (x, y) in enumerate(positions):
                    grid[x][y].append(index)
                for index, position in enumerate(positions):
                    cells = FieldV4._neighbor_cells(position, size)
                    self.assertEqual(len(cells), len(set(cells)))
                    actual = FieldV4._neighbor_indices(grid, cells, index)
                    expected = [
                        other for other, target in enumerate(positions)
                        if other != index and all(
                            min(abs(a - b), size - abs(a - b)) <= 1
                            for a, b in zip(position, target)
                        )
                    ]
                    self.assertEqual(sorted(actual), expected)

    def test_input_choices_receive_exact_local_candidates(self):
        field = self.make_field()
        field.agents = [AgentV4(x, y, 0) for x, y in [(0, 0), (9, 9), (0, 0), (5, 5)]]
        field.grid_food[9, 0] = field.grid_food[5, 5] = 1
        with patch('src.fields.field_v4.random.choice', side_effect=lambda values: values[0]) as choice:
            inputs = field.prepare_step_inputs(field.snapshot())
        self.assertEqual(set(choice.call_args_list[0].args[0]), {1, 2})
        self.assertEqual(choice.call_args_list[1].args[0], [(9, 0)])
        self.assertIsNone(inputs[3].neighbor_index)
        self.assertEqual(inputs[3].food_position, (5, 5))
        self.assertTrue(all(item.neighbor_index != item.agent_index for item in inputs))

    def test_food_helper_wraps_and_preserves_array_return(self):
        field = self.make_field()
        agent = AgentV4(0, 0, 0)
        self.assertIsNone(field.get_nearest_food(agent))
        field.grid_food[9, 9] = 1
        np.testing.assert_array_equal(field.get_nearest_food(agent), [9, 9])

    def test_death_ages_are_final_and_reset_each_step(self):
        field = self.make_field(field_size=1, energy_loss_per_step=1)
        field.agents = [AgentV4(0, 0, 0), AgentV4(0, 0, 0)]
        for agent, age in zip(field.agents, (5, 10)):
            agent.age, agent.energy = age, -49
            agent.weights[:] = 0
        self.assertEqual(field.step(), [6, 11])
        self.assertEqual(field.agents, [])
        self.assertEqual(field.step(), [])

    def test_food_rescue_and_birth_are_not_deaths(self):
        field = self.make_field(field_size=1, energy_loss_per_step=1, reward_food=2, mutation_rate=0)
        field.agents = [AgentV4(0, 0, 0), AgentV4(0, 0, 0)]
        field.agents[0].energy, field.agents[1].energy = -50, 100
        field.grid_food[0, 0] = 5
        self.assertEqual(field.step(), [])
        self.assertEqual(len(field.agents), 3)
        self.assertEqual(field.agents[-1].age, 0)

    def test_runner_uses_returned_ages_and_inclusive_evaluation_boundary(self):
        class DeathField:
            def __init__(self):
                self.agents = [object()]
                self.altruism_count = 0
                self.calls = 0

            def step(self):
                self.calls += 1
                # Agents have no age attribute: aggregation must use the return value.
                return {1: [99], 2: [12, 23], 3: []}[self.calls]

        result = simulation.run_single_simulation(
            DeathField, {}, {'total_steps': 3, 'eval_start_step': 2, 'base_seed': 42}, {},
        )
        self.assertEqual(result['avg_death_age'], 17.5)
        self.assertAlmostEqual(result['std_death_age'], np.std([12, 23], ddof=1))

    def test_runner_handles_extinction_and_no_evaluation_deaths(self):
        for start in (1, 2):
            field = self.make_field(field_size=1, energy_loss_per_step=1, death_threshold=-1)
            field.agents = [AgentV4(0, 0, 0)]
            result = simulation.run_single_simulation(
                lambda: field, {}, {'total_steps': 3, 'eval_start_step': start, 'base_seed': 42}, {},
            )
            self.assertEqual(result['avg_population'], 0)
            self.assertTrue(np.isnan(result['std_death_age']))
            if start == 1:
                self.assertEqual(result['avg_death_age'], 1)
            else:
                self.assertTrue(np.isnan(result['avg_death_age']))

    def test_old_versions_rejected_before_output_creation(self):
        for version in ('V1', 'V2', 'V3'):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                config = {
                    'experiment_name': 'rejected', 'field_class': 'Field' + version,
                    'agent_class': 'Agent' + version,
                    'simulation': {'total_steps': 1, 'eval_start_step': 1},
                    'grid_parameters': {},
                }
                with self.assertRaisesRegex(ValueError, 'requires FieldV4 and AgentV4'):
                    simulation.run_experiment(config, directory)
                self.assertEqual(list(Path(directory).iterdir()), [])
