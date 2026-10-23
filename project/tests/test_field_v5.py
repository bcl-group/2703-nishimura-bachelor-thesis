import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import simu_every_para_up_v5 as simulation
from src import AgentV5, FieldV5
from src.factory import get_agent_class, get_field_class
from src.fields.field_v5 import StepResult


class FieldV5Tests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())

    def make_field(self, count=0, **kwargs):
        field = FieldV5(field_size=1, initial_agents=count, initial_food_prob=0,
                        step_food_prob=0, energy_loss_per_step=0, **kwargs)
        for agent in field.agents:
            agent.weights[:] = 0
        return field

    def test_exports_and_validation(self):
        self.assertIs(get_agent_class('AgentV5'), AgentV5)
        self.assertIs(get_field_class('FieldV5'), FieldV5)
        self.assertIsInstance(self.make_field(count=1).agents[0], AgentV5)
        for value in (0, -1, 1.5, 5.0, True, np.bool_(True), '5'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make_field(food_durability=value)
        with self.assertRaises(TypeError):
            self.make_field(food_lifetime_steps=5)

    def test_food_persists_without_consumption(self):
        field = self.make_field()
        field.grid_food[0, 0] = 3
        for _ in range(10):
            self.assertEqual(field.step(), [])
            self.assertEqual(field.grid_food[0, 0], 3)

    def test_consumption_limits_and_rewards(self):
        for population, durability in ((1, 5), (3, 5), (5, 3), (5, 0)):
            with self.subTest(population=population, durability=durability):
                field = self.make_field(count=population, reward_food=2)
                field.grid_food[0, 0] = durability
                field.step()
                eaten = min(population, durability)
                self.assertEqual(sum(agent.energy == 2 for agent in field.agents), eaten)
                self.assertEqual(sum(agent.energy for agent in field.agents), eaten * 2)
                self.assertEqual(field.grid_food[0, 0], durability - eaten)

    def test_lottery_samples_without_replacement_and_can_select_any_agent(self):
        field = self.make_field(count=5)
        field.grid_food[0, 0] = 3
        state = field.snapshot()
        results = [StepResult(i, (0, 0), None, 0) for i in range(5)]
        with patch('src.fields.field_v5.random.sample', return_value=[4, 2, 1]) as sample:
            rewards, consumed = field.resolve_food(state, results)
        sample.assert_called_once_with([0, 1, 2, 3, 4], 3)
        np.testing.assert_array_equal(rewards, [0, 1, 1, 0, 1])
        self.assertEqual(consumed[0, 0], 3)
        self.assertEqual(state.food[0, 0], 3)

    def test_resolution_is_order_independent_across_cells(self):
        field = FieldV5(field_size=2, initial_agents=8, initial_food_prob=0, step_food_prob=0)
        field.grid_food[:] = 1
        state = field.snapshot()
        results = [StepResult(i, (i % 2, 0), None, 0) for i in range(8)]
        random.seed(42)
        forward = field.resolve_food(state, results)
        random.seed(42)
        reverse = field.resolve_food(state, list(reversed(results)))
        for first, second in zip(forward, reverse):
            np.testing.assert_array_equal(first, second)
        self.assertEqual(np.count_nonzero(forward[0]), 2)

    def test_spawn_resets_durability_and_only_next_step_rewards(self):
        field = self.make_field(count=1, food_durability=5)
        field.step_food_prob = 1
        field.step()
        self.assertEqual(field.agents[0].energy, 0)
        self.assertEqual(field.grid_food[0, 0], 5)
        field.grid_food[0, 0] = 2
        field.step()
        self.assertEqual(field.agents[0].energy, 1)
        self.assertEqual(field.grid_food[0, 0], 5)
        field.step_food_prob = 0
        field.step()
        self.assertEqual(field.grid_food[0, 0], 4)

    def test_lottery_precedes_death_and_reproduction(self):
        field = self.make_field(count=2, reward_food=2, death_threshold=-1)
        for agent in field.agents:
            agent.energy = -1
        field.grid_food[0, 0] = 1
        self.assertEqual(field.step(), [1])
        self.assertEqual(len(field.agents), 1)
        self.assertEqual(field.agents[0].energy, 1)
        parent = field.agents[0]
        parent.energy = 99
        field.grid_food[0, 0] = 1
        self.assertEqual(field.step(), [])
        self.assertEqual(len(field.agents), 2)
        self.assertIsInstance(field.agents[1], AgentV5)
        self.assertEqual([agent.age for agent in field.agents], [2, 0])
        self.assertEqual([agent.energy for agent in field.agents], [0, 0])

    def test_transfers_use_snapshot_and_are_conserved(self):
        field = self.make_field(count=2, transfer_amount=3)
        donor, receiver = field.agents
        donor.energy, receiver.energy = 10, -50
        donor.weights[:, 1] = (0, 1)
        self.assertEqual(field.step(), [])
        self.assertEqual([agent.energy for agent in field.agents], [7, -47])
        self.assertEqual(field.altruism_count, 1)

    def test_snapshot_and_individual_calculations_are_pure(self):
        field = self.make_field(count=3)
        field.grid_food[0, 0] = 1
        state = field.snapshot()
        inputs = field.prepare_step_inputs(state)
        forward = [field.compute_agent_step(state, item) for item in inputs]
        reverse = [field.compute_agent_step(state, item) for item in reversed(inputs)]
        self.assertEqual(forward, list(reversed(reverse)))
        self.assertEqual([agent.energy for agent in field.agents], [0, 0, 0])
        field.grid_food[:] = 0
        self.assertEqual(state.food[0, 0], 1)
        for array in (state.positions, state.energies, state.weights, state.food):
            self.assertFalse(array.flags.writeable)

    def test_seeded_multistep_energy_balance(self):
        def run():
            random.seed(73)
            np.random.seed(73)
            field = FieldV5(field_size=4, initial_agents=12, initial_food_prob=0.3,
                            step_food_prob=0.2, food_durability=3,
                            energy_loss_per_step=0.2, reward_food=0.7,
                            death_threshold=-1000000, clone_threshold=1000000)
            for _ in range(100):
                before = field.grid_food.copy()
                energy = sum(agent.energy for agent in field.agents)
                field.step_food_prob = 0
                field.step()
                consumed = int((before - field.grid_food).sum())
                self.assertAlmostEqual(sum(agent.energy for agent in field.agents),
                                       energy - 12 * 0.2 + consumed * 0.7)
                self.assertTrue(np.all(field.grid_food >= 0))
                field.spawn_objects(prob=0.2)
            return [(a.x, a.y, a.energy, a.age) for a in field.agents], field.grid_food.tolist()
        self.assertEqual(run(), run())

    def test_complete_step_is_independent_of_evaluation_order(self):
        field = self.make_field(count=5)
        field.grid_food[0, 0] = 2
        reversed_field = copy.deepcopy(field)
        prepare = FieldV5.prepare_step_inputs
        random.seed(91)
        field.step()
        random.seed(91)
        with patch.object(FieldV5, 'prepare_step_inputs',
                          side_effect=lambda state: list(reversed(prepare(state)))):
            reversed_field.step()
        self.assertEqual([a.energy for a in field.agents],
                         [a.energy for a in reversed_field.agents])
        np.testing.assert_array_equal(field.grid_food, reversed_field.grid_food)

    def test_serial_and_parallel_outputs_match(self):
        config = {
            'experiment_name': 'parallel_check',
            'agent_class': 'AgentV5', 'field_class': 'FieldV5',
            'simulation': {'total_steps': 20, 'eval_start_step': 5,
                           'num_runs': 2, 'base_seed': 42, 'workers': 1},
            'grid_parameters': {'food_durability': [1, 3]},
            'fixed_field_parameters': {'field_size': 2, 'initial_agents': 8,
                                       'initial_food_prob': 1, 'step_food_prob': 0.1},
        }
        with tempfile.TemporaryDirectory() as directory:
            serial, serial_summary = simulation.run_experiment(config, directory)
            config['simulation']['workers'] = 2
            parallel, parallel_summary = simulation.run_experiment(config, directory)
        from pandas.testing import assert_frame_equal
        assert_frame_equal(serial.sort_values(['food_durability', 'run_id']).reset_index(drop=True),
                           parallel.sort_values(['food_durability', 'run_id']).reset_index(drop=True))
        assert_frame_equal(serial_summary, parallel_summary)

    def test_runner_smoke_and_version_guard(self):
        config = json.loads((Path(__file__).resolve().parents[1] / 'configs/exp_v5.json').read_text())
        config['simulation'].update(total_steps=10, eval_start_step=5, workers=1)
        config['grid_parameters']['death_threshold'] = [-100]
        with tempfile.TemporaryDirectory() as directory:
            raw, summary = simulation.run_experiment(config, directory)
            self.assertEqual(len(raw), 1)
            self.assertEqual(len(summary), 1)
            output = Path(directory) / config['experiment_name']
            self.assertEqual({p.name for p in output.iterdir()}, {
                'config_used.json', 'raw_data.csv', 'summary_data.csv',
                'time_series_data.csv', 'genotype_time_series_data.csv'})
            self.assertEqual(json.loads((output / 'config_used.json').read_text()), config)
            invalid = copy.deepcopy(config)
            invalid.update(agent_class='AgentV4', field_class='FieldV4', experiment_name='invalid')
            with self.assertRaisesRegex(ValueError, 'requires FieldV5 and AgentV5'):
                simulation.run_experiment(invalid, directory)
            self.assertFalse((Path(directory) / 'invalid').exists())


if __name__ == '__main__':
    unittest.main()
