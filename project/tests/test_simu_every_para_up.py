import contextlib
import inspect
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from scripts import simu_every_para_up as simulation
from src import FieldV4


def make_agent(age=0, genotype='00000000'):
    return SimpleNamespace(age=age, weights=np.array(list(genotype), dtype=int).reshape(2, 4))


class V4SimulationTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())

    def simulate_death_ages(self, eval_start):
        first, second, third = make_agent(0), make_agent(10), make_agent(20)
        child = make_agent()
        field = SimpleNamespace(agents=[first, second, third], altruism_count=0)
        calls = 0

        def step():
            nonlocal calls
            calls += 1
            for agent in field.agents:
                agent.age += 1
            dead = {1: first, 2: second, 3: third}.get(calls)
            field.agents = [agent for agent in field.agents if agent is not dead]
            if calls == 2:
                field.agents.append(child)
            return [int(dead.age)] if dead is not None else []

        field.step = step
        return simulation.run_single_simulation(
            lambda: field, {}, {'total_steps': 4, 'eval_start_step': eval_start, 'base_seed': 42}, {},
        )

    def test_death_age_window_excludes_warmup_and_living_individuals(self):
        result = self.simulate_death_ages(eval_start=2)
        self.assertEqual(result['avg_death_age'], 17.5)
        self.assertAlmostEqual(result['std_death_age'], np.std([12, 23], ddof=1))
        self.assertEqual(result['avg_population'], 4 / 3)
        one = self.simulate_death_ages(eval_start=3)
        self.assertEqual(one['avg_death_age'], 23)
        self.assertTrue(np.isnan(one['std_death_age']))
        zero = self.simulate_death_ages(eval_start=4)
        self.assertTrue(np.isnan(zero['avg_death_age']))
        self.assertTrue(np.isnan(zero['std_death_age']))

    def test_actual_v4_rescue_and_reproduction_are_not_deaths(self):
        field = FieldV4(initial_agents=2, initial_food_prob=0, step_food_prob=0,
                        energy_loss_per_step=1, transfer_amount=3, mutation_rate=0)
        donor, receiver = field.agents
        donor.x = receiver.x = donor.y = receiver.y = 0
        donor.weights[:] = receiver.weights[:] = 0
        donor.weights[1, 1] = 1
        donor.energy, receiver.energy = 105, -50
        result = simulation.run_single_simulation(
            lambda: field, {}, {'total_steps': 1, 'eval_start_step': 1, 'base_seed': 42}, {},
        )
        self.assertEqual(len(field.agents), 3)
        self.assertEqual(receiver.energy, -48)
        self.assertEqual(field.agents[1].age, 0)  # New child of the donor.

    def test_small_serial_parallel_csv_outputs_match_and_preserve_other_data(self):
        config = {
            'experiment_name': 'v4_test', 'agent_class': 'AgentV4', 'field_class': 'FieldV4',
            'simulation': {'total_steps': 1001, 'eval_start_step': 500, 'num_runs': 2, 'base_seed': 43, 'workers': 1},
            'grid_parameters': {'death_threshold': [-0.2, -1000]},
            'fixed_field_parameters': {
                'initial_agents': 3, 'field_size': 3, 'mutation_rate': 0,
                'initial_food_prob': 0.1, 'step_food_prob': 0.01,
                'food_durability': 2, 'clone_threshold': 1000000,
            },
        }
        outputs = []
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            legacy = Path(directory) / 'legacy.csv'
            legacy.write_text('preserve me\n')
            for workers in (1, 2):
                config['simulation']['workers'] = workers
                raw, summary = simulation.run_experiment(config, directory)
                folder = Path(directory) / config['experiment_name']
                self.assertEqual({p.name for p in folder.iterdir()}, {
                    'config_used.json', 'raw_data.csv', 'summary_data.csv'})
                self.assertEqual(len(raw), 4)
                self.assertEqual(len(summary), 2)
                self.assertEqual(json.loads((folder / 'config_used.json').read_text()), config)
                self.assertEqual(len(pd.read_csv(folder / 'raw_data.csv')), 4)
                self.assertEqual(len(pd.read_csv(folder / 'summary_data.csv')), 2)
                outputs.append((raw.sort_values(['death_threshold', 'run_id']).reset_index(drop=True), summary))
            for serial, parallel in zip(*outputs):
                pd.testing.assert_frame_equal(serial, parallel)
            self.assertEqual(legacy.read_text(), 'preserve me\n')

    def test_empty_population_outputs_headers_and_zero_series(self):
        config = {
            'experiment_name': 'empty', 'agent_class': 'AgentV4', 'field_class': 'FieldV4',
            'simulation': {'total_steps': 1000, 'eval_start_step': 500},
            'grid_parameters': {}, 'fixed_field_parameters': {'initial_agents': 0},
        }
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            raw, summary = simulation.run_experiment(config, directory)
            folder = Path(directory) / 'empty'
            self.assertEqual({p.name for p in folder.iterdir()}, {
                'config_used.json', 'raw_data.csv', 'summary_data.csv'})
            self.assertEqual(raw.iloc[0]['avg_population'], 0)
            self.assertTrue(np.isnan(summary.iloc[0]['avg_death_age']))


if __name__ == '__main__':
    unittest.main()
