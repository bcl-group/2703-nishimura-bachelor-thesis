import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import src
from src import agents, fields
from src.factory import get_agent_class, get_field_class
from scripts import simu_every_para as simulation


class ConfigAndPackageTests(unittest.TestCase):
    def test_all_versions_are_exported_and_construct_matching_agents(self):
        for suffix in ('', 'V2', 'V3'):
            agent = get_agent_class('Agent' + suffix)
            field = get_field_class('Field' + suffix)
            self.assertIs(agent, getattr(src, 'Agent' + suffix))
            self.assertIs(agent, getattr(agents, 'Agent' + suffix))
            self.assertIs(field, getattr(fields, 'Field' + suffix))
            self.assertIsInstance(field(initial_agents=1).agents[0], agent)
        for getter in (get_agent_class, get_field_class):
            for name in ('Missing', '__name__'):
                with self.assertRaises(ValueError):
                    getter(name)

    def test_numeric_ranges_are_exclusive_and_support_decimal_and_descending(self):
        grid = {'x': {'start': 0, 'stop': 0.3, 'step': 0.1}, 'y': [1, 2]}
        combinations = simulation.expand_grid_parameters(grid)
        self.assertEqual(len(combinations), 6)
        self.assertEqual([row['x'] for row in combinations[::2]], [0, 0.1, 0.2])
        self.assertEqual(simulation.expand_grid_parameters({'x': {'start': 3, 'stop': 0, 'step': -1}}),
                         [{'x': 3}, {'x': 2}, {'x': 1}])
        self.assertEqual(simulation.expand_grid_parameters({}), [{}])
        for values in ([], {'start': 0, 'stop': 1, 'step': 0},
                       {'start': 1, 'stop': 0, 'step': 1}, {'start': 1}):
            with self.assertRaises(ValueError):
                simulation.expand_grid_parameters({'x': values})

    def test_invalid_classes_and_arguments_fail_before_creating_output(self):
        base = json.loads(Path('configs/default_config.json').read_text())
        variants = []
        for key, value in [('agent_class', 'Missing'), ('field_class', 'Missing'),
                           ('agent_class', 'AgentV2'), ('experiment_name', '../outside')]:
            variants.append({**base, key: value})
        variants.append({**base, 'grid_parameters': {'transfer_amount': [1]}})
        with tempfile.TemporaryDirectory() as directory:
            for config in variants:
                with self.assertRaises((ValueError, TypeError)):
                    simulation.run_experiment(config, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_short_real_experiments_for_each_version_and_parallel_execution(self):
        base = json.loads(Path('configs/default_config.json').read_text())
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            legacy = Path(directory) / 'legacy.csv'
            legacy.write_text('existing results\n')
            for suffix, workers in [('', 1), ('V2', 2), ('V3', 1)]:
                config = copy.deepcopy(base)
                config.update(experiment_name='short' + suffix,
                              agent_class='Agent' + suffix, field_class='Field' + suffix)
                config['simulation'].update(total_steps=8, eval_start_step=5, num_runs=2, workers=workers)
                config['grid_parameters'] = {'max_transfer': [5, 10]}
                raw, summary = simulation.run_experiment(config, directory)
                output = Path(directory) / config['experiment_name']
                self.assertEqual(len(raw), 4)
                self.assertEqual(len(summary), 2)
                self.assertEqual(summary['num_runs'].tolist(), [2, 2])
                self.assertEqual(json.loads((output / 'config_used.json').read_text()), config)
                self.assertEqual(len(pd.read_csv(output / 'raw_data.csv')), 4)
                self.assertEqual(len(pd.read_csv(output / 'summary_data.csv')), 2)
                self.assertEqual(set(raw['seed']), {42, 43})
            self.assertEqual(legacy.read_text(), 'existing results\n')

    def test_fixed_only_experiment_and_default_run_settings(self):
        config = json.loads(Path('configs/default_config.json').read_text())
        config['grid_parameters'] = {}
        config['simulation'] = {'total_steps': 3, 'eval_start_step': 2}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            raw, summary = simulation.run_experiment(config, directory)
        self.assertEqual(len(raw), 1)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]['num_runs'], 1)
        self.assertEqual(raw.iloc[0]['seed'], 42)


if __name__ == '__main__':
    unittest.main()
