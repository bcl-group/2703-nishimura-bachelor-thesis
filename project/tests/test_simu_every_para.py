import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from scripts import simu_every_para as simulation


class SimulationTests(unittest.TestCase):
    def test_experiment_has_200_trials_with_100000_evaluation_steps(self):
        config = json.loads(Path('configs/exp_v2_grid.json').read_text())
        self.assertEqual(len(simulation.expand_grid_parameters(config['grid_parameters'])), 20)
        self.assertEqual(config['simulation']['num_runs'], 10)
        self.assertEqual(config['simulation']['eval_start_step'], 100001)
        self.assertEqual(config['simulation']['total_steps'], 200000)

    def run_history(self, history):
        field = Mock()
        field.altruism_count = 0

        def step():
            population, transfers = next(history)
            field.agents = [None] * population
            field.altruism_count = len(transfers)
            field.transfer_total = sum(transfers)
            field.transfer_squared_total = sum(value ** 2 for value in transfers)

        field.step.side_effect = step
        constructor = Mock(return_value=field)
        result = simulation.run_single_simulation(
            constructor, {'death_threshold': -70, 'mutation_rate': 0.05, 'max_transfer': 5},
            {'total_steps': 8, 'eval_start_step': 5, 'base_seed': 42}, {}, run_id=2,
        )
        constructor.assert_called_once_with(death_threshold=-70, mutation_rate=0.05, max_transfer=5)
        return result

    def test_period_means_exclude_warmup_and_reset_action_counts(self):
        history = iter([(100, [9] * 5)] * 4 + [(2, [1]), (4, [1, 3]), (6, []), (8, [4])])
        record = self.run_history(history)
        self.assertEqual(record['avg_population'], 5)
        self.assertEqual(record['avg_altruism_count'], 1)
        self.assertEqual(record['avg_transfer'], 2.25)
        self.assertAlmostEqual(record['std_population'], np.std([2, 4, 6, 8], ddof=1))
        self.assertAlmostEqual(record['std_altruism_count'], np.std([1, 2, 0, 1], ddof=1))
        self.assertAlmostEqual(record['std_transfer'], np.std([1, 1, 3, 4], ddof=1))
        self.assertEqual(record['seed'], 42 + 2)
        self.assertEqual(record['run_id'], 2)
        self.assertNotIn('step', record)

    def test_extinction_retains_full_period_denominator_and_last_transfer(self):
        record = self.run_history(iter([(100, [])] * 4 + [(6, [4]), (0, [2])]))
        self.assertEqual(record['avg_population'], 1.5)
        self.assertEqual(record['avg_altruism_count'], 0.5)
        self.assertEqual(record['avg_transfer'], 3)
        self.assertAlmostEqual(record['std_population'], np.std([6, 0, 0, 0], ddof=1))
        self.assertAlmostEqual(record['std_altruism_count'], np.std([1, 1, 0, 0], ddof=1))
        self.assertAlmostEqual(record['std_transfer'], np.sqrt(2))
        early = self.run_history(iter([(0, [])]))
        self.assertEqual(early['avg_population'], 0)
        self.assertEqual(early['avg_altruism_count'], 0)
        self.assertTrue(np.isnan(early['avg_transfer']))
        self.assertEqual(early['std_population'], 0)
        self.assertEqual(early['std_altruism_count'], 0)
        self.assertTrue(np.isnan(early['std_transfer']))

    def test_surviving_run_without_transfers_has_missing_transfer_mean(self):
        record = self.run_history(iter([(2, [])] * 8))
        self.assertEqual(record['avg_population'], 2)
        self.assertEqual(record['avg_altruism_count'], 0)
        self.assertTrue(np.isnan(record['avg_transfer']))
        self.assertEqual(record['std_population'], 0)
        self.assertEqual(record['std_altruism_count'], 0)
        self.assertTrue(np.isnan(record['std_transfer']))

    def test_single_transfer_has_mean_but_no_sample_sd(self):
        record = self.run_history(iter([(2, [])] * 4 + [(2, [3])] + [(2, [])] * 3))
        self.assertEqual(record['avg_transfer'], 3)
        self.assertTrue(np.isnan(record['std_transfer']))

    def test_identical_transfers_in_one_step_have_zero_event_sd(self):
        record = self.run_history(iter([(2, [])] * 4 + [(2, [3, 3])] + [(2, [])] * 3))
        self.assertEqual(record['avg_transfer'], 3)
        self.assertEqual(record['std_transfer'], 0)

    def test_one_evaluation_step_has_no_temporal_sample_sd(self):
        field = Mock(agents=[None], altruism_count=0, transfer_total=0, transfer_squared_total=0)
        record = simulation.run_single_simulation(
            Mock(return_value=field), {},
            {'total_steps': 5, 'eval_start_step': 5, 'base_seed': 42}, {},
        )
        self.assertEqual(record['avg_population'], 1)
        self.assertTrue(np.isnan(record['std_population']))
        self.assertTrue(np.isnan(record['std_altruism_count']))

    def test_summary_weights_run_means_equally_and_uses_sample_sd(self):
        records = pd.DataFrame([
            {'mutation_rate': 0.05, 'max_transfer': 5, 'death_threshold': -70,
             'run_id': 0, 'avg_population': 2, 'avg_altruism_count': 0.25, 'avg_transfer': 2},
            {'mutation_rate': 0.05, 'max_transfer': 5, 'death_threshold': -70,
             'run_id': 1, 'avg_population': 4, 'avg_altruism_count': 0.75, 'avg_transfer': 4},
        ])
        # Within-run SDs must not affect the summary of run means.
        records['std_population'] = [100, 200]
        records['std_altruism_count'] = [300, 400]
        records['std_transfer'] = [500, 600]
        row = simulation.summarize_runs(records, ['mutation_rate', 'max_transfer', 'death_threshold']).iloc[0]
        self.assertEqual(row['avg_population'], 3)
        self.assertAlmostEqual(row['std_population'], np.sqrt(2))
        self.assertEqual(row['avg_within_run_std_population'], 150)
        self.assertEqual(row['avg_altruism_count'], 0.5)
        self.assertAlmostEqual(row['std_altruism_count'], np.std([0.25, 0.75], ddof=1))
        self.assertEqual(row['avg_transfer'], 3)
        self.assertAlmostEqual(row['std_transfer'], np.sqrt(2))
        self.assertNotEqual(row['avg_transfer'], (2 + 3 * 4) / 4)
        self.assertEqual(row['num_runs'], 2)
        self.assertEqual(row['transfer_valid_runs'], 2)

    def test_summary_missing_transfers_uses_only_valid_runs(self):
        for transfers, expected_mean, valid_count in [
            ([2, 4, np.nan], 3, 2), ([2, np.nan, np.nan], 2, 1),
            ([np.nan, np.nan, np.nan], np.nan, 0),
        ]:
            with self.subTest(transfers=transfers):
                records = pd.DataFrame({
                    'mutation_rate': [0] * 3, 'max_transfer': [5] * 3, 'death_threshold': [-70] * 3,
                    'run_id': range(3), 'avg_population': [0, 2, 4],
                    'std_population': [1, 2, 3],
                    'avg_altruism_count': [0, 0, 0], 'std_altruism_count': [0, 0, 0],
                    'avg_transfer': transfers, 'std_transfer': [np.nan] * 3,
                })
                row = simulation.summarize_runs(records, ['mutation_rate', 'max_transfer', 'death_threshold']).iloc[0]
                np.testing.assert_allclose(row['avg_transfer'], expected_mean, equal_nan=True)
                self.assertEqual(row['transfer_valid_runs'], valid_count)
                self.assertEqual(row['num_runs'], 3)
                self.assertEqual(row['avg_population'], 2)
                self.assertEqual(row['std_population'], 2)
                if valid_count < 2:
                    self.assertTrue(np.isnan(row['std_transfer']))
                else:
                    self.assertAlmostEqual(row['std_transfer'], np.sqrt(2))

    def test_paired_initial_conditions_between_mutation_modes(self):
        initial = []

        def make_field(**kwargs):
            initial.append((simulation.random.random(), np.random.random()))
            field = Mock(agents=[], transfer_total=0, transfer_squared_total=0)
            field.step.side_effect = lambda: setattr(field, 'altruism_count', 0)
            return field

        settings = {'total_steps': 8, 'eval_start_step': 5, 'base_seed': 42}
        simulation.run_single_simulation(make_field, {'mutation_rate': 0.05}, settings, {}, run_id=0)
        simulation.run_single_simulation(make_field, {'mutation_rate': 0.0}, settings, {}, run_id=0)
        simulation.run_single_simulation(make_field, {'mutation_rate': 0.0}, settings, {}, run_id=1)
        self.assertEqual(initial[0], initial[1])
        self.assertNotEqual(initial[0], initial[2])

    def test_csv_has_200_run_rows_and_20_condition_rows_without_changing_old_data(self):
        def fake_run(field_class, param_combination, sim_config, fixed_params, run_id):
            dt, rate, mt = (param_combination[key] for key in ("death_threshold", "mutation_rate", "max_transfer"))
            return {
                'mutation_rate': rate, 'max_transfer': mt, 'death_threshold': dt,
                'run_id': run_id, 'seed': 42 + run_id,
                'avg_population': run_id + 1, 'std_population': 100,
                'avg_altruism_count': run_id / 10, 'std_altruism_count': 200,
                'avg_transfer': run_id if run_id else np.nan, 'std_transfer': 300,
            }

        executor = Mock()
        executor.map.side_effect = lambda func, tasks: map(func, tasks)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(simulation, 'DATA_DIR', Path(directory)), \
             patch.object(simulation, 'run_single_simulation', side_effect=fake_run), \
             patch.object(simulation.concurrent.futures, 'ProcessPoolExecutor') as pool, \
             contextlib.redirect_stdout(io.StringIO()):
            legacy = Path(directory) / 'step_comparison.csv'
            legacy.write_text('previous results\n')
            legacy_dir = Path(directory) / 'step_comparison_runs'
            legacy_dir.mkdir()
            legacy_raw = legacy_dir / 'old_run.csv'
            legacy_raw.write_text('old raw data\n')
            pool.return_value.__enter__.return_value = executor
            config = json.loads(Path('configs/exp_v2_grid.json').read_text())
            config['simulation']['workers'] = 2
            config_path = Path(directory) / 'config.json'
            config_path.write_text(json.dumps(config))
            simulation.main(['--config', str(config_path)])
            output_dir = Path(directory) / config['experiment_name']
            self.assertEqual(json.loads((output_dir / 'config_used.json').read_text()), config)
            raw = pd.read_csv(output_dir / 'raw_data.csv')
            summary = pd.read_csv(output_dir / 'summary_data.csv')
            self.assertEqual(legacy.read_text(), 'previous results\n')
            self.assertEqual(legacy_raw.read_text(), 'old raw data\n')
        self.assertEqual(len(raw), 200)
        self.assertEqual(len(summary), 20)
        self.assertIn('avg_transfer', raw.columns)
        self.assertIn('avg_within_run_std_transfer', summary.columns)
        self.assertEqual(len(raw[['mutation_rate', 'max_transfer', 'death_threshold', 'run_id']].drop_duplicates()), 200)
        self.assertEqual(summary['num_runs'].tolist(), [10] * 20)
        self.assertEqual(summary['transfer_valid_runs'].tolist(), [9] * 20)
        np.testing.assert_allclose(summary['avg_population'], 5.5)
        np.testing.assert_allclose(summary['std_population'], np.std(range(1, 11), ddof=1))
        np.testing.assert_allclose(summary['avg_altruism_count'], 0.45)
        np.testing.assert_allclose(summary['std_altruism_count'], np.std(np.arange(10) / 10, ddof=1))
        np.testing.assert_allclose(summary['avg_transfer'], 5)
        np.testing.assert_allclose(summary['std_transfer'], np.std(range(1, 10), ddof=1))
        pool.assert_called_once_with(max_workers=2)

    def test_invalid_settings_fail_before_execution(self):
        with self.assertRaises(ValueError):
            simulation.run_single_simulation(Mock(), {},
                {'total_steps': 8, 'eval_start_step': 0, 'base_seed': 42}, {})
        config = json.loads(Path('configs/default_config.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            for changes in ({'workers': 0}, {'num_runs': 0}, {'eval_start_step': 0}):
                invalid = {**config, 'simulation': {**config['simulation'], **changes}}
                with self.assertRaises(ValueError):
                    simulation.run_experiment(invalid, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
