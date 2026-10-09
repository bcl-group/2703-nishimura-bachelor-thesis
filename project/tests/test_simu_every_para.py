import contextlib
import io
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from scripts import simu_every_para as simulation


class SimulationTests(unittest.TestCase):
    def test_experiment_has_200_trials_with_100000_evaluation_steps(self):
        self.assertEqual(simulation.DEATH_THRESHOLDS, (-70, -55, -40, -25, -10))
        self.assertEqual(simulation.MAX_TRANSFERS, (5, 10))
        self.assertEqual(simulation.MUTATION_RATES, (0.0, 0.05))
        self.assertEqual(simulation.NUM_RUNS, 10)
        self.assertEqual(simulation.EVAL_START_STEP, 100001)
        self.assertEqual(simulation.TOTAL_STEPS, 200000)
        self.assertEqual(len(list(itertools.product(
            simulation.DEATH_THRESHOLDS, simulation.MUTATION_RATES, simulation.MAX_TRANSFERS,
        ))), 20)

    def run_history(self, history):
        field = Mock()
        field.altruism_count = 0

        def step():
            population, transfers = next(history)
            field.agents = [None] * population
            field.altruism_count += len(transfers)
            field.transfer_total = sum(transfers)
            field.transfer_squared_total = sum(value ** 2 for value in transfers)

        field.step.side_effect = step
        with patch.object(simulation, 'TOTAL_STEPS', 8), \
             patch.object(simulation, 'EVAL_START_STEP', 5), \
             patch.object(simulation, 'Field', return_value=field) as constructor:
            result = simulation.run_single_simulation(-70, 0.05, 5, run_id=2)
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
        self.assertEqual(record['seed'], simulation.BASE_SEED + 2)
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
        with patch.object(simulation, 'TOTAL_STEPS', 5), patch.object(simulation, 'EVAL_START_STEP', 5):
            field = Mock(agents=[None], altruism_count=0, transfer_total=0, transfer_squared_total=0)
            with patch.object(simulation, 'Field', return_value=field):
                record = simulation.run_single_simulation(-70, 0, 5)
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
        row = simulation.summarize_runs(records).iloc[0]
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
                    'avg_altruism_count': [0, 0, 0], 'avg_transfer': transfers,
                })
                row = simulation.summarize_runs(records).iloc[0]
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

        with patch.object(simulation, 'Field', side_effect=make_field):
            simulation.run_single_simulation(-70, 0.05, 5, run_id=0)
            simulation.run_single_simulation(-70, 0.0, 5, run_id=0)
            simulation.run_single_simulation(-70, 0.0, 5, run_id=1)
        self.assertEqual(initial[0], initial[1])
        self.assertNotEqual(initial[0], initial[2])

    def test_csv_has_200_run_rows_and_20_condition_rows_without_changing_old_data(self):
        def fake_run(dt, rate, mt, run_id):
            return {
                'mutation_rate': rate, 'max_transfer': mt, 'death_threshold': dt,
                'run_id': run_id, 'seed': simulation.BASE_SEED + run_id,
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
            simulation.main(workers=2)
            raw = pd.read_csv(Path(directory) / 'raw_data.csv')
            summary = pd.read_csv(Path(directory) / 'summary_data.csv')
            self.assertEqual(legacy.read_text(), 'previous results\n')
            self.assertEqual(legacy_raw.read_text(), 'old raw data\n')
        self.assertEqual(len(raw), 200)
        self.assertEqual(len(summary), 20)
        self.assertEqual(list(raw.columns), [
            'mutation_rate', 'max_transfer', 'death_threshold', 'run_id', 'seed',
            'avg_population', 'std_population', 'avg_altruism_count', 'std_altruism_count',
            'avg_transfer', 'std_transfer',
        ])
        self.assertEqual(list(summary.columns), [
            'mutation_rate', 'max_transfer', 'death_threshold', 'num_runs',
            'avg_population', 'std_population', 'avg_within_run_std_population',
            'avg_altruism_count', 'std_altruism_count',
            'avg_transfer', 'std_transfer', 'transfer_valid_runs',
        ])
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
        with patch.object(simulation, 'EVAL_START_STEP', 0):
            with self.assertRaises(ValueError):
                simulation.run_single_simulation(-70, 0, 5)
        with self.assertRaises(ValueError):
            simulation.main(workers=0)


if __name__ == '__main__':
    unittest.main()
