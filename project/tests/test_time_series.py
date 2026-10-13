import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from scripts import simu_every_para as simulation


class TimeSeriesTests(unittest.TestCase):
    def simulate(self, total_steps=2500, extinction_step=None, run_id=0, collect=True):
        field = Mock()
        current_step = 0

        def step():
            nonlocal current_step
            current_step += 1
            field.agents = [None] * (0 if current_step == extinction_step else current_step % 7 + 1)
            field.altruism_count = current_step % 3

        field.step.side_effect = step
        samples = []
        result = simulation.run_single_simulation(
            Mock(return_value=field), {'death_threshold': -10},
            {'total_steps': total_steps, 'eval_start_step': min(1500, total_steps), 'base_seed': 42},
            {}, run_id=run_id, time_series_records=samples if collect else None,
        )
        return result, samples, field.step.call_count

    def test_snapshot_and_interval_sum_include_warmup(self):
        result, samples, calls = self.simulate()
        self.assertEqual(calls, 2500)
        self.assertEqual([r['step'] for r in samples], [1000, 2000])
        self.assertEqual([r['population'] for r in samples], [1000 % 7 + 1, 2000 % 7 + 1])
        self.assertEqual([r['altruism_count'] for r in samples], [sum(i % 3 for i in range(1, 1001)), sum(i % 3 for i in range(1001, 2001))])
        self.assertTrue(all(r['run_id'] == 0 and r['seed'] == 42 and r['death_threshold'] == -10 for r in samples))
        pd.testing.assert_series_equal(pd.Series(result), pd.Series(self.simulate(collect=False)[0]))

    def test_other_runs_and_short_runs_have_no_samples(self):
        self.assertEqual(self.simulate(run_id=1)[1], [])
        self.assertEqual(self.simulate(total_steps=999)[1], [])

    def test_extinction_pads_without_losing_actions(self):
        for extinction in [1, 999, 1000, 1001, 2000]:
            with self.subTest(extinction=extinction):
                result, samples, calls = self.simulate(total_steps=3500, extinction_step=extinction)
                self.assertEqual(calls, extinction)
                self.assertEqual([r['step'] for r in samples], [1000, 2000, 3000])
                for row in samples:
                    start, end = row['step'] - 999, min(row['step'], extinction)
                    self.assertEqual(row['altruism_count'], sum(i % 3 for i in range(start, end + 1)))
                    self.assertEqual(row['population'], 0 if row['step'] >= extinction else row['step'] % 7 + 1)
                pd.testing.assert_series_equal(pd.Series(result), pd.Series(self.simulate(total_steps=3500, extinction_step=extinction, collect=False)[0]))

    def test_csv_serial_parallel_and_empty(self):
        config = {
            'experiment_name': 'time_series_test', 'agent_class': 'AgentV1', 'field_class': 'FieldV1',
            'simulation': {'total_steps': 2500, 'eval_start_step': 1500, 'num_runs': 2, 'base_seed': 42, 'workers': 1},
            'grid_parameters': {'death_threshold': [-10, -20], 'mutation_rate': [0], 'transfer_amount': [1, 2]},
            'fixed_field_parameters': {'initial_agents': 0},
        }
        outputs = []
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            for workers in [1, 2]:
                config['simulation']['workers'] = workers
                raw, summary = simulation.run_experiment(config, directory)
                series = pd.read_csv(Path(directory) / config['experiment_name'] / 'time_series_data.csv')
                self.assertEqual(len(raw), 8)
                self.assertEqual(len(summary), 4)
                self.assertEqual(len(series), 8)
                self.assertEqual(series['run_id'].unique().tolist(), [0])
                self.assertTrue((series[['population', 'altruism_count']] == 0).all().all())
                outputs.append((raw, summary, series))
            for serial, parallel in zip(*outputs):
                pd.testing.assert_frame_equal(serial, parallel)
            config['simulation'].update(workers=1, total_steps=999, eval_start_step=500)
            simulation.run_experiment(config, directory)
            series = pd.read_csv(Path(directory) / config['experiment_name'] / 'time_series_data.csv')
            self.assertTrue(series.empty)
            self.assertEqual(series.columns.tolist(), ['death_threshold', 'mutation_rate', 'transfer_amount', 'run_id', 'seed', 'step', 'population', 'altruism_count'])


if __name__ == '__main__':
    unittest.main()
