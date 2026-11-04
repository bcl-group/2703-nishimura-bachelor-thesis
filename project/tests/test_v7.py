"""v7 retains v6 statistics and state while never recording history."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    import pandas as pd
    from src import AgentV7, FieldV6, FieldV7
    from src.factory import get_agent_class, get_field_class
    from src.fields.batch_v6 import BatchV6
    from scripts import batch_runner_v6 as batch_runner
    from scripts import simu_every_para_up_v6 as v6
    from scripts import simu_every_para_up_v7 as v7


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed')
class V7Tests(unittest.TestCase):
    def config(self, **simulation):
        config = json.loads(Path('configs/exp_v7.json').read_text())
        config['simulation'].update(device='cpu', total_steps=25, eval_start_step=5,
                                    num_runs=2, batch_size=None)
        config['simulation'].update(simulation)
        config['grid_parameters'] = {'field_size': [1, 3], 'death_threshold': [-2, -4]}
        config['fixed_field_parameters'].update(initial_agents=4, clone_threshold=2,
            initial_food_prob=.5, step_food_prob=.1, food_durability=3, transfer_amount=2)
        return config

    def test_registration_and_cpu_state_match_v6(self):
        self.assertIs(get_agent_class('AgentV7'), AgentV7)
        self.assertIs(get_field_class('FieldV7'), FieldV7)
        for device in ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []):
            kwargs = dict(device=device, field_size=3, initial_agents=12,
                          clone_threshold=2, death_threshold=-2, initial_food_prob=.5,
                          step_food_prob=.1, transfer_amount=2, mutation_rate=.2, seed=7)
            a, b = FieldV6(**kwargs), FieldV7(**kwargs)
            with v6.deterministic_execution():
                for _ in range(30):
                    torch.testing.assert_close(a.step(), b.step(), rtol=0, atol=0)
                    for name in ('positions', 'energies', 'ages', 'weights', 'grid_food', 'altruism_count'):
                        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
            clone = FieldV7(device=device, field_size=1, initial_agents=3,
                            energy_loss_per_step=0, clone_threshold=1, mutation_rate=1,
                            initial_food_prob=0, step_food_prob=0)
            clone.weights.zero_()
            clone.energies.fill_(1)
            batch = BatchV6([clone])
            batch.step()
            self.assertEqual(batch.counts, [6])
            self.assertEqual(batch.ages[0, :6].tolist(), [1, 0]*3)
            self.assertTrue(torch.all(batch.weights[0, 1:6:2] == 1).item())

    def test_statistics_match_v6_across_batch_partitions(self):
        config = self.config()
        old = copy.deepcopy(config)
        old.update(experiment_name='reference_v6', field_class='FieldV6', agent_class='AgentV6')
        with tempfile.TemporaryDirectory() as directory:
            expected = v6.run_experiment(old, directory)
            for size in (None, 1, 3):
                config['simulation']['batch_size'] = size
                config['experiment_name'] = f'v7_batch_{size}'
                with patch.object(BatchV6, 'genotype_counts', side_effect=AssertionError('History must not be sampled')):
                    actual = v7.run_experiment(config, directory)
                for left, right in zip(expected, actual):
                    pd.testing.assert_frame_equal(left, right, check_exact=True)
                output = Path(directory) / config['experiment_name']
                self.assertEqual({p.name for p in output.iterdir()},
                                 {'config_used.json', 'raw_data.csv', 'summary_data.csv'})
                pd.testing.assert_frame_equal(pd.read_csv(output / 'raw_data.csv'), actual[0])

    def test_no_history_at_recording_boundaries_or_after_extinction(self):
        # Pass step 1000 with a living population, then exercise immediate and
        # initial extinction with a complete evaluation-window denominator.
        for count, loss, death in ((2, 0, -50), (2, 1, 0), (0, 0, -50)):
            simulation = dict(device='cpu', total_steps=1005, eval_start_step=1, base_seed=42)
            fixed = dict(initial_agents=count, initial_food_prob=0, step_food_prob=0,
                         energy_loss_per_step=loss, death_threshold=death, transfer_amount=0)
            with patch.object(BatchV6, 'genotype_counts', side_effect=AssertionError('History must not be sampled')):
                records, series, genotypes = batch_runner.run_batch(FieldV7, [({}, 0)], simulation, fixed)
            self.assertEqual(series, ())
            self.assertEqual(genotypes, ())
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['avg_population'], count if loss == 0 else 0)
            if count and loss:
                self.assertEqual(records[0]['avg_death_age'], 1)
            else:
                self.assertTrue(pd.isna(records[0]['avg_death_age']))
        config = self.config(total_steps=2000, eval_start_step=5)
        config['grid_parameters'] = {}
        config['fixed_field_parameters'].update(initial_agents=2, initial_food_prob=0,
            step_food_prob=0, death_threshold=0, energy_loss_per_step=1)
        with tempfile.TemporaryDirectory() as directory:
            raw, _ = v7.run_experiment(config, directory)
            self.assertTrue(raw['avg_population'].eq(0).all())
            self.assertTrue(raw['avg_death_age'].isna().all())

    def test_v7_retry_uses_fresh_streams_and_no_history(self):
        simulation = dict(device='cuda:0', total_steps=5, eval_start_step=1, base_seed=42)
        jobs = [({'initial_agents': n}, 0) for n in (2, 3, 4)]
        original = batch_runner.run_batch
        def simulated_oom(cls, chunk, options, fixed):
            result = original(cls, chunk, {**options, 'device': 'cpu'}, fixed)
            if len(chunk) > 1:
                raise torch.cuda.OutOfMemoryError('synthetic limit')
            return result
        with patch.object(batch_runner, 'run_batch', side_effect=simulated_oom), \
                patch.object(torch.cuda, 'empty_cache'), \
                patch.object(BatchV6, 'genotype_counts', side_effect=AssertionError('No history')):
            results = list(batch_runner.run_with_retry(FieldV7, jobs, simulation, {}))
        expected, _, _ = original(FieldV7, jobs, {**simulation, 'device': 'cpu'}, {})
        actual = [row for rows, _, _ in results for row in rows]
        pd.testing.assert_frame_equal(pd.DataFrame(expected), pd.DataFrame(actual), check_exact=True)
        self.assertTrue(all(series == () and genes == () for _, series, genes in results))

    def test_cli_dispatch_and_invalid_inputs_before_output(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(config))
            with patch.object(v6, 'run_experiment') as run:
                v7.main(['--config', str(path)])
                run.assert_called_once_with(config, version=7)
            bad = copy.deepcopy(config)
            bad['field_class'] = 'FieldV6'
            with self.assertRaisesRegex(ValueError, 'requires FieldV7'):
                v7.run_experiment(bad, directory)
            if not torch.cuda.is_available():
                bad = copy.deepcopy(config)
                bad['simulation']['device'] = 'cuda:0'
                with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                    v7.run_experiment(bad, directory)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['config.json'])


if __name__ == '__main__':
    unittest.main()
