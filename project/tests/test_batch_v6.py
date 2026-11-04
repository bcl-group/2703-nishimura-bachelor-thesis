"""Behavioral checks for v6 stages, batching, recording and memory retries."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from src.fields.field_v6 import FieldV6
    from src.fields.batch_v6 import BatchV6
    from scripts import batch_runner_v6
    from scripts import simu_every_para_up_v6 as runner


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed')
class BatchTests(unittest.TestCase):
    def field(self, **kwargs):
        return FieldV6(**{'field_size': 3, 'initial_agents': 12,
                          'initial_food_prob': .5, 'step_food_prob': .1,
                          'clone_threshold': 3., 'death_threshold': -3.,
                          'transfer_amount': 2., 'mutation_rate': .25, **kwargs})

    def compare(self, batch, row, field):
        n = field.population
        self.assertEqual(batch.counts[row], n)
        for name in ('positions', 'energies', 'ages', 'weights'):
            torch.testing.assert_close(getattr(batch, name)[row, :n], getattr(field, name), rtol=0, atol=0)
        torch.testing.assert_close(batch.grid_food[row], field.grid_food, rtol=0, atol=0)
        self.assertEqual(batch.altruism_count[row].item(), field.altruism_count.item())
        self.assertEqual(batch.genotype_counts()[row].sum().item(), n)

    def test_single_batch_reordering_and_capacity_independent_streams(self):
        for device in ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []):
            options = [dict(initial_agents=0, seed=1), dict(initial_agents=1, seed=2),
                       dict(initial_agents=25, seed=3), dict(initial_agents=7, seed=4)]
            singles = [self.field(device=device, **item) for item in options]
            batch = BatchV6([self.field(device=device, **item) for item in options])
            reverse = BatchV6([self.field(device=device, **item) for item in reversed(options)])
            with runner.deterministic_execution():
                for _ in range(35):
                    batch.step()
                    reverse.step()
                    for row, single in enumerate(singles):
                        deaths = single.step()
                        self.compare(batch, row, single)
                        self.compare(reverse, len(singles)-row-1, single)
                        self.assertEqual(sorted(deaths.tolist()),
                                         sorted(batch.death_ages[row][batch.death_ages[row] > 0].tolist()))

    def test_fixed_stages_competition_transfers_death_and_clone(self):
        # Agents 0 and 2 aid agent 1; one food winner survives starvation,
        # agent 2 clones and agent 3 dies. Every decision uses the old state.
        field = self.field(field_size=1, initial_agents=4, initial_food_prob=0,
                           step_food_prob=0, energy_loss_per_step=1,
                           death_threshold=-2, clone_threshold=4, transfer_amount=2,
                           reward_food=3, mutation_rate=1)
        field.energies[:] = torch.tensor([2., -3., 7., -3.])
        field.weights.zero_()
        field.weights[[0, 2], 1, 1] = 1
        field.grid_food.fill_(1)
        batch = BatchV6([field])
        inputs = (torch.tensor([[1, 0, 1, 0]]), torch.zeros((1, 4), dtype=torch.int64),
                  torch.zeros((1, 4, 2), dtype=torch.int64))
        positions, recipients, transfers, requests = batch.decide_actions(inputs)
        self.assertEqual(transfers.tolist(), [[2., 0., 2., 0.]])
        won, consumed = batch.resolve_grid(requests, torch.tensor([[1, 3, 0, 2]]))
        self.assertEqual(won.tolist(), [[False, True, False, False]])
        self.assertEqual(consumed.tolist(), [[[1]]])
        self.assertEqual(batch.grid_food.item(), 1)
        batch.update_agents(positions, recipients, transfers, won)
        self.assertEqual(batch.counts, [4])
        self.assertEqual(batch.energies[0, :4].tolist(), [-1., 3., 0., 0.])
        self.assertEqual(batch.ages[0, :4].tolist(), [1, 1, 1, 0])
        self.assertEqual(batch.death_counts.tolist(), [1])
        self.assertTrue(torch.all(batch.weights[0, 3] == 1-batch.weights[0, 2]).item())
        self.assertEqual(batch.altruism_count.item(), 2)

    def test_capacity_growth_preserves_all_children(self):
        field = self.field(initial_agents=5, clone_threshold=1, mutation_rate=0,
                           energy_loss_per_step=0, initial_food_prob=0, step_food_prob=0)
        field.energies.fill_(1)
        field.weights.zero_()
        batch = BatchV6([field])
        batch.step()
        self.assertEqual(batch.counts, [10])
        self.assertGreaterEqual(batch.capacity, 10)
        self.assertEqual(batch.ages[0, :10].tolist(), [1, 0]*5)
        self.assertEqual(batch.energies[0, :10].tolist(), [0.]*10)
        torch.testing.assert_close(batch.weights[0, :10:2], batch.weights[0, 1:10:2])

    def test_batched_observation_matches_single_spatial_index(self):
        for size in (1, 2, 5):
            fields = [self.field(field_size=size, initial_agents=n, seed=n) for n in (1, 9, 30)]
            # Reuse RNG draws with cloned generator states to compare the indexes.
            originals = [f.generator.get_state() for f in fields]
            batch = BatchV6(fields)
            nearby, food, directions, _ = batch._random_inputs()
            observed = batch.prepare_step_inputs(nearby, food, directions)
            for row, field in enumerate(fields):
                field.generator.set_state(originals[row])
                expected = field.prepare_step_inputs()
                for actual, reference in zip(observed, expected):
                    torch.testing.assert_close(actual[row, :field.population], reference)

    def test_runner_batch_partition_and_mixed_field_sizes(self):
        import pandas as pd
        config = json.loads(Path('configs/exp_v6.json').read_text())
        config['simulation'].update(device='cpu', total_steps=20, eval_start_step=5, num_runs=2)
        config['grid_parameters'] = {'field_size': [1, 3], 'death_threshold': [-2, -4]}
        config['fixed_field_parameters'].update(initial_agents=4, clone_threshold=2, step_food_prob=.1)
        results = []
        with tempfile.TemporaryDirectory() as directory:
            for size in (None, 1, 3):
                config['simulation']['batch_size'] = size
                config['experiment_name'] = f'batch_{size}'
                raw, summary = runner.run_experiment(config, directory)
                output = Path(directory) / config['experiment_name']
                series = pd.read_csv(output / 'time_series_data.csv')
                genotypes = pd.read_csv(output / 'genotype_time_series_data.csv')
                results.append((raw, summary, series, genotypes))
        for actual in results[1:]:
            for i, (reference, frame) in enumerate(zip(results[0], actual)):
                if i >= 2:
                    keys = ['field_size', 'death_threshold', 'run_id', 'step']
                    if i == 3:
                        keys.append('genotype')
                    reference = reference.sort_values(keys).reset_index(drop=True)
                    frame = frame.sort_values(keys).reset_index(drop=True)
                pd.testing.assert_frame_equal(reference, frame, check_exact=True)

    def test_memory_retry_restarts_without_duplicate_records(self):
        config = dict(device='cuda:0', total_steps=3, eval_start_step=1, base_seed=42)
        jobs = [({'initial_agents': n}, 0) for n in (2, 3, 4)]
        calls = []
        original = batch_runner_v6.run_batch
        def failing_batch(cls, chunk, simulation, fixed):
            # Fail after the calculation, exercising release and seeded restart.
            result = original(cls, chunk, {**simulation, 'device': 'cpu'}, fixed)
            calls.append(len(chunk))
            if len(chunk) > 1:
                raise torch.cuda.OutOfMemoryError('synthetic memory limit')
            return result
        with patch.object(batch_runner_v6, 'run_batch', side_effect=failing_batch), \
                patch.object(torch.cuda, 'empty_cache'):
            output = list(batch_runner_v6.run_with_retry(FieldV6, jobs, config, {}))
        records = [row for records, _, _ in output for row in records]
        expected, _, _ = original(FieldV6, jobs, {**config, 'device': 'cpu'}, {})
        # NaN fields compare through pandas.
        import pandas as pd
        pd.testing.assert_frame_equal(pd.DataFrame(records), pd.DataFrame(expected), check_exact=True)
        self.assertEqual(calls, [3, 1, 2, 1, 1])
        with patch.object(batch_runner_v6, '_attempt_batch', return_value=None), \
                patch.object(torch.cuda, 'empty_cache'):
            with self.assertRaisesRegex(RuntimeError, 'single v6 simulation'):
                list(batch_runner_v6.run_with_retry(FieldV6, jobs[:1], config, {}))

    def test_invalid_config_and_cuda_fail_before_output(self):
        config = json.loads(Path('configs/exp_v6.json').read_text())
        config['simulation']['device'] = 'cpu'
        with tempfile.TemporaryDirectory() as directory:
            for size in (0, -1, True, 1.5):
                config['simulation']['batch_size'] = size
                with self.assertRaisesRegex(ValueError, 'batch_size'):
                    runner.run_experiment(config, directory)
                self.assertEqual(list(Path(directory).iterdir()), [])
            if not torch.cuda.is_available():
                config['simulation'].update(batch_size=None, device='cuda:0')
                with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                    runner.run_experiment(config, directory)
                self.assertEqual(list(Path(directory).iterdir()), [])

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cpu_cuda_fixed_stages(self):
        cpu = self.field(field_size=2, initial_agents=10, mutation_rate=1, step_food_prob=0,
                         clone_threshold=2, death_threshold=-2)
        cpu.energies[:] = torch.arange(10, dtype=torch.float64) - 4
        gpu = self.field(field_size=2, initial_agents=10, mutation_rate=1, step_food_prob=0,
                         clone_threshold=2, death_threshold=-2, device='cuda:0')
        for name in ('positions', 'energies', 'ages', 'weights', 'grid_food'):
            setattr(gpu, name, getattr(cpu, name).cuda())
        inputs = tuple(x.unsqueeze(0) for x in cpu.prepare_step_inputs())
        lottery = torch.arange(9, -1, -1)[None, :]
        a, b = BatchV6([cpu]), BatchV6([gpu])
        with runner.deterministic_execution():
            a.step(inputs, lottery)
            b.step(tuple(x.cuda() for x in inputs), lottery.cuda())
        self.assertEqual(a.counts, b.counts)
        for name in ('positions', 'energies', 'ages', 'weights', 'grid_food', 'death_ages'):
            torch.testing.assert_close(getattr(a, name), getattr(b, name).cpu())


if __name__ == '__main__':
    unittest.main()
