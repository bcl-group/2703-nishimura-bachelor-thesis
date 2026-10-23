"""変更(v6/v7): 同期更新のルール、テンソル版の集計とCUDAの一致を検証する。"""
import copy
import importlib.util
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from src import AgentV6, AgentV7, FieldV5, FieldV6, FieldV7
from src.factory import get_agent_class, get_field_class
from scripts import simu_every_para_up_v6 as runner
from scripts import simu_every_para_up_v7 as cuda_runner

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch


class RegistrationTests(unittest.TestCase):
    def test_registration_without_eager_torch_dependency(self):
        self.assertIs(get_agent_class('AgentV6'), AgentV6)
        self.assertIs(get_agent_class('AgentV7'), AgentV7)
        self.assertIs(get_field_class('FieldV6'), FieldV6)
        self.assertIs(get_field_class('FieldV7'), FieldV7)

    def test_missing_dependency_before_output(self):
        if HAS_TORCH:
            self.skipTest('PyTorch is installed')
        config = json.loads(Path('configs/exp_v6.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'requires PyTorch'):
                runner.run_experiment(config, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


@unittest.skipUnless(HAS_TORCH, 'PyTorch is not installed')
class TensorFieldTests(unittest.TestCase):
    def make_field(self, count=0, **kwargs):
        field = FieldV6(field_size=1, initial_agents=count, initial_food_prob=0,
                        step_food_prob=0, energy_loss_per_step=0, **kwargs)
        field.weights.zero_()
        return field

    def test_food_limits_and_energy(self):
        for count, durability in ((1, 5), (3, 5), (5, 3), (5, 0)):
            field = self.make_field(count, reward_food=2)
            field.grid_food.fill_(durability)
            field.step()
            self.assertEqual(field.energies.sum().item(), min(count, durability) * 2)
            self.assertEqual(field.grid_food.item(), max(0, durability-count))

    def test_food_lottery_without_replacement_and_replenishment(self):
        field = self.make_field(5, food_durability=3)
        field.grid_food.fill_(3)
        with patch.object(torch, 'randperm', return_value=torch.tensor([4, 2, 1, 0, 3])):
            rewards, consumed = field.resolve_food(field.positions)
        self.assertEqual(rewards.tolist(), [0., 1., 1., 0., 1.])
        self.assertEqual(consumed.item(), 3)
        self.assertEqual(field.grid_food.item(), 3)
        field.step_food_prob = 1
        field.step()
        self.assertEqual(field.energies.sum().item(), 3)
        self.assertEqual(field.grid_food.item(), 3)

    def test_empty_and_food_reset(self):
        field = self.make_field()
        field.grid_food.fill_(3)
        for _ in range(4):
            self.assertEqual(field.step().numel(), 0)
        self.assertEqual(field.grid_food.item(), 3)
        field.step_food_prob = 1
        field.step()
        self.assertEqual(field.grid_food.item(), 5)

    def test_transfers_and_death_clone(self):
        field = self.make_field(2, transfer_amount=3)
        field.energies[:] = torch.tensor([10., -50.])
        field.weights[0, :, 1] = torch.tensor([0, 1])
        field.step()
        torch.testing.assert_close(field.energies, torch.tensor([7., -47.], dtype=torch.float64))
        self.assertEqual(field.altruism_count.item(), 1)
        field = self.make_field(2, reward_food=2, death_threshold=-1)
        field.energies.fill_(-1)
        field.grid_food.fill_(1)
        self.assertEqual(field.step().tolist(), [1])
        field.energies.fill_(99)
        field.grid_food.fill_(1)
        field.step()
        self.assertEqual(field.ages.tolist(), [2, 0])
        self.assertEqual(field.energies.tolist(), [0., 0.])

    def test_mutation_and_parent_child_order(self):
        field = self.make_field(2, mutation_rate=1, clone_threshold=1)
        field.energies[:] = torch.tensor([1., 2.])
        field.ages[:] = torch.tensor([3, 7])
        field.step()
        self.assertEqual(field.ages.tolist(), [4, 0, 8, 0])
        self.assertTrue(torch.all(field.weights[[1, 3]] == 1))
        self.assertTrue(torch.all(field.weights[[0, 2]] == 0))

    def test_neighbors_exclude_self_and_deduplicate(self):
        for size in (1, 2, 5):
            field = FieldV6(field_size=size, initial_agents=30, seed=9)
            self.assertEqual(field.neighbor_cells.shape[1], min(size, 3) ** 2)
            for _ in range(10):
                neighbors, foods, _ = field.prepare_step_inputs()
                valid = neighbors >= 0
                self.assertTrue(torch.all(neighbors[valid] != torch.arange(30)[valid]))
                diffs = (field.positions[neighbors[valid]] - field.positions[valid]).abs()
                self.assertTrue(torch.all(torch.minimum(diffs, size-diffs) <= 1))
                self.assertTrue(torch.all(field.grid_food.flatten()[foods[foods >= 0]] > 0))
        field = self.make_field(1)
        self.assertEqual(field.prepare_step_inputs()[0].tolist(), [-1])

    def test_v5_fixed_observations_match(self):
        from src.fields.field_v5 import StepInput
        random.seed(13)
        np.random.seed(13)
        old = FieldV5(field_size=4, initial_agents=12, initial_food_prob=0,
                      step_food_prob=0, transfer_amount=3)
        old.grid_food[:] = 20  # 全員が給餌され、競合抽選による差をなくす。
        field = FieldV6(field_size=4, initial_agents=12, initial_food_prob=0,
                        step_food_prob=0, transfer_amount=3)
        field.positions = torch.tensor([(a.x, a.y) for a in old.agents])
        field.energies = torch.arange(12, dtype=torch.float64) - 49
        for agent, energy in zip(old.agents, field.energies.tolist()):
            agent.energy = energy
        field.weights = torch.tensor(np.stack([a.weights for a in old.agents]))
        field.grid_food.fill_(20)
        inputs = field.prepare_step_inputs()
        state = old.snapshot()
        results = []
        for i, (neighbor, food, direction) in enumerate(zip(*[x.tolist() for x in inputs])):
            item = StepInput(i, None if neighbor < 0 else neighbor,
                             None if food < 0 else divmod(food, 4), tuple(direction))
            results.append(old.compute_agent_step(state, item))
        positions, transfers = field.compute_agent_step(*inputs)
        self.assertEqual(positions.tolist(), [list(r.position) for r in results])
        np.testing.assert_allclose(transfers.numpy(), [r.transfer for r in results])
        incoming = np.zeros(12)
        for result in results:
            if result.recipient_index is not None:
                incoming[result.recipient_index] += result.transfer
        expected = state.energies - .1 - transfers.numpy() + incoming + 1
        field.step(inputs)
        np.testing.assert_allclose(field.energies.numpy(), expected)

    def test_seeded_energy_balance_and_genotypes(self):
        def run():
            field = FieldV6(field_size=4, initial_agents=12, initial_food_prob=.5,
                            step_food_prob=0, death_threshold=-1e6, clone_threshold=1e6)
            for _ in range(20):
                before = field.energies.sum().item()
                food = field.grid_food.sum().item()
                field.step()
                eaten = food - field.grid_food.sum().item()
                self.assertAlmostEqual(field.energies.sum().item(), before - 1.2 + eaten)
                field.spawn_objects(.2)
            self.assertEqual(sum(field.genotype_counts().values()), field.population)
            return field
        a, b = run(), run()
        for name in ('positions', 'energies', 'ages', 'weights', 'grid_food'):
            torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)

    def test_runner_csv_and_extinction(self):
        import pandas as pd
        config = json.loads(Path('configs/exp_v6.json').read_text())
        config['simulation'].update(total_steps=2000, eval_start_step=5)
        config['grid_parameters'] = {}
        config['fixed_field_parameters'].update(initial_agents=2, death_threshold=0,
                                                 initial_food_prob=0, step_food_prob=0)
        with tempfile.TemporaryDirectory() as directory:
            raw, summary = runner.run_experiment(config, directory)
            self.assertEqual(raw.iloc[0]['avg_population'], 0)
            self.assertTrue(np.isnan(raw.iloc[0]['avg_death_age']))
            output = Path(directory) / config['experiment_name']
            self.assertEqual(len(list(output.iterdir())), 5)
            series = pd.read_csv(output / 'time_series_data.csv')
            self.assertEqual(series['step'].tolist(), [1, 1000, 2000])
            self.assertEqual(series['population'].tolist(), [0, 0, 0])
            invalid = copy.deepcopy(config)
            invalid['simulation']['workers'] = 2
            with self.assertRaisesRegex(ValueError, 'workers must be 1'):
                runner.run_experiment(invalid, directory)
            config['simulation'].update(total_steps=10, eval_start_step=1)
            raw, _ = runner.run_experiment(config, directory)
            self.assertEqual(raw.iloc[0]['avg_death_age'], 1)

    def test_cuda_unavailable_before_output(self):
        if torch.cuda.is_available():
            self.skipTest('CUDA is available')
        config = json.loads(Path('configs/exp_v7.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                cuda_runner.run_experiment(config, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cuda_native_agent_api(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA is unavailable')
        weights = torch.randint(2, (20, 2, 4), dtype=torch.int64)
        observations = torch.randint(2, (20, 4), dtype=torch.int64)
        expected = AgentV6.action_from_weights(weights, observations)
        actual = AgentV7.action_from_weights(weights.cuda(), observations.cuda())
        torch.testing.assert_close(actual.cpu(), expected)
        generator = torch.Generator(device='cuda:0').manual_seed(91)
        for rate in (0., 1.):
            actual = AgentV7.mutate(weights.cuda(), rate, generator)
            torch.testing.assert_close(actual.cpu(), weights if rate == 0 else 1-weights)

    def test_cuda_native_seeded_balance_and_empty(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA is unavailable')
        def run():
            field = FieldV7(field_size=3, initial_agents=20, initial_food_prob=.5,
                            step_food_prob=0, death_threshold=-1e6, clone_threshold=1e6,
                            transfer_amount=3, seed=99)
            with runner.deterministic_execution():
                for _ in range(10):
                    before = field.energies.sum().item()
                    food = field.grid_food.sum().item()
                    field.step()
                    eaten = food - field.grid_food.sum().item()
                    self.assertAlmostEqual(field.energies.sum().item(), before-2+eaten)
                    field.spawn_objects(.2)
            return field
        a, b = run(), run()
        for name in ('positions', 'energies', 'ages', 'weights', 'grid_food'):
            torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
        empty = FieldV7(initial_agents=0, initial_food_prob=0, step_food_prob=1)
        self.assertEqual(empty.step().numel(), 0)
        self.assertTrue(torch.all(empty.grid_food == empty.food_durability).item())
        clone = FieldV7(field_size=1, initial_agents=2, initial_food_prob=0,
                        step_food_prob=0, energy_loss_per_step=0, clone_threshold=1,
                        mutation_rate=1)
        clone.weights.zero_()
        clone.energies.fill_(1)
        clone.step()
        self.assertEqual(clone.ages.tolist(), [1, 0, 1, 0])
        self.assertTrue(torch.all(clone.weights[[1, 3]] == 1).item())
        with self.assertRaisesRegex(ValueError, 'out of range'):
            clone.step((torch.full((4,), 4, device='cuda:0'),
                        torch.full((4,), -1, device='cuda:0'),
                        torch.zeros((4, 2), dtype=torch.int64, device='cuda:0')))

    def test_cpu_cuda_fixed_step(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA is unavailable')
        cpu = FieldV6(field_size=3, initial_agents=20, initial_food_prob=0,
                      step_food_prob=0, death_threshold=-1e6, clone_threshold=1e6)
        cpu.grid_food.fill_(20)
        gpu = FieldV7(field_size=3, initial_agents=20, initial_food_prob=0,
                      step_food_prob=0, death_threshold=-1e6, clone_threshold=1e6)
        for name in ('positions', 'energies', 'ages', 'weights', 'grid_food'):
            setattr(gpu, name, getattr(cpu, name).to('cuda:0'))
        inputs = cpu.prepare_step_inputs()
        with runner.deterministic_execution():
            cpu.step(inputs)
            gpu.step(tuple(item.to('cuda:0') for item in inputs))
        for name in ('positions', 'energies', 'ages', 'weights', 'grid_food'):
            torch.testing.assert_close(getattr(cpu, name), getattr(gpu, name).cpu())


if __name__ == '__main__':
    unittest.main()
