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

    def test_config_has_exactly_3300_matching_v4_conditions(self):
        config = json.loads(Path('configs/exp_v4.json').read_text())
        expected = {
            'initial_food_prob': [0.05, 0.1, 0.15, 0.2],
            'step_food_prob': [0.002, 0.004, 0.006, 0.008, 0.01],
            'food_lifetime_steps': [1, 2, 3],
            'death_threshold': [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, -5],
            'transfer_amount': [1, 3, 5, 7, 9],
        }
        self.assertEqual(config['grid_parameters'], expected)
        combinations = simulation.expand_grid_parameters(config['grid_parameters'])
        self.assertEqual(len(combinations), 3300)
        self.assertEqual(config['agent_class'], 'AgentV4')
        self.assertEqual(config['field_class'], 'FieldV4')
        self.assertEqual(config['simulation'], {
            'total_steps': 200000, 'eval_start_step': 100001,
            'num_runs': 1, 'base_seed': 43, 'workers': 5,
        })
        self.assertEqual(config['fixed_field_parameters']['mutation_rate'], 0.05)
        for combination in combinations:
            inspect.signature(FieldV4).bind(**{**config['fixed_field_parameters'], **combination})

    def simulate_history(self, total_steps=2500, extinction=None, run_id=0, collect=True):
        agent = make_agent(genotype='00000101')
        field = SimpleNamespace(agents=[agent], altruism_count=0)
        calls = 0

        def step():
            nonlocal calls
            calls += 1
            agent.age += 1
            if calls == extinction:
                field.agents = []
            field.altruism_count = calls % 3
            return [int(agent.age)] if calls == extinction else []

        field.step = step
        series, genotypes = [], []
        result = simulation.run_single_simulation(
            lambda **kwargs: field, {'death_threshold': -10},
            {'total_steps': total_steps, 'eval_start_step': min(1500, total_steps), 'base_seed': 42},
            {}, run_id, series if collect else None, genotypes if collect else None,
        )
        return result, series, genotypes, calls

    def test_series_starts_at_one_and_counts_nonoverlapping_intervals(self):
        result, rows, genotypes, calls = self.simulate_history()
        self.assertEqual(calls, 2500)
        self.assertEqual([r['step'] for r in rows], [1, 1000, 2000])
        self.assertEqual([r['altruism_count'] for r in rows], [
            1, sum(i % 3 for i in range(2, 1001)), sum(i % 3 for i in range(1001, 2001)),
        ])
        self.assertEqual([r['step'] for r in genotypes], [1, 1000, 2000])
        self.assertTrue(all(r['genotype'] == '00000101' and r['population'] == 1 for r in genotypes))
        pd.testing.assert_series_equal(pd.Series(result), pd.Series(self.simulate_history(collect=False)[0]))

    def test_short_runs_record_step_one_and_other_trials_do_not_record(self):
        self.assertEqual([r['step'] for r in self.simulate_history(total_steps=999)[1]], [1])
        result, rows, genotypes, _ = self.simulate_history(total_steps=1)
        self.assertEqual([r['step'] for r in rows], [1])
        self.assertEqual(len(genotypes), 1)
        self.assertEqual(result['death_count'], 0)
        self.assertEqual(self.simulate_history(run_id=1)[1:3], ([], []))

    def test_extinction_preserves_pending_actions_and_pads_with_zeros(self):
        for extinction in (1, 999, 1000, 1001, 1500, 2000):
            with self.subTest(extinction=extinction):
                result, rows, genotypes, calls = self.simulate_history(3500, extinction)
                self.assertEqual(calls, extinction)
                self.assertEqual([r['step'] for r in rows], [1, 1000, 2000, 3000])
                last = 0
                for row in rows:
                    self.assertEqual(row['population'], int(row['step'] < extinction))
                    self.assertEqual(row['altruism_count'], sum(
                        i % 3 for i in range(last + 1, min(row['step'], extinction) + 1)
                    ))
                    last = row['step']
                self.assertEqual([r['step'] for r in genotypes], [r['step'] for r in rows if r['population']])
                self.assertEqual(result['death_count'], int(extinction >= 1500))
                if extinction >= 1500:
                    self.assertEqual(result['avg_death_age'], extinction)
                    self.assertTrue(np.isnan(result['std_death_age']))

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
        self.assertEqual(result['death_count'], 2)
        self.assertEqual(result['death_age_sum'], 12 + 23)
        self.assertEqual(result['death_age_squared_sum'], 12**2 + 23**2)
        self.assertEqual(result['avg_death_age'], 17.5)
        self.assertAlmostEqual(result['std_death_age'], np.std([12, 23], ddof=1))
        self.assertEqual(result['avg_population'], 4 / 3)
        one = self.simulate_death_ages(eval_start=3)
        self.assertEqual(one['death_count'], 1)
        self.assertEqual(one['avg_death_age'], 23)
        self.assertTrue(np.isnan(one['std_death_age']))
        zero = self.simulate_death_ages(eval_start=4)
        self.assertEqual(zero['death_count'], 0)
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
        self.assertEqual(result['death_count'], 0)
        self.assertEqual(len(field.agents), 3)
        self.assertEqual(receiver.energy, -48)
        self.assertEqual(field.agents[1].age, 0)  # New child of the donor.

    def test_genotype_counts_include_newborns_in_row_major_order(self):
        field = FieldV4(initial_agents=3, initial_food_prob=0, step_food_prob=0,
                        energy_loss_per_step=0, mutation_rate=0)
        field.agents[0].weights[:] = make_agent(genotype='00000101').weights
        field.agents[1].weights[:] = make_agent(genotype='00000101').weights
        field.agents[2].weights[:] = make_agent(genotype='10000001').weights
        field.agents[0].energy = 100
        genotypes = []
        simulation.run_single_simulation(
            lambda: field, {}, {'total_steps': 1, 'eval_start_step': 1, 'base_seed': 42}, {},
            genotype_records=genotypes,
        )
        self.assertEqual([(r['genotype'], r['population']) for r in genotypes], [
            ('00000101', 3), ('10000001', 1),
        ])

    def test_pooled_death_statistics_weight_individuals_and_handle_missing_data(self):
        def record(run_id, ages):
            mean, std = simulation.mean_and_std(sum(ages), sum(a*a for a in ages), len(ages))
            return {
                'condition': 1, 'run_id': run_id, 'avg_population': run_id + 1,
                'std_population': 10, 'avg_altruism_count': 0, 'std_altruism_count': 0,
                'death_count': len(ages), 'death_age_sum': sum(ages),
                'death_age_squared_sum': sum(a*a for a in ages),
                'avg_death_age': mean, 'std_death_age': std,
            }
        for histories in ([[], []], [[4], []], [[2, 4], [10], []]):
            records = pd.DataFrame([record(i, ages) for i, ages in enumerate(histories)])
            for keys in ([], ['condition']):
                with self.subTest(histories=histories, keys=keys):
                    row = simulation.summarize_runs(records, keys).iloc[0]
                    ages = [age for history in histories for age in history]
                    self.assertEqual(row['death_count'], len(ages))
                    np.testing.assert_allclose(row['avg_death_age'], np.mean(ages) if ages else np.nan, equal_nan=True)
                    np.testing.assert_allclose(row['std_death_age'], np.std(ages, ddof=1) if len(ages) > 1 else np.nan, equal_nan=True)
                    self.assertEqual(row['avg_population'], records['avg_population'].mean())

    def test_bounded_parallel_results_preserve_task_order(self):
        submitted = []
        executor = Mock()

        def submit(function, task):
            submitted.append(task)
            return SimpleNamespace(result=lambda: (task, len(submitted)))

        executor.submit.side_effect = submit
        with patch.object(simulation.concurrent.futures, 'ProcessPoolExecutor') as pool:
            pool.return_value.__enter__.return_value = executor
            outputs = simulation._iter_condition_results(list(range(7)), 2)
            self.assertEqual(submitted, [])
            self.assertEqual(next(outputs)[0], 0)
            self.assertEqual(len(submitted), 3)
            self.assertEqual([item[0] for item in outputs], list(range(1, 7)))

    def test_small_serial_parallel_csv_outputs_match_and_preserve_other_data(self):
        config = {
            'experiment_name': 'v4_test', 'agent_class': 'AgentV4', 'field_class': 'FieldV4',
            'simulation': {'total_steps': 1001, 'eval_start_step': 500, 'num_runs': 2, 'base_seed': 43, 'workers': 1},
            'grid_parameters': {'death_threshold': [-0.2, -1000]},
            'fixed_field_parameters': {
                'initial_agents': 3, 'field_size': 3, 'mutation_rate': 0,
                'initial_food_prob': 0.1, 'step_food_prob': 0.01,
                'food_lifetime_steps': 2, 'clone_threshold': 1000000,
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
                series = pd.read_csv(folder / 'time_series_data.csv')
                genotypes = pd.read_csv(folder / 'genotype_time_series_data.csv', dtype={'genotype': str})
                self.assertEqual(len(raw), 4)
                self.assertEqual(len(summary), 2)
                self.assertEqual(len(series), 4)
                self.assertEqual(series['run_id'].unique().tolist(), [0])
                self.assertEqual(series['step'].tolist(), [1, 1000, 1, 1000])
                self.assertTrue(genotypes['genotype'].str.fullmatch('[01]{8}').all())
                counts = genotypes.groupby(['death_threshold', 'step'])['population'].sum()
                for row in series.itertuples():
                    self.assertEqual(counts.get((row.death_threshold, row.step), 0), row.population)
                self.assertEqual(json.loads((folder / 'config_used.json').read_text()), config)
                self.assertEqual(len(pd.read_csv(folder / 'raw_data.csv')), 4)
                self.assertEqual(len(pd.read_csv(folder / 'summary_data.csv')), 2)
                outputs.append((raw, summary, series, genotypes))
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
            genotypes = pd.read_csv(folder / 'genotype_time_series_data.csv')
            self.assertTrue(genotypes.empty)
            self.assertEqual(genotypes.columns.tolist(), ['run_id', 'seed', 'step', 'genotype', 'population'])
            series = pd.read_csv(folder / 'time_series_data.csv')
            self.assertEqual(series['step'].tolist(), [1, 1000])
            self.assertTrue((series[['population', 'altruism_count']] == 0).all().all())
            self.assertEqual(raw.iloc[0]['death_count'], 0)
            self.assertTrue(np.isnan(summary.iloc[0]['avg_death_age']))


if __name__ == '__main__':
    unittest.main()
