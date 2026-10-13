import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from scripts import simu_heatmap as simulation
from src.agents.agent_v3 import AgentV3
from src.fields.field_v3 import FieldV3


class HeatmapTests(unittest.TestCase):
    def test_window_majority_and_extinction(self):
        field = Mock()
        # Warmup majority, exact half, majority, extinction with a majority,
        # followed by implicit zero steps through the end of the window.
        history = iter([(2, 2, 2), (4, 2, 4), (3, 2, 3), (1, 1, 0)])

        def step():
            actors, donors, remaining = next(history)
            field.acting_agent_count = actors
            field.positive_altruism_count = donors
            field.agents = [None] * remaining

        field.step.side_effect = step
        result = simulation.run_single_simulation(
            Mock(return_value=field), {},
            {'base_seed': 43, 'total_steps': 6, 'evaluation_steps': 5}, {}, 2)
        self.assertEqual(result['majority_steps'], 2)
        self.assertEqual(result['seed'], 45)
        self.assertEqual(field.step.call_count, 4)

    def test_empty_population_is_not_a_majority(self):
        field = Mock(acting_agent_count=0, positive_altruism_count=0, agents=[])
        record = simulation.run_single_simulation(
            Mock(return_value=field), {},
            {'base_seed': 43, 'total_steps': 6, 'evaluation_steps': 6}, {})
        self.assertEqual(record['majority_steps'], 0)

    def test_summary_sample_standard_deviation(self):
        raw = pd.DataFrame({
            'death_threshold': [-200] * 10, 'transfer_amount': [1] * 10,
            'run_id': range(10), 'majority_steps': range(10),
        })
        row = simulation.summarize_runs(raw).iloc[0]
        self.assertEqual(row['num_runs'], 10)
        self.assertEqual(row['avg_majority_steps'], 4.5)
        self.assertAlmostEqual(row['std_majority_steps'], np.std(range(10), ddof=1))

    def test_field_counters_exclude_nonpositive_transfers_and_reset(self):
        for energy, expected in [(10, 1), (0, 0), (-1, 0)]:
            with self.subTest(energy=energy):
                field = FieldV3(initial_agents=0, initial_food_prob=0,
                                step_food_prob=0, energy_loss_per_step=0)
                donor = AgentV3(0, 0, 0)
                receiver = AgentV3(0, 0, 0)
                dying = AgentV3(0, 0, 0)
                donor.energy = energy
                dying.energy = -50

                def give(*args):
                    donor.altruism_level = 1
                    return (1, 1)

                donor.decide_action = give
                receiver.decide_action = lambda *args: (1, 1)
                field.agents = [donor, receiver, dying]
                field.step()
                self.assertEqual(field.acting_agent_count, 2)
                self.assertEqual(field.positive_altruism_count, expected)
                self.assertEqual(field.altruism_count, 1)
                field.agents = []
                field.step()
                self.assertEqual(field.acting_agent_count, 0)
                self.assertEqual(field.positive_altruism_count, 0)

    def test_birth_does_not_change_denominator(self):
        field = FieldV3(initial_agents=0, initial_food_prob=0, step_food_prob=0)
        agent = AgentV3(0, 0, 0)
        agent.energy = 101
        agent.decide_action = lambda *args: (1, 1)
        field.agents = [agent]
        field.step()
        self.assertEqual(len(field.agents), 2)
        self.assertEqual(field.acting_agent_count, 1)

    def test_grid_csv_and_parallel_reproducibility(self):
        config = json.loads(Path('configs/exp_v3_heatmap.json').read_text())
        combinations = simulation.expand_grid_parameters(config['grid_parameters'])
        self.assertEqual(len(combinations), 200)
        self.assertEqual(combinations[0], {'death_threshold': -200, 'transfer_amount': 1})
        self.assertEqual(combinations[-1], {'death_threshold': -10, 'transfer_amount': 10})
        self.assertEqual(config['simulation']['num_runs'], 10)
        self.assertEqual(config['simulation']['total_steps'] - config['simulation']['evaluation_steps'] + 1, 190001)
        config['grid_parameters'] = {'death_threshold': [-200, -10], 'transfer_amount': [1, 10]}
        config['simulation'].update(total_steps=12, evaluation_steps=5, workers=1)
        config['fixed_field_parameters']['initial_agents'] = 4
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            raw, summary = simulation.run_experiment(config, directory)
            output = Path(directory) / config['experiment_name']
            self.assertEqual(len(pd.read_csv(output / 'raw_data.csv')), 40)
            self.assertEqual(len(pd.read_csv(output / 'summary_data.csv')), 4)
            self.assertEqual(json.loads((output / 'config_used.json').read_text()), config)
            parallel = copy.deepcopy(config)
            parallel['simulation']['workers'] = 2
            parallel_raw, parallel_summary = simulation.run_experiment(parallel, directory)
            pd.testing.assert_frame_equal(raw, parallel_raw)
            pd.testing.assert_frame_equal(summary, parallel_summary)
            invalid = copy.deepcopy(config)
            invalid['simulation']['evaluation_steps'] = 13
            with self.assertRaises(ValueError):
                simulation.run_experiment(invalid, directory)


if __name__ == '__main__':
    unittest.main()
