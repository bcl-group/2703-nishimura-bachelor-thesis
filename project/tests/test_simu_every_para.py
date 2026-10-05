import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from scripts import simu_every_para as simulation


class SimulationTests(unittest.TestCase):
    def test_evaluation_window_has_100000_steps(self):
        self.assertEqual(simulation.TOTAL_STEPS - simulation.EVAL_START_STEP + 1, 100000)

    def run_history(self, history):
        field = Mock()
        field.altruism_count = 0

        def step():
            population, altruism_count = next(history)
            field.agents = [None] * population
            field.altruism_count += altruism_count

        field.step.side_effect = step
        with patch.object(simulation, 'TOTAL_STEPS', 6), \
             patch.object(simulation, 'EVAL_START_STEP', 4), \
             patch.object(simulation, 'Field', return_value=field) as constructor:
            result = simulation.run_single_simulation(-200, 20)
        constructor.assert_called_once_with(death_threshold=-200, max_transfer=20)
        return result

    def test_only_second_half_contributes_to_mean(self):
        history = iter([(100, 90), (100, 90), (100, 90), (2, 1), (4, 2), (6, 3)])
        self.assertEqual(self.run_history(history), (4, 2))

    def test_extinction_keeps_zero_population_in_denominator(self):
        history = iter([(100, 90), (100, 90), (100, 90), (6, 4), (0, 2)])
        self.assertEqual(self.run_history(history), (2, 2))
        self.assertEqual(self.run_history(iter([(0, 0)])), (0, 0))

    def test_both_metrics_are_averaged_across_runs(self):
        with patch.object(simulation, 'NUM_RUNS', 2), \
             patch.object(simulation, 'run_single_simulation', side_effect=[(2, 1), (4, 3)]):
            self.assertEqual(simulation.evaluate_parameter_set((-200, 20)), (-200, 20, 3, 2))

    def test_csv_contains_population_results_for_every_pair(self):
        executor = Mock()
        executor.map.side_effect = lambda func, pairs: (
            (dt, mt, 2.5, 1.5) for dt, mt in pairs
        )
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(simulation, 'DATA_DIR', Path(directory)), \
             patch.object(simulation.concurrent.futures, 'ProcessPoolExecutor') as pool, \
             patch('builtins.print'):
            pool.return_value.__enter__.return_value = executor
            simulation.main()
            data = pd.read_csv(Path(directory) / 'grid_search_results_food=5.0.csv')

        self.assertEqual(list(data.columns), ['death_threshold', 'max_transfer', 'avg_population', 'avg_altruism_count'])
        expected_pairs = set(itertools.product(simulation.DEATH_THRESHOLDS, simulation.MAX_TRANSFERS))
        self.assertEqual(len(data), len(expected_pairs))
        self.assertEqual(set(zip(data['death_threshold'], data['max_transfer'])), expected_pairs)
        self.assertTrue((data['avg_population'] == 2.5).all())
        self.assertTrue((data['avg_altruism_count'] == 1.5).all())


if __name__ == '__main__':
    unittest.main()
