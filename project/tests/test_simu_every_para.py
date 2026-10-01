import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from scripts import simu_every_para as simulation


class SimulationTests(unittest.TestCase):
    def test_parameter_grid_includes_endpoints_and_all_400_pairs(self):
        self.assertEqual(list(simulation.DEATH_THRESHOLDS), list(range(-200, 0, 10)))
        self.assertEqual(list(simulation.MAX_TRANSFERS), list(range(1, 21)))
        self.assertEqual(len(list(itertools.product(simulation.DEATH_THRESHOLDS, simulation.MAX_TRANSFERS))), 400)
        self.assertEqual(simulation.TOTAL_STEPS - simulation.EVAL_START_STEP + 1, 100000)

    def run_population_history(self, populations):
        field = Mock()
        field.step.side_effect = lambda: setattr(field, 'agents', [None] * next(populations))
        with patch.object(simulation, 'TOTAL_STEPS', 6), \
             patch.object(simulation, 'EVAL_START_STEP', 4), \
             patch.object(simulation, 'Field', return_value=field) as constructor:
            result = simulation.run_single_simulation(-200, 20)
        constructor.assert_called_once_with(death_threshold=-200, max_transfer=20)
        return result

    def test_only_second_half_contributes_to_mean(self):
        self.assertEqual(self.run_population_history(iter([100, 100, 100, 2, 4, 6])), 4)

    def test_extinction_keeps_zero_population_in_denominator(self):
        self.assertEqual(self.run_population_history(iter([100, 100, 100, 6, 0])), 2)
        self.assertEqual(self.run_population_history(iter([0])), 0)

    def test_csv_contains_population_results_for_every_pair(self):
        executor = Mock()
        executor.map.side_effect = lambda func, pairs: (
            (dt, mt, 2.5) for dt, mt in pairs
        )
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(simulation, 'DATA_DIR', Path(directory)), \
             patch.object(simulation.concurrent.futures, 'ProcessPoolExecutor') as pool, \
             patch('builtins.print'):
            pool.return_value.__enter__.return_value = executor
            simulation.main()
            data = pd.read_csv(Path(directory) / 'grid_search_results.csv')

        self.assertEqual(list(data.columns), ['death_threshold', 'max_transfer', 'avg_population'])
        self.assertEqual(len(data), 400)
        self.assertEqual(len(data[['death_threshold', 'max_transfer']].drop_duplicates()), 400)
        self.assertTrue((data['avg_population'] == 2.5).all())


if __name__ == '__main__':
    unittest.main()
