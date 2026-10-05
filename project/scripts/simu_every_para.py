import itertools
import concurrent.futures
import numpy as np
import pandas as pd
from pathlib import Path

from src import Field

DATA_DIR = Path("./data")

# Simulation parameters
TOTAL_STEPS = 200000
EVAL_START_STEP = 100001
NUM_RUNS = 10

# search space for parameters
DEATH_THRESHOLDS = range(-200, 0, 5)
MAX_TRANSFERS = range(10, 31, 10)

def run_single_simulation(death_threshold, max_transfer):
    """Return mean population and altruistic actions per step in the evaluation window."""
    field = Field(
        death_threshold=death_threshold,
        max_transfer=max_transfer
    )
    
    population_sum = 0
    altruism_sum = 0

    for step_num in range(1, TOTAL_STEPS + 1):
        field.altruism_count = 0
        field.step()
        
        if step_num >= EVAL_START_STEP:
            population_sum += len(field.agents)
            altruism_sum += field.altruism_count
        # An extinct population stays zero; retain the full evaluation denominator.
        if not field.agents:
            break
        
    evaluation_steps = TOTAL_STEPS - EVAL_START_STEP + 1
    return population_sum / evaluation_steps, altruism_sum / evaluation_steps

def evaluate_parameter_set(params):
    """Average both temporal means over NUM_RUNS independent runs."""
    death_threshold, max_transfer = params
    results = [run_single_simulation(death_threshold, max_transfer) for _ in range(NUM_RUNS)]
    avg_population, avg_altruism_count = np.mean(results, axis=0)
    return death_threshold, max_transfer, float(avg_population), float(avg_altruism_count)

def main():
    print("Starting grid search for parameter evaluation...")
    
    # Create a grid of parameter combinations to evaluate
    param_grid = list(itertools.product(DEATH_THRESHOLDS, MAX_TRANSFERS))
    results_data = []
    
    # Use ProcessPoolExecutor to parallelize the evaluation of parameter sets
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for dt, mt, avg_population, avg_altruism_count in executor.map(evaluate_parameter_set, param_grid):
            print(
                f"Completed: death_threshold={dt:6.1f}, max_transfer={mt:4.1f} "
                f"-> Average Population: {avg_population:.3f}, "
                f"Average Altruistic Actions: {avg_altruism_count:.3f}"
            )
            results_data.append({
                "death_threshold": dt,
                "max_transfer": mt,
                "avg_population": avg_population,
                "avg_altruism_count": avg_altruism_count
            })

    # Save the results to a CSV file for further analysis
    df = pd.DataFrame(results_data)
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / "grid_search_results_food=5.0.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSimulation completed. Results saved to: {csv_path}")

if __name__ == "__main__":
    main()
