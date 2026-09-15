import itertools
import concurrent.futures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src import Field

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# Simulation parameters
TOTAL_STEPS = 100000
EVAL_START_STEP = 90001
NUM_RUNS = 10

# search space for parameters
DEATH_THRESHOLDS = np.linspace(-200.0, -10.0, 10) 
TRANSFER_AMOUNTS = np.linspace(0.5, 10.0, 5)     

def run_single_simulation(death_threshold, transfer_amount):
    """Run a single simulation with specified parameters and return the count of steps where majority of agents performed altruistic actions in the last 10,000 steps."""
    field = Field(
        death_threshold=death_threshold,
        transfer_amount=transfer_amount
    )
    
    target_step_count = 0

    for step_num in range(1, TOTAL_STEPS + 1):
        field.step()
        
        # Evaluate only after EVAL_START_STEP to focus on the last 10,000 steps
        if step_num >= EVAL_START_STEP:
            population = len(field.agents)
            # Count the step if more than half of the agents performed altruistic actions
            if population > 0 and field.altruism_count > (population / 2.0):
                target_step_count += 1
        
        field.altruism_count = 0
        
    return target_step_count

def evaluate_parameter_set(params):
    """Evaluate a specific parameter set by running 100 simulations and returning the average"""
    dt, ta = params
    results = []
    
    for _ in range(NUM_RUNS):
        res = run_single_simulation(dt, ta)
        results.append(res)
    
    avg_steps = np.mean(results)
    return dt, ta, avg_steps

def main():
    print("Starting grid search for parameter evaluation...")
    
    # Create a grid of parameter combinations to evaluate
    param_grid = list(itertools.product(DEATH_THRESHOLDS, TRANSFER_AMOUNTS))
    results_data = []
    
    # Use ProcessPoolExecutor to parallelize the evaluation of parameter sets
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for dt, ta, avg_steps in executor.map(evaluate_parameter_set, param_grid):
            print(f"Completed: death_threshold={dt:6.1f}, transfer_amount={ta:4.1f} -> Average Steps: {avg_steps:.1f}")
            results_data.append({
                "death_threshold": dt,
                "transfer_amount": ta,
                "avg_majority_altruism_steps": avg_steps
            })

    # Save the results to a CSV file for further analysis
    df = pd.DataFrame(results_data)
    csv_path = DATA_DIR / "grid_search_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSimulation completed. Results saved to: {csv_path}")

if __name__ == "__main__":
    main()