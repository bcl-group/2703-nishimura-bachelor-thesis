import itertools
import concurrent.futures
import numpy as np
import pandas as pd
from pathlib import Path

from src import Field

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

# Simulation parameters
TOTAL_STEPS = 100000
NUM_RUNS = 1

# search space for parameters
DEATH_THRESHOLDS = np.linspace(-200.0, -10.0, 10) 

def run_single_simulation(death_threshold):
    """Run a single simulation with specified parameters and return the history of the simulation."""
    field = Field(
        death_threshold=death_threshold,
    )
    
    history = []

    for step_num in range(1, TOTAL_STEPS + 1):
        field.step()
        
        population = len(field.agents)
        altruism = field.altruism_count
       
        if population > 0:
            avg_energy = sum(a.energy for a in field.agents) / population
        else:
            avg_energy = 0.0
            
        history.append({
            'death_threshold': death_threshold,
            'step': step_num,
            'altruism_count': altruism,
            'population': population,
            'avg_energy': avg_energy
        })
        
        field.altruism_count = 0 
        
    return history

def evaluate_parameter_set(params):
    dt = params
    history = run_single_simulation(dt)
    return dt, history

def main():
    print("Starting parameter evaluation and data generation...")
    
    all_results = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for dt, history in executor.map(evaluate_parameter_set, DEATH_THRESHOLDS):
            print(f"Completed: death_threshold={dt:6.1f}")
            all_results.extend(history)

    print("Saving data to CSV...")
    df = pd.DataFrame(all_results)
    csv_path = DATA_DIR / "simulation_history.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSimulation completed. Data saved to: {csv_path}")

if __name__ == "__main__":
    main()