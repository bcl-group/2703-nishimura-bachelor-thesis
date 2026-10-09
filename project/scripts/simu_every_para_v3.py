import argparse
import concurrent.futures
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src import FieldV3

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TOTAL_STEPS = 200000
EVAL_START_STEP = 100001
NUM_RUNS = 1
BASE_SEED = 42

DEATH_THRESHOLDS = (-100, -85, -70, -55, -40, -25, -10)
MAX_TRANSFERS = (5, 10)
MUTATION_RATES = (0.0, 0.05)


def mean_and_std(total, squared_total, count):
    if count == 0:
        return np.nan, np.nan
    mean = total / count
    std = np.sqrt(max(0.0, (squared_total - total * mean) / (count - 1))) if count > 1 else np.nan
    return mean, std


def run_single_simulation(death_threshold, mutation_rate, max_transfer, run_id=0):
    if not 1 <= EVAL_START_STEP <= TOTAL_STEPS:
        raise ValueError("The evaluation start must be within the simulation.")
    seed = BASE_SEED + run_id
    random.seed(seed)
    np.random.seed(seed)
    field = FieldV3(
        death_threshold=death_threshold,
        mutation_rate=mutation_rate,
        max_transfer=max_transfer,
    )
    population_sum = 0
    population_squared_sum = 0
    altruism_sum = 0
    altruism_squared_sum = 0
    transfer_sum = 0.0
    transfer_squared_sum = 0.0
    for step in range(1, TOTAL_STEPS + 1):
        field.step()
        if step >= EVAL_START_STEP:
            population = len(field.agents)
            population_sum += population
            population_squared_sum += population ** 2
            altruism_sum += field.altruism_count
            altruism_squared_sum += field.altruism_count ** 2
            transfer_sum += field.transfer_total
            transfer_squared_sum += field.transfer_squared_total
        if not field.agents:
            break

    evaluation_steps = TOTAL_STEPS - EVAL_START_STEP + 1
    avg_population, std_population = mean_and_std(population_sum, population_squared_sum, evaluation_steps)
    avg_altruism_count, std_altruism_count = mean_and_std(altruism_sum, altruism_squared_sum, evaluation_steps)
    avg_transfer, std_transfer = mean_and_std(transfer_sum, transfer_squared_sum, altruism_sum)
    return {
        "mutation_rate": mutation_rate,
        "max_transfer": max_transfer,
        "death_threshold": death_threshold,
        "run_id": run_id,
        "seed": seed,
        "avg_population": avg_population,
        "std_population": std_population,
        "avg_altruism_count": avg_altruism_count,
        "std_altruism_count": std_altruism_count,
        "avg_transfer": avg_transfer,
        "std_transfer": std_transfer,
    }


def summarize_runs(records):
    return records.groupby(["mutation_rate", "max_transfer", "death_threshold"], as_index=False).agg(
        num_runs=("run_id", "count"),
        avg_population=("avg_population", "mean"),
        std_population=("avg_population", "std"),
        avg_within_run_std_population=("std_population", "mean"),
        avg_altruism_count=("avg_altruism_count", "mean"),
        std_altruism_count=("avg_altruism_count", "std"),
        avg_within_run_std_altruism_count=("std_altruism_count", "mean"),
        avg_transfer=("avg_transfer", "mean"),
        std_transfer=("avg_transfer", "std"),
        avg_within_run_std_transfer=("std_transfer", "mean"),
        transfer_valid_runs=("avg_transfer", "count"),
    )


def run_condition(params):
    death_threshold, mutation_rate, max_transfer = params
    records = []
    for run_id in range(NUM_RUNS):
        records.append(run_single_simulation(death_threshold, mutation_rate, max_transfer, run_id))
        print(
            f"Completed: death_threshold={death_threshold}, mutation_rate={mutation_rate:g}, "
            f"max_transfer={max_transfer}, run {run_id + 1}/{NUM_RUNS}",
            flush=True,
        )
    return records


def main(workers=5):
    if workers <= 0 or NUM_RUNS <= 0 or not 1 <= EVAL_START_STEP <= TOTAL_STEPS:
        raise ValueError("Workers/runs must be positive and the evaluation window must be valid.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "raw_data.csv"
    summary_path = DATA_DIR / "summary_data.csv"
    conditions = list(itertools.product(DEATH_THRESHOLDS, MUTATION_RATES, MAX_TRANSFERS))
    print(f"Starting {len(conditions)} conditions x {NUM_RUNS} runs with {workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for index, records in enumerate(executor.map(run_condition, conditions)):
            raw = pd.DataFrame(records)
            summary = summarize_runs(raw)
            mode = "w" if index == 0 else "a"
            raw.to_csv(raw_path, mode=mode, header=index == 0, index=False)
            summary.to_csv(summary_path, mode=mode, header=index == 0, index=False)
    print(f"Run averages: {raw_path}\nCondition means and sample SD: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save evaluation-period means per run and statistics across runs.")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel conditions (default: 5).")
    args = parser.parse_args()
    main(workers=args.workers)
