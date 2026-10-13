import argparse
import concurrent.futures
import itertools
import inspect
import json
import random
from pathlib import Path
from decimal import Decimal

import numpy as np
import pandas as pd

from src.factory import get_agent_class, get_field_class

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def mean_and_std(total, squared_total, count):
    if count == 0:
        return np.nan, np.nan
    mean = total / count
    std = (
        np.sqrt(max(0.0, (squared_total - total * mean) / (count - 1)))
        if count > 1
        else np.nan
    )
    return mean, std

def run_single_simulation(
    field_class,
    param_combination,
    sim_config,
    fixed_params,
    run_id=0,
    time_series_records=None,
):
    total_steps = sim_config["total_steps"]
    eval_start_step = sim_config["eval_start_step"]
    base_seed = sim_config["base_seed"]

    if not 1 <= eval_start_step <= total_steps:
        raise ValueError("The evaluation start must be within the simulation steps.")

    seed = base_seed + run_id
    random.seed(seed)
    np.random.seed(seed)

    field_kwargs = {**fixed_params, **param_combination}

    field = field_class(**field_kwargs)

    population_sum = 0
    population_squared_sum = 0
    altruism_sum = 0
    altruism_squared_sum = 0
    record_time_series = run_id == 0 and time_series_records is not None
    interval_altruism_count = 0

    def record_sample(sample_step, population):
        time_series_records.append({
            **param_combination,
            "run_id": run_id,
            "seed": seed,
            "step": sample_step,
            "population": population,
            "altruism_count": interval_altruism_count,
        })

    for step in range(1, total_steps + 1):
        field.step()
        if step >= eval_start_step:
            if record_time_series:
                interval_altruism_count += field.altruism_count
                if step % 1000 == 0:
                    record_sample(step, len(field.agents))
                    interval_altruism_count = 0
            population = len(field.agents)
            population_sum += population
            population_squared_sum += population**2
            altruism_sum += field.altruism_count
            altruism_squared_sum += field.altruism_count**2

        if not field.agents:
            if record_time_series:
                for sample_step in range((step // 1000 + 1) * 1000, total_steps + 1, 1000):
                    record_sample(sample_step, 0)
                    interval_altruism_count = 0
            break

    evaluation_steps = total_steps - eval_start_step + 1
    avg_population, std_population = mean_and_std(
        population_sum, population_squared_sum, evaluation_steps
    )
    avg_altruism_count, std_altruism_count = mean_and_std(
        altruism_sum, altruism_squared_sum, evaluation_steps
    )

    result = {
        "run_id": run_id,
        "seed": seed,
        "avg_population": avg_population,
        "std_population": std_population,
        "avg_altruism_count": avg_altruism_count,
        "std_altruism_count": std_altruism_count,
    }
    result.update(param_combination)
    return result

def summarize_runs(records, grid_param_keys):
    groups = (records.groupby(grid_param_keys, as_index=False) if grid_param_keys
              else records.groupby(lambda _: 0))
    return (
        groups
        .agg(
            num_runs=("run_id", "count"),
            avg_population=("avg_population", "mean"),
            std_population=("avg_population", "std"),
            avg_within_run_std_population=("std_population", "mean"),
            avg_altruism_count=("avg_altruism_count", "mean"),
            std_altruism_count=("avg_altruism_count", "std"),
            avg_within_run_std_altruism_count=("std_altruism_count", "mean"),
        )
    )


def run_condition(args_tuple):
    field_class, param_combination, sim_config, fixed_params = args_tuple
    num_runs = sim_config["num_runs"]
    records = []
    time_series_records = []

    for run_id in range(num_runs):
        rec = run_single_simulation(
            field_class=field_class,
            param_combination=param_combination,
            sim_config=sim_config,
            fixed_params=fixed_params,
            run_id=run_id,
            time_series_records=time_series_records if run_id == 0 else None,
        )
        records.append(rec)

        param_str = ", ".join([f"{k}={v}" for k, v in param_combination.items()])
        print(
            f"Completed: [{param_str}], run {run_id + 1}/{num_runs}",
            flush=True,
        )

    return records, time_series_records


def expand_grid_parameters(grid_params):
    expanded = {}
    for name, values in grid_params.items():
        if isinstance(values, dict):
            if set(values) != {"start", "stop", "step"}:
                raise ValueError(f"Range for {name} needs start, stop and step.")
            start, stop, step = (Decimal(str(values[key])) for key in ("start", "stop", "step"))
            if not all(value.is_finite() for value in (start, stop, step)) or step == 0:
                raise ValueError(f"Range for {name} must be finite with a nonzero step.")
            values = []
            current = start
            while (current < stop if step > 0 else current > stop):
                values.append(int(current) if current == current.to_integral_value() else float(current))
                current += step
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid parameter {name} must contain at least one value.")
        expanded[name] = values
    keys = list(expanded)
    return [dict(zip(keys, values)) for values in itertools.product(*expanded.values())]


def run_experiment(config, data_dir=None):
    exp_name = config.get("experiment_name", "default_exp")
    if not isinstance(exp_name, str) or not exp_name or exp_name in {".", ".."} or "/" in exp_name or "\\" in exp_name:
        raise ValueError("experiment_name must be a directory name.")
    sim_config = {"workers": 1, "num_runs": 1, "base_seed": 42, **config["simulation"]}
    grid_params = config["grid_parameters"]
    fixed_params = config.get("fixed_field_parameters", {})
    workers = sim_config["workers"]
    num_runs = sim_config["num_runs"]
    eval_start_step = sim_config["eval_start_step"]
    total_steps = sim_config["total_steps"]
    if any(type(value) is not int for value in (workers, num_runs, total_steps, eval_start_step)) or workers <= 0 or num_runs <= 0 or not 1 <= eval_start_step <= total_steps:
        raise ValueError("Workers/runs must be positive integers and the evaluation window must be valid.")

    field_class = get_field_class(config["field_class"])
    agent_class = get_agent_class(config["agent_class"])
    if agent_class.__name__.removeprefix("Agent") != field_class.__name__.removeprefix("Field"):
        raise ValueError("agent_class and field_class must use the same version.")
    combinations = expand_grid_parameters(grid_params)
    for combination in combinations:
        inspect.signature(field_class).bind(**{**fixed_params, **combination})

    exp_data_dir = Path(DATA_DIR if data_dir is None else data_dir) / exp_name
    exp_data_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_data_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    raw_path = exp_data_dir / "raw_data.csv"
    summary_path = exp_data_dir / "summary_data.csv"
    time_series_path = exp_data_dir / "time_series_data.csv"

    grid_keys = list(grid_params.keys())

    print(f"--- Experiment: {exp_name} ---")
    print(f"Target Field Class: {config['field_class']}")
    print(
        f"Starting {len(combinations)} conditions x {num_runs} runs with {workers} workers..."
    )

    tasks = [
        (field_class, combo, sim_config, fixed_params) for combo in combinations
    ]

    all_raw_records = []
    all_time_series_records = []

    if workers == 1:
        for task in tasks:
            records, time_series_records = run_condition(task)
            all_raw_records.extend(records)
            all_time_series_records.extend(time_series_records)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for records, time_series_records in executor.map(run_condition, tasks):
                all_raw_records.extend(records)
                all_time_series_records.extend(time_series_records)

    raw_df = pd.DataFrame(all_raw_records)
    summary_df = summarize_runs(raw_df, grid_keys)
    time_series_df = pd.DataFrame(
        all_time_series_records,
        columns=[*grid_keys, "run_id", "seed", "step", "population", "altruism_count"],
    ).sort_values([*grid_keys, "step"]).reset_index(drop=True)

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    time_series_df.to_csv(time_series_path, index=False)

    print(f"\nSimulation Complete!")
    print(f"Raw data saved to: {raw_path}")
    print(f"Summary data saved to: {summary_path}")
    print(f"Time series data saved to: {time_series_path}")

    return raw_df, summary_df


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a simulation experiment from JSON config.")
    parser.add_argument("--config", required=True, help="Path to an experiment JSON file")
    args = parser.parse_args(argv)
    with Path(args.config).open(encoding="utf-8") as file:
        config = json.load(file)
    run_experiment(config)


if __name__ == "__main__":
    main()
