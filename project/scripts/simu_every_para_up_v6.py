import argparse
import csv
import itertools
import inspect
import json
import time
from contextlib import ExitStack, contextmanager
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
    field_class, param_combination, sim_config, fixed_params, run_id=0,
    time_series_records=None, genotype_records=None,
):
    from .batch_runner_v6 import run_batch
    records, series, genotypes = run_batch(field_class, [(param_combination, run_id)],
        {"device": "cpu", **sim_config}, fixed_params)
    if run_id == 0 and time_series_records is not None:
        time_series_records.extend(series)
    if run_id == 0 and genotype_records is not None:
        genotype_records.extend(genotypes)
    return records[0]


@contextmanager
def deterministic_execution():
    import torch
    # Require deterministic GPU reductions and restore the caller settings.
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def summarize_runs(records, grid_param_keys):
    groups = (records.groupby(grid_param_keys, as_index=False) if grid_param_keys
              else records.groupby(lambda _: 0))
    summary = (
        groups
        .agg(
            num_runs=("run_id", "count"),
            avg_population=("avg_population", "mean"),
            std_population=("avg_population", "std"),
            avg_within_run_std_population=("std_population", "mean"),
            avg_altruism_count=("avg_altruism_count", "mean"),
            std_altruism_count=("avg_altruism_count", "std"),
            avg_within_run_std_altruism_count=("std_altruism_count", "mean"),
            avg_altruism_rate=("avg_altruism_rate", "mean"),
            std_altruism_rate=("avg_altruism_rate", "std"),
            avg_death_age=("avg_death_age", "mean"),
            std_death_age=("avg_death_age", "std"),
            avg_within_run_std_death_age=("std_death_age", "mean"),
        )
    )

    return summary

def run_condition(args_tuple):
    from .batch_runner_v6 import run_batch
    field_class, combination, simulation, fixed = args_tuple
    jobs = [(combination, run_id) for run_id in range(simulation['num_runs'])]
    return run_batch(field_class, jobs, {"device": "cpu", **simulation}, fixed)


def _iter_condition_results(tasks, workers):
    from .batch_runner_v6 import run_with_retry
    # Group incompatible spatial shapes, then batch conditions AND repetitions.
    groups = {}
    for field_class, combo, simulation, fixed in tasks:
        size = {**fixed, **combo}.get("field_size", 10)
        groups.setdefault(size, []).extend((combo, run_id) for run_id in range(simulation['num_runs']))
    if not tasks:
        return
    field_class, _, simulation, fixed = tasks[0]
    for jobs in groups.values():
        batch_size = simulation.get('batch_size') or len(jobs)
        for start in range(0, len(jobs), batch_size):
            yield from run_with_retry(field_class, jobs[start:start+batch_size], simulation, fixed)


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


def run_experiment(config, data_dir=None, *, version=6):
    # Measure total wall time, including worker startup, aggregation and CSV writes.
    experiment_started_at = time.perf_counter()
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
    if version not in (6, 7):
        raise ValueError("This runner supports v6 and v7 only.")
    record_history = version == 6
    batch_size = sim_config.get("batch_size")
    if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
        raise ValueError("batch_size must be a positive integer or null.")
    # Validate dependencies, device and arguments before creating output files.
    if field_class.__name__ != f"FieldV{version}" or agent_class.__name__ != f"AgentV{version}":
        raise ValueError(f"runner v{version} requires FieldV{version} and AgentV{version}.")
    if workers != 1:
        raise ValueError("v6/v7 uses one GPU process; workers must be 1.")
    from src.fields.field_v6 import FieldV6, require_device
    sim_config.setdefault("device", "cuda:0")
    require_device(sim_config["device"])
    if agent_class.__name__.removeprefix("Agent") != field_class.__name__.removeprefix("Field"):
        raise ValueError("agent_class and field_class must use the same version.")
    combinations = expand_grid_parameters(grid_params)
    for combination in combinations:
        kwargs = {**fixed_params, **combination}
        if {"device", "seed"} & kwargs.keys():
            raise ValueError("Set device and base_seed in simulation, not field parameters.")
        inspect.signature(FieldV6).bind(**kwargs)
        # 初期化による検証も出力作成前に済ませる。
        field_class(**kwargs, device=sim_config["device"], seed=sim_config["base_seed"])

    exp_data_dir = Path(DATA_DIR if data_dir is None else data_dir) / exp_name
    exp_data_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_data_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    raw_path = exp_data_dir / "raw_data.csv"
    summary_path = exp_data_dir / "summary_data.csv"
    time_series_path = exp_data_dir / "time_series_data.csv"
    genotype_path = exp_data_dir / "genotype_time_series_data.csv"

    grid_keys = list(grid_params.keys())

    print(f"--- Experiment: {exp_name} ---")
    print(f"Target Field Class: {config['field_class']}")
    print(
        f"Starting {len(combinations)} conditions x {num_runs} runs in batches on {sim_config['device']}..."
    )

    tasks = [
        (field_class, combo, sim_config, fixed_params) for combo in combinations
    ]

    all_raw_records = []
    with ExitStack() as stack:
        if record_history:
            series_file = stack.enter_context(time_series_path.open("w", newline="", encoding="utf-8"))
            genotype_file = stack.enter_context(genotype_path.open("w", newline="", encoding="utf-8"))
            series_writer = csv.DictWriter(
                series_file,
                fieldnames=[*grid_keys, "run_id", "seed", "step", "population", "altruism_count", "altruism_rate"],
            )
            genotype_writer = csv.DictWriter(
                genotype_file,
                fieldnames=[*grid_keys, "run_id", "seed", "step", "genotype", "population"],
            )
            series_writer.writeheader()
            genotype_writer.writeheader()
        for records, time_series_records, genotype_records in _iter_condition_results(tasks, workers):
            all_raw_records.extend(records)
            if record_history:
                series_writer.writerows(time_series_records)
                genotype_writer.writerows(genotype_records)

    condition_order = {tuple(combo[key] for key in grid_keys): i for i, combo in enumerate(combinations)}
    all_raw_records.sort(key=lambda record: (
        condition_order[tuple(record[key] for key in grid_keys)], record['run_id']))
    raw_df = pd.DataFrame(all_raw_records)
    summary_df = summarize_runs(raw_df, grid_keys)
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    experiment_elapsed_seconds = time.perf_counter() - experiment_started_at

    print(f"\nSimulation Complete!")
    print(f"Total elapsed time: {experiment_elapsed_seconds:.3f} s")
    print(f"Raw data saved to: {raw_path}")
    print(f"Summary data saved to: {summary_path}")
    if record_history:
        print(f"Time series data saved to: {time_series_path}")
        print(f"Genotype time series data saved to: {genotype_path}")

    return raw_df, summary_df


def main(argv=None, *, version=6):
    parser = argparse.ArgumentParser(description="Run a simulation experiment from JSON config.")
    parser.add_argument("--config", required=True, help="Path to an experiment JSON file")
    args = parser.parse_args(argv)
    with Path(args.config).open(encoding="utf-8") as file:
        config = json.load(file)
    run_experiment(config, version=version)


if __name__ == "__main__":
    main()
