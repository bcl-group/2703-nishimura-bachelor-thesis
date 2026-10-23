import argparse
import csv
import itertools
import inspect
import json
import time
from contextlib import contextmanager
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
    import torch
    total_steps = sim_config["total_steps"]
    eval_start_step = sim_config["eval_start_step"]
    seed = sim_config["base_seed"] + run_id
    # 変更(v6/v7): 乱数生成器はFieldごとに分離し、統計も同じデバイスで蓄積する。
    with deterministic_execution():
        field = field_class(**{**fixed_params, **param_combination},
                            device=sim_config.get("device", "cpu"), seed=seed)
        stats = torch.zeros(7, dtype=torch.float64, device=field.device)
        interval = torch.zeros(2, dtype=torch.float64, device=field.device)
        record_series = run_id == 0 and time_series_records is not None
        record_genotypes = run_id == 0 and genotype_records is not None

        def record_sample(step, population):
            identifiers = {**param_combination, "run_id": run_id, "seed": seed, "step": step}
            if record_series:
                denominator, altruism = interval.cpu().tolist()
                time_series_records.append({**identifiers, "population": population,
                    "altruism_count": int(altruism),
                    "altruism_rate": altruism / denominator if denominator > 0 else None})
            if record_genotypes and population:
                genotype_records.extend(
                    {**identifiers, "genotype": code, "population": count}
                    for code, count in field.genotype_counts().items())

        for step in range(1, total_steps + 1):
            deaths = field.step()
            population = field.population
            if record_series or record_genotypes:
                interval[0] += population
                interval[1] += field.altruism_count
                if step == 1 or step % 1000 == 0:
                    record_sample(step, population)
                    interval.zero_()
            if step >= eval_start_step:
                stats[0] += population
                stats[1] += population ** 2
                stats[2] += field.altruism_count
                stats[3] += field.altruism_count.to(torch.float64).square()
                stats[4] += deaths.numel()
                death_values = deaths.to(torch.float64)
                stats[5] += death_values.sum()
                stats[6] += death_values.square().sum()
            if population == 0:
                if record_series or record_genotypes:
                    for sample_step in range((step // 1000 + 1) * 1000, total_steps + 1, 1000):
                        record_sample(sample_step, 0)
                        interval.zero_()
                break
        # 変更(v7): 最終転送はCUDAの完了待ちも兼ねるため、実行時間にGPU演算を含む。
        values = stats.cpu().tolist()
    evaluation_steps = total_steps - eval_start_step + 1
    avg_population, std_population = mean_and_std(values[0], values[1], evaluation_steps)
    avg_altruism, std_altruism = mean_and_std(values[2], values[3], evaluation_steps)
    avg_age, std_age = mean_and_std(values[5], values[6], values[4])
    return {"run_id": run_id, "seed": seed,
            "avg_population": avg_population, "std_population": std_population,
            "avg_altruism_count": avg_altruism, "std_altruism_count": std_altruism,
            "avg_altruism_rate": avg_altruism / avg_population if avg_population > 0 else None,
            "avg_death_age": avg_age, "std_death_age": std_age, **param_combination}


@contextmanager
def deterministic_execution():
    import torch
    # 変更(v6/v7): CUDAの加算も決定的実装を要求し、呼出元の設定は終了時に復元する。
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
    field_class, param_combination, sim_config, fixed_params = args_tuple
    num_runs = sim_config["num_runs"]
    records = []
    time_series_records = []
    genotype_records = []

    for run_id in range(num_runs):
        # 変更(v6/v7): 初期化・記録・CUDA完了待ちを含めて実行時間を測る。
        run_started_at = time.perf_counter()
        rec = run_single_simulation(
            field_class=field_class,
            param_combination=param_combination,
            sim_config=sim_config,
            fixed_params=fixed_params,
            run_id=run_id,
            time_series_records=time_series_records if run_id == 0 else None,
            genotype_records=genotype_records if run_id == 0 else None,
        )
        run_elapsed_seconds = time.perf_counter() - run_started_at
        records.append(rec)

        param_str = ", ".join([f"{k}={v}" for k, v in param_combination.items()])
        print(
            f"Completed: [{param_str}], run {run_id + 1}/{num_runs}, "
            f"elapsed: {run_elapsed_seconds:.3f} s",
            flush=True,
        )

    return records, time_series_records, genotype_records

def _iter_condition_results(tasks, workers):
    # 変更(v6/v7): 条件は逐次実行し、CUDAを複数プロセスから使用しない。
    yield from map(run_condition, tasks)


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
    # 変更(v6/v7): CUDA確認と引数検証は、出力ディレクトリを作る前に実施する。
    if field_class.__name__ != f"FieldV{version}" or agent_class.__name__ != f"AgentV{version}":
        raise ValueError(f"runner v{version} requires FieldV{version} and AgentV{version}.")
    if workers != 1:
        raise ValueError("v6/v7 runs conditions sequentially; workers must be 1.")
    from src.fields.field_v6 import FieldV6, require_device
    sim_config.setdefault("device", "cuda:0" if version == 7 else "cpu")
    require_device(sim_config["device"], cuda=version == 7)
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
        f"Starting {len(combinations)} conditions x {num_runs} runs with {workers} workers..."
    )

    tasks = [
        (field_class, combo, sim_config, fixed_params) for combo in combinations
    ]

    all_raw_records = []
    with time_series_path.open("w", newline="", encoding="utf-8") as series_file, \
            genotype_path.open("w", newline="", encoding="utf-8") as genotype_file:
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
            series_writer.writerows(time_series_records)
            genotype_writer.writerows(genotype_records)

    raw_df = pd.DataFrame(all_raw_records)
    summary_df = summarize_runs(raw_df, grid_keys)
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    experiment_elapsed_seconds = time.perf_counter() - experiment_started_at

    print(f"\nSimulation Complete!")
    print(f"Total elapsed time: {experiment_elapsed_seconds:.3f} s")
    print(f"Raw data saved to: {raw_path}")
    print(f"Summary data saved to: {summary_path}")
    print(f"Time series data saved to: {time_series_path}")
    print(f"Genotype time series data saved to: {genotype_path}")

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
