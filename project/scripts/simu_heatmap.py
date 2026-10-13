import argparse
import concurrent.futures
import inspect
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.simu_every_para import expand_grid_parameters
from src.fields.field_v3 import FieldV3

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "exp_v3_heatmap.json"
PARAMETER_KEYS = ["death_threshold", "transfer_amount"]


def run_single_simulation(field_class, parameters, simulation, fixed_params, run_id=0):
    seed = simulation["base_seed"] + run_id
    random.seed(seed)
    np.random.seed(seed)
    field = field_class(**{**fixed_params, **parameters})
    eval_start = simulation["total_steps"] - simulation["evaluation_steps"] + 1
    majority_steps = 0
    for step in range(1, simulation["total_steps"] + 1):
        field.step()
        if (step >= eval_start and field.acting_agent_count > 0
                and 2 * field.positive_altruism_count > field.acting_agent_count):
            majority_steps += 1
        if not field.agents:
            break
    return {**parameters, "run_id": run_id, "seed": seed,
            "majority_steps": majority_steps}


def run_condition(task):
    parameters, simulation, fixed_params = task
    records = []
    for run_id in range(simulation["num_runs"]):
        records.append(run_single_simulation(
            FieldV3, parameters, simulation, fixed_params, run_id))
        print(f"Completed: {parameters}, run {run_id + 1}/{simulation['num_runs']}",
              flush=True)
    return records


def summarize_runs(records):
    return records.groupby(PARAMETER_KEYS, as_index=False).agg(
        num_runs=("run_id", "count"),
        avg_majority_steps=("majority_steps", "mean"),
        std_majority_steps=("majority_steps", "std"),
    )


def run_experiment(config, data_dir=None):
    if config.get("field_class") != "FieldV3" or config.get("agent_class") != "AgentV3":
        raise ValueError("This experiment requires FieldV3 and AgentV3.")
    simulation = config["simulation"]
    for key in ("total_steps", "evaluation_steps", "num_runs", "workers"):
        if type(simulation[key]) is not int or simulation[key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    if simulation["evaluation_steps"] > simulation["total_steps"]:
        raise ValueError("evaluation_steps cannot exceed total_steps.")
    if type(simulation["base_seed"]) is not int or not (
        0 <= simulation["base_seed"] <= 2**32 - simulation["num_runs"]
    ):
        raise ValueError("All run seeds must fit NumPy's seed range.")
    name = config["experiment_name"]
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or "/" in name or "\\" in name):
        raise ValueError("experiment_name must be a directory name.")
    if set(config["grid_parameters"]) != set(PARAMETER_KEYS):
        raise ValueError("Grid must contain death_threshold and transfer_amount only.")
    combinations = expand_grid_parameters(config["grid_parameters"])
    fixed_params = config["fixed_field_parameters"]
    for parameters in combinations:
        inspect.signature(FieldV3).bind(**{**fixed_params, **parameters})
    tasks = [(parameters, simulation, fixed_params) for parameters in combinations]
    print(f"Starting {len(tasks)} conditions x {simulation['num_runs']} runs "
          f"with {simulation['workers']} workers...", flush=True)
    if simulation["workers"] == 1:
        groups = [run_condition(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=simulation["workers"]
        ) as executor:
            groups = list(executor.map(run_condition, tasks))
    raw = pd.DataFrame([record for group in groups for record in group])
    summary = summarize_runs(raw)
    output_dir = Path(DATA_DIR if data_dir is None else data_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_used.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    raw.to_csv(output_dir / "raw_data.csv", index=False)
    summary.to_csv(output_dir / "summary_data.csv", index=False)
    print(f"Results saved to: {output_dir}", flush=True)
    return raw, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as file:
        config = json.load(file)
    run_experiment(config)


if __name__ == "__main__":
    main()
