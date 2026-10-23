"""変更(v7): .cuのCUDAカーネルを使うrunner。集計・CSV形式はv6と共通。"""
import argparse
import json
from pathlib import Path
from scripts.simu_every_para_up_v6 import run_experiment as _run_experiment


def run_experiment(config, data_dir=None):
    # 変更(v7): CUDA専用のバージョン検証とデバイス初期化を選択する。
    return _run_experiment(config, data_dir, version=7)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a CUDA simulation experiment.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    with Path(args.config).open(encoding="utf-8") as file:
        run_experiment(json.load(file))


if __name__ == "__main__":
    main()
