# エージェントシミュレーション

JSON 設定から v1〜v3 のモデルを選択し、条件ごとにシミュレーションと集計を実行します。

## ディレクトリ構成

```text
project/
├── .gitignore
├── README.md
├── proposed.md
├── project_env/
├── configs/
│   ├── default_config.json
│   ├── exp_v1_baseline.json
│   ├── exp_v2_grid.json
│   └── exp_v3.json
├── data/
│   └── <experiment_name>/
│       ├── config_used.json
│       ├── raw_data.csv
│       └── summary_data.csv
├── notebooks/
│   └── analysis_exp_v1.ipynb
├── scripts/
│   └── simu_every_para.py
├── src/
│   ├── __init__.py
│   ├── factory.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── agent_v2.py
│   │   └── agent_v3.py
│   └── fields/
│       ├── __init__.py
│       ├── field.py
│       ├── field_v2.py
│       └── field_v3.py
└── tests/
```

既存の `data/` と `images/` の内容は保持しています。実験別の出力フォルダは実行時に作成されます。

## 実行

`project/` を作業ディレクトリにして実行します。

```bash
source project_env/bin/activate
python -m scripts.simu_every_para --config configs/exp_v1_baseline.json
python -m scripts.simu_every_para --config configs/exp_v2_grid.json
```

実行には NumPy と pandas、ノートブックには Jupyter と matplotlib が必要です。仮想環境を新規作成する場合は次のように準備できます。

```bash
python3 -m venv project_env
source project_env/bin/activate
python -m pip install numpy pandas matplotlib jupyter
```

## JSON 設定

`default_config.json` をコピーし、実験名と条件を編集してください。

- `experiment_name`: 出力先のフォルダ名。同じ名前で再実行すると、その実験の3つの出力ファイルを上書きします。
- `agent_class` / `field_class`: `Agent` / `Field`、`AgentV2` / `FieldV2`、`AgentV3` / `FieldV3` の対応する組合せを指定します。
- `simulation`: `total_steps`、`eval_start_step`（1始まり・開始ステップを含む）、`num_runs`、`base_seed`、`workers` を指定します。試行数と並列数の省略時は1、シードは42です。
- `grid_parameters`: 配列の直積で探索します。`{"start": -70, "stop": -10, "step": 15}` の形式も使え、終端 `stop` は含みません。空のオブジェクトは固定条件で1条件を実行します。
- `fixed_field_parameters`: 全条件に共通する Field の引数です。同じ引数を探索条件にも指定した場合は探索値を使用します。

移譲上限の引数名は `max_transfer` です。`exp_v3.json` もこの名前に統一しています。

v1 のテンプレートと基準実験は1条件・1試行です。v2 は死亡閾値5通り、移譲上限2通り、突然変異率2通りの20条件を各10試行します。設定例は200,000ステップで、100,001ステップ以降を評価します。動作確認にはステップ数と試行数を小さくした設定を使用してください。

`workers: 1` は逐次実行、それ以上は条件単位のプロセス並列です。同じ試行番号では条件間で `base_seed + run_id` の同じシードを使用します。

## 出力と解析

- `config_used.json`: 読み込んだ設定のコピー。
- `raw_data.csv`: 条件・試行ごとの評価期間の平均と標本標準偏差、試行番号、シード。
- `summary_data.csv`: 条件ごとの試行平均の平均・標本標準偏差、試行内標準偏差の平均。

絶滅後の個体数・利他行動数は評価期間の残りを0として扱います。移譲量は実際の移譲イベントごとに集計します。移譲がない場合の平均は欠損値、観測が1つの場合の標本標準偏差も欠損値です。条件別の移譲量平均は有効な試行の平均を等しく重み付けし、有効試行数を `transfer_valid_runs` に保存します。

```bash
jupyter notebook notebooks/analysis_exp_v1.ipynb
```

ノートブックの `experiment_name` を変更すると、別の実験も解析できます。旧スクリプトの履歴出力・専用作図機能は統合対象外です。

## パッケージとテスト

```python
from src import Agent, AgentV2, AgentV3, Field, FieldV2, FieldV3
from src.factory import get_field_class

field_class = get_field_class("FieldV2")
```

旧 `src.agent` / `src.field` などの import は `src.agents` / `src.fields` に変更してください。

```bash
python -m unittest discover -s tests -v
```
