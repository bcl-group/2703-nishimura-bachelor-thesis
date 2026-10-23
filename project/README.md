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

## v6（CPUテンソル）・v7（CUDA）

変更(v6/v7): v6はPyTorchのCPUテンソル版、v7はCUDA C++（`.cu`）版です。
v7はPyTorchをテンソルの保持・乱数生成・ソート・累積和・統計とPython連携に使用し、
近傍探索・行動・移動・餌の競合・移譲集計・死亡・繁殖・餌生成をCUDAカーネルで実行します。
v5以前の利用にはPyTorchは不要です。
新しい版では個体リスト`field.agents`の代わりに、`positions`（N×2）、
`energies`（N、float64）、`ages`（N）、`weights`（N×2×4）のテンソルを保持します。
個体数は`field.population`、遺伝子の集計は`field.genotype_counts()`で取得します。
`step()`の戻り値は死亡年齢のテンソルです。
`AgentV6` / `AgentV7`は個体オブジェクトではなく、行動・突然変異の一括計算を担当します。

利用先の仮想環境にPyTorchを別途導入してください。CPU版は`python -m pip install torch`、
CUDA版は[公式インストール案内](https://pytorch.org/get-started/locally/)でOSとドライバーに
対応するCUDA版を選びます。RTX 3060（12 GB）1台を想定しています。

```bash
python -m scripts.simu_every_para_up_v6 --config configs/exp_v6.json
python -m scripts.simu_every_para_up_v7 --config configs/exp_v7.json
python -m unittest discover -s tests -p 'test_field_v6_v7.py' -v
```

設定の変更点は、版番号・実験名・`simulation.workers: 1`と
`simulation.device`（v6: `cpu`、v7: `cuda:0`）です。
JSONはコメント非対応なので、設定の変更理由はここに記載しています。
`device`と乱数シードは`simulation`で指定し、フィールド引数には指定しません。
条件・試行は逐次実行し、各ステップ内の個体処理を並列化します。
CUDAが利用できない場合は出力ファイルの作成前にエラーになります。

餌の耐久度、同期更新、移譲、死亡・繁殖、CSVの列と記録間隔はv5と共通です。
乱数生成器が異なるため、v5・v6・v7の同じシードの軌跡は一致しません。
runnerは決定的演算を有効にし、同一版・同一ソフトウェア／機器構成での再現性を
検証対象とします。エネルギーと統計は倍精度です。
GPUへの入力・状態はデバイス上に保持しますが、可変個体数の圧縮・繁殖時には
CPU/GPU同期が発生します（v7では出力個体数・死亡数を取得する際）。
初期20個体などの小規模条件では起動・同期の費用が上回るため、高速化は保証しません。
時間表示には初期化・記録・GPU計算完了待ちが含まれます。

GPU未搭載またはPyTorch未導入の環境では、該当テストはスキップされます。
実機での比較時は、各版の設定で条件・ステップ数を揃え、ウォームアップ後に複数回測定してください。
既定の10万ステップ実験は自動実行しません。

### v7のCUDAソースとビルド

変更(v7): 計算本体は`src/cuda/simulation_v7.cu`です。
`src/cuda/bindings_v7.cpp`がPythonへの接続を定義し、
`src/fields/field_v7.py`からカーネルを起動します。
`src/agents/agent_v7.py`の行動・突然変異APIも同じ`.cu`を使用します。
v6はCPUテンソル版のままです。

v7の実行には、CUDA対応PyTorchに加えて以下が必要です。

- NVIDIAドライバーとCUDA対応GPU
- PyTorchのCUDAバージョンと互換性のあるCUDA Toolkit（`nvcc`を含む）
- CUDA Toolkitが対応するC++コンパイラー（LinuxではGCC/G++）
- Ninja（`python -m pip install ninja`）

CUDA対応PyTorchだけのインストールでは`nvcc`は用意されません。
`python -c "import torch; print(torch.version.cuda)"`と`nvcc --version`で確認します。
Toolkitが自動検出されない場合は、そのインストール先を`CUDA_HOME`に設定してください。

従来と同じv7の実行コマンドで、初回に`.cu`と`.cpp`を自動コンパイルします。
生成物は`.cache/cuda_v7/`に保存し、次回以降は再利用します。
必要なら`MAX_JOBS=1`でコンパイル時の並列数を制限できます。
ビルド失敗時にテンソル版へ戻す処理はありません。
高速近似演算は使用せず、CUDAコンパイル時も`--fmad=false`で計算順序を維持します。
この`.cu`はPyTorch拡張としてビルドするソースで、単体の実行ファイルではありません。

```bash
python -m unittest discover -s tests -p 'test_cuda_v7_source.py' -v
python -m unittest discover -s tests -p 'test_field_v6_v7.py' -v
```

前者は`.cu`のカーネル本体を通常C++として実行する計算式のテストで、CUDA不要です。
CUDAコンパイル・カーネル起動・デバイス連携は、後者のGPUテストで検証します。
今回の開発環境にはPyTorch・`nvcc`・利用可能なGPUがないため、
CUDA拡張のビルドとGPU上での動作・性能は未検証です。
