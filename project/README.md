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

## v6：条件×試行のGPUバッチ並列化

v6はPyTorchを使用し、全パラメータ条件×全試行をバッチにまとめて実行します。
`simulation.device`で`cuda:0`などのGPU、または`cpu`を指定できます。
設定例`configs/exp_v6.json`はCUDAを使用します。v5以前の実行にPyTorchは不要です。

```bash
python -m pip install torch
python -m scripts.simu_every_para_up_v6 --config configs/exp_v6.json
```

GPUではドライバーに対応するCUDA版PyTorchを
[公式インストール案内](https://pytorch.org/get-started/locally/)に従って導入してください。
CUDA C++拡張を使用しないため、nvccやNinjaは不要です。
CUDAが利用できなければ出力ディレクトリの作成前にエラーになります。
CPUで実行する場合は設定の`simulation.device`を`cpu`に変更してください。

### 3段階の更新

`src/fields/batch_v6.py`が、次の3段階をバッチ全体に適用します。
これは論理的な段階分けであり、実際のCUDAカーネル起動数は3回に限定しません。
近傍索引・ソート・集計・個体配列の整理には補助GPU演算を使います。

1. **行動決定 `[Params, Agents]`**：更新前の状態から観測・行動を決定し、移動先・利他ターゲット・移譲量・要求エサマスを出力します。要求先は移動後のマスです。
2. **空間マス更新 `[Params, Field_X, Field_Y]`**：要求をマス単位で集約し、耐久度を上限に公平な非復元抽選で勝者を選びます。マス別の消費量と個体別の当選フラグを出力します。
3. **個体更新 `[Params, Agents]`**：利他受給を集計し、代謝・移譲・採食報酬を反映して移動・加齢・死亡・分裂を処理します。生存親の直後に子を置き、子だけに突然変異を適用します。

餌は消費後に設定確率で耐久度を所定値まで補充します。
観測・移譲可能量はステップ開始時の状態に基づき、受領分の同一ステップ内の再移譲はありません。
死亡判定は利他受給・採食後です。これらは従来v6の規則を維持します。

### バッチ・メモリ・再現性

Paramsはパラメータ組合せとrun_idの組です。異なるフィールドサイズは別バッチにします。
内部状態は`[Params, Capacity, ...]`で保持し、有効個体マスクで空き領域を除外します。
個体数は条件ごとに増減し、出生時に容量を自動拡張します。上限で個体を切り捨てる処理はありません。

`simulation.batch_size`には同時に実行する試行数の正整数、または`null`を指定します。
`null`または省略時は同じフィールドサイズの全試行をまとめます。
CUDAメモリ不足時はバッチを半分に分け、初期seedから再実行します。
失敗した計算の記録は出力せず、1試行でも収まらない場合はエラーにします。
`workers`は`1`とし、1プロセスからGPUを使います。

試行ごとに独立した乱数生成器を持ち、seedは`base_seed + run_id`です。
実個体数・出生数に応じて乱数を生成するため、パディング容量・バッチ分割・条件の並べ替え・
他試行の絶滅によって乱数系列は変化しません。
乱数生成と初期化・記録には条件別のPythonループがありますが、状態更新はバッチ演算です。
利他受給は送信者順を固定して区間ごとに加算し、浮動小数点atomic加算を避けます。
同一デバイス・ソフトウェア構成での再現性を検証対象とし、旧v6やCPU/GPU間の乱数軌跡の一致は求めません。
エネルギー・統計はfloat64です。

可変個体数と乱数形状の取得には毎ステップCPU/GPU同期が発生します。
小規模な条件では起動・同期・ソートの費用が大きく、高速化は条件数や個体数に依存します。
時間表示には初期化・記録・GPU計算完了待ちを含むバッチ時間と実験全体の時間を表示します。

### API・出力・検証

`FieldV6`の単一条件APIもバッチ計算の1条件版を使用します。
`positions`（N×2）、`energies`（N）、`ages`（N）、`weights`（N×2×4）、
`grid_food`、`population`、`genotype_counts()`を維持し、`step()`は死亡年齢テンソルを返します。
`AgentV6`は行動・突然変異の一括計算APIです。

CSVの列・集計方法・記録間隔は従来v6と共通です。時系列・遺伝子頻度は各条件のrun_id=0を
ステップ1と1000ステップごとに記録します。絶滅後も評価期間の残りをゼロとして集計します。
異なる条件の時系列行はバッチ処理順に出力されるため、解析時は条件とstepで並べてください。

```bash
python -m unittest discover -s tests -p 'test_batch_v6.py' -v
python -m unittest discover -s tests -p 'test_field_v6_v7.py' -v
python -m unittest discover -s tests -v
```

PyTorch未導入時はテンソル計算テストを、GPUがない環境ではGPU比較をスキップします。
CPUでは既存ルール、バッチ分割・並べ替えの一致、容量拡張、絶滅、CSV、メモリ不足時の再試行を検証します。
GPUテストは固定入力でCPUと比較し、バッチ構成によらない再現性も確認します。
今回の環境ではCPU版PyTorchで検証し、GPUドライバーを利用できないためGPU動作・性能は未検証です。
実機性能の比較では、条件・ステップ数を揃え、ウォームアップ後に複数回測定してください。
既定の10万ステップ実験は自動実行しません。

## v7：時系列を保存せず、パラメータによる推移を比較する

v7はv6と同じ3段階のバッチ計算・行動規則を使い、CPUとCUDAの両方に対応します。
`FieldV7`と`AgentV7`はv6のクラスを継承し、実行・集計処理も共有しています。
旧CUDA C++版v7の復元ではなく、現在のv6を基にした版です。

v7は個体数・利他行動・遺伝子型の時系列を記録しません。記録用リスト、区間集計、
定期サンプリング、遺伝子頻度集計、記録用のCPU転送を省きます。
評価期間の合計・二乗和・死亡数から統計を計算するため、記録用メモリがステップ数に比例して増えません。
個体数の増加に伴う状態テンソルのメモリは引き続き必要です。
同じ条件・seed・デバイス構成では、v6と同じ試行別・条件別の集計値になります。

```bash
python -m scripts.simu_every_para_up_v7 --config configs/exp_v7.json
jupyter notebook notebooks/analysis_exp_v7.ipynb
```

設定例は現在のv6と同じ240条件×10試行・20万ステップで、後半10万ステップを評価します。
`batch_size: 1200`、`device: cuda:0`が既定です。CPUで実行する場合は`device`を`cpu`へ変更してください。
バッチ分割・乱数系列・メモリ不足時の再試行・絶滅後の集計はv6と共通です。
v7には時系列記録を再有効化する設定はありません。v6の出力と解析ノートブックは維持しています。

実験結果は`data/exp_v7_grid_search/`に次の3ファイルを保存します。

- `config_used.json`：実行時の設定
- `raw_data.csv`：各条件・各試行の平均と標本標準偏差
- `summary_data.csv`：条件別に集計した試行平均と試行間標準偏差

`time_series_data.csv`と`genotype_time_series_data.csv`は作成しません。
既存の実験データを残して比較する場合は、`experiment_name`を変更して別ディレクトリに保存してください。

解析ノートブックは、この3ファイルだけを読み込みます。既定の横軸は`death_threshold`、
線の分類は`transfer_amount`で、初期餌確率・餌補充確率・餌耐久度の組合せごとに図を分けます。
各図は平均個体数・利他行動件数・利他行動率・平均死亡年齢の4パネルです。
点付き折れ線に試行間標準偏差のエラーバーを表示します（信頼区間ではありません）。
欠損した平均をゼロに置換せず、標準偏差のみが欠損の場合は平均の点と線を表示します。

`x_parameter`・`series_parameter`で軸と線の分類、`comparison_filters`で条件の絞り込みを変更できます。
全条件のままなら現在の設定では8枚の図になります。他の条件を暗黙に平均する処理はありません。
可変パラメータが1つの場合は単一系列、固定条件だけの場合は点で表示します。
同じ集計形式のv6データも、`experiment_name`を変更すれば読み込めます。

```bash
python -m unittest discover -s tests -p 'test_v7.py' -v
python -m unittest discover -s tests -p 'test*6*.py' -v
```

v7はCPUでv6との状態・統計の一致、時系列記録の停止、バッチ分割、容量拡張、絶滅と再試行を検証しています。
GPUテストはCUDAが利用できる環境で実施します。現在の環境ではGPU実行・性能は未検証です。

## v4の行動と出力

`python -m scripts.simu_every_para_up --config configs/exp_v4.json` で実行します。
入力は順に、餌の有無、他個体の有無、自身の飢餓状態、他個体の飢餓状態です。
2ビット出力は `00`: 餌に接近、`01`: 他個体に接近し利他行動、`10`: 餌から離れる、`11`: 他個体から離れる、に対応します。
対象がない場合はランダムに移動します。

出力は `config_used.json`、`raw_data.csv`、`summary_data.csv` のみです。個体数・利他行動および遺伝子型の時系列は収集しません。既存の時系列CSVは削除・更新しません。

 v4の餌は `food_durability`（正の整数、既定値5）を採用します。時間では減らず、獲得個体1体につき1減ります。移動後に同じセルにいる個体数が残り耐久値を超える場合、重複なしの抽選で獲得個体を選び、各個体に `reward_food` を与えます。再出現時は耐久値を設定値に戻します。旧設定名 `food_lifetime_steps` は使用できません。

## v3・v5・v7の共通CPUモデル

3方式ともv5の同期更新・移動・移譲・採食抽選を使用します。
v3は他者の利他レベルを入力して自身のレベルを更新（3×3）、v5は自身と他者の飢餓状態（2×4）、v7は餌と他者の有無のみ（2×2）を入力します。
移譲は全方式で行動01の場合に実施します。v7は旧GPU版を置き換えています。
実行は `python -m scripts.simu_every_para_up_v3 --config configs/exp_v3.json` とし、v5・v7も同じ形式です。
時系列と遺伝子型は各条件のrun_id=0について1・3000・6000…stepで保存します。
旧v7のGPU実行・時系列非保存の説明はこの仕様で置き換えます。
