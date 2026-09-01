# config.py

# --- 基本パラメータ ---
FIELD_SIZE = 10          # 10x10のフィールド
INITIAL_AGENTS = 20      # 初期個体数
MUTATION_RATE = 0.05     # 突然変異率
TOTAL_STEPS = 100000     # 全ステップ数
EVAL_STEPS = 10000       # 評価（記録）対象とする最後のステップ数

INITIAL_FOOD_PROB = 0.2  # 初期餌配置の確率
STEP_FOOD_PROB = 0.005   # 毎ステップの餌発生の確率
FOOD_DURABILITY = 5.0    # 餌の耐久値

# 報酬・閾値パラメータ
ENERGY_LOSS_PER_STEP = 0.1 # 1ステップの消費エネルギー
REWARD_FOOD = 1.0          # 餌獲得時の報酬
CLONE_THRESHOLD = 100.0    # 複製閾値
TRANSFER_AMOUNT = 1.0      # 利他行動1回あたりのエネルギー譲渡量
STARVING_THRESHOLD = 0.0   # 他者の飢餓状態を判定する閾値