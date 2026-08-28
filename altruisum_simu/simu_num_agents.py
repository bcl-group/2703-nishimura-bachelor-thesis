import numpy as np
import random
import matplotlib.pyplot as plt

# --- 基本パラメータ ---
FIELD_SIZE = 10          # 10x10のフィールド
INITIAL_AGENTS = 20      # 初期個体数
MUTATION_RATE = 0.05     # 突然変異率
TOTAL_STEPS = 100000     # 全ステップ数
RECORD_INTERVAL = 1      # 記録・出力を行うインターバル
INITIAL_FOOD_PROB = 0.2  # 初期餌配置の確率
STEP_FOOD_PROB = 0.005   # 毎ステップの餌発生の確率
FOOD_DURABILITY = 5.0      # 餌の耐久値

# 報酬・閾値パラメータ
ENERGY_LOSS_PER_STEP = 0.1 # 1ステップの消費エネルギー
REWARD_FOOD = 1.0          # 餌獲得時の報酬
CLONE_THRESHOLD = 100.0    # 複製閾値
DEATH_THRESHOLD = -50.0    # 死亡閾値
TRANSFER_AMOUNT = 1.0      # 利他行動1回あたりのエネルギー譲渡量
STARVING_THRESHOLD = 0.0  # 他者の「低エネルギー」を判定する閾値

class Agent:
    def __init__(self, x, y, weights=None):
        self.x = x
        self.y = y
        self.energy = 0.0
        self.age = 0
        if weights is None:
            self.weights = np.random.randint(0, 2, (3, 3))
        else:
            self.weights = self.mutate(weights)
            
    def mutate(self, weights):
        new_weights = weights.copy()
        for i in range(3):
            for j in range(3):
                if random.random() < MUTATION_RATE:
                    new_weights[i, j] = 1 - new_weights[i, j]
        return new_weights

    def decide_action(self, on_food, see_other, other_is_starving):
        input_vec = np.array([on_food, see_other, other_is_starving])
        output_vec = np.dot(self.weights, input_vec) % 2
        
        move_action = output_vec[0:2]
        give_energy = output_vec[2]
        return move_action, give_energy

class Field:
    def __init__(self):
        self.grid_food = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=int)
        self.agents = [Agent(random.randint(0, 9), random.randint(0, 9)) for _ in range(INITIAL_AGENTS)]
        self.spawn_objects(is_initial=True)
        self.altruism_count = 0  # 区間内の利他行動回数をカウントする変数

    def spawn_objects(self, is_initial=False):
        prob = INITIAL_FOOD_PROB if is_initial else STEP_FOOD_PROB
        for i in range(FIELD_SIZE):
            for j in range(FIELD_SIZE):
                if random.random() < prob: # 餌の発生確率
                    self.grid_food[i, j] = FOOD_DURABILITY # 餌の耐久値

    def get_nearest_agent(self, agent):
        # 周囲1マス以内にいる他個体をすべて抽出
        neighbors = [
            other for other in self.agents 
            if other is not agent and abs(other.x - agent.x) <= 1 and abs(other.y - agent.y) <= 1
        ]
        return random.choice(neighbors) if neighbors else None

    def step(self):
        actions = []
        nearest_agents = []
        
        for agent in self.agents:
            agent.age += 1
            agent.energy -= ENERGY_LOSS_PER_STEP
            
            on_food = 1 if self.grid_food[agent.x, agent.y] > 0 else 0
            nearest = self.get_nearest_agent(agent)
            nearest_agents.append(nearest)
            
            see_other = 1 if nearest else 0
            other_is_starving = 1 if nearest and nearest.energy < STARVING_THRESHOLD else 0
            
            move_action, give_energy = agent.decide_action(on_food, see_other, other_is_starving)
            actions.append((move_action, give_energy))

        for i, agent in enumerate(self.agents):
            move_action, give_energy = actions[i]
            nearest = nearest_agents[i]
            
            # 利他行動の実行とカウント
            if give_energy == 1 and nearest and agent.energy > STARVING_THRESHOLD:
                agent.energy -= TRANSFER_AMOUNT
                nearest.energy += TRANSFER_AMOUNT
                self.altruism_count += 1  # 利他行動が行われたらカウントアップ
                
            dx, dy = 0, 0

            def get_torus_direction(src_x, src_y, target_x, target_y, size):
                diff_x = (target_x - src_x + size // 2) % size - size // 2
                diff_y = (target_y - src_y + size // 2) % size - size // 2

                dir_x = 1 if diff_x > 0 else (-1 if diff_x < 0 else 0)
                dir_y = 1 if diff_y > 0 else (-1 if diff_y < 0 else 0)
                return dir_x, dir_y

            if move_action[0] == 0 and move_action[1] == 1:
                if nearest:
                    dx, dy = get_torus_direction(agent.x, agent.y, nearest.x, nearest.y, FIELD_SIZE)
                else:
                    dx, dy = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])
            elif move_action[0] == 1 and move_action[1] == 0:
                if nearest:
                    dir_x, dir_y = get_torus_direction(agent.x, agent.y, nearest.x, nearest.y, FIELD_SIZE)
                    dx, dy = -dir_x, -dir_y
                else:
                    dx, dy = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])
            elif move_action[0] == 0 and move_action[1] == 0:
                dx = random.choice([-1, 0, 1])
                dy = random.choice([-1, 0, 1])
            elif move_action[0] == 1 and move_action[1] == 1:
                dx, dy = 0, 0

            agent.x = (agent.x + dx) % FIELD_SIZE
            agent.y = (agent.y + dy) % FIELD_SIZE
            
            if self.grid_food[agent.x, agent.y] > 0:
                agent.energy += REWARD_FOOD
                self.grid_food[agent.x, agent.y] -= 1

        self.spawn_objects()

        next_agents = []
        for agent in self.agents:
            if agent.energy <= DEATH_THRESHOLD:
                continue
            if agent.energy >= CLONE_THRESHOLD:
                agent.energy = 0
                child = Agent(agent.x, agent.y, agent.weights)
                next_agents.extend([agent, child])
            else:
                next_agents.append(agent)
        self.agents = next_agents

# --- 実行とグラフ化 ---
simulation = Field()
history_steps = []
history_altruism = []
history_population = []  # 個体数を記録するリストを追加

print("シミュレーションを開始します...")

for step_num in range(1, TOTAL_STEPS + 1):
    simulation.step()
    
    # 指定インターバルごとの記録と出力
    if step_num % RECORD_INTERVAL == 0:
        history_steps.append(step_num)
        history_altruism.append(simulation.altruism_count)
        history_population.append(len(simulation.agents))  # 現在の個体数を記録
        
        # 次の区間のためにカウントをリセット
        simulation.altruism_count = 0

print("シミュレーション完了。グラフを保存します。")

# 1. 利他行動のグラフ描画と保存
plt.figure(figsize=(10, 6))
plt.plot(history_steps, history_altruism, marker='', linestyle='-', color='b')
plt.title('Number of Altruistic Actions per Step', fontsize=18)
plt.xlabel('Step', fontsize=18)
plt.ylabel('Altruistic Actions Count', fontsize=18)
plt.tick_params(labelsize=18)
plt.grid(True)
plt.savefig('altruism_history.png')  
plt.close()

# 2. 個体数推移のグラフ描画と保存
plt.figure(figsize=(10, 6))
plt.plot(history_steps, history_population, marker='', linestyle='-', color='g')
plt.title('Population Size over Time', fontsize=18)
plt.xlabel('Step', fontsize=18)
plt.ylabel('Number of Agents', fontsize=18)
plt.tick_params(labelsize=18)
plt.grid(True)
plt.savefig('population_history.png')  
plt.close()

print("保存が完了しました。")