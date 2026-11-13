import random
from dataclasses import dataclass

import numpy as np

from src.agents.agent_v2_1 import AgentV2_1


@dataclass(frozen=True)
class StepSnapshot:
    """ステップ計算全体で使用される不変の状態およびパラメータ設定。"""

    positions: np.ndarray
    energies: np.ndarray
    weights: np.ndarray
    food: np.ndarray
    field_size: int
    energy_loss: float
    reward_food: float
    death_threshold: float
    levels: np.ndarray
    starvation_threshold: float
    transfer_amount: float


@dataclass(frozen=True)
class StepInput:
    agent_index: int
    neighbor_index: int | None
    food_position: tuple[int, int] | None
    random_direction: tuple[int, int]
    safe_cells_for_agent: list[tuple[int, int]]


@dataclass(frozen=True)
class StepResult:
    agent_index: int
    position: tuple[int, int]
    recipient_index: int | None
    transfer: float
    level: int = 0


class FieldV2_1:
    agent_class = AgentV2_1
    def __init__(
        self,
        field_size=10,
        initial_agents=20,
        mutation_rate=0.05,
        initial_food_prob=0.1,
        step_food_prob=0.003,
        food_durability=5,
        energy_loss_per_step=0.1,
        reward_food=1.0,
        clone_threshold=100.0,
        death_threshold=-50.0,
        life_left=5.0,
        transfer_amount=1.0,
    ):
        if type(food_durability) is not int or food_durability <= 0:
            raise ValueError("food_durability must be a positive integer.")
        self.field_size = field_size
        self.mutation_rate = mutation_rate
        self.step_food_prob = step_food_prob
        self.food_durability = food_durability
        self.energy_loss_per_step = energy_loss_per_step
        self.reward_food = reward_food
        self.clone_threshold = clone_threshold
        self.death_threshold = death_threshold
        self.starvation_threshold = death_threshold + life_left
        self.transfer_amount = transfer_amount

        self.grid_food = np.zeros((field_size, field_size), dtype=int)
        self.agents = [
            self.agent_class(
                random.randint(0, field_size - 1),
                random.randint(0, field_size - 1),
                mutation_rate=mutation_rate,
            )
            for _ in range(initial_agents)
        ]
        self.spawn_objects(prob=initial_food_prob)
        self.altruism_count = 0
        self.acting_agent_count = 0

    def spawn_objects(self, prob=None):
        """食料を生成または既存食料の耐久値をリセット。"""
        if prob is None:
            prob = self.step_food_prob
        spawn_mask = np.random.rand(self.field_size, self.field_size) < prob
        self.grid_food[spawn_mask] = self.food_durability

    @staticmethod
    def _neighbor_cells(position, field_size):
        x, y = map(int, position)
        xs = dict.fromkeys((x + dx) % field_size for dx in (-1, 0, 1))
        ys = dict.fromkeys((y + dy) % field_size for dy in (-1, 0, 1))
        return [(nx, ny) for nx in xs for ny in ys]

    @staticmethod
    def _neighbor_indices(grid, cells, exclude_index=None):
        return [
            index for x, y in cells for index in grid[x][y]
            if index != exclude_index
        ]

    @staticmethod
    def _torus_direction(src_x, src_y, target_x, target_y, field_size):
        diff_x = (target_x - src_x + field_size // 2) % field_size - field_size // 2
        diff_y = (target_y - src_y + field_size // 2) % field_size - field_size // 2
        return int(np.sign(diff_x)), int(np.sign(diff_y))

    def snapshot(self):
        """ステップ開始時の環境状態のスナップショットを作成（不変オブジェクト）。"""
        positions = np.array([(a.x, a.y) for a in self.agents], dtype=int).reshape(-1, 2)
        energies = np.array([a.energy for a in self.agents], dtype=float)
        weights = np.array([a.weights for a in self.agents], dtype=int).reshape(-1, *self.agent_class.weight_shape)
        levels = np.array([getattr(a, "altruism_level", 0) for a in self.agents], dtype=int)
        levels.setflags(write=False)
        food = self.grid_food.copy()
        for array in (positions, energies, weights, food):
            array.setflags(write=False)
        return StepSnapshot(
            positions, energies, weights, food, self.field_size,
            self.energy_loss_per_step, self.reward_food, self.death_threshold,
            levels, self.starvation_threshold, self.transfer_amount,
        )

    @staticmethod
    def prepare_step_inputs(state):
        """ステップ計算前の観測・乱数決定を一括処理。"""
        grid = [[[] for _ in range(state.field_size)] for _ in range(state.field_size)]
        for index, (x, y) in enumerate(state.positions):
            grid[x][y].append(index)
        inputs = []
        for index, position in enumerate(state.positions):
            cells = FieldV2_1._neighbor_cells(position, state.field_size)
            neighbors = FieldV2_1._neighbor_indices(grid, cells, exclude_index=index)
            foods = [cell for cell in cells if state.food[cell] > 0]
            neighbor = random.choice(neighbors) if neighbors else None
            food = random.choice(foods) if foods else None
            direction = (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))
            x, y = map(int, position)
            safe_cells = [
                (cx, cy) for cx, cy in cells
                if (len(grid[cx][cy]) == 0) or (grid[cx][cy] == [index])
            ]
            
            inputs.append(StepInput(index, neighbor, food, direction, safe_cells))
        return inputs

    @classmethod
    def compute_agent_step(cls, state, step_input):
        """Pure calculation: reads a snapshot and returns one agent's effects."""
        index = step_input.agent_index
        neighbor = step_input.neighbor_index
        food = step_input.food_position
        output = tuple(cls.agent_class.action_from_weights(
            state.weights[index], int(food is not None), int(neighbor is not None),
            int(state.energies[index] <= state.starvation_threshold),
            int(neighbor is not None and state.energies[neighbor] <= state.starvation_threshold),
        ))
        action = tuple(output[:2])
        level = state.levels[index]

        transfer = 0.0
        recipient = None
        if level == 1 and neighbor is not None and state.energies[index] > state.starvation_threshold and state.energies[neighbor] <= state.starvation_threshold:
            transfer = max(0.0, state.transfer_amount)
            recipient = neighbor

        x, y = map(int, state.positions[index])
        
        if action in ((0, 0), (0, 1)):
            target = food if action == (0, 0) else (state.positions[neighbor] if neighbor is not None else None)
            if target is None:
                dx, dy = step_input.random_direction
            else:
                dx, dy = cls._torus_direction(x, y, *target, state.field_size)

        elif action in ((1, 0), (1, 1)):
            neighbor_cells = cls._neighbor_cells((x, y), state.field_size)
            if action == (1, 0):
                safe_cells = [cell for cell in neighbor_cells if state.food[cell] <= 0]
            else:
                safe_cells = step_input.safe_cells_for_agent

            if safe_cells:
                target_cell = random.choice(safe_cells)
                dx, dy = cls._torus_direction(x, y, *target_cell, state.field_size)
            else:
                dx, dy = step_input.random_direction
        else:
            dx, dy = step_input.random_direction

        position = ((x + dx) % state.field_size, (y + dy) % state.field_size)
        return StepResult(index, position, recipient, transfer, level)

    @staticmethod
    def resolve_food(state, results):
        """同一セルでの食料獲得競合を解決。"""
        candidates = {}
        for result in results:
            if state.food[result.position] > 0:
                candidates.setdefault(result.position, []).append(result.agent_index)
        rewards = np.zeros(len(state.positions), dtype=float)
        consumed = np.zeros_like(state.food)
        for position in sorted(candidates):
            indices = sorted(candidates[position])
            count = min(len(indices), int(state.food[position]))
            winners = random.sample(indices, count) if count < len(indices) else indices
            rewards[winners] = state.reward_food
            consumed[position] = count
        return rewards, consumed

    def step(self) -> list[int]:
        """1ステップ環境を進め、死亡したエージェントの年齢リストを返却。"""
        state = self.snapshot()
        inputs = self.prepare_step_inputs(state)
        results = sorted(
            (self.compute_agent_step(state, item) for item in inputs),
            key=lambda result: result.agent_index,
        )
        food_rewards, consumed = self.resolve_food(state, results)

        incoming = np.zeros(len(self.agents), dtype=float)
        for result in results:
            if result.recipient_index is not None:
                incoming[result.recipient_index] += result.transfer

        self.acting_agent_count = sum(
            1 for item in inputs
            if item.neighbor_index is not None
            and state.energies[item.agent_index] > state.starvation_threshold
            and state.energies[item.neighbor_index] <= state.starvation_threshold
        )
        self.altruism_count = sum(result.transfer > 0 for result in results)

        for result in results:
            index = result.agent_index
            agent = self.agents[index]
            agent.x, agent.y = result.position
            agent.age += 1
            agent.energy = (
                state.energies[index] - state.energy_loss - result.transfer
                + incoming[index] + food_rewards[index]
            )

        next_agents = []
        death_ages = []
        for agent in self.agents:
            if agent.energy <= self.death_threshold:
                death_ages.append(int(agent.age))
                continue
            if agent.energy >= self.clone_threshold:
                agent.energy = 0
                child = self.agent_class(agent.x, agent.y, mutation_rate=self.mutation_rate, weights=agent.weights)
                next_agents.extend([agent, child])
            else:
                next_agents.append(agent)
        self.agents = next_agents

        self.grid_food[:] = state.food - consumed
        self.spawn_objects()
        return death_ages