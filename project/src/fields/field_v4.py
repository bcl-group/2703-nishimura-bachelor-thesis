import random
from dataclasses import dataclass

import numpy as np

from src.agents.agent_v4 import AgentV4


@dataclass(frozen=True)
class StepSnapshot:
    """Read-only state and parameters used by every independent step calculation."""

    positions: np.ndarray
    energies: np.ndarray
    weights: np.ndarray
    food: np.ndarray
    field_size: int
    energy_loss: float
    reward_food: float
    death_threshold: float
    starvation_threshold: float
    transfer_amount: float


@dataclass(frozen=True)
class StepInput:
    agent_index: int
    neighbor_index: int | None
    food_position: tuple[int, int] | None
    random_direction: tuple[int, int]


@dataclass(frozen=True)
class StepResult:
    agent_index: int
    position: tuple[int, int]
    recipient_index: int | None
    transfer: float
    food_reward: float


class FieldV4:
    def __init__(
        self,
        field_size=10,
        initial_agents=20,
        mutation_rate=0.05,
        initial_food_prob=0.1,
        step_food_prob=0.003,
        food_lifetime_steps=5,
        energy_loss_per_step=0.1,
        reward_food=1.0,
        clone_threshold=100.0,
        death_threshold=-50.0,
        life_left=5.0,
        transfer_amount=1.0,
    ):

        self.field_size = field_size
        self.mutation_rate = mutation_rate
        self.step_food_prob = step_food_prob
        self.food_lifetime_steps = food_lifetime_steps
        self.energy_loss_per_step = energy_loss_per_step
        self.reward_food = reward_food
        self.clone_threshold = clone_threshold
        self.death_threshold = death_threshold
        self.starvation_threshold = death_threshold + life_left
        self.transfer_amount = transfer_amount

        # Positive values are remaining lifetimes, not consumable food quantities.
        self.grid_food = np.zeros((field_size, field_size), dtype=int)
        self.agents = [
            AgentV4(
                random.randint(0, field_size - 1),
                random.randint(0, field_size - 1),
                mutation_rate=mutation_rate,
            )
            for _ in range(initial_agents)
        ]
        self.spawn_objects(prob=initial_food_prob)
        self.altruism_count = 0
        self.acting_agent_count = 0
        self.positive_altruism_count = 0

    def spawn_objects(self, prob=None):
        """Spawn food or refresh existing food to its full lifetime."""
        if prob is None:
            prob = self.step_food_prob
        spawn_mask = np.random.rand(self.field_size, self.field_size) < prob
        self.grid_food[spawn_mask] = self.food_lifetime_steps

    @staticmethod
    def _neighbor_cells(position, field_size):
        x, y = map(int, position)
        # Preserve traversal order while avoiding repeated cells on size 1/2 tori.
        xs = dict.fromkeys((x + dx) % field_size for dx in (-1, 0, 1))
        ys = dict.fromkeys((y + dy) % field_size for dy in (-1, 0, 1))
        return [(nx, ny) for nx in xs for ny in ys]

    @staticmethod
    def _neighbor_indices(grid, cells, exclude_index=None):
        # Read only the local cells and exclude self without list.remove().
        return [
            index for x, y in cells for index in grid[x][y]
            if index != exclude_index
        ]

    def get_nearest_agent(self, agent):
        # Preserve the existing helper's random selection within the 3x3 neighborhood.
        neighbors = [
            other for other in self.agents
            if other is not agent
            and min(abs(other.x - agent.x), self.field_size - abs(other.x - agent.x)) <= 1
            and min(abs(other.y - agent.y), self.field_size - abs(other.y - agent.y)) <= 1
        ]
        return random.choice(neighbors) if neighbors else None

    def get_nearest_food(self, agent):
        # Food is already spatially indexed; no full-field search is necessary.
        neighbors = [
            cell for cell in self._neighbor_cells((agent.x, agent.y), self.field_size)
            if self.grid_food[cell] > 0
        ]
        return np.array(random.choice(neighbors), dtype=int) if neighbors else None

    @staticmethod
    def _torus_direction(src_x, src_y, target_x, target_y, field_size):
        diff_x = (target_x - src_x + field_size // 2) % field_size - field_size // 2
        diff_y = (target_y - src_y + field_size // 2) % field_size - field_size // 2
        return int(np.sign(diff_x)), int(np.sign(diff_y))

    def get_torus_direction(self, src_x, src_y, target_x, target_y):
        return self._torus_direction(src_x, src_y, target_x, target_y, self.field_size)

    def snapshot(self):
        """Copy the beginning-of-step environment; no live agent references escape."""
        positions = np.array([(a.x, a.y) for a in self.agents], dtype=int).reshape(-1, 2)
        energies = np.array([a.energy for a in self.agents], dtype=float)
        weights = np.array([a.weights for a in self.agents], dtype=int).reshape(-1, 2, 4)
        food = self.grid_food.copy()
        for array in (positions, energies, weights, food):
            array.setflags(write=False)
        return StepSnapshot(
            positions, energies, weights, food, self.field_size,
            self.energy_loss_per_step, self.reward_food, self.death_threshold,
            self.starvation_threshold, self.transfer_amount,
        )

    @staticmethod
    def prepare_step_inputs(state):
        """Resolve all random choices before evaluating any individual agent."""
        # Build one field_size x field_size map from the immutable step snapshot.
        grid = [[[] for _ in range(state.field_size)] for _ in range(state.field_size)]
        for index, (x, y) in enumerate(state.positions):
            grid[x][y].append(index)
        inputs = []
        for index, position in enumerate(state.positions):
            cells = FieldV4._neighbor_cells(position, state.field_size)
            neighbors = FieldV4._neighbor_indices(grid, cells, exclude_index=index)
            foods = [cell for cell in cells if state.food[cell] > 0]
            neighbor = random.choice(neighbors) if neighbors else None
            food = random.choice(foods) if foods else None
            direction = (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))
            inputs.append(StepInput(index, neighbor, food, direction))
        return inputs

    @staticmethod
    def compute_agent_step(state, step_input):
        """Pure calculation: reads a snapshot and returns one agent's effects."""
        index = step_input.agent_index
        neighbor = step_input.neighbor_index
        food = step_input.food_position
        action = tuple(AgentV4.action_from_weights(
            state.weights[index], int(food is not None), int(neighbor is not None),
            int(state.energies[index] <= state.starvation_threshold),
            int(neighbor is not None and state.energies[neighbor] <= state.starvation_threshold),
        ))
        transfer = 0.0
        recipient = None
        if action == (0, 1) and neighbor is not None:
            available = max(0.0, state.energies[index] - state.energy_loss - state.death_threshold)
            transfer = max(0.0, min(state.transfer_amount, available))
            recipient = neighbor

        target = None
        if action in ((0, 0), (1, 0)) and food is not None:
            target = food
        elif action in ((0, 1), (1, 1)) and neighbor is not None:
            target = state.positions[neighbor]
        x, y = map(int, state.positions[index])
        if target is None:
            dx, dy = step_input.random_direction
        else:
            dx, dy = FieldV4._torus_direction(x, y, *target, state.field_size)
            if action in ((1, 0), (1, 1)):
                dx, dy = -dx, -dy
        position = ((x + dx) % state.field_size, (y + dy) % state.field_size)
        food_reward = state.reward_food if state.food[position] > 0 else 0.0
        return StepResult(index, position, recipient, transfer, food_reward)

    def step(self) -> list[int]:
        """Advance the field and return the ages of agents that died this step."""
        state = self.snapshot()
        inputs = self.prepare_step_inputs(state)
        results = [self.compute_agent_step(state, item) for item in inputs]

        # Accumulate all transfers before committing any live agent state.
        incoming = np.zeros(len(self.agents), dtype=float)
        for result in results:
            if result.recipient_index is not None:
                incoming[result.recipient_index] += result.transfer
        self.acting_agent_count = len(results)
        self.altruism_count = sum(result.transfer > 0 for result in results)
        self.positive_altruism_count = self.altruism_count
        for result in results:
            index = result.agent_index
            agent = self.agents[index]
            agent.x, agent.y = result.position
            agent.age += 1
            agent.energy = (
                state.energies[index] - state.energy_loss - result.transfer
                + incoming[index] + result.food_reward
            )

        next_agents = []
        death_ages = []
        for agent in self.agents:
            if agent.energy <= self.death_threshold:
                # Record the incremented age only after all energy effects settle.
                death_ages.append(int(agent.age))
                continue
            if agent.energy >= self.clone_threshold:
                agent.energy = 0
                child = AgentV4(agent.x, agent.y, mutation_rate=self.mutation_rate, weights=agent.weights)
                next_agents.extend([agent, child])
            else:
                next_agents.append(agent)
        self.agents = next_agents

        np.maximum(self.grid_food - 1, 0, out=self.grid_food)
        self.spawn_objects()
        return death_ages
