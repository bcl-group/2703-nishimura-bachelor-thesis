"""Synchronous CPU field with predators, eight movement actions and altruism."""
from dataclasses import dataclass
import random

import numpy as np

from src.agents.agent_v4_2 import AgentV4_2


@dataclass(frozen=True)
class StepSnapshot:
    positions: np.ndarray
    energies: np.ndarray
    weights: np.ndarray
    food: np.ndarray
    predators: np.ndarray
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
    predator_position: tuple[int, int] | None
    action: int
    altruism: int
    fallback_direction: tuple[int, int]
    move_target: tuple[int, int] | None


class FieldV4_2:
    agent_class = AgentV4_2

    def __init__(self, field_size=10, initial_agents=20, mutation_rate=0.05,
                 initial_food_prob=0.1, step_food_prob=0.003, food_durability=5,
                 energy_loss_per_step=0.1, reward_food=1.0, clone_threshold=100.0,
                 death_threshold=-50.0, life_left=5.0, transfer_amount=1.0,
                 initial_predator_prob=0.005, step_predator_prob=0.01,
                 predator_attack_cost=3.0, predator_empty_steps=3):
        if type(field_size) is not int or field_size < 2:
            raise ValueError("field_size must be an integer >= 2")
        if type(initial_agents) is not int or initial_agents < 0:
            raise ValueError("initial_agents must be a nonnegative integer")
        if type(food_durability) is not int or food_durability <= 0:
            raise ValueError("food_durability must be a positive integer")
        for name, probability in (("initial_food_prob", initial_food_prob),
                                  ("step_food_prob", step_food_prob),
                                  ("initial_predator_prob", initial_predator_prob),
                                  ("step_predator_prob", step_predator_prob),
                                  ("mutation_rate", mutation_rate)):
            if not 0 <= probability <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if predator_empty_steps != 3:
            if type(predator_empty_steps) is not int or predator_empty_steps < 1:
                raise ValueError("predator_empty_steps must be a positive integer")
        if min(energy_loss_per_step, reward_food, transfer_amount, predator_attack_cost) < 0:
            raise ValueError("energy amounts must be nonnegative")

        self.field_size = field_size
        self.mutation_rate = mutation_rate
        self.step_food_prob = step_food_prob
        self.food_durability = food_durability
        self.energy_loss_per_step = float(energy_loss_per_step)
        self.reward_food = float(reward_food)
        self.clone_threshold = float(clone_threshold)
        self.death_threshold = float(death_threshold)
        self.starvation_threshold = self.death_threshold + float(life_left)
        self.transfer_amount = float(transfer_amount)
        self.initial_predator_prob = float(initial_predator_prob)
        self.step_predator_prob = float(step_predator_prob)
        self.predator_attack_cost = float(predator_attack_cost)
        self.predator_empty_steps = int(predator_empty_steps)

        self.grid_food = np.zeros((field_size, field_size), dtype=np.int32)
        self.predator_grid = np.zeros((field_size, field_size), dtype=bool)
        self.predator_empty_count = np.zeros((field_size, field_size), dtype=np.int8)
        self.agents = [
            self.agent_class(random.randrange(field_size), random.randrange(field_size), mutation_rate)
            for _ in range(initial_agents)
        ]
        self.altruism_count = 0
        self.acting_agent_count = 0
        self.predator_attack_count = 0
        self.spawn_food(initial_food_prob)
        self.spawn_predators(initial_predator_prob)

    @staticmethod
    def _neighbor_cells(position, field_size):
        x, y = map(int, position)
        xs = dict.fromkeys((x + dx) % field_size for dx in (-1, 0, 1))
        ys = dict.fromkeys((y + dy) % field_size for dy in (-1, 0, 1))
        return [(nx, ny) for nx in xs for ny in ys]

    @staticmethod
    def _torus_direction(src, target, size):
        dx = (target[0] - src[0] + size // 2) % size - size // 2
        dy = (target[1] - src[1] + size // 2) % size - size // 2
        return int(np.sign(dx)), int(np.sign(dy))

    def spawn_food(self, probability=None):
        probability = self.step_food_prob if probability is None else probability
        mask = np.random.random(self.grid_food.shape) < probability
        self.grid_food[mask] = self.food_durability

    def spawn_predators(self, probability=None):
        """Independent Bernoulli trial per cell; never stack predators."""
        probability = self.step_predator_prob if probability is None else probability
        trials = np.random.random(self.predator_grid.shape) < probability
        new_cells = trials & ~self.predator_grid
        self.predator_grid[new_cells] = True
        self.predator_empty_count[new_cells] = 0

    def snapshot(self):
        positions = np.asarray([(a.x, a.y) for a in self.agents], dtype=int).reshape(-1, 2)
        energies = np.asarray([a.energy for a in self.agents], dtype=float)
        weights = np.asarray([a.weights for a in self.agents], dtype=int).reshape(-1, 4, 5)
        food = self.grid_food.copy()
        predators = self.predator_grid.copy()
        for array in (positions, energies, weights, food, predators):
            array.setflags(write=False)
        return StepSnapshot(positions, energies, weights, food, predators,
                            self.field_size, self.energy_loss_per_step, self.reward_food,
                            self.death_threshold, self.starvation_threshold, self.transfer_amount)

    @classmethod
    def prepare_step_inputs(cls, state):
        size = state.field_size
        grid = [[[] for _ in range(size)] for _ in range(size)]
        for i, (x, y) in enumerate(state.positions):
            grid[x][y].append(i)
        prepared = []
        for i, (x, y) in enumerate(state.positions):
            cells = cls._neighbor_cells((x, y), size)
            other_indices = [j for cx, cy in cells for j in grid[cx][cy] if j != i]
            foods = [cell for cell in cells if state.food[cell] > 0]
            predator_cells = [cell for cell in cells if state.predators[cell]]
            other = random.choice(other_indices) if other_indices else None
            food = random.choice(foods) if foods else None
            predator = random.choice(predator_cells) if predator_cells else None
            observations = (int(bool(foods)), int(bool(other_indices)),
                            int(state.energies[i] <= state.starvation_threshold),
                            int(other is not None and state.energies[other] <= state.starvation_threshold),
                            int(bool(predator_cells)))
            output = cls.agent_class.action_from_weights(state.weights[i], *observations)
            action = int(output[0]) * 4 + int(output[1]) * 2 + int(output[2])
            altruism = int(output[3])
            # Select an actual target object for the three approach actions.
            if action == 0:
                target = food
            elif action == 1:
                target = tuple(map(int, state.positions[other])) if other is not None else None
            elif action == 2:
                target = predator
            else:
                target = None
            prepared.append(StepInput(i, other, food, predator, action, altruism,
                                      (random.choice((-1, 0, 1)), random.choice((-1, 0, 1))), target))
        return prepared

    @classmethod
    def _choose_destination(cls, state, item, occupied_by_agents):
        x, y = map(int, state.positions[item.agent_index])
        current = (x, y)
        cells = cls._neighbor_cells(current, state.field_size)
        if item.action in (0, 1, 2):
            if item.move_target is None:
                dx, dy = item.fallback_direction
                return ((x + dx) % state.field_size, (y + dy) % state.field_size)
            dx, dy = cls._torus_direction(current, item.move_target, state.field_size)
            return ((x + dx) % state.field_size, (y + dy) % state.field_size)

        candidates = []
        for cell in cells:
            food_here = state.food[cell] > 0
            other_here = any(j != item.agent_index for j in occupied_by_agents.get(cell, ()))
            predator_here = bool(state.predators[cell])
            has_object = food_here or other_here or predator_here
            valid = {
                3: not food_here,       # 011: no food
                4: not other_here,      # 100: no other agent
                5: not predator_here,   # 101: no predator
                6: not has_object,      # 110: no object of any type
                7: has_object,          # 111: at least one object
            }[item.action]
            if valid:
                candidates.append(cell)
        if candidates:
            return random.choice(candidates)
        dx, dy = item.fallback_direction
        return ((x + dx) % state.field_size, (y + dy) % state.field_size)

    def _attack_predators(self):
        """Each predator independently chooses at most one nearby agent; duplicates allowed."""
        positions_by_cell = {}
        for index, agent in enumerate(self.agents):
            positions_by_cell.setdefault((agent.x, agent.y), []).append(index)
        attacks = []
        for px, py in np.argwhere(self.predator_grid):
            nearby = [i for cell in self._neighbor_cells((int(px), int(py)), self.field_size)
                      for i in positions_by_cell.get(cell, ())]
            if nearby:
                attacks.append(random.choice(nearby))
        for index in attacks:
            self.agents[index].energy -= self.predator_attack_cost
        self.predator_attack_count = len(attacks)

    def _update_predators(self):
        for px, py in np.argwhere(self.predator_grid):
            px, py = int(px), int(py)
            nearby = any(
                min(abs(a.x - px), self.field_size - abs(a.x - px)) <= 1 and
                min(abs(a.y - py), self.field_size - abs(a.y - py)) <= 1
                for a in self.agents
            )
            if nearby:
                self.predator_empty_count[px, py] = 0
            else:
                self.predator_empty_count[px, py] += 1
                if self.predator_empty_count[px, py] >= self.predator_empty_steps:
                    self.predator_grid[px, py] = False
                    self.predator_empty_count[px, py] = 0

    @property
    def predator_count(self):
        return int(self.predator_grid.sum())

    def genotype_counts(self):
        counts = {}
        for agent in self.agents:
            genotype = "".join(str(int(bit)) for bit in agent.weights.flat)
            counts[genotype] = counts.get(genotype, 0) + 1
        return counts

    def step(self):
        """Run one step in the specified order and return ages of agents that died."""
        state = self.snapshot()                         # 1. snapshot environment
        inputs = self.prepare_step_inputs(state)       # 2-3. observe and decide
        occupied = {}
        for i, (x, y) in enumerate(state.positions):
            occupied.setdefault((int(x), int(y)), []).append(i)
        destinations = [self._choose_destination(state, item, occupied) for item in inputs]
        for agent, destination in zip(self.agents, destinations):  # 4. move simultaneously
            agent.x, agent.y = destination

        self._attack_predators()                        # 5. predator attacks after movement

        # 6. food, altruism and energy accounting. Recipients are selected from the
        # beginning-of-step neighborhood, as were all agent observations.
        rewards = np.zeros(len(self.agents), dtype=float)
        food_claims = {}
        for i, agent in enumerate(self.agents):
            if state.food[(agent.x, agent.y)] > 0:
                food_claims.setdefault((agent.x, agent.y), []).append(i)
        consumed = np.zeros_like(self.grid_food)
        for cell, claimants in food_claims.items():
            available = int(state.food[cell])
            winners = claimants if len(claimants) <= available else random.sample(claimants, available)
            rewards[winners] = state.reward_food
            consumed[cell] = len(winners)

        incoming = np.zeros(len(self.agents), dtype=float)
        transfers = 0
        for item in inputs:
            if item.altruism and item.neighbor_index is not None:
                donor = item.agent_index
                recipient = item.neighbor_index
                amount = state.transfer_amount
                self.agents[donor].energy -= amount
                incoming[recipient] += amount
                transfers += 1
        self.altruism_count = transfers
        self.acting_agent_count = len(self.agents)
        for i, agent in enumerate(self.agents):
            agent.age += 1
            agent.energy += -state.energy_loss + rewards[i] + incoming[i]
        self.grid_food[:] = state.food - consumed

        # 7. deaths and cloning are resolved after every energy effect has been applied.
        next_agents = []
        death_ages = []
        for agent in self.agents:
            if agent.energy <= self.death_threshold:
                death_ages.append(int(agent.age))
                continue
            if agent.energy >= self.clone_threshold:
                agent.energy = 0.0
                child = self.agent_class(agent.x, agent.y, self.mutation_rate, weights=agent.weights)
                next_agents.extend((agent, child))
            else:
                next_agents.append(agent)
        self.agents = next_agents

        # 8. expire predators, then independently attempt new spawns per cell.
        self._update_predators()
        self.spawn_predators()
        # 9. replenish food; 10. step ends.
        self.spawn_food()
        return death_ages
