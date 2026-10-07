import random
import numpy as np
from src.agent import Agent

class Field:
    def __init__(
        self,
        field_size=10,
        initial_agents=20,
        mutation_rate=0.05,
        initial_food_prob=0.1,
        step_food_prob=0.003,
        food_durability=5.0,
        energy_loss_per_step=0.1,
        reward_food=1.0,
        clone_threshold=100.0,
        death_threshold=-50.0,
        max_transfer=10.0

    ):
        self.field_size = field_size
        self.mutation_rate = mutation_rate
        self.step_food_prob = step_food_prob
        self.food_durability = food_durability
        self.energy_loss_per_step = energy_loss_per_step
        self.reward_food = reward_food
        self.clone_threshold = clone_threshold
        self.death_threshold = death_threshold
        self.max_transfer = max_transfer

        self.grid_food = np.zeros((field_size, field_size), dtype=int)
        self.agents = [
            Agent(
                random.randint(0, field_size - 1), 
                random.randint(0, field_size - 1), 
                mutation_rate=mutation_rate
            ) 
            for _ in range(initial_agents)
        ]
        self.spawn_objects(prob=initial_food_prob)
        self.altruism_count = 0
        self.transfer_total = 0.0
        self.transfer_squared_total = 0.0

    def spawn_objects(self, prob=None):
        if prob is None:
            prob = self.step_food_prob
        spawn_mask = np.random.rand(self.field_size, self.field_size) < prob
        self.grid_food[spawn_mask] = self.food_durability

    def get_nearest_agent(self, agent):
        neighbors = []
        for other in self.agents:
            if other is agent:
                continue
            
            dx = min(abs(other.x - agent.x), self.field_size - abs(other.x - agent.x))
            dy = min(abs(other.y - agent.y), self.field_size - abs(other.y - agent.y))
            
            if dx <= 1 and dy <= 1:
                neighbors.append(other)
                
        return random.choice(neighbors) if neighbors else None

    def get_nearest_food(self, agent):
        neighbors_food = []
        food_positions = np.argwhere(self.grid_food > 0)
        for food in food_positions:
            dx = min(abs(food[0] - agent.x), self.field_size - abs(food[0] - agent.x))
            dy = min(abs(food[1] - agent.y), self.field_size - abs(food[1] - agent.y))

            if dx <= 1 and dy <= 1:
                neighbors_food.append(food)

        return random.choice(neighbors_food) if neighbors_food else None

    def get_torus_direction(self, src_x, src_y, target_x, target_y):
        diff_x = (target_x - src_x + self.field_size // 2) % self.field_size - self.field_size // 2
        diff_y = (target_y - src_y + self.field_size // 2) % self.field_size - self.field_size // 2

        dir_x = 1 if diff_x > 0 else (-1 if diff_x < 0 else 0)
        dir_y = 1 if diff_y > 0 else (-1 if diff_y < 0 else 0)
        return dir_x, dir_y

    def step(self):
        self.transfer_total = 0.0
        self.transfer_squared_total = 0.0
        initial_states = {
            agent: (agent.x, agent.y, agent.energy) for agent in self.agents
        }
        surviving_agents = []
        for agent in self.agents:
            agent.age += 1
            agent.energy -= self.energy_loss_per_step
            if agent.energy > self.death_threshold:
                surviving_agents.append(agent)
        self.agents = surviving_agents

        actions = []
        
        for agent in self.agents:
            nearest = self.get_nearest_agent(agent)
            nearest_food = self.get_nearest_food(agent)

            see_other = 1 if nearest else 0
            see_food = 1 if nearest_food is not None else 0
            other_energy = initial_states[nearest][2] if nearest else 0
            
            move_action, give_flag, transfer_amount = agent.decide_action(
                self.clone_threshold, 
                self.death_threshold, 
                see_food,
                initial_states[agent][2],
                see_other, 
                other_energy,
                max_transfer=self.max_transfer
            )
            actions.append((agent, tuple(move_action), give_flag, transfer_amount, nearest, nearest_food))

        # Iterate over saved actions so removing a dead donor does not skip anyone.
        for agent, move_action, give_flag, transfer_amount, nearest, nearest_food in actions:
            if nearest is not None and nearest.energy <= self.death_threshold:
                nearest = None

            if give_flag and nearest is not None and transfer_amount > 0:
                actual_transfer = min(transfer_amount, agent.energy - self.death_threshold)
                agent.energy -= actual_transfer
                nearest.energy += actual_transfer
                self.altruism_count += 1
                self.transfer_total += actual_transfer
                self.transfer_squared_total += actual_transfer ** 2
                if agent.energy <= self.death_threshold:
                    self.agents.remove(agent)
                    continue

            dx, dy = 0, 0
            if move_action == (0, 1) and nearest_food is not None:
                dx, dy = self.get_torus_direction(agent.x, agent.y, nearest_food[0], nearest_food[1])
            elif move_action == (1, 0) and nearest is not None:
                target_x, target_y, _ = initial_states[nearest]
                dir_x, dir_y = self.get_torus_direction(agent.x, agent.y, target_x, target_y)
                dx, dy = -dir_x, -dir_y
            elif move_action != (1, 1):
                dx, dy = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])

            agent.x = (agent.x + dx) % self.field_size
            agent.y = (agent.y + dy) % self.field_size

            if self.grid_food[agent.x, agent.y] > 0:
                agent.energy += self.reward_food
                self.grid_food[agent.x, agent.y] -= self.reward_food

        self.spawn_objects()

        next_agents = []
        for agent in self.agents:
            if agent.energy >= self.clone_threshold:
                agent.energy = 0
                child = Agent(agent.x, agent.y, mutation_rate=self.mutation_rate, weights=agent.weights)
                next_agents.extend([agent, child])
            else:
                next_agents.append(agent)
        self.agents = next_agents
