import random
import numpy as np
from src.agent import Agent

class Field:
    def __init__(
        self,
        field_size=10,
        initial_agents=20,
        mutation_rate=0.05,
        initial_food_prob=0.2,
        step_food_prob=0.005,
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

    def get_torus_direction(self, src_x, src_y, target_x, target_y):
        diff_x = (target_x - src_x + self.field_size // 2) % self.field_size - self.field_size // 2
        diff_y = (target_y - src_y + self.field_size // 2) % self.field_size - self.field_size // 2

        dir_x = 1 if diff_x > 0 else (-1 if diff_x < 0 else 0)
        dir_y = 1 if diff_y > 0 else (-1 if diff_y < 0 else 0)
        return dir_x, dir_y

    def step(self):
        actions = []
        nearest_agents = []
        
        for agent in self.agents:
            agent.age += 1
            on_food = 1 if self.grid_food[agent.x, agent.y] > 0 else 0
            nearest = self.get_nearest_agent(agent)
            nearest_agents.append(nearest)
            
            see_other = 1 if nearest else 0
            other_energy = nearest.energy if nearest else 0
            
            move_action, give_flag, transfer_amount = agent.decide_action(
                self.clone_threshold, 
                self.death_threshold, 
                on_food, 
                agent.energy, 
                see_other, 
                other_energy,
                max_transfer=self.max_transfer
            )
            actions.append((move_action, give_flag, transfer_amount))

        for i, agent in enumerate(self.agents):
            agent.energy -= self.energy_loss_per_step
            move_action, give_flag, transfer_amount = actions[i]
            nearest = nearest_agents[i]

            if give_flag and nearest is not None:
                actual_transfer = min(transfer_amount, agent.energy - self.death_threshold)
                agent.energy -= actual_transfer
                nearest.energy += actual_transfer
                self.altruism_count += 1

            dx, dy = 0, 0
            if move_action[0] == 0 and move_action[1] == 1:
                if nearest:
                    dx, dy = self.get_torus_direction(agent.x, agent.y, nearest.x, nearest.y)
                else:
                    dx, dy = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])
            elif move_action[0] == 1 and move_action[1] == 0:
                if nearest:
                    dir_x, dir_y = self.get_torus_direction(agent.x, agent.y, nearest.x, nearest.y)
                    dx, dy = -dir_x, -dir_y
                else:
                    dx, dy = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])
            elif move_action[0] == 0 and move_action[1] == 0:
                dx, dy = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])
            elif move_action[0] == 1 and move_action[1] == 1:
                dx, dy = 0, 0

            agent.x = (agent.x + dx) % self.field_size
            agent.y = (agent.y + dy) % self.field_size

            if self.grid_food[agent.x, agent.y] > 0:
                agent.energy += self.reward_food
                self.grid_food[agent.x, agent.y] -= self.reward_food

        self.spawn_objects()

        next_agents = []
        for agent in self.agents:
            if agent.energy <= self.death_threshold:
                continue
            if agent.energy >= self.clone_threshold:
                agent.energy = 0
                child = Agent(agent.x, agent.y, mutation_rate=self.mutation_rate, weights=agent.weights)
                next_agents.extend([agent, child])
            else:
                next_agents.append(agent)
        self.agents = next_agents