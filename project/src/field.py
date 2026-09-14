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
        transfer_amount=1.0,
        starving_threshold=0.0
    ):
        self.field_size = field_size
        self.mutation_rate = mutation_rate
        self.step_food_prob = step_food_prob
        self.food_durability = food_durability
        self.energy_loss_per_step = energy_loss_per_step
        self.reward_food = reward_food
        self.clone_threshold = clone_threshold
        self.death_threshold = death_threshold
        self.transfer_amount = transfer_amount
        self.starving_threshold = starving_threshold

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
        for i in range(self.field_size):
            for j in range(self.field_size):
                if random.random() < prob:
                    self.grid_food[i, j] = self.food_durability

    def get_nearest_agent(self, agent):
        neighbors = [
            other for other in self.agents 
            if other is not agent and abs(other.x - agent.x) <= 1 and abs(other.y - agent.y) <= 1
        ]
        return random.choice(neighbors) if neighbors else None

    def _get_torus_direction(self, src_x, src_y, target_x, target_y):
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
            agent.energy -= self.energy_loss_per_step
            
            on_food = 1 if self.grid_food[agent.x, agent.y] > 0 else 0
            nearest = self.get_nearest_agent(agent)
            nearest_agents.append(nearest)
            
            see_other = 1 if nearest else 0
            other_is_starving = 1 if nearest and nearest.energy < self.starving_threshold else 0
            
            move_action, give_energy = agent.decide_action(on_food, see_other, other_is_starving)
            actions.append((move_action, give_energy))

        for i, agent in enumerate(self.agents):
            move_action, give_energy = actions[i]
            nearest = nearest_agents[i]
            
            if give_energy == 1 and nearest and agent.energy > self.starving_threshold:
                agent.energy -= self.transfer_amount
                nearest.energy += self.transfer_amount
                self.altruism_count += 1
                
            dx, dy = 0, 0
            if move_action[0] == 0 and move_action[1] == 1:
                if nearest:
                    dx, dy = self._get_torus_direction(agent.x, agent.y, nearest.x, nearest.y)
                else:
                    dx, dy = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])
            elif move_action[0] == 1 and move_action[1] == 0:
                if nearest:
                    dir_x, dir_y = self._get_torus_direction(agent.x, agent.y, nearest.x, nearest.y)
                    dx, dy = -dir_x, -dir_y
                else:
                    dx, dy = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])
            elif move_action[0] == 0 and move_action[0] == 0:
                dx, dy = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])
            elif move_action[0] == 1 and move_action[1] == 1:
                dx, dy = 0, 0

            agent.x = (agent.x + dx) % self.field_size
            agent.y = (agent.y + dy) % self.field_size
            
            if self.grid_food[agent.x, agent.y] > 0:
                agent.energy += self.reward_food
                self.grid_food[agent.x, agent.y] -= 1

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