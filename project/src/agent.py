import numpy as np
import random

class Agent:
    def __init__(self, x, y, mutation_rate=0.05, weights=None):
        self.x = x
        self.y = y
        self.energy = 0.0
        self.age = 0
        self.mutation_rate = mutation_rate
        
        if weights is None:
            self.weights = np.random.uniform(-1.0, 1.0, (4, 4))
        else:
            self.weights = self.mutate(weights)
            
    def mutate(self, weights):
        mutation_mask = np.random.rand(4, 4) < self.mutation_rate
        mutation_amounts = np.random.uniform(-0.2, 0.2, (4, 4))
        return weights + (mutation_mask * mutation_amounts)

    def decide_action(self, clone_threshold, death_threshold, on_food, our_energy, see_other, other_energy, max_transfer):
        normalized_my_energy = (our_energy - death_threshold) / (clone_threshold - death_threshold)
        normalized_other_energy = (other_energy - death_threshold) / (clone_threshold - death_threshold)

        input_vec = np.array([on_food, normalized_my_energy, see_other, normalized_other_energy], dtype=float)
        raw_output = np.dot(self.weights, input_vec)
        activated_output = 1 / (1 + np.exp(-raw_output))

        threshold = 0.5
        move_action = (activated_output[0:2] >= threshold).astype(int)
        give_flag = activated_output[2] >= threshold

        if give_flag:
            transfer_amount = activated_output[3] * max_transfer
            
        return move_action, give_flag, transfer_amount