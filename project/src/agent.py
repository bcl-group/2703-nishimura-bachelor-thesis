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
            self.weights = np.random.randint(0, 2, (3, 4))
        else:
            self.weights = self.mutate(weights)
            
    def mutate(self, weights):
        new_weights = weights.copy()
        for i in range(3):
            for j in range(4):
                if random.random() < self.mutation_rate:
                    new_weights[i, j] = 1 - new_weights[i, j]
        return new_weights

    def decide_action(self, on_food, our_energy, see_other, other_is_starving):
        input_vec = np.array([on_food, our_energy, see_other, other_is_starving])
        output_vec = np.dot(self.weights, input_vec) % 2
        
        move_action = output_vec[0:2]
        give_energy = output_vec[2]
        return move_action, give_energy