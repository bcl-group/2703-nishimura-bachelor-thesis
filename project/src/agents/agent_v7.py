import numpy as np


class AgentV7:
    weight_shape = (2, 2)

    def __init__(self, x, y, mutation_rate, weights=None):
        self.x = x
        self.y = y
        self.energy = 0.0
        self.age = 0
        self.mutation_rate = mutation_rate
        if weights is None:
            self.weights = np.random.randint(0, 2, self.weight_shape)
        else:
            self.weights = self.mutate(weights)

    def mutate(self, weights):
        mutation_mask = np.random.rand(*self.weight_shape) < self.mutation_rate
        new_weights = np.where(mutation_mask, 1 - weights, weights)
        return new_weights.astype(int)

    def decide_action(self, see_food, see_other):
        return self.action_from_weights(self.weights, see_food, see_other)

    @staticmethod
    def action_from_weights(weights, *observations):
        return np.dot(weights, observations) % 2
