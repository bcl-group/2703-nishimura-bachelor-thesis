import numpy as np


class AgentV4:
    def __init__(self, x, y, mutation_rate, weights=None):
        self.x = x
        self.y = y
        self.energy = 0.0
        self.age = 0
        self.mutation_rate = mutation_rate
        if weights is None:
            self.weights = np.random.randint(0, 2, (2, 4))
        else:
            self.weights = self.mutate(weights)

    def mutate(self, weights):
        mutation_mask = np.random.rand(2, 4) < self.mutation_rate
        new_weights = np.where(mutation_mask, 1 - weights, weights)
        return new_weights.astype(int)

    def decide_action(self, see_food, see_other, me_starving, other_starving):
        return self.action_from_weights(
            self.weights, see_food, see_other, me_starving, other_starving
        )

    @staticmethod
    def action_from_weights(weights, see_food, see_other, me_starving, other_starving):
        input_vec = np.array([see_food, see_other, me_starving, other_starving])
        raw_output = np.dot(weights, input_vec) % 2
        return raw_output[0:2]
