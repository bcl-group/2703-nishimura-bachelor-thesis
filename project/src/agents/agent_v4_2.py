"""Binary 4-output controller for the v4.2 predator/altruism model."""
import numpy as np


class AgentV4_2:
    # Rows 0..2 encode the 3-bit movement action; row 3 is altruism.
    # Columns: food, other agent, self starving, other starving, predator.
    weight_shape = (4, 5)

    def __init__(self, x, y, mutation_rate, weights=None):
        self.x = int(x)
        self.y = int(y)
        self.energy = 0.0
        self.age = 0
        self.mutation_rate = float(mutation_rate)
        if weights is None:
            self.weights = np.random.randint(0, 2, self.weight_shape, dtype=int)
        else:
            weights = np.asarray(weights, dtype=int)
            if weights.shape != self.weight_shape:
                raise ValueError(f"weights must have shape {self.weight_shape}")
            self.weights = self.mutate(weights)

    def mutate(self, weights):
        mask = np.random.random(self.weight_shape) < self.mutation_rate
        return np.where(mask, 1 - weights, weights).astype(int)

    @staticmethod
    def action_from_weights(weights, *observations):
        observations = np.asarray(observations, dtype=int)
        if observations.shape != (5,):
            raise ValueError("v4.2 requires exactly five binary observations")
        return np.dot(np.asarray(weights, dtype=int), observations) % 2
