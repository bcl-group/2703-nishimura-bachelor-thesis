import numpy as np

class AgentV3:
    def __init__(self, x, y, mutation_rate, weights=None):
        self.x = x
        self.y = y
        self.energy = 0.0
        self.age = 0
        self.altruism_level = 0
        self.mutation_rate = mutation_rate
        
        if weights is None:
            self.weights = np.random.randint(0, 2, (3, 3))
        else:
            self.weights = self.mutate(weights)
            
    def mutate(self, weights):
        mutation_mask = np.random.rand(3, 3) < self.mutation_rate
        new_weights = np.where(mutation_mask, 1 - weights, weights)
        return new_weights.astype(int)

    def decide_action(self, see_food, see_other, other_altruism_level):
        input_vec = np.array([see_food, see_other, other_altruism_level])
        raw_output = np.dot(self.weights, input_vec) % 2
    
        move_action = raw_output[0:2] 
        new_altruism_level = raw_output[2]  
        self.altruism_level = new_altruism_level

        return move_action