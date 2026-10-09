import numpy as np

class AgentV3:
    def __init__(self, x, y, mutation_rate, weights=None):
        self.x = x
        self.y = y
        self.energy = 0.0
        self.age = 0
        self.mutation_rate = mutation_rate
        
        if weights is None:
            self.weights = np.random.randint(0, 2, (4, 2)) 
        else:
            self.weights = self.mutate(weights)
            
    def mutate(self, weights):
        mutation_mask = np.random.rand(4, 2) < self.mutation_rate
        new_weights = np.where(mutation_mask, 1 - weights, weights)
        return new_weights.astype(int)

    def decide_action(self, see_food, see_other, max_transfer):
        input_vec = np.array([see_food, see_other])
        raw_output = np.dot(self.weights, input_vec) % 2

        move_action = raw_output[0:2] 
        give_flag = raw_output[2] 

        if give_flag:
            transfer_amount = float((raw_output[3] / 2.0) * max_transfer)
        else:
            transfer_amount = 0.0

        return move_action, give_flag, transfer_amount