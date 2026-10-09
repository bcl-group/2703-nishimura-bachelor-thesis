import numpy as np

class AgentV2:
    def __init__(self, x, y, mutation_rate, weights=None):
        self.x = x
        self.y = y
        self.energy = 0.0
        self.age = 0
        self.mutation_rate = mutation_rate
        
        if weights is None:
            self.weights = np.random.randint(0, 2, (4, 4)) 
        else:
            self.weights = self.mutate(weights)
            
    def mutate(self, weights):
        mutation_mask = np.random.rand(4, 4) < self.mutation_rate
        new_weights = np.where(mutation_mask, 1 - weights, weights)
        return new_weights.astype(int)

    def decide_action(self, clone_threshold, death_threshold, see_food, my_energy, see_other, other_energy, max_transfer):
        norm_my_e = np.clip((my_energy - death_threshold) / (clone_threshold - death_threshold), 0.0, 1.0)
        norm_other_e = np.clip((other_energy - death_threshold) / (clone_threshold - death_threshold), 0.0, 1.0)

        input_vec = np.array([see_food, norm_my_e, see_other, norm_other_e], dtype=float)
        raw_output = np.dot(self.weights, input_vec)
        
        threshold = 2.0
        move_action = (raw_output[0:2] >= threshold).astype(int)
        give_flag = raw_output[2] >= threshold

        if give_flag:
            transfer_amount = float((raw_output[3] / 4.0) * max_transfer)
        else:
            transfer_amount = 0.0

        return move_action, give_flag, transfer_amount