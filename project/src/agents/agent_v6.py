class AgentV6:
    @staticmethod
    def action_from_weights(weights, observations):
        return (weights * observations[..., None, :]).sum(dim=-1).remainder(2)

    @staticmethod
    def mutate(weights, mutation_rate, generator):
        import torch
        mask = torch.rand(weights.shape, device=weights.device, generator=generator,
                          dtype=torch.float64) < mutation_rate
        return torch.where(mask, 1 - weights, weights)
