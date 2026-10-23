from src.cuda import load_v7_cuda


class AgentV7:
    # 変更(v7): テンソル演算の継承から、.cu内の行動・突然変異カーネル呼出しへ変更。
    @staticmethod
    def action_from_weights(weights, observations):
        import torch
        return load_v7_cuda().actions(weights.contiguous(),
                                      observations.to(dtype=torch.int64).contiguous())

    @staticmethod
    def mutate(weights, mutation_rate, generator):
        import torch
        uniforms = torch.rand(weights.shape, device=weights.device,
                              generator=generator, dtype=torch.float64)
        return load_v7_cuda().mutate(weights.contiguous(), uniforms, mutation_rate)
