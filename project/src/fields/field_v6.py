from src.agents.agent_v6 import AgentV6


def require_device(device, cuda=False):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("v6/v7 requires PyTorch; see README.md.") from exc
    device = torch.device(device)
    if cuda:
        if device.type != "cuda":
            raise ValueError("FieldV7 requires a CUDA device.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check PyTorch and the NVIDIA driver.")
        torch.empty(0, device=device)  # 出力作成前にGPU番号も検証する。
    elif device.type != "cpu":
        raise ValueError("FieldV6 requires device='cpu'.")
    return device


class FieldV6:
    agent_class = AgentV6
    cuda = False

    def __init__(self, field_size=10, initial_agents=20, mutation_rate=0.05,
                 initial_food_prob=0.1, step_food_prob=0.003, food_durability=5,
                 energy_loss_per_step=0.1, reward_food=1.0, clone_threshold=100.0,
                 death_threshold=-50.0, life_left=5.0, transfer_amount=1.0,
                 *, device="cpu", seed=42):
        self.device = require_device(device, self.cuda)
        import torch
        if type(field_size) is not int or field_size <= 0:
            raise ValueError("field_size must be a positive integer.")
        if type(initial_agents) is not int or initial_agents < 0:
            raise ValueError("initial_agents must be a nonnegative integer.")
        if type(food_durability) is not int or food_durability <= 0:
            raise ValueError("food_durability must be a positive integer.")
        self.field_size = field_size
        self.mutation_rate = mutation_rate
        self.step_food_prob = step_food_prob
        self.food_durability = food_durability
        self.energy_loss_per_step = energy_loss_per_step
        self.reward_food = reward_food
        self.clone_threshold = clone_threshold
        self.death_threshold = death_threshold
        self.starvation_threshold = death_threshold + life_left
        self.transfer_amount = transfer_amount
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
       
        self.positions = torch.randint(field_size, (initial_agents, 2),
                                       generator=self.generator, device=self.device)
        self.energies = torch.zeros(initial_agents, dtype=torch.float64, device=self.device)
        self.ages = torch.zeros(initial_agents, dtype=torch.int64, device=self.device)
        self.weights = torch.randint(2, (initial_agents, 2, 4),
                                     generator=self.generator, device=self.device)
        self.grid_food = torch.zeros((field_size, field_size), dtype=torch.int64,
                                     device=self.device)
     
        offsets = list(dict.fromkeys(i % field_size for i in (-1, 0, 1)))
        delta = torch.tensor([(x, y) for x in offsets for y in offsets], device=self.device)
        cells = torch.arange(field_size ** 2, device=self.device)
        xy = torch.stack((cells // field_size, cells % field_size), dim=1)
        adjacent = (xy[:, None, :] + delta).remainder(field_size)
        self.neighbor_cells = adjacent[..., 0] * field_size + adjacent[..., 1]
        self.altruism_count = torch.zeros((), dtype=torch.int64, device=self.device)
        self.acting_agent_count = initial_agents
        self.spawn_objects(initial_food_prob)

    @property
    def population(self):
        return self.positions.shape[0]

    def _random(self, shape):
        import torch
        return torch.rand(shape, dtype=torch.float64, device=self.device,
                          generator=self.generator)

    def spawn_objects(self, prob=None):
        prob = self.step_food_prob if prob is None else prob
        self.grid_food[self._random(self.grid_food.shape) < prob] = self.food_durability

    def prepare_step_inputs(self):
        import torch
        n = self.population
        ids = torch.arange(n, device=self.device)
        cells = self.positions[:, 0] * self.field_size + self.positions[:, 1]
        nearby = self.neighbor_cells[cells]
        order = torch.argsort(cells, stable=True)
        counts = torch.bincount(cells, minlength=self.field_size ** 2)
        starts = counts.cumsum(0) - counts
        own_cell = nearby == cells[:, None]
        available = counts[nearby] - own_cell.to(torch.int64)

        totals = available.sum(1)
        rank = (self._random((n,)) * totals).to(torch.int64)
        cumulative = available.cumsum(1)
        slot = (rank[:, None] >= cumulative).sum(1).clamp(max=nearby.shape[1] - 1)
        selected_cell = nearby.gather(1, slot[:, None]).squeeze(1)
        preceding = (cumulative - available).gather(1, slot[:, None]).squeeze(1)
        local_rank = rank - preceding
        inverse = torch.empty_like(order)
        inverse[order] = ids
        self_rank = inverse - starts[cells]
        local_rank += ((selected_cell == cells) & (local_rank >= self_rank)).to(torch.int64)
        selected_index = (starts[selected_cell] + local_rank).clamp(max=max(n - 1, 0))
        neighbors = order[selected_index]
        neighbors = torch.where(totals > 0, neighbors, -1)
        food_available = self.grid_food.flatten()[nearby] > 0
        food_totals = food_available.sum(1)
        food_rank = (self._random((n,)) * food_totals).to(torch.int64)
        food_slot = (food_rank[:, None] >= food_available.cumsum(1)).sum(1)
        food_slot = food_slot.clamp(max=nearby.shape[1] - 1)
        food_cells = nearby.gather(1, food_slot[:, None]).squeeze(1)
        food_cells = torch.where(food_totals > 0, food_cells, -1)
        directions = torch.randint(-1, 2, (n, 2), generator=self.generator, device=self.device)
        return neighbors, food_cells, directions

    def compute_agent_step(self, neighbors, food_cells, directions):
        import torch
        has_neighbor, has_food = neighbors >= 0, food_cells >= 0
        safe_neighbors = neighbors.clamp(min=0)
        observations = torch.stack((has_food, has_neighbor,
            self.energies <= self.starvation_threshold,
            has_neighbor & (self.energies[safe_neighbors] <= self.starvation_threshold)), 1)
        actions = self.agent_class.action_from_weights(self.weights, observations)
        seek_agent = actions[:, 1].bool()
        has_target = torch.where(seek_agent, has_neighbor, has_food)
        food_positions = torch.stack((food_cells // self.field_size,
                                      food_cells % self.field_size), 1)
        target = torch.where(seek_agent[:, None], self.positions[safe_neighbors], food_positions)
        diff = (target - self.positions + self.field_size // 2).remainder(self.field_size)
        diff -= self.field_size // 2
        movement = torch.sign(diff) * (1 - 2 * actions[:, :1])
        movement = torch.where(has_target[:, None], movement, directions)
        positions = (self.positions + movement).remainder(self.field_size)
        available = (self.energies - self.energy_loss_per_step - self.death_threshold).clamp(min=0)
        transfer = torch.minimum(available, torch.full_like(available, self.transfer_amount)).clamp(min=0)
        transfer = torch.where((actions[:, 0] == 0) & seek_agent & has_neighbor, transfer, 0.)
        return positions, transfer

    def resolve_food(self, positions):
        import torch
        n = self.population
        cells = positions[:, 0] * self.field_size + positions[:, 1]
        # 変更(v6): 一様ランダム順列をセルで安定ソートし、耐久度以内の個体に給餌。
        # 乱数キーの同点偏りを避け、非復元抽選を維持する。
        lottery = torch.randperm(n, generator=self.generator, device=self.device)
        order = lottery[torch.argsort(cells[lottery], stable=True)]
        sorted_cells = cells[order]
        counts = torch.bincount(cells, minlength=self.field_size ** 2)
        starts = counts.cumsum(0) - counts
        ranks = torch.arange(n, device=self.device) - starts[sorted_cells]
        winners = ranks < self.grid_food.flatten()[sorted_cells]
        rewards = torch.zeros(n, dtype=torch.float64, device=self.device)
        rewards[order] = winners.to(torch.float64) * self.reward_food
        consumed = torch.minimum(counts, self.grid_food.flatten()).reshape(self.grid_food.shape)
        return rewards, consumed

    def step(self, inputs=None):
        import torch
        n = self.population
        self.acting_agent_count = n
        if n == 0:
            self.altruism_count = torch.zeros((), dtype=torch.int64, device=self.device)
            self.positive_altruism_count = self.altruism_count
            self.spawn_objects()
            return self.ages.clone()
        # 変更(v6): 全ての計算が更新前の状態を参照し、最後に一括コミットする。
        neighbors, foods, directions = self.prepare_step_inputs() if inputs is None else inputs
        positions, transfers = self.compute_agent_step(neighbors, foods, directions)
        rewards, consumed = self.resolve_food(positions)
        incoming = torch.zeros_like(self.energies)
        incoming.scatter_add_(0, neighbors.clamp(min=0), transfers)
        energies = self.energies - self.energy_loss_per_step - transfers + incoming + rewards
        ages = self.ages + 1
        self.altruism_count = (transfers > 0).sum()
        self.positive_altruism_count = self.altruism_count
        alive = energies > self.death_threshold
        death_ages = ages[~alive]
        positions, energies, ages, weights = positions[alive], energies[alive], ages[alive], self.weights[alive]
        clone = energies >= self.clone_threshold
        # 変更(v6): 生存親・子の順序をv5と同じ「親の直後に子」で維持する。
        repeats = 1 + clone.to(torch.int64)
        parents = torch.repeat_interleave(torch.arange(energies.shape[0], device=self.device), repeats)
        starts = repeats.cumsum(0) - repeats
        child = torch.arange(parents.shape[0], device=self.device) != starts[parents]
        energies = torch.where(clone, 0., energies)
        self.positions = positions[parents]
        self.energies = energies[parents]
        self.ages = torch.where(child, 0, ages[parents])
        self.weights = weights[parents].clone()
        self.weights[child] = self.agent_class.mutate(self.weights[child], self.mutation_rate, self.generator)
        self.grid_food -= consumed
        self.spawn_objects()
        return death_ages

    def genotype_counts(self):
        import torch
        # 変更(v6): 8ビットの重みを整数化してデバイス上で集計し、256値だけ転送する。
        powers = 2 ** torch.arange(7, -1, -1, device=self.device)
        codes = (self.weights.reshape(-1, 8) * powers).sum(1)
        counts = torch.bincount(codes, minlength=256).cpu().tolist()
        return {format(code, '08b'): count for code, count in enumerate(counts) if count}
