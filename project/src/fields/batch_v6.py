class BatchV6:
    def __init__(self, fields):
        import torch
        if not fields:
            raise ValueError("A batch needs at least one field.")
        first = fields[0]
        self.device, self.field_size = first.device, first.field_size
        if any(f.device != self.device or f.field_size != self.field_size for f in fields):
            raise ValueError("Batch fields must have the same device and field_size.")
        self.size = len(fields)
        self.generators = [f.generator for f in fields]
        self.counts = [f.population for f in fields]
        self.capacity = max(1, max(self.counts))
        self.neighbor_cells = first.neighbor_cells
        self.grid_food = torch.stack([f.grid_food for f in fields])
        self.parameters = {
            name: torch.tensor([getattr(f, name) for f in fields], device=self.device,
                               dtype=torch.int64 if name == "food_durability" else torch.float64)[:, None]
            for name in ("energy_loss_per_step", "reward_food", "death_threshold",
                         "clone_threshold", "starvation_threshold", "transfer_amount",
                         "mutation_rate", "step_food_prob", "food_durability")
        }
        for name, tail, dtype in (("positions", (2,), torch.int64),
                                  ("energies", (), torch.float64),
                                  ("ages", (), torch.int64),
                                  ("weights", (2, 4), torch.int64)):
            value = torch.zeros((self.size, self.capacity, *tail), device=self.device, dtype=dtype)
            for row, field in enumerate(fields):
                value[row, :field.population] = getattr(field, name)
            setattr(self, name, value)
        self.populations = torch.tensor(self.counts, device=self.device)
        self.altruism_count = torch.zeros_like(self.populations)

    def _mask(self):
        import torch
        return torch.arange(self.capacity, device=self.device)[None, :] < self.populations[:, None]

    def _random_inputs(self):
        """Only real agents consume random numbers; capacity is irrelevant."""
        import torch
        b, c = self.size, self.capacity
        nearby = torch.zeros((b, c), dtype=torch.float64, device=self.device)
        food = torch.zeros_like(nearby)
        directions = torch.zeros((b, c, 2), dtype=torch.int64, device=self.device)
        lottery = torch.arange(c, device=self.device).expand(b, c).clone()
        for row, (n, generator) in enumerate(zip(self.counts, self.generators)):
            if not n:
                continue
            nearby[row, :n] = torch.rand(n, generator=generator, device=self.device, dtype=torch.float64)
            food[row, :n] = torch.rand(n, generator=generator, device=self.device, dtype=torch.float64)
            directions[row, :n] = torch.randint(-1, 2, (n, 2), generator=generator, device=self.device)
            lottery[row, :n] = torch.randperm(n, generator=generator, device=self.device)
        return nearby, food, directions, lottery

    def prepare_step_inputs(self, nearby_random, food_random, directions):
        import torch
        b, c, s = self.size, self.capacity, self.field_size
        g = s * s
        rows = torch.arange(b, device=self.device)[:, None]
        ids = torch.arange(c, device=self.device)[None, :].expand(b, c)
        valid = self._mask()
        cells = self.positions[..., 0] * s + self.positions[..., 1]
        # Padding sorts behind all real cells and never enters the spatial index.
        keys = torch.where(valid, cells, g)
        order = keys.argsort(dim=1, stable=True)
        counts = torch.bincount((rows * g + cells)[valid], minlength=b*g).reshape(b, g)
        starts = counts.cumsum(1) - counts
        nearby = self.neighbor_cells[cells]
        available = counts[rows[..., None], nearby] - (nearby == cells[..., None]).to(torch.int64)
        available = available.clamp(min=0)
        totals = available.sum(2)
        rank = (nearby_random * totals).to(torch.int64)
        cumulative = available.cumsum(2)
        slot = (rank[..., None] >= cumulative).sum(2).clamp(max=nearby.shape[2]-1)
        selected = nearby.gather(2, slot[..., None]).squeeze(2)
        preceding = (cumulative - available).gather(2, slot[..., None]).squeeze(2)
        inverse = torch.empty_like(order)
        inverse.scatter_(1, order, ids)
        self_rank = inverse - starts.gather(1, cells)
        local_rank = rank - preceding
        local_rank += ((selected == cells) & (local_rank >= self_rank)).to(torch.int64)
        index = (starts.gather(1, selected) + local_rank).clamp(0, c-1)
        neighbors = order.gather(1, index)
        neighbors = torch.where(valid & (totals > 0), neighbors, -1)
        food_available = self.grid_food.flatten(1)[rows[..., None], nearby] > 0
        food_rank = (food_random * food_available.sum(2)).to(torch.int64)
        food_slot = (food_rank[..., None] >= food_available.cumsum(2)).sum(2)
        food_slot = food_slot.clamp(max=nearby.shape[2]-1)
        food_cells = nearby.gather(2, food_slot[..., None]).squeeze(2)
        food_cells = torch.where(valid & food_available.any(2), food_cells, -1)
        return neighbors, food_cells, directions

    def decide_actions(self, inputs):
        """Stage 1: read old state and emit movement, recipient and food request."""
        import torch
        from src.agents.agent_v6 import AgentV6
        neighbors, foods, directions = inputs
        rows = torch.arange(self.size, device=self.device)[:, None]
        safe = neighbors.clamp(min=0)
        has_neighbor, has_food = neighbors >= 0, foods >= 0
        p = self.parameters
        observations = torch.stack((has_food, has_neighbor,
            self.energies <= p['starvation_threshold'],
            has_neighbor & (self.energies[rows, safe] <= p['starvation_threshold'])), -1)
        actions = AgentV6.action_from_weights(self.weights, observations)
        seek_agent = actions[..., 1].bool()
        target = torch.where(seek_agent[..., None], self.positions[rows, safe],
                             torch.stack((foods // self.field_size, foods % self.field_size), -1))
        diff = (target - self.positions + self.field_size//2).remainder(self.field_size)
        diff -= self.field_size//2
        movement = torch.sign(diff) * (1 - 2 * actions[..., :1])
        has_target = torch.where(seek_agent, has_neighbor, has_food)
        movement = torch.where(has_target[..., None], movement, directions)
        positions = (self.positions + movement).remainder(self.field_size)
        available = (self.energies - p['energy_loss_per_step'] - p['death_threshold']).clamp(min=0)
        transfers = torch.minimum(available, p['transfer_amount']).clamp(min=0)
        transfers = torch.where(self._mask() & (actions[..., 0] == 0) & seek_agent & has_neighbor,
                                transfers, 0.)
        requests = positions[..., 0] * self.field_size + positions[..., 1]
        return positions, neighbors, transfers, requests

    def resolve_grid(self, requests, lottery):
        """Stage 2: uniform sampling without replacement within every cell."""
        import torch
        b, c, g = self.size, self.capacity, self.field_size**2
        rows = torch.arange(b, device=self.device)[:, None]
        valid = self._mask()
        keys = torch.where(valid, requests, g)
        order = lottery.gather(1, keys.gather(1, lottery).argsort(dim=1, stable=True))
        sorted_cells = keys.gather(1, order)
        counts = torch.bincount((rows*g + requests)[valid], minlength=b*g).reshape(b, g)
        starts = counts.cumsum(1) - counts
        safe_cells = sorted_cells.clamp(max=g-1)
        rank = torch.arange(c, device=self.device)[None, :] - starts.gather(1, safe_cells)
        winners = (sorted_cells < g) & (rank < self.grid_food.flatten(1).gather(1, safe_cells))
        won = torch.zeros_like(winners)
        won.scatter_(1, order, winners)
        consumed = torch.minimum(counts, self.grid_food.flatten(1)).reshape_as(self.grid_food)
        # Replenishment happens after state update, preserving RNG ordering.
        return won, consumed

    def update_agents(self, positions, recipients, transfers, won):
        """Stage 3: deterministic recipient reduction, energy, death and birth."""
        import torch
        b, c = self.size, self.capacity
        rows = torch.arange(b, device=self.device)[:, None]
        valid = self._mask()
        # A stable sort fixes summation order; segment_reduce avoids floating atomics
        # and prefix subtraction across unrelated recipients.
        keys = (rows*c + recipients.clamp(min=0))[valid]
        values = transfers[valid]
        order = keys.argsort(stable=True)
        lengths = torch.bincount(keys, minlength=b*c)
        incoming = torch.segment_reduce(values[order], 'sum', lengths=lengths).reshape(b, c)
        p = self.parameters
        energies = self.energies - p['energy_loss_per_step'] - transfers + incoming
        energies += won.to(torch.float64) * p['reward_food']
        ages = self.ages + 1
        alive = valid & (energies > p['death_threshold'])
        dead = valid & ~alive
        self.death_ages = torch.where(dead, ages, 0)
        self.death_counts = dead.sum(1)
        self.altruism_count = (transfers > 0).sum(1)
        clone = alive & (energies >= p['clone_threshold'])
        next_counts = (alive.sum(1) + clone.sum(1))
        # One host synchronization supplies both dynamic capacity and RNG shapes.
        count_data = torch.stack((next_counts, clone.sum(1))).cpu().tolist()
        counts, children = count_data
        capacity = max(c, max(counts))
        if capacity > c:
            capacity = max(capacity, c*2)
        # Interleaved candidates preserve the original parent-immediately-child order.
        keep = torch.stack((alive, clone), 2).reshape(b, 2*c)
        destinations = keep.to(torch.int64).cumsum(1) - 1
        row_ids = rows.expand(b, 2*c)[keep]
        dest = destinations[keep]
        parents = torch.arange(c, device=self.device).repeat_interleave(2)[None, :].expand(b, 2*c)
        parent_ids = parents[keep]
        child = (torch.arange(2*c, device=self.device)[None, :].expand(b, 2*c)[keep] % 2 == 1)
        next_positions = torch.zeros((b, capacity, 2), dtype=torch.int64, device=self.device)
        next_energies = torch.zeros((b, capacity), dtype=torch.float64, device=self.device)
        next_ages = torch.zeros((b, capacity), dtype=torch.int64, device=self.device)
        next_weights = torch.zeros((b, capacity, 2, 4), dtype=torch.int64, device=self.device)
        next_positions[row_ids, dest] = positions[row_ids, parent_ids]
        next_energies[row_ids, dest] = torch.where(clone, 0., energies)[row_ids, parent_ids]
        next_ages[row_ids, dest] = torch.where(child, 0, ages[row_ids, parent_ids])
        next_weights[row_ids, dest] = self.weights[row_ids, parent_ids]
        # Draw only for actual children, in their compacted order.
        mutation = torch.zeros((b, c, 2, 4), dtype=torch.bool, device=self.device)
        for row, (n, generator) in enumerate(zip(children, self.generators)):
            if n:
                mutation[row, :n] = torch.rand((n, 2, 4), generator=generator,
                    device=self.device, dtype=torch.float64) < p['mutation_rate'][row]
        child_rank = clone.to(torch.int64).cumsum(1) - 1
        child_rows, child_dest = row_ids[child], dest[child]
        flips = mutation[child_rows, child_rank[child_rows, parent_ids[child]]]
        next_weights[child_rows, child_dest] ^= flips.to(torch.int64)
        self.positions, self.energies = next_positions, next_energies
        self.ages, self.weights = next_ages, next_weights
        self.capacity, self.counts, self.populations = capacity, counts, next_counts

    def step(self, inputs=None, lottery=None):
        import torch
        random_neighbor, random_food, directions, random_lottery = self._random_inputs()
        if inputs is None:
            inputs = self.prepare_step_inputs(random_neighbor, random_food, directions)
        self.acting_agent_count = self.populations.clone()
        positions, recipients, transfers, requests = self.decide_actions(inputs)
        won, consumed = self.resolve_grid(requests, random_lottery if lottery is None else lottery)
        self.update_agents(positions, recipients, transfers, won)
        self.grid_food -= consumed
        # Spawn uses preexisting streams even for empty simulations.
        spawn_random = torch.stack([torch.rand((self.field_size, self.field_size),
            generator=g, device=self.device, dtype=torch.float64) for g in self.generators])
        p = self.parameters
        self.grid_food = torch.where(spawn_random < p['step_food_prob'][..., None],
                                     p['food_durability'].to(torch.int64)[..., None], self.grid_food)
        return self.death_ages

    def genotype_counts(self):
        """Return B x 256 device counts, excluding padding."""
        import torch
        powers = 2 ** torch.arange(7, -1, -1, device=self.device)
        codes = (self.weights.flatten(2) * powers).sum(2)
        rows = torch.arange(self.size, device=self.device)[:, None]
        return torch.bincount((rows*256 + codes)[self._mask()], minlength=self.size*256).reshape(self.size, 256)

    def copy_to_field(self, field, row=0):
        n = self.counts[row]
        for name in ('positions', 'energies', 'ages', 'weights'):
            setattr(field, name, getattr(self, name)[row, :n].clone())
        field.grid_food = self.grid_food[row].clone()
        field.altruism_count = self.altruism_count[row]
        field.acting_agent_count = int(self.acting_agent_count[row].item())
