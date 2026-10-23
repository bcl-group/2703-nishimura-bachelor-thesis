"""変更(v7): Pythonは状態保持と起動を担当し、計算本体はCUDA C++で実行する。"""
from src.agents.agent_v7 import AgentV7
from src.cuda import load_v7_cuda
from .field_v6 import FieldV6, require_device


class FieldV7(FieldV6):
    agent_class = AgentV7
    cuda = True

    def __init__(self, *args, device="cuda:0", **kwargs):
        require_device(device, cuda=True)
        # 変更(v7): 出力作成前のField初期化時にnvccによるビルドも検証する。
        self._cuda = load_v7_cuda()
        super().__init__(*args, device=device, **kwargs)

    def spawn_objects(self, prob=None):
        import torch
        probability = self.step_food_prob if prob is None else prob
        # 変更(v7): 初期餌生成にも同じCUDAカーネルを使う。
        consumed = torch.zeros_like(self.grid_food)
        self._cuda.spawn(self.grid_food, consumed, self._random(self.grid_food.shape),
                         probability, self.food_durability)

    def prepare_step_inputs(self):
        import torch
        directions = torch.randint(-1, 2, (self.population, 2),
                                   generator=self.generator, device=self.device)
        return self._cuda.prepare(self.positions, self.grid_food, self.neighbor_cells,
                                  self._random((self.population, 2)), directions)

    def compute_agent_step(self, neighbors, food_cells, directions):
        return self._cuda.compute(self.positions, self.energies, self.weights,
                                  neighbors, food_cells, directions, self.field_size,
                                  self.starvation_threshold, self.energy_loss_per_step,
                                  self.death_threshold, self.transfer_amount)

    def resolve_food(self, positions):
        import torch
        # 変更(v7): 一様順列はPyTorchのGPU乱数器で生成し、競合処理は.cuで行う。
        priorities = torch.randperm(self.population, generator=self.generator, device=self.device)
        return self._cuda.food(positions, self.grid_food, priorities, self.reward_food)

    def step(self, inputs=None):
        # 変更(v7): 近傍探索→行動→餌→移譲集計→死亡・繁殖→餌生成をCUDAで実行する。
        # カーネルはPyTorchの現在のストリームを使い、更新前の状態を共通参照する。
        self.acting_agent_count = self.population
        if inputs is not None:
            self._validate_inputs(inputs)
        neighbors, foods, directions = self.prepare_step_inputs() if inputs is None else inputs
        positions, transfers = self.compute_agent_step(neighbors, foods, directions)
        rewards, consumed = self.resolve_food(positions)
        outputs = self._cuda.finish(
            positions, self.energies, self.ages, self.weights, neighbors, transfers, rewards,
            self._random((self.population, 8)), self.energy_loss_per_step,
            self.death_threshold, self.clone_threshold, self.mutation_rate)
        (self.positions, self.energies, self.ages, self.weights,
         death_ages, self.altruism_count) = outputs
        self.positive_altruism_count = self.altruism_count
        self._cuda.spawn(self.grid_food, consumed, self._random(self.grid_food.shape),
                         self.step_food_prob, self.food_durability)
        return death_ages

    def _validate_inputs(self, inputs):
        import torch
        # 外部から固定観測を注入するテスト時だけ範囲を検証。通常ステップには転送を追加しない。
        if len(inputs) != 3:
            raise ValueError("inputs must contain neighbors, food cells and directions")
        shapes = [(self.population,), (self.population,), (self.population, 2)]
        for tensor, shape in zip(inputs, shapes):
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                    or tensor.dtype != torch.int64 or tensor.device != self.positions.device
                    or not tensor.is_contiguous()):
                raise ValueError("Invalid CUDA step input shape, dtype or device")
        neighbors, food, directions = inputs
        if (torch.any((neighbors < -1) | (neighbors >= self.population)).item()
                or torch.any((food < -1) | (food >= self.field_size ** 2)).item()
                or torch.any((directions < -1) | (directions > 1)).item()):
            raise ValueError("CUDA step input index or direction is out of range")
