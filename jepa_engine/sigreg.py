"""
jepa_engine/sigreg.py —— JEPA 防坍塌正则化（SIGReg）

SIGReg 通过将潜空间投影到随机方向并用草图高斯特征函数检验分布，
惩罚"坍塌"（所有向量退化到同一点）和"过度集中"，保持潜空间多样性。

v8.1 优化：
- 投影矩阵从每次 forward 随机生成改为 init 时预生成并注册为 buffer
- 消除 D=16384 时每次创建 128MB 临时矩阵的开销
- 支持延迟初始化（第一次看到 embedding 维度时才创建投影矩阵）
"""
import torch
import torch.nn as nn
from mvp_config import config


class SIGReg(nn.Module):
    def __init__(self, knots=None, num_proj=None):
        super().__init__()
        knots = knots or config.jepa_sigreg_knots
        self.num_proj = num_proj or config.jepa_sigreg_num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

        # v8.1: 预生成固定投影矩阵（延迟初始化，第一次 forward 时根据维度创建）
        # 避免每次 forward 创建临时矩阵，D=16384 时从 128MB/次 降为 0
        self._proj_dim = None
        self.register_buffer("proj", torch.empty(0))  # 占位，延迟填充

    def _lazy_init_proj(self, dim: int, device: torch.device):
        """延迟初始化投影矩阵：首次调用时创建，后续复用。"""
        proj = torch.randn(dim, self.num_proj, device=device)
        proj = proj / proj.norm(p=2, dim=0, keepdim=True)
        self.proj = proj  # 覆盖 buffer
        self._proj_dim = dim

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.dim() == 3:
            embeddings = embeddings.reshape(-1, embeddings.size(-1))

        dim = embeddings.size(-1)

        # 延迟初始化：第一次看到数据维度时创建投影矩阵
        if self._proj_dim != dim:
            self._lazy_init_proj(dim, embeddings.device)

        p = embeddings @ self.proj
        x_t = p.unsqueeze(-1) * self.t
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        return (err @ self.weights * embeddings.size(0)).mean()
