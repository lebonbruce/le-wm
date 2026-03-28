"""
jepa_engine/predictor.py —— AdaLN-zero Transformer Predictor（JEPA 核心世界模型）

核心机制：条件信号（策略意图嵌入）通过 AdaLN-zero 调制主流特征，
初始时 gate=0 使模块输出为零，训练中逐渐打开影响力。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mvp_config import config


class AdaLNBlock(nn.Module):
    """
    AdaLN-zero Transformer 块 -- 条件化注意力和 FFN。

    核心机制：条件信号（策略意图嵌入）通过 AdaLN-zero 调制主流特征，
    初始时 gate=0 使模块输出为零，训练中逐渐打开影响力。
    """
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        # 自注意力
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        # FFN
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

        # AdaLN-zero 调制：从条件信号生成 6 个调制参数
        # (shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        # Zero-init：初始时 gate=0，模块输出为零，不干扰主流
        nn.init.constant_(self.adaLN[-1].weight, 0)
        nn.init.constant_(self.adaLN[-1].bias, 0)

    def forward(self, x, cond):
        """
        x: (B, T, D) 主流特征
        cond: (B, T, D) 条件信号（策略意图嵌入）
        """
        # 生成 6 个调制参数
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = \
            self.adaLN(cond).chunk(6, dim=-1)

        # 调制后的因果自注意力
        h = self.norm1(x) * (1 + scale_attn) + shift_attn
        B, T, _ = h.size()
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.heads, -1).transpose(1, 2) for t in qkv]
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + gate_attn * self.to_out(attn)

        # 调制后的 FFN
        h = self.norm2(x) * (1 + scale_ffn) + shift_ffn
        x = x + gate_ffn * self.ffn(h)
        return x


class CognitivePredictor(nn.Module):
    """
    认知版 ARPredictor -- 使用 AdaLN-zero 条件化机制，
    条件信号 = 策略意图嵌入。
    """
    def __init__(self):
        super().__init__()
        D = config.jepa_core_dim
        # 位置编码长度需同时支持 NLP rollout（3+3=6）和 ARC 上下文（2×5+1=11）
        nlp_frames = config.jepa_rollout_history + config.jepa_rollout_depth
        arc_frames = 2 * config.arc_context_max_pairs + 1
        max_frames = max(nlp_frames, arc_frames)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_frames, D) * 0.02)
        self.dropout = nn.Dropout(0.1)

        # AdaLN-zero Transformer 层
        self.layers = nn.ModuleList([
            AdaLNBlock(
                dim=D,
                heads=config.jepa_predictor_heads,
                dim_head=config.jepa_predictor_dim_head,
                mlp_dim=config.jepa_predictor_mlp_dim
            )
            for _ in range(config.jepa_predictor_depth)
        ])
        self.norm = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) 观测序列嵌入
        cond: (B, T, D) 条件信号（策略意图嵌入，沿时间步广播）
        返回: (B, T, D) 预测的未来状态嵌入
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        for block in self.layers:
            x = block(x, cond)
        return self.norm(x)
