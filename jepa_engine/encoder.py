"""
jepa_engine/encoder.py —— JEPA 多层 Transformer Encoder（A+B 混合路线 · 阶段 1）

替代原有的 Linear+LayerNorm 简单投影，引入 4 层 Transformer 块，
使 Encoder 能够学到比线性变换更丰富的层次化潜空间表示。

设计决策：
1. 仍接收冻结 LLM 的隐层特征（而非从 token 开始），复用 LLM 的上下文理解能力
2. 使用标准 Pre-Norm Transformer 块（LayerNorm → Attention → Residual → LN → FFN → Residual）
3. 支持单向量输入 (B, D_llm) 和序列输入 (B, T, D_llm) 两种模式
4. EMA Target Encoder 可直接 deepcopy 此 Encoder

核心区别 vs 旧 encoder：
- 旧：Linear(896→1536) + LN → 仿射变换，表示质量完全依赖 LLM
- 新：Proj(896→1536) + 4层Transformer → 非线性层次化特征提取，潜空间有独立学习能力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mvp_config import config


class EncoderTransformerBlock(nn.Module):
    """
    Pre-Norm Transformer 块（标准 ViT/BERT 范式）。

    Pre-Norm 比 Post-Norm 训练更稳定（不需要 warmup），
    适合中等深度网络（4-12 层）。

    data flow:
        x → LN → MHSA → + residual → LN → FFN → + residual
    """
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        # 自注意力（Pre-Norm）
        self.norm1 = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        # FFN（Pre-Norm）
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) 输入序列
        返回: (B, T, D) 输出序列
        """
        # Pre-Norm 自注意力
        h = self.norm1(x)
        B, T, _ = h.shape
        qkv = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.heads, self.dim_head).transpose(1, 2)
                    for t in qkv]
        # 双向注意力（Encoder 不需要因果掩码）
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.to_out(attn_out)

        # Pre-Norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class JEPAEncoder(nn.Module):
    """
    JEPA 多层 Transformer Encoder。

    将 LLM 隐层特征（896 维）通过投影 + 多层 Transformer 变换到
    JEPA 潜空间（1536 维），学习到层次化的抽象表示。

    与旧 encoder 的关键区别：
    - 旧：Linear → 只能做仿射变换，潜空间完全映射自 LLM 空间
    - 新：4 层 Transformer → 可以学习非线性特征组合和多头注意力交互，
      潜空间有独立于 LLM 的学习自由度

    输入模式：
    - 单向量: (B, D_llm) → 自动加时间维 → (B, 1, D_jepa) → squeeze → (B, D_jepa)
    - 序列:   (B, T, D_llm) → (B, T, D_jepa)
    """
    def __init__(self):
        super().__init__()
        d_llm = config.llm_hidden_size       # 896
        d_jepa = config.jepa_core_dim         # 1536

        # 线性投影到 JEPA 维度（保底：即使 Transformer 退化，至少有投影能力）
        self.input_proj = nn.Linear(d_llm, d_jepa)

        # 位置编码（可学习，最大支持 32 个 token 的序列）
        # 对于单向量输入 T=1，pos_embed 只取第一个位置
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.jepa_encoder_max_seq_len, d_jepa) * 0.02
        )

        # 多层 Transformer 块
        self.layers = nn.ModuleList([
            EncoderTransformerBlock(
                dim=d_jepa,
                heads=config.jepa_encoder_heads,
                dim_head=config.jepa_encoder_dim_head,
                mlp_dim=config.jepa_encoder_mlp_dim,
                dropout=config.jepa_encoder_dropout,
            )
            for _ in range(config.jepa_encoder_depth)
        ])

        # 最终归一化（Pre-Norm arch 需要额外的 final LN）
        self.norm = nn.LayerNorm(d_jepa)

    def forward(self, llm_features: torch.Tensor) -> torch.Tensor:
        """
        将 LLM 特征编码到 JEPA 潜空间。

        llm_features: (B, D_llm) 或 (B, T, D_llm)
        返回:
          - 如果输入 (B, D_llm) → 返回 (B, D_jepa)
          - 如果输入 (B, T, D_llm) → 返回 (B, T, D_jepa)
        """
        squeezed = False
        if llm_features.dim() == 2:
            llm_features = llm_features.unsqueeze(1)  # (B, 1, D_llm)
            squeezed = True

        B, T, _ = llm_features.shape

        # 投影到 JEPA 维度
        x = self.input_proj(llm_features)  # (B, T, D_jepa)

        # 加位置编码
        x = x + self.pos_embed[:, :T, :]

        # 通过 Transformer 层
        for layer in self.layers:
            x = layer(x)

        # 最终归一化
        x = self.norm(x)  # (B, T, D_jepa)

        if squeezed:
            x = x.squeeze(1)  # (B, D_jepa)

        return x
