"""
jepa_engine/injection.py —— Soft Prompt 轨迹注入层 (v7.0)

v7.0 架构重设计（替换 v6.0 KV-Cache 注入）：
- JEPA rollout 完整推理轨迹 → 投影为虚拟 token embedding
- 海马体记忆 embedding → 投影为 1 个虚拟 token
- 虚拟 token 拼在 LLM 输入序列头部，因果掩码天然允许后续 token attend
- per-token 学习门控（sigmoid router）控制每个虚拟 token 的激活强度
- gamma 整体门控保持（softplus 确保非负）

为什么 KV-Cache 注入失败：
  虚拟 KV tokens 拼在序列末尾 → 被 SDPA 因果掩码完全屏蔽 → CE 恒定不变。
  Soft Prompt 在序列头部 → 因果掩码的下三角特性天然允许所有后续 token attend。

为什么输出完整轨迹而非仅终态：
  终态压缩丢失推理过程。轨迹的每一步都携带 JEPA 的思维链信息，
  LLM 可以通过 attention 选择性参考不同推理阶段的状态。
  per-token gate 自动学习哪些步骤对当前问题有价值。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mvp_config import config


class TrajectoryInjection(nn.Module):
    """
    Soft Prompt 轨迹注入层。

    将 JEPA rollout 的完整推理轨迹 + 海马体记忆投影为虚拟 token embedding，
    拼在 LLM 输入序列头部作为 Soft Prompt。

    数据流：
        trajectory (B, T_traj, D_jepa)  → trajectory_proj → (B, T_traj, D_llm)
        mem_emb    (B, D_llm)           → memory_proj     → (B, 1, D_llm)
        拼接 → (B, T_traj+1, D_llm)   → gate * gamma     → gated_embeds

    门控机制（两级阀门）：
        gamma: 全局注入强度（softplus，学习标量）
        gate:  per-token 激活控制（sigmoid，学习 router）
        实际输出 = gamma * gate_i * embed_i
        → gamma=0 时全局静音，gate_i≈0 时单个 token 静音
    """
    def __init__(self):
        super().__init__()

        d_jepa = config.jepa_core_dim     # 1536
        d_llm = config.llm_hidden_size    # 896

        # 轨迹步投影：将 JEPA 潜空间映射到 LLM embedding 空间
        # 每个 rollout 步的 JEPA 状态 → 1 个虚拟 token embedding
        self.trajectory_proj = nn.Sequential(
            nn.Linear(d_jepa, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm),
            nn.LayerNorm(d_llm),
        )

        # 记忆投影：将海马体检索到的记忆 embedding 变换为虚拟 token
        # 独立于轨迹投影，因为记忆 embedding 已在 LLM 空间，
        # 但需要学习"如何将记忆以对 LLM 有用的方式呈现"
        self.memory_proj = nn.Sequential(
            nn.Linear(d_llm, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm),
            nn.LayerNorm(d_llm),
        )

        # Per-token 门控 router：根据每个虚拟 token 的内容决定激活强度
        # 简单决策 → 大部分 gate ≈ 0（等效于少量虚拟 token）
        # 复杂决策 → 多个 gate > 0（等效于多条推理线索）
        self.gate_head = nn.Sequential(
            nn.Linear(d_llm, d_llm // 4),
            nn.GELU(),
            nn.Linear(d_llm // 4, 1),
        )
        # Gate bias 初始化为负值，使训练初期大部分 gate ≈ 0（sigmoid(-1) ≈ 0.27）
        # 随训练进展，gate 逐渐打开有价值的 token
        nn.init.constant_(self.gate_head[-1].bias, -1.0)

        # Gamma 全局门控（softplus 确保非负）
        # 初始 0.0 → softplus ≈ 0.69，让 CE 能尽早感受到注入信号
        self._raw_gamma = nn.Parameter(torch.tensor([0.0]))

    @property
    def gamma(self):
        """实际门控值（始终非负）"""
        return F.softplus(self._raw_gamma)

    def compute_virtual_embeds(self, trajectory: torch.Tensor,
                               mem_emb: torch.Tensor):
        """
        将 JEPA 推理轨迹 + 海马体记忆投影为虚拟 token embeddings。

        Args:
            trajectory: (B, T_traj, D_jepa) — JEPA rollout 的完整轨迹
                        T_traj = rollout_depth + 1 (含初始状态)
            mem_emb: (B, D_llm) — 海马体检索到的记忆 embedding

        Returns:
            gated_embeds: (B, T_traj+1, D_llm) — gamma * gate 双重门控后的虚拟 embedding
            gate_scores:  (B, T_traj+1) — 每个虚拟 token 的门控分数（调试/日志用）
        """
        # 1. 轨迹步投影: (B, T_traj, D_jepa) → (B, T_traj, D_llm)
        traj_embeds = self.trajectory_proj(trajectory.to(torch.float32))

        # 2. 记忆投影: (B, D_llm) → (B, 1, D_llm)
        mem_embed = self.memory_proj(mem_emb.to(torch.float32)).unsqueeze(1)

        # 3. 拼接轨迹 + 记忆: (B, T_traj+1, D_llm)
        all_embeds = torch.cat([traj_embeds, mem_embed], dim=1)

        # 4. Per-token 门控: sigmoid(gate_head(each_token)) → (B, T_traj+1)
        gate_scores = torch.sigmoid(
            self.gate_head(all_embeds).squeeze(-1))  # (B, T_traj+1)

        # 5. Gamma + gate 双重门控
        # gamma 控制整体注入强度，gate 控制每个 token 的激活程度
        gated_embeds = self.gamma * gate_scores.unsqueeze(-1) * all_embeds

        return gated_embeds, gate_scores
"""
Complexity: 7
Description: Completely rewrote injection layer from KV-Cache monkey-patch to Soft Prompt trajectory injection. Key design: JEPA rollout trajectory steps each become a virtual token, with per-token learned gating and gamma global gating. The gate bias is initialized negative so most tokens start near-silent and learn to activate during training.
"""
