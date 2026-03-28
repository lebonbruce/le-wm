"""
jepa_engine/meta_decoder.py — 元语解码器 v20

核心职责：
  将 JEPA rollout 轨迹的预演结果解码为 LLM 能理解的结构化文本"元语"。
  
  这是断裂点 #2 的修复：替代 Soft Prompt embedding 级注入。
  
  为什么文本级注入不会中毒（而 Soft Prompt 会）：
  - LLM 的注意力流形是基于自然语言 token 建立的
  - 文本 token 是 LLM 的"母语"，Soft Prompt 向量是"外星噪声"
  - 元语文本本质上是一种高质量的结构化 prompt engineering
  
设计选择：最简规则映射方案（无可训练参数）
  - 意图索引 → 可读标签（从 intent_bank 的语义距离推导）
  - 轨迹终态 → 查询海马体 → 检索最近的记忆文本 → "关键记忆"
  - 轨迹与 goal 的距离 → 量化为"置信度"
  
  为什么不用神经网络解码器：
  - 额外网络引入新的训练不稳定性（在架构刚修复时风险太高）
  - 规则映射是确定性的、可调试的、零参数的
  - 当系统稳定后可升级为可训练的解码器（Phase 2）
"""

import torch
import torch.nn.functional as F
from mvp_config import config


# 意图标签（人可读的语义方向描述）
# 对应 intent_bank 的 6 个可学习意图位置
# 这些标签在初始阶段是启发式命名，随训练进化会逐步校准
INTENT_LABELS = [
    "情绪安抚",      # intent 0
    "行动建议",      # intent 1
    "原因分析",      # intent 2
    "经验回忆",      # intent 3
    "认知重构",      # intent 4
    "共情理解",      # intent 5
]


class MetaLanguageDecoder:
    """
    元语解码器：JEPA 轨迹 → 结构化文本。
    
    不继承 nn.Module（无可训练参数），纯规则映射。
    """
    
    def __init__(self):
        self.intent_labels = INTENT_LABELS
    
    def decode_trajectory(self,
                          trajectory: torch.Tensor,
                          best_intent_idx: int,
                          goal_emb: torch.Tensor,
                          memories: list,
                          gate_confidence: float = None) -> str:
        """
        将 JEPA 预演轨迹解码为结构化文本元语。
        
        trajectory: (1, T, D_jepa) JEPA rollout 的完整轨迹
        best_intent_idx: 最优意图的索引（来自 compute_prediction_loss 或 plan_with_prior）
        goal_emb: (D_jepa,) 或 None — 海马体目标节点的编码
        memories: list[dict] — 海马体检索的记忆列表
        gate_confidence: float 或 None — 旧 injection_layer 的 gate 分数（兼容性参数，不再使用）
        
        返回: str — 结构化元语文本
        """
        parts = []
        
        # 1. 方向意图标签
        intent_label = self._get_intent_label(best_intent_idx)
        parts.append(f"[方向] {intent_label}")
        
        # 2. 关键记忆（从海马体检索结果中提取最相关的 1-2 条）
        memory_texts = self._extract_memory_context(memories)
        if memory_texts:
            parts.append(f"[关键记忆] {memory_texts}")
        
        # 3. 目标置信度（轨迹终态与 goal 的距离）
        if goal_emb is not None and trajectory is not None:
            confidence = self._compute_confidence(trajectory, goal_emb)
            parts.append(f"[置信度] {confidence:.0f}%")
        
        # 4. 轨迹稳定性（轨迹各步之间的变化幅度）
        if trajectory is not None and trajectory.size(1) > 1:
            stability = self._assess_stability(trajectory)
            parts.append(f"[推理稳定性] {stability}")
        
        # 组合为结构化上下文
        meta_text = " | ".join(parts)
        return f"[JEPA预演] {meta_text}"
    
    def _get_intent_label(self, intent_idx: int) -> str:
        """将意图索引映射为人可读标签"""
        if 0 <= intent_idx < len(self.intent_labels):
            return self.intent_labels[intent_idx]
        # 超出预定义标签范围的（memory-driven intents）
        return f"记忆驱动策略#{intent_idx - len(self.intent_labels) + 1}"
    
    def _extract_memory_context(self, memories: list) -> str:
        """从海马体记忆中提取最相关的文本片段"""
        if not memories:
            return ""
        
        # 取最相关的 1-2 条记忆的文本摘要
        texts = []
        for mem in memories[:2]:
            text = mem.get("text", "")
            if text:
                # 截取前 60 字符作为摘要
                summary = text[:60].strip()
                if len(text) > 60:
                    summary += "..."
                texts.append(summary)
        
        return " → ".join(texts)
    
    @torch.no_grad()
    def _compute_confidence(self, trajectory: torch.Tensor,
                            goal_emb: torch.Tensor) -> float:
        """
        计算轨迹终态与目标的接近程度（0-100%）。
        
        使用余弦相似度：1.0 = 完全对齐，0.0 = 正交，-1.0 = 完全对立
        映射到 [0, 100] 置信度范围
        """
        final_state = trajectory[0, -1, :]  # (D_jepa,)
        
        if goal_emb.dim() == 1:
            goal = goal_emb
        else:
            goal = goal_emb.squeeze()
        
        # 如果维度不匹配，无法计算
        if final_state.shape != goal.shape:
            return 50.0  # 默认中等置信度
        
        cos_sim = F.cosine_similarity(
            final_state.unsqueeze(0), goal.unsqueeze(0)).item()
        
        # [-1, 1] → [0, 100]
        confidence = (cos_sim + 1.0) / 2.0 * 100.0
        return max(0.0, min(100.0, confidence))
    
    @torch.no_grad()
    def _assess_stability(self, trajectory: torch.Tensor) -> str:
        """
        评估推理轨迹的稳定性。
        
        通过相邻步之间的余弦距离来判断推理过程是否收敛/发散/振荡。
        """
        T = trajectory.size(1)
        if T < 2:
            return "单步"
        
        # 计算相邻步之间的余弦距离
        step_dists = []
        for t in range(T - 1):
            s1 = trajectory[0, t, :]
            s2 = trajectory[0, t + 1, :]
            dist = 1.0 - F.cosine_similarity(
                s1.unsqueeze(0), s2.unsqueeze(0)).item()
            step_dists.append(dist)
        
        avg_dist = sum(step_dists) / len(step_dists)
        
        # 判断趋势
        if len(step_dists) >= 2:
            first_half = sum(step_dists[:len(step_dists)//2]) / max(len(step_dists)//2, 1)
            second_half = sum(step_dists[len(step_dists)//2:]) / max(len(step_dists) - len(step_dists)//2, 1)
            
            if second_half < first_half * 0.5:
                return "收敛"
            elif second_half > first_half * 1.5:
                return "发散"
        
        if avg_dist < 0.05:
            return "稳定"
        elif avg_dist < 0.2:
            return "适度探索"
        else:
            return "剧烈探索"
