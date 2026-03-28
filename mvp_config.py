import os
from dataclasses import dataclass
import torch

@dataclass
class BrainConfig:
    """
    统一配置中心 v5.6 —— JEPA 核心 + SQLite 海马体 + 多候选评分。
    所有维度/参数量通过此文件统一管理，升级 LLM 只需改 llm_hidden_size。

    v5.6 审计全修复：
    - P0: SIGReg batch 化 / 意图进化投影漂移 / 微梦 Adam 隔离
    - P1: softmax pooling / query_head 对齐 loss / 时间临近边防爆
    - P2: hook context manager / sha256 key / 死配置清理
    """
    # =============================================================
    # 1. 语言中枢 (Broca's Area) —— 冻结的语法引擎
    #    升级 LLM 只需修改下面两行，所有模块自动适配
    # =============================================================
    llm_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_hidden_size: int = 896

    # =============================================================
    # 2. 潜意识引擎 (JEPA) —— 核心大脑（真正的世界模型）
    #
    #    【扩展指南】基于 Transformer Scaling Laws (Kaplan 2020) + DiT：
    #    - 只需调 jepa_core_dim，其余按比例自动计算:
    #      · inner_dim = heads × dim_head ≈ 0.5 × core_dim
    #      · mlp_dim ≈ 1.3 × core_dim
    #      · intent_dim ≈ core_dim / 12
    #      · depth: 按 depth ∝ sqrt(core_dim/128) 粗略估算
    #    - 当前配置 core_dim=1536 → 总参约 32M（不含 LLM）
    #    - 扩展到 500B: core_dim≈16384, depth≈36, heads≈128
    # =============================================================
    jepa_core_dim: int = 1536
    jepa_steer_dim: int = 256

    # Predictor Transformer（v5.3: 重新分配参数比例）
    jepa_predictor_depth: int = 3       # Transformer 层数
    jepa_predictor_heads: int = 8       # 注意力头数（v5.3: 4→8，更丰富的多角度推理）
    jepa_predictor_dim_head: int = 96   # 每头维度（v5.3: 64→96，更精细的特征交互）
    jepa_predictor_mlp_dim: int = 2048  # FFN 隐藏层（v5.3: 1024→2048，更强的非线性变换）

    # 策略意图空间（方案 C: 可学习意图 + 记忆驱动意图）
    jepa_intent_dim: int = 128          # 意图维度（v5.3: 64→128，更丰富的策略表达）
    jepa_num_intents: int = 6           # 可学习策略意图数量（S 条预演路径）
    jepa_max_memory_intents: int = 4    # 从海马体提取的最大记忆驱动意图数
    jepa_memory_intent_weight: float = 0.8  # 记忆意图投影的缩放因子
    evolve_intent_ema_alpha: float = 0.1    # 意图进化 EMA 步长（P0-3: 消除硬编码）

    # Rollout 多步推理
    jepa_rollout_depth: int = 3         # 向前推演步数
    jepa_rollout_history: int = 3       # 推演时使用的历史窗口

    # 自监督训练
    jepa_sigreg_knots: int = 17
    jepa_sigreg_num_proj: int = 512
    jepa_pred_loss_weight: float = 1.0
    jepa_sigreg_weight: float = 0.1
    jepa_intent_diversity_weight: float = 0.05  # 意图多样性正则化权重
    jepa_softmin_temperature: float = 0.1       # soft-min 温度

    # v20 InfoNCE 对比损失（断裂点 #3 修复：锐化训练信号）
    jepa_contrastive_weight: float = 0.3         # InfoNCE 在联合 loss 中的权重
    # v20-audit T6 修复: 0.07 对高维归一化向量过于激进（14.3x 放大）
    # 初始 cosine ≈ 0 时 softmax 完全均匀化, InfoNCE 信号丢失
    # 0.1 是对比学习的标准值 (SimCLR/CLIP 均使用 0.07-0.1 范围, 但需匹配维度)
    jepa_contrastive_temperature: float = 0.1    # InfoNCE 温度

    # EMA Target Encoder（v5.3: BYOL/JEPA 标准范式）
    ema_momentum: float = 0.996         # EMA 动量（每步只更新 0.4% 参数）

    # JEPA Encoder Transformer（v8.0 A+B 混合路线：替代 Linear+LN）
    jepa_encoder_depth: int = 4          # Encoder Transformer 层数
    jepa_encoder_heads: int = 8          # 注意力头数
    jepa_encoder_dim_head: int = 96      # 每头维度（inner_dim = heads * dim_head = 768）
    jepa_encoder_mlp_dim: int = 2048     # FFN 隐层维度
    jepa_encoder_dropout: float = 0.1    # Dropout 比率
    jepa_encoder_max_seq_len: int = 32   # 位置编码最大序列长度

    # ARC Grid 编码器/解码器（多模态 JEPA 世界模型）
    grid_max_size: int = 30              # ARC 网格最大尺寸（30×30）
    grid_num_colors: int = 10            # ARC 颜色数（0-9）
    grid_encoder_depth: int = 2          # Grid Encoder Transformer 层数
    grid_decoder_depth: int = 2          # Grid Decoder Transformer 层数
    grid_encoder_mlp_dim: int = 512     # FFN 隐层（ARC 独立，与 NLP 解耦）
    grid_patch_size: int = 3             # Grid patch 大小（3×3 cells → 1 token，加速 ~9x）
    # v20-audit M1: 修复重复定义（原来 L83-84 和 L87-88 重复，Python 后定义覆盖前定义）
    grid_encoder_depth_heads: int = 4    # 注意力头数（ARC 独立）
    grid_encoder_dim_head: int = 64      # 每头维度

    # ARC 独立维度空间（与 NLP 的 jepa_core_dim=1536 完全解耦）
    # ARC 任务复杂度远低于自然语言，256 维足够且大幅降低参数量和训练时间
    arc_dim: int = 256                   # ARC 潜空间维度（NLP 用 1536）
    arc_train_epochs: int = 200          # ARC 训练轮次（升级：100→200）
    arc_lr: float = 1e-4                 # ARC 训练学习率
    arc_context_max_pairs: int = 5       # ARC 单题最大示例对数

    # ARC 独立 Predictor（6层，不影响 NLP 路径的 3 层）
    arc_predictor_depth: int = 6         # ARC Predictor Transformer 层数
    arc_predictor_heads: int = 4         # ARC Predictor 注意力头数
    arc_predictor_dim_head: int = 64     # ARC Predictor 每头维度
    arc_predictor_mlp_dim: int = 512     # ARC Predictor FFN 隐层

    # ARC 数据增强配置
    arc_augment_rotations: bool = True   # 启用旋转增强（90°/180°/270°）
    arc_augment_flips: bool = True       # 启用翻转增强（水平/垂直）
    arc_augment_color_perms: int = 2     # 颜色置换随机排列数量

    # JEPA Stochastic World Model（v8.0 A路线：随机状态 + Latent Overshooting）
    jepa_z_dim: int = 128                # 随机状态维度
    jepa_kl_weight: float = 0.1          # KL(posterior||prior) 损失权重
    jepa_kl_free_bits: float = 1.0       # Free bits 阈值（低于此值不惩罚 KL，避免过度压制 posterior）

    # 双模学习
    online_lr: float = 5e-4
    replay_buffer_size: int = 64
    micro_dream_steps: int = 3          # 实时微梦训练步数
    dream_lr: float = 2e-3              # 深度做梦阶段的学习率

    # 深度做梦
    deep_dream_merge_threshold: float = 0.92  # 记忆合并的余弦相似度阈值
    deep_dream_pattern_min_count: int = 3     # 至少出现 N 次才提精为模式

    # =============================================================
    # 3. 海马体 (SQLite + NetworkX + FAISS 三引擎)
    # =============================================================
    memory_top_k: int = 3
    enable_multi_hop: bool = True
    memory_auto_extract: bool = True
    memory_db_path: str = "hippocampus.db"      # SQLite 路径

    # KGE (TransE)
    kge_embed_dim: int = 128
    kge_margin: float = 1.0
    kge_lr: float = 1e-3
    kge_train_epochs: int = 20
    kge_batch_size: int = 16            # TransE mini-batch 大小
    kge_score_weight: float = 0.3       # find_seeds 中 TransE 评分的融合权重（v5.3: 0.2→0.3, 投影层使其真正有效）
    kge_proj_epochs: int = 30           # LLM→KGE 投影层训练轮数（v5.3 新增）

    # v5.5: 联合训练中 KGE/投影层的 loss 权重（消灭魔法数字）
    kge_loss_weight: float = 0.1        # TransE margin ranking loss 的权重
    proj_loss_weight: float = 0.01      # LLM→KGE 投影层 MSE loss 的权重
    query_alignment_loss_weight: float = 0.05  # v5.6 P1-7: query_head 对齐损失权重

    # Personalized PageRank
    ppr_damping: float = 0.85
    ppr_max_iter: int = 50
    ppr_top_k: int = 5

    # 多层记忆边构建 (进化 2)
    semantic_sim_threshold: float = 0.75  # 语义相似边的余弦相似度阈值
    temporal_window_sec: float = 3600.0   # 时间临近边的窗口 (秒)
    temporal_min_gap_sec: float = 0.5     # v5.6 P1-3: 最小时间间隔，低于此值不建时间临近边

    # v5.5: 结局节点拓扑判定参数（消灭硬编码阈值）
    outcome_min_in_degree: int = 2      # 入度 >= 此值才视为终端特征
    outcome_max_out_degree: int = 1     # 出度 <= 此值才视为终端特征

    # =============================================================
    # 4. 注入层 (v7.0 Soft Prompt 轨迹注入)
    # =============================================================
    # v7.0: 不再需要 injection_target_layer / num_kv_heads / head_dim / num_virtual_tokens
    # Soft Prompt 在 LLM 输入序列头部拼接虚拟 token，不修改 LLM 内部层
    generation_max_tokens: int = 80     # 交互时 LLM 生成的最大 token 数
    abstraction_max_tokens: int = 30    # 做梦期间 Cortex 归纳摘要的最大 token 数
    pooling_temperature: float = 0.5    # v5.6 P1-6: Attention-weighted pooling 的 softmax 温度

    # v5.5: 摘要 Prompt 模板（做梦期间 Cortex 归纳用，支持按语言皮层类型切换）
    # 可用占位符: {text1}, {text2}
    summary_prompt_template: str = (
        "请用一句话概括以下两段经历的共同主题：\n"
        "1. {text1}\n2. {text2}\n概括："
    )

    # =============================================================
    # 5. 运行环境
    # =============================================================
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True
    emb_cache_max_size: int = 2048          # 全局 Embedding 缓存上限（P1-1: 防 OOM）

config = BrainConfig()

