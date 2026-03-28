"""
jepa_engine/subconscious.py —— JEPA 潜意识引擎（A+B 混合世界模型）

v8.0 A+B 混合重构：
- Encoder 从 Linear+LN 升级为 4 层 Transformer（JEPAEncoder）
- Encoder 学习非线性层次化特征提取，潜空间有独立学习自由度
- EMA Target Encoder 从深拷贝 JEPAEncoder 获得，动量演进

v5.7 梯度流修复：
- memory_intent_proj 新增两阶段 API（extract_memory_cluster_centers + project_cluster_centers），
  使训练循环中投影层能正常接收梯度

v5.6 审计全修复：
- P0-2: SIGReg B=1 失效 → 新增 compute_sigreg_on_batch 方法，由 brain.py 积累 batch 后调用
- P0-3: replay_buffer 存储 intent 空间向量，消除投影漂移
- P1-7: 新增 query_head 对齐 loss
- P2-2: k-means 向量化（scatter_add_ 替代 for 循环）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from mvp_config import config
from jepa_engine.sigreg import SIGReg
from jepa_engine.predictor import AdaLNBlock, CognitivePredictor
from jepa_engine.encoder import JEPAEncoder


class ExperienceReplayBuffer:
    """
    经验回放缓冲区。
    v5.6 P0-3: 同时存储 intent 空间的投影向量，消除 evolve_intents 的投影漂移。
    """
    def __init__(self, capacity=None):
        self.buffer = deque(maxlen=capacity or config.replay_buffer_size)

    def push(self, inp: torch.Tensor, tgt: torch.Tensor,
             intent_emb: torch.Tensor = None):
        """
        存储一条经验。
        v5.6 P0-3: intent_emb 为已投影到 intent 空间的目标向量，
        避免 evolve_intents 时重新投影造成漂移。
        """
        entry = {
            'inp': inp.detach().clone(),
            'tgt': tgt.detach().clone(),
            'intent_emb': intent_emb.detach().clone() if intent_emb is not None else None,
        }
        self.buffer.append(entry)

    def sample(self, k: int) -> list:
        return random.sample(list(self.buffer), min(k, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


def _kmeans_cluster(embeddings: torch.Tensor, k: int, max_iter: int = 10) -> torch.Tensor:
    """
    简单 k-means 聚类（纯 torch 实现，无外部依赖）。

    用于将记忆 embeddings 聚类为 k 个语义簇心，
    代替按检索类型（种子/PPR）的硬分组。

    embeddings: (N, D) 记忆 embedding 集合
    k: 目标簇数
    返回: (k, D) 簇心向量
    """
    N, D = embeddings.shape
    if N <= k:
        return embeddings  # 记忆数 <= k，每条记忆本身就是一个意图

    # 随机初始化簇心（从 embeddings 中采样）
    indices = torch.randperm(N, device=embeddings.device)[:k]
    centers = embeddings[indices].clone()

    for _ in range(max_iter):
        # 分配：计算每个点到每个簇心的距离 → 选最近簇
        dists = torch.cdist(embeddings, centers)  # (N, k)
        assignments = dists.argmin(dim=1)  # (N,)

        # v5.6 P2-2: 用 scatter_add_ 向量化替代 for 循环
        # 计算每个簇的成员数量
        counts = torch.zeros(k, device=embeddings.device)
        counts.scatter_add_(0, assignments, torch.ones(N, device=embeddings.device))
        # 计算每个簇的成员嵌入总和
        new_centers = torch.zeros_like(centers)  # (k, D)
        new_centers.scatter_add_(0, assignments.unsqueeze(1).expand(-1, D), embeddings)
        # 归一化（空簇保留原心）
        valid = counts > 0
        new_centers[valid] = new_centers[valid] / counts[valid].unsqueeze(1)
        new_centers[~valid] = centers[~valid]

        if torch.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers

    return centers


class SubconsciousJEPA(nn.Module):
    """
    JEPA 潜意识引擎 v8.0 -- A+B 混合世界模型（深度 Encoder + EMA + 混合意图）。

    核心机制：
    1. encoder: 4 层 Transformer 在线编码器（LLM -> JEPA 潜空间），接收梯度更新
    2. ema_encoder: EMA 目标编码器（encoder 的动量平滑副本），无梯度
    3. intent_bank: 可学习的策略意图库
    4. memory_intent_proj: 记忆驱动意图投影
    5. intent_encoder: 意图 -> Predictor 条件信号
    6. predictor: AdaLN-zero Transformer
    7. rollout: 批量并行化自回归多步推演
    8. query_head: 最优路径 -> 检索查询信号

    v8.0 A+B 混合路线升级：
    - Encoder 从 Linear+LN 升级为 4 层 Transformer（JEPAEncoder）
    - 潜空间表示质量不再完全依赖冻结 LLM，有独立学习能力
    - EMA Target Encoder 保护深层 Encoder 的训练稳定性
    """
    def __init__(self):
        super().__init__()
        import copy

        # ---- 1. Online Encoder: 4 层 Transformer（v8.0 升级） ----
        # 接收 LLM 隐层特征 (B, D_llm) → 输出 JEPA 潜空间 (B, D_jepa)
        # 比旧的 Linear+LN 有更强的非线性表示学习能力
        self.encoder = JEPAEncoder()

        # ---- 1b. EMA Target Encoder (v8.0: 深度 Encoder 的 EMA 副本) ----
        # 深层 Encoder 的 EMA 副本，动量演进提供稳定训练目标
        # 因为 Encoder 现在有 4 层 Transformer，EMA 的稳定化效果显著增强
        self.ema_encoder = copy.deepcopy(self.encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False

        # ---- 2. 可学习策略意图库（S 个方向，训练中自动分化） ----
        self.intent_bank = nn.Parameter(
            torch.randn(config.jepa_num_intents, config.jepa_intent_dim) * 0.1
        )

        # ---- 3. 记忆驱动意图投影（方案 C 新增） ----
        #    将 LLM 空间的记忆 embedding 投影到意图空间
        self.memory_intent_proj = nn.Sequential(
            nn.Linear(config.llm_hidden_size, config.jepa_intent_dim),
            nn.LayerNorm(config.jepa_intent_dim)
        )

        # ---- 4. Intent Encoder: 意图 -> 条件信号（对齐 Predictor 维度） ----
        self.intent_encoder = nn.Sequential(
            nn.Linear(config.jepa_intent_dim, config.jepa_core_dim),
            nn.GELU(),
            nn.Linear(config.jepa_core_dim, config.jepa_core_dim)
        )

        # ---- 5. Predictor: AdaLN-zero Transformer（核心世界模型） ----
        self.predictor = CognitivePredictor()

        # ---- 6. 输出头 ----
        # v20-audit M3: steer_head 已移除（v20 不再有 KV-Cache 注入路径）
        self.query_head = nn.Linear(config.jepa_core_dim, config.llm_hidden_size)

        # ---- 7. SIGReg 防坍塌 ----
        self.sigreg = SIGReg()

        # ---- 8. 经验回放 ----
        self.replay_buffer = ExperienceReplayBuffer()

        # ---- 9. SSL Predictor（v8.0 B路线：自监督预训练专用） ----
        D = config.jepa_core_dim
        self.ssl_predictor = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.LayerNorm(D),
            nn.Linear(D, D),
        )

        # ---- 10. Stochastic World Model（v8.0 A路线） ----
        # 随机状态分支：处理多模态未来（同一问题可能有多种合理回复）
        z_dim = config.jepa_z_dim  # 128

        # Posterior: 训练时使用，看到 context + target → 更准确的分布
        # 输入 = cat(h_context, h_target)，输出 = (mu, logvar)
        self.posterior_net = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.GELU(),
            nn.Linear(D, z_dim * 2),  # 输出 mu 和 logvar
        )

        # Prior: 推理时使用，只看到 context → 需要想象的分布
        # 输入 = h_context，输出 = (mu, logvar)
        self.prior_net = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, z_dim * 2),
        )

        # State Combiner: 将确定性状态 h + 随机状态 z 合并回 D_jepa 维
        # 使得下游的 predictor/injection 无需修改
        self.state_combiner = nn.Sequential(
            nn.Linear(D + z_dim, D),
            nn.LayerNorm(D),
        )

        # Step Embeddings: Latent Overshooting 用的步数编码
        # 每个 rollout 步有独立的步数信号，打破 autoregressive 的串行依赖
        max_steps = config.jepa_rollout_depth
        self.step_embeddings = nn.Parameter(
            torch.randn(max_steps, D) * 0.02
        )

    # ---- EMA 更新 ----

    @torch.no_grad()
    def _update_ema(self):
        """
        v5.3: EMA 目标编码器参数更新（BYOL/JEPA 标准范式）。

        ema_params = momentum × ema_params + (1 - momentum) × online_params

        每个训练步后调用。momentum=0.996 意味着每步只更新 0.4% 的参数，
        使目标表示缓慢且平滑地演进，避免训练不稳定。
        """
        m = config.ema_momentum
        for ema_p, online_p in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            ema_p.data.mul_(m).add_(online_p.data, alpha=1.0 - m)

    def encode(self, llm_features: torch.Tensor) -> torch.Tensor:
        """
        在线编码：LLM 隐状态 -> JEPA 潜空间（带梯度）。

        v8.0：JEPAEncoder 内部自动处理维度（2D→unsqueeze→squeeze）。

        llm_features: (B, D_llm)
        返回: (B, 1, D_jepa) 添加时间维以兼容序列操作
        """
        encoded = self.encoder(llm_features)  # (B, D_jepa)，JEPAEncoder 处理 2D 输入
        return encoded.unsqueeze(1)  # (B, 1, D_jepa)

    @torch.no_grad()
    def encode_target(self, llm_features: torch.Tensor) -> torch.Tensor:
        """
        EMA 目标编码 —— 用 EMA encoder 编码目标嵌入。

        天然无梯度（@torch.no_grad + 参数 requires_grad=False），
        不需要手动 .detach()。

        注意: v20 后此方法仅用于 SSL 预训练阶段。
        世界模型训练阶段应使用 encode_fact_target() 编码海马体事实。

        llm_features: (B, D_llm)
        返回: (B, 1, D_jepa)
        """
        encoded = self.ema_encoder(llm_features)  # (B, D_jepa)
        return encoded.unsqueeze(1)  # (B, 1, D_jepa)

    @torch.no_grad()
    def encode_fact_target(self, fact_llm_features: torch.Tensor) -> torch.Tensor:
        """
        v20: 编码来自海马体的外部事实作为预测目标。

        与 encode_target() 的关键区别：
        - encode_target() 用 EMA encoder → 自蒸馏 → 垃圾进垃圾出
        - encode_fact_target() 用 online encoder (detached) → 外部事实 → 打破死循环

        为什么用 online encoder 而非 EMA encoder：
        - EMA encoder 在自蒸馏场景下是 JEPA 标准范式（稳定目标）
        - 但当训练信号本身就来自外部 ground truth 时，EMA 的"稳定化"反而引入滞后
        - online encoder 反映最新的表示空间，detach 后只截断梯度不截断表示质量

        fact_llm_features: (B, D_llm) 海马体中真实事件的 LLM embedding
        返回: (B, D_jepa) 作为 prediction loss 的目标（无梯度，常量）
        """
        encoded = self.encoder(fact_llm_features)  # (B, D_jepa)
        return encoded.detach()  # 截断梯度，作为常量目标

    # ---- v8.0 A路线：随机状态世界模型 ----

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化采样（VAE 标准技术）：z = mu + sigma * epsilon
        使得采样操作可微分，梯度能通过 mu 和 logvar 回传。
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def compute_stochastic_state(self, h_context: torch.Tensor,
                                  h_target: torch.Tensor = None,
                                  use_posterior: bool = True) -> tuple:
        """
        计算随机增强状态 = deterministic h + stochastic z。

        训练时（use_posterior=True）：
          posterior 看到 context + target → 更精确的 z 采样
        推理时（use_posterior=False）：
          prior 只看 context → 想象 z

        h_context: (B, D_jepa) 确定性编码
        h_target:  (B, D_jepa) 目标编码（仅 posterior 需要）
        use_posterior: 是否使用 posterior（训练时 True，推理时 False）

        返回: (state, mu_post, logvar_post, mu_prior, logvar_prior)
              state: (B, D_jepa) 随机增强状态
              其余为分布参数（用于 KL loss 计算）
        """
        z_dim = config.jepa_z_dim

        # Prior 分布（始终计算，训练和推理都需要）
        prior_out = self.prior_net(h_context)  # (B, z_dim*2)
        mu_prior, logvar_prior = prior_out.chunk(2, dim=-1)
        logvar_prior = logvar_prior.clamp(-10, 2)  # 数值稳定性

        if use_posterior and h_target is not None:
            # Posterior 分布：看到了目标，分布更精确
            post_in = torch.cat([h_context, h_target], dim=-1)  # (B, D*2)
            post_out = self.posterior_net(post_in)  # (B, z_dim*2)
            mu_post, logvar_post = post_out.chunk(2, dim=-1)
            logvar_post = logvar_post.clamp(-10, 2)
            z = self.reparameterize(mu_post, logvar_post)
        else:
            # Prior 分布：只看到上下文，需要想象
            mu_post = mu_prior
            logvar_post = logvar_prior
            z = self.reparameterize(mu_prior, logvar_prior)

        # 合并确定性 + 随机状态 → D_jepa 维（下游兼容）
        state = self.state_combiner(torch.cat([h_context, z], dim=-1))

        return state, mu_post, logvar_post, mu_prior, logvar_prior

    def compute_kl_loss(self, mu_post: torch.Tensor, logvar_post: torch.Tensor,
                        mu_prior: torch.Tensor, logvar_prior: torch.Tensor) -> torch.Tensor:
        """
        KL(posterior || prior) with free bits。

        Free bits 机制：当 KL 低于阈值时不惩罚，
        防止 KL 过度压制 posterior（导致 posterior ≈ prior，失去信息）。
        """
        # 标准高斯 KL 公式：
        # KL(q || p) = 0.5 * sum(logvar_p - logvar_q + (var_q + (mu_q-mu_p)^2)/var_p - 1)
        var_post = logvar_post.exp()
        var_prior = logvar_prior.exp()

        kl_per_dim = 0.5 * (
            logvar_prior - logvar_post
            + var_post / var_prior
            + (mu_post - mu_prior).pow(2) / var_prior
            - 1
        )  # (B, z_dim)

        # Free bits: 每维度独立 clamp
        free_bits = config.jepa_kl_free_bits
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

        return kl_per_dim.sum(dim=-1).mean()  # scalar

    # ---- v8.0 B路线：自监督 Context Masking ----

    def compute_context_masking_loss(self, context_llm_embs: torch.Tensor,
                                     target_llm_embs: torch.Tensor) -> tuple:
        """
        v8.0 B路线核心：Context Masking 自监督损失。

        I-JEPA 在对话场景的适配：
        - 给定一段对话的上下文（context），预测被遮蔽段（target）的潜空间表示
        - Online Encoder 编码 context → ssl_predictor 预测 target 表示
        - EMA Encoder 编码 target → 作为预测目标（无梯度，缓慢演进）
        - SIGReg 防止所有预测坍塌到同一点

        为什么用独立的 ssl_predictor 而非 CognitivePredictor：
        - I-JEPA 原则：predictor 越简单，encoder 被迫学到越好的表示
        - CognitivePredictor 有 AdaLN 条件化（需要 intent），预训练不需要 intent
        - 预训练和世界模型训练解耦，避免目标冲突

        context_llm_embs: (B, D_llm) — 可见上下文的 LLM 特征
        target_llm_embs:  (B, D_llm) — 被遮蔽段的 LLM 特征

        返回: (ssl_loss, sigreg_loss, z_pred)
               ssl_loss: 预测损失标量
               sigreg_loss: 防坍塌损失标量
               z_pred: (B, D_jepa) 预测的潜表示（用于外部 batch SIGReg）
        """
        # 1. Online Encoder 编码可见上下文（带梯度 → 训练 Encoder）
        z_context = self.encoder(context_llm_embs)  # (B, D_jepa)

        # 2. SSL Predictor 从上下文预测被遮蔽段的表示
        z_pred = self.ssl_predictor(z_context)  # (B, D_jepa)

        # 3. EMA Encoder 编码目标（无梯度 → 稳定目标）
        with torch.no_grad():
            z_target = self.ema_encoder(target_llm_embs)  # (B, D_jepa)

        # 4. MSE loss in latent space
        ssl_loss = F.mse_loss(z_pred, z_target)

        # 5. SIGReg 防坍塌（确保预测向量不都坍塌到同一点）
        sigreg_loss = self.sigreg(z_pred)

        return ssl_loss, sigreg_loss, z_pred

    def predict(self, emb_seq: torch.Tensor, intent_cond: torch.Tensor) -> torch.Tensor:
        """
        在意图条件下预测下一步潜状态。
        emb_seq: (B, T, D) 观测序列
        intent_cond: (B, T, D) 条件信号
        返回: (B, T, D) 预测输出
        """
        return self.predictor(emb_seq, intent_cond)

    # ---- 方案 C: 记忆驱动意图（v5.2 修复 #8：k-means 语义聚类） ----

    def extract_memory_intents(self, memories: list, cortex) -> torch.Tensor:
        """
        从海马体记忆中提取记忆驱动意图。

        v5.3 P2-2 修复：优先复用 retrieve 返回的 embedding（numpy array），
        仅当不存在时才调用 cortex.get_real_embedding（避免重复 LLM forward）。

        memories: list of dict (来自 hippocampus.retrieve，含可选的 'embedding' 字段)
        cortex: LinguisticCortex 实例（仅在缺少 embedding 时使用）
        返回: (M, intent_dim), M <= max_memory_intents
        """
        if not memories:
            return torch.zeros(0, config.jepa_intent_dim, device=config.device)

        # 收集所有记忆的 embedding（优先复用已有的）
        mem_embeddings = []
        for mem in memories:
            text = mem.get('text', '')
            if not text:
                continue
            stored_emb = mem.get('embedding')
            if stored_emb is not None:
                emb = torch.from_numpy(np.asarray(stored_emb)).to(torch.float32).to(config.device)
            else:
                with torch.no_grad():
                    emb = cortex.get_real_embedding(text).to(torch.float32)
            mem_embeddings.append(emb)

        if not mem_embeddings:
            return torch.zeros(0, config.jepa_intent_dim, device=config.device)

        mem_stack = torch.stack(mem_embeddings, dim=0)  # (N, D_llm)

        # k-means 语义聚类：将 N 条记忆聚类为 M 个语义簇
        M = min(config.jepa_max_memory_intents, len(mem_embeddings))
        with torch.no_grad():
            cluster_centers = _kmeans_cluster(mem_stack, M)  # (M, D_llm)

        # 投影每个簇心到意图空间
        intent_vecs = self.memory_intent_proj(cluster_centers)  # (M, intent_dim)
        return intent_vecs * config.jepa_memory_intent_weight

    def extract_memory_cluster_centers(self, memories: list, cortex) -> torch.Tensor:
        """
        v5.7 梯度修复：仅执行记忆收集 + k-means 聚类，返回簇心（无梯度）。

        与 extract_memory_intents 的区别：不调用 memory_intent_proj，
        使 brain.py 训练循环可以在有梯度环境下单独调用 project_cluster_centers，
        让 memory_intent_proj 的权重正常接收梯度更新。

        返回: (M, D_llm) 簇心向量，或 None（无记忆时）
        """
        if not memories:
            return None

        mem_embeddings = []
        for mem in memories:
            text = mem.get('text', '')
            if not text:
                continue
            stored_emb = mem.get('embedding')
            if stored_emb is not None:
                emb = torch.from_numpy(np.asarray(stored_emb)).to(torch.float32).to(config.device)
            else:
                with torch.no_grad():
                    emb = cortex.get_real_embedding(text).to(torch.float32)
            mem_embeddings.append(emb)

        if not mem_embeddings:
            return None

        mem_stack = torch.stack(mem_embeddings, dim=0)  # (N, D_llm)
        M = min(config.jepa_max_memory_intents, len(mem_embeddings))
        with torch.no_grad():
            cluster_centers = _kmeans_cluster(mem_stack, M)  # (M, D_llm)
        return cluster_centers.detach()  # 确保簇心作为常量输入

    def project_cluster_centers(self, cluster_centers: torch.Tensor) -> torch.Tensor:
        """
        v5.7 梯度修复：将 k-means 簇心投影到意图空间（带梯度）。

        此方法在训练循环内调用，memory_intent_proj 的梯度通过
        compute_prediction_loss → intent_encoder → memory_intents 路径正常回传。

        cluster_centers: (M, D_llm) 来自 extract_memory_cluster_centers 的簇心
        返回: (M, intent_dim) 记忆驱动意图向量
        """
        intent_vecs = self.memory_intent_proj(cluster_centers)  # (M, intent_dim)，带梯度
        return intent_vecs * config.jepa_memory_intent_weight

    def get_all_intents(self, memory_intents: torch.Tensor = None) -> torch.Tensor:
        """
        融合可学习意图 + 记忆驱动意图。
        返回: (S+M, intent_dim), 其中 S=固定可学习, M=动态记忆驱动
        """
        all_intents = [self.intent_bank]  # (S, intent_dim)
        if memory_intents is not None and memory_intents.numel() > 0:
            all_intents.append(memory_intents.to(self.intent_bank.device))
        return torch.cat(all_intents, dim=0)

    # ---- 泛化 Rollout（接受任意意图向量） ----

    def rollout_with_intent_vec(self, init_emb: torch.Tensor,
                                intent_vec: torch.Tensor) -> torch.Tensor:
        """
        在潜空间对单条策略意图路径做多步自回归预演。
        泛化版本：接受任意意图向量（不限于 intent_bank 索引）。

        init_emb: (B, 1, D) 初始观测嵌入
        intent_vec: (intent_dim,) 意图向量
        返回: (B, rollout_depth+1, D) 完整推演轨迹
        """
        B = init_emb.size(0)
        D = config.jepa_core_dim
        HS = config.jepa_rollout_history

        # 编码策略意图为条件信号
        intent_cond = self.intent_encoder(intent_vec)  # (D,)
        intent_cond_base = intent_cond.unsqueeze(0).unsqueeze(0).expand(B, 1, D)

        # 自回归推演
        emb_trajectory = init_emb  # (B, 1, D)

        for step in range(config.jepa_rollout_depth):
            T_current = emb_trajectory.size(1)
            start_idx = max(0, T_current - HS)
            emb_window = emb_trajectory[:, start_idx:, :]
            W = emb_window.size(1)
            cond_window = intent_cond_base.expand(B, W, D)

            pred_seq = self.predict(emb_window, cond_window)
            next_pred = pred_seq[:, -1:, :]
            emb_trajectory = torch.cat([emb_trajectory, next_pred], dim=1)

        return emb_trajectory  # (B, rollout_depth+1, D)

    # v7.1: rollout() 旧接口已移除（无调用者）

    def compute_prediction_loss(self, init_emb: torch.Tensor,
                                target_emb: torch.Tensor,
                                memory_intents: torch.Tensor = None) -> tuple:
        """
        v20 JEPA 核心训练损失：随机状态 + Latent Overshooting + 意图多样性 + InfoNCE 对比。

        v20 升级（断裂点 #1 + #3 修复）：
        - target_emb 应来自 encode_fact_target()（海马体外部事实），不是 EMA 自蒸馏
        - 新增 InfoNCE 对比损失：迫使预测不仅靠近正确目标，还远离错误目标
        - InfoNCE 使用 batch 内其他样本的 target 作为 hard negatives

        init_emb: (B, D_jepa) 输入嵌入（确定性编码）
        target_emb: (B, D_jepa) 目标嵌入（应来自 encode_fact_target，已 detach）
        memory_intents: (M, intent_dim) 可选的记忆驱动意图

        返回: (total_loss, pred_loss, sigreg_loss, diversity_loss,
               final_pred_2d, best_intent_idx, kl_loss, contrastive_loss)
        """
        B = init_emb.size(0)
        D = config.jepa_core_dim
        rollout_depth = config.jepa_rollout_depth
        HS = config.jepa_rollout_history

        # ---- 随机状态强化（v8.0 A路线核心） ----
        # 训练时用 posterior（看到了 target，分布更准确）
        stoch_state, mu_post, logvar_post, mu_prior, logvar_prior = \
            self.compute_stochastic_state(init_emb, target_emb, use_posterior=True)

        # KL(posterior || prior)
        kl_loss = self.compute_kl_loss(mu_post, logvar_post, mu_prior, logvar_prior)

        # 用随机增强状态作为 rollout 起点
        init_seq = stoch_state.unsqueeze(1)  # (B, 1, D)

        # ---- 融合所有意图 ----
        all_intents = self.get_all_intents(memory_intents)
        S_total = all_intents.size(0)

        # 批量并行化 rollout
        init_expanded = init_seq.repeat_interleave(S_total, dim=0)  # (B*S, 1, D)
        all_conds = self.intent_encoder(all_intents)  # (S, D)
        cond_expanded = (all_conds.unsqueeze(0)
                         .expand(B, -1, -1)
                         .reshape(B * S_total, D)
                         .unsqueeze(1))  # (B*S, 1, D)

        # ---- Latent Overshooting: 每步加 step_embedding 减少累积误差 ----
        emb_trajectory = init_expanded  # (B*S, 1, D)
        step_predictions = []

        for step in range(rollout_depth):
            T_current = emb_trajectory.size(1)
            start_idx = max(0, T_current - HS)
            emb_window = emb_trajectory[:, start_idx:, :]
            W = emb_window.size(1)

            # Latent Overshooting: 加入步数编码
            step_emb = self.step_embeddings[step].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            step_emb_expanded = step_emb.expand(emb_window.size(0), W, -1)  # (B*S, W, D)
            emb_window_stepped = emb_window + step_emb_expanded

            cond_window = cond_expanded[:, :1, :].expand(-1, W, -1)
            pred_seq = self.predict(emb_window_stepped, cond_window)
            next_pred = pred_seq[:, -1:, :]  # (B*S, 1, D)
            step_predictions.append(next_pred.squeeze(1))  # (B*S, D)
            emb_trajectory = torch.cat([emb_trajectory, next_pred], dim=1)

        # ---- 目标展开 & 每步 MSE ----
        target_expanded = target_emb.unsqueeze(1).expand(B, S_total, D)  # (B, S, D)

        # v20-audit: 计算 per-sample per-intent loss (B, S_total)
        total_step_loss_per_sample = torch.zeros(B, S_total, device=init_emb.device)
        total_weight = 0.0
        for step_idx, step_pred in enumerate(step_predictions):
            step_weight = 1.0 / (step_idx + 1)
            total_weight += step_weight
            step_pred_2d = step_pred.view(B, S_total, D)
            # (B, S_total): 每个样本、每个意图的 MSE
            step_mse = (step_pred_2d - target_expanded).pow(2).mean(dim=-1)
            total_step_loss_per_sample = total_step_loss_per_sample + step_weight * step_mse

        per_sample_intent_loss = total_step_loss_per_sample / total_weight  # (B, S_total)

        # v20-audit F1 修复: per-sample 最优意图索引 (B,)
        # 原来 argmin 跨 batch 平均后取全局标量 → InfoNCE 退化为 ln(B)
        # 现在每个样本独立选自己最优意图，InfoNCE query 才能反映 per-sample 判别力
        best_intent_indices = per_sample_intent_loss.argmin(dim=-1)  # (B,)
        # 兼容性: 保留全局 best_intent_idx 用于返回值和其他接口
        per_intent_loss = per_sample_intent_loss.mean(dim=0)  # (S_total,) 跨 batch 平均
        best_intent_idx = per_intent_loss.argmin().item()

        # soft-min（仍在跨 batch 平均后的 per_intent_loss 上计算）
        soft_weights = F.softmax(
            -per_intent_loss / config.jepa_softmin_temperature, dim=0)
        pred_loss = (soft_weights * per_intent_loss).sum()

        # SIGReg
        final_pred_2d = step_predictions[-1].view(B, S_total, D)
        sigreg_loss = self.sigreg(final_pred_2d)

        # 意图多样性正则化
        intent_normalized = F.normalize(self.intent_bank, dim=-1)
        sim_matrix = intent_normalized @ intent_normalized.t()
        mask = ~torch.eye(config.jepa_num_intents, dtype=torch.bool,
                         device=self.intent_bank.device)
        diversity_loss = sim_matrix[mask].pow(2).mean()

        # ---- v20 断裂点 #3 修复：InfoNCE 对比损失 ----
        # v20-audit F1 修复：使用 per-sample best_intent_indices
        contrastive_loss = self._compute_infonce(
            final_pred_2d, target_emb, best_intent_indices)

        total = (pred_loss
                 + config.jepa_intent_diversity_weight * diversity_loss
                 + config.jepa_kl_weight * kl_loss
                 + config.jepa_contrastive_weight * contrastive_loss)

        return (total, pred_loss, sigreg_loss, diversity_loss,
                final_pred_2d, best_intent_idx, kl_loss, contrastive_loss)

    def _compute_infonce(self, final_pred_2d: torch.Tensor,
                         target_emb: torch.Tensor,
                         best_intent_indices: torch.Tensor) -> torch.Tensor:
        """
        v20 InfoNCE 对比损失：锐化 JEPA 的判别能力。

        v20-audit F1 修复：best_intent_indices 改为 per-sample 向量 (B,)。
        原来使用全局标量导致所有样本用同一意图索引，InfoNCE = ln(B) 恒定。

        核心思想：
        - query_i = 样本 i 的最优意图的预测终态（per-sample 独立选择）
        - positive = 该样本的真实后继事件编码
        - negatives = batch 内其他样本的后继事件编码（in-batch negatives）

        final_pred_2d: (B, S_total, D) 所有意图的预测终态
        target_emb: (B, D) 真实目标编码
        best_intent_indices: (B,) per-sample 最优意图索引

        返回: InfoNCE loss 标量
        """
        B = target_emb.size(0)

        # B=1 时无法构造 negatives（需要至少 2 个样本做对比）
        if B < 2:
            return torch.tensor(0.0, device=target_emb.device)

        # v20-audit F1: per-sample 提取最优意图的预测终态
        # best_intent_indices: (B,) → gather 出 (B, D)
        idx = best_intent_indices.view(B, 1, 1).expand(B, 1, final_pred_2d.size(2))  # (B, 1, D)
        queries = final_pred_2d.gather(1, idx).squeeze(1)  # (B, D)

        # L2 归一化（InfoNCE 在归一化空间效果更好）
        queries_norm = F.normalize(queries, dim=-1)
        targets_norm = F.normalize(target_emb, dim=-1)

        # 计算 (B, B) 相似度矩阵
        # sim[i][j] = query_i · target_j
        sim_matrix = queries_norm @ targets_norm.t()  # (B, B)

        # 温度缩放
        sim_matrix = sim_matrix / config.jepa_contrastive_temperature

        # InfoNCE: 每个样本的正确目标在对角线上
        # 等价于 B-way 分类问题：第 i 个 query 的正确类是第 i 个 target
        labels = torch.arange(B, device=target_emb.device)
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    @torch.no_grad()
    def plan_with_prior(self, llm_features: torch.Tensor,
                        goal_emb: torch.Tensor = None,
                        memory_intents: torch.Tensor = None,
                        n_perturb: int = 4,
                        noise_scale: float = 0.1) -> torch.Tensor:
        """
        v8.0 统一推理规划器（CEM-lite + Prior + Latent Overshooting）。

        替代旧的 generate_candidates()，整合所有 v8.0 升级：
        1. Prior sampling（无目标信息的随机状态）
        2. Step embeddings（Latent Overshooting）
        3. CEM-lite 噪声探索（围绕每个意图添加扰动副本，扩大搜索空间）

        CEM-lite 流程：
        - 基础候选：S+M 个意图（learned + memory）
        - 扰动候选：每个基础意图 × n_perturb 个噪声副本
        - 总候选数 = base + base × n_perturb
        - 全部 rollout → 评估 → 选最优

        llm_features: (1, D_llm) 或 (D_llm,) 用户输入特征
        goal_emb: (D_llm,) 可选的海马体目标记忆嵌入
        memory_intents: (M, intent_dim) 可选的记忆驱动意图
        n_perturb: 每个基础意图生成的噪声副本数
        noise_scale: 扰动强度（标准差倍数）

        返回: trajectory (1, T, D_jepa) 最优路径的完整轨迹
        """
        if llm_features.dim() == 1:
            llm_features = llm_features.unsqueeze(0)

        D = config.jepa_core_dim

        # 1. Encode → Prior stochastic state
        h_context = self.encoder(llm_features)  # (1, D_jepa)
        stoch_state, _, _, _, _ = self.compute_stochastic_state(
            h_context, h_target=None, use_posterior=False)

        # 2. CEM-lite: 基础意图 + 噪声扰动
        base_intents = self.get_all_intents(memory_intents)  # (S+M, intent_dim)
        S_base = base_intents.size(0)

        if n_perturb > 0:
            # 围绕每个基础意图生成噪声副本
            noise = torch.randn(S_base, n_perturb, config.jepa_intent_dim,
                                device=base_intents.device) * noise_scale
            perturbed = base_intents.unsqueeze(1) + noise  # (S_base, n_perturb, intent_dim)
            perturbed = perturbed.view(-1, config.jepa_intent_dim)  # (S_base*n_perturb, intent_dim)
            all_intents = torch.cat([base_intents, perturbed], dim=0)  # (S_total, intent_dim)
        else:
            all_intents = base_intents

        S_total = all_intents.size(0)

        # 3. 批量 rollout（所有候选意图）
        init_seq = stoch_state.unsqueeze(1)  # (1, 1, D)
        init_expanded = init_seq.expand(S_total, -1, -1)  # (S, 1, D)
        all_conds = self.intent_encoder(all_intents)  # (S, D)
        cond_expanded = all_conds.unsqueeze(1)  # (S, 1, D)

        emb_trajectory = init_expanded
        for step in range(config.jepa_rollout_depth):
            T_cur = emb_trajectory.size(1)
            start_idx = max(0, T_cur - config.jepa_rollout_history)
            emb_window = emb_trajectory[:, start_idx:, :]
            W = emb_window.size(1)
            # Latent Overshooting: step_embedding
            step_emb = self.step_embeddings[step].unsqueeze(0).unsqueeze(0)
            emb_window = emb_window + step_emb.expand(S_total, W, -1)
            cond_window = cond_expanded[:, :1, :].expand(-1, W, -1)
            pred_seq = self.predict(emb_window, cond_window)
            next_pred = pred_seq[:, -1:, :]
            emb_trajectory = torch.cat([emb_trajectory, next_pred], dim=1)

        # 4. 选最优路径
        if goal_emb is not None:
            if goal_emb.dim() == 1:
                goal_emb = goal_emb.unsqueeze(0)
            goal_jepa = self.encoder(goal_emb)  # (1, D_jepa)
            final_states = emb_trajectory[:, -1, :]  # (S, D)
            costs = (final_states - goal_jepa.expand(S_total, -1)).pow(2).mean(dim=-1)
            best_idx = costs.argmin().item()
        else:
            best_idx = 0

        trajectory = emb_trajectory[best_idx:best_idx+1]  # (1, T, D)
        return trajectory

    def compute_sigreg_on_batch(self, all_final_preds: torch.Tensor) -> torch.Tensor:
        """
        v5.6 P0-2: 在积累的 batch 上计算 SIGReg 损失。

        解决 B=1 时 SIGReg 统计量不稳定的问题。brain.py 在遍历所有样本后，
        将各样本的 final_pred 堆叠为 (N*S_total, D) Tensor 后调用此方法。

        all_final_preds: (N_total, D) 累积的所有样本预测向量
        返回: sigreg_loss 标量
        """
        return self.sigreg(all_final_preds)

    def compute_query_alignment_loss(self, final_states: torch.Tensor,
                                      target_llm_embs: torch.Tensor) -> torch.Tensor:
        """
        v5.6 P1-7: query_head 对齐损失。

        query_head 此前从未被直接训练，导致其输出与 LLM embedding 空间没有对齐约束。
        此 loss 确保 query_head(最优路径终态) 与目标 LLM embedding 在余弦空间对齐。

        final_states: (B, D_jepa) JEPA 最优路径的终态
        target_llm_embs: (B, D_llm) 目标 LLM 嵌入
        返回: cosine alignment loss 标量
        """
        query_emb = self.query_head(final_states)  # (B, D_llm)
        # 1 - cosine_similarity 作为 loss（完美对齐时 loss=0）
        cos_sim = F.cosine_similarity(query_emb, target_llm_embs.detach(), dim=-1)
        return (1.0 - cos_sim).mean()

    # ---- 意图进化（从 brain.py 下沉，修复 #13） ----

    def evolve_intents(self):
        """
        根据累积经验调整 intent_bank 的方向。

        v5.7 修复：从 replay_buffer 的原始 LLM embedding（entry['tgt']）出发，
        用当前 memory_intent_proj 权重重新投影。消除 v5.6 P0-3 中因存储旧权重
        投影向量导致的意图漂移问题（memory_intent_proj 现在会被训练，权重会变化）。

        排他性分配：每个目标嵌入只能被一个意图占用，
        防止多个意图竞争同一目标导致所有意图收敛到同一方向。
        """
        if len(self.replay_buffer) < 2:
            return

        # v5.7: 从原始 LLM embedding 用当前权重重新投影（消除旧权重投影漂移）
        tgt_embs = [entry['tgt'] for entry in self.replay_buffer.buffer]
        if not tgt_embs:
            return

        with torch.no_grad():
            tgt_stack = torch.stack(tgt_embs)  # (N, D_llm)
            projected = self.memory_intent_proj(tgt_stack)  # (N, intent_dim) 用当前权重

            # 排他性分配：每个目标只能被一个意图占用
            S = config.jepa_num_intents
            used_indices = set()
            for s in range(S):
                intent_vec = self.intent_bank[s]
                sims = F.cosine_similarity(
                    intent_vec.unsqueeze(0), projected, dim=-1)  # (N,)
                # 将已被其他意图占用的目标标记为不可选
                for used_idx in used_indices:
                    sims[used_idx] = -1.0
                best_idx = sims.argmax().item()
                used_indices.add(best_idx)
                alpha = config.evolve_intent_ema_alpha
                self.intent_bank.data[s] = (
                    (1.0 - alpha) * self.intent_bank.data[s]
                    + alpha * projected[best_idx]
                )
