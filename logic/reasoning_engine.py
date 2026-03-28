"""
logic/reasoning_engine.py —— 递归推理引擎

核心创新: 递归推理循环（Recursive Reasoning Loop）
不是一次 forward 给答案，而是迭代式推理:
  每轮 JEPA rollout → 中间结论 → 存入工作记忆 → 下一轮取出继续推

组件:
1. TripleEncoder     — 将 (h, r, t) 三元组编码到 JEPA 潜空间
2. WorkingMemory     — 固定大小的张量槽位，存储中间推理结果
3. ConclusionDecoder — 将 JEPA 预测解码为关系概率 + 实体概率
4. HaltingMechanism  — ACT (Adaptive Computation Time) 动态停止
5. LogicReasoningEngine — 主推理循环编排器

架构映射:
  TransE embeddings    → 事实三元组的向量表示
  JEPA Encoder         → 三元组 → 潜空间
  JEPA Predictor       → 如果 A 则 B 的因果推理
  WorkingMemory        → 海马体工作记忆（中间结论缓存）
  intent_bank          → 推理策略（传递、逆向、组合）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from logic.logic_config import logic_config
from mvp_config import config as brain_config


class TripleEncoder(nn.Module):
    """
    三元组编码器: (h_emb, r_emb, t_emb) → JEPA 潜空间向量

    设计决策:
    - 拼接 (h, r, t) 三个 kge_embed_dim 向量 → 投影到 jepa_core_dim
    - 比简单的 h+r-t TransE scoring 更有表达力
    - 保留三元组内部结构信息（谁是头、谁是尾、什么关系）
    """
    def __init__(self):
        super().__init__()
        kge_dim = logic_config.kge_embed_dim     # 128
        jepa_dim = logic_config.jepa_core_dim     # 1536
        hidden = logic_config.triple_encoder_hidden  # 512

        # 拼接三个 128 维向量 → 384 → 512 → 1536
        self.encoder = nn.Sequential(
            nn.Linear(kge_dim * 3, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, jepa_dim),
            nn.GELU(),
            nn.LayerNorm(jepa_dim),
        )

    def forward(self, h_emb: torch.Tensor, r_emb: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:
        """
        h_emb: (B, kge_dim) 头实体嵌入
        r_emb: (B, kge_dim) 关系嵌入
        t_emb: (B, kge_dim) 尾实体嵌入
        返回: (B, jepa_dim) 三元组的潜空间表示
        """
        combined = torch.cat([h_emb, r_emb, t_emb], dim=-1)  # (B, kge_dim*3)
        return self.encoder(combined)  # (B, jepa_dim)

    def encode_query(self, h_emb: torch.Tensor, r_emb: torch.Tensor) -> torch.Tensor:
        """
        编码查询 (h, r, ?)：尾实体用零向量占位

        h_emb: (B, kge_dim) 头实体嵌入
        r_emb: (B, kge_dim) 关系嵌入
        返回: (B, jepa_dim) 查询的潜空间表示
        """
        zero_t = torch.zeros_like(h_emb)
        return self.forward(h_emb, r_emb, zero_t)


class WorkingMemory(nn.Module):
    """
    工作记忆: 固定大小的张量槽位，存储中间推理结论。

    类似人脑的短期工作记忆:
    - 容量有限 (working_memory_size 个槽位)
    - 新结论写入 → 旧结论保留（FIFO 当满时）
    - 读取时用 attention 加权（让推理器自动选择相关结论）

    注意: WorkingMemory 是 per-problem 实例化的，不跨问题共享。
    """
    def __init__(self, capacity: int = None):
        super().__init__()
        self.capacity = capacity or logic_config.working_memory_size
        self.jepa_dim = logic_config.jepa_core_dim

        # 读取注意力: 用 query 向量读取记忆
        self.read_query_proj = nn.Linear(self.jepa_dim, self.jepa_dim)
        self.read_key_proj = nn.Linear(self.jepa_dim, self.jepa_dim)
        self.read_value_proj = nn.Linear(self.jepa_dim, self.jepa_dim)
        self.read_norm = nn.LayerNorm(self.jepa_dim)

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        初始化空工作记忆。

        返回: (B, capacity, jepa_dim) 全零槽位
        """
        return torch.zeros(
            batch_size, self.capacity, self.jepa_dim,
            device=device
        )

    def write(self, memory: torch.Tensor, new_content: torch.Tensor,
              write_idx: int) -> torch.Tensor:
        """
        写入新结论到工作记忆的指定位置。

        memory: (B, capacity, jepa_dim) 当前记忆状态
        new_content: (B, jepa_dim) 新的中间结论
        write_idx: 写入槽位索引（FIFO 循环）
        返回: 更新后的记忆 (B, capacity, jepa_dim)
        """
        idx = write_idx % self.capacity
        memory = memory.clone()
        memory[:, idx, :] = new_content
        return memory

    def read(self, memory: torch.Tensor, query: torch.Tensor,
             num_filled: int) -> torch.Tensor:
        """
        用注意力机制从工作记忆中读取相关信息。

        memory: (B, capacity, jepa_dim) 当前记忆状态
        query: (B, jepa_dim) 或 (jepa_dim,) 读取查询
        num_filled: 已填充的槽位数（未填充的不参与注意力）
        返回: (B, jepa_dim) 加权聚合的记忆内容
        """
        # 保证 query 为 2D (B, D)
        if query.dim() == 1:
            query = query.unsqueeze(0)

        if num_filled <= 0:
            return torch.zeros_like(query)

        # 只读取已填充部分
        filled_slots = min(num_filled, self.capacity)
        active_memory = memory[:, :filled_slots, :]  # (B, F, D)

        # 标准 attention 读取
        q = self.read_query_proj(query).unsqueeze(1)       # (B, 1, D)
        k = self.read_key_proj(active_memory)              # (B, F, D)
        v = self.read_value_proj(active_memory)             # (B, F, D)

        # 缩放点积注意力
        d_k = q.size(-1) ** 0.5
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / d_k  # (B, 1, F)
        attn_weights = F.softmax(attn_scores, dim=-1)          # (B, 1, F)
        context = torch.bmm(attn_weights, v).squeeze(1)        # (B, D)

        return self.read_norm(context)


class ConclusionDecoder(nn.Module):
    """
    结论解码器: 将 JEPA 预测的潜空间向量解码为逻辑判断。

    两个输出头:
    1. answer_head: 二分类——查询是否成立 (True/False)
    2. relation_head: 多分类——在所有可能的关系类型中，
       预测最可能的推导关系（辅助训练信号）

    设计决策:
    - 不解码回具体实体（太难且不必要），而是直接判断 query 是否成立
    - 输入 = JEPA 预测 + 原始查询编码 的拼接（让网络对比推理结果和问题）
    """
    def __init__(self, num_relations: int):
        super().__init__()
        jepa_dim = logic_config.jepa_core_dim
        hidden = logic_config.conclusion_decoder_hidden

        # 拼接 JEPA 预测 + 查询编码 → 判断
        self.answer_head = nn.Sequential(
            nn.Linear(jepa_dim * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        # 关系分类头（辅助任务）
        self.relation_head = nn.Sequential(
            nn.Linear(jepa_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_relations),
        )

    def forward(self, reasoning_output: torch.Tensor,
                query_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        reasoning_output: (B, jepa_dim) 递归推理循环的最终输出
        query_encoding: (B, jepa_dim) 查询三元组的编码
        返回: (answer_logit, relation_logits)
            answer_logit: (B, 1) 查询成立的 logit
            relation_logits: (B, num_relations) 关系类型概率 logits
        """
        # 拼接推理结果和查询，让解码器对比
        combined = torch.cat([reasoning_output, query_encoding], dim=-1)
        answer_logit = self.answer_head(combined)          # (B, 1)
        relation_logits = self.relation_head(reasoning_output)  # (B, R)
        return answer_logit, relation_logits


class HaltingMechanism(nn.Module):
    """
    ACT (Adaptive Computation Time) 停止机制。

    学习何时停止推理循环:
    - 每轮推理后输出 halt_prob ∈ [0, 1]
    - 累积 halt_prob 超过阈值 → 停止推理
    - 不同问题需要不同推理深度（1步题 vs 3步题）
    - 训练时强制跑满 max_depth 轮但用 halt_prob 做加权
    """
    def __init__(self):
        super().__init__()
        jepa_dim = logic_config.jepa_core_dim

        self.halt_head = nn.Sequential(
            nn.Linear(jepa_dim, jepa_dim // 4),
            nn.GELU(),
            nn.Linear(jepa_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B, jepa_dim) 当前推理状态
        返回: (B, 1) halt_prob ∈ [0, 1]
        """
        return self.halt_head(state)


class EmbeddingTable(nn.Module):
    """
    实体和关系的嵌入表。

    独立于 TransE（TransE 用于海马体知识图谱），
    逻辑推理模块维护自己的嵌入表，维度为 kge_embed_dim。
    可在 Phase 2 与 TransE 嵌入对接。
    """
    def __init__(self):
        super().__init__()
        self.embed_dim = logic_config.kge_embed_dim
        # 动态注册的嵌入（用 ParameterDict）
        self.entity_embeddings = nn.ParameterDict()
        self.relation_embeddings = nn.ParameterDict()
        # 名称到安全键的映射
        self._name_to_key: Dict[str, str] = {}
        self._counter = 0

    def _safe_key(self, name: str) -> str:
        """生成合法的 ParameterDict 键"""
        if name not in self._name_to_key:
            self._name_to_key[name] = f"e_{self._counter}"
            self._counter += 1
        return self._name_to_key[name]

    def register_entity(self, name: str) -> None:
        """注册实体（如果尚未注册）"""
        key = self._safe_key(name)
        if key not in self.entity_embeddings:
            self.entity_embeddings[key] = nn.Parameter(
                torch.randn(self.embed_dim) * 0.1
            )

    def register_relation(self, name: str) -> None:
        """注册关系（如果尚未注册）"""
        key = self._safe_key(name)
        if key not in self.relation_embeddings:
            self.relation_embeddings[key] = nn.Parameter(
                torch.randn(self.embed_dim) * 0.1
            )

    def get_entity(self, name: str) -> torch.Tensor:
        """获取实体嵌入 (embed_dim,)"""
        key = self._safe_key(name)
        return self.entity_embeddings[key]

    def get_relation(self, name: str) -> torch.Tensor:
        """获取关系嵌入 (embed_dim,)"""
        key = self._safe_key(name)
        return self.relation_embeddings[key]

    def register_from_dataset(self, entities: List[str],
                                relations: List[str]) -> None:
        """从数据集中批量注册所有实体和关系"""
        for e in entities:
            self.register_entity(e)
        for r in relations:
            self.register_relation(r)


class LogicReasoningEngine(nn.Module):
    """
    递归推理引擎 —— Phase 1 核心。

    推理循环:
    ```
    for step in range(max_depth):
        1. 从工作记忆读取当前已知信息
        2. 与查询编码融合
        3. JEPA Predictor 在意图条件下预测
        4. 中间结论写入工作记忆
        5. ACT 检查是否停止
    ```

    与 JEPA 世界模型的对接:
    - 复用 CognitivePredictor 的权重结构（AdaLN-zero Transformer）
    - 复用 intent_bank 的意图条件化机制
    - 但新建独立实例，不修改 NLP 路径的 SubconsciousJEPA
    """
    def __init__(self, num_relations: int):
        super().__init__()
        from jepa_engine.predictor import CognitivePredictor
        from jepa_engine.sigreg import SIGReg

        self.jepa_dim = logic_config.jepa_core_dim
        self.max_depth = logic_config.max_reasoning_depth

        # 子组件
        self.embedding_table = EmbeddingTable()
        self.triple_encoder = TripleEncoder()
        self.working_memory = WorkingMemory()
        self.conclusion_decoder = ConclusionDecoder(num_relations)
        self.halting = HaltingMechanism()
        self.sigreg = SIGReg()

        # JEPA 风格的 Predictor（独立实例，不与 NLP 共享）
        self.predictor = CognitivePredictor()

        # 推理策略意图（逻辑推理专用 intent_bank）
        # 不同的 intent 对应不同推理策略（传递、逆向、组合等）
        self.num_intents = brain_config.jepa_num_intents  # 6
        self.intent_bank = nn.Parameter(
            torch.randn(self.num_intents, brain_config.jepa_intent_dim) * 0.1
        )
        self.intent_encoder = nn.Sequential(
            nn.Linear(brain_config.jepa_intent_dim, self.jepa_dim),
            nn.GELU(),
            nn.Linear(self.jepa_dim, self.jepa_dim),
        )

        # 融合层: 将工作记忆读取结果 + 当前查询状态融合
        self.fusion = nn.Sequential(
            nn.Linear(self.jepa_dim * 2, self.jepa_dim),
            nn.GELU(),
            nn.LayerNorm(self.jepa_dim),
        )

        # 查询→意图空间投影（用于意图选择）
        # query_encoding 是 jepa_dim=1536，intent_bank 是 intent_dim=128
        self.query_to_intent = nn.Linear(
            self.jepa_dim, brain_config.jepa_intent_dim)

        # 步数编码（每步推理有独立的位置信号）
        self.step_embeddings = nn.Parameter(
            torch.randn(self.max_depth, self.jepa_dim) * 0.02
        )

    def _batch_encode_triples(
            self, triples_list: List[List[Tuple[str, str, str]]],
            max_triples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量编码多组三元组（padding到统一长度）。

        triples_list: 每个样本的三元组列表 [[(h,r,t), ...], ...]
        max_triples: padding 目标长度
        返回: (encodings, mask)
            encodings: (B, max_triples, jepa_dim) padding后的编码
            mask: (B, max_triples) bool，True=有效位置
        """
        device = logic_config.device
        B = len(triples_list)
        D = self.jepa_dim

        encodings = torch.zeros(B, max_triples, D, device=device)
        mask = torch.zeros(B, max_triples, dtype=torch.bool, device=device)

        # 批量收集所有三元组的 h/r/t 嵌入
        all_h, all_r, all_t = [], [], []
        sample_indices = []  # (sample_idx, slot_idx)

        for i, triples in enumerate(triples_list):
            for j, (h, r, t) in enumerate(triples[:max_triples]):
                all_h.append(self.embedding_table.get_entity(h))
                all_r.append(self.embedding_table.get_relation(r))
                all_t.append(self.embedding_table.get_entity(t))
                sample_indices.append((i, j))
                mask[i, j] = True

        if not all_h:
            return encodings, mask

        # 一次性编码所有三元组
        h_stack = torch.stack(all_h)  # (total, kge_dim)
        r_stack = torch.stack(all_r)
        t_stack = torch.stack(all_t)
        encoded = self.triple_encoder(h_stack, r_stack, t_stack)  # (total, D)

        # 散布回 (B, max_triples, D)
        for idx, (si, sj) in enumerate(sample_indices):
            encodings[si, sj] = encoded[idx]

        return encodings, mask

    def _batch_encode_queries(
            self, queries: List[Tuple[str, str, str]]) -> torch.Tensor:
        """
        批量编码查询三元组。

        queries: [(h, r, t), ...]
        返回: (B, jepa_dim)
        """
        all_h, all_r, all_t = [], [], []
        for h, r, t in queries:
            all_h.append(self.embedding_table.get_entity(h))
            all_r.append(self.embedding_table.get_relation(r))
            all_t.append(self.embedding_table.get_entity(t))

        h_stack = torch.stack(all_h)
        r_stack = torch.stack(all_r)
        t_stack = torch.stack(all_t)
        return self.triple_encoder(h_stack, r_stack, t_stack)  # (B, D)

    def forward_batch(self, batch: List[Dict],
                       max_depth: int = None) -> Dict[str, torch.Tensor]:
        """
        批量并行递归推理 + ACT (Adaptive Computation Time)。

        核心创新（Phase 2）:
        - 每步产生 halt_prob，决定该步的输出权重
        - 最终 answer = Σ(halt_weight_t × answer_t) 加权组合
        - 训练时: 固定跑 max_depth 步，但用 ACT 权重做加权
        - 推理时: 累积 halt_prob > threshold 后提前终止

        ACT halting 分布 (Graves 2016):
          p_t = halt_prob_t × Π_{i<t}(1 - halt_prob_i)   几何分布
          remainder = 1 - Σp_t  分配给最后一步
        """
        max_depth = max_depth or logic_config.initial_reasoning_depth
        device = logic_config.device
        B = len(batch)

        # 1. 批量编码所有事实和查询
        facts_list = [s["facts"] for s in batch]
        queries = [s["query"] for s in batch]
        max_facts = max(len(f) for f in facts_list)
        max_facts = min(max_facts, self.working_memory.capacity)

        fact_encodings, fact_mask = self._batch_encode_triples(
            facts_list, max_facts)  # (B, F, D), (B, F)
        query_encodings = self._batch_encode_queries(queries)  # (B, D)

        # 2. 初始化工作记忆 + 写入已知事实
        memory = self.working_memory.init_memory(B, device)  # (B, cap, D)
        num_facts_per_sample = fact_mask.sum(dim=1)  # (B,)
        max_filled = int(num_facts_per_sample.max().item())

        for slot in range(max_filled):
            memory = self.working_memory.write(
                memory, fact_encodings[:, slot, :], slot)
        num_filled = max_filled

        # 3. 意图选择
        with torch.no_grad():
            avg_query = query_encodings.mean(dim=0, keepdim=True)
            query_in_intent = self.query_to_intent(avg_query)
            intent_sims = F.cosine_similarity(
                query_in_intent.expand(self.num_intents, -1),
                self.intent_bank, dim=-1
            )
            best_intent_idx = intent_sims.argmax().item()

        intent_vec = self.intent_bank[best_intent_idx]
        intent_cond = self.intent_encoder(intent_vec)
        intent_cond = intent_cond.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)

        # 4. ACT 递归推理循环
        current_state = query_encodings  # (B, D)

        # ACT 状态追踪
        cumulative_halt = torch.zeros(B, device=device)       # Σ step_weight
        running_product = torch.ones(B, device=device)         # Π (1 - halt_prob)
        accumulated_output = torch.zeros(B, 1, device=device)  # 加权 answer logits
        accumulated_rel = torch.zeros(
            B, self.conclusion_decoder.relation_head[-1].out_features,
            device=device)  # 加权 relation logits
        actual_steps = torch.zeros(B, device=device)           # 实际推理步数
        all_halt_probs = []  # 用于简单的 halting 正则化

        # 全量解码: 每步都调用 ConclusionDecoder
        # 注: 稀疏 Decoder（只在后半段解码）因 ACT halt_prob 坍塌问题被禁用。
        # 当 halt_prob→1 时所有权重集中在首步，跳过首步解码会导致零输出。
        decode_start = 0

        # 缓存最近一次 Decoder 输出（用于 remainder 分配）
        last_answer = torch.zeros(B, 1, device=device)
        last_rel = torch.zeros(
            B, self.conclusion_decoder.relation_head[-1].out_features,
            device=device)

        for step in range(max_depth):
            # 4a. 批量读取工作记忆
            memory_context = self.working_memory.read(
                memory, current_state, num_filled)

            # 4b. 融合
            fused = self.fusion(
                torch.cat([current_state, memory_context], dim=-1))

            # 4c. 步数编码（动态扩展）
            if step < self.step_embeddings.size(0):
                fused = fused + self.step_embeddings[step].unsqueeze(0)

            # 4d. JEPA Predictor 批量推理
            pred_input = fused.unsqueeze(1)
            predicted = self.predictor(pred_input, intent_cond)
            predicted_state = predicted.squeeze(1)

            # 4e. 写入工作记忆
            memory = self.working_memory.write(
                memory, predicted_state, num_filled)
            num_filled += 1

            # 4f. ACT halting（每步都运行，保持完整的停止分布）
            halt_prob = self.halting(predicted_state).squeeze(-1)  # (B,)
            halt_prob = halt_prob.clamp(min=1e-6, max=1.0 - 1e-6)
            all_halt_probs.append(halt_prob.mean())

            # 几何分布权重
            step_weight = halt_prob * running_product  # (B,)

            # 4g. 稀疏 Decoder: 只在后半段调用 ConclusionDecoder
            if step >= decode_start:
                step_answer, step_rel = self.conclusion_decoder(
                    predicted_state, query_encodings)
                last_answer = step_answer
                last_rel = step_rel

                # 加权累积
                accumulated_output += step_weight.unsqueeze(-1) * step_answer
                accumulated_rel += step_weight.unsqueeze(-1) * step_rel
            # 前半段: 不调 Decoder，step_weight 分配给后续步骤
            # （running_product 会自然把概率传递给后面的步骤）

            # 更新 ACT 状态
            cumulative_halt = cumulative_halt + step_weight
            running_product = (running_product * (1 - halt_prob)).clamp(min=1e-8)
            actual_steps = actual_steps + 1

            # 4h. 更新状态
            current_state = predicted_state

            # 4i. 推理时动态停止
            if not self.training:
                if (cumulative_halt >= logic_config.halt_threshold).all():
                    break

        # 5. remainder 分配给最后一步的 Decoder 输出
        remainder = (1.0 - cumulative_halt).clamp(min=0.0)  # (B,)
        accumulated_output += remainder.unsqueeze(-1) * last_answer
        accumulated_rel += remainder.unsqueeze(-1) * last_rel

        # 6. 组装标签
        labels = torch.tensor(
            [[1.0 if s["answer"] else 0.0] for s in batch],
            device=device)

        # 7. 简化的 halting 正则化: 鼓励尽早停止
        # 使用平均 halt_prob（越高 = 越早停止 = 越便宜）
        # 负号: 最小化 -mean(halt_prob) = 最大化 halt_prob
        if all_halt_probs:
            ponder_loss = -torch.stack(all_halt_probs).mean()
        else:
            ponder_loss = torch.tensor(0.0, device=device)

        # 8. SIGReg 防坍塌
        sigreg_loss = self.sigreg(current_state) if B > 1 else torch.tensor(0.0, device=device)

        return {
            "answer_logits": accumulated_output,   # (B, 1) ACT 加权
            "labels": labels,
            "halt_loss": ponder_loss,
            "sigreg_loss": sigreg_loss,
            "avg_steps": actual_steps.mean().item(),
        }
