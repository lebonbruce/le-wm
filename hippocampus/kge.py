"""
hippocampus/kge.py —— TransE 知识图谱嵌入 + LLM→KGE 投影层

v5.4 架构修复：
- P0-2: train_on_graph / train_projection 不再内建优化器，
  改为 compute_xxx_loss 返回 loss 值，由 brain.py 的统一优化器驱动
- P1-5: 新增 _remove_entity 方法，merge_similar 时清理幽灵参数
- P1-1: _safe_key 使用 SHA256 哈希防碰撞（v5.6 P2-3: MD5→SHA256）
- v5.2 #9: mini-batch 向量化训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import hashlib
import numpy as np
import networkx as nx
from mvp_config import config


class TransE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = config.kge_embed_dim
        self.margin = config.kge_margin
        self.entity_embeddings = nn.ParameterDict()
        self.relation_embeddings = nn.ParameterDict()

        # v5.3: LLM→KGE 投影层（将 LLM 空间的查询映射到 TransE 嵌入空间）
        # 在 consolidate 阶段用已有节点的 (llm_emb, transe_emb) 配对训练
        self.llm_to_kge_proj = nn.Sequential(
            nn.Linear(config.llm_hidden_size, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )
        # 投影层是否已训练过的标记
        self._proj_trained = False

    @staticmethod
    def _safe_key(key: str) -> str:
        """v5.6 P2-3: 确定性哈希，将任意字符串转为合法的 ParameterDict key（SHA256 替代 MD5）"""
        return "k_" + hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]

    def _ensure_entity(self, eid: str):
        k = self._safe_key(eid)
        if k not in self.entity_embeddings:
            self.entity_embeddings[k] = nn.Parameter(torch.randn(self.embed_dim) * 0.1)
        return k

    def _ensure_relation(self, rtype: str):
        k = self._safe_key(rtype)
        if k not in self.relation_embeddings:
            self.relation_embeddings[k] = nn.Parameter(torch.randn(self.embed_dim) * 0.1)
        return k

    def _remove_entity(self, eid: str):
        """P1-5: 从 ParameterDict 中移除指定实体的嵌入，防止幽灵参数残留"""
        k = self._safe_key(eid)
        if k in self.entity_embeddings:
            del self.entity_embeddings[k]

    def _cleanup_orphan_relations(self, graph: nx.MultiDiGraph):
        """
        BUG-4 修复：扫描 relation_embeddings，删除图中已无任何边使用该关系类型的孤立嵌入。

        适用场景：当某种自定义 edge_type 只存在于被删除节点的边上，节点删除后
        该关系类型在整个图中已无任何边，其嵌入参数应同步销毁，防止：
        ① 梯度计算时噪声项无限积累；② 内存泄漏（ParameterDict 无限增长）。

        注意：全局共享的关系类型（如"时间临近"、"语义相似"）只要图中还有一条
        同类型的边，就不会被删除，确保不误伤正常参数。
        """
        # 收集图中当前实际使用的所有边类型（去重）
        # 补丁C：不对缺少 'type' 属性的脏边做 fallback='rel'，
        # 否则会产生一个名为 'rel' 的伪关系嵌入永远免疫清理。
        active_relation_keys: set[str] = set()
        for _, _, data in graph.edges(data=True):
            rtype = data.get('type')
            if rtype:  # 忽略无 type 属性的脏边，不污染 active_relation_keys
                active_relation_keys.add(self._safe_key(rtype))

        # 找出 ParameterDict 中不再被任何边引用的关系嵌入 key
        orphan_keys = [k for k in list(self.relation_embeddings.keys())
                       if k not in active_relation_keys]
        for k in orphan_keys:
            del self.relation_embeddings[k]

        return len(orphan_keys)  # 返回清理数量，便于日志记录

    def get_entity_embedding(self, eid: str) -> torch.Tensor:
        k = self._safe_key(eid)
        if k in self.entity_embeddings:
            return self.entity_embeddings[k]
        return torch.zeros(self.embed_dim, device=config.device)

    def project_query(self, llm_emb: torch.Tensor) -> torch.Tensor:
        """
        v5.3: 将 LLM 空间的查询 embedding 投影到 TransE 嵌入空间。
        只有在投影层训练完成后才有意义。

        llm_emb: (D_llm,) 或 (B, D_llm)
        返回: (D_kge,) 或 (B, D_kge)
        """
        with torch.no_grad():
            if llm_emb.dim() == 1:
                llm_emb = llm_emb.unsqueeze(0)
            projected = self.llm_to_kge_proj(llm_emb.to(torch.float32))
            return projected.squeeze(0)

    def compute_projection_loss(self, node_llm_embs: dict) -> torch.Tensor:
        """
        v5.4 P0-2: 计算 LLM→KGE 投影层的 MSE 损失（不内建优化器）。

        用图谱中已有节点的 (llm_embedding, transe_embedding) 配对做 MSE 回归。
        返回 loss 标量，由统一优化器驱动反向传播。

        node_llm_embs: dict {node_id → numpy array (D_llm)} 来自 memory.node_embeddings
        返回: loss tensor（如数据不足则返回 0 tensor）
        """
        llm_list, kge_list = [], []
        for nid, llm_np in node_llm_embs.items():
            k = self._safe_key(nid)
            if k in self.entity_embeddings:
                llm_t = torch.from_numpy(llm_np).to(torch.float32).to(config.device)
                # v5.6 P1-4 设计决策：有意 detach TransE embedding 作为固定目标。
                # 投影层去适应 TransE 空间，而非反过来。与 GAN 交替训练类似：
                # TransE embedding 由 consolidation_loss 更新，投影层由 proj_loss 更新，
                # 两者在统一优化器中交替开火，收敛性由 Adam 的动量平滑保证。
                kge_t = self.entity_embeddings[k].detach().to(config.device)
                llm_list.append(llm_t)
                kge_list.append(kge_t)

        if len(llm_list) < 2:
            return torch.tensor(0.0, device=config.device)

        llm_batch = torch.stack(llm_list)
        kge_batch = torch.stack(kge_list)

        self.llm_to_kge_proj.to(config.device)
        projected = self.llm_to_kge_proj(llm_batch)
        loss = F.mse_loss(projected, kge_batch)
        self._proj_trained = True
        return loss

    def compute_consolidation_loss(self, graph: nx.MultiDiGraph,
                                   batch_size: int = None) -> torch.Tensor:
        """
        v5.4 P0-2: 计算一个 mini-batch 的 TransE margin ranking loss（不内建优化器）。

        每次调用采样一个 batch 的三元组计算 loss 并返回，
        由 brain.py 的统一优化器管理梯度更新。

        返回: loss tensor（如无边则返回 0 tensor）
        """
        batch_size = batch_size or config.kge_batch_size
        # 过滤无 type 属性的脏边（与 _cleanup_orphan_relations 策略一致，
        # 防止产生永不被清理的 'rel' 幽灵关系嵌入）
        edges = [(u, v, d) for u, v, d in graph.edges(data=True) if d.get('type')]
        if not edges:
            return torch.tensor(0.0, device=config.device)

        # 确保所有实体/关系已注册
        for u, v, d in edges:
            self._ensure_entity(u)
            self._ensure_entity(v)
            self._ensure_relation(d['type'])

        all_nodes = list(graph.nodes)

        # 随机采样一个 batch
        batch = random.sample(edges, min(batch_size, len(edges)))

        h_list, r_list, t_list, neg_list = [], [], [], []
        for h_id, t_id, data in batch:
            r = data['type']  # 已在上方过滤，此处必有 type
            h_list.append(self.entity_embeddings[self._safe_key(h_id)])
            r_list.append(self.relation_embeddings[self._safe_key(r)])
            t_list.append(self.entity_embeddings[self._safe_key(t_id)])
            neg_id = random.choice(all_nodes)
            self._ensure_entity(neg_id)
            neg_list.append(self.entity_embeddings[self._safe_key(neg_id)])

        h_batch = torch.stack(h_list)
        r_batch = torch.stack(r_list)
        t_batch = torch.stack(t_list)
        neg_batch = torch.stack(neg_list)

        pos_dist = (h_batch + r_batch - t_batch).norm(p=2, dim=-1)
        neg_dist = (h_batch + r_batch - neg_batch).norm(p=2, dim=-1)
        loss = F.relu(self.margin + pos_dist - neg_dist).mean()
        return loss
