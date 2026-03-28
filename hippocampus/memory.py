"""
hippocampus/memory.py —— SQLite + NetworkX + FAISS 三引擎海马体

v5.7 数据守恒全修复（地基层审计）：
- 漏洞A/F: _add_edge 独立路径加 with self.conn: 事务包裹，SQL 失败时内存引擎不写入
- 漏洞B/C/D: merge_similar 重构为「SQLite 优先原子事务」模式，每对节点作为一个完整事务，
             SQL 成功后才更新内存引擎，崩溃时 FAISS/NetworkX 不产生孤岛
- 漏洞E: 删除 memorize_dialogue 末尾多余 conn.commit()（与内部事务重叠)
- 漏洞F: _load_from_sqlite 对 NULL embedding 节点发出明确 WARNING
- v5.6 全部功能保留（FAISS、TransE 投影融合、MultiDiGraph）
"""
import torch
import numpy as np
import networkx as nx
import sqlite3
import time
import uuid
import faiss
from collections import deque
from mvp_config import config
from hippocampus.kge import TransE
from hippocampus.pagerank import personalized_pagerank


class HippocampalMemory:
    """
    SQLite + NetworkX + FAISS 三引擎海马体。

    v5.6 P2-4: 不再继承 nn.Module（原因：SQLite Connection 不可 pickle）。
    kge (TransE) 作为普通属性存储，由 brain.py 将其单独注册为 nn.Module 子模块。

    SQLite: 持久化存储 (ACID, BLOB embedding, 无限容量)
    NetworkX: 内存中图算法计算 (PPR, BFS)
    FAISS: O(logN) 向量检索索引（语义检索 + 自动建edge）
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        # v5.6 P2-4: kge 作为普通属性（不通过 nn.Module 自动注册）
        # brain.py 中通过 self.kge = hippocampus.kge 单独注册为子模块
        self.kge = TransE()
        self.node_embeddings = {}  # node_id -> numpy array (内存缓存)

        # v5.5 #5: FAISS 向量索引升级为 IndexIDMap2
        # 支持 add_with_ids 增量添加和 remove_ids 局部删除
        # 底层仍用 IndexFlatIP（inner product = cosine sim，需 L2 归一化后 add）
        self._faiss_dim = config.llm_hidden_size
        self._faiss_base = faiss.IndexFlatIP(self._faiss_dim)
        self._faiss_index = faiss.IndexIDMap2(self._faiss_base)
        self._faiss_next_id = 0          # 自增整数 ID（FAISS 要求 int64 ID）
        self._faiss_nid_to_int = {}      # node_id(str) → faiss int64 ID
        self._faiss_int_to_nid = {}      # faiss int64 ID → node_id(str)

        # v8.1: journal_mode=MEMORY 避免 Docker 卷挂载的 WAL 文件 I/O 冲突
        # WAL 模式会创建 -wal/-shm 辅助文件，在 Windows↔Linux 跨文件系统挂载时可能失败
        # MEMORY 模式将日志保持在内存中，牺牲崩溃恢复换取 I/O 稳定性
        self.conn = sqlite3.connect(config.memory_db_path)
        self.conn.execute("PRAGMA journal_mode=MEMORY")
        self.conn.execute("PRAGMA synchronous=OFF")
        self._init_tables()
        self._load_from_sqlite()

    def _init_tables(self):
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding BLOB,
            created_at REAL NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS edges (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            type TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            created_at REAL NOT NULL,
            PRIMARY KEY (src, dst, type)
        )""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)""")
        self.conn.commit()

    def _load_from_sqlite(self):
        """启动时从 SQLite 加载到内存（MultiDiGraph + FAISS + TransE 忠实还原）"""
        c = self.conn.cursor()
        node_count = 0
        null_emb_count = 0  # 漏洞F修复：追踪 NULL embedding 节点数量
        for row in c.execute("SELECT id, text, embedding FROM nodes"):
            nid, text, emb_blob = row
            self.graph.add_node(nid, text=text)
            if emb_blob:
                self.node_embeddings[nid] = np.frombuffer(emb_blob, dtype=np.float32).copy()
            else:
                # 漏洞F修复：NULL embedding 节点会进入 NetworkX 和 TransE 但不进入 FAISS，
                # 导致四引擎计数不一致。发出明确警告，便于运维排查。
                null_emb_count += 1
            # BUG-5 修复：重启后同步恢复 TransE 实体参数表，保持四引擎实体名单一致。
            # 注：此处是随机初始化而非恢复权重（权重持久化需由 brain.py checkpoint 保证），
            # 但至少确保 entity_embeddings 键集合与 SQLite 节点集合一致，
            # 防止 find_seeds() 中 TransE 融合评分静默退化为零向量。
            self.kge._ensure_entity(nid)
            node_count += 1

        # 漏洞F修复：NULL embedding 节点导致 FAISS 与其他引擎计数不一致，需明确警告
        if null_emb_count > 0:
            print(f"  >> [海马体] WARNING: {null_emb_count} 个节点的 embedding 为 NULL，"
                  f"这些节点会进入 NetworkX/TransE 但不进入 FAISS，四引擎计数不一致！"
                  f"请检查是否存在遗留的脏数据或老版本数据。")

        edge_count_sqlite = c.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        for row in c.execute("SELECT src, dst, type, weight FROM edges"):
            src, dst, etype, weight = row
            if src in self.graph and dst in self.graph:
                self.graph.add_edge(src, dst, key=etype, type=etype, weight=weight)
                # 同步恢复关系嵌入名单（与实体同理，权重由 checkpoint 负责）
                self.kge._ensure_relation(etype)

        # 构建 FAISS 索引（启动恢复时全量构建，运行期增量更新）
        self._init_faiss_index()

        if node_count > 0:
            print(f"  >> [海马体]: 恢复 {node_count} 节点, {edge_count_sqlite} 突触 (FAISS 索引就绪)。")

    def _init_faiss_index(self):
        """
        v5.5: 完整重建 FAISS 索引（仅在启动恢复时调用）。
        运行期间使用 _add_to_faiss / _remove_from_faiss 做增量更新。
        """
        self._faiss_base = faiss.IndexFlatIP(self._faiss_dim)
        self._faiss_index = faiss.IndexIDMap2(self._faiss_base)
        self._faiss_next_id = 0
        self._faiss_nid_to_int = {}
        self._faiss_int_to_nid = {}
        if not self.node_embeddings:
            return
        ids = list(self.node_embeddings.keys())
        embs = np.stack([self.node_embeddings[nid] for nid in ids]).astype(np.float32)
        # L2 归一化后用 inner product = cosine similarity
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embs_normed = embs / norms
        # 分配连续整数 ID
        int_ids = np.arange(len(ids), dtype=np.int64)
        self._faiss_index.add_with_ids(embs_normed, int_ids)
        for idx, nid in enumerate(ids):
            self._faiss_nid_to_int[nid] = idx
            self._faiss_int_to_nid[idx] = nid
        self._faiss_next_id = len(ids)

    def _add_to_faiss(self, nid: str, emb_np: np.ndarray):
        """
        v5.5: 增量添加单个向量到 FAISS 索引（使用 add_with_ids）。
        """
        norm = np.linalg.norm(emb_np)
        if norm < 1e-8:
            return
        emb_normed = (emb_np / norm).reshape(1, -1).astype(np.float32)
        fid = self._faiss_next_id
        self._faiss_next_id += 1
        self._faiss_index.add_with_ids(emb_normed, np.array([fid], dtype=np.int64))
        self._faiss_nid_to_int[nid] = fid
        self._faiss_int_to_nid[fid] = nid

    def _remove_from_faiss(self, nid: str):
        """
        v5.5 #5: 从 FAISS 索引中局部删除单个节点。
        使用 IndexIDMap2 的 remove_ids 能力，无需全量重建。
        """
        if nid in self._faiss_nid_to_int:
            fid = self._faiss_nid_to_int[nid]
            self._faiss_index.remove_ids(np.array([fid], dtype=np.int64))
            del self._faiss_nid_to_int[nid]
            if fid in self._faiss_int_to_nid:
                del self._faiss_int_to_nid[fid]

    def _update_faiss_vector(self, nid: str, new_emb_np: np.ndarray):
        """
        v5.5: 原子更新 FAISS 中某节点的向量（先删后加）。
        用于 merge_similar 中幸存节点的 embedding 更新。
        """
        self._remove_from_faiss(nid)
        self._add_to_faiss(nid, new_emb_np)

    # ---- 记忆写入 (多层表示) ----

    def memorize(self, text: str, embedding: torch.Tensor,
                 linked_from: str = None, edge_type: str = None) -> str:
        """
        写入一条记忆（多层表示：原文 + embedding + 时间戳 + 可选边）。

        BUG-1 修复：使用 with self.conn: 显式事务保证 SQLite 原子性。
        操作顺序：先写内存引擎（NetworkX/FAISS/TransE），再在事务块中写 SQLite。
        若 SQLite 事务失败，立即回滚内存引擎，确保四引擎始终一致。
        """
        nid = str(uuid.uuid4())[:8]
        ts = time.time()
        emb_np = embedding.detach().cpu().to(torch.float32).numpy().flatten()

        # 第一步：写内存引擎（NetworkX + FAISS + TransE）
        # 内存操作不跨越事务边界，执行速度极快，几乎不会失败
        self.graph.add_node(nid, text=text)
        self.node_embeddings[nid] = emb_np
        self.kge._ensure_entity(nid)
        self._add_to_faiss(nid, emb_np)

        # 第二步：在显式事务块中原子写入 SQLite
        # with self.conn: 保证成功时自动 commit，任何异常自动 rollback
        try:
            with self.conn:
                emb_blob = emb_np.tobytes()
                self.conn.execute(
                    "INSERT OR REPLACE INTO nodes (id, text, embedding, created_at) VALUES (?,?,?,?)",
                    (nid, text, emb_blob, ts))
                if linked_from and edge_type and linked_from in self.graph:
                    self._add_edge_sql(linked_from, nid, edge_type)
                self._auto_build_edges_sql(nid, emb_np, ts)
        except Exception:
            # SQLite 已自动 rollback；同步回滚内存引擎，恢复四引擎一致性
            if nid in self.graph:
                self.graph.remove_node(nid)
            self.node_embeddings.pop(nid, None)
            self.kge._remove_entity(nid)
            self._remove_from_faiss(nid)
            raise

        # 第三步：SQLite 写入成功后，同步写内存引擎的边（NetworkX + TransE）
        # 此时 SQL 已落盘，内存边不一致只影响本次运行，重启后可从 SQLite 恢复
        if linked_from and edge_type and linked_from in self.graph:
            self._add_edge_memory(linked_from, nid, edge_type)
        self._auto_build_edges_memory(nid, emb_np, ts)

        return nid

    def _add_edge_sql(self, src: str, dst: str, etype: str, weight: float = 1.0):
        """
        仅写 SQLite 的边（在事务块内调用）。

        BUG-3 修复：使用 INSERT OR IGNORE 替代 INSERT OR REPLACE，
        防止边迁移时静默覆盖原有边权重。若边已存在且新权重更高，则用 UPDATE 更新。
        """
        ts = time.time()
        # 先尝试插入（已存在则忽略）
        self.conn.execute(
            "INSERT OR IGNORE INTO edges (src, dst, type, weight, created_at) VALUES (?,?,?,?,?)",
            (src, dst, etype, weight, ts))
        # 若已存在且新权重更大，则更新（保留最高权重，让强语义连接不被弱连接覆盖）
        self.conn.execute(
            "UPDATE edges SET weight = ?, created_at = ? "
            "WHERE src = ? AND dst = ? AND type = ? AND weight < ?",
            (weight, ts, src, dst, etype, weight))

    def _add_edge_memory(self, src: str, dst: str, etype: str, weight: float = 1.0):
        """仅写内存引擎（NetworkX + TransE）的边。"""
        # NetworkX MultiDiGraph：若同类边已存在则自然覆盖（取新权重）
        # 此处与 SQL 的 max-weight 策略保持一致：只在新权重更大时才更新
        existing_weight = 0.0
        if self.graph.has_edge(src, dst, key=etype):
            existing_weight = self.graph[src][dst][etype].get('weight', 0.0)
        if weight >= existing_weight:
            self.graph.add_edge(src, dst, key=etype, type=etype, weight=weight)
        self.kge._ensure_entity(src)
        self.kge._ensure_entity(dst)
        self.kge._ensure_relation(etype)

    def _add_edge(self, src: str, dst: str, etype: str, weight: float = 1.0):
        """
        原子操作：同时写 SQLite + NetworkX + TransE。
        用于单次独立的边写入（非事务块内）。

        漏洞A/F修复：SQL 操作用 with self.conn: 事务包裹。
        若 SQL 失败则自动 rollback，内存引擎不执行写入，确保两者一致。
        """
        with self.conn:
            self._add_edge_sql(src, dst, etype, weight)
        # SQL 成功提交后，再写内存引擎（内存操作失败只影响本次运行，重启可从 SQLite 恢复）
        self._add_edge_memory(src, dst, etype, weight)

    def _update_node_in_sqlite(self, nid: str, text: str, emb_np: np.ndarray):
        """同步更新 SQLite 中的节点数据"""
        emb_blob = emb_np.astype(np.float32).tobytes()
        self.conn.execute(
            "UPDATE nodes SET text = ?, embedding = ? WHERE id = ?",
            (text, emb_blob, nid))

    def _delete_node_from_sqlite(self, nid: str):
        """从 SQLite 中删除节点及其关联边"""
        self.conn.execute("DELETE FROM edges WHERE src = ? OR dst = ?", (nid, nid))
        self.conn.execute("DELETE FROM nodes WHERE id = ?", (nid,))

    def _auto_build_edges_sql(self, new_id: str, new_emb: np.ndarray, new_ts: float):
        """
        BUG-1 修复辅助：仅在 SQLite 事务块内调用，写边的 SQL 部分。
        注意：此时新节点已在 SQLite 中（同一事务内），可以安全查询。
        """
        c = self.conn.cursor()

        # 1. 时间临近边 SQL（v5.6 P1-3: 跳过时间差异过小的记忆）
        for row in c.execute(
            "SELECT id, created_at FROM nodes WHERE id != ? AND abs(created_at - ?) < ? "
            "ORDER BY created_at DESC LIMIT 3",
            (new_id, new_ts, config.temporal_window_sec)
        ):
            neighbor_id, neighbor_ts = row[0], row[1]
            delta_t = abs(new_ts - neighbor_ts)
            if delta_t < config.temporal_min_gap_sec:
                continue
            weight = 0.5 * (1.0 - delta_t / config.temporal_window_sec)
            self._add_edge_sql(new_id, neighbor_id, "时间临近", weight=max(weight, 0.05))

        # 2. 语义相似边 SQL（依赖 FAISS 检索结果，在建边前 FAISS 已有 new_id）
        new_norm = np.linalg.norm(new_emb)
        if new_norm < 1e-8 or self._faiss_index.ntotal < 2:
            return
        search_k = min(10, self._faiss_index.ntotal)
        query_normed = (new_emb / new_norm).reshape(1, -1).astype(np.float32)
        sims, indices = self._faiss_index.search(query_normed, search_k)
        for rank in range(search_k):
            fid = int(indices[0, rank])
            sim = float(sims[0, rank])
            if fid < 0 or fid not in self._faiss_int_to_nid:
                continue
            other_id = self._faiss_int_to_nid[fid]
            if other_id == new_id:
                continue
            if sim > config.semantic_sim_threshold:
                self._add_edge_sql(new_id, other_id, "语义相似", weight=sim)

    def _auto_build_edges_memory(self, new_id: str, new_emb: np.ndarray, new_ts: float):
        """
        BUG-1 修复辅助：在 SQLite 事务成功提交后，同步写内存边（NetworkX + TransE）。
        """
        c = self.conn.cursor()

        # 时间临近边（从 SQLite 读取邻居，保持与 SQL 侧完全一致的逻辑）
        for row in c.execute(
            "SELECT id, created_at FROM nodes WHERE id != ? AND abs(created_at - ?) < ? "
            "ORDER BY created_at DESC LIMIT 3",
            (new_id, new_ts, config.temporal_window_sec)
        ):
            neighbor_id, neighbor_ts = row[0], row[1]
            delta_t = abs(new_ts - neighbor_ts)
            if delta_t < config.temporal_min_gap_sec:
                continue
            weight = 0.5 * (1.0 - delta_t / config.temporal_window_sec)
            self._add_edge_memory(new_id, neighbor_id, "时间临近", weight=max(weight, 0.05))

        # 语义相似边
        new_norm = np.linalg.norm(new_emb)
        if new_norm < 1e-8 or self._faiss_index.ntotal < 2:
            return
        search_k = min(10, self._faiss_index.ntotal)
        query_normed = (new_emb / new_norm).reshape(1, -1).astype(np.float32)
        sims, indices = self._faiss_index.search(query_normed, search_k)
        for rank in range(search_k):
            fid = int(indices[0, rank])
            sim = float(sims[0, rank])
            if fid < 0 or fid not in self._faiss_int_to_nid:
                continue
            other_id = self._faiss_int_to_nid[fid]
            if other_id == new_id:
                continue
            if sim > config.semantic_sim_threshold:
                self._add_edge_memory(new_id, other_id, "语义相似", weight=sim)

    def _auto_build_edges(self, new_id: str, new_emb: np.ndarray, new_ts: float):
        """
        向后兼容接口：同时写 SQL + 内存（用于 extract_patterns 等非事务路径）。
        v5.3: FAISS 加速的自动建边。
        """
        c = self.conn.cursor()

        # 1. 时间临近边（v5.6 P1-3: 跳过时间差异过小的记忆，防止 batch 导入时边爆炸）
        for row in c.execute(
            "SELECT id, created_at FROM nodes WHERE id != ? AND abs(created_at - ?) < ? ORDER BY created_at DESC LIMIT 3",
            (new_id, new_ts, config.temporal_window_sec)
        ):
            neighbor_id, neighbor_ts = row[0], row[1]
            delta_t = abs(new_ts - neighbor_ts)
            # v5.6 P1-3: 时间间隔太小说明是 batch 导入，无时序信息量，跳过
            if delta_t < config.temporal_min_gap_sec:
                continue
            if not self.graph.has_edge(new_id, neighbor_id, key="时间临近"):
                # 权重按时间距离衰减（越近权重越高）
                weight = 0.5 * (1.0 - delta_t / config.temporal_window_sec)
                self._add_edge(new_id, neighbor_id, "时间临近", weight=max(weight, 0.05))

        # 2. 语义相似边：FAISS top-k 近邻检索（替代 for 循环全量扫描）
        new_norm = np.linalg.norm(new_emb)
        if new_norm < 1e-8 or self._faiss_index.ntotal < 2:
            return

        # 查询比实际需要多一些（排除自身 + 可能已有边的节点）
        search_k = min(10, self._faiss_index.ntotal)
        query_normed = (new_emb / new_norm).reshape(1, -1).astype(np.float32)
        sims, indices = self._faiss_index.search(query_normed, search_k)

        for rank in range(search_k):
            fid = int(indices[0, rank])
            sim = float(sims[0, rank])
            if fid < 0 or fid not in self._faiss_int_to_nid:
                continue
            other_id = self._faiss_int_to_nid[fid]
            if other_id == new_id:
                continue
            if sim > config.semantic_sim_threshold:
                if not self.graph.has_edge(new_id, other_id, key="语义相似"):
                    self._add_edge(new_id, other_id, "语义相似", weight=sim)

    def memorize_dialogue(self, user_text: str, ai_text: str, cortex,
                          user_emb: 'torch.Tensor' = None) -> list:
        """对话记忆：两条记忆 + 对话共现边。可传入已有 user_emb 避免重复 LLM forward。"""
        if not config.memory_auto_extract:
            return []
        u_emb = user_emb if user_emb is not None else cortex.get_real_embedding(user_text)
        a_emb = cortex.get_real_embedding(ai_text)
        uid = self.memorize(f"[用户] {user_text}", u_emb)
        aid = self.memorize(f"[回应] {ai_text}", a_emb)
        # 漏洞E修复：_add_edge 内部已包含 with self.conn: 事务，无需额外 commit。
        # 原来末尾的 self.conn.commit() 与 memorize 内部事务重叠，已删除。
        self._add_edge(uid, aid, "对话共现")
        return [uid, aid]

    # ---- 记忆合并 ----

    def merge_similar(self, get_embedding_fn=None, summarize_fn=None) -> int:
        """
        合并语义高度相似的记忆节点。

        v5.7 漏洞B/C/D 修复（SQLite优先原子事务模式）：
        每对节点的合并被拆分为两个严格顺序的阶段：
          阶段1：在单个 with self.conn: 块内原子提交所有 SQLite 操作
                 （节点更新、边迁移、节点删除）
          阶段2：SQLite 成功提交后，再同步更新内存引擎（NetworkX/FAISS/TransE）

        这确保：若进程在任何时刻崩溃，SQLite 要么完整提交要么完整回滚，
        而内存引擎只在 SQLite 确认落盘后才被修改，重启后可从 SQLite 完整恢复。

        BUG-2/3/4 原有修复全部保留。
        """
        merged = 0
        nodes = list(self.node_embeddings.keys())
        if len(nodes) < 2:
            return 0

        emb_matrix = np.stack([self.node_embeddings[n] for n in nodes])
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        valid_mask = (norms.squeeze() > 1e-8)
        if valid_mask.sum() < 2:
            return 0
        normed = emb_matrix / np.maximum(norms, 1e-8)
        sim_matrix = normed @ normed.T

        i_indices, j_indices = np.triu_indices(len(nodes), k=1)
        sims = sim_matrix[i_indices, j_indices]
        above_threshold = sims > config.deep_dream_merge_threshold
        to_merge = []
        for idx in np.where(above_threshold)[0]:
            i, j = int(i_indices[idx]), int(j_indices[idx])
            if valid_mask[i] and valid_mask[j]:
                to_merge.append((nodes[i], nodes[j], float(sims[idx])))

        merged_nodes = set()
        for n1, n2, sim in sorted(to_merge, key=lambda x: -x[2]):
            if n1 in merged_nodes or n2 in merged_nodes:
                continue
            if n1 not in self.graph or n2 not in self.graph:
                continue

            t1 = self.graph.nodes[n1].get('text', '')
            t2 = self.graph.nodes[n2].get('text', '')
            if summarize_fn:
                abstract_text = summarize_fn(t1, t2)
            else:
                abstract_text = f"[抽象] {t1[:50]}... + {t2[:50]}..."

            merged_emb = (self.node_embeddings[n1] + self.node_embeddings[n2]) / 2.0

            # 预取边迁移数据（在修改任何引擎之前），确保边读取完整
            # n2 的出边（排除指向 n1 的，避免建自环）
            # 补丁B：只迁移有 'type' 属性的边，防止 MultiDiGraph key（整数）被误当 etype 写入 SQLite
            n2_out_edges = [
                (neighbor, data['type'], data.get('weight', 1.0))
                for _, neighbor, key, data in self.graph.out_edges(n2, data=True, keys=True)
                if neighbor != n1 and 'type' in data
            ]
            # n2 的入边（排除来自 n1 的）
            n2_in_edges = [
                (predecessor, data['type'], data.get('weight', 1.0))
                for predecessor, _, key, data in self.graph.in_edges(n2, data=True, keys=True)
                if predecessor != n1 and 'type' in data
            ]

            # ================================================================
            # 阶段1：原子 SQLite 事务（漏洞B/C/D 核心修复）
            # with self.conn: 保证本块内所有 SQL 操作要么全部提交，要么全部回滚。
            # 若此块抛出任何异常，SQLite 数据完整回滚，内存引擎不会被修改。
            # ================================================================
            try:
                with self.conn:
                    # 1a. 更新幸存节点 n1
                    self._update_node_in_sqlite(n1, abstract_text, merged_emb)
                    # 1b. 迁移 n2 的出边到 n1（SQL 侧，使用 max-weight 策略）
                    for neighbor, etype, weight in n2_out_edges:
                        self._add_edge_sql(n1, neighbor, etype, weight)
                    # 1c. 迁移 n2 的入边到 n1（SQL 侧）
                    for predecessor, etype, weight in n2_in_edges:
                        self._add_edge_sql(predecessor, n1, etype, weight)
                    # 1d. 删除 n2 的所有 SQLite 记录（节点 + 所有关联边）
                    self._delete_node_from_sqlite(n2)
            except Exception as e:
                # SQLite 已自动 rollback，内存引擎未修改，跳过此对节点的合并
                print(f"  >> [海马体] WARNING: 合并 ({n1}, {n2}) 时 SQLite 事务失败，已回滚: {e}")
                continue

            # ================================================================
            # 阶段2：SQLite 成功落盘后，同步更新内存引擎
            # 此时即使进程崩溃，重启后可从 SQLite 完整恢复，内存引擎的临时不一致可接受。
            # ================================================================
            # 2a. 更新幸存节点 n1 的内存状态
            self.graph.nodes[n1]['text'] = abstract_text
            self.graph.nodes[n1]['abstract'] = True
            self.node_embeddings[n1] = merged_emb
            # BUG-2 修复：node_embeddings[n1] 已更新，FAISS 立即同步，消除不一致窗口
            self._update_faiss_vector(n1, merged_emb)

            # 2b. 迁移边到内存引擎（NetworkX + TransE，使用 max-weight 策略）
            for neighbor, etype, weight in n2_out_edges:
                self._add_edge_memory(n1, neighbor, etype, weight)
            for predecessor, etype, weight in n2_in_edges:
                self._add_edge_memory(predecessor, n1, etype, weight)

            # 2c. 从内存引擎中彻底清除 n2
            # 顺序：先清 FAISS（需要 nid 映射），再从图中移除，再清 node_embeddings，最后清 TransE
            self._remove_from_faiss(n2)
            self.graph.remove_node(n2)    # 同时移除所有 n2 的出入边（NetworkX 自动）
            del self.node_embeddings[n2]
            self.kge._remove_entity(n2)   # v5.4 P1-5: 清理 TransE 实体幽灵参数

            merged_nodes.add(n2)
            merged += 1

        # BUG-4 修复：合并完成后，清理所有已无边引用的孤立关系嵌入
        orphan_count = self.kge._cleanup_orphan_relations(self.graph)
        if orphan_count > 0:
            print(f"  >> [海马体]: 清理 {orphan_count} 个孤立关系嵌入（防幽灵参数）。")

        return merged

    # ---- 模式提精 ----

    def extract_patterns(self, get_embedding_fn, summarize_fn=None) -> list:
        """
        从图谱中识别重复出现的边类型模式，写入为抽象节点。

        v5.4 P2-3: 限制遍历范围，对大图谱只采样部分节点进行模式发现，
        避免 O(N×E²) 的全遍历性能瓶颈。
        """
        patterns = []
        path_counts = {}

        # P2-3: 大图谱采样——超过 500 节点时只采样部分节点
        all_nodes = list(self.graph.nodes)
        max_sample = 500
        if len(all_nodes) > max_sample:
            import random
            sample_nodes = random.sample(all_nodes, max_sample)
        else:
            sample_nodes = all_nodes

        for node in sample_nodes:
            for n1 in self.graph.neighbors(node):
                for key1, data1 in self.graph[node][n1].items():
                    e1 = data1.get('type', key1)
                    for n2 in self.graph.neighbors(n1):
                        if n2 == node:
                            continue
                        for key2, data2 in self.graph[n1][n2].items():
                            e2 = data2.get('type', key2)
                            pattern_key = f"{e1}->{e2}"
                            if pattern_key not in path_counts:
                                path_counts[pattern_key] = []
                            path_counts[pattern_key].append((node, n1, n2))

        for pattern_key, instances in path_counts.items():
            if len(instances) >= config.deep_dream_pattern_min_count:
                n0, n1, n2 = instances[0]
                t0 = self.graph.nodes[n0].get('text', '')[:20]
                t1 = self.graph.nodes[n1].get('text', '')[:20]
                t2 = self.graph.nodes[n2].get('text', '')[:20]
                if summarize_fn:
                    pattern_text = summarize_fn(
                        f"路径模式 {pattern_key}:",
                        f"{t0} → {t1} → {t2}，出现{len(instances)}次")
                else:
                    pattern_text = (f"[模式] {pattern_key} "
                                    f"(例:{t0}..→{t1}..→{t2}..)"
                                    f" 出现{len(instances)}次")
                patterns.append({
                    "key": pattern_key, "count": len(instances),
                    "text": pattern_text, "instances": instances})

        for p in patterns:
            emb = get_embedding_fn(p["text"])
            pid = self.memorize(p["text"], emb)
            self.graph.nodes[pid]['pattern'] = True
            for n0, n1, n2 in p["instances"][:5]:
                if n0 in self.graph:
                    self._add_edge(pid, n0, "模式实例")

        # 补丁A：删除裸 commit（memorize/_add_edge 均已自管理事务，此处 commit 为 no-op 且违反设计纪律）
        return patterns

    # ---- 记忆检索 ----

    def find_seeds(self, query_emb: torch.Tensor, top_k: int = 3) -> list:
        """
        v5.3: FAISS + TransE 投影的真正融合检索。

        score = (1 - α) * cosine_sim(FAISS) + α * cosine_sim(proj(query), transe_emb)
        α = config.kge_score_weight
        """
        if self._faiss_index.ntotal == 0:
            return []

        q = query_emb.detach().cpu().to(torch.float32).numpy().flatten()
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return []

        # FAISS 检索：top-K 余弦相似度（已 L2 归一化）
        search_k = min(top_k * 3, self._faiss_index.ntotal)
        q_normed = (q / q_norm).reshape(1, -1).astype(np.float32)
        cos_sims, indices = self._faiss_index.search(q_normed, search_k)

        # TransE 投影：将查询映射到 KGE 空间
        alpha = config.kge_score_weight
        q_kge = None
        if alpha > 0 and self.kge._proj_trained:
            q_tensor = torch.from_numpy(q).to(torch.float32).to(config.device)
            q_kge = self.kge.project_query(q_tensor).detach().cpu().numpy()
            q_kge_norm = np.linalg.norm(q_kge)
            if q_kge_norm < 1e-8:
                q_kge = None

        # 融合评分（v5.5: 使用 int→nid 映射替代数组索引）
        scores = []
        for rank in range(search_k):
            fid = int(indices[0, rank])
            if fid < 0 or fid not in self._faiss_int_to_nid:
                continue
            nid = self._faiss_int_to_nid[fid]
            cos_sim = float(cos_sims[0, rank])

            # TransE 投影余弦相似度（真正的语义融合）
            kge_sim = 0.0
            if q_kge is not None and alpha > 0:
                kge_emb = self.kge.get_entity_embedding(nid).detach().cpu().numpy()
                kge_norm = np.linalg.norm(kge_emb)
                if kge_norm > 1e-8:
                    kge_sim = float(np.dot(q_kge, kge_emb) / (np.linalg.norm(q_kge) * kge_norm))

            score = (1.0 - alpha) * cos_sim + alpha * kge_sim
            scores.append((nid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scores[:top_k]]

    def find_outcome_nodes(self, seed_nodes: list, max_hops: int = 2) -> list:
        """
        BFS 搜索"结局"节点（终端性：入度>=2 且出度<=1 / abstract / pattern）。

        v5.4 P1-6: 优雅降级——当找不到符合严格条件的终端节点时，
        返回 BFS 范围内拓扑得分最高的非种子节点（按 in_degree/(out_degree+1) 排序），
        确保在小图谱早期阶段也能提供有意义的 goal_emb。
        """
        strict_outcomes = []
        all_visited_non_seed = []  # 降级候选池
        visited = set(seed_nodes)
        frontier = deque((s, 0) for s in seed_nodes if s in self.graph)

        while frontier:
            node, depth = frontier.popleft()  # v5.6 P1-5: O(1) deque.popleft() 替代 O(N) list.pop(0)
            if depth > max_hops:
                continue
            for neighbor in self.graph.neighbors(node):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                node_data = self.graph.nodes.get(neighbor, {})
                out_degree = self.graph.out_degree(neighbor)
                in_degree = self.graph.in_degree(neighbor)
                # v5.5 #2: 拓扑判定阈值走 config（消灭硬编码）
                is_terminal = (out_degree <= config.outcome_max_out_degree
                               and in_degree >= config.outcome_min_in_degree)
                is_abstract = node_data.get('abstract', False)
                is_pattern = node_data.get('pattern', False)

                # 收集所有非种子的已访问节点作为降级候选
                topo_score = in_degree / (out_degree + 1.0)
                all_visited_non_seed.append((neighbor, node_data.get('text', ''),
                                             depth + 1, topo_score))

                if is_abstract or is_pattern or is_terminal:
                    strict_outcomes.append((neighbor, node_data.get('text', ''), depth + 1))

                if depth + 1 < max_hops:
                    frontier.append((neighbor, depth + 1))

        # 严格模式有结果则返回
        if strict_outcomes:
            strict_outcomes.sort(key=lambda x: x[2])
            return strict_outcomes

        # P1-6 优雅降级：返回拓扑得分最高的非种子节点
        if all_visited_non_seed:
            all_visited_non_seed.sort(key=lambda x: -x[3])
            # 只返回前 3 个（避免返回过多弱候选）
            return [(nid, text, hops) for nid, text, hops, _ in all_visited_non_seed[:3]]

        return []

    def retrieve(self, query_emb: torch.Tensor) -> list:
        """完整检索管线：FAISS 种子 + PPR 联想。返回结果包含 embedding。"""
        if self.graph.number_of_nodes() == 0:
            return []
        seeds = self.find_seeds(query_emb, config.memory_top_k)
        if not seeds:
            return []
        results = []
        for s in seeds:
            results.append({
                "id": s, "text": self.graph.nodes[s].get('text', ''),
                "type": "种子", "ppr_score": 0.0,
                "embedding": self.node_embeddings.get(s)
            })
        ppr = personalized_pagerank(self.graph, seeds)
        for r in ppr:
            r["type"] = "PPR联想"
            r["embedding"] = self.node_embeddings.get(r["id"])
            results.append(r)
        return results

    # ---- KGE 巩固（v5.4: 返回 loss，不内建优化器） ----

    def compute_consolidation_losses(self) -> tuple:
        """
        v5.4 P0-2: 计算记忆巩固的损失（TransE + 投影层），返回 loss 元组。
        由 brain.py 的统一优化器驱动反向传播，不再内部 .backward()。

        返回: (kge_loss, proj_loss) 两个 tensor
        """
        kge_loss = self.kge.compute_consolidation_loss(self.graph)
        proj_loss = self.kge.compute_projection_loss(self.node_embeddings)
        return kge_loss, proj_loss

    # ---- 状态报告 ----

    def stats(self) -> str:
        node_count = self.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edge_count = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        faiss_count = self._faiss_index.ntotal
        # 补丁D：监控 FAISS 映射表与索引的一致性（漏洞#6：幽灵向量检测）
        faiss_mapped = len(self._faiss_nid_to_int)
        if faiss_mapped != faiss_count:
            print(f"  >> [海马体] WARNING: FAISS映射表({faiss_mapped}) != "
                  f"FAISS索引({faiss_count})，存在幽灵向量，建议重启重建索引。")
        return (f"{node_count} 节点, {edge_count} 突触, "
                f"FAISS:{faiss_count} 向量 "
                f"(SQLite: {config.memory_db_path})")

    def close(self):
        self.conn.close()
