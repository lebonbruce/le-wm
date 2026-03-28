"""
hippocampus/pagerank.py —— Personalized PageRank 多跳联想检索

v5.3 P1-4 修复：保持有向图做 PPR，保留因果方向信息。
对悬挂节点（出度=0）做 teleport-to-seed 处理。
"""
import numpy as np
from scipy import sparse
import networkx as nx
from mvp_config import config


def personalized_pagerank(graph: nx.MultiDiGraph, seed_nodes: list,
                          damping=None, max_iter=None, top_k=None) -> list:
    """
    Personalized PageRank 多跳联想检索（有向图版本）。

    v5.3 P1-4 修复：保持有向图方向性，不再 to_undirected()。
    边的方向承载因果/时间/语义信息（"焦虑"→"深呼吸"），必须保留。
    对悬挂节点（出度=0）做 teleport-to-seed 处理。
    """
    damping = damping or config.ppr_damping
    max_iter = max_iter or config.ppr_max_iter
    top_k = top_k or config.ppr_top_k

    nodes = list(graph.nodes)
    if not nodes or not seed_nodes:
        return []
    n = len(nodes)
    idx = {nd: i for i, nd in enumerate(nodes)}
    seeds_valid = [s for s in seed_nodes if s in idx]
    if not seeds_valid:
        return []

    seed_prob = np.zeros(n)
    for s in seeds_valid:
        seed_prob[idx[s]] = 1.0 / len(seeds_valid)

    # ★ P1-4 修复：构建有向转移矩阵（保留因果方向）
    # MultiDiGraph: edges(data=True) 产生 (u, v, data)，遍历所有有向边
    rows, cols, vals = [], [], []
    out_degree = np.zeros(n)
    for u, v, _ in graph.edges(data=True):
        if u in idx and v in idx:
            out_degree[idx[u]] += 1.0

    for u, v, _ in graph.edges(data=True):
        if u in idx and v in idx:
            ui, vi = idx[u], idx[v]
            # 转移概率 = 1 / 出度（均匀分配给所有出边）
            if out_degree[ui] > 0:
                rows.append(vi)  # 目标行
                cols.append(ui)  # 源列
                vals.append(1.0 / out_degree[ui])

    T = sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))

    # 悬挂节点处理：出度=0 的节点 teleport 回 seed（经典 PageRank 做法）
    dangling = (out_degree == 0).astype(np.float64)

    # Power iteration
    rank = seed_prob.copy()
    for _ in range(max_iter):
        # 悬挂节点的概率质量回流到 seed
        dangling_mass = damping * np.dot(dangling, rank)
        new_rank = ((1 - damping) * seed_prob
                    + damping * (T @ rank)
                    + dangling_mass * seed_prob)
        if np.linalg.norm(new_rank - rank) < 1e-8:
            break
        rank = new_rank

    results = []
    for i, nd in enumerate(nodes):
        if nd not in seed_nodes and rank[i] > 1e-10:
            results.append({
                "id": nd, "text": graph.nodes[nd].get('text', ''),
                "ppr_score": float(rank[i])
            })
    results.sort(key=lambda x: x['ppr_score'], reverse=True)
    return results[:top_k]
