"""
logic/clutrr_loader.py —— CLUTRR 标准基准数据加载器

将 CLUTRR 的 parquet 数据转换为实体无关的内部格式。
关键设计: 不使用实体名称，只使用图中的位置索引和关系类型。
"""
import ast
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter

# CLUTRR 边类型 (输入关系) — 14 种
EDGE_TYPES = [
    'aunt', 'brother', 'daughter', 'father',
    'granddaughter', 'grandfather', 'grandmother', 'grandson',
    'husband', 'mother', 'sister', 'son', 'uncle', 'wife'
]
EDGE_TO_ID = {e: i for i, e in enumerate(EDGE_TYPES)}

# CLUTRR 目标关系 (输出分类) — 21 种
TARGET_RELATIONS = [
    'aunt', 'son-in-law', 'grandfather', 'brother', 'sister',
    'father', 'mother', 'grandmother', 'uncle', 'daughter-in-law',
    'grandson', 'granddaughter', 'father-in-law', 'mother-in-law',
    'nephew', 'son', 'daughter', 'niece', 'husband', 'wife',
    'sister-in-law'
]
NUM_EDGE_TYPES = len(EDGE_TYPES)
NUM_TARGET_CLASSES = len(TARGET_RELATIONS)


def parse_chain_length(task_name: str) -> int:
    """从 task_name 中提取推理链长度: 'task_1.3' → 3"""
    return int(task_name.split('.')[-1])


def load_clutrr(split: str = 'train',
                data_dir: str = '/app/logic/clutrr_data') -> List[Dict]:
    """
    加载 CLUTRR 数据并转换为内部格式。

    内部格式:
    {
        "facts": [(src_pos, rel_id, dst_pos), ...],  # 事实三元组 (位置索引, 关系ID, 位置索引)
        "query_src": int,          # 查询源位置
        "query_dst": int,          # 查询目标位置
        "target": int,             # 目标关系类别 (0-20)
        "chain_length": int,       # 推理链长度
        "edge_types": List[str],   # 原始关系类型 (用于调试)
    }
    """
    path = f'{data_dir}/{split}.parquet'
    df = pd.read_parquet(path)
    samples = []

    for _, row in df.iterrows():
        # 解析图结构
        story_edges = ast.literal_eval(row['story_edges'])  # [(0,1), (1,2)]
        edge_types = ast.literal_eval(row['edge_types'])    # ['daughter', 'brother']
        query_edge = ast.literal_eval(row['query_edge'])    # (0, 2)
        target = int(row['target'])
        chain_length = parse_chain_length(row['task_name'])

        # 转换为内部格式: (src_pos, rel_id, dst_pos)
        facts = []
        for (src, dst), rel_name in zip(story_edges, edge_types):
            rel_id = EDGE_TO_ID.get(rel_name, -1)
            if rel_id == -1:
                continue  # 跳过未知关系
            facts.append((src, rel_id, dst))

        samples.append({
            'facts': facts,
            'query_src': query_edge[0],
            'query_dst': query_edge[1],
            'target': target,
            'chain_length': chain_length,
            'edge_types': edge_types,
        })

    return samples


def print_clutrr_stats(samples: List[Dict], name: str = 'Dataset'):
    """打印 CLUTRR 数据集统计"""
    print(f"\n{'='*50}")
    print(f"  CLUTRR {name} 统计")
    print(f"{'='*50}")
    print(f"  总数: {len(samples)}")

    # 按推理链长度分布
    chain_counts = Counter(s['chain_length'] for s in samples)
    print(f"  推理链长度分布:")
    for k in sorted(chain_counts.keys()):
        print(f"    k={k}: {chain_counts[k]} 题")

    # 目标关系分布
    target_counts = Counter(s['target'] for s in samples)
    print(f"  目标关系种类: {len(target_counts)}")


if __name__ == '__main__':
    train = load_clutrr('train')
    test = load_clutrr('test')
    print_clutrr_stats(train, 'Train')
    print_clutrr_stats(test, 'Test')
    print(f"\n  样本示例:")
    s = train[0]
    print(f"    facts: {s['facts']}")
    print(f"    query: ({s['query_src']}, ?, {s['query_dst']})")
    print(f"    target: {s['target']} ({TARGET_RELATIONS[s['target']]})")
    print(f"    chain: {s['chain_length']}")
    print(f"    edges: {s['edge_types']}")
