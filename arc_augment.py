"""
arc_augment.py —— ARC 数据增强模块

对 ARC 任务实施几何变换和颜色置换，同步应用于同一道题的所有 grid
（train pairs + test pair），保持变换规律不变。

增强策略（确定性 + 随机混合）：
    - 旋转 90°/180°/270°（×3）
    - 水平翻转 / 垂直翻转（×2）
    - 颜色置换：随机重排非背景色 1-9 的映射（×N，默认2）
    
    总增强：1(原始) + 3(旋转) + 2(翻转) + N(颜色) = 8× (N=2时)
    400 题 → 3200 题

核心约束：
    同一道题的所有 grid 必须使用完全相同的变换参数，
    否则输入→输出的变换规律会被破坏。
"""

import copy
import random
from typing import Callable


# ──────────────────────────────────────────────────────────────────────────────
#  原子变换函数（操作单个 grid: list[list[int]]）
# ──────────────────────────────────────────────────────────────────────────────

def rotate_90(grid: list[list[int]]) -> list[list[int]]:
    """顺时针旋转 90°：(H, W) → (W, H)"""
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    return [[grid[H - 1 - r][c] for r in range(H)] for c in range(W)]


def rotate_180(grid: list[list[int]]) -> list[list[int]]:
    """旋转 180°：行倒序 + 每行倒序"""
    return [row[::-1] for row in grid[::-1]]


def rotate_270(grid: list[list[int]]) -> list[list[int]]:
    """顺时针旋转 270°（= 逆时针 90°）：(H, W) → (W, H)"""
    H = len(grid)
    W = len(grid[0]) if H > 0 else 0
    return [[grid[r][W - 1 - c] for r in range(H)] for c in range(W)]


def flip_horizontal(grid: list[list[int]]) -> list[list[int]]:
    """水平翻转（左右镜像）"""
    return [row[::-1] for row in grid]


def flip_vertical(grid: list[list[int]]) -> list[list[int]]:
    """垂直翻转（上下镜像）"""
    return grid[::-1]


def apply_color_permutation(grid: list[list[int]], perm: dict[int, int]) -> list[list[int]]:
    """
    应用颜色置换映射。
    perm: {原始颜色: 新颜色}，背景色 0 始终映射到 0。
    """
    return [[perm.get(cell, cell) for cell in row] for row in grid]


# ──────────────────────────────────────────────────────────────────────────────
#  颜色置换生成器
# ──────────────────────────────────────────────────────────────────────────────

def generate_color_permutation(seed: int) -> dict[int, int]:
    """
    生成一个非背景色（1-9）的随机置换映射。
    背景色 0 始终保持不变（0→0）。
    使用确定性种子保证可复现。
    """
    rng = random.Random(seed)
    colors = list(range(1, 10))  # [1, 2, 3, ..., 9]
    shuffled = colors[:]
    rng.shuffle(shuffled)
    
    # 确保至少有一个颜色发生了变化（避免生成恒等映射）
    while shuffled == colors:
        rng.shuffle(shuffled)
    
    perm = {0: 0}  # 背景色不变
    for original, new_color in zip(colors, shuffled):
        perm[original] = new_color
    return perm


# ──────────────────────────────────────────────────────────────────────────────
#  任务级变换（同步变换一道题的所有 grid）
# ──────────────────────────────────────────────────────────────────────────────

def transform_task(task: dict, grid_fn: Callable) -> dict:
    """
    对一道 ARC 任务的所有 grid 应用同一个变换函数。
    
    保持 task 的结构不变，只修改所有 grid 内容。
    关键：train pairs 和 test pair 的 input/output 都必须用同一个 grid_fn，
    这样输入→输出的变换规律在新坐标系下依然成立。
    """
    new_task = {"_id": task.get("_id", "") + "_aug", "train": [], "test": []}
    
    for pair in task["train"]:
        new_task["train"].append({
            "input": grid_fn(pair["input"]),
            "output": grid_fn(pair["output"]),
        })
    
    for pair in task["test"]:
        new_task["test"].append({
            "input": grid_fn(pair["input"]),
            "output": grid_fn(pair["output"]),
        })
    
    return new_task


def augment_single_task(
    task: dict,
    do_rotations: bool = True,
    do_flips: bool = True,
    num_color_perms: int = 2,
    task_idx: int = 0,
) -> list[dict]:
    """
    对单道 ARC 任务生成所有增强版本。
    
    返回增强后的任务列表（不含原始任务）。
    task_idx 用于颜色置换的确定性种子。
    """
    augmented = []
    task_id = task.get("_id", f"task_{task_idx}")
    
    # 几何变换：旋转
    if do_rotations:
        for angle, fn in [(90, rotate_90), (180, rotate_180), (270, rotate_270)]:
            aug = transform_task(task, fn)
            aug["_id"] = f"{task_id}_rot{angle}"
            augmented.append(aug)
    
    # 几何变换：翻转
    if do_flips:
        for name, fn in [("flipH", flip_horizontal), ("flipV", flip_vertical)]:
            aug = transform_task(task, fn)
            aug["_id"] = f"{task_id}_{name}"
            augmented.append(aug)
    
    # 颜色置换
    for perm_idx in range(num_color_perms):
        # 用 (task_idx, perm_idx) 组合生成确定性种子
        seed = task_idx * 1000 + perm_idx + 42
        perm = generate_color_permutation(seed)
        color_fn = lambda grid, p=perm: apply_color_permutation(grid, p)
        aug = transform_task(task, color_fn)
        aug["_id"] = f"{task_id}_color{perm_idx}"
        augmented.append(aug)
    
    return augmented


def augment_tasks(
    tasks: list[dict],
    do_rotations: bool = True,
    do_flips: bool = True,
    num_color_perms: int = 2,
) -> list[dict]:
    """
    对整个任务列表进行数据增强。
    
    返回：原始任务 + 所有增强任务的合并列表。
    
    扩增倍数 = 1 + 3(旋转) + 2(翻转) + num_color_perms
             = 8× (默认 num_color_perms=2)
    """
    all_tasks = list(tasks)  # 保留原始数据
    
    for idx, task in enumerate(tasks):
        aug_list = augment_single_task(
            task,
            do_rotations=do_rotations,
            do_flips=do_flips,
            num_color_perms=num_color_perms,
            task_idx=idx,
        )
        all_tasks.extend(aug_list)
    
    return all_tasks


# ──────────────────────────────────────────────────────────────────────────────
#  在线数据增强（Online Augmentation）
#  ──────────────────────────────────────────────────────────────────────────────
#  标准 CV/NLP 做法：不预先生成增强副本，而是每次访问时随机变换。
#  优势：(1) 等效于无限增强，每 epoch 看到不同的变体
#        (2) 0 额外内存，题数不变
#        (3) 代码简洁，无需 steps_per_epoch hack
# ──────────────────────────────────────────────────────────────────────────────

# 所有可用的几何变换
_GEOMETRIC_TRANSFORMS = [
    ("rot90", rotate_90),
    ("rot180", rotate_180),
    ("rot270", rotate_270),
    ("flipH", flip_horizontal),
    ("flipV", flip_vertical),
]


def online_augment_task(task: dict) -> dict:
    """
    在线随机增强：每次调用时随机选择一种变换应用到整道题。

    概率分配（经验值，平衡多样性）：
        - 40% 几何变换（5种中随机选1种）
        - 30% 颜色置换（随机排列非背景色）
        - 30% 不变换（保留原始数据，防止模型只学增强后的分布）
    """
    r = random.random()

    if r < 0.4:
        # 几何变换
        _, fn = random.choice(_GEOMETRIC_TRANSFORMS)
        return transform_task(task, fn)
    elif r < 0.7:
        # 颜色置换（完全随机种子，每次不同）
        perm = generate_color_permutation(seed=random.randint(0, 999999))
        color_fn = lambda grid, p=perm: apply_color_permutation(grid, p)
        return transform_task(task, color_fn)
    else:
        # 不变换
        return task


# ──────────────────────────────────────────────────────────────────────────────
#  自测入口
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 构造一个简单的测试任务
    test_task = {
        "_id": "test_001",
        "train": [
            {"input": [[0, 1, 0], [2, 3, 0]], "output": [[1, 0, 1], [0, 3, 2]]},
            {"input": [[0, 0, 1], [1, 0, 0]], "output": [[1, 1, 0], [0, 0, 1]]},
        ],
        "test": [
            {"input": [[1, 2, 3], [0, 0, 0]], "output": [[3, 2, 1], [0, 0, 0]]},
        ],
    }
    
    # 测试单任务增强
    augmented = augment_single_task(test_task, task_idx=0)
    print(f"单任务增强: 1 → {1 + len(augmented)} 个版本")
    for aug in augmented:
        print(f"  {aug['_id']}: train[0].input = {aug['train'][0]['input']}")
    
    # 测试批量增强
    tasks = [test_task]
    all_tasks = augment_tasks(tasks)
    print(f"\n批量增强: {len(tasks)} → {len(all_tasks)} 个任务")
    
    # 验证旋转的正确性
    original = [[1, 2], [3, 4]]
    r90 = rotate_90(original)
    r180 = rotate_180(original)
    r270 = rotate_270(original)
    
    assert r90 == [[3, 1], [4, 2]], f"rotate_90 错误: {r90}"
    assert r180 == [[4, 3], [2, 1]], f"rotate_180 错误: {r180}"
    assert r270 == [[2, 4], [1, 3]], f"rotate_270 错误: {r270}"
    
    # 验证翻转的正确性
    fh = flip_horizontal(original)
    fv = flip_vertical(original)
    assert fh == [[2, 1], [4, 3]], f"flip_horizontal 错误: {fh}"
    assert fv == [[3, 4], [1, 2]], f"flip_vertical 错误: {fv}"
    
    # 验证颜色置换保持 0 不变
    perm = generate_color_permutation(seed=42)
    assert perm[0] == 0, "颜色置换必须保持背景色 0 不变"
    
    print("\n✅ 所有增强变换验证通过")
