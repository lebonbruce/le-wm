"""
arc_trainer.py -- ARC-AGI JEPA v5.0 (Transformation-Centric Architecture)

核心思想：ARC 不是序列预测问题，而是规则归纳问题。
架构把"变换规则"设计为一等公民：

    1. TransformEncoder: 从 (input, output) 对中提取变换表示 delta
       - 可学习的 rule query tokens 通过 cross-attention 提取变换信息
    2. TransformConsensus: 多对 delta 做 self-attention 找共同规则
    3. TransformApplier: 把共识规则 + 测试输入送入 GridDecoder

数据流：
    For each (in_i, out_i) pair:
        in_tokens = GridEncoder(in_i)       # (1, N, D)
        out_tokens = GridEncoder(out_i)     # (1, N', D)
        delta_i = TransformEncoder(in_tokens, out_tokens)  # (1, K, D)

    delta_consensus = ConsensusAttention([delta_1, ..., delta_k])  # (1, K, D)
    test_tokens = GridEncoder(test_input)   # (1, N_test, D)
    context = concat([test_tokens, delta_consensus])
    pred_grid = GridDecoder(context, H, W)
"""

import argparse
import json
import time
import copy
import random
import math
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from mvp_config import config
from jepa_engine.grid_codec import GridEncoder, GridDecoder, CrossAttentionBlock
from jepa_engine.encoder import EncoderTransformerBlock
from arc_augment import online_augment_task, transform_task
from arc_augment import rotate_90, rotate_180, rotate_270
from arc_augment import flip_horizontal, flip_vertical


# ──────────────────────────────────────────────────────────────────────────────
#  ARC Data Loading
# ──────────────────────────────────────────────────────────────────────────────

CACHE_DIR = Path("/app/arc_cache")


def _ensure_cache_dir(folder: str) -> Path:
    cache_path = CACHE_DIR / folder
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def load_arc_tasks(n: int, folder: str = "training") -> list[dict]:
    cache_path = _ensure_cache_dir(folder)
    cached_files = sorted(cache_path.glob("*.json"))
    if len(cached_files) >= n:
        print(f"  Loading {folder} from cache ({n} tasks)...")
        tasks = []
        for f in cached_files[:n]:
            task = json.loads(f.read_text(encoding="utf-8"))
            if "train" in task and "test" in task and task["test"]:
                task["_id"] = f.stem
                tasks.append(task)
        print(f"  Loaded: {len(tasks)} tasks")
        return tasks
    return _download_and_cache(n, folder, cache_path)


def _download_and_cache(n: int, folder: str, cache_path: Path) -> list[dict]:
    api_url = f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{folder}"
    base_url = f"https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/{folder}"
    print(f"  Downloading ARC {folder} ({n} tasks)...")
    json_files = []
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "arc-trainer"})
        resp = urllib.request.urlopen(req, timeout=30)
        file_list = json.loads(resp.read().decode())
        json_files = [f["name"] for f in file_list if f["name"].endswith(".json")][:n]
    except Exception as e:
        print(f"  GitHub API failed: {e}")
        return []
    tasks = []
    for idx, fname in enumerate(json_files):
        if len(tasks) >= n:
            break
        local_file = cache_path / fname
        if local_file.exists():
            task = json.loads(local_file.read_text(encoding="utf-8"))
        else:
            url = f"{base_url}/{fname}"
            req = urllib.request.Request(url, headers={"User-Agent": "arc-trainer"})
            resp = urllib.request.urlopen(req, timeout=30)
            raw = resp.read().decode()
            task = json.loads(raw)
            local_file.write_text(raw, encoding="utf-8")
        if "train" in task and "test" in task and task["test"]:
            task["_id"] = fname.replace(".json", "")
            tasks.append(task)
    print(f"  Downloaded: {len(tasks)} tasks (cached to {cache_path})")
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
#  TransformEncoder: 从 (input, output) 对中提取变换表示
# ──────────────────────────────────────────────────────────────────────────────

# 变换规则查询 token 数（类似 DETR 的 object queries）
# 每个 rule token 可以学习提取变换的一个方面（颜色映射、空间变换等）
NUM_RULE_TOKENS = 8


class TransformEncoder(nn.Module):
    """
    从一对 (input, output) 中提取变换规则表示。

    核心机制：可学习的 rule query tokens 通过 cross-attention
    同时关注 input 和 output 的 patch tokens，提取"发生了什么变换"。

    类似 DETR 的 object queries — 但这里提取的不是目标检测框，
    而是抽象的变换规则向量。
    """
    def __init__(self):
        super().__init__()
        D = config.arc_dim  # 256

        # 可学习 rule query tokens — 变换规则的"探针"
        self.rule_queries = nn.Parameter(torch.randn(1, NUM_RULE_TOKENS, D) * 0.02)

        # 用于区分 input tokens 和 output tokens 的 type embedding
        self.input_type_embed = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.output_type_embed = nn.Parameter(torch.randn(1, 1, D) * 0.02)

        # Cross-attention: rule queries attend to (input + output) tokens
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=D,
                heads=config.arc_predictor_heads,
                dim_head=config.arc_predictor_dim_head,
                mlp_dim=config.arc_predictor_mlp_dim,
                dropout=0.1,
            )
            for _ in range(3)  # 3 层足够提取变换
        ])

        self.norm = nn.LayerNorm(D)

    def forward(self, in_tokens: torch.Tensor,
                out_tokens: torch.Tensor) -> torch.Tensor:
        """
        in_tokens:  (1, N_in, D)  输入 grid 的 patch tokens
        out_tokens: (1, N_out, D) 输出 grid 的 patch tokens
        返回: (1, K, D) 变换规则 tokens（K = NUM_RULE_TOKENS）

        机制：将 input + output tokens 拼接（加 type embedding 区分来源），
        rule queries 通过 cross-attention 提取两者之间的变换关系。
        """
        # 添加 type embedding 区分输入和输出
        in_typed = in_tokens + self.input_type_embed
        out_typed = out_tokens + self.output_type_embed

        # 拼接为完整的 pair context
        pair_context = torch.cat([in_typed, out_typed], dim=1)  # (1, N_in+N_out, D)

        # Rule queries 提取变换规则
        rules = self.rule_queries.expand(in_tokens.shape[0], -1, -1)  # (B, K, D)
        for layer in self.cross_layers:
            rules = layer(rules, pair_context)

        return self.norm(rules)  # (B, K, D)


# ──────────────────────────────────────────────────────────────────────────────
#  TransformConsensus: 从多对变换表示中找到共同规则
# ──────────────────────────────────────────────────────────────────────────────

class TransformConsensus(nn.Module):
    """
    从多个 (input, output) 对的变换表示中找到共同规则。

    核心假设：如果模型正确理解了规则，所有 pair 产出的 delta 应该表示同一变换。
    self-attention 让模型在多个 delta 之间做比较和整合，找到不变的共同模式。
    """
    def __init__(self):
        super().__init__()
        D = config.arc_dim

        # 可学习的 consensus query tokens
        # 从所有 pair deltas 中提取共识
        self.consensus_queries = nn.Parameter(
            torch.randn(1, NUM_RULE_TOKENS, D) * 0.02
        )

        # Cross-attention: consensus queries attend to all pair deltas
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=D,
                heads=config.arc_predictor_heads,
                dim_head=config.arc_predictor_dim_head,
                mlp_dim=config.arc_predictor_mlp_dim,
                dropout=0.1,
            )
            for _ in range(2)
        ])
        self.norm = nn.LayerNorm(D)

    def forward(self, deltas: list[torch.Tensor]) -> torch.Tensor:
        """
        deltas: list of (1, K, D) 每个 pair 的变换表示
        返回: (1, K, D) 共识变换表示

        机制：将所有 pair 的 rule tokens 拼接成长序列，
        consensus queries 通过 cross-attention 提取跨 pair 的共同模式。
        """
        # 拼接所有 pair 的 rule tokens
        all_deltas = torch.cat(deltas, dim=1)  # (1, num_pairs * K, D)

        # Consensus queries 提取共识
        consensus = self.consensus_queries.expand(1, -1, -1)  # (1, K, D)
        for layer in self.cross_layers:
            consensus = layer(consensus, all_deltas)

        return self.norm(consensus)  # (1, K, D)


# ──────────────────────────────────────────────────────────────────────────────
#  TransformApplier: 将共识规则应用到测试输入
# ──────────────────────────────────────────────────────────────────────────────

class TransformApplier(nn.Module):
    """
    将共识变换规则应用到测试输入上，生成预测的输出 tokens。

    机制：
    1. 将 test_input_tokens 和 rule_tokens 拼接
    2. 通过 self-attention 让 test tokens 和 rule tokens 交互
    3. 输出 test token 位置的 tokens 作为 decoder 的上下文
    """
    def __init__(self):
        super().__init__()
        D = config.arc_dim

        # 标记 test input tokens 和 rule tokens 的 type embedding
        self.test_type_embed = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.rule_type_embed = nn.Parameter(torch.randn(1, 1, D) * 0.02)

        # Self-attention: test tokens 和 rule tokens 自由交互
        self.layers = nn.ModuleList([
            EncoderTransformerBlock(
                dim=D,
                heads=config.arc_predictor_heads,
                dim_head=config.arc_predictor_dim_head,
                mlp_dim=config.arc_predictor_mlp_dim,
                dropout=0.1,
            )
            for _ in range(4)  # 4 层应用变换
        ])
        self.norm = nn.LayerNorm(D)

    def forward(self, test_tokens: torch.Tensor,
                rule_tokens: torch.Tensor) -> torch.Tensor:
        """
        test_tokens: (1, N_test, D) 测试输入的 patch tokens
        rule_tokens: (1, K, D) 共识变换规则 tokens
        返回: (1, N_test + K, D) 混合 context tokens（供 decoder 使用）

        decoder 的 query tokens 会通过 cross-attention
        同时关注 spatial（test tokens）和 rule（rule tokens）信息。
        """
        # 添加 type embedding
        test_typed = test_tokens + self.test_type_embed
        rule_typed = rule_tokens + self.rule_type_embed

        # 拼接并做 self-attention
        combined = torch.cat([test_typed, rule_typed], dim=1)  # (1, N+K, D)
        for layer in self.layers:
            combined = layer(combined)

        return self.norm(combined)  # (1, N+K, D)


# ──────────────────────────────────────────────────────────────────────────────
#  ARC JEPA v5.0 Model
# ──────────────────────────────────────────────────────────────────────────────

class ArcJEPA(nn.Module):
    """
    ARC JEPA v5.0: Transformation-Centric Architecture.

    数据流：
      1. 对每个 (input, output) 示例对：
         - GridEncoder 编码两个 grid -> token 序列
         - TransformEncoder 提取变换规则 delta
      2. TransformConsensus 找到所有 pair 的共同规则
      3. GridEncoder 编码测试输入 -> token 序列
      4. TransformApplier 将规则应用到测试 tokens
      5. GridDecoder 从应用后的 context 解码输出 grid

    正则化：
      - 变换一致性损失：鼓励各 pair 的 delta 接近共识
      - 潜空间对齐损失：EMA target encoder 提供目标
      - 像素 CE 损失：最终解码输出与 gold grid 对比
    """
    def __init__(self):
        super().__init__()
        self.grid_encoder = GridEncoder()
        self.grid_decoder = GridDecoder()
        self.transform_encoder = TransformEncoder()
        self.transform_consensus = TransformConsensus()
        self.transform_applier = TransformApplier()

        # EMA target encoder
        self.target_encoder = copy.deepcopy(self.grid_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_target_encoder(self):
        for p_online, p_target in zip(
            self.grid_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_target.data.mul_(config.ema_momentum).add_(
                p_online.data, alpha=1.0 - config.ema_momentum
            )

    def _encode_grid(self, grid: list[list[int]]) -> torch.Tensor:
        """编码单个 grid -> (1, N_objects, D) 物体级 token 序列。"""
        return self.grid_encoder.encode_single(grid)  # (1, N, D)

    def _encode_grid_target(self, grid: list[list[int]]) -> torch.Tensor:
        """用 EMA target encoder 编码 grid。"""
        device = next(self.parameters()).device
        h, w = len(grid), len(grid[0])
        t = torch.tensor(grid, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            tokens, _, _ = self.target_encoder(t)
        return tokens  # (1, N, D)

    def forward(self, task: dict) -> dict:
        """
        单题前向传播。

        核心流程：
        1. 从 train pairs 中提取变换规则
        2. 在 train pairs 间找共识
        3. 将共识规则应用到测试输入
        4. 解码预测输出 grid
        """
        device = next(self.parameters()).device
        train_pairs = task["train"][:config.arc_context_max_pairs]
        test_pair = task["test"][0]
        H_out = len(test_pair["output"])
        W_out = len(test_pair["output"][0])

        # -- 1. 提取每个 pair 的变换规则 --
        deltas = []
        for pair in train_pairs:
            in_tokens = self._encode_grid(pair["input"])    # (1, N_in, D)
            out_tokens = self._encode_grid(pair["output"])  # (1, N_out, D)
            delta = self.transform_encoder(in_tokens, out_tokens)  # (1, K, D)
            deltas.append(delta)

        # -- 2. 找共识变换规则 --
        if len(deltas) == 1:
            consensus = deltas[0]
        else:
            consensus = self.transform_consensus(deltas)  # (1, K, D)

        # -- 3. 编码测试输入 --
        test_tokens = self._encode_grid(test_pair["input"])  # (1, N_test, D)

        # -- 4. 应用变换 -> 解码 --
        applied_context = self.transform_applier(
            test_tokens, consensus
        )  # (1, N_test + K, D)

        logits = self.grid_decoder(applied_context, H_out, W_out)
        gold = torch.tensor(test_pair["output"], dtype=torch.long, device=device)

        # -- 5. 损失计算 --
        # 主损失：pixel CE
        loss_pixel = F.cross_entropy(
            logits.reshape(-1, config.grid_num_colors),
            gold.reshape(-1),
        )

        # 辅助损失：潜空间对齐（EMA target）
        target_tokens = self._encode_grid_target(test_pair["output"])  # (1, N_t, D)
        # 用 applied context 的前 N_test 个 token 与 target 对齐
        min_t = min(test_tokens.shape[1], target_tokens.shape[1])
        loss_latent = F.mse_loss(
            applied_context[:, :min_t, :],
            target_tokens[:, :min_t, :].detach()
        )

        # 辅助损失：变换一致性（所有 pair 的 delta 应该接近共识）
        loss_consistency = torch.tensor(0.0, device=device)
        if len(deltas) > 1:
            for delta in deltas:
                loss_consistency = loss_consistency + F.mse_loss(delta, consensus.detach())
            loss_consistency = loss_consistency / len(deltas)

        # 总损失
        loss_total = loss_pixel + 0.5 * loss_latent + 0.1 * loss_consistency

        pred_grid = logits.argmax(dim=-1).squeeze(0)
        return {
            "loss_pixel": loss_pixel,
            "loss_latent": loss_latent,
            "loss_consistency": loss_consistency,
            "loss_total": loss_total,
            "pred_grid": pred_grid,
            "gold_grid": gold,
        }

    def forward_batch(self, tasks: list[dict]) -> dict:
        """Mini-batch: 逐题 forward 然后平均梯度。"""
        device = next(self.parameters()).device
        B = len(tasks)

        total_loss = 0.0
        total_cells_correct = 0
        total_cells = 0
        exact_matches = 0

        for task in tasks:
            result = self.forward(task)
            total_loss += result["loss_total"]

            pred = result["pred_grid"]
            gold = result["gold_grid"]
            match = (pred == gold).sum().item()
            total_cells_correct += match
            total_cells += gold.numel()
            if torch.equal(pred, gold):
                exact_matches += 1

        loss_total = total_loss / B

        return {
            "loss_total": loss_total,
            "cells_correct": total_cells_correct,
            "cells_total": total_cells,
            "exact_matches": exact_matches,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Test-Time Augmentation
# ──────────────────────────────────────────────────────────────────────────────

_TTA_TRANSFORMS = [
    ("identity", lambda g: g, lambda g: g),
    ("rot90",    rotate_90,    rotate_270),
    ("rot180",   rotate_180,   rotate_180),
    ("rot270",   rotate_270,   rotate_90),
    ("flipH",    flip_horizontal, flip_horizontal),
    ("flipV",    flip_vertical,   flip_vertical),
]


@torch.no_grad()
def predict_with_tta(model: ArcJEPA, task: dict) -> torch.Tensor:
    """TTA: 多种几何变换 + majority voting。"""
    model.eval()
    device = next(model.parameters()).device
    test_pair = task["test"][0]
    H = len(test_pair["output"])
    W = len(test_pair["output"][0])
    all_preds = []

    for name, fwd_fn, inv_fn in _TTA_TRANSFORMS:
        if name == "identity":
            aug_task = task
        else:
            aug_task = transform_task(task, fwd_fn)
        result = model(aug_task)
        pred = result["pred_grid"].cpu()
        if name != "identity":
            pred_list = pred.tolist()
            pred_list = inv_fn(pred_list)
            pred = torch.tensor(pred_list, dtype=torch.long)
        if pred.shape == (H, W):
            all_preds.append(pred)

    if not all_preds:
        result = model(task)
        return result["pred_grid"]

    stacked = torch.stack(all_preds, dim=0)
    votes = F.one_hot(stacked.long(), num_classes=config.grid_num_colors)
    vote_counts = votes.sum(dim=0)
    final_pred = vote_counts.argmax(dim=-1)
    return final_pred.to(device)


# ──────────────────────────────────────────────────────────────────────────────
#  Test-Time Training (TTT) — 推理时梯度适应
# ──────────────────────────────────────────────────────────────────────────────

TTT_STEPS = 25       # 每道题的微调步数（越多越准但越慢）
TTT_LR = 3e-4        # TTT 学习率（比训练 LR 高，因为要快速适应）


def ttt_predict(model: ArcJEPA, task: dict,
                ttt_steps: int = TTT_STEPS,
                ttt_lr: float = TTT_LR) -> torch.Tensor:
    """
    Test-Time Training: 遇到新题时现场微调，然后预测。

    核心流程：
    1. 保存模型原始权重
    2. Leave-one-out: 轮流把每个 train pair 当"测试题"，
       用其余 pair 做 context，对已知答案做梯度下降
    3. 微调后的模型对真正的 test input 做预测
    4. 恢复原始权重（不污染基础模型）

    TTT 的本质：让模型在回答前先"审题思考"——
    通过在当前题目的已知示例上做梯度更新，
    让权重临时适配到"这道题的规则"。
    这是 ARC 2024 冠军 MindsAI 的核心技术。
    """
    device = next(model.parameters()).device

    # 保存原始权重
    original_state = copy.deepcopy(model.state_dict())

    # 创建 TTT 优化器（只优化部分关键参数以加快速度）
    # 优化 TransformEncoder + TransformApplier（规则理解和应用的核心）
    ttt_params = list(model.transform_encoder.parameters()) + \
                 list(model.transform_applier.parameters()) + \
                 list(model.grid_decoder.parameters())
    optimizer = torch.optim.Adam(ttt_params, lr=ttt_lr)

    train_pairs = task["train"]

    # -- TTT 微调循环 --
    model.train()
    use_amp = config.use_fp16 and config.device == "cuda"

    for step in range(ttt_steps):
        total_loss = 0.0

        # Leave-one-out: 每个 train pair 轮流当"测试题"
        for i in range(len(train_pairs)):
            # 构造子任务：pair i 是"测试"，其余是 context
            other_pairs = [p for j, p in enumerate(train_pairs) if j != i]
            if not other_pairs:
                # 只有一个 pair 的情况：用自己做 context 和测试
                other_pairs = train_pairs

            sub_task = {
                "train": other_pairs,
                "test": [train_pairs[i]],
                "_id": task.get("_id", ""),
            }

            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model(sub_task)
                total_loss = total_loss + result["loss_total"]

        avg_loss = total_loss / len(train_pairs)
        optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(ttt_params, max_norm=1.0)
        optimizer.step()

    # -- 用微调后的模型预测 --
    model.eval()
    with torch.no_grad():
        result = model(task)
    pred = result["pred_grid"]

    # -- 恢复原始权重 --
    model.load_state_dict(original_state)

    return pred


def evaluate_with_ttt(model: ArcJEPA, tasks: list[dict],
                      ttt_steps: int = TTT_STEPS,
                      ttt_lr: float = TTT_LR,
                      verbose: bool = True) -> dict:
    """用 TTT 做评测：每道题先微调再预测。"""
    exact = 0
    cell_correct = 0
    cell_total = 0

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  ARC JEPA v5.0 Eval + TTT ({len(tasks)} tasks)")
        print(f"  TTT Steps: {ttt_steps}  LR: {ttt_lr}")
        print(f"{'=' * 70}\n")

    for idx, task in enumerate(tasks):
        device = next(model.parameters()).device
        gold = torch.tensor(
            task["test"][0]["output"], dtype=torch.long, device=device
        )

        t0 = time.time()
        pred = ttt_predict(model, task, ttt_steps=ttt_steps, ttt_lr=ttt_lr)
        elapsed = time.time() - t0

        is_exact = torch.equal(pred, gold)
        if is_exact:
            exact += 1
        match = (pred == gold).sum().item()
        cells = gold.numel()
        cell_correct += match
        cell_total += cells

        if verbose and ((idx + 1) % 5 == 0 or is_exact):
            cell_acc = match / cells * 100
            status = " <<< EXACT MATCH >>>" if is_exact else ""
            print(f"  [{idx+1:3d}/{len(tasks)}]  cell={cell_acc:.0f}%  "
                  f"{elapsed:.1f}s{status}")

    total = len(tasks)
    final_cell = cell_correct / max(cell_total, 1) * 100
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  TTT Results: Exact={exact}/{total} ({exact/max(total,1)*100:.1f}%)  "
              f"Cell={final_cell:.1f}%")
        print(f"{'=' * 70}")
    return {"exact": exact, "total": total, "cell_acc": final_cell}


# ──────────────────────────────────────────────────────────────────────────────
#  Learning Rate Schedule
# ──────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch: int):
        if epoch <= self.warmup_epochs:
            lr = self.base_lr * epoch / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


# ──────────────────────────────────────────────────────────────────────────────
#  Training Loop v5.0
# ──────────────────────────────────────────────────────────────────────────────

MINI_BATCH_SIZE = 4
ACCUM_STEPS = 8


def train(model: ArcJEPA, tasks: list[dict], epochs: int = 200,
          eval_tasks: list[dict] | None = None,
          patience: int = 30,
          augment: bool = True,
          batch_size: int = MINI_BATCH_SIZE) -> ArcJEPA:

    use_amp = config.use_fp16 and config.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.arc_lr, weight_decay=0.01
    )
    warmup_epochs = min(10, epochs // 10)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs, epochs, config.arc_lr
    )

    best_cell_acc = 0.0
    best_epoch = 0
    best_state = None

    print(f"\n{'=' * 70}")
    print(f"  ARC JEPA v5.0 (Transformation-Centric)")
    print(f"  Tasks: {len(tasks)}  Augment: {'ON' if augment else 'OFF'}")
    print(f"  Epochs: {epochs}  Warmup: {warmup_epochs}")
    print(f"  AMP: {'ON' if use_amp else 'OFF'}")
    print(f"  Batch: {batch_size}  Accum: {ACCUM_STEPS}  Eff: {batch_size * ACCUM_STEPS}")
    print(f"  Rule Tokens: {NUM_RULE_TOKENS}")
    print(f"  TransformEncoder: 3L CrossAttn")
    print(f"  TransformConsensus: 2L CrossAttn")
    print(f"  TransformApplier: 4L SelfAttn")
    print(f"  Early Stopping: patience={patience}")
    print(f"{'=' * 70}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        exact_match = 0
        batch_count = 0

        order = list(range(len(tasks)))
        random.shuffle(order)
        current_lr = scheduler.step(epoch)
        optimizer.zero_grad()
        t0 = time.time()

        for batch_start in range(0, len(order), batch_size):
            batch_indices = order[batch_start:batch_start + batch_size]
            batch_tasks = [tasks[i] for i in batch_indices]
            if augment:
                batch_tasks = [online_augment_task(t) for t in batch_tasks]

            with torch.amp.autocast("cuda", enabled=use_amp):
                result = model.forward_batch(batch_tasks)
                scaled_loss = result["loss_total"] / ACCUM_STEPS

            scaler.scale(scaled_loss).backward()

            step_idx = batch_start // batch_size
            is_last = (batch_start + batch_size) >= len(order)
            if (step_idx + 1) % ACCUM_STEPS == 0 or is_last:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                model._update_target_encoder()

            total_loss += result["loss_total"].item() * len(batch_tasks)
            correct += result["cells_correct"]
            total += result["cells_total"]
            exact_match += result["exact_matches"]
            batch_count += len(batch_tasks)

        elapsed = time.time() - t0
        avg_loss = total_loss / max(batch_count, 1)
        cell_acc = correct / max(total, 1) * 100

        if cell_acc > best_cell_acc:
            best_cell_acc = cell_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  [Epoch {epoch:3d}/{epochs}]  "
                f"Loss: {avg_loss:.4f}  "
                f"Cell: {cell_acc:.1f}%  "
                f"Exact: {exact_match}/{batch_count}  "
                f"Best: {best_cell_acc:.1f}%@{best_epoch}  "
                f"LR: {current_lr:.2e}  "
                f"{elapsed:.1f}s"
            )

        if eval_tasks and epoch % 25 == 0:
            eval_result = evaluate(model, eval_tasks, verbose=False, use_tta=False)
            print(
                f"    Eval: cell={eval_result['cell_acc']:.1f}%  "
                f"exact={eval_result['exact']}/{eval_result['total']}"
            )

        if epoch - best_epoch >= patience and epoch > warmup_epochs:
            print(
                f"\n  Early stopping @ epoch {epoch} "
                f"(best={best_cell_acc:.1f}% @ {best_epoch})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Best: epoch {best_epoch}, cell {best_cell_acc:.1f}%")
    return model


# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model: ArcJEPA, tasks: list[dict],
             verbose: bool = True, use_tta: bool = True) -> dict:
    model.eval()
    exact = 0
    cell_correct = 0
    cell_total = 0

    if verbose:
        mode = "TTA" if use_tta else "Direct"
        print(f"\n{'=' * 70}")
        print(f"  ARC JEPA v5.0 Eval ({len(tasks)} tasks, {mode})")
        print(f"{'=' * 70}\n")

    with torch.no_grad():
        for idx, task in enumerate(tasks):
            device = next(model.parameters()).device
            gold = torch.tensor(
                task["test"][0]["output"], dtype=torch.long, device=device
            )
            if use_tta:
                pred = predict_with_tta(model, task)
            else:
                result = model(task)
                pred = result["pred_grid"]

            is_exact = torch.equal(pred, gold)
            if is_exact:
                exact += 1
            match = (pred == gold).sum().item()
            cells = gold.numel()
            cell_correct += match
            cell_total += cells

            if verbose and ((idx + 1) % 10 == 0 or is_exact):
                cell_acc = match / cells * 100
                status = "EXACT" if is_exact else ""
                print(f"  [{idx+1:3d}/{len(tasks)}]  cell={cell_acc:.0f}%  {status}")

    total = len(tasks)
    final_cell = cell_correct / max(cell_total, 1) * 100
    if verbose:
        print(f"\n  Exact: {exact}/{total} ({exact/max(total,1)*100:.1f}%)  "
              f"Cell: {final_cell:.1f}%")
    return {"exact": exact, "total": total, "cell_acc": final_cell}


# ──────────────────────────────────────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARC JEPA v5.0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train_n", type=int, default=400)
    parser.add_argument("--eval_n", type=int, default=100)
    parser.add_argument("--save", type=str, default="arc_jepa_model.pt")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=MINI_BATCH_SIZE)
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--ttt", action="store_true",
                        help="Enable TTT during final evaluation")
    parser.add_argument("--ttt_steps", type=int, default=TTT_STEPS)
    parser.add_argument("--ttt_lr", type=float, default=TTT_LR)
    parser.add_argument("--ttt_only", action="store_true",
                        help="Skip training, load saved model, run TTT eval only")
    args = parser.parse_args()

    eval_tasks = load_arc_tasks(args.eval_n, folder="evaluation")

    if args.ttt_only:
        # TTT-only 模式: 加载已有模型直接做 TTT 评测
        model = ArcJEPA().to(config.device)
        model.load_state_dict(torch.load(args.save, map_location=config.device, weights_only=True))
        print(f"\n  Loaded model from {args.save}")
        if eval_tasks:
            evaluate_with_ttt(model, eval_tasks,
                              ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr)
        return

    train_tasks = load_arc_tasks(args.train_n, folder="training")
    if not train_tasks:
        print("  No training data")
        return

    model = ArcJEPA().to(config.device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  ARC JEPA v5.0 params: {param_count:,} ({param_count/1e6:.1f}M)")

    model = train(
        model, train_tasks, epochs=args.epochs,
        eval_tasks=eval_tasks,
        patience=args.patience,
        augment=not args.no_augment,
        batch_size=args.batch_size,
    )

    torch.save(model.state_dict(), args.save)
    print(f"  Saved to {args.save}")

    if eval_tasks:
        print("\n  --- Final Eval (no TTA) ---")
        evaluate(model, eval_tasks, use_tta=False)

        if args.ttt:
            print("\n  --- Final Eval (TTT) ---")
            evaluate_with_ttt(model, eval_tasks,
                              ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr)


if __name__ == "__main__":
    main()
