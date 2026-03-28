"""
jepa_engine/grid_codec.py -- Grid 编码器 / 解码器 v6.0（认知先验架构）

v6.0 核心设计 -- 模拟人类视觉认知系统：

  1. CNN Backbone（视觉皮层 V1/V2）：
     - 3x3 卷积堆叠，天然编码局部空间结构（边缘、角落、纹理）
     - 平移等变性：同一图案在不同位置产生相同响应
     - 输出 per-cell 特征图 (B, D, H, W)

  2. Object Segmentation（视觉皮层 V4 / 物体感知）：
     - 连通区域检测：同色相邻格子 = 一个"物体"
     - 每个物体 = 一个 token（CNN 特征池化 + 属性向量）
     - 这是结构性先验，不是硬编码规则

  3. Object Attention（关系推理 / 前额叶逻辑）：
     - 物体级 self-attention：物体之间的关系推理
     - "物体 A 在物体 B 的左边" "物体 C 和 D 颜色相同"

  4. GridDecoder（Cross-Attention 解码）：
     - 从物体级 context tokens 解码回像素级 grid

数据流：
  Grid (H, W) -> CNN (局部特征) -> 连通区域分割 -> 物体 tokens
    -> Object Attention (关系推理) -> (num_objects, D) token 序列
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mvp_config import config
from jepa_engine.encoder import EncoderTransformerBlock


# 每个 grid 最多保留的物体数（按面积排序，保留最大的）
MAX_OBJECTS_PER_GRID = 48
# 物体属性维度（color, size, center_y, center_x, height, width）
OBJECT_ATTR_DIM = 6


# ──────────────────────────────────────────────────────────────────────────────
#  CNN Backbone: 模拟视觉皮层 V1/V2
# ──────────────────────────────────────────────────────────────────────────────

def _make_norm(channels: int) -> nn.Module:
    """GroupNorm 替代 BatchNorm — 不依赖 batch 统计量，单样本也能用。"""
    # groups 取 min(16, channels)，确保 channels 能被 groups 整除
    groups = min(16, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    """残差块：2 层 3x3 卷积 + skip connection。"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = _make_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = _make_norm(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)


class CNNBackbone(nn.Module):
    """
    CNN 视觉皮层：3x3 卷积堆叠，提取 per-cell 局部空间特征。

    天然编码的认知先验：
    - 3x3 kernel = 感知 8 邻域关系（上下左右 + 对角线）
    - 多层堆叠 = 层级抽象（边缘 → 纹理 → 形状）
    - 平移等变性 = 同一图案在不同位置有相同响应
    """
    def __init__(self):
        super().__init__()
        D = config.arc_dim  # 256

        # 输入嵌入：one-hot(10) → 64 channels
        self.embed = nn.Conv2d(config.grid_num_colors, 64, 3, padding=1, bias=False)
        self.embed_bn = _make_norm(64)

        # 残差层堆叠：64 → 128 → 256
        self.res1 = ResBlock(64, 64)
        self.res2 = ResBlock(64, 128)
        self.res3 = ResBlock(128, 128)
        self.res4 = ResBlock(128, D)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        grid: (B, H, W) long tensor, values 0-9
        返回: (B, D, H, W) per-cell 特征图
        """
        # One-hot 编码
        x = F.one_hot(grid.long(), num_classes=config.grid_num_colors)
        x = x.float().permute(0, 3, 1, 2)  # (B, 10, H, W)

        # CNN forward
        x = F.gelu(self.embed_bn(self.embed(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        return x  # (B, D, H, W)


# ──────────────────────────────────────────────────────────────────────────────
#  Object Segmentation: 连通区域检测 + 特征池化
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
from scipy.ndimage import label as scipy_label


@torch.no_grad()
def find_objects(grid: torch.Tensor) -> list[dict]:
    """
    用 scipy.ndimage.label（C 优化）检测连通区域。
    比 Python BFS 快 50-100 倍。

    grid: (H, W) long tensor
    返回: list of {color, label_id, size, bbox, mask}，按面积降序排列
    """
    H, W = grid.shape
    grid_np = grid.cpu().numpy()
    objects = []

    # 对每种颜色分别做连通区域检测
    for color in range(config.grid_num_colors):
        color_mask = (grid_np == color)
        if not color_mask.any():
            continue
        labeled, num_features = scipy_label(color_mask)
        for label_id in range(1, num_features + 1):
            obj_mask = (labeled == label_id)
            rows, cols = np.where(obj_mask)
            if len(rows) == 0:
                continue
            objects.append({
                "color": color,
                "size": len(rows),
                "bbox": (rows.min().item(), cols.min().item(),
                         rows.max().item(), cols.max().item()),
                "rows": torch.from_numpy(rows.astype(np.int64)),
                "cols": torch.from_numpy(cols.astype(np.int64)),
            })

    objects.sort(key=lambda o: o["size"], reverse=True)
    return objects[:MAX_OBJECTS_PER_GRID]


def compute_object_attributes(obj: dict, H: int, W: int,
                              device: torch.device) -> torch.Tensor:
    """计算物体的归一化属性向量 [color, size, cy, cx, h, w]。"""
    r_min, c_min, r_max, c_max = obj["bbox"]
    return torch.tensor([
        obj["color"] / 9.0,
        min(obj["size"] / (H * W), 1.0),
        (r_min + r_max) / 2.0 / max(H - 1, 1),
        (c_min + c_max) / 2.0 / max(W - 1, 1),
        (r_max - r_min + 1) / H,
        (c_max - c_min + 1) / W,
    ], dtype=torch.float32, device=device)


class ObjectTokenizer(nn.Module):
    """将 CNN 特征图 + 物体分割结果转换为物体级 tokens（向量化版）。"""
    def __init__(self):
        super().__init__()
        D = config.arc_dim
        self.attr_embed = nn.Sequential(
            nn.Linear(OBJECT_ATTR_DIM, D), nn.GELU(), nn.Linear(D, D),
        )
        self.fuse = nn.Sequential(
            nn.Linear(D * 2, D), nn.GELU(), nn.Linear(D, D),
        )
        self.norm = nn.LayerNorm(D)

    def forward(self, feature_map: torch.Tensor,
                objects: list[dict],
                H: int, W: int) -> torch.Tensor:
        """
        feature_map: (D, H, W) 特征图
        objects: 检测到的物体列表（含 rows, cols 索引）
        返回: (num_objects, D) 物体 tokens
        """
        device = feature_map.device
        D = feature_map.shape[0]

        if not objects:
            return torch.zeros(1, D, device=device)

        # 批量计算属性嵌入
        attrs = torch.stack([
            compute_object_attributes(obj, H, W, device) for obj in objects
        ])  # (N_obj, 6)
        attr_embs = self.attr_embed(attrs)  # (N_obj, D)

        # 向量化特征池化：用 index_select 批量提取
        # feature_map: (D, H, W) -> (D, H*W)
        feat_flat = feature_map.view(D, -1)  # (D, H*W)
        pooled_list = []
        for obj in objects:
            # 将 (row, col) 转为 flat index
            flat_idx = (obj["rows"].to(device) * W + obj["cols"].to(device))
            # 用 gather 的方式避免循环
            obj_feats = feat_flat[:, flat_idx]  # (D, num_cells)
            pooled_list.append(obj_feats.mean(dim=1))  # (D,)

        pooled = torch.stack(pooled_list)  # (N_obj, D)

        # 融合
        fused = self.fuse(torch.cat([pooled, attr_embs], dim=-1))  # (N_obj, D)
        return self.norm(fused)


# ──────────────────────────────────────────────────────────────────────────────
#  Object Attention: 物体级关系推理
# ──────────────────────────────────────────────────────────────────────────────

class ObjectAttention(nn.Module):
    """
    物体级 Self-Attention：模拟前额叶的关系推理。

    物体 token 之间做 self-attention，学习捕捉：
    - 空间关系（"A 在 B 的上方"）
    - 属性关系（"A 和 C 颜色相同"）
    - 逻辑模式（"所有红色物体都在边缘"）
    """
    def __init__(self):
        super().__init__()
        D = config.arc_dim
        self.layers = nn.ModuleList([
            EncoderTransformerBlock(
                dim=D,
                heads=config.arc_predictor_heads,
                dim_head=config.arc_predictor_dim_head,
                mlp_dim=config.arc_predictor_mlp_dim,
                dropout=0.1,
            )
            for _ in range(2)  # 2 层关系推理
        ])
        self.norm = nn.LayerNorm(D)

    def forward(self, object_tokens: torch.Tensor) -> torch.Tensor:
        """
        object_tokens: (1, N, D) 物体 token 序列
        返回: (1, N, D) 经过关系推理的 token 序列
        """
        x = object_tokens
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ──────────────────────────────────────────────────────────────────────────────
#  Cross-Attention Block (Decoder 用)
# ──────────────────────────────────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """Cross-Attention: query attend to context。"""
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head

        self.norm1 = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q_norm = self.norm1(query)
        c_norm = self.norm_ctx(context)
        B, Nq, _ = q_norm.shape
        Nc = c_norm.shape[1]

        q = self.to_q(q_norm).view(B, Nq, self.heads, self.dim_head).transpose(1, 2)
        kv = self.to_kv(c_norm).chunk(2, dim=-1)
        k, v = [t.view(B, Nc, self.heads, self.dim_head).transpose(1, 2) for t in kv]

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Nq, -1)
        query = query + self.to_out(attn_out)
        query = query + self.ffn(self.norm2(query))
        return query


# ──────────────────────────────────────────────────────────────────────────────
#  GridEncoder v6.0: CNN + Object Segmentation + Object Attention
# ──────────────────────────────────────────────────────────────────────────────

class GridEncoder(nn.Module):
    """
    ARC Grid -> 物体级 token 序列。

    完整认知处理流程：
    1. CNN 提取 per-cell 视觉特征
    2. 连通区域分割检测物体
    3. 物体特征池化 + 属性嵌入 → 物体 tokens
    4. 物体级 self-attention 做关系推理
    """
    def __init__(self):
        super().__init__()
        self.cnn = CNNBackbone()
        self.tokenizer = ObjectTokenizer()
        self.object_attn = ObjectAttention()

    def forward(self, grid: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        grid: (B, H, W) long tensor
        返回: (tokens, 0, 0)
            tokens: (B, max_objects, D) 物体 token 序列（padded）
            后两个值为 0（兼容旧接口）
        """
        B, H, W = grid.shape
        feature_maps = self.cnn(grid)  # (B, D, H, W)

        all_tokens = []
        max_n = 0

        for b in range(B):
            objects = find_objects(grid[b])
            tokens = self.tokenizer(feature_maps[b], objects, H, W)  # (N_b, D)
            all_tokens.append(tokens)
            max_n = max(max_n, tokens.shape[0])

        # Pad to max_n and stack
        device = grid.device
        D = config.arc_dim
        padded = torch.zeros(B, max_n, D, device=device)
        for b, tokens in enumerate(all_tokens):
            padded[b, :tokens.shape[0], :] = tokens

        # Object-level attention
        padded = self.object_attn(padded)

        return padded, 0, 0

    def forward_single(self, grid: torch.Tensor) -> torch.Tensor:
        """编码单个 grid -> (1, N, D) 不做 padding。"""
        B, H, W = grid.shape
        feature_map = self.cnn(grid)  # (1, D, H, W)
        objects = find_objects(grid[0])
        tokens = self.tokenizer(feature_map[0], objects, H, W)  # (N, D)
        tokens = tokens.unsqueeze(0)  # (1, N, D)
        tokens = self.object_attn(tokens)
        return tokens

    def forward_pooled(self, grid: torch.Tensor) -> torch.Tensor:
        """向后兼容：返回 GAP 后的 (B, D)。"""
        tokens, _, _ = self.forward(grid)
        return tokens.mean(dim=1)

    def encode_batch(self, grids: list[list[list[int]]]) -> tuple[torch.Tensor, int, int]:
        """便捷接口：Python list -> (N, max_objects, D)。"""
        max_h = max(len(g) for g in grids)
        max_w = max(len(g[0]) for g in grids)
        B = len(grids)
        padded = torch.zeros(B, max_h, max_w, dtype=torch.long,
                             device=next(self.parameters()).device)
        for i, g in enumerate(grids):
            h, w = len(g), len(g[0])
            padded[i, :h, :w] = torch.tensor(g, dtype=torch.long)
        return self.forward(padded)

    def encode_single(self, grid: list[list[int]]) -> torch.Tensor:
        """编码单个 grid -> (1, N_objects, D)。无 padding。"""
        device = next(self.parameters()).device
        h, w = len(grid), len(grid[0])
        t = torch.tensor(grid, dtype=torch.long, device=device).unsqueeze(0)
        return self.forward_single(t)


# ──────────────────────────────────────────────────────────────────────────────
#  GridDecoder v6.0: Cross-Attention from Object Tokens to Output Grid
# ──────────────────────────────────────────────────────────────────────────────

class GridDecoder(nn.Module):
    """
    物体 token 序列 -> ARC Grid。

    使用 per-cell query tokens + cross-attention 从物体级 context 解码。
    每个输出 cell 的 query 通过 cross-attention 从物体 tokens 获取信息，
    然后分类为 10 种颜色之一。
    """
    def __init__(self):
        super().__init__()
        D = config.arc_dim
        max_cells = config.grid_max_size * config.grid_max_size

        # Per-cell query tokens
        self.cell_queries = nn.Parameter(torch.randn(1, max_cells, D) * 0.02)
        # 2D 位置编码
        self.row_embed = nn.Parameter(torch.randn(1, config.grid_max_size, D) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, config.grid_max_size, D) * 0.02)

        # Cross-attention 层
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=D,
                heads=config.arc_predictor_heads,
                dim_head=config.arc_predictor_dim_head,
                mlp_dim=config.arc_predictor_mlp_dim,
                dropout=0.1,
            )
            for _ in range(config.grid_decoder_depth)
        ])
        self.norm = nn.LayerNorm(D)
        # 每个 cell 分类为 10 种颜色
        self.output_head = nn.Linear(D, config.grid_num_colors)

    def forward(self, context_tokens: torch.Tensor,
                H: int, W: int) -> torch.Tensor:
        """
        context_tokens: (B, Nc, D) 物体级 context tokens
        H, W: 输出 grid 尺寸
        返回: (B, H, W, 10) logits
        """
        B = context_tokens.shape[0]
        N = H * W

        # 构建 per-cell query tokens + 2D 位置编码
        queries = self.cell_queries[:, :N, :].expand(B, -1, -1)
        row_pos = self.row_embed[:, :H, :]
        col_pos = self.col_embed[:, :W, :]
        pos_2d = (row_pos.unsqueeze(2) + col_pos.unsqueeze(1)).reshape(1, N, -1)
        queries = queries + pos_2d

        # Cross-attention 从 context 解码
        x = queries
        for layer in self.cross_layers:
            x = layer(x, context_tokens)
        x = self.norm(x)

        # 分类
        logits = self.output_head(x)  # (B, N, 10)
        logits = logits.view(B, H, W, config.grid_num_colors)
        return logits

    def predict_grid(self, context_tokens: torch.Tensor,
                     H: int, W: int) -> torch.Tensor:
        logits = self.forward(context_tokens, H, W)
        return logits.argmax(dim=-1)
