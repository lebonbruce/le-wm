"""
arc_eval.py —— ARC-AGI 基线评测器 v1.0

架构：
  1. 从 HuggingFace 加载 ARC-AGI 数据集（或本地 JSON）
  2. 将 Grid 序列化为紧凑可读文本（行压缩 + 颜色符号化）
  3. 通过 Ollama (/api/chat) 用 qwen3:8b 做 few-shot 推理
  4. 解析模型输出，还原为 Grid，与标准答案做精确匹配
  5. 输出精度报告 + 错误分析

评测方式：
  - 精确匹配 (exact match): 每个 cell 都必须完全正确
  - 部分 IoU：宽容指标（方便追踪微小进步）

用法：
  python arc_eval.py --n 100 --model qwen3:8b
  python arc_eval.py --n 400 --model qwen3:8b --mode evaluation  # 完整测试集
"""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
#  颜色映射：ARC 使用 0-9 十种颜色
# ──────────────────────────────────────────────────────────────────────────────
COLOR_SYMBOLS = {
    0: '.',  # 黑（背景）
    1: 'B',  # 蓝
    2: 'R',  # 红
    3: 'G',  # 绿
    4: 'Y',  # 黄
    5: 'W',  # 灰/白
    6: 'M',  # 品红
    7: 'O',  # 橙
    8: 'A',  # 天蓝
    9: 'P',  # 棕/紫
}
SYMBOL_TO_COLOR = {v: k for k, v in COLOR_SYMBOLS.items()}


# ──────────────────────────────────────────────────────────────────────────────
#  Grid 序列化 / 反序列化
# ──────────────────────────────────────────────────────────────────────────────

def grid_to_text(grid: list[list[int]]) -> str:
    """将 Grid（二维整数列表）转为紧凑文本。每行一串符号，行间换行。

    示例:
      [[0, 1, 0],
       [1, 0, 1]]
    →
      .B.
      B.B
    """
    lines = []
    for row in grid:
        lines.append(''.join(COLOR_SYMBOLS.get(c, '?') for c in row))
    return '\n'.join(lines)


def text_to_grid(text: str) -> list[list[int]] | None:
    """将模型输出文本还原为 Grid。失败返回 None。"""
    lines = []
    for raw in text.strip().split('\n'):
        row_str = raw.strip()
        if not row_str:
            continue
        row = []
        for ch in row_str:
            if ch in SYMBOL_TO_COLOR:
                row.append(SYMBOL_TO_COLOR[ch])
            elif ch.isdigit():  # 模型有时直接输出数字
                row.append(int(ch))
            else:
                return None  # 无法解析
        if row:
            lines.append(row)
    return lines if lines else None


def cells_correct(pred: list, gold: list) -> tuple[int, int]:
    """返回 (正确 cell 数, 总 cell 数)，用于宽容指标。"""
    if not pred or not gold:
        return 0, max(sum(len(r) for r in gold), 1)
    total = sum(len(r) for r in gold)
    correct = 0
    for i, row in enumerate(gold):
        if i >= len(pred):
            break
        for j, val in enumerate(row):
            if j < len(pred[i]) and pred[i][j] == val:
                correct += 1
    return correct, total


# ──────────────────────────────────────────────────────────────────────────────
#  Ollama 客户端
# ──────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, model: str, url: str = "http://host.docker.internal:11434"):
        self.model = model
        self.url = url
        # 测试连通性
        try:
            req = urllib.request.Request(f"{url}/api/tags")
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            raise RuntimeError(f"Ollama 不可达 ({url}): {e}")
        print(f"  ✅ Ollama 连接成功: {url}  模型: {model}")

    def chat(self, user_prompt: str, max_tokens: int = 512) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": user_prompt}],
            "stream": False,
            "think": False,
            "options": {"num_predict": max_tokens, "temperature": 0.0},  # 贪心解码，确定性
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=600)
        raw = resp.read().decode("utf-8-sig")
        data = json.loads(raw)
        msg = data.get("message", {})
        content = msg.get("content", "").strip()
        if "</think>" in content:
            content = content.split("</think>", 1)[1].strip()
        return content


# ──────────────────────────────────────────────────────────────────────────────
#  Prompt 构建
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(task: dict) -> str:
    """将一道 ARC 题目构建为 few-shot prompt。"""
    lines = [
        "你是一个擅长规律识别的智能系统。",
        "我会给你若干组「输入网格 → 输出网格」的示例，请识别规律，",
        "并对最后的「测试输入」生成正确的输出网格。",
        "",
        "网格编码规则：",
        "  . = 黑(0)  B = 蓝(1)  R = 红(2)  G = 绿(3)  Y = 黄(4)",
        "  W = 灰(5)  M = 品红(6)  O = 橙(7)  A = 天蓝(8)  P = 棕(9)",
        "",
        "== 示例 ==",
    ]

    for i, ex in enumerate(task["train"], 1):
        lines.append(f"[示例 {i}] 输入:")
        lines.append(grid_to_text(ex["input"]))
        lines.append(f"[示例 {i}] 输出:")
        lines.append(grid_to_text(ex["output"]))
        lines.append("")

    lines.append("== 测试 ==")
    lines.append("[测试] 输入:")
    lines.append(grid_to_text(task["test"][0]["input"]))
    lines.append("")
    lines.append("[测试] 输出（只输出网格，不要任何解释）:")

    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────────────
#  数据加载
# ──────────────────────────────────────────────────────────────────────────────

def load_arc_tasks(n: int, mode: str = "evaluation") -> list[dict]:
    """
    从 HuggingFace datasets 加载 ARC 数据。
    mode: 'evaluation' (公开测试集400题) 或 'training' (800题训练集)
    """
    print(f"  📥 加载 ARC-AGI 数据集（模式: {mode}, 前 {n} 题）...")
    try:
        from datasets import load_dataset
        # ARC-AGI 在 HF 上的 ID
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=mode, trust_remote_code=True)
    except Exception:
        pass

    # 尝试本地 JSON（如果已下载）
    local_paths = [
        Path("/app/arc_data"),
        Path("arc_data"),
        Path("/data/arc"),
    ]
    for base in local_paths:
        glob_pattern = "*.json"
        files = list(base.glob(glob_pattern)) if base.exists() else []
        if files:
            print(f"  📂 从本地加载: {base}")
            return _load_local_arc(base, n)

    # 直接从 GitHub 下载原始 ARC JSON（最可靠的方式）
    return _download_arc_from_github(n, mode)


def _download_arc_from_github(n: int, mode: str) -> list[dict]:
    """从 fchollet/ARC-AGI GitHub 仓库下载任务 JSON。"""
    base_url = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"
    folder = "evaluation" if mode == "evaluation" else "training"
    index_url = f"{base_url}/{folder}/"

    print(f"  🌐 从 GitHub 下载 ARC {folder} 数据...")

    # 先获取文件列表（通过 GitHub API）
    api_url = f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{folder}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "arc-eval"})
        resp = urllib.request.urlopen(req, timeout=30)
        file_list = json.loads(resp.read().decode())
        json_files = [f["name"] for f in file_list if f["name"].endswith(".json")]
        json_files = json_files[:n]
    except Exception as e:
        print(f"  ⚠ GitHub API 请求失败: {e}")
        # 如果 API 请求失败，使用一些已知的任务 ID 作为后备
        json_files = [f"{i:08x}.json" for i in range(n)]

    tasks = []
    ok_count = 0
    for fname in json_files:
        if ok_count >= n:
            break
        url = f"{base_url}/{folder}/{fname}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "arc-eval"})
            resp = urllib.request.urlopen(req, timeout=15)
            task = json.loads(resp.read().decode())
            # ARC JSON 格式: {"train": [...], "test": [...]}
            if "train" in task and "test" in task and task["test"]:
                task["_id"] = fname.replace(".json", "")
                tasks.append(task)
                ok_count += 1
                if ok_count % 10 == 0:
                    print(f"      已加载 {ok_count}/{n} 题...")
        except Exception as e:
            print(f"  ⚠ 跳过 {fname}: {e}")

    print(f"  ✅ 加载完成: {len(tasks)} 道题")
    return tasks


def _load_local_arc(base: Path, n: int) -> list[dict]:
    """从本地 JSON 文件加载"""
    tasks = []
    for f in sorted(base.glob("*.json"))[:n]:
        data = json.loads(f.read_text())
        if "train" in data and "test" in data and data["test"]:
            data["_id"] = f.stem
            tasks.append(data)
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
#  提取 Grid 输出
# ──────────────────────────────────────────────────────────────────────────────

def extract_grid_from_response(response: str) -> list[list[int]] | None:
    """
    鲁棒地从模型回复中提取 Grid。
    策略：
    1. 找连续的符号行（只含 .BRGYWMOAP 和空格）
    2. 尝试数字格式（"0 1 2 0" 或 "[0,1,2]"）
    """
    # 策略1：符号格式
    symbol_chars = set(''.join(COLOR_SYMBOLS.values()))
    candidate_lines = []
    for line in response.strip().split('\n'):
        cleaned = line.strip()
        if not cleaned:
            if candidate_lines:  # 空行终止当前候选
                break
            continue
        if all(c in symbol_chars for c in cleaned):
            candidate_lines.append(cleaned)

    if candidate_lines:
        grid = text_to_grid('\n'.join(candidate_lines))
        if grid:
            return grid

    # 策略2：数字格式 "0 1 2 ..." 每行
    num_lines = []
    for line in response.strip().split('\n'):
        nums = re.findall(r'\b[0-9]\b', line)
        if nums:
            num_lines.append([int(x) for x in nums])

    if num_lines:
        return num_lines

    return None


# ──────────────────────────────────────────────────────────────────────────────
#  主评测循环
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(tasks: list[dict], client: OllamaClient, max_tokens: int = 256) -> dict:
    """逐题评测，返回汇总统计。"""
    results = {
        "total": len(tasks),
        "exact_match": 0,
        "parse_fail": 0,
        "wrong_shape": 0,
        "wrong_values": 0,
        "total_cell_correct": 0,
        "total_cells": 0,
        "errors": [],
    }

    for idx, task in enumerate(tasks):
        task_id = task.get("_id", f"task_{idx}")
        gold = task["test"][0]["output"]
        prompt = build_prompt(task)

        t0 = time.time()
        try:
            response = client.chat(prompt, max_tokens=max_tokens)
        except Exception as e:
            print(f"  [{idx+1}/{len(tasks)}] {task_id} ⚠ Ollama 错误: {e}")
            results["parse_fail"] += 1
            continue

        elapsed = time.time() - t0
        pred = extract_grid_from_response(response)

        if pred is None:
            results["parse_fail"] += 1
            results["errors"].append({
                "id": task_id, "type": "parse_fail", "response_preview": response[:100]
            })
            status = "❌ parse_fail"
        elif len(pred) != len(gold) or any(
            len(pred[i]) != len(gold[i]) for i in range(len(gold))
        ):
            results["wrong_shape"] += 1
            results["errors"].append({"id": task_id, "type": "wrong_shape"})
            status = "❌ wrong_shape"
        elif pred == gold:
            results["exact_match"] += 1
            status = "✅ exact"
        else:
            results["wrong_values"] += 1
            status = "❌ wrong_values"

        # 宽容指标（cell 级别精度）
        correct, total = cells_correct(pred, gold)
        results["total_cell_correct"] += correct
        results["total_cells"] += total

        cell_acc = correct / max(total, 1) * 100
        print(f"  [{idx+1:3d}/{len(tasks)}] {task_id}  {status}  "
              f"cell={cell_acc:.0f}%  {elapsed:.1f}s")

    return results


def print_summary(results: dict):
    total = results["total"]
    exact = results["exact_match"]
    cell_acc = results["total_cell_correct"] / max(results["total_cells"], 1) * 100

    print("\n" + "=" * 60)
    print("  🏆 ARC-AGI 评测结果摘要")
    print("=" * 60)
    print(f"  总题数：    {total}")
    print(f"  精确匹配：  {exact} / {total}  ({exact/max(total,1)*100:.1f}%)")
    print(f"  Cell 精度： {cell_acc:.1f}%")
    print(f"  解析失败：  {results['parse_fail']}")
    print(f"  形状错误：  {results['wrong_shape']}")
    print(f"  值错误：    {results['wrong_values']}")
    print("=" * 60)

    # 参考基准
    print("\n  参考基准:")
    print("  GPT-4o (2024):        ~5%")
    print("  Claude 3.5 Sonnet:    ~21%")
    print("  o3-mini (大量推理):   ~85%")
    print(f"\n  我们的系统:           {exact/max(total,1)*100:.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
#  入口
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARC-AGI 基线评测器")
    parser.add_argument("--n", type=int, default=20,
                        help="评测题目数量（默认 20；完整测试集 400）")
    parser.add_argument("--model", type=str, default="qwen3:8b",
                        help="Ollama 模型名（默认 qwen3:8b）")
    parser.add_argument("--url", type=str,
                        default="http://host.docker.internal:11434",
                        help="Ollama API 地址")
    parser.add_argument("--mode", type=str, default="evaluation",
                        choices=["evaluation", "training"],
                        help="ARC 数据集模式（evaluation=公开测试集，training=训练集）")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="生成最大 token 数（默认 256，网格通常不大）")
    parser.add_argument("--save", type=str, default="arc_results.json",
                        help="保存详细结果到 JSON 文件")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🧩 ARC-AGI 基线评测器 v1.0")
    print(f"  模型: {args.model}  题数: {args.n}  模式: {args.mode}")
    print("=" * 60)

    # 初始化 Ollama 客户端
    client = OllamaClient(model=args.model, url=args.url)

    # 加载任务
    tasks = load_arc_tasks(n=args.n, mode=args.mode)
    if not tasks:
        print("  ❌ 没有加载到任何任务，退出")
        return

    print(f"\n  开始评测 {len(tasks)} 道题...\n")

    # 评测
    results = evaluate(tasks, client, max_tokens=args.max_tokens)

    # 输出摘要
    print_summary(results)

    # 保存结果
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  💾 详细结果已保存到 {args.save}")


if __name__ == "__main__":
    main()
