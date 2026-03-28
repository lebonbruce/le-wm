"""
arc_solver/verifier.py — 外部绝对验证器

这是 V20 架构中打破"衔尾蛇自蒸馏死循环"的核心组件。

设计原则：
  1. 精确匹配：候选 output 必须与 gold output 逐 cell 完全相同
  2. 交叉验证：程序必须在所有 train pairs 上通过才算有效
  3. 零容忍：部分正确 = 完全错误（ARC 评分标准）
  4. Ground Truth 注入：验证信号来自外部数据，不来自模型自身

为什么 Cosine/MSE 在这里无效（V19 教训）：
  - "把左上角 2x2 涂红" vs "把中上偏左 2x2 涂粉" → Cosine 0.85 → ARC 得分 0
  - 唯一可靠的信号是 cell-by-cell 精确匹配
"""

from typing import Optional
from arc_solver.sandbox_executor import execute_program, ExecutionResult


class VerificationResult:
    """单次验证的结果"""
    __slots__ = (
        "program_valid",       # 程序是否通过了所有 train pairs 的交叉验证
        "pass_count",          # 通过验证的 pair 数
        "total_count",         # 总 pair 数
        "cell_accuracy",       # 所有 pair 的平均 cell 精度（软指标，用于排序）
        "per_pair_results",    # 每个 pair 的详细结果
        "error_summary",       # 失败原因汇总
    )

    def __init__(self):
        self.program_valid = False
        self.pass_count = 0
        self.total_count = 0
        self.cell_accuracy = 0.0
        self.per_pair_results = []
        self.error_summary = ""

    def __repr__(self):
        status = "✅ VALID" if self.program_valid else "❌ INVALID"
        return (f"VerificationResult({status}, "
                f"pass={self.pass_count}/{self.total_count}, "
                f"cell_acc={self.cell_accuracy:.1f}%)")


class PairVerification:
    """单个 pair 的验证详情"""
    __slots__ = (
        "pair_index",
        "execution_ok",        # 程序是否成功执行
        "shape_match",         # 输出形状是否匹配
        "exact_match",         # 逐 cell 精确匹配
        "cells_correct",       # 正确的 cell 数
        "cells_total",         # 总 cell 数
        "error_message",       # 错误信息（如有）
    )

    def __init__(self, pair_index: int):
        self.pair_index = pair_index
        self.execution_ok = False
        self.shape_match = False
        self.exact_match = False
        self.cells_correct = 0
        self.cells_total = 0
        self.error_message = None


def _compare_grids(pred: list[list[int]],
                   gold: list[list[int]]) -> tuple[bool, int, int]:
    """
    逐 cell 精确比较两个 grid。

    返回: (exact_match, cells_correct, cells_total)
    """
    if len(pred) != len(gold):
        total = sum(len(row) for row in gold)
        return False, 0, total

    cells_correct = 0
    cells_total = 0
    exact = True

    for i, (pred_row, gold_row) in enumerate(zip(pred, gold)):
        if len(pred_row) != len(gold_row):
            exact = False
            cells_total += len(gold_row)
            # 部分比较
            for j in range(min(len(pred_row), len(gold_row))):
                cells_total_already = True  # 已在上面加过
                if pred_row[j] == gold_row[j]:
                    cells_correct += 1
            continue

        for j, (p, g) in enumerate(zip(pred_row, gold_row)):
            cells_total += 1
            if p == g:
                cells_correct += 1
            else:
                exact = False

    return exact, cells_correct, cells_total


def verify_program(code: str,
                   train_pairs: list[dict],
                   timeout_sec: float = 5.0) -> VerificationResult:
    """
    在所有 train pairs 上交叉验证候选程序。

    核心逻辑：
    1. 对每个 train pair: 用 input 执行程序 → 比较 output 与 gold
    2. 程序必须在所有 pair 上精确匹配才算 valid
    3. 即使 invalid，也计算 cell_accuracy 作为排序信号

    code: 候选 Python 程序文本
    train_pairs: [{"input": grid, "output": grid}, ...]
    timeout_sec: 每个 pair 的执行超时

    返回: VerificationResult
    """
    result = VerificationResult()
    result.total_count = len(train_pairs)
    all_exact = True
    total_correct = 0
    total_cells = 0
    errors = []

    for idx, pair in enumerate(train_pairs):
        pv = PairVerification(idx)
        gold_output = pair["output"]
        pv.cells_total = sum(len(row) for row in gold_output)

        # 执行程序
        exec_result = execute_program(code, pair["input"], timeout_sec)

        if not exec_result.success:
            pv.execution_ok = False
            pv.error_message = exec_result.error_message
            all_exact = False
            errors.append(f"pair[{idx}]: {exec_result.error_message}")
            total_cells += pv.cells_total
            result.per_pair_results.append(pv)
            continue

        pv.execution_ok = True
        pred_output = exec_result.output_grid

        # 形状检查
        if (len(pred_output) != len(gold_output) or
                any(len(pred_output[i]) != len(gold_output[i])
                    for i in range(len(gold_output)))):
            pv.shape_match = False
            pv.error_message = (
                f"形状不匹配: pred={len(pred_output)}x"
                f"{len(pred_output[0]) if pred_output else 0} vs "
                f"gold={len(gold_output)}x{len(gold_output[0])}")
            all_exact = False
            errors.append(f"pair[{idx}]: {pv.error_message}")
            # 仍然计算部分 cell 精度
            exact, correct, total = _compare_grids(pred_output, gold_output)
            pv.cells_correct = correct
            pv.cells_total = total
            total_correct += correct
            total_cells += total
            result.per_pair_results.append(pv)
            continue

        pv.shape_match = True

        # 逐 cell 精确比较
        exact, correct, total = _compare_grids(pred_output, gold_output)
        pv.exact_match = exact
        pv.cells_correct = correct
        pv.cells_total = total
        total_correct += correct
        total_cells += total

        if exact:
            result.pass_count += 1
        else:
            all_exact = False
            errors.append(
                f"pair[{idx}]: cell {correct}/{total} "
                f"({correct / max(total, 1) * 100:.0f}%) 不精确")

        result.per_pair_results.append(pv)

    # 汇总
    result.program_valid = all_exact and result.pass_count == result.total_count
    result.cell_accuracy = (total_correct / max(total_cells, 1)) * 100.0
    result.error_summary = "; ".join(errors) if errors else ""

    return result


def verify_on_test(code: str,
                   test_input: list[list[int]],
                   timeout_sec: float = 5.0) -> ExecutionResult:
    """
    在测试输入上执行已验证的程序，获取预测输出。

    这个方法只在程序已通过 train pairs 验证后调用。
    返回 ExecutionResult（包含 output_grid 或错误信息）。
    """
    return execute_program(code, test_input, timeout_sec)
