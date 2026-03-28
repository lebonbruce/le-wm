"""
arc_solver/sandbox_executor.py — 安全沙盒执行器

核心职责：在隔离环境中执行 LLM 生成的候选 Python 变换程序。
这是 V20 架构中"外部绝对验证"的基础设施层。

安全机制：
  1. 受限 globals（禁止 import, exec, eval, open, __builtins__ 子集）
  2. 超时控制（防止无限循环 / 指数爆炸）
  3. 异常捕获（语法错误、运行时错误 → 返回 None）
  4. 输出校验（grid 维度 / 值域合法性检查）

为什么不用 subprocess / Docker 隔离：
  项目本身已运行在 Docker 中，再套一层 subprocess 引入不必要的延迟。
  受限 globals + timeout 对 ARC 场景（纯 grid 操作）已经足够安全。
"""

import signal
import copy
import threading
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
#  受限执行环境
# ──────────────────────────────────────────────────────────────────────────────

# 允许在沙盒中使用的内置函数白名单
# 只保留纯计算和数据操作相关的，禁止一切 I/O 和代码执行
_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": lambda *args, **kwargs: None,  # 静默print，不允许输出
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}

# 在沙盒中禁止的关键字/函数调用（静态检查）
_FORBIDDEN_PATTERNS = [
    "import ",
    "from ",
    "__import__",
    "exec(",
    "eval(",
    "compile(",
    "open(",
    "os.",
    "sys.",
    "subprocess",
    "shutil",
    "pathlib",
    "__builtins__",
    "globals()",
    "locals()",
    "getattr(",
    "setattr(",
    "delattr(",
    "__class__",
    "__subclasses__",
]


def _static_safety_check(code: str) -> Optional[str]:
    """
    静态安全检查：在执行前扫描代码文本，拒绝包含危险模式的代码。

    返回: None 如果安全；str 错误描述如果不安全
    """
    code_lower = code.lower()
    for pattern in _FORBIDDEN_PATTERNS:
        if pattern.lower() in code_lower:
            return f"禁止使用 '{pattern}'"
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  Grid 合法性校验
# ──────────────────────────────────────────────────────────────────────────────

ARC_MAX_SIZE = 30
ARC_NUM_COLORS = 10


def validate_grid(grid: object) -> Optional[list[list[int]]]:
    """
    校验 grid 是否满足 ARC 规范：
    - 类型：list[list[int]]
    - 尺寸：1×1 ~ 30×30
    - 值域：0-9（10 种颜色）
    - 行长一致（矩形）

    返回: 校验后的标准化 grid，或 None（不合法）
    """
    if not isinstance(grid, list) or len(grid) == 0:
        return None
    if len(grid) > ARC_MAX_SIZE:
        return None

    width = None
    result = []
    for row in grid:
        if not isinstance(row, list) or len(row) == 0:
            return None
        if len(row) > ARC_MAX_SIZE:
            return None
        if width is None:
            width = len(row)
        elif len(row) != width:
            return None  # 非矩形

        int_row = []
        for cell in row:
            if not isinstance(cell, (int, float)):
                return None
            val = int(cell)
            if val < 0 or val >= ARC_NUM_COLORS:
                return None
            int_row.append(val)
        result.append(int_row)

    return result


# ──────────────────────────────────────────────────────────────────────────────
#  带超时的沙盒执行
# ──────────────────────────────────────────────────────────────────────────────

class ExecutionResult:
    """沙盒执行的结果封装"""
    __slots__ = ("success", "output_grid", "error_message")

    def __init__(self, success: bool,
                 output_grid: Optional[list[list[int]]] = None,
                 error_message: Optional[str] = None):
        self.success = success
        self.output_grid = output_grid
        self.error_message = error_message

    def __repr__(self):
        if self.success:
            h = len(self.output_grid)
            w = len(self.output_grid[0]) if self.output_grid else 0
            return f"ExecutionResult(success=True, grid={h}x{w})"
        return f"ExecutionResult(success=False, error='{self.error_message}')"


def execute_program(code: str, input_grid: list[list[int]],
                    timeout_sec: float = 5.0) -> ExecutionResult:
    """
    在受限沙盒中执行候选变换程序。

    程序约定：
    - 接收变量 `input_grid`: list[list[int]]，原始输入 grid
    - 必须将结果赋值给变量 `output_grid`: list[list[int]]
    - 不得使用 import / exec / eval 等危险操作

    code: 候选 Python 程序文本
    input_grid: ARC 输入 grid（深拷贝传入，防止变异）
    timeout_sec: 最大执行时间（秒）

    返回: ExecutionResult
    """
    # 1. 静态安全检查
    safety_error = _static_safety_check(code)
    if safety_error is not None:
        return ExecutionResult(False, error_message=f"安全检查失败: {safety_error}")

    # 2. 语法检查
    try:
        compiled = compile(code, "<sandbox>", "exec")
    except SyntaxError as e:
        return ExecutionResult(False, error_message=f"语法错误: {e}")

    # 3. 构建受限执行环境
    safe_globals = {"__builtins__": _SAFE_BUILTINS}
    # 深拷贝输入 grid，防止程序修改影响原始数据
    safe_globals["input_grid"] = copy.deepcopy(input_grid)

    # 4. 带超时的执行（使用线程 + join timeout）
    result_container = {"output_grid": None, "error": None}

    def _run():
        try:
            exec(compiled, safe_globals)
            result_container["output_grid"] = safe_globals.get("output_grid")
        except Exception as e:
            result_container["error"] = f"运行时错误: {type(e).__name__}: {e}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if thread.is_alive():
        # 超时：线程无法直接杀死（Python 限制），但 daemon=True 确保主线程退出时清理
        return ExecutionResult(False, error_message=f"执行超时 ({timeout_sec}s)")

    # 5. 检查执行结果
    if result_container["error"] is not None:
        return ExecutionResult(False, error_message=result_container["error"])

    raw_output = result_container["output_grid"]
    if raw_output is None:
        return ExecutionResult(
            False, error_message="程序未定义 output_grid 变量")

    # 6. Grid 合法性校验
    validated = validate_grid(raw_output)
    if validated is None:
        return ExecutionResult(
            False, error_message="输出 grid 不合法（维度/值域/类型错误）")

    return ExecutionResult(True, output_grid=validated)


def execute_on_pairs(code: str,
                     pairs: list[dict],
                     timeout_sec: float = 5.0) -> list[ExecutionResult]:
    """
    在多个 (input, output) 对上执行同一程序。

    pairs: [{"input": grid, "output": grid}, ...]
    返回: 每个 pair 的 ExecutionResult 列表
    """
    results = []
    for pair in pairs:
        result = execute_program(code, pair["input"], timeout_sec)
        results.append(result)
    return results
