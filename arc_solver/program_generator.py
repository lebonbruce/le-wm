"""
arc_solver/program_generator.py — LLM 驱动的候选程序生成器

核心职责：
  将 ARC 任务的 train pairs 交给 LLM，让它归纳变换规则并输出可执行 Python 代码。

设计要点：
  1. Prompt 工程：向 LLM 精确描述 ARC 任务格式和输出要求
  2. 多候选采样：通过 temperature sampling 生成多个不同的候选程序
  3. 代码提取：从 LLM 自由格式回复中鲁棒提取 Python 代码块
  4. 后端选择：默认使用 Ollama API（qwen3:8b），与 arc_eval.py 复用基础设施

为什么用 LLM 而非 DSL 搜索：
  - ARC 规则种类极其多样（颜色映射、几何变换、拓扑操作、计数逻辑等）
  - 可枚举的 DSL 要覆盖所有规则类型 → 组合爆炸
  - LLM 的 few-shot ICL 能力天然适合从示例归纳规则
  - LLM 生成的 Python 代码具有图灵完备的表达力
"""

import json
import re
import urllib.request
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
#  Grid 文本化（与 arc_eval.py 保持一致）
# ──────────────────────────────────────────────────────────────────────────────

# 用数字直接表示 grid（比符号更不容易被 LLM 误解）
def grid_to_text(grid: list[list[int]]) -> str:
    """将 Grid 转为文本。每行用空格分隔数字，行间换行。"""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def grid_dimensions(grid: list[list[int]]) -> str:
    """返回 grid 的尺寸描述"""
    h = len(grid)
    w = len(grid[0]) if grid else 0
    return f"{h}x{w}"


# ──────────────────────────────────────────────────────────────────────────────
#  Prompt 构建
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert ARC-AGI puzzle solver. Your task is to analyze input-output grid pairs, identify the transformation rule, and write a Python function that implements it.

CRITICAL RULES:
1. Read the variable `input_grid` (a list of lists of ints, values 0-9)
2. Write your transformation and assign the result to `output_grid`
3. Do NOT use import statements
4. Do NOT use exec, eval, open, or any I/O operations
5. You can use basic Python: loops, conditions, list comprehensions, len, range, etc.
6. Your code must be deterministic (no random)
7. Colors are integers 0-9

COMMON ARC PATTERNS to look for:
- Color mapping/replacement
- Copying/moving objects
- Reflection (horizontal/vertical)
- Rotation (90/180/270 degrees)
- Scaling (upscale/downscale)
- Filling regions
- Counting objects
- Pattern completion
- Boolean operations on grids
- Border/frame operations
- Gravity/stacking
- Connected component operations"""


def build_generation_prompt(task: dict,
                            attempt_num: int = 0,
                            previous_errors: list[str] = None) -> str:
    """
    构建向 LLM 请求生成变换程序的 prompt。

    task: ARC 任务 dict (含 train pairs 和 test pair)
    attempt_num: 第几次尝试（0-based），影响 prompt 中的引导
    previous_errors: 之前失败尝试的错误信息（用于自我纠正）

    返回: prompt 文本
    """
    lines = [_SYSTEM_PROMPT, ""]

    # 展示所有训练示例
    train_pairs = task.get("train", [])
    lines.append(f"=== TASK ({len(train_pairs)} training examples) ===\n")

    for i, pair in enumerate(train_pairs, 1):
        inp = pair["input"]
        out = pair["output"]
        lines.append(f"Example {i}:")
        lines.append(f"  Input ({grid_dimensions(inp)}):")
        for row in inp:
            lines.append(f"    {row}")
        lines.append(f"  Output ({grid_dimensions(out)}):")
        for row in out:
            lines.append(f"    {row}")
        lines.append("")

    # 展示测试输入
    test_pair = task.get("test", [{}])[0]
    test_input = test_pair.get("input", [[]])
    lines.append(f"Test Input ({grid_dimensions(test_input)}):")
    for row in test_input:
        lines.append(f"  {row}")
    lines.append("")

    # 引导不同的思考方向
    if attempt_num == 0:
        lines.append(
            "Analyze the transformation from input to output. "
            "What rule transforms each input into its output? "
            "Write Python code that implements this rule.")
    elif attempt_num == 1:
        lines.append(
            "Look carefully at the spatial structure and colors. "
            "Consider: rotations, reflections, color swaps, object movements, "
            "or pattern repetitions. Write Python code.")
    elif attempt_num == 2:
        lines.append(
            "Think about this differently. Consider connected components, "
            "symmetry, counting, filling, or border operations. "
            "Write Python code.")
    else:
        lines.append(
            "Try an entirely different approach. Consider combinations of "
            "basic operations: crop, pad, tile, transpose, color remap. "
            "Write Python code.")

    # 自我纠正：展示之前失败的错误信息
    if previous_errors:
        lines.append("")
        lines.append("PREVIOUS ATTEMPTS FAILED:")
        for err in previous_errors[-3:]:  # 最多展示最近 3 个错误
            lines.append(f"  - {err}")
        lines.append("Please try a DIFFERENT approach.")

    lines.append("")
    lines.append(
        "Write ONLY the Python code. "
        "Read from `input_grid`, write to `output_grid`. "
        "Wrap your code in ```python ... ``` markers.")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
#  代码提取
# ──────────────────────────────────────────────────────────────────────────────

def extract_code_from_response(response: str) -> Optional[str]:
    """
    从 LLM 回复中提取 Python 代码块。

    提取策略（按优先级）：
    1. ```python ... ``` 标记的代码块
    2. ``` ... ``` 通用代码块
    3. 如果整个回复看起来像 Python 代码（包含 = 和缩进），直接使用

    返回: 提取的代码文本，或 None
    """
    # 策略 1: ```python ... ``` 标记
    pattern_python = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern_python, response, re.DOTALL)
    if matches:
        return matches[-1].strip()  # 取最后一个（通常是最终版本）

    # 策略 2: ``` ... ``` 通用标记
    pattern_generic = r"```\s*\n(.*?)```"
    matches = re.findall(pattern_generic, response, re.DOTALL)
    if matches:
        code = matches[-1].strip()
        # 快速检查是否像 Python 代码
        if "output_grid" in code or "input_grid" in code:
            return code

    # 策略 3: 整个回复作为代码（如果包含关键标记）
    if "output_grid" in response and "input_grid" in response:
        # 移除明显的非代码行
        code_lines = []
        in_code = False
        for line in response.split("\n"):
            stripped = line.strip()
            # 跳过以 # 开头的说明性文本，保留 Python 注释
            if stripped.startswith("output_grid") or stripped.startswith(
                    "input_grid") or stripped.startswith("for ") or \
                    stripped.startswith("if ") or stripped.startswith(
                    "def ") or stripped.startswith("#") or \
                    stripped.startswith("    ") or stripped == "" or \
                    "=" in stripped:
                code_lines.append(line)
                in_code = True
            elif in_code and (stripped.startswith("    ") or stripped == ""):
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

    return None


# ──────────────────────────────────────────────────────────────────────────────
#  Ollama LLM 客户端
# ──────────────────────────────────────────────────────────────────────────────

class ProgramGeneratorLLM:
    """
    通过 Ollama API 调用 LLM 生成候选 Python 变换程序。

    复用 arc_eval.py 中验证过的 Ollama 连接方式。
    """

    def __init__(self,
                 model: str = "qwen3:8b",
                 ollama_url: str = "http://host.docker.internal:11434",
                 max_tokens: int = 1024):
        self.model = model
        self.url = ollama_url
        self.max_tokens = max_tokens
        self._connected = False

        # 测试连通性
        try:
            req = urllib.request.Request(f"{self.url}/api/tags")
            urllib.request.urlopen(req, timeout=10)
            self._connected = True
            print(f"  ✅ ProgramGenerator: Ollama 连接成功 ({model})")
        except Exception as e:
            print(f"  ⚠ ProgramGenerator: Ollama 不可达 ({self.url}): {e}")
            print(f"     将使用 fallback 模板程序")

    def generate_raw(self, prompt: str,
                     temperature: float = 0.7) -> Optional[str]:
        """
        调用 Ollama 生成原始回复文本。

        temperature: 采样温度（0=贪心，1=创造性，>1=高随机性）
        返回: 生成的文本，或 None (失败时)
        """
        if not self._connected:
            return None

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": temperature,
            },
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{self.url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=120)
            raw = resp.read().decode("utf-8-sig")
            data = json.loads(raw)
            content = data.get("message", {}).get("content", "").strip()
            # 处理 think 标签
            if "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            return content
        except Exception as e:
            print(f"  ⚠ Ollama 调用失败: {e}")
            return None

    def generate_candidates(self,
                            task: dict,
                            n_candidates: int = 8,
                            temperatures: list[float] = None,
                            previous_errors: list[str] = None,
                            ) -> list[str]:
        """
        为一道 ARC 题目生成多个候选 Python 程序。

        task: ARC 任务 dict
        n_candidates: 生成的候选程序数量
        temperatures: 每个候选的采样温度，None 则自动分配
        previous_errors: 之前失败的错误信息（用于自我纠正循环）

        返回: 提取出的 Python 代码列表（已去重）
        """
        if temperatures is None:
            # 默认温度梯度：从确定性到探索性
            temperatures = [
                0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5
            ][:n_candidates]
            # 如果需要更多候选，重复高温区间
            while len(temperatures) < n_candidates:
                temperatures.append(0.7 + len(temperatures) * 0.1)

        codes = set()  # 去重
        code_list = []

        for i in range(n_candidates):
            temp = temperatures[i] if i < len(temperatures) else 0.7
            prompt = build_generation_prompt(
                task, attempt_num=i, previous_errors=previous_errors)

            raw_response = self.generate_raw(prompt, temperature=temp)
            if raw_response is None:
                continue

            code = extract_code_from_response(raw_response)
            if code is None:
                continue

            # 去重：相同代码不重复添加
            code_normalized = code.strip()
            if code_normalized not in codes:
                codes.add(code_normalized)
                code_list.append(code_normalized)

        return code_list


# ──────────────────────────────────────────────────────────────────────────────
#  Fallback: 基于模板的简单程序生成（无 LLM 时使用）
# ──────────────────────────────────────────────────────────────────────────────

def generate_template_candidates(task: dict) -> list[str]:
    """
    无 LLM 时的 fallback：生成一组基于常见 ARC 模式的模板程序。

    这些模板覆盖最常见的 ARC 变换类型。
    虽然覆盖率有限，但提供了一个不依赖外部服务的基线。
    """
    templates = []

    # 模板 1: 直接复制（identity 变换 — 极少数题目的正确答案）
    templates.append(
        "output_grid = [row[:] for row in input_grid]"
    )

    # 模板 2: 水平翻转
    templates.append(
        "output_grid = [row[::-1] for row in input_grid]"
    )

    # 模板 3: 垂直翻转
    templates.append(
        "output_grid = input_grid[::-1]"
    )

    # 模板 4: 转置
    templates.append("""
h = len(input_grid)
w = len(input_grid[0])
output_grid = [[input_grid[r][c] for r in range(h)] for c in range(w)]
""")

    # 模板 5: 90度旋转
    templates.append("""
h = len(input_grid)
w = len(input_grid[0])
output_grid = [[input_grid[h - 1 - r][c] for r in range(h)] for c in range(w)]
""")

    # 模板 6: 颜色计数后填充最多的颜色
    templates.append("""
h = len(input_grid)
w = len(input_grid[0])
counts = {}
for r in range(h):
    for c in range(w):
        v = input_grid[r][c]
        counts[v] = counts.get(v, 0) + 1
# 移除背景色 0 后找最多的颜色
non_bg = {k: v for k, v in counts.items() if k != 0}
if non_bg:
    dominant = max(non_bg, key=non_bg.get)
else:
    dominant = 0
output_grid = [[dominant if input_grid[r][c] != 0 else 0
                 for c in range(w)] for r in range(h)]
""")

    # 模板 7: 去除背景色，保留非零区域（crop to bounding box）
    templates.append("""
h = len(input_grid)
w = len(input_grid[0])
min_r, max_r, min_c, max_c = h, -1, w, -1
for r in range(h):
    for c in range(w):
        if input_grid[r][c] != 0:
            min_r = min(min_r, r)
            max_r = max(max_r, r)
            min_c = min(min_c, c)
            max_c = max(max_c, c)
if max_r < 0:
    output_grid = [[0]]
else:
    output_grid = [input_grid[r][min_c:max_c+1]
                   for r in range(min_r, max_r+1)]
""")

    # 模板 8: 上下左右各方向翻转并比较
    templates.append("""
h = len(input_grid)
w = len(input_grid[0])
# 180度旋转
output_grid = [row[::-1] for row in input_grid[::-1]]
""")

    return [t.strip() for t in templates]
