"""
auto_expand.py —— 自动知识图谱扩展器

利用 Qwen2.5 LLM 从马斯洛需求出发，递归拆解人类社会全景知识图谱。

核心机制：
1. 从种子节点（马斯洛 5 层需求）出发
2. BFS 逐层扩展：对每个"待展开"节点，让 LLM 生成 5-8 个子节点
3. 每个子节点自带边类型（供需关系 / 产业链 / 行业职业 / 工序链 / 原材料）
4. 自动写入海马体知识图谱
5. 每积累一批（batch_size）新知识后，训练 JEPA 世界模型
6. 循环直到达到目标节点数或展开深度

运行方式：
    docker exec v19-dev python auto_expand.py --target_nodes 1000 --max_depth 6
"""
import json
import os
import re
import sys
import time
import torch
from mvp_config import config
from cortex import LinguisticCortex
from hippocampus import HippocampalMemory
from brain import TheBrainMVP


# ---- 拆解 Prompt 模板（简化版，适配 0.5B 小模型） ----
DECOMPOSE_PROMPT = """请将"{node_text}"细分为5个具体的子概念。每行一个，格式：子概念描述 [边类型]

边类型从以下选一个：需求供给、产业链、行业职业、工序链、原材料、产出物、组成部分

示例（细分"农业"）：
种植业负责粮食和蔬菜的生产 [组成部分]
畜牧业提供肉蛋奶等动物蛋白 [组成部分]
农民是种植业的核心劳动者 [行业职业]
农业需要土地和水资源 [原材料]
农产品运往食品加工厂 [产业链]

请细分"{node_text}"："""

# ---- 生成对话 Prompt（简化版） ----
DIALOGUE_PROMPT = """根据以下知识写一个问答。
知识：{parent} 包含 {children}
格式：
问：（问题）
答：（回答，30字以内）"""


class KnowledgeGraphExpander:
    """
    自动知识图谱扩展器。

    BFS 策略：
    1. 初始化种子节点（马斯洛 5 层需求）
    2. 从 expansion_queue 取出待展开节点
    3. 调用 LLM 生成子节点
    4. 写入海马体
    5. 将 expand=true 的子节点加入队列
    6. 每 train_interval 个新节点后自动训练 JEPA
    """

    def __init__(self, brain: TheBrainMVP,
                 ollama_model: str = "qwen3:8b",
                 ollama_url: str = "http://host.docker.internal:11434"):
        self.brain = brain
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        # 待展开队列：[(node_id, node_text, depth, context_path)]
        self.expansion_queue = []
        # 已展开节点集合（防重复）
        self.expanded_texts = set()
        # 统计
        self.total_nodes = 0
        self.total_edges = 0
        # 新增的对话对（用于 JEPA 训练）
        self.pending_dialogues = []

    def _call_llm(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        通过 Ollama HTTP API (/api/chat) 调用本地大模型生成回复。
        Qwen3 thinking 模式下 num_predict 需要足够大，因为 thinking + content 共享 token 配额。
        只读取 message.content（实际回答），忽略 message.thinking（内部推理）。
        """
        import urllib.request
        import json as json_mod

        payload = json_mod.dumps({
            "model": self.ollama_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            },
            "think": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        resp = urllib.request.urlopen(req, timeout=600)
        raw_bytes = resp.read()
        result = json_mod.loads(raw_bytes.decode("utf-8-sig"))
        msg = result.get("message", {})
        # 只读取 content（实际回答），忽略 thinking（内部推理链）
        content = msg.get("content", "").strip()
        # 剥离可能残留的 <think>...</think> 标签
        if "</think>" in content:
            content = content.split("</think>", 1)[1].strip()
        return content

    def _parse_line_children(self, response: str) -> list:
        """
        解析逐行格式的 LLM 输出，带严格质量过滤。
        只接受包含 [边类型] 标签的行——这是区分实际回答和思考内容的关键标志。
        """
        valid_edge_types = {
            "需求供给", "产业链", "行业职业", "工序链",
            "原材料", "产出物", "组成部分"
        }
        results = []
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            # 去除行首序号
            line = re.sub(r'^[\d]+[.、)\s]+', '', line)
            line = re.sub(r'^[-*·]\s*', '', line)

            # 必须包含 [边类型] 标签——这是区分回答和思考的核心
            match = re.search(r'\[([^\]]+)\]\s*$', line)
            if not match:
                continue

            edge_type = match.group(1).strip()
            text = line[:match.start()].strip()

            # 边类型必须是预定义的
            if edge_type not in valid_edge_types:
                continue

            # 质量门槛
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            if chinese_chars < 3 or len(text) < 6 or len(text) > 80:
                continue

            results.append({"text": text, "edge_type": edge_type})

        return results

    def seed_maslow(self):
        """播种马斯洛 5 层需求作为根节点。"""
        seeds = [
            "人类生理需求：食物、水、睡眠、呼吸、衣物、住所——这是一切生存活动的基础",
            "人类安全需求：人身安全、健康保障、财产安全、工作稳定、社会秩序",
            "人类社交需求：友情、爱情、亲情、归属感、社群认同、社会参与",
            "人类尊重需求：自我尊重、被他人认可、专业成就、社会地位、自信心",
            "人类自我实现需求：发挥最大潜能、追求理想、创造价值、精神自由、终身成长",
        ]
        print("\n" + "=" * 60)
        print("  知识图谱自动扩展器 v1.0")
        print(f"  种子: 马斯洛 5 层需求")
        print("=" * 60)

        for i, seed_text in enumerate(seeds):
            nid = self.brain.ingest(seed_text)
            self.expansion_queue.append((nid, seed_text, 0, seed_text[:10]))
            self.expanded_texts.add(seed_text)
            self.total_nodes += 1

        print(f"\n  ✅ 种子播种完成: {len(seeds)} 个根节点")

    def expand_one_node(self, node_id: str, node_text: str,
                        depth: int, context_path: str) -> int:
        """展开一个节点，返回新增子节点数。"""
        if node_text in self.expanded_texts and depth > 0:
            return 0

        self.expanded_texts.add(node_text)

        # 调用 LLM 拆解（简短版 prompt）
        short_node = node_text[:50]
        prompt = DECOMPOSE_PROMPT.format(node_text=short_node)
        response = self._call_llm(prompt, max_tokens=300)
        children = self._parse_line_children(response)

        if not children:
            print(f"    ⚠ LLM 未返回有效子节点: {short_node}...")
            return 0

        new_count = 0
        child_texts = []
        for child in children[:8]:  # 最多取 8 个
            text = child["text"]
            edge_type = child["edge_type"]

            # 去重
            if text in self.expanded_texts:
                continue

            # 写入海马体
            child_id = self.brain.ingest(text, node_id, edge_type)
            self.total_nodes += 1
            self.total_edges += 1
            new_count += 1
            child_texts.append(text)
            self.expanded_texts.add(text)

            # 所有子节点都加入展开队列（由 max_depth 控制深度）
            new_path = f"{context_path} → {text[:15]}"
            self.expansion_queue.append(
                (child_id, text, depth + 1, new_path))

        # 为这批新知识生成对话对
        if child_texts:
            self._generate_dialogue(node_text, child_texts)

        return new_count

    def _generate_dialogue(self, parent_text: str, child_texts: list):
        """用 LLM 为新知识生成训练用的对话对。"""
        children_str = "、".join(t[:15] for t in child_texts[:5])
        prompt = DIALOGUE_PROMPT.format(
            parent=parent_text[:30], children=children_str)
        response = self._call_llm(prompt, max_tokens=150)

        # 解析 "问：.../答：..." 格式
        q_match = re.search(r'问[：:]\s*(.+?)(?:\n|$)', response)
        a_match = re.search(r'答[：:]\s*(.+?)(?:\n|$)', response)
        if q_match and a_match:
            q = q_match.group(1).strip()
            a = a_match.group(1).strip()
            if len(q) > 3 and len(a) > 3:
                self.pending_dialogues.append((q, a))

    def train_jepa_on_new_knowledge(self):
        """用积累的新对话对训练 JEPA 世界模型。"""
        if not self.pending_dialogues:
            return

        n = len(self.pending_dialogues)
        print(f"\n  📚 训练 JEPA（{n} 条新对话）...")

        # 自监督预训练
        self.brain.pretrain_encoder_phase(self.pending_dialogues, epochs=10)
        # 深度做梦
        self.brain.train_dream_phase(self.pending_dialogues, epochs=20)

        self.pending_dialogues = []
        print(f"  ✅ JEPA 训练完成")

    def run(self, target_nodes: int = 500, max_depth: int = 5,
            train_interval: int = 30):
        """
        主循环：BFS 展开 + 定期训练。

        target_nodes: 目标总节点数
        max_depth: 最大展开深度
        train_interval: 每新增 N 个节点后训练一次 JEPA
        """
        self.seed_maslow()

        nodes_since_last_train = 0
        round_num = 0

        while self.expansion_queue and self.total_nodes < target_nodes:
            # 从队列头部取出
            node_id, node_text, depth, context_path = self.expansion_queue.pop(0)

            if depth > max_depth:
                continue

            round_num += 1
            short_text = node_text[:40].replace('\n', ' ')
            print(f"\n  🔍 [{round_num}] 深度={depth} | "
                  f"展开: {short_text}...")

            new_count = self.expand_one_node(
                node_id, node_text, depth, context_path)
            nodes_since_last_train += new_count

            print(f"    → +{new_count} 子节点 | "
                  f"总计: {self.total_nodes} 节点, "
                  f"{self.total_edges} 边 | "
                  f"队列: {len(self.expansion_queue)}")

            # 定期训练 JEPA
            if nodes_since_last_train >= train_interval:
                self.train_jepa_on_new_knowledge()
                nodes_since_last_train = 0

        # 最后一次训练
        if self.pending_dialogues:
            self.train_jepa_on_new_knowledge()

        print(f"\n{'=' * 60}")
        print(f"  🏁 扩展完成!")
        print(f"  总节点: {self.total_nodes}")
        print(f"  总边: {self.total_edges}")
        print(f"  图谱: {self.brain.hippocampus.stats()}")
        print(f"  JEPA 经验池: {len(self.brain.jepa.replay_buffer)}"
              f"/{config.replay_buffer_size}")
        print(f"{'=' * 60}")

    def test_knowledge(self, queries: list = None):
        """测试图谱知识是否被 JEPA 吸收。"""
        if queries is None:
            queries = [
                "食物是怎么从农田到达我们餐桌的？",
                "建一栋房子需要哪些人和材料？",
                "为什么人需要社交？",
                "一个面包是怎么做出来的？",
            ]

        print(f"\n{'=' * 60}")
        print(f"  知识图谱推理测试")
        print(f"{'=' * 60}")

        for q in queries:
            self.brain.benchmark_interact(q)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="自动知识图谱扩展器")
    parser.add_argument("--target_nodes", type=int, default=200,
                        help="目标节点数（默认 200）")
    parser.add_argument("--max_depth", type=int, default=4,
                        help="最大展开深度（默认 4）")
    parser.add_argument("--train_interval", type=int, default=30,
                        help="每 N 个新节点训练一次 JEPA（默认 30）")
    parser.add_argument("--ollama_model", type=str, default="qwen3:4b",
                        help="Ollama 模型名（默认 qwen3:4b）")
    parser.add_argument("--ollama_url", type=str,
                        default="http://host.docker.internal:11434",
                        help="Ollama API 地址（默认 host.docker.internal:11434）")
    args = parser.parse_args()

    # 初始化大脑（0.5B 用于 JEPA 训练/推理）
    brain = TheBrainMVP()

    # 创建扩展器（大模型用于知识生成）
    print(f"\n  📡 Ollama 模型: {args.ollama_model}")
    print(f"  📡 Ollama 地址: {args.ollama_url}")
    expander = KnowledgeGraphExpander(
        brain, ollama_model=args.ollama_model, ollama_url=args.ollama_url)

    # 运行自动扩展
    expander.run(
        target_nodes=args.target_nodes,
        max_depth=args.max_depth,
        train_interval=args.train_interval,
    )

    # 测试推理
    expander.test_knowledge()

    # 持久化验证
    print(f"\n最终图谱: {brain.hippocampus.stats()}")
    brain.hippocampus.close()


if __name__ == "__main__":
    main()
