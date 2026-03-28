"""
logic/data_gen.py —— 合成逻辑推理数据集生成器

核心能力:
1. 程序化生成家族关系图（DAG）
2. 从图中提取多步推理链
3. 生成正样本（可推导）和负样本（不可推导）
4. 推理类型: 传递闭包、逆关系、对称关系、属性继承、组合推理

设计目标: 数据无限生成、复杂度可控、与 v19 TransE 三元组格式完全兼容
"""
import random
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from logic.logic_config import logic_config


class FamilyTree:
    """
    程序化家族树生成器。

    生成过程保证:
    - 父/母关系构成 DAG（无环）
    - 婚姻关系对称
    - 代际关系一致（同代不会互为父子）
    """

    # 人名池（确保多样性）
    FIRST_NAMES = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace",
        "Henry", "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia",
        "Paul", "Quinn", "Rose", "Sam", "Tina", "Uma", "Victor", "Wendy",
        "Xavier", "Yuki", "Zoe", "Aaron", "Beth", "Carl", "Dora",
        "Ethan", "Fiona", "George", "Helen", "Ian", "Julia", "Kevin",
        "Lily", "Mike", "Nina", "Oscar", "Penny", "Ray", "Sara",
        "Tom", "Ursula", "Vince", "Wanda", "Xena", "Yara",
        "Alex", "Bella", "Caleb", "Daisy", "Eli", "Faye", "Gus",
        "Hana", "Iris", "Joel", "Kira", "Liam", "Maya", "Nora",
        "Owen", "Pia", "Rex", "Sia", "Troy", "Una", "Vera",
        "Wade", "Xia", "Yael", "Zara", "Amos", "Bria", "Cruz",
        "Devi", "Elio", "Finn", "Glen", "Hope", "Ivan", "Jade",
        "Kent", "Luna", "Max", "Nell", "Otis", "Remy", "Sky",
        "Theo", "Uri", "Val", "Wren", "Yves", "Zion", "Alma",
    ]

    def __init__(self, num_entities: int = None):
        self.num_entities = num_entities or logic_config.num_entities
        self.entities: List[str] = []
        # 关系存储: (head, relation, tail) 三元组集合
        self.base_facts: Set[Tuple[str, str, str]] = set()
        # 代际信息: entity → generation (0 = 最老一代)
        self.generation: Dict[str, int] = {}
        self._name_counter: int = 0  # 名字池耗尽后用于生成编号名
        # 性别信息（用于区分 father/mother）
        self.gender: Dict[str, str] = {}

    def generate(self, num_generations: int = None,
                 children_per_couple: int = None) -> None:
        """
        生成完整家族树。

        num_generations: 代数（10 代可产生 10 步 ancestor 链）
        children_per_couple: 每对夫妻的孩子数
        """
        num_generations = num_generations or logic_config.num_generations
        children_per_couple = children_per_couple or logic_config.children_per_couple
        self.entities = []
        self.base_facts = set()
        self.generation = {}
        self.gender = {}
        self._name_counter = 0

        # 洗牌名字池
        names = list(self.FIRST_NAMES)
        random.shuffle(names)
        self._name_pool = list(names)
        self._name_pool_idx = 0

        def next_name() -> str:
            if self._name_pool_idx < len(self._name_pool):
                name = self._name_pool[self._name_pool_idx]
                self._name_pool_idx += 1
                return name
            # 名字池耗尽：生成编号名
            self._name_counter += 1
            return f"Person_{self._name_counter}"

        # 第 0 代: 创建初始夫妻
        # 对于深度家族树(10代+)，需要足够多的初始夫妻
        # 保证每代都有跨家庭婚配的可能性
        gen0_couples = max(4, min(self.num_entities // 4, 10))

        for _ in range(gen0_couples):
            husband = next_name()
            wife = next_name()
            self.entities.extend([husband, wife])
            self.gender[husband] = "male"
            self.gender[wife] = "female"
            self.generation[husband] = 0
            self.generation[wife] = 0
            # 婚姻关系（对称）
            self.base_facts.add((husband, "married_to", wife))
            self.base_facts.add((wife, "married_to", husband))

        # 逐代繁衍
        current_gen_couples = [
            (e1, e2) for (e1, r, e2) in self.base_facts
            if r == "married_to" and self.gender.get(e1) == "male"
        ]

        for gen_idx in range(1, num_generations):
            next_gen_couples = []
            for father, mother in current_gen_couples:
                num_children = random.randint(1, children_per_couple)
                children = []
                for _ in range(num_children):
                    try:
                        child = next_name()
                    except StopIteration:
                        break
                    child_gender = random.choice(["male", "female"])
                    self.entities.append(child)
                    self.gender[child] = child_gender
                    self.generation[child] = gen_idx
                    # 父/母关系
                    self.base_facts.add((father, "father_of", child))
                    self.base_facts.add((mother, "mother_of", child))
                    children.append(child)

                # 部分同代孩子配对结婚（不近亲：只有来自不同家庭的）
                # 这在后面统一处理

            # 跨家庭配对：从不同夫妻的后代中配对
            gen_children = [
                e for e in self.entities
                if self.generation.get(e) == gen_idx
            ]
            random.shuffle(gen_children)
            males = [e for e in gen_children if self.gender[e] == "male"]
            females = [e for e in gen_children if self.gender[e] == "female"]

            # 跨家庭配对（检查不是兄妹）
            for m, f in zip(males, females):
                m_parents = self._get_parents(m)
                f_parents = self._get_parents(f)
                # 不共享任何父母 → 可以结婚
                if not m_parents.intersection(f_parents):
                    self.base_facts.add((m, "married_to", f))
                    self.base_facts.add((f, "married_to", m))
                    next_gen_couples.append((m, f))

            current_gen_couples = next_gen_couples
            if not current_gen_couples:
                break

    def _get_parents(self, entity: str) -> Set[str]:
        """获取实体的所有父母"""
        parents = set()
        for h, r, t in self.base_facts:
            if t == entity and r in ("father_of", "mother_of"):
                parents.add(h)
        return parents

    def get_all_base_facts(self) -> List[Tuple[str, str, str]]:
        """返回所有基础事实的三元组列表"""
        return list(self.base_facts)


class InferenceRule:
    """
    一条逻辑推理规则。

    格式: 前提 → 结论
    例如: father_of(A,B) ∧ father_of(B,C) → grandparent_of(A,C)
    """
    def __init__(self, name: str,
                 premises: List[Tuple[str, str]],
                 conclusion: Tuple[str, str],
                 binding: str = "chain"):
        """
        name: 规则名称
        premises: [(rel, direction), ...] 前提关系列表
                  direction = "fwd" 表示 (X_i, rel, X_{i+1})
                  direction = "rev" 表示 (X_{i+1}, rel, X_i)
        conclusion: (rel, direction) 结论关系
                    "fwd" → (X_0, rel, X_n)
                    "rev" → (X_n, rel, X_0)
        binding: "chain" = 链式 A→B→C, "shared_parent" = A←P→B
        """
        self.name = name
        self.premises = premises
        self.conclusion = conclusion
        self.binding = binding

    def __repr__(self):
        prem_str = " ∧ ".join(f"{r}({d})" for r, d in self.premises)
        conc_r, conc_d = self.conclusion
        return f"{prem_str} → {conc_r}({conc_d})"


# 预定义推理规则集
INFERENCE_RULES = [
    # 1. 父/母 → 父母
    InferenceRule(
        "parent_from_father",
        premises=[("father_of", "fwd")],
        conclusion=("parent_of", "fwd"),
        binding="chain",
    ),
    InferenceRule(
        "parent_from_mother",
        premises=[("mother_of", "fwd")],
        conclusion=("parent_of", "fwd"),
        binding="chain",
    ),
    # 2. parent_of 的逆 → child_of
    InferenceRule(
        "child_from_parent",
        premises=[("parent_of", "fwd")],
        conclusion=("child_of", "rev"),
        binding="chain",
    ),
    # 3. 传递闭包: parent ∘ parent → grandparent
    InferenceRule(
        "grandparent_2hop",
        premises=[("parent_of", "fwd"), ("parent_of", "fwd")],
        conclusion=("grandparent_of", "fwd"),
        binding="chain",
    ),
    # 4. grandparent 的逆
    InferenceRule(
        "grandchild_2hop",
        premises=[("parent_of", "fwd"), ("parent_of", "fwd")],
        conclusion=("grandchild_of", "rev"),
        binding="chain",
    ),
    # 5. 传递闭包: parent → ancestor（任意长度）
    InferenceRule(
        "ancestor_from_parent",
        premises=[("parent_of", "fwd")],
        conclusion=("ancestor_of", "fwd"),
        binding="chain",
    ),
    # 6. 传递闭包的传递性: ancestor ∘ parent → ancestor
    InferenceRule(
        "ancestor_transitive",
        premises=[("ancestor_of", "fwd"), ("parent_of", "fwd")],
        conclusion=("ancestor_of", "fwd"),
        binding="chain",
    ),
    # 7. ancestor 的逆
    InferenceRule(
        "descendant_from_ancestor",
        premises=[("ancestor_of", "fwd")],
        conclusion=("descendant_of", "rev"),
        binding="chain",
    ),
    # 8. shared parent → sibling
    InferenceRule(
        "sibling_from_shared_parent",
        premises=[("parent_of", "fwd"), ("parent_of", "fwd")],
        conclusion=("sibling_of", "fwd"),
        binding="shared_parent",
    ),
    # 9. 对称关系: married_to → spouse_of
    InferenceRule(
        "spouse_from_married",
        premises=[("married_to", "fwd")],
        conclusion=("spouse_of", "fwd"),
        binding="chain",
    ),
]


class ReasoningChainExtractor:
    """
    从家族树中提取多步推理链。

    核心算法:
    1. 对每条推理规则，在图中搜索匹配的变量绑定
    2. 对链式规则 (chain)，找 A→B→C→... 路径
    3. 对共享父规则 (shared_parent)，找 A←P→B 结构
    4. 返回 (事实集, 查询, 答案, 推理链) 元组
    """

    def __init__(self, family_tree: FamilyTree):
        self.tree = family_tree
        # 构建关系索引: rel → [(head, tail), ...]
        self.rel_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # 构建邻接表: head → [(rel, tail), ...]
        self.adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # 构建反向邻接表: tail → [(rel, head), ...]
        self.rev_adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for h, r, t in family_tree.base_facts:
            self.rel_index[r].append((h, t))
            self.adj[h].append((r, t))
            self.rev_adj[t].append((r, h))

        # 预计算推导关系（1步推导：father/mother → parent）
        self._compute_derived_1hop()

    def _compute_derived_1hop(self):
        """计算 1 步推导关系并加入索引"""
        # father_of / mother_of → parent_of
        for r in ("father_of", "mother_of"):
            for h, t in self.rel_index[r]:
                self.rel_index["parent_of"].append((h, t))
                self.adj[h].append(("parent_of", t))
                self.rev_adj[t].append(("parent_of", h))

        # married_to → spouse_of
        for h, t in self.rel_index["married_to"]:
            self.rel_index["spouse_of"].append((h, t))
            self.adj[h].append(("spouse_of", t))
            self.rev_adj[t].append(("spouse_of", h))

    def extract_ancestor_chains(self, max_problems: int = 500,
                                 max_chain_length: int = 10) -> List[Dict]:
        """
        DFS 提取任意长度的祖先链。

        通过沿 parent_of 关系进行 DFS，找到所有长度为 3-max_chain_length 的
        ancestor 链（e.g. A→B→C→D→E = 4 步链）。

        这是 2 步以上推理的核心数据源。
        """
        problems = []
        parent_adj: Dict[str, List[str]] = defaultdict(list)
        for h, t in self.rel_index.get("parent_of", []):
            parent_adj[h].append(t)

        # 对每个实体做 DFS，找所有后代链
        all_entities = list(set(
            e for pairs in self.rel_index.get("parent_of", [])
            for e in pairs
        ))

        for root in all_entities:
            # DFS 找所有从 root 出发的 parent_of 链
            stack = [(root, [root])]  # (当前节点, 路径)
            while stack:
                node, path = stack.pop()
                if len(problems) >= max_problems:
                    break

                children = parent_adj.get(node, [])
                for child in children:
                    new_path = path + [child]
                    chain_len = len(new_path) - 1  # 边数 = 节点数-1

                    if 3 <= chain_len <= max_chain_length:
                        # 构建事实链: 每一步都是 parent_of
                        facts = [
                            (new_path[i], "parent_of", new_path[i+1])
                            for i in range(chain_len)
                        ]
                        query = (new_path[0], "ancestor_of", new_path[-1])
                        reasoning_chain = facts + [f"=> {query}"]

                        problems.append({
                            "facts": facts,
                            "query": query,
                            "answer": True,
                            "chain_length": chain_len,
                            "rule_name": f"ancestor_{chain_len}hop",
                            "reasoning_chain": reasoning_chain,
                        })

                    # 继续 DFS（只要路径没超过上限）
                    if chain_len < max_chain_length:
                        stack.append((child, new_path))

        random.shuffle(problems)
        return problems[:max_problems]

    def extract_chain_problems(self, rule: InferenceRule,
                                max_problems: int = 100) -> List[Dict]:
        """
        对指定规则提取推理问题。

        返回格式: [
            {
                "facts": [(h, r, t), ...],    # 已知事实（前提）
                "query": (h, r, t),           # 待推问题（结论）
                "answer": True,               # 正确答案
                "chain_length": int,          # 推理步数 = len(premises)
                "rule_name": str,             # 使用的规则名称
                "reasoning_chain": [...]      # 推理过程
            }
        ]
        """
        problems = []

        if rule.binding == "chain":
            problems = self._extract_chain_binding(rule, max_problems)
        elif rule.binding == "shared_parent":
            problems = self._extract_shared_parent_binding(rule, max_problems)

        return problems

    def _extract_chain_binding(self, rule: InferenceRule,
                                max_problems: int) -> List[Dict]:
        """提取链式绑定的推理问题: A→B→C→..."""
        problems = []
        num_steps = len(rule.premises)

        # 从第一个前提的所有实例出发，尝试延伸
        first_rel, first_dir = rule.premises[0]
        if first_rel not in self.rel_index:
            return problems

        candidates = list(self.rel_index[first_rel])
        random.shuffle(candidates)

        for start_pair in candidates:
            if len(problems) >= max_problems:
                break

            # 构建链: 找到满足所有前提的实体序列
            if first_dir == "fwd":
                chain_entities = [start_pair[0], start_pair[1]]
            else:
                chain_entities = [start_pair[1], start_pair[0]]

            chain_facts = [(start_pair[0], first_rel, start_pair[1])]
            valid = True

            for step_idx in range(1, num_steps):
                step_rel, step_dir = rule.premises[step_idx]
                last_entity = chain_entities[-1]

                # 在邻接表中找下一跳
                next_hops = [
                    (r, t) for r, t in self.adj[last_entity]
                    if r == step_rel
                ]

                if not next_hops:
                    valid = False
                    break

                chosen_rel, chosen_tail = random.choice(next_hops)
                if step_dir == "fwd":
                    chain_entities.append(chosen_tail)
                    chain_facts.append((last_entity, step_rel, chosen_tail))
                else:
                    chain_entities.append(chosen_tail)
                    chain_facts.append((last_entity, step_rel, chosen_tail))

            if not valid or len(chain_facts) != num_steps:
                continue

            # 构建结论
            conc_rel, conc_dir = rule.conclusion
            if conc_dir == "fwd":
                query = (chain_entities[0], conc_rel, chain_entities[-1])
            else:
                query = (chain_entities[-1], conc_rel, chain_entities[0])

            # 构建推理链描述
            reasoning_chain = list(chain_facts) + [
                f"=> {query}"
            ]

            problems.append({
                "facts": chain_facts,
                "query": query,
                "answer": True,
                "chain_length": num_steps,
                "rule_name": rule.name,
                "reasoning_chain": reasoning_chain,
            })

        return problems

    def _extract_shared_parent_binding(self, rule: InferenceRule,
                                        max_problems: int) -> List[Dict]:
        """提取共享父绑定的推理问题: A←P→B"""
        problems = []
        prem_rel, _ = rule.premises[0]

        # 找所有有两个以上 prem_rel 出边的节点（共享父母）
        parent_children: Dict[str, List[str]] = defaultdict(list)
        for h, t in self.rel_index.get(prem_rel, []):
            parent_children[h].append(t)

        for parent, children in parent_children.items():
            if len(children) < 2:
                continue
            if len(problems) >= max_problems:
                break

            # 从 children 中选取不同的两个
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    if len(problems) >= max_problems:
                        break
                    child_a, child_b = children[i], children[j]
                    conc_rel, _ = rule.conclusion
                    query = (child_a, conc_rel, child_b)
                    facts = [
                        (parent, prem_rel, child_a),
                        (parent, prem_rel, child_b),
                    ]
                    reasoning_chain = facts + [f"=> {query}"]
                    problems.append({
                        "facts": facts,
                        "query": query,
                        "answer": True,
                        "chain_length": 2,
                        "rule_name": rule.name,
                        "reasoning_chain": reasoning_chain,
                    })

        return problems


class LogicDataGenerator:
    """
    合成逻辑推理数据集的统一入口。

    生成流程:
    1. 创建家族树
    2. 对每条推理规则提取正样本
    3. 构造负样本（随机替换结论中的实体）
    4. 混合 & 打乱
    """

    def __init__(self, config: 'LogicConfig' = None):
        self.config = config or logic_config

    def generate_dataset(self, n_problems: int = None,
                          chain_lengths: List[int] = None) -> List[Dict]:
        """
        生成完整数据集。

        核心改进: 生成**多棵独立家族树**，每棵树的实体名带唯一前缀，
        确保即使单棵树产出有限，也能通过数量弥补。

        n_problems: 总问题数（正+负）
        chain_lengths: 限制推理链长度列表，如 [2, 3]

        返回: 问题列表，每题包含 facts / query / answer / chain_length / rule_name
        """
        n_problems = n_problems or self.config.num_train_problems

        if chain_lengths is None:
            chain_lengths = list(range(
                self.config.min_chain_length,
                self.config.max_chain_length + 1,
            ))

        all_positive = []
        all_entities_pool = []  # 所有树的实体汇总
        tree_idx = 0

        # 循环生成家族树，直到收集到足够多的正样本
        # 目标正样本数 = n_problems / (1 + negative_ratio)
        target_positive = int(n_problems / (1 + self.config.negative_ratio)) + 10

        while len(all_positive) < target_positive and tree_idx < 200:
            # 每棵树用唯一前缀避免实体名碰撞
            prefix = f"T{tree_idx}_"
            tree_idx += 1

            tree = FamilyTree(num_entities=self.config.num_entities)
            tree.generate()  # 使用 config 中的 num_generations 和 children_per_couple

            # 给所有实体加前缀
            prefixed_facts = set()
            for h, r, t in tree.base_facts:
                prefixed_facts.add((prefix + h, r, prefix + t))
            tree.base_facts = prefixed_facts
            tree.entities = [prefix + e for e in tree.entities]
            all_entities_pool.extend(tree.entities)

            # 提取推理链（包含规则单步 + DFS 长链）
            extractor = ReasoningChainExtractor(tree)
            for rule in INFERENCE_RULES:
                if len(rule.premises) not in chain_lengths:
                    continue
                per_rule_max = max(50, target_positive // len(INFERENCE_RULES) + 10)
                problems = extractor.extract_chain_problems(rule, max_problems=per_rule_max)
                all_positive.extend(problems)

            # DFS 提取长链 ancestor 推理（3-10 步）
            long_chain_lengths = [l for l in chain_lengths if l >= 3]
            if long_chain_lengths:
                max_cl = max(long_chain_lengths)
                ancestor_problems = extractor.extract_ancestor_chains(
                    max_problems=target_positive // 2,
                    max_chain_length=max_cl,
                )
                # 只保留目标链长范围内的
                ancestor_problems = [
                    p for p in ancestor_problems
                    if p["chain_length"] in chain_lengths
                ]
                all_positive.extend(ancestor_problems)

        # 构造负样本
        n_positive = len(all_positive)
        n_negative = int(n_positive * self.config.negative_ratio)
        negatives = self._generate_negatives(
            all_positive, all_entities_pool, n_negative
        )

        # 合并、截断、打乱
        dataset = all_positive + negatives
        random.shuffle(dataset)

        if len(dataset) > n_problems:
            dataset = dataset[:n_problems]

        return dataset

    def _generate_negatives(self, positives: List[Dict],
                             entities: List[str],
                             n_negative: int) -> List[Dict]:
        """
        生成负样本：保持事实不变，替换查询中的尾实体为随机实体。

        确保替换后的查询不在正样本中（避免假负样本）。
        """
        positive_queries = {
            (q["query"][0], q["query"][1], q["query"][2])
            for q in positives
        }

        negatives = []
        attempts = 0
        max_attempts = n_negative * 10

        while len(negatives) < n_negative and attempts < max_attempts:
            attempts += 1
            # 随机选一个正样本作为基础
            pos = random.choice(positives)
            h, r, t = pos["query"]

            # 替换尾实体
            new_t = random.choice(entities)
            if new_t == t or new_t == h:
                continue

            new_query = (h, r, new_t)
            if new_query in positive_queries:
                continue

            negatives.append({
                "facts": pos["facts"],  # 事实保持不变
                "query": new_query,
                "answer": False,
                "chain_length": pos["chain_length"],
                "rule_name": pos["rule_name"] + "_neg",
                "reasoning_chain": pos["facts"] + [f"≠> {new_query}"],
            })

        return negatives

    def get_all_entities(self, dataset: List[Dict]) -> List[str]:
        """从数据集中收集所有出现的实体"""
        entities = set()
        for problem in dataset:
            for h, r, t in problem["facts"]:
                entities.add(h)
                entities.add(t)
            h, r, t = problem["query"]
            entities.add(h)
            entities.add(t)
        return sorted(entities)

    def get_all_relations(self, dataset: List[Dict]) -> List[str]:
        """从数据集中收集所有出现的关系"""
        relations = set()
        for problem in dataset:
            for _, r, _ in problem["facts"]:
                relations.add(r)
            _, r, _ = problem["query"]
            relations.add(r)
        return sorted(relations)

    def print_statistics(self, dataset: List[Dict]) -> None:
        """打印数据集统计信息"""
        from collections import Counter
        total = len(dataset)
        positive = sum(1 for d in dataset if d["answer"])
        negative = total - positive
        chain_dist = Counter(d["chain_length"] for d in dataset)
        rule_dist = Counter(d["rule_name"] for d in dataset)

        print(f"\n{'='*50}")
        print(f"  数据集统计")
        print(f"{'='*50}")
        print(f"  总数: {total} (正: {positive}, 负: {negative})")
        print(f"  推理链长度分布:")
        for length, count in sorted(chain_dist.items()):
            print(f"    {length} 步: {count} 题")
        print(f"  规则分布:")
        for rule, count in sorted(rule_dist.items(), key=lambda x: -x[1])[:10]:
            print(f"    {rule}: {count}")
        print(f"  实体数: {len(self.get_all_entities(dataset))}")
        print(f"  关系数: {len(self.get_all_relations(dataset))}")


if __name__ == "__main__":
    # 独立测试: 生成数据集并打印统计
    gen = LogicDataGenerator()
    dataset = gen.generate_dataset(n_problems=500, chain_lengths=[1, 2])
    gen.print_statistics(dataset)

    # 打印前 5 个样本
    print(f"\n--- 样例 ---")
    for i, sample in enumerate(dataset[:5]):
        print(f"\n[{i+1}] 规则: {sample['rule_name']}, "
              f"链长: {sample['chain_length']}, "
              f"答案: {sample['answer']}")
        print(f"  事实: {sample['facts']}")
        print(f"  查询: {sample['query']}")
        print(f"  推理: {sample['reasoning_chain']}")
