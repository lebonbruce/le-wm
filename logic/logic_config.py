"""
logic/logic_config.py —— 逻辑推演模块配置

独立于主 mvp_config.py，管理逻辑推理特有的参数。
核心维度（jepa_core_dim, kge_embed_dim 等）从 mvp_config 继承。
"""
from dataclasses import dataclass, field
from typing import Tuple
from mvp_config import config as brain_config


@dataclass
class LogicConfig:
    """逻辑推演模块的独立配置中心"""

    # =============================================================
    # 1. 递归推理循环参数
    # =============================================================
    max_reasoning_depth: int = 50        # 最大递归推理步数（安全硬上限，防无限循环）
    initial_reasoning_depth: int = 10    # 训练时默认推理深度
    working_memory_size: int = 64        # 工作记忆槽位数（扩大以支持长链）
    halt_threshold: float = 0.95         # ACT 停止阈值——累积 halt_prob 超过此值则终止
    confidence_threshold: float = 0.9    # 推理时动态停止: answer 置信度超过此值提前终止

    # =============================================================
    # 2. 合成数据生成
    # =============================================================
    num_entities: int = 80               # 实体池大小（增大以支持深家族树）
    num_train_problems: int = 5000       # 训练题数（增大以覆盖多种链长）
    num_test_problems: int = 500         # 测试题数
    min_chain_length: int = 1            # 最短推理链
    max_chain_length: int = 10           # 最长推理链（Phase 2）
    negative_ratio: float = 1.0          # 负样本比例 1:1（解决负样本准确率低的问题）
    num_generations: int = 10            # 家族树代数（支持 10 步 ancestor 链）
    children_per_couple: int = 3         # 每对夫妻最大孩子数

    # 关系类型定义（显式、符号化、可组合）
    # 基础关系：直接在图上建边
    base_relations: Tuple[str, ...] = (
        "father_of",      # A 是 B 的父亲
        "mother_of",       # A 是 B 的母亲
        "married_to",      # A 和 B 结婚（对称）
    )
    # 推导关系：由基础关系通过规则推出
    derived_relations: Tuple[str, ...] = (
        "parent_of",       # father_of ∨ mother_of → parent_of
        "child_of",        # parent_of 的逆
        "grandparent_of",  # parent_of ∘ parent_of
        "grandchild_of",   # grandparent_of 的逆
        "ancestor_of",     # parent_of 的传递闭包
        "descendant_of",   # ancestor_of 的逆
        "sibling_of",      # 共享 parent_of（对称）
        "uncle_of",        # parent_of 的 sibling
        "spouse_of",       # married_to 的别名（对称）
    )

    # =============================================================
    # 3. 推理规则（显式、符号化）
    #    格式: (前提关系列表, 结论关系, 变量绑定模式)
    #    变量绑定: "chain" = A→B→C 链式, "shared" = A←B→C 共父
    # =============================================================
    # 在 __post_init__ 中构建（dataclass 不能直接存复杂对象）

    # =============================================================
    # 4. 神经网络维度（继承 + 扩展）
    # =============================================================
    # 三元组编码器将 (h, r, t) 三个 kge_embed_dim 向量映射到 jepa_core_dim
    triple_encoder_hidden: int = 512     # 三元组编码器中间层维度
    conclusion_decoder_hidden: int = 512 # 结论解码器中间层维度

    # =============================================================
    # 5. 训练参数
    # =============================================================
    reasoning_lr: float = 1e-3           # 推理训练初始学习率
    reasoning_epochs: int = 100          # 训练轮数
    batch_size: int = 64                 # 训练 batch 大小（增大利用 batch 并行）
    answer_loss_weight: float = 1.0      # 最终答案 BCE 损失权重
    chain_loss_weight: float = 0.5       # 中间推理步 MSE 损失权重
    halting_loss_weight: float = 0.1     # ACT 停止正则化权重（提高，鼓励学习何时停）
    sigreg_weight: float = 0.1           # SIGReg 防坍塌权重
    eval_interval: int = 10              # 每 N epoch 评测一次
    use_cosine_lr: bool = True           # 使用 cosine annealing 学习率调度

    # =============================================================
    # 6. 从 brain_config 继承的关键维度（只读引用）
    # =============================================================
    @property
    def jepa_core_dim(self) -> int:
        return brain_config.jepa_core_dim  # 1536

    @property
    def kge_embed_dim(self) -> int:
        return brain_config.kge_embed_dim  # 128

    @property
    def device(self) -> str:
        return brain_config.device


logic_config = LogicConfig()
