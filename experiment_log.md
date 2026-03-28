# 实验日志 (Experiment Log)

> 记录每次架构演进的关键变更、实验结果和思考。按版本倒序排列。

---

## v5.2_Final — 消除架构原罪的彻底重构 (2026-03-26)

### 背景

基于极其苛刻的架构设计原则，开展了覆盖所有层面代码的 “17 项缺陷清洗”。本次升级不引入新模型，只引入更高阶的结构与数学抽象。

### 核心决议与落地

**彻底的器官解耦 (职责归位)**
- 剥夺了 `brain.py` 的上帝视角，将其拥有的 `merge_similar`、`extract_patterns`、`evolve_intents` 全部下放给海马体与 JEPA。让 `brain` 退化为纯粹的事件编排器。

**物理级拓扑守恒 (双向边迁移)**
- 记忆在图谱中向上坍缩为 “抽象概念” 节点时，彻底打通了历史遗留的图断裂漏洞。底层全面换装 `MultiDiGraph` 容纳并发叠加态软边，并在节点合并时，强制开启了对由于时间序列形成的入度（predecessors）的双向安全迁移，维持了多拓扑时空连贯。

**零损算力 (全局 Embedding 缓存化)**
- 语言模型的 forward() 极为昂贵，在深度做梦这种密集循环重推演期间我们全面启用了 `emb_cache` 池回调架构。对同样的数据文本拒绝任何复用请求，算力百分之百向 KGE 和 KMeans 方向剥离。

**纯数学意图驱动 (无偏语义 KMeans)**
- 摒弃了原有的通过检索途径粗暴分类记忆意图的做法，基于 Pytorch 重构了基于 896 维隐空间的真实 `K-Means` 自动漂移聚类算法，动态压缩意图簇群，系统达成了真正的“语义凝集”，而非行为设定。

**10K+图谱解禁 (O(N²) 降级为稀疏 O(E))**
- 将导致内存几何爆炸的 `np.zeros(N,N)` 邻接转移矩阵彻底废弃，更换为专攻大尺度图谱搜索的 `scipy.sparse.csc_matrix`，一举赋予系统存储十万级关联记忆节点的物理支持极限。

**多维 Batch 并发压榨 (GPU 利用率极效提升)**
- 将在 Python 层面逐条串行计算特征 Loss 的愚蠢循环彻底消灭。强行对齐管线，利用 `repeat_interleave` 和极速拉平 `view` 重写了整个多源 Intent 成本核算，一趟全量 Batch 并行通过，执行时延断崖下降。

**全能物尽其用 (TransE 破局融合)**
- 先前为了训练而训练，产出无用的 128 维空间悬空表征的问题彻底被掐断。经过严谨论证，抛弃了在临时推断上进行生拉硬拽的 L2 匹配，而是把其活跃度降维归范，完美作为权重辅信号注入 `find_seeds()` 打分系统。

### 思考

只有把一切为了敷衍运行而拼凑的妥协擦除（硬编码字典、强制类型判断、单向图覆盖、多重循环低效推理），系统的 “认知水管” 才会真正畅通。所有的规则应源自拓扑和数学本质，而非人工 if-else 的堆砌。v5.2 这个终局级重修彻底做到了这一点。

---

## v5.1 — 三条架构红线修复 + Attention Pooling (2026-03-26)

### 背景

外部审核（Gemini）发现四个架构级问题：

1. JEPA 用 LLM 自身输出做训练目标 → 自引用/幻觉晶体化 → **致命**
2. 微梦在白天跑 LLM forward + CE 反传 → 违反"光速推理"哲学 → **高**
3. goal_emb = 离输入最近的记忆（问题节点） → rollout 形同虚设 → **高**
4. Mean Pooling 导致 Grey Mush（语义失真） → **高**

### 变更

**红线 1: 延迟训练目标**
- 新增 `_pending_experience`，存储上一轮用户输入的 embedding
- `interact()` 中：当新一轮用户输入到来时，用它作为上一轮 JEPA 的真实训练目标
- 回放池中存储的是 `(prev_user_input, next_user_input)` 外部真实对，不再有 LLM 自身输出

**红线 2: 微梦轻量化**
- `_micro_dream()` v2：去除 LLM CE Loss，只做纯 JEPA Pred Loss
- 白天不走 LLM forward，injection_layer 和 Gamma 只在做梦阶段更新
- 做梦阶段保留 CE（用的是人类提供的真实对话对，无自引用问题）

**红线 3: goal_emb 重定义**
- 新增 `HippocampalMemory.find_outcome_nodes()`：从种子节点出发 BFS 搜索下游"结局"节点
- 结局类型：`应对策略`、`效果验证`、`行动经验`、`模式实例`、抽象概念节点、模式节点
- `interact()` 中：goal_emb = 结局节点的 embedding（不再是最近记忆）
- 无结局节点时 fallback 到 `goal_emb=None`（JEPA 自由探索）

**P0: Attention Pooling**
- `get_real_embedding()` 从 `mean(dim=1)` 改为 L2-norm weighted attention pooling
- 信息量高的 token（内容词/实体）自然获得更高权重，padding/停用词被压制

### 实验结果

| 指标 | v5.0（修复前） | v5.1（修复后） |
|---|---|---|
| CE Loss | 5.47 → 2.69 | 5.47 → 2.22 |
| JEPA Pred Loss | 0.16 → 0.04 | 0.17 → 0.07 |
| 微梦训练目标 | LLM 自身输出 ❌ | 用户真实下一轮输入 ✅ |
| 微梦延迟 | 含 LLM forward ❌ | 纯 JEPA (零 LLM 开销) ✅ |
| goal_emb | 最近记忆（问题节点）❌ | 结局节点 1 跳 ✅ |
| 延迟微梦 pred | — | 0.059 |
| 回放池内容 | (input, LLM_output) | (input, next_real_input) ✅ |

### 关键日志验证

```
# 第1轮: 无延迟微梦（首次无 pending）
# 第2轮: 延迟微梦 (prev→current pred: 0.0596, 回放池: 1/64) ← 用真实输入训练
# 第3轮: 延迟微梦 (prev→current pred: 0.0591, 回放池: 2/64) ← 持续收敛

# goal_emb 修复:
第2轮: 结局目标: [抽象] 今天和老板开会... (1跳)  ← 下游抽象概念
第3轮: 结局目标: 小目标逐个攻克焦虑下降了 (1跳) ← 真正的"效果验证"结局
```

---

## v5.0 — 方案C 混合意图 + 双模学习 (2026-03-26)

### 变更
- **方案C 混合意图**：`intent_bank`（6个可学习）+ `memory_intent_proj`（动态从记忆提取）
  - `extract_memory_intents()`: 按边类型分组提取代表性记忆 embedding → 投影到意图空间
  - `get_all_intents()`: 融合所有意图，总路径数 S+M 随记忆积累动态增长
  - `rollout_with_intent_vec()`: 泛化 rollout，接受任意意图向量（不限于 intent_bank 索引）
- **实时微梦** (`_micro_dream`): 每次交互后 3 步联合训练（JEPA Pred + Hook CE）
- **深度做梦 v5** 新增三个高级操作：
  - `_merge_similar_memories()`: cosine > 0.92 的节点对合并为抽象概念节点（涌现基础）
  - `_extract_recurring_patterns()`: 两跳路径频率统计 → 模式节点写回图谱
  - `_evolve_intents_from_experience()`: 回放池目标嵌入 EMA 更新 intent_bank（意图进化）
- **`mvp_config.py`** 新增：`jepa_max_memory_intents`、`micro_dream_steps/ce_weight`、`deep_dream_merge_threshold/pattern_min_count`

### 思考

**关于意图边数量是否自适应**：
- 边类型由 `ingest()` 调用时传入的字符串决定，完全自由
- 系统额外自动建两种软边："时间临近"（3600秒窗口）、"语义相似"（cosine > 0.75）
- `memory_intents` 数量 = 见到的边类型数量（上限 `jepa_max_memory_intents=4`）
- **结论：不写死，完全由写入内容驱动**

**关于二阶抽象涌现（重要研究方向）**：
- v5 已实现一阶抽象：多个具体经历 → 合并为抽象概念节点
- 二阶抽象：在下次做梦时，对一阶抽象节点再次执行合并 → "焦虑管理策略" 这类高阶概念
- 理论上只需积累足够多的一阶抽象节点后多次调用 `train_dream_phase`，不需要额外代码改动
- **这是系统产生真正泛化能力的关键路径**

---

## v4 — 真 JEPA 世界模型集成 (2026-03-26)

### 变更
- 将 `SubconsciousJEPA` 从 3 层 MLP 重构为真正的 JEPA 世界模型引擎
- 新增 `_AdaLNBlock`：AdaLN-zero 条件化 Transformer 块
- 新增 `CognitivePredictor`：3 层 Transformer Predictor
- 新增 `intent_bank`：6 个可学习策略意图（替代噪声微扰方案）
- 新增 `rollout()`：自回归多步预演（rollout_depth=3）
- 新增 `criterion()`：与海马体目标记忆的 MSE 成本评估
- 新增 `generate_candidates()`：cost-based 决策流程
- 新增 `compute_prediction_loss()`：latent prediction + SIGReg + 意图多样性联合损失
- 更新 `train_dream_phase`：使用 JEPA latent prediction loss 替代旧 MSE loss
- 更新 `interact`：rollout → criterion → best path → steer 决策链

### 实验结果

| 指标 | 结果 |
|---|---|
| JEPA Pred Loss | 0.15 → 0.04 (8 epochs) ✅ |
| Cost 分布 | best=0.057, worst=2.13 (37x差距) ✅ |
| Gamma | 0.004 → 0.010 ✅ |
| SQLite 持久化 | 11节点, 62突触 ✅ |

---

## v3 — SQLite 海马体 + 多候选评分 (早期版本)

### 变更
- 引入 SQLite 持久化
- 用噪声微扰生成多个候选（已在 v4 废弃）
- PPR 联想检索
- TransE KGE 记忆巩固

---

## 审计问题追踪表

| # | 问题 | 发现时间 | 修复版本 | 状态 |
|---|---|---|---|---|
| 1 | SubconsciousJEPA 是假 JEPA（3层MLP） | v3 审计 | v4 | ✅ 已修复 |
| 2 | Grey Mush（Mean Pooling 语义失真） | v3 审计 | v5.1 | ✅ 已修复 |
| 3 | 自引用训练（LLM 输出做 JEPA 目标） | v3 审计 + Gemini 审核 | v5.1 | ✅ 已修复 |
| 4 | 噪声微扰多候选（非真 JEPA 意图空间） | v3 审计 | v4 | ✅ 已修复 |
| 5 | 微梦白天跑 LLM forward（延迟高） | Gemini 审核 | v5.1 | ✅ 已修复 |
| 6 | goal_emb 绕路（最近记忆做目标） | Gemini 审核 | v5.1 | ✅ 已修复 |
| 7 | find_seeds O(N) 遍历（可扩展性差） | Gemini 审核 | 待定 | ⏳ 低优先级 |
| 8 | 上帝文件（brain_mvp.py 1490行） | v5.1 审计 | v5.2 | ✅ 模块化拆分 |
| 9 | 14个死代码文件 | v5.1 审计 | v5.2 | ✅ 归档到 archive/ |
| 10 | SQLite↔NetworkX 状态不同步 | v5.1 审计 | v5.2 | ✅ 原子操作同步 |
| 11 | 注入层 Attention 退化（softmax→1.0） | v5.1 审计 | v5.2 | ✅ Gated Residual Modulation |
| 12 | find_outcome_nodes 硬编码边类型 | v5.1 审计 | v5.2 | ✅ 拓扑特征判断 |
| 13 | rollout 串行循环 | v5.1 审计 | v5.2 | ✅ repeat_interleave 并行化 |
| 14 | min-loss winner-takes-all | v5.1 审计 | v5.2 | ✅ soft-min 加权 |
| 15 | 意图进化伪K-means（无排他性） | v5.2 审查 | v5.2 | ✅ used_indices 排他分配 |
| 16 | O(N²) 合并 Python 循环 | v5.2 审查 | v5.2 | ✅ numpy 矩阵化 |
| 17 | rollout 并行化维度 bug（B>1错配） | v5.2 审查 | v5.2 | ✅ 统一排列约定 |
