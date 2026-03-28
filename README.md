# 用结构换算力，用记忆换智商

> 一个人、一块 RTX 4060 8G、一个脑洞——能做出什么？

这是我的个人研究实验项目。核心问题是：**LLM 的智能上限真的是参数量决定的吗？**

我不信。所以我试着拼凑出一个"三位一体"的认知系统，用结构代替算力，用记忆代替智商。

## 🧠 是个啥？

本质上是一个把三种东西缝合在一起的实验：

- **JEPA 世界模型**（潜意识引擎）：在脑子里预演未来，而不是硬算
- **海马体记忆系统**：SQLite + 知识图谱 + FAISS + TransE，把经历变成可检索的智慧
- **冻结的 LLM**（Qwen2.5-0.5B）：只当"嘴替"，把 JEPA 想好的方向翻译成人话

```
输入文字
  ↓
冻结 LLM → 896维语义特征
  ↓
JEPA 潜意识引擎（在脑子里滚 3 步，试 50 条路径，找最优意图）
  ↓
MetaLanguage 解码器（把轨迹翻译成 "[方向] xxx | [置信度] 78%"）
  ↓
冻结 LLM 读懂元语 → 生成回答
```

JEPA 不预测 token，它预测**潜空间里的抽象状态**。训练目标不是让它说出正确答案，而是让它的预演轨迹收敛到海马体里真实发生过的后继事件。

## ✨ 折腾了哪些东西

**v20 三个关键修复**：

1. **打破衔尾蛇**：以前 JEPA 用自己的 EMA 副本做训练目标（自己评估自己），现在改成海马体里的真实后继事件
2. **消除注意力中毒**：以前把 JEPA 轨迹当连续向量注入 LLM（LLM 完全不认识那是啥），现在翻译成结构化文本
3. **InfoNCE 锐化信号**：不只是让预测靠近正确，还要远离错误

**用到的技术**（都是站在巨人肩膀上）：

- **SIGReg**（来自 [LeWorldModel](https://arxiv.org/abs/2603.19312)）：随机投影到 1D 后做高斯分布拟合，防止潜空间坍塌
- **随机世界模型**（借鉴 [DreamerV3](https://arxiv.org/abs/2301.04104)）：Posterior/Prior + 重参数化，让模型能表达"同一问题可能有多种合理回复"
- **InfoNCE 对比学习**（来自 [CPC](https://arxiv.org/abs/1807.03748)）：不只靠近正确目标，同时远离错误目标
- **可学习意图库 + K-Means 记忆聚类**：这个是我自己瞎想的，让策略方向随经历动态扩展，不只依赖固定参数
- **Latent Overshooting**：每步推演加步数编码，借鉴自 [Dreamer](https://arxiv.org/abs/1912.01603) 的思路

## 📊 测试结果（跑在 Docker 里，RTX 4060 8G）

```
v20 审计测试套件 (P0: 梯度流 + 世界模型 | P1: MetaLanguage)

[T1] encode_fact_target 梯度阻断验证  ✅ encoder 49/49 个参数有非零梯度
[T2] InfoNCE 梯度传播验证            ✅ predictor + encoder 全部有梯度
[T4] memory_intent_proj 梯度连通验证  ✅
[T5] Prediction loss 收敛性测试      ✅ 改善幅度: 77.9% (2.02 → 0.44)
[T6] InfoNCE 有效性测试              ✅ (⚠️ 收敛需调参，非代码 bug)
[T8] 意图区分度测试                  ✅ 6个意图平均余弦距离 0.97（近似正交）
[T9] Rollout vs 随机基线             ✅ rollout MSE 0.44 vs 随机基线 1.90（4.3x↓）
[T10] 元语格式完整性                 ✅
[T12] 置信度计算正确性               ✅ 完全对齐100% / 正交50% / 反向0%
[T13] 意图标签边界                   ✅

结果: 10 通过 / 1 失败（faiss 宿主机未安装，Docker 内正常）/ 11 总计
```

值得注意的是 **T9**：JEPA rollout 找到的最优路径比随机猜测好 4 倍多。这至少说明世界模型在潜空间里确实学到了点什么，不是完全在随机游走。

## 🔧 项目结构

```
le-wm/
├── brain.py                   # 主入口：编排三大系统
├── mvp_config.py              # 统一配置中心
├── cortex/                    # 语言皮层（冻结 LLM 封装）
├── hippocampus/               # 海马体（SQLite+图谱+FAISS+TransE）
├── jepa_engine/               # JEPA 潜意识引擎
│   ├── subconscious.py        # 世界模型核心
│   ├── predictor.py           # AdaLN-zero Transformer
│   ├── encoder.py             # 4层 Transformer Encoder
│   ├── meta_decoder.py        # 轨迹→结构化文本
│   └── sigreg.py              # 防坍塌正则化
└── scenarios.json             # 测试场景（可自定义）
```

## 🚀 跑起来

项目跑在 Docker 里（用了些只在 Docker 里能用的技术），需要的依赖都在容器里：

```bash
# 清数据库重新开始
rm hippocampus.db

# 三阶段：记忆建库 → 深度做梦 → 认知交互
python brain.py

# 跑测试
python test_audit_v20.py
```

## 😅 说说限制和现实

先说好，这不是什么工业级系统，就是一个**兴趣驱动的个人实验**。

**硬件实况**：只有一块 RTX 4060 8G。当前可训练参数 ~63M（对比 Meta V-JEPA 2 的 1.2B），测试数据就 4 条对话对，50 epochs。

也就是说，很多设计是**在概念上验证可行性**，但没有经过大规模训练和 benchmark 洗礼。T5 那个"77.9% 改善"是在少量数据上的，不代表通用性。

**与前沿的差距**：
- 规模差 ~20 倍（63M vs 1.2B+）
- 没有视觉、听觉输入，纯文本
- 没在标准 benchmark（ARC-AGI / SSv2 等）上系统验证

**相比 Meta 官方 JEPA 路线有一个比较不同的地方**：
- Meta 的 JEPA 系列训练完就固定了；这个实验加了持久记忆（海马体），每次对话都会写入并影响下次推演

## 🔭 如果有更多资源，未来想做的

（以下都是想象，我那块 4060 8G 做不了）

- [ ] 把 Encoder 从全局 pooling 改成序列级编码（保留 token 粒度信息）
- [ ] 去掉 EMA，全面 SIGReg（对齐 LeWorldModel，省 14M 参数）
- [ ] 加一个 Cross-Attention 解码层替代 MetaLanguage 手写规则
- [ ] 接图像输入（ViT + 海马体存视觉记忆）
- [ ] 在 ARC-AGI 或 CLUTRR 上做系统 benchmark
- [ ] 规模 x10：core_dim 4096，encoder 8层，predictor 6层 → 约 1.2B 参数
- [ ] 改用 Mini 版本的 Llama/Mistral 替代 Qwen2.5-0.5B，看看接口兼容性

如果有大佬愿意在更大的卡上跑跑，非常欢迎开 issue 聊聊！

## 📖 设计理念

| 原则 | 实现 |
|------|------|
| 冻结语法，训练策略 | LLM 参数锁定，JEPA + 记忆系统持续进化 |
| 从外部真实反馈学习 | JEPA 训练目标 = 海马体真实后继事件 |
| LLM 读人话不读噪声 | 元语文本注入替代 embedding 注入 |
| 意图自适应 | 可学习意图 + 记忆驱动意图随经历增长 |

核心参考：
- [I-JEPA](https://arxiv.org/abs/2301.08243) / [V-JEPA 2](https://arxiv.org/abs/2506.09985) (Meta AI)
- [LeWorldModel](https://arxiv.org/abs/2603.19312) (2026.03) — SIGReg 思路
- [VL-JEPA](https://arxiv.org/abs/2412.04139) (2025.12) — selective decoding

## License

MIT — 随便用，随便改，但别说这是工业级的就行 😂

---

*一个人一块 4060 8G 能做到这里，已经很满足了。如果这个实验对你有任何启发，那就值了。*
