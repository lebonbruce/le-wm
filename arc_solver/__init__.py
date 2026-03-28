"""
arc_solver — V20 搜索+验证 ARC 求解管线

核心理念：ARC 不是序列预测问题，而是程序合成+确定性验证问题。
- LLM 生成候选变换程序（Python DSL）
- 沙盒执行程序 → 产生候选输出
- 外部验证器精确匹配 → 打破 JEPA 自蒸馏死循环
- Value Network 在多个候选中选最优（Phase 2）

为什么放弃 JEPA Soft Prompt 注入：
  1. JEPA 连续潜空间 vs ARC 离散刚性 = 不可调和
  2. 自蒸馏 Qwen 隐状态 = 垃圾进垃圾出死循环
  3. Soft Prompt 中毒 Qwen 注意力流形
  4. MSE/Cosine loss 对离散任务极度宽容
"""
