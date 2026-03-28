"""
brain.py —— 三位一体认知架构 v20 编排层

v20 三断裂点修复：
- 断裂点 #1: 训练 target 从 EMA 自蒸馏切换为海马体外部事实 (encode_fact_target)
- 断裂点 #2: JEPA→LLM 接口从 Soft Prompt embedding 注入切换为 MetaLanguage 文本注入
- 断裂点 #3: Loss 增加 InfoNCE 对比损失锐化训练信号

核心哲学不变：
  JEPA 在潜空间预演现实，海马体抲经历结晶成智慧，
  LLM 作为嘴替将 JEPA 的方向性元语翻译为人话。
"""
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from mvp_config import config
from cortex import LinguisticCortex
from hippocampus import HippocampalMemory
from jepa_engine import SubconsciousJEPA, TrajectoryInjection
from jepa_engine.meta_decoder import MetaLanguageDecoder


class TheBrainMVP(nn.Module):
    def __init__(self):
        super().__init__()

        # cortex 不注册为 nn.Module 子模块
        # 使用 object.__setattr__ 绕过 nn.Module 的自动注册机制
        # 这样 cortex.model 的 5亿冻结参数不会出现在 state_dict 中
        object.__setattr__(self, 'cortex', LinguisticCortex())

        # v5.6 P2-4: hippocampus 不再继承 nn.Module，作为普通属性存储
        # kge (TransE) 单独注册为 nn.Module 子模块，确保参数被训练体系发现
        object.__setattr__(self, 'hippocampus', HippocampalMemory())
        self.kge = self.hippocampus.kge  # nn.Module 子模块注册

        self.jepa = SubconsciousJEPA().to(config.device)
        self.injection_layer = TrajectoryInjection().to(config.device)

        # v20: 元语解码器（断裂点 #2 修复：替代 Soft Prompt embedding 级注入）
        # 无可训练参数，纯规则映射
        object.__setattr__(self, 'meta_decoder', MetaLanguageDecoder())

        # v5.6 P0-4: 参数组分离——微梦只更新 group 0 (JEPA)，
        # 做梦更新所有组。避免微梦的 step() 污染 hippocampus 的 Adam 动量状态。
        # v20-audit F3: injection_layer 不再参与 v20 训练路径，从 optimizer 中移除
        # 原来 ~9M 参数在每个训练步都接收梯度更新但完全无用，浪费算力且引入噪声
        self.optimizer = torch.optim.AdamW([
            {'params': list(self.jepa.parameters()),
             'lr': config.online_lr},       # group 0: JEPA
            {'params': list(self.kge.parameters()),
             'lr': config.online_lr},       # group 1: hippocampus KGE
        ])

        # 延迟微梦：缓存上一轮用户的输入，用下一轮真实反馈做训练
        self._pending_experience = None

        # LRU 嵌入缓存（OrderedDict，超过上限时移除最早的条目）
        self._emb_cache = OrderedDict()

    # ==== v7.0: Soft Prompt 注入辅助方法 ====

    def _build_virtual_embeds(self, trajectory: torch.Tensor,
                              mem_emb: torch.Tensor):
        """
        v7.0: 将 JEPA 推理轨迹 + 海马体记忆投影为门控后的虚拟 token embeddings。

        trajectory: (B, T_traj, D_jepa) — JEPA rollout 完整轨迹
        mem_emb: (B, D_llm) — 海马体检索的记忆 embedding

        返回: (gated_embeds, gate_scores)
            gated_embeds: (B, T_traj+1, D_llm) 门控后的虚拟 token embedding
            gate_scores: (B, T_traj+1) 每个虚拟 token 的门控分数
        """
        return self.injection_layer.compute_virtual_embeds(trajectory, mem_emb)

    def _prepend_virtual_tokens_for_ce(self, virtual_embeds: torch.Tensor,
                                       full_text: str, user_token_len: int):
        """
        v7.0: 将虚拟 token 拼在文本 embedding 前，构造 CE loss 所需的输入。

        核心设计：
        - 虚拟 token 放在序列头部 → 因果掩码的下三角特性天然允许后续 token attend
        - label 中虚拟 token + 用户输入部分设为 -100（ignore_index）
        - 仅 AI 回复部分参与 CE 梯度

        virtual_embeds: (1, V, D_llm) 门控后的虚拟 token embedding
        full_text: 完整文本 (user_input + "\n" + target_output)
        user_token_len: 用户输入部分的 token 数量

        返回: (logits, labels)
        """
        # Tokenize 完整文本
        inputs = self.cortex.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=512
        ).to(config.device)

        input_ids = inputs["input_ids"]  # (1, T)
        T = input_ids.shape[1]
        V = virtual_embeds.shape[1]

        # 获取文本部分的 token embedding
        # 需要通过 LLM 的 embedding 层转换 input_ids → 连续嵌入
        text_embeds = self.cortex.model.model.embed_tokens(input_ids)  # (1, T, D_llm)

        # 拼接：虚拟 token 在前，文本 token 在后
        combined_embeds = torch.cat([
            virtual_embeds.to(text_embeds.dtype),
            text_embeds
        ], dim=1)  # (1, V+T, D_llm)

        # 构造 attention mask（全 1，虚拟 token 和文本 token 都可见）
        attention_mask = torch.ones(
            1, V + T, dtype=torch.long, device=config.device)

        # 前向传播（使用 inputs_embeds 而非 input_ids）
        outputs = self.cortex.model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (1, V+T, vocab_size)

        # 构造 labels：
        # - 虚拟 token 位置: -100 (不计算 loss)
        # - 用户输入 token 位置: -100 (不计算 loss)
        # - AI 回复 token 位置: 真实 token id
        labels = torch.full(
            (1, V + T), -100, dtype=torch.long, device=config.device)
        # AI 回复部分的 label = 对应的 input_ids（从 V + user_token_len 开始）
        labels[:, V + user_token_len:] = input_ids[:, user_token_len:]

        return logits, labels

    def _generate_with_virtual_tokens(self, virtual_embeds: torch.Tensor,
                                      user_input: str):
        """
        v7.0: 将虚拟 token 拼在用户输入 embedding 前，然后生成回答。

        virtual_embeds: (1, V, D_llm) 门控后的虚拟 token embedding
        user_input: 用户输入文本

        返回: 生成的文本 (str)
        """
        # Tokenize 用户输入
        input_ids = self.cortex.tokenizer(
            user_input, return_tensors="pt"
        ).to(config.device)["input_ids"]  # (1, T)

        T = input_ids.shape[1]
        V = virtual_embeds.shape[1]

        # 获取文本 embedding
        text_embeds = self.cortex.model.model.embed_tokens(input_ids)  # (1, T, D_llm)

        # 拼接虚拟 token
        combined_embeds = torch.cat([
            virtual_embeds.to(text_embeds.dtype),
            text_embeds
        ], dim=1)  # (1, V+T, D_llm)

        attention_mask = torch.ones(
            1, V + T, dtype=torch.long, device=config.device)

        # 生成（使用 inputs_embeds）
        output = self.cortex.model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=config.generation_max_tokens,
            pad_token_id=self.cortex.tokenizer.eos_token_id,
            do_sample=False,
        )

        # output 包含完整序列（虚拟 token 位置 + 原文 + 生成），
        # 需要跳过前 V+T 个位置（虚拟 token + 原文）提取新生成的内容。
        # 注意：model.generate 的 inputs_embeds 模式下，
        # 返回的 output 只包含生成的新 token（不含 inputs_embeds 对应的位置）。
        generated_text = self.cortex.tokenizer.decode(
            output[0], skip_special_tokens=True)

        return generated_text

    # ==== 学习率动态切换 ====

    def _set_lr(self, lr: float):
        """统一修改优化器学习率"""
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    # ==== 阶段 1: 记忆建库 ====

    def ingest(self, text: str, linked_from: str = None, edge_type: str = None) -> str:
        """将经验写入海马体"""
        emb = self.cortex.get_real_embedding(text)
        nid = self.hippocampus.memorize(text, emb, linked_from, edge_type)
        print(f"  >> 记忆写入: {text[:50]}... → 节点 {nid}")
        return nid

    # ==== P1-1: LRU 嵌入缓存 ====

    def _get_cached_emb(self, text: str) -> torch.Tensor:
        """
        LRU 嵌入缓存：相同文本只走一次 LLM forward，跨调用复用。
        超过上限时自动淘汰最久未使用的条目（FIFO）。
        """
        if text in self._emb_cache:
            # 移到末尾（标记为最近使用）
            self._emb_cache.move_to_end(text)
            return self._emb_cache[text]

        with torch.no_grad():
            emb = self.cortex.get_real_embedding(text).to(torch.float32)

        self._emb_cache[text] = emb
        # 超过容量上限，移除最早的条目
        while len(self._emb_cache) > config.emb_cache_max_size:
            self._emb_cache.popitem(last=False)

        return emb


    # ==== v8.0 B路线：自监督 Encoder 预训练 ====

    def pretrain_encoder_phase(self, dialogue_pairs: list, epochs: int = 30):
        """
        v8.0 B路线核心：自监督 Context Masking 预训练。

        在深度做梦之前调用，用无标签的对话文本训练 Encoder 学习高质量表示。
        不涉及 injection/CE/海马体，纯粹训练 Encoder + SSL Predictor。

        数据流（对话场景的 I-JEPA 适配）：
        - 对每对对话 (user, assistant)：
          · 随机选择 mask 方向：user→assistant 或 assistant→user
          · visible（context）→ Online Encoder → SSL Predictor → 预测 masked 表示
          · masked（target）→ EMA Encoder → 作为预测目标
        - 双向 masking 让 Encoder 学习：
          · user→assistant：从问题预测回答的抽象表示（理解因果）
          · assistant→user：从回答逆推问题的抽象表示（理解意图）

        dialogue_pairs: [(user_text, assistant_text), ...]
        epochs: 预训练轮数（建议 20-50，视 loss 收敛情况）
        """
        print("\n" + "=" * 60)
        print("  自监督预训练 v8.0 (Context Masking · B路线)")
        print("  目标：训练 Encoder 学习高质量对话表示")
        print("=" * 60)

        N = len(dialogue_pairs)

        # 预加载所有对话文本的 LLM embedding 到缓存
        for u, a in dialogue_pairs:
            self._get_cached_emb(u)
            self._get_cached_emb(a)

        # 堆叠为 Tensor
        user_embs = torch.stack([self._get_cached_emb(u) for u, _ in dialogue_pairs])   # (N, D_llm)
        asst_embs = torch.stack([self._get_cached_emb(a) for _, a in dialogue_pairs])   # (N, D_llm)

        # 预训练只更新 JEPA 参数（group 0），不动 KGE（group 1）
        self.jepa.train()
        self._set_lr(config.dream_lr)

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # 双向 Context Masking：两个方向都训练
            # 方向 1：user（context）→ 预测 assistant（target）
            ssl_loss_ua, sigreg_ua, z_pred_ua = self.jepa.compute_context_masking_loss(
                user_embs, asst_embs)

            # 方向 2：assistant（context）→ 预测 user（target）
            ssl_loss_au, sigreg_au, z_pred_au = self.jepa.compute_context_masking_loss(
                asst_embs, user_embs)

            # 合并双向 loss
            ssl_loss = (ssl_loss_ua + ssl_loss_au) / 2
            sigreg_loss = (sigreg_ua + sigreg_au) / 2

            total = ssl_loss + config.jepa_sigreg_weight * sigreg_loss

            total.backward()

            # 预训练不更新 KGE（group 1）
            for p in self.optimizer.param_groups[1]['params']:
                if p.grad is not None:
                    p.grad.zero_()

            self.optimizer.step()
            self.jepa._update_ema()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [预训练 {epoch+1}/{epochs}] "
                      f"SSL: {ssl_loss.item():.4f} | "
                      f"SIGReg: {sigreg_loss.item():.4f}")

        self.jepa.eval()
        self._set_lr(config.online_lr)
        print(f"  >> 自监督预训练完成（{epochs} epochs）")

    # ==== 阶段 2: 深度做梦（v8.0 Soft Prompt 重构） ====

    def train_dream_phase(self, dialogue_pairs: list, epochs: int = 5):
        """
        v7.0 Soft Prompt 深度做梦训练。

        核心改造（对比 v6.0）：
        - 注入机制：KV-Cache monkey-patch → Soft Prompt 序列头部拼接
        - CE 梯度通路：attention → virtual_embeds → trajectory_proj → JEPA trajectory
        - 完整轨迹传递：不再只用 final_state，将 rollout 全轨迹投射为虚拟 token
        """
        print("\n" + "=" * 60)
        print("  深度做梦 v7.0 (Soft Prompt 轨迹注入 + Batch 化)")
        print("=" * 60)

        N = len(dialogue_pairs)

        # 提前全部加载到全局缓存
        for u, a in dialogue_pairs:
            self._get_cached_emb(u)
            self._get_cached_emb(a)

        # ==== 预处理：记忆巩固 ====
        print(f"  >> [海马体]: TransE 巩固将在训练循环中与 JEPA 联合优化。")

        # v5.5 #6: summarize_fn Prompt 走配置模板
        def summarize_fn(text1: str, text2: str) -> str:
            prompt = config.summary_prompt_template.format(
                text1=text1, text2=text2)
            input_ids = self.cortex.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).to(config.device)["input_ids"]
            with torch.no_grad():
                out = self.cortex.model.generate(
                    input_ids, max_new_tokens=config.abstraction_max_tokens,
                    pad_token_id=self.cortex.tokenizer.eos_token_id, do_sample=False)
            summary = self.cortex.tokenizer.decode(
                out[0][input_ids.shape[-1]:], skip_special_tokens=True)
            return f"[抽象] {summary.strip()}"

        # ==== 预处理：记忆合并 ====
        merged = self.hippocampus.merge_similar(
            get_embedding_fn=self._get_cached_emb, summarize_fn=summarize_fn)
        if merged > 0:
            print(f"  >> 记忆合并: {merged} 对相似记忆合并为抽象概念")

        # ==== 预处理：模式提精 ====
        patterns = self.hippocampus.extract_patterns(
            get_embedding_fn=self._get_cached_emb, summarize_fn=summarize_fn)
        if patterns:
            print(f"  >> 模式提精: 发现 {len(patterns)} 个重复模式")
            for p in patterns:
                print(f"     * {p['key']} (出现 {p['count']} 次)")

        # ==== v5.5 Batch 预构建：将所有对话对的 embedding 堆叠为 Tensor ====
        user_embs = torch.stack([self._get_cached_emb(u) for u, _ in dialogue_pairs])   # (N, D_llm)
        tgt_embs = torch.stack([self._get_cached_emb(a) for _, a in dialogue_pairs])    # (N, D_llm)

        # 提前为每条对话对检索记忆并提取 k-means 簇心（检索+聚类不需要梯度）
        # v5.7 梯度流修复：训练循环内带梯度重新投影 cluster_centers
        per_sample_memories = []
        per_sample_cluster_centers = []  # v5.7: 存储无梯度的 k-means 簇心
        for u, _ in dialogue_pairs:
            user_emb = self._get_cached_emb(u)
            mems = self.hippocampus.retrieve(user_emb)
            per_sample_memories.append(mems)
            # v5.7 梯度修复：只做聚类（no_grad），投影留到 epoch 循环内执行
            cluster_centers = self.jepa.extract_memory_cluster_centers(mems, self.cortex)
            per_sample_cluster_centers.append(cluster_centers)

        # ==== v20 断裂点 #1 修复: target 来自海马体外部事实，不是 EMA 自蒸馏 ====
        # encode_fact_target 用 online encoder (detached) 编码真实后继事件
        # 打破「模型自己评估自己」的衔尾蛇死循环
        with torch.no_grad():
            target_jepa_all = self.jepa.encode_fact_target(tgt_embs)  # (N, D_jepa)

        self.jepa.train()
        self._set_lr(config.dream_lr)

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # ==== v8.0 关键：每 epoch 重新编码 init_emb（带梯度） ====
            # 必须在 epoch 循环内，因为 backward() 会释放计算图
            # 每 epoch 需要新的计算图让梯度从 loss 回传到 Encoder
            init_jepa_all = self.jepa.encode(user_embs)  # (N, 1, D_jepa)，带梯度
            init_jepa_all = init_jepa_all.squeeze(1)     # (N, D_jepa)

            # v5.7 梯度修复：每 epoch 重新投影簇心 → 意图空间（带梯度）
            per_sample_mem_intents = []
            for cc in per_sample_cluster_centers:
                if cc is not None:
                    mem_intents = self.jepa.project_cluster_centers(cc)
                else:
                    mem_intents = torch.zeros(0, config.jepa_intent_dim, device=config.device)
                per_sample_mem_intents.append(mem_intents)

            # ============ v20 JEPA Prediction Loss (批量化 B=N + InfoNCE) ============
            # target_jepa_all 来自 encode_fact_target（海马体外部事实）
            # v20-audit S3 修复: 传入 memory_intents 而非 None
            # 原来训练时永远 memory_intents=None，只用固定 intent_bank，
            # 而推理时使用 memory_intents，导致训练/推理分布不一致。
            # 另外 memory_intent_proj 梯度路径也因此永远不活跃（T4 发现）。
            # 现在合并所有样本的 memory_intents，传入 compute_prediction_loss。
            merged_mem_intents = None
            non_empty = [mi for mi in per_sample_mem_intents if mi.numel() > 0]
            if non_empty:
                # 取所有样本的 memory intents 的并集（去重复）
                merged_mem_intents = torch.cat(non_empty, dim=0)  # (M_total, intent_dim)
                # 截止到 max_memory_intents 防止 S_total 爆炸
                M_max = config.jepa_max_memory_intents
                if merged_mem_intents.size(0) > M_max:
                    merged_mem_intents = merged_mem_intents[:M_max]

            jepa_total, pred_loss_val, sig_loss_val, div_loss_val, \
                final_pred_batch, _, kl_loss_val, contrast_loss_val = \
                self.jepa.compute_prediction_loss(
                    init_jepa_all, target_jepa_all, memory_intents=merged_mem_intents)

            total_pred_loss = jepa_total
            total_div_loss = div_loss_val
            total_kl_loss = kl_loss_val
            total_contrast_loss = contrast_loss_val
            all_final_preds = [final_pred_batch]  # 已经是 (N*S, D)

            # v8.1 批量 best_intent_idx：为 CE rollout 选最优意图
            # 每样本仍需独立选意图（因为 per-sample memory_intents 影响选择）
            best_intent_indices = []
            for i in range(N):
                with torch.no_grad():
                    # 快速评估：用 init_emb 和 target_emb 的 MSE 选意图
                    all_intents_i = self.jepa.get_all_intents(per_sample_mem_intents[i])
                    S_i = all_intents_i.size(0)
                    init_exp = init_jepa_all[i:i+1].unsqueeze(1).expand(S_i, 1, -1)
                    conds = self.jepa.intent_encoder(all_intents_i).unsqueeze(1)
                    traj = init_exp
                    for step in range(config.jepa_rollout_depth):
                        pred = self.jepa.predict(
                            traj[:, -config.jepa_rollout_history:, :],
                            conds[:, :1, :].expand(-1, traj[:, -config.jepa_rollout_history:, :].size(1), -1))
                        traj = torch.cat([traj, pred[:, -1:, :]], dim=1)
                    final = traj[:, -1, :]
                    costs = (final - target_jepa_all[i:i+1].expand(S_i, -1)).pow(2).mean(dim=-1)
                    best_intent_indices.append(costs.argmin().item())

            # v8.1 批量 query alignment loss
            all_intents_base = self.jepa.get_all_intents(None)  # 共享意图
            qa_losses = []
            for i in range(N):
                all_intents_i = self.jepa.get_all_intents(per_sample_mem_intents[i])
                best_intent_vec = all_intents_i[best_intent_indices[i]]
                init_emb_i = init_jepa_all[i:i+1].unsqueeze(1)
                trajectory_i = self.jepa.rollout_with_intent_vec(init_emb_i, best_intent_vec)
                final_state_i = trajectory_i[:, -1, :]
                qa_loss_i = self.jepa.compute_query_alignment_loss(
                    final_state_i, tgt_embs[i:i+1])
                qa_losses.append(qa_loss_i)
            total_query_align_loss = sum(qa_losses)

            # v20-audit S2 修复: compute_prediction_loss 内部的 MSE 已经是 .mean()，
            # 再除以 N 会让 pred_loss 权重被人为缩小 N 倍，导致与其他 loss 项比例失调
            avg_jepa_loss = total_pred_loss
            avg_query_align_loss = total_query_align_loss / N

            # v5.6 P0-2: batch SIGReg（用所有样本积累的预测向量，解决 B=1 统计失效）
            all_final_preds_cat = torch.cat(all_final_preds, dim=0)  # (N*S_total, D)
            batch_sigreg_loss = self.jepa.compute_sigreg_on_batch(
                all_final_preds_cat.reshape(-1, config.jepa_core_dim))

            # ============ v20: CE loss 已移除 ============
            # v19 的 CE loss 通过 Soft Prompt 注入训练 injection_layer
            # v20 不再使用 Soft Prompt 注入，injection_layer 不参与世界模型训练
            # JEPA 的训练信号完全来自：
            #   1. MSE prediction loss（目标 = 海马体事实）
            #   2. InfoNCE 对比损失（锐化判别）
            #   3. KL / SIGReg / 意图多样性（正则化）

            # ============ KGE/Proj Loss（每 epoch 只算一次） ============
            kge_loss, proj_loss = self.hippocampus.compute_consolidation_losses()

            # ============ v20 联合损失（无 CE） ============
            total_loss = (config.jepa_pred_loss_weight * avg_jepa_loss
                          + config.jepa_sigreg_weight * batch_sigreg_loss
                          + config.kge_loss_weight * kge_loss
                          + config.proj_loss_weight * proj_loss
                          + config.query_alignment_loss_weight * avg_query_align_loss)

            # ============ 单次 Batch 更新 ============
            total_loss.backward()
            self.optimizer.step()

            # v20-audit M2: EMA 目标编码器在 v20 dream 阶段不再使用
            # （v20 用 encode_fact_target 而非 encode_target，EMA 更新纯属无效计算）
            # EMA 仅保留用于 SSL 预训练阶段（compute_context_masking_loss 调用 ema_encoder）

            print(f"  [周期 {epoch+1}/{epochs}] "
                  f"Pred: {(total_pred_loss/N).item():.4f} | "
                  f"InfoNCE: {total_contrast_loss.item():.4f} | "
                  f"SIGReg: {batch_sigreg_loss.item():.4f} | "
                  f"IntDiv: {(total_div_loss/N).item():.4f} | "
                  f"KL: {(total_kl_loss/N).item():.4f} | "
                  f"KGE: {kge_loss.item():.4f} | "
                  f"QAlign: {avg_query_align_loss.item():.4f}")

        # ==== 后处理：意图进化 ====
        self.jepa.evolve_intents()
        print(f"  >> 意图进化: intent_bank 已根据经验更新")

        self.jepa.eval()

        # P0-5: 恢复在线学习率
        self._set_lr(config.online_lr)

    # ==== 实时微梦 (Micro-Dream) v2 ====

    def _micro_dream(self, prev_input_emb: torch.Tensor,
                     real_next_input_emb: torch.Tensor,
                     memory_intents: torch.Tensor = None):
        """
        延迟微梦：用上一轮的输入和本轮真实输入训练 JEPA predictor。

        v5.6 P0-4: 只更新 JEPA+injection 参数组（group 0），
        不让 optimizer.step() 污染 hippocampus 的 Adam 动量状态。
        v5.6 P0-3: push 时同时存储 intent 空间投影向量。
        """
        self.jepa.train()
        for step in range(config.micro_dream_steps):
            self.optimizer.zero_grad()

            init_jepa = self.jepa.encode(prev_input_emb.unsqueeze(0)).squeeze(1)
            # v20: 微梦也用 encode_fact_target（外部真实事件），与做梦阶段一致
            target_jepa = self.jepa.encode_fact_target(real_next_input_emb.unsqueeze(0))
            jepa_total, pred_loss, _, _, _, _, _, _ = \
                self.jepa.compute_prediction_loss(init_jepa, target_jepa, memory_intents)
            jepa_total.backward()

            # v5.6 P0-4: 微梦只更新 group 0 (JEPA+injection)，不触碰 group 1 (hippocampus)。
            for p in self.optimizer.param_groups[1]['params']:
                if p.grad is not None:
                    p.grad.zero_()

            self.optimizer.step()
            self.jepa._update_ema()

        # v5.6 P0-3: 存入 replay_buffer 时同时存储当前时刻的 intent 空间投影
        with torch.no_grad():
            intent_emb = self.jepa.memory_intent_proj(
                real_next_input_emb.unsqueeze(0)).squeeze(0)  # (intent_dim,)
        self.jepa.replay_buffer.push(prev_input_emb, real_next_input_emb, intent_emb)
        self.jepa.eval()
        return pred_loss.item()

    # ==== 阶段 3: 认知交互 ====

    def interact(self, user_input: str) -> str:
        """
        v20 认知交互流程（MetaLanguage 文本元语注入）。

        v20 数据流（断裂点 #2 修复）：
        1. 用户输入 → LLM embedding
        2. 海马体检索 → 记忆驱动意图
        3. 延迟微梦（可选）
        4. JEPA rollout → 完整轨迹
        5. MetaLanguageDecoder → 轨迹解码为结构化文本元语
        6. 元语 + 用户输入 → LLM generate（文本级注入，不是 embedding 级）
        """
        with torch.no_grad():
            current_feat = self.cortex.get_real_embedding(
                user_input).to(torch.float32)

        # 海马体检索（先检索，用于微梦和后续推演）
        with torch.no_grad():
            pre_memories = self.hippocampus.retrieve(current_feat)
            memory_intents = self.jepa.extract_memory_intents(
                pre_memories, self.cortex)

        # 延迟微梦（传入 memory_intents）
        if self._pending_experience is not None:
            prev_emb = self._pending_experience
            with torch.enable_grad():
                pred_l = self._micro_dream(prev_emb, current_feat, memory_intents)

        # 海马体目标搜索
        with torch.no_grad():
            basic_seeds = self.hippocampus.find_seeds(
                current_feat, config.memory_top_k)

            goal_emb = None
            if basic_seeds:
                outcomes = self.hippocampus.find_outcome_nodes(basic_seeds)
                if outcomes:
                    outcome_id, outcome_text, hops = outcomes[0]
                    goal_emb = self.cortex.get_real_embedding(
                        outcome_text).to(torch.float32)

        # JEPA 统一规划器（CEM-lite + Prior + Latent Overshooting）
        trajectory = self.jepa.plan_with_prior(
            current_feat, goal_emb=goal_emb, memory_intents=memory_intents)

        # v20 断裂点 #2 修复：元语解码 + 文本级注入（替代 Soft Prompt）
        # JEPA 轨迹 → 结构化文本元语 → LLM 通过自然语言理解能力解析方向指引
        # 不再在 embedding 层注入连续向量（那对 LLM 是“外星噪声”）
        best_intent_idx = 0  # plan_with_prior 已内部选择最优意图
        meta_text = self.meta_decoder.decode_trajectory(
            trajectory, best_intent_idx, goal_emb, pre_memories)

        with torch.no_grad():
            steered_text = self.cortex.generate_with_context(
                meta_text, user_input)

        # v20-audit F4 增强: 输出侧清洗泄漏的元语标记
        # prompt-level 指令对小模型（0.5B）不够有效，LLM 仍可能复制内部标记
        # 用正则清洗确保用户永远看不到 [JEPA预演] [方向] [关键记忆] 等系统结构
        import re
        steered_text = re.sub(
            r'\[(?:JEPA预演|方向|关键记忆|置信度|推理稳定性)[^\]]*\]\s*', '', steered_text)
        # 清理可能残留的 "| [标记] 值%" 格式片段
        steered_text = re.sub(r'\|\s*\[[^\]]+\]\s*\d*%?\s*', '', steered_text)
        steered_text = steered_text.strip()

        self._pending_experience = current_feat.detach().clone()

        # 对话记忆化
        if config.memory_auto_extract and steered_text.strip():
            self.hippocampus.memorize_dialogue(
                user_input, steered_text, self.cortex,
                user_emb=current_feat)

        return steered_text

    def benchmark_interact(self, user_input: str) -> str:
        """v20: 带详细终端打印和 Benchmark 对比的包装器"""
        print(f"\n{'='*60}")
        print(f"[用户]: {user_input}")
        print(f"{'='*60}")

        # 1. 跑一次纯 LLM 进行对比（无注入）
        input_ids = self.cortex.tokenizer(
            user_input, return_tensors="pt").to(config.device)["input_ids"]
        with torch.no_grad():
            base_out = self.cortex.model.generate(
                input_ids, max_new_tokens=config.generation_max_tokens,
                pad_token_id=self.cortex.tokenizer.eos_token_id,
                do_sample=False)
            baseline = self.cortex.tokenizer.decode(
                base_out[0][input_ids.shape[-1]:], skip_special_tokens=True)

        # 2. 调用 v20 核心业务流（MetaLanguage 文本元语注入）
        steered_text = self.interact(user_input)

        print(f"\n  >>> v20 架构效果对比:")
        print(f"  [纯 LLM (冻结)]: {baseline}")
        print(f"  [JEPA预演 + 海马体元语]: {steered_text}")

        return steered_text


# =====================================================================
# 主程序（P2-1: 测试数据从外部 JSON 加载）
# =====================================================================
def _load_scenarios(path: str = None) -> dict:
    """从外部 JSON 文件加载测试场景。找不到文件时使用内嵌默认数据。"""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "scenarios.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # 内嵌默认数据（兜底，确保无 JSON 文件时也能运行）
    return {
        "memories": [
            {"text": "今天和老板开会，他否定了我做了一周的方案。"},
            {"text": "每次努力不被认可时，我都会感到极度焦虑。",
             "link_from_idx": 0, "edge_type": "心理反馈"},
            {"text": "深呼吸，这不是我的错，慢慢来。",
             "link_from_idx": 1, "edge_type": "应对策略"},
            {"text": "上次被否定后，我用番茄工作法把方案拆分成了小目标。",
             "link_from_idx": 0, "edge_type": "行动经验"},
            {"text": "小目标逐个攻克的过程中，焦虑感明显下降了。",
             "link_from_idx": 3, "edge_type": "效果验证"}
        ],
        "dialogues": [
            ["啊啊啊又被退回重写了，我真的要崩溃了！", "深呼吸，这不是我的错，慢慢来。"],
            ["这次怎么做都不对，老板就是针对我。", "上次被否定后用番茄工作法拆分，焦虑就下降了，试试？"],
            ["又一个通宵白写了，我好累", "先休息，小目标一步步来。"],
            ["为什么每次努力都不被认可呢", "被否定很难受，但试试拆解问题。"]
        ],
        "test_queries": [
            "啊啊啊又被退回重写了，我真的要崩溃了！",
            "为什么我总是这么焦虑",
            "有没有什么具体的办法缓解这种压力"
        ]
    }


def main():
    import sys
    scenario_path = sys.argv[1] if len(sys.argv) > 1 else None
    scenarios = _load_scenarios(scenario_path)

    n_mem = len(scenarios.get("memories", []))
    n_dia = len(scenarios.get("dialogues", []))
    n_test = len(scenarios.get("test_queries", []))
    source = scenario_path or "默认内嵌"

    print("=" * 60)
    print("  三位一体认知架构 v20 (MetaLanguage 文本元语注入)")
    print(f"  课程: {source} ({n_mem} 记忆 · {n_dia} 对话 · {n_test} 测试)")
    print("  梯度通路: 海马体事实 → InfoNCE → JEPA predictor + Encoder")
    print("=" * 60)

    brain = TheBrainMVP()

    # ---- 阶段 1: 记忆建库 ----
    print("\n--- 阶段 1: 记忆建库 (自动建边) ---")
    memory_ids = []
    for mem in scenarios["memories"]:
        linked_from = None
        if "link_from_idx" in mem:
            linked_from = memory_ids[mem["link_from_idx"]]
        nid = brain.ingest(mem["text"], linked_from, mem.get("edge_type"))
        memory_ids.append(nid)

    edge_types = {}
    for _, _, d in brain.hippocampus.graph.edges(data=True):
        t = d.get('type', '?')
        edge_types[t] = edge_types.get(t, 0) + 1
    print(f"\n图谱: {brain.hippocampus.stats()}")
    print(f"边类型分布: {edge_types}")

    # ---- 阶段 1.5: 自监督预训练（v8.0 B路线） ----
    dialogues = [tuple(pair) for pair in scenarios["dialogues"]]
    brain.pretrain_encoder_phase(dialogues, epochs=30)

    # ---- 阶段 2: 深度做梦 ----
    brain.train_dream_phase(dialogues, epochs=50)

    # ---- 阶段 3: 认知交互 ----
    print("\n--- 阶段 3: 认知交互 (Benchmark 模式) ---")
    for query in scenarios["test_queries"]:
        brain.benchmark_interact(query)

    print(f"\n最终图谱: {brain.hippocampus.stats()}")
    print(f"JEPA 经验池: {len(brain.jepa.replay_buffer)}/{config.replay_buffer_size}")

    # ---- 验证 SQLite 恢复 ----
    print("\n--- 验证 SQLite 持久化恢复与双引一致性 ---")
    brain.hippocampus.close()
    test_db = HippocampalMemory()
    print(f"恢复后图谱状态: {test_db.stats()}")
    test_db.close()


if __name__ == "__main__":
    main()
