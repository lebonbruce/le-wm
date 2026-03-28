"""
cortex/linguistic.py —— 冻结的语言皮层（Broca's Area）

v5.4 P1-2 修复：不继承 nn.Module（所有参数已冻结，不应出现在 state_dict 中）。
只负责：① 输入 → 896维语义特征  ② 接受注入后的隐状态 → 自然语言输出
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mvp_config import config


class LinguisticCortex:
    """
    冻结的语言皮层。

    v5.4 P1-2: 不继承 nn.Module。原因：
    - LLM 参数全部冻结（requires_grad=False），不参与训练
    - 如果注册为 nn.Module 子模块，TheBrainMVP.state_dict() 会包含完整 LLM 权重（~1GB），
      导致 checkpoint 体积爆炸且不必要——LLM 应始终从 HuggingFace 原模型加载。
    - 参数统计会被 LLM 的 5亿参数淹没，掩盖真正可训练参数的规模。
    """
    def __init__(self):
        print(f"Loading Base LLM ({config.llm_model_id}) on {config.device}...")
        self.dtype = torch.float16 if config.use_fp16 and config.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config.llm_model_id, torch_dtype=self.dtype, device_map=config.device
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def get_real_embedding(self, text: str) -> torch.Tensor:
        """
        Softmax Attention-weighted pooling（替代 mean pooling）。

        问题：mean pooling 对所有 token 等权平均，长文本中语义信号互相抵消，
        产生"Grey Mush"效应——向量趋于所有 token 的平均灰色，失去区分度。

        解法：用每个 token 的 hidden state 的 L2 范数作为注意力权重，
        通过 softmax(norms / temperature) 归一化产生尖锐分布。
        v5.6 P1-6 修复：旧版使用线性归一化（norms/sum），当 token 范数差异小时
        退化为 uniform 权重 ≈ mean pooling。改用 softmax + 温度系数产生
        更有区分度的权重分布，信息量高的 token 获得指数级更高权重。
        """
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512,
            padding=True
        ).to(config.device)
        out = self.model(**inputs, output_hidden_states=True)

        # 取最后一层 hidden states: (1, T, D)
        hidden = out.hidden_states[-1].float()  # 统一转 float32 避免精度问题
        mask = inputs["attention_mask"].float()  # (1, T)

        # 权重 = softmax(L2 范数 / 温度)，在有效 token 上做 masked softmax
        token_norms = hidden.norm(dim=-1)  # (1, T)
        # 将 padding 位置设为 -inf 使 softmax 输出为 0
        token_norms = token_norms.masked_fill(mask == 0, float('-inf'))
        # v5.6: softmax 归一化（温度系数控制尖锐度，越小越尖锐）
        weights = torch.softmax(token_norms / config.pooling_temperature, dim=1)  # (1, T)
        weights = weights.unsqueeze(-1)  # (1, T, 1)

        # 加权求和
        pooled = (hidden * weights).sum(dim=1).squeeze()  # (D,)
        return pooled

    @torch.no_grad()
    def generate_with_context(self, context: str, user_input: str,
                              max_new_tokens: int = None) -> str:
        """
        v20: 将元语上下文 + 用户输入拼接为 prompt，让 LLM 生成回答。

        这是断裂点 #2 的修复：替代 Soft Prompt embedding 级注入。
        
        为什么文本级注入有效：
        - LLM 的 Transformer 注意力流形是基于自然语言 token 建立的
        - 文本 token 是 LLM 的"母语"，它天然能理解结构化文本指令
        - 元语文本本质上是一种高质量的系统级 prompt
        - Soft Prompt 连续向量是"外星噪声"，文本是"人话"
        
        context: JEPA 预演产出的结构化文本元语
                 （来自 MetaLanguageDecoder.decode_trajectory）
        user_input: 用户原始输入
        max_new_tokens: 最大生成 token 数（None 则使用配置默认值）
        
        返回: LLM 生成的回答文本
        """
        max_tokens = max_new_tokens or config.generation_max_tokens
        
        # 构造完整 prompt：系统级元语上下文在前，用户输入在后
        # v20-audit F4 修复: 添加分隔指令防止元语标记泄漏到 LLM 输出
        # 原来直接拼接 "{context}\n用户: ..." 导致 LLM 把 [方向] [关键记忆] 等标记
        # 当作需要生成的格式，在输出中泄漏内部结构
        prompt = (
            f"系统参考信息（仅供参考，不要在回答中包含方括号标记）：\n"
            f"{context}\n"
            f"---\n"
            f"用户: {user_input}\n"
            f"回答:"
        )
        
        input_ids = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512
        ).to(config.device)["input_ids"]
        
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
        )
        
        # 只提取新生成的 token（跳过输入部分）
        generated = self.tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        return generated.strip()
