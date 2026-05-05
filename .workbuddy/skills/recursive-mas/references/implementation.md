# RecursiveMAS 实现指南

## 目录

1. [环境准备](#1-环境准备)
2. [RecursiveLink 实现](#2-recursivelink-实现)
3. [Agent 封装](#3-agent-封装)
4. [隐空间自回归生成](#4-隐空间自回归生成)
5. [Inner-Loop 训练](#5-inner-loop-训练)
6. [Outer-Loop 训练](#6-outer-loop-训练)
7. [推理流程](#7-推理流程)
8. [四种协作模式实现](#8-四种协作模式实现)
9. [训练数据准备](#9-训练数据准备)

---

## 1. 环境准备

```bash
# 基础依赖
pip install torch transformers accelerate

# 可选：Flash Attention 加速
pip install flash-attn --no-build-isolation

# 训练依赖
pip install peft datasets wandb
```

硬件要求：
- Inner-Loop 训练：单 GPU（≥16GB VRAM per 1.5B agent）
- Outer-Loop 训练：多 GPU 推荐（需同时加载所有 Agent）
- 推理：单 GPU 可行（Agent 按顺序加载/卸载）

---

## 2. RecursiveLink 实现

```python
import torch
import torch.nn as nn

class InnerRecursiveLink(nn.Module):
    """Agent 内部隐空间自回归的残差投影模块
    
    ℛ_in(h) = h + W₂·GELU(W₁·h)
    """
    def __init__(self, hidden_dim: int, intermediate_dim: int = None):
        super().__init__()
        intermediate_dim = intermediate_dim or hidden_dim * 2
        self.W1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.W2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act = nn.GELU()
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.W2(self.act(self.W1(h)))


class OuterRecursiveLink(nn.Module):
    """跨 Agent 隐状态传递的残差投影模块
    
    ℛ_out(h) = W₃·h + W₂·GELU(W₁·h)
    
    W₃ 处理异构模型的维度映射：d_src → d_tgt
    """
    def __init__(self, src_dim: int, tgt_dim: int, intermediate_dim: int = None):
        super().__init__()
        intermediate_dim = intermediate_dim or max(src_dim, tgt_dim) * 2
        self.W3 = nn.Linear(src_dim, tgt_dim, bias=False)  # 维度映射
        self.W1 = nn.Linear(src_dim, intermediate_dim, bias=False)
        self.W2 = nn.Linear(intermediate_dim, tgt_dim, bias=False)
        self.act = nn.GELU()
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.W3(h) + self.W2(self.act(self.W1(h)))
```

**参数量分析**（dₕ=2048, intermediate=4096）：
- Inner Link: 2048×4096 + 4096×2048 ≈ 16.8M 参数
- Outer Link (同维度): 2048×2048 + 16.8M ≈ 21.0M 参数
- 对比 Agent 全参数（1.5B）: 仅占 1.1%–1.4%

---

## 3. Agent 封装

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class RecursiveAgent:
    def __init__(self, model_name: str, role: str, device: str = "cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.role = role
        self.device = device
        self.hidden_dim = self.model.config.hidden_size
        
        # 冻结 LLM 参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 初始化 RecursiveLink
        self.inner_link = InnerRecursiveLink(self.hidden_dim).to(device)
    
    def encode_input(self, text: str) -> torch.Tensor:
        """文本 → 输入嵌入"""
        tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
        return self.model.get_input_embeddings()(tokens.input_ids), tokens
    
    def get_last_hidden(self, input_embeds: torch.Tensor) -> torch.Tensor:
        """前向传播获取 last-layer 隐状态"""
        with torch.no_grad():
            outputs = self.model(inputs_embeds=input_embeds, output_hidden_states=True)
        return outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
```

---

## 4. 隐空间自回归生成

```python
def latent_thoughts_generation(
    agent: RecursiveAgent,
    input_embeds: torch.Tensor,
    num_steps: int = 4
) -> torch.Tensor:
    """在隐空间中自回归生成 latent thoughts
    
    每步：
    1. 前向传播得到 hₜ
    2. ℛ_in(hₜ) → eₜ₊₁（映射回输入空间）
    3. 拼接 eₜ₊₁，继续下一步
    
    返回所有 latent thoughts 的序列
    """
    all_latent_thoughts = []
    current_embeds = input_embeds
    
    for step in range(num_steps):
        # 前向获取 last-layer 隐状态
        h_t = agent.get_last_hidden(current_embeds)
        
        # 记录最后位置的隐状态
        last_h = h_t[:, -1:, :]  # (batch, 1, hidden_dim)
        all_latent_thoughts.append(last_h)
        
        # Inner Link: 隐状态 → 下一步输入嵌入
        next_embed = agent.inner_link(last_h)  # (batch, 1, hidden_dim)
        
        # 拼接继续自回归
        current_embeds = torch.cat([current_embeds, next_embed], dim=1)
    
    return torch.cat(all_latent_thoughts, dim=1)  # (batch, num_steps, hidden_dim)
```

---

## 5. Inner-Loop 训练

```python
def train_inner_loop(
    agent: RecursiveAgent,
    train_data: list,  # [(question, answer), ...]
    epochs: int = 3,
    lr: float = 5e-4
):
    """Stage 1: 训练每个 Agent 的 Inner RecursiveLink
    
    损失: ℒ_in = 1 - cos(ℛ_in(H), Emb_θ(y))
    目标: 对齐 latent thoughts 与标准嵌入的语义分布
    """
    optimizer = torch.optim.AdamW(agent.inner_link.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        for question, answer in train_data:
            optimizer.zero_grad()
            
            # 1. 编码输入
            input_embeds, _ = agent.encode_input(question)
            
            # 2. 生成 latent thoughts
            H = latent_thoughts_generation(agent, input_embeds, num_steps=4)
            
            # 3. 编码 ground-truth 文本为嵌入
            with torch.no_grad():
                answer_tokens = agent.tokenizer(answer, return_tensors="pt").to(agent.device)
                target_embeds = agent.model.get_input_embeddings()(answer_tokens.input_ids)
                target_embeds = target_embeds.mean(dim=1, keepdim=True)  # (batch, 1, dim)
            
            # 4. Inner Link 映射
            predicted = agent.inner_link(H[:, -1:, :])  # 最后一步的隐状态
            
            # 5. 余弦相似度损失
            cos_sim = torch.nn.functional.cosine_similarity(predicted, target_embeds, dim=-1)
            loss = 1 - cos_sim.mean()
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    return agent
```

---

## 6. Outer-Loop 训练

```python
class RecursiveMASSystem:
    """递归多智能体系统"""
    
    def __init__(self, agents: list, collaboration_pattern: str = "sequential"):
        self.agents = agents
        self.pattern = collaboration_pattern
        self.outer_links = nn.ModuleList()
        
        # 初始化 Agent 间的 Outer RecursiveLinks
        for i in range(len(agents)):
            src_dim = agents[i].hidden_dim
            tgt_dim = agents[(i + 1) % len(agents)].hidden_dim
            self.outer_links.append(
                OuterRecursiveLink(src_dim, tgt_dim).to(agents[i].device)
            )
        
        # 闭环：最后一个 Agent 的输出回到第一个
        self.close_link = OuterRecursiveLink(
            agents[-1].hidden_dim, agents[0].hidden_dim
        ).to(agents[0].device)
    
    def forward_recursive(
        self,
        question: str,
        recursion_rounds: int = 2,
        latent_steps: int = 4
    ) -> torch.Tensor:
        """执行 n 轮递归"""
        
        # 初始输入
        first_agent = self.agents[0]
        current_embeds, _ = first_agent.encode_input(question)
        
        for round_r in range(recursion_rounds):
            transferred_info = None
            
            for i, agent in enumerate(self.agents):
                # 拼接输入：自身指令 + 来自前一个 Agent 的隐信息
                if transferred_info is not None:
                    agent_input = torch.cat([current_embeds, transferred_info], dim=1)
                else:
                    agent_input = current_embeds
                
                # 隐空间自回归
                H = latent_thoughts_generation(agent, agent_input, latent_steps)
                
                # Outer Link 传递给下一个 Agent
                if i < len(self.agents) - 1:
                    transferred_info = self.outer_links[i](H)
                else:
                    # 最后一个 Agent：闭环回到第一个
                    transferred_info = self.close_link(H)
                    current_embeds = transferred_info  # 下一轮的输入
        
        # 最终轮：最后一个 Agent 解码文本
        last_agent = self.agents[-1]
        last_h = H[:, -1:, :]
        logits = last_agent.model.lm_head(last_h)
        return logits


def train_outer_loop(
    system: RecursiveMASSystem,
    train_data: list,
    recursion_rounds: int = 2,
    epochs: int = 5,
    lr: float = 5e-4
):
    """Stage 2: 系统级 Outer-Loop 训练
    
    梯度沿完整递归路径回传，共享信用分配
    """
    all_outer_params = (
        list(system.outer_links.parameters()) + 
        list(system.close_link.parameters())
    )
    optimizer = torch.optim.AdamW(all_outer_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        for question, answer in train_data:
            optimizer.zero_grad()
            
            # 前向：n 轮递归
            logits = system.forward_recursive(question, recursion_rounds)
            
            # 计算交叉熵损失
            answer_ids = system.agents[-1].tokenizer(
                answer, return_tensors="pt"
            ).input_ids.to(system.agents[-1].device)
            
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                answer_ids.view(-1)
            )
            
            # 梯度沿完整递归路径回传
            loss.backward()
            optimizer.step()
        
        scheduler.step()
    
    return system
```

---

## 7. 推理流程

```python
def inference(
    system: RecursiveMASSystem,
    question: str,
    recursion_rounds: int = 2,
    latent_steps: int = 4,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_new_tokens: int = 512
) -> str:
    """RecursiveMAS 推理
    
    n 轮递归隐空间协作 → 最终轮解码文本输出
    """
    with torch.no_grad():
        first_agent = system.agents[0]
        current_embeds, _ = first_agent.encode_input(question)
        
        for round_r in range(recursion_rounds):
            transferred_info = None
            
            for i, agent in enumerate(system.agents):
                if transferred_info is not None:
                    agent_input = torch.cat([current_embeds, transferred_info], dim=1)
                else:
                    agent_input = current_embeds
                
                H = latent_thoughts_generation(agent, agent_input, latent_steps)
                
                if i < len(system.agents) - 1:
                    transferred_info = system.outer_links[i](H)
                else:
                    transferred_info = system.close_link(H)
                    current_embeds = transferred_info
        
        # 最终解码：仅由最后一个 Agent 生成文本
        last_agent = system.agents[-1]
        output_ids = last_agent.model.generate(
            inputs_embeds=current_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        
        return last_agent.tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

---

## 8. 四种协作模式实现

### Sequential Style

```
Planner → Critic → Solver (链式)
```
- Planner: 分析问题，生成计划
- Critic: 审查计划，指出缺陷
- Solver: 基于审查意见，输出最终答案
- 信息流：单向链式传递

### Mixture Style

```
Code Expert ──┐
Science Expert ──→ Summarizer
Math Expert  ──┘
```
- 多领域专家并行推理
- Summarizer 聚合所有专家的隐状态
- 信息流：并行汇聚

### Distillation Style

```
Expert ↔ Learner (双向)
```
- Expert 生成高质量隐表示
- Learner 从 Expert 的隐状态中学习
- 信息流：双向交互

### Deliberation Style

```
Reflector ↔ Tool-Caller (迭代直到共识)
```
- Reflector 反思当前方案
- Tool-Caller 调用外部工具获取信息
- 信息流：迭代循环直到收敛

---

## 9. 训练数据准备

### 数据集格式

```json
{
    "question": "Solve the equation x² - 5x + 6 = 0",
    "answer": "x = 2 or x = 3",
    "domain": "math",
    "difficulty": "medium"
}
```

### 数据集来源

| 数据集 | 领域 | 规模 | 获取方式 |
|--------|------|------|---------|
| s1K | 数学推理 | 1K | HuggingFace |
| m1k | 医学/科学 | 1K | HuggingFace |
| OpenCodeReasoning | 代码生成 | ~10K | HuggingFace |
| ARPO-SFT | 工具调用 | ~5K | HuggingFace |

### 数据预处理

```python
def preprocess_for_pattern(data: list, pattern: str) -> list:
    """按协作模式预处理训练数据"""
    processed = []
    for item in data:
        if pattern == "sequential":
            # 分解为 plan → critique → solve 三阶段
            processed.append({
                "question": item["question"],
                "plan": generate_plan(item),     # Planner 目标
                "critique": generate_critique(item),  # Critic 目标
                "answer": item["answer"]          # Solver 目标
            })
        elif pattern == "mixture":
            # 按领域标签分配
            processed.append({
                "question": item["question"],
                "domain": item["domain"],
                "answer": item["answer"]
            })
        elif pattern in ("distillation", "deliberation"):
            processed.append({
                "question": item["question"],
                "answer": item["answer"]
            })
    return processed
```
