---
id: "bidir-lm-causal-to-bidirectional-encoder"
title: "BidirLM：将因果 LLM 转换为全模态双向编码器的开源配方"
category: "多模态 RAG"
date: "2026-07-06"
order: 1
readTime: "15 min"
tags: ["Encoder", "Bidirectional", "Causal LLM", "Weight Merging", "Multimodal", "Representation Learning", "Gemma3", "Qwen3"]
summary: "BidirLM 提出一套完整的开源框架，将因果生成式 LLM（如 Gemma3、Qwen3）转换为双向编码器。核心发现是：被近期对比学习方法忽略的 Masking 阶段对微调性能至关重要；通过线性权重合并 + 轻量多域数据混合可缓解灾难性遗忘；还能通过合并专用因果模型无缝扩展到新模态（视觉、音频）。最终发布的 BidirLM-Omni-2.5B 在文本、图像、音频表征基准上超越更大模型。"
hero: "assets/papers/bidir-lm/training_method_delta.png"
---

## 一句话总结
BidirLM 的核心贡献不是某个单一技巧，而是一套**可复现、可扩展、可组合**的开源配方：先用 MNTP masking 解锁双向注意力潜力，再用对比学习提升通用嵌入质量，用权重合并 + 多域混合缓解遗忘，最后通过合并专用因果模型扩展到视觉和音频模态。这套方法让因果 LLM 生态中积累的数百万 GPU 小时知识能被编码器复用。

<div class="metric-grid">
  <div class="metric"><strong>5 个</strong><span>发布的 BidirLM 编码器变体</span></div>
  <div class="metric"><strong>+17</strong><span>BidirLM-Omni 相对 Nemotron-Omni-3B 的文本基准提升</span></div>
  <div class="metric"><strong>250 GPU·h</strong><span>构建全模态编码器所需额外计算量</span></div>
  <div class="metric"><strong>2.5B</strong><span>BidirLM-Omni 参数量，仅为竞品一半</span></div>
</div>

## 论文信息
- **标题**：BidirLM: From Text to Omnimodal Bidirectional Encoders by Adapting and Composing Causal LLMs
- **作者**：Nicolas Boizard, Théo Deschamps-Berger, Hippolyte Gisserot-Boukhlef, Céline Hudelot, Pierre Colombo
- **机构**：Diabolocom, Artefact Research Center, CentraleSupélec (Université Paris-Saclay), Cohere
- **版本**：arXiv:2604.02045v1, 2026-04-02
- **链接**：[arXiv](https://arxiv.org/abs/2604.02045) · [PDF](https://arxiv.org/pdf/2604.02045)

<figure class="figure">
  <img src="assets/papers/bidir-lm/training_method_delta.png" alt="不同适配策略的性能对比" loading="lazy" />
  <figcaption>图 1：五种适配变体在下游任务上的性能变化。仅开启双向注意力（Bi+Base）效果不稳定；加入 MNTP 后全面提升；对比学习单独使用牺牲微调性能换取嵌入质量；顺序执行 MNTP→Contrastive 兼顾两者。</figcaption>
</figure>

## 背景：为什么要把因果 LLM 变成编码器？
因果 LLM 已经成为生成任务的主流基座，并且围绕它积累了庞大的专用模型生态：代码、数学、安全、视觉、音频等变体代表了数百万 GPU 小时的开源知识。然而，**表征学习任务仍然绑定在 BERT 风格的双向编码器上**，这些因果模型的知识无法被直接复用。

将因果模型转换为编码器面临三个核心问题：

1. **什么驱动适配质量？** 现有方法混淆了训练目标、注意力机制等关键设计选择，缺乏共识。近期流行的纯对比学习方法跳过了 masking 阶段，但这是否合理？
2. **没有原始预训练数据时如何扩展？** 许多适配模型由基座模型的同一团队开发，可能隐式受益于未公开的预训练语料对齐，掩盖了分布偏移下的灾难性遗忘。独立复现者如何在严格数据约束下扩展？
3. **适配后的编码器能否与因果生态组合？** 当前方法采用固定管线，无法与同基座的其他专用因果模型组合，浪费了整个生态的专业化知识。

## 方法：四步开源配方
### 1. 双向注意力 + MNTP 是基础
论文定义了五种适配变体来解耦关键因素：

- **Base**：原始因果模型
- **Bi+Base**：仅切换为双向注意力
- **Bi+MNTP**：双向注意力 + Masked Next-Token Prediction 适配
- **Bi+Contrastive**：双向注意力 + InfoNCE 对比训练
- **Bi+MNTP+Contrastive**：先 MNTP 再对比训练

关键发现：**仅开启双向注意力（Bi+Base）效果不稳定**——在 token 分类和检索上有提升，但在 XNLI 和 Seahorse 上下降。**MNTP 阶段是解锁双向注意力潜力的关键**，加入后所有任务均有提升（Gemma3 在 XNLI 上 +0.8、Seahorse 上 +9.0）。

对比学习单独使用时，嵌入质量（MTEB）比 Bi+MNTP 高 13+ 分，但**牺牲了需要全参数微调的任务性能**。顺序执行 MNTP→Contrastive 能兼顾两者。

### 2. 权重合并 + 多域混合缓解灾难性遗忘
在没有原始预训练数据的情况下扩展适配会引发遗忘。论文提出双重策略：

**线性权重合并**：适配模型与基座模型在权重空间中距离很近（Gemma 余弦相似度 0.78，Qwen 0.97）。以约 50% 比例合并 30B token 适配检查点与原始基座，可恢复跨域性能：Gemma 在阿拉伯语 MNLI 和代码上 +6 分，Qwen 在数学上 +4 分。

**轻量多域数据混合**：将 20%-30% 的训练数据替换为多语言、数学、代码混合数据即可达到性能平台期。固定 20% 比例后再做权重合并，得到最优配置 **Multilingual+Merge**，在 XNLI/MIRACL 上平均 +2 分，代码基准上最高 +11 分。

<figure class="figure">
  <img src="assets/papers/bidir-lm/forgetting_interaction.png" alt="长程适配中的遗忘与缓解策略" loading="lazy" />
  <figcaption>图 2：从 10B 扩展到 30B token 适配时的性能演变。实线显示相对初始 10B 的绝对分数变化，虚线标注互补缓解方案的效果。单纯扩展导致明显遗忘，合并 + 多域混合可有效恢复。</figcaption>
</figure>

<figure class="figure">
  <img src="assets/papers/bidir-lm/merging_ratio.png" alt="不同合并比例下的性能" loading="lazy" />
  <figcaption>图 3：合并比例对各任务的影响。约 50% 比例在适配后的双向能力与基座分布覆盖之间取得最佳平衡。</figcaption>
</figure>

### 3. 与专用因果模型组合扩展新能力
权重合并不仅能缓解遗忘，还能**将专用因果模型的能力注入编码器**：

- **安全领域**：将 Qwen3Guard-Gen-0.6B 与 Bi+MNTP Qwen3-0.6B 以 50% 合并，仅需 500 步微调（MI250X 上 2 分钟），即在 Beaver/Safe/Aegis 上超越所有基线 1+ 分，且 20 步即达峰值性能的 93%。
- **视觉模态**：合并 Qwen3-VL-2B-Instruct 与 Bi+MNTP Qwen3-1.7B，在 e-SNLI-VE 上超越 Bi+Specialist 1+ 分、超越未合并基线 30+ 分。
- **音频模态**：合并 Qwen3-ASR-0.6B 与 Bi+MNTP Qwen3-0.6B，在 BoolQ-Audio 上超越 Bi+Specialist 15+ 分、超越未合并基线 19+ 分。值得注意的是，音频专家模型原本只支持单模态语音识别，与编码器**无共享模态**，合并仍然成功。

### 4. 构建全模态编码器 BidirLM-Omni
将三个 Qwen3-1.7B 变体（ASR、VL、Bi+MNTP）的文本骨干等比合并，拼接各自的音频和视觉头，再在多模态对比语料（Omni-Contrastive，180 万对）上训练，得到 **BidirLM-Omni-2.5B**。

<figure class="figure">
  <img src="assets/papers/bidir-lm/multimodal.png" alt="模态 specialization 训练曲线" loading="lazy" />
  <figcaption>图 4：视觉和音频模态适配的训练曲线。合并变体整体性能最高，但存在约 100-175 步的预热期，用于对齐内部表征与新模态头。</figcaption>
</figure>

## 数据与评测
### 适配语料
全部使用开源数据，分三类：

- **English**：FineWeb-Edu（masking）+ KaLM-embedding 英文子集（对比，7 hard negatives）
- **Multi-domain**：FineWeb-Edu + FineWeb2-HQ（20 语言）+ FineMath + Stack V2（34 语言代码）；对比训练用 89 个数据集合并语料
- **Multimodal（Omni-Contrastive）**：180 万对，65% 文本-文本 + 17.5% 音频-文本（Laion-Audio-300M + LibriSpeech ASR）+ 17.5% 图像-文本（Colpali + NatCap + MSCOCO）

### 评测协议
- **微调评测**：XTREME + 信息检索（MIRACL、CodeSearchNet）+ 序列分类（MNLI、XNLI、PAWS-X、MathShepherd、CodeComplexity）+ token 分类（PAN-X、POS）+ 序列回归（Seahorse）
- **嵌入评测**：MTEB v2（英文 + 多语言）、MIEB lite（图像）、MAEB beta（音频）
- **组合能力评测**：安全（Beaver/Safe/Aegis）、视觉蕴含（e-SNLI-VE）、音频理解（BoolQ-Audio）

## 实验结果
### 文本基准：重新定义开源 Pareto 前沿
在全参数微调的增强 XTREME 基准上，所有 BidirLM 变体均建立新的性能前沿。BidirLM-270M 以少 10% 的参数匹配 mmBERT-base；BidirLM-0.6B 超越 EuroBERT-610m 超过 1 分。

在通用嵌入基准 MTEB 上，同样的 BidirLM 变体也在四个尺寸配置中的三个上推进了开源 Pareto 前沿，且**仅使用经典对比训练，无需知识蒸馏或多轮平均**。

<figure class="figure">
  <img src="assets/papers/bidir-lm/final_results.png" alt="多语言模型按尺寸的性能 Pareto 前沿" loading="lazy" />
  <figcaption>图 5：XTREME + MTEB 上各尺寸多语言模型的平均分数。虚线为开源性能 Pareto 前沿，BidirLM 在多个尺寸上位于前沿之上。</figcaption>
</figure>

### 全模态基准：以小博大
BidirLM-Omni-2.5B 在所有模态上超越最新的 Nemotron-Omni-3B（4.8B 参数）：文本 +17 分、图像 +5 分，参数量仅为其一半。在 MIEB 上排名第一，在 MAEB 上排名第三，超越了数倍大小的双模态架构。

<figure class="figure">
  <img src="assets/papers/bidir-lm/final_results_omni.png" alt="全模态嵌入模型按尺寸的性能" loading="lazy" />
  <figcaption>图 6：MTEB Multilingual V2 + MIEB + MAEB 上各尺寸嵌入模型的平均分数。BidirLM-Omni-2.5B 以 2.5B 参数建立了新的全模态 Pareto 前沿。</figcaption>
</figure>

### 硬件资源
论文使用了 ADASTRA、Jean Zay 和 ROMEO 超算集群。构建 BidirLM-Omni 的合并 + 对比训练仅需 **250 GPU 小时（MI250X）**，证明了增量组装全模态架构的可行性。

## Ablation：哪些设计最关键？
### MNTP 阶段不可省略
近期对比学习方法普遍跳过 masking 阶段。论文的受控消融证明这是错误的：没有 MNTP，双向注意力的潜力无法释放，微调任务性能显著下降。

### 对比学习和 masking 互补而非替代
对比学习驱动通用嵌入质量，masking 驱动微调性能。顺序执行两者才能兼顾。

### 权重合并是无数据的遗忘缓解利器
适配模型与基座在权重空间距离很近，线性合并即可恢复大量原始知识，无需任何额外数据。

### 多域混合只需 20%-30%
不需要精确复制原始预训练分布，少量多域数据即可防止遗忘 plateau。

### 合并可跨模态工作
即使两个模型没有共享模态（如纯 ASR 模型 + 文本编码器），权重合并仍能成功转移能力，说明合并操作利用的是底层表征结构的兼容性。

## 我的理解与启发
这篇论文最有价值的洞察是：**因果 LLM 生态本身就是编码器的最佳资源池**。

1. **不要从零训练编码器**：因果模型已经学到了丰富的语言/多模态表征，适配比从头训练高效得多。
2. **Masking 不是过时的技术**：它在双向适配器中扮演的角色不同于 BERT 时代的 MLM，而是让模型学会在双向注意力下有效利用上下文。
3. **权重合并是被低估的工具**：它不仅用于 ensemble，更是知识迁移、遗忘缓解、模态组合的统一机制。
4. **模块化组合优于端到端重训**：当新的专用因果模型出现时，只需合并即可扩展编码器能力，无需重训整个管线。这对快速迭代的生产系统非常有吸引力。
5. **开源可复现性是核心竞争力**：论文全程使用开源数据和模型，消除了对私有预训练数据的依赖，使结果真正可验证。

## 局限与待观察点
- **对比训练阶段的消融不够深入**：论文承认对比训练已有大量先前研究，但未系统研究数据组成、hard-negative 策略和全模态场景下的缩放行为。
- **仅验证了 Transformer 架构**：状态空间模型等非 Transformer 因果架构是否适用尚未验证。
- **合并比例的调优仍依赖经验**：50% 是最优点，但不同模型对、不同任务可能需要不同比例，缺乏自动化方法。
- **全模态编码器的生成能力未评估**：BidirLM-Omni 是纯编码器，无法生成文本/图像/音频，限制了某些应用场景。
- **评测仍以离线基准为主**：真实 RAG 系统中的检索延迟、索引更新、多模态查询路由等工程指标未涉及。

## 结论
BidirLM 给出了一套将因果 LLM 转换为全模态双向编码器的完整开源配方：**MNTP 解锁双向潜力 → 对比学习提升嵌入质量 → 权重合并 + 多域混合缓解遗忘 → 合并专用模型扩展模态**。它的价值不仅是发布了几个强编码器，更是证明了因果 LLM 生态中积累的海量专业化知识可以被高效复用于表征学习，为未来"按需组装"多模态编码器提供了可行路径。