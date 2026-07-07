---
id: "lco-emb-language-centric-omnimodal-embedding"
title: "LCO-Emb：用语言中心的轻量对比学习激活 MLLM 的全模态表征能力"
category: "全模态模型"
date: "2026-07-07"
order: 6
readTime: "18 min"
tags: ["Multimodal Embedding", "MLLM", "Contrastive Learning", "LoRA", "GRSL", "Representation Learning", "NeurIPS 2025"]
summary: "LCO-Emb 发现 MLLM 在生成式预训练中已实现隐式跨模态对齐，只需轻量级文本对比学习即可激活全模态表征能力。提出 Language-Centric Omnimodal Embedding 框架，仅用 ~370k 训练数据即在 MIEB 上超越使用 8M 数据的 GME。进一步发现 Generation-Representation Scaling Law (GRSL)，证明 MLLM 的生成能力决定了其表征质量的上界，并通过 PAC-Bayesian 理论给出形式化解释。"
hero: "assets/papers/lco-emb/fig3.png"
---

## 一句话总结
LCO-Emb 揭示了一个关键洞察：**MLLM 在生成式预训练中已经隐式完成了跨模态对齐**，对比学习（CL）只需作为轻量级的"激活"步骤，而非从头对齐。基于此提出的语言中心全模态嵌入框架，仅用文本数据微调就能提升所有模态的表征质量，并发现了生成能力与表征质量之间的缩放定律（GRSL）。

<div class="metric-grid">
  <div class="metric"><strong>MIEB SOTA</strong><span>仅用 ~370k 数据超越 8M 数据的 GME</span></div>
  <div class="metric"><strong>+21.69</strong><span>Text-only LCO-Emb 相对 E5-V 的平均提升</span></div>
  <div class="metric"><strong>~4.7h</strong><span>3B LoRA 文本对比学习训练时间（H20）</span></div>
  <div class="metric"><strong>GRSL</strong><span>生成能力 ↔ 表征质量的正相关缩放定律</span></div>
</div>

## 论文信息
- **标题**：Scaling Language-Centric Omnimodal Representation Learning
- **作者**：Chenghao Xiao, Hou Pong Chan, Hao Zhang, Weiwen Xu, Mahani Aljunied, Yu Rong
- **机构**：DAMO Academy, Alibaba Group
- **会议**：NeurIPS 2025；arXiv 2510.11693, Oct 2025
- **链接**：[arXiv](https://arxiv.org/abs/2510.11693) · [PDF](https://arxiv.org/pdf/2510.11693) · [GitHub](https://github.com/LCO-Embedding/LCO-Embedding)

## 背景：为什么 MLLM-based Embedding 比 CLIP-style 更强？

传统 CLIP-style 模型依赖大规模跨模态对比学习来对齐不同模态的表征空间，但在复杂任务（多语言检索、文档理解、交错多模态编码）上性能趋于饱和。近期工作转向用 MLLM 作为 backbone + CL 微调，取得了更好效果，但**背后的原因未被深入探究**。

本文的核心问题是：**MLLM-based embedding 的优势到底来自哪里？**

<figure class="figure">
  <img src="assets/papers/lco-emb/fig1.png" alt="各向异性分析" loading="lazy" />
  <figcaption>图 1：Qwen2.5-Omni-3B 各模态嵌入的各向异性估计。原始模型所有模态都呈现表征退化（高各向异性）。仅用文本对比学习后，图像、音频、视频嵌入也变得更各向同性，表明 MLLM 内部存在隐式的语言中心跨模态对齐。</figcaption>
</figure>

## 方法：核心设计与洞察

### 1. 关键发现：MLLM 预训练中的隐式跨模态对齐

通过两个几何性质分析证实了这一假设：

**各向异性（Anisotropy）分析**：用纯文本 NLI 数据做对比学习后，不仅文本嵌入变得更均匀分散，**图像、音频、视频嵌入也同步改善**。这说明 MLLM 在生成式预训练中已将多模态信号整合到共享表征空间。

**核级相似度（Kernel-level Similarity）分析**：用 mutual kNN 度量视觉-语言内核对齐度。结果显示：(1) 文本 CL 后跨模态对齐度提升；(2) 7B 模型的对齐度始终优于 3B，参数空间越大，预训练中捕获的隐式跨模态关系越强。

<figure class="figure">
  <img src="assets/papers/lco-emb/fig2.png" alt="核级相似度分析" loading="lazy" />
  <figcaption>图 2：逐层视觉-语言内核对齐度。文本对比学习后对齐度提升，且 7B 模型始终优于 3B，表明更大的参数空间能更好地捕获隐式跨模态关系。</figcaption>
</figure>

### 2. LCO-Emb 框架：语言中心的轻量级精炼

基于上述洞察，LCO-Emb 的设计原则是：**保留预训练建立的隐式对齐，仅用最小扰动将嵌入空间投影到相似度匹配子空间**。

**架构**：标准 MLLM 结构（模态编码器 + Projector + LLM Decoder）。Text-only 变体仅微调 LLM Decoder，冻结编码器和 Projector。

**LoRA 的关键作用**：不同于常规的参数高效微调目的，LoRA 在此的核心价值是**最小化对预训练权重的扰动**：
- 保持模型的生成能力
- 维护 LLM Decoder embedding 层中的隐式跨模态对齐（该层不受 LoRA 影响）

**训练数据**：
- **Text-only**：all-NLI（~276k triplets）或 Scale-1M（1M 多语言句子对）
- **Multimodal**：在 all-NLI 基础上加 ~94k 合成多模态对，总计 ~370k

<figure class="figure">
  <img src="assets/papers/lco-emb/fig3.png" alt="LCO-Emb 框架总览" loading="lazy" />
  <figcaption>图 3：LCO-Emb 的核心思想。对比学习前，MLLM 各模态表征呈各向异性坍缩。文本 CL 使文本表征分散，且这一效果泛化到非文本模态——无需直接监督即可缓解各向异性。</figcaption>
</figure>

## 数据与评测

**Backbone 模型**：LLaVA-Next-8B、Qwen2.5-VL（3B/7B）、Qwen2.5-Omni（3B/7B）

**评测基准**：
- **MIEB-Lite**（51 tasks）：覆盖 Linear Probing、Retrieval、Zero-shot Classification、Compositionality、Vision-centric QA、Document Understanding、Clustering、Visual STS 八大类
- **Audio-text**：AudioCaps、Clotho（Recall@1）
- **Video-text**：MSR-VTT、ActivityNet（Recall@1）

**训练设置**：AdamW + cosine schedule，peak lr=4e-4，batch size=768，2 epochs。LoRA rank=64, α=16（text-only）/ α=128（multimodal）。

## 实验结果

### MIEB-Lite 主实验

<figure class="figure">
  <img src="assets/papers/lco-emb/fig4.png" alt="MIEB 性能对比" loading="lazy" />
  <figcaption>图 4：LCO-Emb 与 SOTA 开源及商业嵌入模型的性能对比。LCO-Emb-VL 和 LCO-Emb-Omni 分别基于 Qwen2.5-VL 和 Qwen2.5-Omni，T/M 分别表示 text-only 和 multimodal 变体。</figcaption>
</figure>

<div class="table-wrap">
<table>
<thead>
<tr><th>Model</th><th>Data Size</th><th>Mean (51)</th><th>Mean (en, 47)</th></tr>
</thead>
<tbody>
<tr><td>CLIP-ViT-bigG</td><td>-</td><td>51.3</td><td>56.5</td></tr>
<tr><td>SigLIP-so400m</td><td>9B</td><td>53.5</td><td>57.3</td></tr>
<tr><td>E5-V</td><td>276k</td><td>51.9</td><td>51.8</td></tr>
<tr><td>Voyage Multimodal 3</td><td>-</td><td>58.1</td><td>57.7</td></tr>
<tr><td>mmE5 (11B)</td><td>2.1M</td><td>61.8</td><td>57.7</td></tr>
<tr><td>GME (7B)</td><td>8.0M</td><td>64.5</td><td>63.4</td></tr>
<tr><td><strong>LCO-Emb-VL (7B) T</strong></td><td><strong>276k</strong></td><td><strong>60.4</strong></td><td><strong>57.5</strong></td></tr>
<tr><td><strong>LCO-Emb-VL (7B) M</strong></td><td><strong>370k</strong></td><td><strong>67.6</strong></td><td><strong>66.2</strong></td></tr>
<tr><td><strong>LCO-Emb-Omni (7B) M</strong></td><td><strong>370k</strong></td><td><strong>68.8</strong></td><td><strong>67.6</strong></td></tr>
</tbody>
</table>
</div>

**关键发现**：
- **数据效率极高**：LCO-Emb 仅用 ~370k 数据（GME 的 1/21）即超越 GME，成为 MIEB 新 SOTA
- **Text-only 已超商业模型**：纯文本训练的 LCO-Emb 超越 Voyage Multimodal 3
- **Omni > VL**：Qwen2.5-Omni backbone 优于 Qwen2.5-VL，说明更多模态的预训练带来更强的隐式对齐

### Text-only 变体的详细优势

<figure class="figure">
  <img src="assets/papers/lco-emb/fig5.png" alt="Text-only 变体消融对比" loading="lazy" />
  <figcaption>图 5：Text-only LCO-Emb 与 E5-V、Voyage Multimodal 3 在 MIEB-Sub18 上的消融对比。即使仅用文本数据训练，LCO-Emb 仍全面超越先进开源和商业模型。</figcaption>
</figure>

Text-only LCO-Emb 相对 E5-V 和 Voyage-M3 分别平均提升 **21.69** 和 **13.00** 分，在 Linear Probing（+21.02）、Cross-lingual V-STS（+10.26）、Multilingual Retrieval（+15.35）上提升尤为显著。

### 训练策略对比

<div class="table-wrap">
<table>
<thead>
<tr><th>Strategy</th><th>GPU Hours</th><th>Avg (3B)</th><th>Avg (7B)</th></tr>
</thead>
<tbody>
<tr><td>CLIP-style CL (multimodal)</td><td>~453 / ~550</td><td>53.38</td><td>50.02</td></tr>
<tr><td>Linear Projection (text-only)</td><td>~4.5 / ~8.8</td><td>52.69</td><td>56.22</td></tr>
<tr><td>Full Finetune (text-only)</td><td>~8.5 / ~17.3</td><td>62.10</td><td>66.49</td></tr>
<tr><td><strong>LoRA (text-only)</strong></td><td><strong>~4.7 / ~9.3</strong></td><td><strong>66.28</strong></td><td><strong>71.98</strong></td></tr>
</tbody>
</table>
</div>

**LoRA 全面胜出**：比 CLIP-style CL 快 ~50×，比 Full Finetune 效果更好。原因是 Full Finetune 的对比损失偏离预训练目标，破坏了预训练建立的跨模态对齐；而 LoRA 以最小扰动保留了这些对齐。

## Ablation：哪些设计最关键？

### 数据集互补性

| 数据集 | 优势任务 | 原因 |
|--------|---------|------|
| all-NLI | V-STS、Doc Understanding | NLI 监督锐化了语义相似度感知 |
| Scale-1M | Linear Probing、Multilingual Retrieval | 丰富描述模拟 image-caption 预训练 |
| Model Soup 融合 | 整体最优 | 有效整合两个 checkpoint 的互补优势 |

### GRSL 验证

<figure class="figure">
  <img src="assets/papers/lco-emb/fig6.png" alt="GRSL 缩放关系" loading="lazy" />
  <figcaption>图 6：生成基准性能（X轴）与对比学习后表征基准性能（Y轴）的缩放关系。在 OCR 图像-文本、视频-文本、音频-文本三类任务上均呈现一致的正相关，验证了 GRSL。</figcaption>
</figure>

在不同 backbone 和模态对上，**CL 前的生成能力与 CL 后的表征能力呈一致正相关**。这意味著：提升 MLLM 的生成能力（如继续预训练、SFT）是提升其表征潜力的有效路径。

### SeaDoc 低资源验证

引入 SeaDoc（东南亚低资源语言视觉文档检索），通过在低资源语言上做 OCR-intensive 持续预训练后再做文本 CL，检索性能进一步提升，实证了 GRSL 的实际指导价值。

## 我的理解与启发

1. **"CL 是对齐"这个假设需要修正**。对于 MLLM，CL 更像是一个"激活"或"投影"步骤，而非从头建立跨模态对齐。这改变了我们对 MLLM embedding 训练的认知框架。

2. **LoRA 的价值不只是参数效率**。在 LCO-Emb 中，LoRA 的核心作用是**保持预训练权重不变**，从而保留隐式对齐。这提示我们：在 MLLM 上做 embedding 微调时，应该优先考虑最小扰动策略。

3. **GRSL 提供了清晰的 scaling 方向**。与其在 CL 阶段堆数据，不如先提升 MLLM 的生成能力。这对工程实践有直接指导意义：如果要做 embedding 模型，先在生成任务上把 base model 训好。

4. **Text-only 训练的泛化能力令人惊讶**。仅用文本 NLI 数据就能提升图像、音频、视频的表征质量，这说明 MLLM 的跨模态对齐确实深植于语言解码器中。Scale-1M 的多语言描述数据进一步模拟了 image-caption 预训练的效果。

5. **PAC-Bayesian 理论解释优雅**。Theorem 1 将生成质量 \(I_P(X;Y)\) 直接嵌入表征性能的 upper bound，形式化了"更好的生成先验 → 更好的表征"这一直觉。KL(Q||P) 项也解释了为什么 LoRA（小 KL）优于 Full Finetune（大 KL）。

## 局限与待观察点

- **依赖 MLLM 的预训练质量**：如果 base MLLM 本身跨模态对齐差（如早期模型），LCO-Emb 的效果会受限。GRSL 既是优势也是约束。
- **Text-only 的上限**：虽然 text-only 已很强，但加入 ~94k 多模态数据仍有显著提升（+7 分左右），说明纯文本激活无法完全替代显式多模态对齐。
- **评测覆盖有限**：主要在 MIEB 上评测，缺少与传统 CLIP-style 模型在 zero-shot ImageNet 等经典 benchmark 上的对比。
- **LoRA 超参敏感**：rank 和 α 的选择对结果有影响（Appendix D），不同 backbone 可能需要不同的最优配置。
- **SeaDoc 的特殊性**：低资源语言的验证场景较窄，GRSL 在更广泛场景下的普适性仍需验证。

## 结论

LCO-Emb 重新定义了 MLLM-based embedding 的训练范式：**不是用 CL 从头对齐模态，而是用轻量 CL 激活预训练中已有的隐式对齐**。这一洞察带来了极高的数据效率和训练效率，同时通过 GRSL 揭示了生成能力与表征质量之间的根本联系。对于多模态表征学习的研究和实践，这篇论文提供了一个清晰的方向：先把生成做好，表征自然跟上。
