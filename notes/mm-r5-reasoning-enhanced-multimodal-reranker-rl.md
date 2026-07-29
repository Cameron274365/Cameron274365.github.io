---
id: "mm-r5-reasoning-enhanced-multimodal-reranker-rl"
title: "MM-R5：用推理链 + 强化学习做多模态文档重排"
category: "多模态 RAG"
date: "2026-07-29"
order: 1
readTime: "14 min"
tags: ["Reranker", "Multimodal RAG", "Reinforcement Learning", "GRPO", "Chain-of-Thought", "Document Retrieval", "MMDocIR"]
summary: "MM-R5 把「显式推理链」引入多模态文档重排：基于 Qwen2.5-VL-7B，先用 7,200 条带标注推理路径的数据做 SFT 打好指令遵循与推理基础，再用 GRPO 做任务特定强化学习。奖励设计有两个亮点——结果奖励用排名的三次方倒数（比 NDCG 的对数折扣更强调头部命中），格式奖励用「结构合法性 × 长度准确性 × 索引合法性」的乘性组合做硬约束。在 MMDocIR 上多数指标达 SoTA，7B 模型超过 Qwen2.5-VL-32B-cot；作为即插即用组件接在 5 种检索器之上均一致提升，对弱检索器（CLIP +26%）增益尤大。"
---

## 一句话总结

多模态文档检索的 reranker 研究还很薄弱，且大多是「黑箱打分」——没有显式推理过程，难以分析和优化。MM-R5 的做法是：**把重排变成一个「先显式推理、再输出排名」的生成任务，用两阶段训练（SFT 打推理基础 + GRPO 强化学习精修）把一个 7B 模型训成超过 32B 通用模型的专用重排器**。它一次吃下所有候选页、逐张分析图像，输出推理链和相关性排名，既提性能也带来可解释性。

<div class="metric-grid">
  <div class="metric"><strong>0.6951</strong><span>MMDocIR macro Recall@1（ColQwen 一阶段 0.6481）</span></div>
  <div class="metric"><strong>7B > 32B</strong><span>超过 Qwen2.5-VL-32B-cot（0.6768）</span></div>
  <div class="metric"><strong>+26.08%</strong><span>接在弱检索器 CLIP 上的 macro R@1 相对增益</span></div>
  <div class="metric"><strong>5/5</strong><span>五种检索器上全部一致提升（即插即用）</span></div>
</div>

## 论文信息

- **标题**：MM-R5: MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval
- **作者**：Mingjun Xu, Jinhan Dong, Jue Hou（同等贡献）, Zehui Wang, Sihang Li, Zhifeng Gao, Renxin Zhong, Hengxing Cai
- **版本**：arXiv:2506.12364v2，2025-06-22 修订（v1 为 2025-06-14）
- **链接**：[arXiv](https://arxiv.org/abs/2506.12364) · [PDF](https://arxiv.org/pdf/2506.12364)

## 背景：多模态重排的两个空白

多模态文档检索要从含图像、表格、复杂版式的文档里找信息。检索侧已有不少进展——DSE 直接把文档截图编码成稠密向量（免 OCR），ColPali 用多向量 + late interaction 进一步提升。但作者指出两个空白：

- **重排器探索不足**：现有方法主要盯检索性能，多模态 reranker 的训练策略和效果都有很大提升空间。
- **缺乏显式推理**：没有推理过程，就难以分析模型为什么这样排、也难以进一步优化。

同时，多模态推理（R1-OneVision、Vision-R1、MM-Eureka、Visual-RFT 等）已经在 VQA 到数学推理的广泛任务上证明有效，且 R1-V 让 3B 模型在计数任务上超过 72B。这自然引出本文的问题：**能不能把推理引入多模态重排？**

<figure class="figure">
  <img src="assets/papers/mm-r5/x1_workflow.webp" alt="MM-R5 工作流" loading="lazy" />
  <figcaption>图 1：MM-R5 的工作流。它一次性接收所有候选页、分析全部图像，然后输出推理过程和相关性排名。</figcaption>
</figure>

## 任务形式化

给定查询 \(q\) 和检索到的多模态候选集 \(\mathcal{D}=\{d_1,\dots,d_n\}\)（每个 \(d_i\) 含视觉与文本信息），reranker 输出一个索引列表 \(\hat{\mathcal{I}}\) 表示对 \(\mathcal{D}\) 的排列，索引越小代表预测相关性越高。评测时把相关候选的黄金索引集记为 \(\mathcal{G}\)。

注意这里是**一次性把所有候选喂进去输出完整排列**的形式（listwise 风格），而非逐条打分。

## 方法：两阶段训练

<figure class="figure">
  <img src="assets/papers/mm-r5/x2_pipeline.webp" alt="MM-R5 两阶段训练流水线" loading="lazy" />
  <figcaption>图 2：两阶段训练流水线总览。第一阶段 SFT 侧重构造高质量推理数据，第二阶段 RL 侧重奖励函数设计。</figcaption>
</figure>

两阶段的侧重点不同：**SFT 阶段重点在「构造高质量推理数据」，RL 阶段重点在「奖励函数设计」**。

### 阶段一：监督微调（SFT）

目标是提升指令遵循能力、并引导模型生成**完整且高质量的推理链**。作者设计了一套数据构造策略，产出 **7,200 条高质量训练实例**（带良好标注的推理路径）。这一步对模型的指令遵循和推理能力起到关键作用。

> 注：论文正文对数据构造管线的具体步骤描述较简略（HTML 版该小节内容缺失），只给出了最终规模 7,200 条与其作用。

### 阶段二：强化学习（GRPO）

与 SFT 不同，RL 阶段的数据**直接从原始训练语料采样**，不需要额外构造。

**① 分辨率均衡采样（Resolution-Balanced Sampling）**

为缓解图像分辨率差异带来的偏差，作者把原始训练集按图像尺寸划成 **10 个子集**（每个子集内分辨率相近），再从各子集**无放回按比例采样**，合计 **3,000 条**构成最终训练集。因为子集内分辨率一致，这种比例采样保证**各分辨率区间对训练的贡献均衡**，在不增加数据量的前提下降低了分辨率分布不均导致的梯度估计偏差，让 RL 更稳定。

**② 奖励建模：结果奖励 + 格式奖励**

**结果奖励**用排名的**三次方倒数**加权：

\[
R_{result}=\frac{\sum_{j=1}^{|\hat{\mathcal{I}}|} s_j / j^3}{\sum_{j=1}^{|\mathcal{G}|} 1 / j^3},\quad s_j=\begin{cases}1,& \hat{i}_j \in \mathcal{G}\\ 0,& \hat{i}_j \notin \mathcal{G}\end{cases}
\]

分子把所有正确命中项的排名倒立方求和，**极大加权头部位置（尤其第一名）**；分母是「所有真值都排进前 K」时的最大可能奖励，用于归一化到 \([0,1]\) 并抵消不同样本真值数量的差异。作者明确对比：**相比标准 NDCG 的对数折扣，用排名的三次方能更强地强调头部预测**。

**格式奖励**由三个独立可微分量**乘性**组合：

\[
R_{format}=R_{valid}\times R_{len}\times R_{range}
\]

- **结构合法性 \(R_{valid}\)**：输出必须被 `<think>…</think>` 与 `<answer>…</answer>` 正确包裹，否则为 0——这是**硬约束**，任何格式错误都让整个格式奖励归零（不管内容对不对）。
- **长度准确性 \(R_{len}=1-\frac{|\,|\hat{\mathcal{I}}|-n\,|}{n}\)**：鼓励输出长度与参考一致。
- **索引合法性 \(R_{range}\)**：预测索引落在 \([1,n]\) 内的比例，惩罚越界索引。

这种设计的好处：\(R_{valid}\) 用硬停惩罚强制结构；\(R_{len}\)/\(R_{range}\) 在接近正确时提供**平滑梯度信号**；三个因子**可独立调节**，便于对不同错误类型做细粒度控制。

### 训练配置

- **基座**：Qwen2.5-VL-7B。SFT 用 [swift](https://github.com/modelscope/ms-swift) 框架，RL 用 [VLM-R1](https://github.com/om-ai-lab/VLM-R1) 框架跑 GRPO，全程用 LoRA 提效。
- **SFT**：1 epoch，lr 1e-4，batch 1，梯度累积 4，LoRA r=8 / alpha=32。
- **RL**：rollout 4，1 epoch，lr 1e-5，batch 1，梯度累积 2，LoRA r=64 / alpha=128。
- **硬件**：4× A100 80GB。

## 数据与评测

- **基准**：MMDocIR（跨多领域的长文档多模态检索基准）。
- **指标**：Recall@k，同时报 **micro**（所有样本平均）和 **macro**（先在各子集内平均、再跨子集平均，对分布差异更均衡）。
- **公平性控制**：重排时**固定检索骨干产出 top-10 候选页**，让所有重排策略在相同输入下比较。
- **对比对象**：检索基线（CLIP、E5-V、DSE、GME、ColQwen）；重排基线（RAG-VL，以及 Qwen2.5-VL-7B/32B、Gemma3-12B 通用 VLM，含 cot 变体）。

## 实验结果

### 主结果（候选来自 ColQwen top-10）

<table>
  <thead>
    <tr><th>类别</th><th>方法</th><th>Macro R@1</th><th>Macro R@3</th><th>Micro R@1</th><th>Micro R@3</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="2">仅检索</td><td>GME</td><td>0.5400</td><td>0.7603</td><td>0.5421</td><td>0.7603</td></tr>
    <tr><td>ColQwen（前 SoTA）</td><td>0.6481</td><td>0.8331</td><td>0.6354</td><td>0.8213</td></tr>
    <tr><td rowspan="6">重排</td><td>RagVL</td><td>0.3814</td><td>0.6462</td><td>0.3411</td><td>0.6206</td></tr>
    <tr><td>Gemma3-12B-cot</td><td>0.5403</td><td>0.7729</td><td>0.5166</td><td>0.7536</td></tr>
    <tr><td>Qwen2.5-VL-32B-cot</td><td>0.6768</td><td>0.8500</td><td>0.6609</td><td>0.8448</td></tr>
    <tr><td>Qwen2.5-VL-7B-sft（仅 SFT）</td><td>0.6673</td><td>0.8475</td><td>0.6498</td><td>0.8366</td></tr>
    <tr><td>Qwen2.5-VL-7B-rl（仅 RL）</td><td>0.6586</td><td>0.8370</td><td>0.6454</td><td>0.8309</td></tr>
    <tr><td><strong>MM-R5（完整两阶段）</strong></td><td><strong>0.6951</strong></td><td><strong>0.8520</strong></td><td><strong>0.6759</strong></td><td><strong>0.8401</strong></td></tr>
  </tbody>
</table>

- **超越所有仅检索方法**：相比前 SoTA 检索器 ColQwen，macro R@1 从 0.6481 → 0.6951、micro R@1 从 0.6354 → 0.6759。
- **7B 打赢 32B**：MM-R5 基于 Qwen2.5-VL-7B，却超过 Qwen2.5-VL-32B-cot（0.6768/0.6609）；相比 Gemma3-12B-cot 更是拉开 **+0.1548 / +0.1593** 的巨大差距。
- **RagVL 严重掉点**（0.3814，远低于不重排的 0.6481），说明没有针对性训练的多模态重排会**把排序搞坏**。

### 跨检索器泛化（即插即用）

<table>
  <thead>
    <tr><th>检索器</th><th>Macro R@1（原始）</th><th>+ MM-R5</th><th>相对增益</th></tr>
  </thead>
  <tbody>
    <tr><td>CLIP</td><td>0.3334</td><td>0.5942</td><td><strong>+26.08%</strong></td></tr>
    <tr><td>E5-V</td><td>0.4201</td><td>0.6236</td><td>+20.35%</td></tr>
    <tr><td>DSE</td><td>0.5109</td><td>0.6487</td><td>+13.78%</td></tr>
    <tr><td>GME</td><td>0.5400</td><td>0.6612</td><td>+12.12%</td></tr>
    <tr><td>ColQwen</td><td>0.6481</td><td>0.6951</td><td>+4.70%</td></tr>
  </tbody>
</table>

五种检索器上 macro/micro、R@1/3/5 **全部一致提升**，证明 MM-R5 是稳健的即插即用组件。**弱检索器增益尤其显著**（CLIP +26%、E5-V +20%），说明它不只精修强基线，也能大幅拉起简单检索器。

## Ablation：哪些设计最关键？

**① CoT 推理确实有效**：给通用 VLM 加 CoT 就能涨——Qwen2.5-VL-7B macro/micro R@1 分别 +2.49/+3.24，32B 分别 +3.46/+3.00。本文引入的推理范式（**先显式评估每张图与查询的相关性、再给最终排名**）不仅提性能，也提升了推理过程的可解释性。

**② SFT 与 RL 互补，缺一不可**：

| 变体 | Macro R@1 | 差距 |
|---|---|---|
| MM-R5（完整） | 0.6951 | — |
| 仅 SFT | 0.6673 | −2.78 |
| 仅 RL | 0.6586 | −3.65 |
| 基线 Qwen2.5-VL-7B-cot | 0.6479 | −4.72 |

两个单阶段变体都明显优于基线，但都不及完整模型——**证实 SFT 与 RL 是互补的**，两阶段流水线的必要性与协同性成立。

## 我的理解与启发

- **「显式推理 + 一次看全部候选」的组合值得注意**。MM-R5 让模型一次吃下所有候选、先推理再排名，这与本站 LightOn-rerank 笔记里的 listwise 发现不谋而合——候选之间能互相比较是重排质量的来源。MM-R5 额外加了显式推理链，同时换来可解释性。
- **奖励函数的「排名三次方倒数」是个务实设计**。重排场景下用户几乎只看头部，NDCG 的对数折扣对头部区分不够狠；用 \(1/j^3\) 直接把梯度压向第一名，是把「业务指标」写进奖励的好例子。
- **乘性格式奖励 = 硬约束 + 平滑梯度的巧妙折中**。结构不合法直接归零（硬停），但长度和索引范围给平滑信号，避免「全对或全错」的稀疏奖励。这个模式在训练结构化输出的 agent 时很值得借鉴。
- **分辨率均衡采样提示了多模态 RL 的一个隐蔽坑**：图像尺寸分布不均会导致梯度估计偏差。分箱按比例采样是低成本的修法，比单纯加数据更划算。
- **专用小模型 + 好训练策略 > 通用大模型**：7B 超过 32B，再次印证在垂直任务上「数据与奖励设计」的杠杆比参数量更大。
- **弱检索器上增益最大**，说明重排器在工程上可以作为「补救层」——检索侧不够强时，重排能补回相当多召回质量。

## 局限与待观察点

- **SFT 数据构造细节披露不足**：7,200 条推理路径如何生成、如何过滤质量，正文描述简略，复现有难度。
- **只在 MMDocIR 上评测**：单一基准，跨基准泛化（如 ViDoRe）未验证。
- **重排深度固定 top-10**：更深候选池（如 top-100）下的表现与成本未讨论——而这恰是生产部署的关键权衡。
- **推理链带来的延迟未量化**：一次生成完整推理 + 排列的开销、与逐条打分方案的延迟对比都没有报告。
- **推理链质量本身未独立评估**：可解释性主要靠 case study 定性展示，缺少推理正确性的量化指标。

## 结论

MM-R5 把显式推理引入多模态文档重排，用「SFT 构造高质量推理数据 + GRPO 任务特定奖励」的两阶段流水线，把 Qwen2.5-VL-7B 训成在 MMDocIR 上多数指标达 SoTA 的重排器，超过 32B 级通用 VLM，并作为即插即用组件在五种检索器上一致提升（相比最佳仅检索方法 R@1 提升 4% 以上）。它最有借鉴价值的是奖励设计——**排名三次方倒数强调头部、乘性格式奖励兼顾硬约束与平滑梯度**，以及分辨率均衡采样这个多模态 RL 的实用细节。
