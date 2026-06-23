---
id: "m3kg-rag-multi-hop-multimodal-kg-rag"
title: "M3KG-RAG：多跳多模态知识图谱增强的检索增强生成"
category: "多模态 RAG"
date: "2026-06-23"
order: 10
readTime: "18 min"
tags: ["Multimodal RAG","Knowledge Graph","Multi-hop Retrieval","GRASP","Audio-Visual Reasoning","Multi-Agent","MLLM"]
summary: "M3KG-RAG 构建多跳多模态知识图谱（M3KG），通过模态级检索（Modality-Wise Retrieval）消除跨模态 embedding gap，再经 GRASP（Grounded Retrieval And Selective Pruning）利用视觉/音频 grounding 模型和 LLM filter 逐级剪枝，仅保留与 query 相关且对回答有用的子图，在 Audio/Video/Audio-Visual QA 三个 benchmark 上全面超越现有 MMKG-RAG 方法。"
hero: "assets/papers/m3kg-rag/fig2_pipeline.webp"
---

## 一句话总结
M3KG-RAG 的核心思路是：**先用多 agent 管线从多模态语料中构建多跳知识图谱（M3KG），然后在检索阶段按 query 的模态做同模态 nearest-neighbor search 消除 modality gap，最后用 GRASP 通过视觉/音频 grounding + LLM filter 三重剪枝，只留下 query-aligned 且 answer-supporting 的子图送入 MLLM**。在 GPT-4o 上也能带来显著提升，说明即使模型已有丰富内置知识，外部多跳结构化证据仍然有不可替代的价值。

<div class="metric-grid">
  <div class="metric"><strong>60.77 M.J.</strong><span>Qwen2.5-Omni + M3KG-RAG 在 AudioCaps-QA 上的得分，比 VAT-KG 高 9.5 分</span></div>
  <div class="metric"><strong>44.67 M.J.</strong><span>在 VALOR Audio-Visual QA 上的得分，比最强 baseline VAT-KG 高 9.2 分</span></div>
  <div class="metric"><strong>~7s/query</strong><span>完整 GRASP 管线的平均推理延迟（含 GDino + TAG + LLM filter）</span></div>
  <div class="metric"><strong>~40 GB VRAM</strong><span>完整系统 GPU 显存占用（H100 单卡可运行）</span></div>
</div>

## 论文信息
- **标题**：M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation
- **作者**：Hyeongcheol Park, Jiyoung Seo, Jaewon Mun, Hogun Park, Wonmin Byeon, Sung June Kim, Hyeonsoo Im, JeungSub Lee, Sangpil Kim
- **机构**：Korea University, Sungkyunkwan University, NVIDIA Research, Hanwha Systems
- **版本**：arXiv 2512.20136 (Dec 2025, v3 Apr 2026)
- **链接**：[arXiv](https://arxiv.org/abs/2512.20136) · [PDF](https://arxiv.org/pdf/2512.20136) · [项目主页](https://kuai-lab.github.io/cvpr2026m3kgrag/)

## 背景：为什么现有多模态 RAG 在音视频场景下效果有限？

多模态 RAG 将 MLLM 与外部知识源连接，但在 **audio-visual** 场景下面临两个核心瓶颈：

1. **模态覆盖与多跳连通性不足**：现有 MMKG（如 VTKG、M2ConceptBase）主要覆盖 image-text，对 audio-visual 的时序和因果推理支持有限。VAT-KG 虽然纳入了 audio-visual 流，但构建的是 **concept-level 单跳图**，难以捕捉跨模态的时序依赖关系。

2. **共享 embedding 空间的 modality gap**：多模态编码器（如 InternVL2、CLAP）的共享 embedding 空间中，不同模态之间的距离校准不一致。用 video query 直接搜索 text embedding 索引，或用 audio query 搜索 visual items，往往会检索到 off-topic 的邻居。**similarity-only 检索无法过滤掉与问题无关的冗余知识**。

<figure class="figure">
  <img src="assets/papers/m3kg-rag/fig1_comparison.webp" alt="M3KG-RAG 与现有方法对比" loading="lazy" />
  <figcaption>图 1：(a) 共享 embedding 空间搜索与 audio-visual query 不对齐，检索到错误证据；(b) 单跳图提供 noisy、fragmentary 的上下文，无法改善回答；(c) M3KG-RAG 通过模态级多跳检索获取 answer-supporting 的上下文，产出忠实回答。</figcaption>
</figure>

## 方法：M3KG-RAG 如何实现多跳多模态知识图谱增强？

### 1. M3KG 构建：多 Agent 管线

M3KG 的构建由一个轻量级多 agent 管线完成，包含 7 种专用 LLM agent（rewriter、extractor、normalizer、searcher、selector、refiner、inspector），全部基于 Qwen3-8B 实现。管线分三步：

**Step 1: Context-Enriched Triplet Extraction**
- **Rewriter** 将语义泛化的原始 caption 重写为 knowledge-intensive caption，融入 YouTube title/description 等外部知识。
- **Extractor** 从重写后的 caption 中抽取结构化三元组 \((h, r, t)\)。由于 caption 已经是 knowledge-intensive 的，抽取的三元组往往包含长尾实体和非常见概念。

**Step 2: Knowledge Grounding**
- **Normalizer** 将实体标准化为可搜索的 canonical concept（去掉修饰词、统一为单数名词短语）。
- **Searcher** 查询 Wikipedia/Wiktionary 获取候选描述；对查不到的实体，调用 LLM callback 生成描述。

**Step 3: Context-Aware Description Refinement**
- **Selector** 从多个候选描述中选择最契合当前上下文语境的描述。
- **Refiner** 将选中的描述从 canonical concept 适配回原始实体的表述方式。

**Self-Reflection Loop**：Inspector agent 对每个描述打分（0-10），低于 7 分的退回重新生成，最多 3 次迭代，仍不达标的丢弃。

最终构建的 M3KG 为：
\[
G = \{E, R, \mathcal{T}, \hat{D}, A, V, L\}
\]
其中每个三元组都至少链接到一个 audio 或 visual item，保证全图覆盖性（full graph coverage）。

<figure class="figure">
  <img src="assets/papers/m3kg-rag/fig2_pipeline.webp" alt="M3KG 构建管线" loading="lazy" />
  <figcaption>图 2：M3KG 构建管线概览。三步流程 + Self-Reflection Loop，从多模态语料中构建包含 context-enriched triplet 和 refined description 的多跳知识图谱。</figcaption>
</figure>

### 2. Modality-Wise Retrieval：消除 Modality Gap

核心思想：**只在 query 的同模态空间内做 nearest-neighbor search**，避免跨模态距离校准不准导致的 off-topic 检索。

- 用模态基础模型（InternVL2 处理 video，CLAP 处理 audio）计算 query embedding。
- 在 M3KG 的 audio/visual items 上构建 FAISS 索引，用 L2 距离检索 top-\(k\) 最近邻。
- 通过距离阈值 \(\tau\) 过滤掉过远的候选项。
- 将选中的 items 通过链接 \(L\) 提升为对应的三元组子图 \(G_{\text{init}}\)。

### 3. GRASP：Grounded Retrieval And Selective Pruning

GRASP 是论文的核心检索贡献，分两个阶段逐步剪枝：

**阶段一：Multimodal Grounded Retrieval**

利用 off-the-shelf 多模态 grounding 模型验证三元组中的实体是否真的出现在 query 的 audio/visual 流中：

- **Visual Grounding**：用 GroundingDINO 在 query video 的 4 帧均匀采样上做检测，取最大置信度作为视觉存在分数：
\[
s_v(e | q_v) = \max_{f \in F} \Phi_v(e; f)
\]
对三元组的 head 和 tail 实体分数求和，低于阈值 \(\eta_v\) 的剪掉。

- **Audio Grounding**：用 TAG（Text-to-Audio Grounding）模型，将三元组转为自然语言句子，测量其在 query audio 中的 grounding 强度：
\[
s_a(t | q_a) = \Phi_a(\sigma(t); q_a)
\]
低于阈值 \(\eta_a\) 的剪掉。当 audio 和 visual 同时可用时，融合分数并应用阈值 \(\eta_{av}\)。

**阶段二：LLM-based Selective Pruning**

轻量级 LLM（Qwen3-8B）对 grounded subgraph 中的三元组做 binary keep-or-drop 决策，采用 conservative 策略（uncertain 时倾向保留），去除对回答无用的知识。

<figure class="figure">
  <img src="assets/papers/m3kg-rag/fig3_framework.webp" alt="多模态 RAG 框架" loading="lazy" />
  <figcaption>图 3：多模态 RAG 框架。(a) Modality-Wise Retrieval 在同模态空间内检索多跳三元组；(b) GRASP 通过 visual/audio grounding 验证实体存在性，再用 LLM filter 剪除无用知识，最终子图送入 MLLM。</figcaption>
</figure>

### 4. Graph-Augmented Generation

最终检索到的子图 \(G_{\text{GRASP}}\) 与多模态 query 拼接后送入 MLLM。对每个三元组 \((h, r, t)\)，包含关系 \(r\) 以及 head/tail 的 refined descriptions \(\langle h, d_h \rangle\) 和 \(\langle t, d_t \rangle\)：

\[
p_{\text{aug}} = q \;\|\; \left( \bigcup_{(h,r,t) \in G_{\text{GRASP}}} \langle h, d_h \rangle \xrightarrow{r} \langle t, d_t \rangle \right)
\]

## 数据与评测

**评测基准**：
- **Audio QA**：AudioCaps-QA（基于 AudioSet 的 10 秒音频片段 + 人工标注 QA）
- **Video QA**：VideoChatGPT benchmark（基于 ActivityNet 视频）
- **Audio-Visual QA**：VALOR（需要同步 audio-visual 联合推理）

**Base MLLMs**：VideoLLaMA2、Qwen2.5-Omni（开源）、GPT-4o（商业）

**Baseline**：None（无外部知识）、Wikidata + naïve RAG、VTKG（image-text MMKG）、M2ConceptBase、VAT-KG（audio-visual MMKG）

**M3KG 构建**：使用各 benchmark 的 training split 构建（AudioCaps + ActivityNet + VALOR-32K），不访问 QA 标注。

**评估指标**：
- **Model-as-Judge (M.J.)**：LLM judge 对回答打 0-5 分（映射到 0-100）
- **Reference-Aware Win-Rate**：提供参考答案的 pairwise 比较，评估 Comprehensiveness / Diversity / Empowerment / Overall

## 实验结果

### 主实验：全 benchmark 对比

<div class="table-wrap">
<table>
<thead>
<tr><th>MLLM</th><th>Method</th><th>AudioCaps-QA</th><th>VCGPT</th><th>VALOR</th></tr>
</thead>
<tbody>
<tr><td rowspan="6"><strong>VideoLLaMA2</strong></td><td>None</td><td>43.13</td><td>39.09</td><td>25.66</td></tr>
<tr><td>Wikidata</td><td>43.58</td><td>38.58</td><td>26.43</td></tr>
<tr><td>VTKG</td><td>43.02</td><td>38.88</td><td>25.92</td></tr>
<tr><td>M2ConceptBase</td><td>42.19</td><td>39.31</td><td>25.93</td></tr>
<tr><td>VAT-KG</td><td>44.60</td><td>39.42</td><td>28.30</td></tr>
<tr><td><strong>M3KG-RAG</strong></td><td><strong>53.23</strong></td><td><strong>39.92</strong></td><td><strong>29.25</strong></td></tr>
<tr><td rowspan="6"><strong>Qwen2.5-Omni</strong></td><td>None</td><td>49.00</td><td>42.21</td><td>32.42</td></tr>
<tr><td>Wikidata</td><td>49.78</td><td>40.82</td><td>30.28</td></tr>
<tr><td>VTKG</td><td>48.95</td><td>42.96</td><td>32.70</td></tr>
<tr><td>M2ConceptBase</td><td>49.78</td><td>42.78</td><td>32.31</td></tr>
<tr><td>VAT-KG</td><td>51.30</td><td>43.50</td><td>35.44</td></tr>
<tr><td><strong>M3KG-RAG</strong></td><td><strong>60.77</strong></td><td><strong>44.35</strong></td><td><strong>44.67</strong></td></tr>
</tbody>
</table>
</div>

**关键发现**：
- **Audio QA 提升最显著**：Qwen2.5-Omni + M3KG-RAG 在 AudioCaps-QA 上比 VAT-KG 高 9.47 分，说明多跳 audio 知识对音频理解至关重要。
- **Audio-Visual QA 差距最大**：VALOR 上比 VAT-KG 高 9.23 分（44.67 vs 35.44），需要联合 audio-visual 推理时，多跳证据的价值最大。
- **Video QA 提升相对温和**：VCGPT 上仅提升 0.85 分，可能因为 video-only 任务的知识需求相对简单。
- **GPT-4o 也受益**：即使商业模型已有丰富内置知识，M3KG-RAG 仍能在所有 benchmark 上带来一致提升。

### GPT-4o 实验

<div class="table-wrap">
<table>
<thead>
<tr><th>Method</th><th>AudioCaps-QA</th><th>VCGPT</th><th>VALOR</th></tr>
</thead>
<tbody>
<tr><td>GPT-4o (None)</td><td>56.74</td><td>49.68</td><td>46.02</td></tr>
<tr><td>GPT-4o + VAT-KG</td><td>57.70</td><td>51.49</td><td>55.86</td></tr>
<tr><td><strong>GPT-4o + M3KG-RAG</strong></td><td><strong>59.17</strong></td><td><strong>53.05</strong></td><td><strong>56.53</strong></td></tr>
</tbody>
</table>
</div>

## Ablation：哪些设计最关键？

### Modality-Wise Retrieval vs. GRASP

<div class="table-wrap">
<table>
<thead>
<tr><th>Modality-Wise Retrieval</th><th>GRASP</th><th>M.J. (VALOR)</th></tr>
</thead>
<tbody>
<tr><td>✗</td><td>✗</td><td>36.62</td></tr>
<tr><td>✓</td><td>✗</td><td>40.91 (+4.29)</td></tr>
<tr><td>✗</td><td>✓</td><td>36.96 (+0.34)</td></tr>
<tr><td>✓</td><td>✓</td><td><strong>44.67 (+8.05)</strong></td></tr>
</tbody>
</table>
</div>

**核心结论**：两个组件高度互补——Modality-Wise Retrieval 提供模态对齐的候选子图，GRASP 在此基础上做细粒度剪枝。**单独使用 GRASP 几乎无效**（仅 +0.34），因为如果初始检索的子图本身就与 audio-visual query 不对齐（shared embedding space 的 modality gap），再怎么剪枝也无法挽救。

### GRASP 组件逐步消融

<div class="table-wrap">
<table>
<thead>
<tr><th>GRASP 组件</th><th>VRAM (GB)</th><th>Avg time/query (s)</th><th>M.J.</th></tr>
</thead>
<tbody>
<tr><td>None</td><td>23.0</td><td>4.30</td><td>40.91</td></tr>
<tr><td>+ GDino</td><td>23.7</td><td>5.75</td><td>41.35</td></tr>
<tr><td>+ TAG</td><td>23.6</td><td>4.48</td><td>41.70</td></tr>
<tr><td>+ GDino + TAG</td><td>24.2</td><td>6.02</td><td>42.96</td></tr>
<tr><td><strong>+ GDino + TAG + LLM Filter</strong></td><td><strong>39.8</strong></td><td><strong>7.02</strong></td><td><strong>44.87</strong></td></tr>
</tbody>
</table>
</div>

每个 grounding 模块独立贡献约 0.4-0.8 分，组合后达到 42.96（audio 和 visual grounding 互补）。加上 LLM filter 后再提升 ~2 分，说明**语义级剪枝**（判断知识是否对回答有用）在 grounding 之上仍有显著价值。

## 我的理解与启发

1. **Modality gap 是多模态 RAG 中被低估的瓶颈**。论文的核心洞察之一是：与其在统一 embedding 空间里硬做跨模态检索，不如按 query 的模态分别在同模态空间内搜索。这个思路简单但有效，代价是需要在 MMKG 中显式存储 audio/visual items 及其 embedding。

2. **多跳 KG 的价值在 audio-visual 联合推理时最突出**。从 ablation 看，VALOR（需要同时理解音频和视觉）上的提升远大于纯 Video QA。这说明当任务需要跨模态因果推理时，结构化的多跳证据链比碎片化的单跳知识更有用。

3. **GRASP 的 grounding 思路值得借鉴**。用 GroundingDINO 和 TAG 这类专用 grounding 模型来验证「检索到的知识是否真的存在于 query 中」，是一个很好的 filtering 策略。这比单纯靠 similarity score 或 LLM 判断更可靠，因为 grounding 模型直接在像素/波形层面做验证。

4. **8B LLM 作为 backbone 的 multi-agent 管线很实用**。整个 KG 构建管线都基于 Qwen3-8B，不需要大模型，降低了构建成本。Self-Reflection Loop 的 Inspector 设计保证了图质量。

5. **即使 GPT-4o 级别的模型也能从外部结构化知识中受益**。这暗示了当前商业 MLLM 在 audio-visual 领域的知识仍有盲区，多跳 KG 提供了一种可解释、可更新的知识注入方式。

## 局限与待观察点

- **M3KG 的知识覆盖受限于构建语料**：长尾实体、罕见关系或特定领域的知识可能未被覆盖。论文使用 benchmark 的 training split 构建，实际应用需要更大规模的语料。
- **检索阈值需要按 benchmark 单独调参**：\(\tau\) 和 \(\eta\) 在不同 benchmark 上差异很大（AudioCaps: \(\tau=3.0, \eta_a=0.5\)；VCGPT: \(\tau=0.15, \eta_v=1.5\)），缺乏自适应机制。
- **Grounding 模型的领域迁移性**：当 GroundingDINO/TAG 的训练域与 query 分布不匹配时，grounding 精度下降会影响 GRASP 效果。
- **端到端延迟**：完整管线约 7s/query + KG 构建的前置开销，实时性场景可能受限。
- **缺乏对 text-only KG 的公平对比**：论文主要与 MMKG baseline 比较，但与 GraphRAG、LightRAG 等 text-only graph RAG 的对比不够充分。

## 结论

M3KG-RAG 是一个端到端的多模态 RAG 框架，核心贡献在于：(1) 用多 agent 管线构建多跳 MMKG，使每个三元组都有 audio-visual grounding；(2) Modality-Wise Retrieval 通过同模态检索消除 modality gap；(3) GRASP 通过 grounding + LLM filter 双重剪枝，确保送入 MLLM 的知识既与 query 相关又对回答有用。在三个不同类型的多模态 QA benchmark 上全面超越现有方法，包括对 GPT-4o 的提升，展示了多跳结构化知识在多模态推理中的不可替代性。
