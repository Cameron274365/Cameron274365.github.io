---
id: "m3kg-rag-multi-hop-multimodal-kg-rag"
title: "M³KG-RAG：用多跳知识图谱和精准剪枝解决音视频 RAG 的两大瓶颈"
category: "多模态 RAG"
date: "2026-07-07"
order: 5
readTime: "16 min"
tags: ["RAG", "Multimodal RAG", "Knowledge Graph", "Audio-Visual", "Multi-hop", "GRASP", "MLLM", "CVPR 2026"]
summary: "M³KG-RAG 指出音视频多模态 RAG 的两大瓶颈：现有 MMKG 缺乏多跳连通性和 audio-visual 覆盖，共享 embedding 空间的 modality gap 导致检索偏离。论文用 7-agent 管线构建每个三元组都链接到 audio/visual item 的多跳 MMKG，用同模态检索绕过跨模态距离校准问题，再用 GRASP（grounding 验证 + LLM 语义剪枝）只保留对回答有用的知识。在 AudioCaps-QA、VideoChatGPT、VALOR 上全面超越 VAT-KG 等基线，包括对 GPT-4o 的一致提升。"
hero: "assets/papers/m3kg-rag/fig1_comparison.webp"
---

## 一句话总结
M³KG-RAG 把音视频 RAG 的问题拆解为两层：**知识源不够结构化**（单跳、缺 audio-visual 链接）和**检索不够精确**（modality gap + 无答案相关性过滤）。它用多 agent 管线构建全图覆盖的多跳 MMKG，用同模态检索消除跨模态偏差，再用 GRASP 的双重剪枝确保送入 MLLM 的每一条知识都既存在于 query 中、又对回答有贡献。

<div class="metric-grid">
  <div class="metric"><strong>+9.47</strong><span>Qwen2.5-Omni 在 AudioCaps-QA 上相对 VAT-KG 的提升</span></div>
  <div class="metric"><strong>+9.23</strong><span>VALOR 上相对 VAT-KG 的提升（audio-visual 联合推理）</span></div>
  <div class="metric"><strong>~7s/query</strong><span>完整 GRASP 管线平均推理延迟</span></div>
  <div class="metric"><strong>~40 GB</strong><span>完整系统 VRAM 占用（H100 单卡可运行）</span></div>
</div>

## 论文信息
- **标题**：M³KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation
- **作者**：Hyeongcheol Park, Jiyoung Seo, Jaewon Mun, Hogun Park, Wonmin Byeon, Sung June Kim, Hyeonsoo Im, JeungSub Lee, Sangpil Kim
- **机构**：Korea University, Sungkyunkwan University, NVIDIA Research, Hanwha Systems
- **会议**：CVPR 2026；arXiv v3, 2026-04-13
- **链接**：[arXiv:2512.20136](https://arxiv.org/abs/2512.20136) · [PDF](https://arxiv.org/pdf/2512.20136) · [项目主页](https://kuai-lab.github.io/cvpr2026m3kgrag/)

## 背景：音视频 RAG 的两个结构性瓶颈

多模态 RAG 已经在 image-text 场景取得进展，但在 **audio-visual** 领域仍有两个根本性问题：

### 1. 知识图谱不够"多跳"也不够"多模态"
现有 MMKG 要么只覆盖 image-text（VTKG、M2ConceptBase），要么虽纳入 audio-visual 但只构建 concept-level 单跳图（VAT-KG）。单跳图无法捕捉跨模态的时序依赖和因果链条——例如"这段音乐响起时画面中发生了什么"需要 audio→event→visual 的多跳推理。

### 2. 共享 embedding 空间的 modality gap 让检索失准
多模态编码器（InternVL2、CLAP）的统一向量空间中，不同模态的距离校准不一致。用 video query 搜 text 索引、或用 audio query 搜 visual items，检索到的往往是模态相近而非语义相关的邻居。即使检索到了"看起来相关"的知识，也可能与当前 query 的具体内容无关，成为噪声。

<figure class="figure">
  <img src="assets/papers/m3kg-rag/fig1_comparison.webp" alt="M3KG-RAG 与现有方法对比" loading="lazy" />
  <figcaption>图 1：三种范式对比。(a) 共享 embedding 检索因 modality gap 命中错误证据；(b) 单跳图提供碎片化上下文，无法支撑推理；(c) M³KG-RAG 通过模态级多跳检索 + GRASP 剪枝，获取真正 answer-supporting 的上下文。</figcaption>
</figure>

## 方法：三层设计逐层解决问题

### 1. M³KG 构建：让每个三元组都有 audio-visual 锚点

论文用一个 7-agent 管线（全部基于 Qwen3-8B）从原始多模态语料构建多跳 MMKG，分三步：

**Step 1: Context-Enriched Triplet Extraction**
- **Rewriter** 将语义泛化的 caption 重写为 knowledge-intensive 文本，融入 YouTube title/description 等外部知识。
- **Extractor** 从重写文本中抽取三元组 \((h, r, t)\)。因为输入已是知识密集型文本，抽取的三元组常包含长尾实体和非常见关系。

**Step 2: Knowledge Grounding**
- **Normalizer** 将实体标准化为可搜索的 canonical concept（去修饰词、统一单数名词短语）。
- **Searcher** 查询 Wikipedia/Wiktionary 获取候选描述；未命中的实体由 LLM callback 补全。

**Step 3: Context-Aware Description Refinement**
- **Selector** 从多个候选描述中选择最契合当前上下文的版本。
- **Refiner** 将选中描述从 canonical concept 适配回原始实体的表述。

**Self-Reflection Loop**：**Inspector** agent 对每个描述打分（0-10），低于 7 分退回重生，最多 3 轮，仍不达标则丢弃。

最终得到的 M³KG 形式化为 \(G = \{E, R, \mathcal{T}, \hat{D}, A, V, L\}\)，关键约束是**全图覆盖性**：每个三元组至少链接到一个 audio 或 visual item。这保证了后续模态级检索可以触达图中任意知识。

<figure class="figure">
  <img src="assets/papers/m3kg-rag/fig2_pipeline.webp" alt="M3KG 构建管线" loading="lazy" />
  <figcaption>图 2：M³KG 构建管线。三步流程 + Self-Reflection Loop，从多模态语料中提取 context-enriched triplet，grounding 到外部知识库，再 refine 为上下文一致的描述。</figcaption>
</figure>

### 2. Modality-Wise Retrieval：在同模态空间内搜索

核心原则：**不做跨模态检索**。query 是 video 就只在 visual items 中搜，是 audio 就只在 audio items 中搜。

具体流程：
1. 用模态基础模型（InternVL2 处理 video，CLAP 处理 audio）编码 query。
2. 在 M³KG 对应模态的 items 上用 FAISS 做 L2 近邻搜索。
3. 距离阈值 \(\tau\) 过滤过远候选。
4. 通过链接 \(L\) 将选中 items 提升为三元组子图 \(G_{\text{init}}\)。

这一步绕过了 modality gap：不再让 video embedding 和 text embedding 直接比较，而是利用 M³KG 中预建的 audio/visual 锚点做同模态匹配。

### 3. GRASP：双重剪枝只留有用知识

GRASP（Grounded Retrieval And Selective Pruning）解决"检索到的知识虽然模态对齐，但不一定存在于当前 query 中或对回答有用"的问题。

**阶段一：Multimodal Grounded Retrieval**
用专用 grounding 模型验证三元组实体是否真的出现在 query 中：
- **Visual**：GroundingDINO 在 4 帧均匀采样上检测，取最大置信度 \(s_v(e|q_v) = \max_f \Phi_v(e; f)\)，head+tail 分数和低于 \(\eta_v\) 则剪掉。
- **Audio**：TAG 模型将三元组转为自然语言句子，测量在 query audio 中的 grounding 强度 \(s_a(t|q_a)\)，低于 \(\eta_a\) 则剪掉。

**阶段二：LLM-based Selective Pruning**
Qwen3-8B 对 grounded subgraph 做 binary keep-or-drop 决策，采用 conservative 策略（不确定时倾向保留），去除对回答无贡献的知识。

<figure class="figure">
  <img src="assets/papers/m3kg-rag/fig3_framework.webp" alt="M3KG-RAG 检索框架" loading="lazy" />
  <figcaption>图 3：检索框架总览。(a) Modality-Wise Retrieval 在同模态空间内检索候选子图；(b) GRASP 先用 grounding 模型验证实体存在性，再用 LLM filter 剪除无用知识，最终精简子图送入 MLLM 生成回答。</figcaption>
</figure>

## 数据与评测

### 评测基准
| 基准 | 模态 | 特点 |
|------|------|------|
| AudioCaps-QA | Audio | 基于 AudioSet 10s 片段 + 人工标注 QA |
| VideoChatGPT | Video | 基于 ActivityNet 视频 |
| VALOR | Audio-Visual | 需同步 audio-visual 联合推理 |

### 模型与基线
- **Base MLLMs**：VideoLLaMA2、Qwen2.5-Omni（开源）、GPT-4o（商业）
- **Baselines**：None、Wikidata + naïve RAG、VTKG、M2ConceptBase、VAT-KG
- **M³KG 构建语料**：各 benchmark training split（AudioCaps + ActivityNet + VALOR-32K），不访问 QA 标注

### 评估指标
- **Model-as-Judge (M.J.)**：LLM judge 打 0-5 分（映射到 0-100）
- **Reference-Aware Win-Rate**：pairwise 比较，评估 Comprehensiveness / Diversity / Empowerment / Overall

## 实验结果

### 主实验：全 benchmark 领先

<div class="table-wrap">
<table>
<thead><tr><th>MLLM</th><th>Method</th><th>AudioCaps-QA</th><th>VCGPT</th><th>VALOR</th></tr></thead>
<tbody>
<tr><td rowspan="6"><strong>VideoLLaMA2</strong></td><td>None</td><td>43.13</td><td>39.09</td><td>25.66</td></tr>
<tr><td>VAT-KG</td><td>44.60</td><td>39.42</td><td>28.30</td></tr>
<tr><td><strong>M³KG-RAG</strong></td><td><strong>53.23</strong></td><td><strong>39.92</strong></td><td><strong>29.25</strong></td></tr>
<tr><td rowspan="6"><strong>Qwen2.5-Omni</strong></td><td>None</td><td>49.00</td><td>42.21</td><td>32.42</td></tr>
<tr><td>VAT-KG</td><td>51.30</td><td>43.50</td><td>35.44</td></tr>
<tr><td><strong>M³KG-RAG</strong></td><td><strong>60.77</strong></td><td><strong>44.35</strong></td><td><strong>44.67</strong></td></tr>
</tbody>
</table>
</div>

**关键发现**：
- **Audio QA 提升最大**：Qwen2.5-Omni 在 AudioCaps-QA 上比 VAT-KG 高 **+9.47** 分，多跳 audio 知识对音频理解至关重要。
- **Audio-Visual 联合推理差距最显著**：VALOR 上 **+9.23** 分（44.67 vs 35.44），跨模态因果推理时多跳证据链价值最大。
- **Video-only 提升温和**：VCGPT 仅 +0.85 分，video-only 任务知识需求相对简单。
- **GPT-4o 也受益**：商业模型已有丰富内置知识，M³KG-RAG 仍在所有 benchmark 上一致提升（AudioCaps +1.47, VCGPT +1.56, VALOR +0.67）。

## Ablation：两个组件缺一不可

### Modality-Wise Retrieval × GRASP 交互

<div class="table-wrap">
<table>
<thead><tr><th>Modality-Wise</th><th>GRASP</th><th>M.J. (VALOR)</th><th>Δ</th></tr></thead>
<tbody>
<tr><td>✗</td><td>✗</td><td>36.62</td><td>—</td></tr>
<tr><td>✓</td><td>✗</td><td>40.91</td><td>+4.29</td></tr>
<tr><td>✗</td><td>✓</td><td>36.96</td><td>+0.34</td></tr>
<tr><td>✓</td><td>✓</td><td><strong>44.67</strong></td><td><strong>+8.05</strong></td></tr>
</tbody>
</table>
</div>

**核心结论**：两者高度互补。**单独 GRASP 几乎无效**（+0.34）——如果初始子图因 modality gap 本身就与 query 不对齐，再怎么剪枝也救不回来。Modality-Wise Retrieval 先保证候选子图的模态对齐，GRASP 才能在此基础上做有意义的细粒度筛选。

### GRASP 组件逐步叠加

<div class="table-wrap">
<table>
<thead><tr><th>组件</th><th>VRAM</th><th>延迟</th><th>M.J.</th></tr></thead>
<tbody>
<tr><td>None</td><td>23.0 GB</td><td>4.30s</td><td>40.91</td></tr>
<tr><td>+ GDino</td><td>23.7 GB</td><td>5.75s</td><td>41.35</td></tr>
<tr><td>+ TAG</td><td>23.6 GB</td><td>4.48s</td><td>41.70</td></tr>
<tr><td>+ GDino + TAG</td><td>24.2 GB</td><td>6.02s</td><td>42.96</td></tr>
<tr><td><strong>+ GDino + TAG + LLM Filter</strong></td><td><strong>39.8 GB</strong></td><td><strong>7.02s</strong></td><td><strong>44.87</strong></td></tr>
</tbody>
</table>
</div>

Visual 和 audio grounding 各自贡献 0.4-0.8 分且互补；LLM filter 再叠加 ~2 分，说明**语义级剪枝**（判断知识是否对回答有用）在像素/波形级 grounding 之上仍有独立价值。

## 我的理解与启发

1. **Modality gap 不是"换个更好的 encoder"能解决的**。论文的选择很务实：既然统一空间的跨模态距离不可靠，就不做跨模态检索，转而用 MMKG 中的 audio/visual 锚点做同模态匹配。这对工程系统的启示是：**与其追求完美的统一 embedding，不如在系统设计层面绕过这个问题**。

2. **"知识存在于 query 中"和"知识对回答有用"是两个独立的过滤维度**。Grounding 解决前者（实体是否真的出现），LLM filter 解决后者（是否有助于推理）。很多 RAG 系统只做相似度过滤，混淆了这两个问题。

3. **多跳 KG 的价值与任务复杂度正相关**。VALOR（audio-visual 联合推理）上的提升远大于纯 Video QA，说明当任务需要跨模态因果推理时，结构化证据链比碎片化知识更有用。这提示我们：**RAG 的收益取决于任务是否需要结构化推理，而非简单的知识补充**。

4. **8B 模型足以驱动高质量 KG 构建**。整个管线基于 Qwen3-8B，配合 Self-Reflection Loop 保证质量。这降低了构建门槛，也让 KG 可以更频繁地随语料更新。

5. **即使 GPT-4o 也有 audio-visual 知识盲区**。外部结构化知识对商业模型仍有增量价值，且这种注入方式是可解释、可更新的，比 fine-tuning 更灵活。

## 局限与待观察点

- **知识覆盖受限于构建语料**：使用 benchmark training split 构建，长尾实体和特定领域知识可能缺失，实际应用需更大规模语料。
- **检索阈值需逐 benchmark 调参**：\(\tau\) 和 \(\eta\) 在不同数据集上差异大（AudioCaps: \(\tau=3.0\)；VCGPT: \(\tau=0.15\)），缺乏自适应机制。
- **Grounding 模型的领域迁移性**：GroundingDINO/TAG 训练域与 query 分布不匹配时，grounding 精度下降会级联影响 GRASP。
- **端到端延迟 ~7s/query**：加上 KG 构建的前置开销，实时场景受限。
- **缺少与 text-only GraphRAG 的公平对比**：主要与 MMKG baseline 比较，未充分对比 GraphRAG、LightRAG 等文本图 RAG。

## 结论

M³KG-RAG 把音视频 RAG 的改进路径讲得很清楚：**先让知识源足够结构化和多模态（M³KG），再让检索足够精确（Modality-Wise + GRASP）**。它的贡献不仅是性能数字，更是把"modality gap"和"answer relevance"这两个常被混淆的问题拆开，分别用同模态检索和双重剪枝来解决。在三个差异化 benchmark 上的一致领先，包括对 GPT-4o 的提升，证明了多跳结构化知识在多模态推理中的不可替代性。