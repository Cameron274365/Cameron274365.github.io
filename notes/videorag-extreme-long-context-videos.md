---
id: "videorag-extreme-long-context-videos"
title: "VideoRAG：面向超长视频的检索增强生成"
category: "多模态 RAG"
date: "2026-06-10"
order: 45
readTime: "16 min"
tags: ["Video RAG","Multimodal RAG","Long Video Understanding","Knowledge Graph","Retrieval","ImageBind","MiniCPM-V"]
summary: "VideoRAG 将视频先转化为可检索的跨视频知识库：一条通道用图结构组织视觉 caption 与 ASR 文本，另一条通道保留多模态视觉 embedding，再通过文本语义检索、视觉相似度检索和 LLM 过滤共同定位相关片段。论文提出 LongerVideos，覆盖 164 个视频、134.6 小时和 602 个查询，用于评估跨多个超长视频的问答能力。"
hero: "assets/papers/videorag/framework.png"
---

## 一句话总结
VideoRAG 的核心贡献是把“看很长的视频”转化为“先构建可检索的视频知识库，再按问题检索相关片段并生成答案”。它不是让 VLM 直接吞下数小时视频，而是通过 **图结构文本知识 + 多模态视觉 embedding** 的双通道索引，把多个长视频中的视觉、语音和语义关系组织起来，从而支持跨视频、超长上下文的问答。

<div class="metric-grid">
  <div class="metric"><strong>164</strong><span>LongerVideos 中的视频数量</span></div>
  <div class="metric"><strong>134.6h</strong><span>总视频时长</span></div>
  <div class="metric"><strong>602</strong><span>跨视频问答查询</span></div>
  <div class="metric"><strong>4.45</strong><span>相对 NaiveRAG 的总体评分</span></div>
</div>

## 论文信息
- **标题**：VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos
- **作者**：Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, Chao Huang
- **机构**：The University of Hong Kong, Baidu Inc.
- **版本**：arXiv 2025
- **链接**：[arXiv:2502.01549](https://arxiv.org/abs/2502.01549) · [PDF](https://arxiv.org/pdf/2502.01549) · [代码与数据](https://github.com/HKUDS/VideoRAG)

## 背景：为什么视频 RAG 比文本 RAG 更难？
文本 RAG 已经相对成熟：把文档切 chunk、做 embedding、检索相关段落，再交给 LLM 生成答案。但视频不是文本的简单延伸，它同时包含视觉画面、语音、时间顺序和跨片段语义关系。

论文指出，极长视频场景至少有三类困难：

- **异构信息难以统一组织**：视频包含画面、动作、对白、旁白和场景变化，纯文本索引会丢失视觉细节，纯视觉 embedding 又难以表达复杂语义关系。
- **跨视频语义关系难保留**：真实问题经常不是问某个单独片段，而是问一组课程、纪录片或系列视频中的知识联系。
- **检索空间极大**：当知识库包含任意数量、任意长度的视频时，系统必须快速定位最相关片段，而不是把所有视频帧都塞给模型。

因此，VideoRAG 的目标不是单纯扩展上下文长度，而是为视频建立一个可持续增长、可跨视频连接、可按问题检索的知识层。

## 方法：VideoRAG 的双通道架构
<figure class="figure">
  <img src="assets/papers/videorag/framework.png" alt="VideoRAG 总体框架" loading="lazy" />
  <figcaption>图 1：VideoRAG 总体框架。系统先构建 graph-based textual knowledge grounding 和 multi-modal context encoding 两类索引，再结合文本语义检索、视觉检索和 LLM 过滤完成 query-aware retrieval。</figcaption>
</figure>

### 1. Graph-based Textual Knowledge Grounding：把视频变成结构化知识
VideoRAG 首先把长视频切成 30 秒左右的短 clip。对每个 clip，它做两件事：

- **Vision-Text Grounding**：从 clip 中均匀采样少量帧，使用 VLM 生成视觉 caption，描述人物、物体、动作和场景动态。论文默认初始 caption 使用 \(k=5\) 帧。
- **Audio-Text Grounding**：使用 ASR 提取语音转录，补足画面无法表达的讲解、对白和术语。

随后，系统把视觉 caption 和 ASR transcript 合并成每个 clip 的文本表示，并进一步用 LLM 做实体和关系抽取，构建跨视频知识图谱。这个图谱中的节点是实体或概念，边是它们之间的关系。

我理解这里的关键是：**视频 RAG 不能只做“片段级向量检索”，还要把多个视频里反复出现的概念和关系连起来**。例如一套课程中多次提到 “reinforcement fine-tuning”“graders”“partial credit”，图结构可以把它们合并到统一语义空间里，而不是散落在不同视频 chunk 中。

### 2. Multi-Modal Context Encoding：保留无法文本化的视觉细节
只把视频转成文本会丢失很多信息，例如车的颜色、画面布局、人物动作、光照和镜头细节。为此，VideoRAG 额外使用多模态编码器构建视觉 embedding 索引。

论文使用类似 CLIP / ImageBind 的多模态编码能力，把视频 clip 和文本 query 映射到共享向量空间。这样，系统既能用文本图谱做语义检索，也能用视觉 embedding 找到“画面上相似”的片段。

这条通道解决的是文本 caption 的天然瓶颈：caption 总会选择性描述内容，但用户问题可能刚好问到 caption 没写出的视觉细节。

### 3. Multi-Modal Retrieval：文本语义检索 + 视觉检索 + LLM 过滤
在查询阶段，VideoRAG 不只做一次向量搜索，而是组合三步：

1. **Textual Semantic Matching**：将用户 query 改写成声明式表达，在知识图谱中匹配相关实体，再借助 GraphRAG 风格的方法选出相关 chunk，并映射回对应视频 clip。
2. **Visual Retrieval via Content Embeddings**：用 LLM 从 query 中抽取视觉场景描述，再通过多模态 encoder 与视频 clip embedding 做相似度匹配。
3. **LLM-based Video Clip Filtering**：对候选 clip 做相关性判断，过滤噪声片段，只保留真正能回答问题的视频内容。

这个设计体现了一个实用判断：视频检索经常需要“语义上相关”和“视觉上匹配”同时成立。单纯图检索容易漏掉视觉细节，单纯视觉检索又容易缺少概念级推理。

### 4. Query-Aware Content Integration：按问题重新生成片段描述
检索到片段后，VideoRAG 不是直接使用索引阶段的粗 caption，而是进入更细的内容抽取阶段：

- 先用 LLM 从 query 中提取关键词；
- 再用更多帧重新调用 VLM 生成 query-aware caption，论文默认 \(\hat{k}=15\) 帧；
- 最后把这些 query-specific 视频描述和相关文本 chunk 一起交给通用 LLM 生成答案。

这一点很重要：索引阶段的 caption 是通用摘要，可能不够回答具体问题；查询阶段重新围绕问题生成描述，可以提升答案的针对性。

## 数据与评测：LongerVideos
论文提出了新的评测集 **LongerVideos**，用于测试模型对多个超长视频的跨视频问答能力。

<div class="table-wrap">
  <table>
    <thead><tr><th>类型</th><th>视频列表数</th><th>视频数</th><th>查询数</th><th>平均每列表查询</th><th>总时长</th></tr></thead>
    <tbody>
      <tr><td>Lecture</td><td>12</td><td>135</td><td>376</td><td>31.3</td><td>约 64.3 小时</td></tr>
      <tr><td>Documentary</td><td>5</td><td>12</td><td>114</td><td>22.8</td><td>约 28.5 小时</td></tr>
      <tr><td>Entertainment</td><td>5</td><td>17</td><td>112</td><td>22.4</td><td>约 41.9 小时</td></tr>
      <tr><td><strong>All</strong></td><td><strong>22</strong></td><td><strong>164</strong></td><td><strong>602</strong></td><td><strong>27.4</strong></td><td><strong>约 134.6 小时</strong></td></tr>
    </tbody>
  </table>
</div>

这些视频来自开放 YouTube 内容，覆盖三类场景：

- **Lecture Video**：AI Agents、RAG 技术等教育课程；
- **Documentary Video**：自然、野生动物、访谈等纪录片；
- **Entertainment Video**：颁奖典礼、游戏解说、旅行记录等。

评测协议主要有两种：

- **Win-Rate Comparison**：使用 GPT-4o-mini 比较两个模型答案，从 comprehensiveness、empowerment、trustworthiness、depth、density 等维度判断胜负。
- **Quantitative Comparison**：以 NaiveRAG 为基线，给不同模型答案打 1 到 5 分，衡量其相对基线的好坏。

论文还做了位置交换和多轮重复评测，以降低 LLM judge 的位置偏差和随机方差。

## 实验结果
### 1. 相比文本 RAG 基线：VideoRAG 稳定胜出
VideoRAG 与 NaiveRAG、GraphRAG、LightRAG 等文本 RAG 方法相比，在 LongerVideos 上整体胜率更高。

<div class="table-wrap">
  <table>
    <thead><tr><th>对比方法</th><th>All Overall Winner：VideoRAG</th><th>主要原因</th></tr></thead>
    <tbody>
      <tr><td>NaiveRAG</td><td><strong>53.26%</strong></td><td>VideoRAG 能跨视频组织知识，不只是检索孤立文本 chunk。</td></tr>
      <tr><td>GraphRAG-local</td><td><strong>53.87%</strong></td><td>VideoRAG 在图结构之外额外融合视觉 embedding。</td></tr>
      <tr><td>GraphRAG-global</td><td><strong>55.78%</strong></td><td>多模态检索提升了视觉细节和语义匹配能力。</td></tr>
      <tr><td>LightRAG</td><td><strong>57.14%</strong></td><td>视频专用的 query-aware 片段检索优于纯文本图检索。</td></tr>
    </tbody>
  </table>
</div>

这个结果说明：把视频先 caption/transcript 化再套文本 RAG 是不够的。视频场景下，视觉 embedding 与 query-aware 视觉描述能提供额外增益。

### 2. 相比长视频理解模型：RAG 化路线更适合超长视频
论文还比较了 LLaMA-VID、VideoAgent、NotebookLM 和 VideoRAG。评测采用 1 到 5 分量表，分数越高表示相对 NaiveRAG 越好。

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>Overall Score</th><th>特点</th></tr></thead>
    <tbody>
      <tr><td>LLaMA-VID</td><td>2.44</td><td>直接处理视频帧，受显存和上下文限制明显。</td></tr>
      <tr><td>VideoAgent</td><td>1.98</td><td>依赖外部工具和短视频级分析，跨多个长视频较弱。</td></tr>
      <tr><td>NotebookLM</td><td>3.37</td><td>擅长 transcript 分析，但视觉信息利用不足。</td></tr>
      <tr><td><strong>VideoRAG</strong></td><td><strong>4.45</strong></td><td>同时利用跨视频图谱、ASR、视觉 caption 和多模态检索。</td></tr>
    </tbody>
  </table>
</div>

我认为这是本文最有价值的结论之一：面对 100+ 小时视频知识库，**“检索增强的视频知识系统”比“直接把长视频喂给 VLM”更现实**。直接处理帧会遇到显存、token 和时间成本瓶颈，而索引-检索-生成架构更容易扩展到海量视频。

## Ablation：哪些模块最关键？
<figure class="figure">
  <img src="assets/papers/videorag/ablation_study.png" alt="VideoRAG 消融实验" loading="lazy" />
  <figcaption>图 2：VideoRAG 消融实验。移除 graph-based index-retrieval 或 visual indexing/retrieval 都会导致性能下降，说明图结构和视觉通道都不可替代。</figcaption>
</figure>

论文主要比较了两个消融版本：

- **-Graph**：移除图结构索引与检索，只保留非图式检索能力。
- **-Vision**：移除视觉索引和视觉检索，只依赖文本化知识。

结果显示，两者都会明显退化。我的理解是：

- 图结构负责跨视频概念合并和关系推理，解决“信息散落在多个视频里”的问题；
- 视觉 embedding 负责保留 caption 无法完整表达的画面信息，解决“问题问到具体视觉细节”的问题；
- 两者结合才构成视频 RAG 的完整能力。

## Case Study：从 12 Days of OpenAI 中定位 graders
论文用 OpenAI “12 Days of OpenAI” 系列做案例，问题是：解释 reinforcement fine-tuning 中 graders 的目的和功能。

VideoRAG 能从 12 个视频中定位到 Day 2 “Reinforcement Fine-Tuning” 里 10:00 到 12:00 的连续片段，并生成关于 graders 的详细说明，包括：grader 如何比较模型输出与标准答案、如何给 0 到 1 的分数、如何支持 partial credit，以及这些分数如何反馈到强化微调过程中。

<figure class="figure">
  <img src="assets/papers/videorag/openai-1.png" alt="VideoRAG 检索到的 OpenAI 案例片段 1" loading="lazy" />
  <figcaption>图 3：案例中检索到的关键片段之一，来自 OpenAI Day 2 reinforcement fine-tuning 相关视频。</figcaption>
</figure>

<figure class="figure">
  <img src="assets/papers/videorag/openai-2.png" alt="VideoRAG 检索到的 OpenAI 案例片段 2" loading="lazy" />
  <figcaption>图 4：VideoRAG 利用连续片段中的视觉与文本证据，支撑对 graders 功能的解释。</figcaption>
</figure>

<figure class="figure">
  <img src="assets/papers/videorag/openai-3.png" alt="VideoRAG 检索到的 OpenAI 案例片段 3" loading="lazy" />
  <figcaption>图 5：该案例展示了跨系列视频定位目标知识点的能力，而不是只在单个视频内做检索。</figcaption>
</figure>

这个案例很好地说明了 VideoRAG 的适用场景：用户可能只知道一个概念词，但不知道它在哪个视频、哪个时间段出现。系统需要先跨视频定位，再综合相邻片段生成答案。

## 我的理解与启发
VideoRAG 对我最大的启发是：多模态 RAG 的难点不只是“如何检索图片/视频”，而是 **如何为多模态内容建立适合推理的知识结构**。

传统文本 RAG 的基本单位是 chunk；但在视频里，合理单位可能同时包括：

- 时间片段；
- 视觉 caption；
- ASR transcript；
- 实体和关系；
- 原始视觉 embedding；
- query-aware 的二次片段描述。

这些单位对应不同层级的信息：clip 解决定位，caption/transcript 解决语义，知识图谱解决跨视频连接，视觉 embedding 解决画面细节。VideoRAG 的价值就在于把这些层级组织成一个可检索系统。

对实际产品来说，这很适合以下场景：

- 企业内部培训视频、会议录屏、技术分享的知识问答；
- 教育课程合集的跨章节检索与答疑；
- 媒资库、纪录片、长节目中的内容定位；
- 视频知识库与文档知识库统一接入的多模态搜索系统。

## 局限与待观察点
- **评测依赖 LLM judge**：GPT-4o-mini 打分具备可扩展性，但仍可能存在偏好、位置偏差和解释不稳定问题。
- **索引成本不低**：VideoRAG 需要 VLM caption、ASR、LLM 抽取实体关系、多模态 embedding 和 query-time 重 caption，离线与在线成本都需要工程优化。
- **视频时间结构还可以更强**：论文主要用 clip 和图谱组织语义，但复杂事件的时间因果、前后依赖和长程演化仍有提升空间。
- **caption 质量影响上限**：如果 VLM 对片段描述不准确，后续图谱构建和检索都会被污染。
- **Benchmark 仍偏知识问答**：LongerVideos 覆盖课程、纪录片和娱乐内容，但对实时交互、监控、机器人等场景的迁移还需要验证。

## 结论
VideoRAG 给出了一条很务实的长视频理解路线：不要试图让模型直接处理无限长度视频，而是把视频转化为 **图结构文本知识 + 多模态视觉索引**，再通过问题驱动的混合检索定位相关片段。它的贡献不仅是提出一个视频 RAG 框架，也是在说明未来多模态知识库需要同时保留文本语义、视觉细节和跨内容关系。对于需要管理大量视频资产的应用来说，这种架构比单纯扩展 VLM 上下文更可落地。