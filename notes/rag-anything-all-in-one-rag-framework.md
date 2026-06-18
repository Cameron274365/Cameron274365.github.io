---
id: "rag-anything-all-in-one-rag-framework"
title: "RAG-Anything：把文本、图片、表格和公式都变成可检索知识实体"
category: "多模态 RAG"
date: "2026-06-11"
order: 3
readTime: "14 min"
tags: ["RAG", "Multimodal RAG", "GraphRAG", "Document QA", "Long Context", "Table", "Equation", "Retrieval"]
summary: "RAG-Anything 提出一个面向多模态文档的统一 RAG 框架：先把文本、图片、表格、公式等内容拆成原子知识单元，再用双图结构同时建模跨模态关系和文本语义，最后通过结构导航与语义匹配结合的混合检索完成长文档问答。"
hero: "assets/papers/rag-anything/x1_framework.webp"
---

## 一句话总结
RAG-Anything 的核心观点是：**多模态文档不应该被压扁成文本 chunk，而应该被表示为一组相互连接的知识实体**。它用双图结构分别保留跨模态上下文和细粒度文本语义，再用结构化图检索 + 向量语义检索的混合方式，把图片、表格、公式和文本统一纳入 RAG 证据链。实验显示，这种设计在长文档、多模态证据分散、表格/图像布局复杂的场景中更有优势。

<div class="metric-grid">
  <div class="metric"><strong>63.4%</strong><span>DocBench overall accuracy，高于 MMGraphRAG 61.0%</span></div>
  <div class="metric"><strong>42.8%</strong><span>MMLongBench overall accuracy，高于 LightRAG 38.9%</span></div>
  <div class="metric"><strong>13+ pts</strong><span>DocBench 100 页以上文档相对 MMGraphRAG 的最高优势</span></div>
  <div class="metric"><strong>4 模态</strong><span>文本、图片、表格、公式均作为一等知识单元处理</span></div>
</div>

## 论文信息
- **标题**：RAG-Anything: All-in-One RAG Framework
- **作者**：Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, Chao Huang
- **机构**：The University of Hong Kong
- **版本**：arXiv:2510.12323，2025-10-14
- **链接**：[arXiv](https://arxiv.org/abs/2510.12323) · [PDF](https://arxiv.org/pdf/2510.12323) · [代码](https://github.com/HKUDS/RAG-Anything)

<figure class="figure">
  <img src="assets/papers/rag-anything/x1_framework.webp" alt="RAG-Anything 统一多模态 RAG 框架" loading="lazy" />
  <figcaption>图 1：RAG-Anything 框架总览。系统先做多模态知识统一，将文本、图片、表格、公式拆成原子单元；再构建双图索引；查询时结合结构化图导航与语义相似度检索，并把原始视觉证据还原给 VLM 生成答案。</figcaption>
</figure>

## 背景：为什么文本 RAG 不够？
传统 RAG 默认知识库主要由纯文本组成：切块、向量化、检索、拼接上下文，然后让 LLM 回答。但真实知识库往往不是这样。论文特别强调，学术论文、金融报告、政府文件、医疗资料、技术文档中，很多关键信息存在于非文本形态：

- **图像和图表**：实验曲线、结构图、诊断影像、统计可视化可能直接承载结论。
- **表格**：财报、指标对比、消融实验和临床数据依赖行列关系，不能只按自然语言顺序读取。
- **公式**：数学表达式包含变量、定义、推导关系和理论约束。
- **版面结构**：标题、图注、子图、脚注、单元格、页间引用构成了证据定位线索。

如果把这些内容全部转成 plain text，会出现两类损失：一是视觉/空间信息丢失，二是结构关系丢失。RAG-Anything 要解决的正是这个问题：让多模态内容以结构化方式进入 RAG，而不是在进入索引前就被压扁。

## 方法：All-in-One Multimodal RAG
### 1. Multimodal Knowledge Unification：先拆成原子知识单元
RAG-Anything 首先把每个知识源分解成原子内容单元，每个单元包含模态类型和原始内容。模态可以是 text、image、table、equation 等。

这一步的关键不是“都转成文本”，而是**保留每种模态自己的结构和上下文**：

- 文本被切成段落、列表项等语义块。
- 图片会保留图像本身、图注、引用关系和周边上下文。
- 表格会保留表头、单元格、数值、行列关系。
- 公式会转成可理解的符号表示，并关联周围定义。

这样做的好处是，后续索引可以同时访问“内容本身”和“内容在文档中的位置/关系”。例如，一张图不仅是图片 embedding，也和 caption、相邻段落、子图标题、引用它的文字共同组成证据。

### 2. Dual-Graph Construction：双图建模跨模态与文本语义
论文最核心的设计是双图结构：

- **Cross-Modal Knowledge Graph**：以非文本内容为锚点，把图片、表格、公式等转成结构化实体。系统会用 MLLM 为每个非文本单元生成两类文本代理：一类是面向检索的详细描述，另一类是面向图构建的实体摘要。再从描述中抽取实体和关系，并用 `belongs_to` 等边连接回对应的多模态锚点。
- **Text-based Knowledge Graph**：对文本块使用类似 LightRAG / GraphRAG 的实体和关系抽取，捕捉纯文本中的语义结构。

之后，系统通过实体名等信号对两张图做对齐和融合，得到一个统一知识图谱。这个设计的直觉很清楚：**非文本内容需要跨模态 grounding，文本内容需要细粒度语义关系；两者不能只用一张粗糙图统一处理**。

### 3. Graph Fusion + Dense Index：图结构和向量表并存
融合后的图谱用于结构化推理，但 RAG 仍然需要语义相似度检索。因此 RAG-Anything 同时维护两类索引：

- **统一知识图谱**：存实体、关系、跨模态锚点和结构边。
- **Dense embedding table**：为实体、关系和原子内容 chunk 生成向量表示。

这相当于把 GraphRAG 和 dense retrieval 的优势放在一起：图适合找显式关系、多跳路径和结构约束；向量适合找语义相似但图上未直接相连的内容。

### 4. Cross-Modal Hybrid Retrieval：结构导航 + 语义匹配
查询阶段，系统先做 modality-aware query analysis。例如 query 中出现 “figure / chart / table / equation” 这类词时，会提示系统更关注对应模态。

随后走两条检索路径：

1. **Structural Knowledge Navigation**：根据关键词和实体识别定位图中相关节点，再扩展邻居，找出相关实体、关系和关联 chunk。它适合多跳推理、跨模态引用和版面结构导航。
2. **Semantic Similarity Matching**：用 query embedding 在 dense table 中做相似度检索，覆盖那些没有显式结构边但语义相关的证据。

最后，系统将两个候选池合并，并用多信号打分：结构重要性、语义相似度、query 推断出的模态偏好等共同决定最终排序。

### 5. From Retrieval to Synthesis：还原原始视觉证据
一个很重要但容易被忽视的细节是：RAG-Anything 不只把图片、表格、公式的文字描述交给 LLM。对于视觉证据，系统会 dereference 回原始 visual artifacts，再连同结构化文本上下文一起交给 VLM 生成答案。

这避免了“检索阶段靠文本代理，生成阶段也只能看文本代理”的信息瓶颈。文本代理用于高效检索，原始视觉内容用于最终推理。

## 数据与评测设置
论文主要在两个长文档多模态 DQA benchmark 上评测：

<div class="table-wrap">
  <table>
    <thead><tr><th>数据集</th><th>文档数</th><th>平均页数</th><th>平均 tokens</th><th>文档类型</th><th>问题数</th></tr></thead>
    <tbody>
      <tr><td>DocBench</td><td>229</td><td>66</td><td>46,377</td><td>5</td><td>1,102</td></tr>
      <tr><td>MMLongBench</td><td>135</td><td>47.5</td><td>21,214</td><td>7</td><td>1,082</td></tr>
    </tbody>
  </table>
</div>

- **DocBench**：覆盖 Academia、Finance、Government、Laws、News 五类文档，强调真实文档阅读。
- **MMLongBench**：覆盖 research reports、tutorials、academic papers、guidebooks、brochures、administration/industry files、financial reports 等长文档。

实验中，所有方法统一使用 GPT-4o-mini 作为 backbone LLM；文档解析使用 MinerU；embedding 使用 `text-embedding-3-large`；reranker 使用 `bge-reranker-v2-m3`；最终答案由 GPT-4o-mini 评估准确性。对 GPT-4o-mini 直接长上下文 baseline，论文把文档渲染成图片，最多输入 50 页。

## 实验结果
### 1. DocBench：整体最好，尤其擅长多模态问题
在 DocBench 上，RAG-Anything overall accuracy 达到 **63.4%**，高于 MMGraphRAG 的 61.0%、LightRAG 的 58.4% 和 GPT-4o-mini 直接处理的 51.2%。

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>Text-only</th><th>Multimodal</th><th>Unanswerable</th><th>Overall</th></tr></thead>
    <tbody>
      <tr><td>GPT-4o-mini</td><td>61.0</td><td>43.8</td><td>49.6</td><td>51.2</td></tr>
      <tr><td>LightRAG</td><td>85.0</td><td>59.7</td><td>46.8</td><td>58.4</td></tr>
      <tr><td>MMGraphRAG</td><td>67.6</td><td>66.0</td><td>60.5</td><td>61.0</td></tr>
      <tr><td>RAG-Anything</td><td><strong>85.0</strong></td><td><strong>76.3</strong></td><td>46.0</td><td><strong>63.4</strong></td></tr>
    </tbody>
  </table>
</div>

最明显的提升来自 Multimodal 类型问题：RAG-Anything 达到 76.3%，比 MMGraphRAG 高 10.3 个点。这说明它对表格、公式、图片等非文本内容的统一结构建模确实有效。

### 2. MMLongBench：长文档场景优势更稳定
在 MMLongBench 上，RAG-Anything overall accuracy 为 **42.8%**，高于 LightRAG 38.9% 和 MMGraphRAG 37.7%。其中 Research Reports、Guidebooks、Financial Reports 等信息密集型长文档上优势更明显。

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>Research</th><th>Tutorial</th><th>Academic</th><th>Guidebook</th><th>Financial</th><th>Overall</th></tr></thead>
    <tbody>
      <tr><td>GPT-4o-mini</td><td>35.5</td><td>44.0</td><td>24.6</td><td>33.1</td><td>31.1</td><td>33.5</td></tr>
      <tr><td>LightRAG</td><td>40.8</td><td>34.1</td><td>36.2</td><td>39.4</td><td>38.3</td><td>38.9</td></tr>
      <tr><td>MMGraphRAG</td><td>40.8</td><td>36.5</td><td>35.7</td><td>35.8</td><td>38.5</td><td>37.7</td></tr>
      <tr><td>RAG-Anything</td><td><strong>46.6</strong></td><td>43.5</td><td><strong>38.7</strong></td><td><strong>43.9</strong></td><td><strong>43.6</strong></td><td><strong>42.8</strong></td></tr>
    </tbody>
  </table>
</div>

### 3. 文档越长，双图检索越有价值
论文专门按页数分桶评估。结果显示，在短文档上 RAG-Anything 和 MMGraphRAG 差距不大；但文档越长，RAG-Anything 的优势越明显。

<figure class="figure">
  <img src="assets/papers/rag-anything/x2_length_performance.webp" alt="RAG-Anything 在不同页数文档上的性能" loading="lazy" />
  <figcaption>图 2：不同文档长度下的性能。DocBench 中，当文档超过 100 页，RAG-Anything 相对 MMGraphRAG 的优势扩大到 13 个点以上；MMLongBench 中也在 51–100 页和 101–200 页区间保持明显优势。</figcaption>
</figure>

这点很关键：长文档的难点不是“上下文放不下”这么简单，而是证据往往跨页、跨模态、跨结构分散。双图结构能够把这些分散证据通过实体、图注、表头、行列关系和文本语义连接起来。

## Ablation：哪些设计最关键？
论文做了两个主要消融：

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>Text-only</th><th>Multimodal</th><th>Overall</th><th>解释</th></tr></thead>
    <tbody>
      <tr><td>Chunk-only</td><td>81.6</td><td>66.2</td><td>60.0</td><td>不用图，只做传统 chunk 检索</td></tr>
      <tr><td>w/o Reranker</td><td>81.7</td><td>74.7</td><td>62.4</td><td>保留图结构，去掉重排序</td></tr>
      <tr><td>RAG-Anything</td><td><strong>85.0</strong></td><td><strong>76.3</strong></td><td><strong>63.4</strong></td><td>完整双图 + 混合检索 + reranker</td></tr>
    </tbody>
  </table>
</div>

结论很直接：**图结构是主要增益来源，reranker 是锦上添花**。Chunk-only overall 只有 60.0%，说明传统切块无法捕捉多模态文档中的结构关系；去掉 reranker 只下降到 62.4%，说明核心能力来自图构建和跨模态整合。

## Case Study：结构化检索到底解决了什么？
### 多面板图像理解
<figure class="figure">
  <img src="assets/papers/rag-anything/x3_visual_case.webp" alt="RAG-Anything 多面板图像理解案例" loading="lazy" />
  <figcaption>图 3：多面板图像理解案例。查询要求比较 style-space panel 中的聚类分离模式，RAG-Anything 通过面板、坐标轴、图例和 caption 的结构关系定位正确子图，避免被相邻 content-space panel 混淆。</figcaption>
</figure>

多面板图是学术论文中很常见的结构。普通检索可能只看到“这张图包含 t-SNE / cluster / style space”等词，但不知道 query 应该指向哪个 panel。RAG-Anything 将 panel、axis title、legend、caption 等建成节点和边，因此能把问题定位到正确视觉区域。

### 财务表格导航
<figure class="figure">
  <img src="assets/papers/rag-anything/x4_financial_table_case.webp" alt="RAG-Anything 财务表格导航案例" loading="lazy" />
  <figcaption>图 4：财务表格导航案例。问题要求定位 “Wages and salaries” 行与 “2020” 列的交叉单元格，RAG-Anything 通过 row-of、column-of、header-applies-to、unit-of 等关系避免混淆相近条目。</figcaption>
</figure>

财务表格最容易暴露文本化处理的缺陷。线性化后的表格常把相邻年份、相邻科目和单位混在一起。RAG-Anything 显式建模表头、行头、数据单元格和单位关系，因此能定位到目标 cell，而不是只检索到一段相似文本。

### 复杂表格结构
<figure class="figure">
  <img src="assets/papers/rag-anything/x5_tabular_case.webp" alt="RAG-Anything 复杂表格结构案例" loading="lazy" />
  <figcaption>图 5：复杂表格结构案例。RAG-Anything 能在重复行标签和多数据集表格中定位 Evidence Inference 数据集下最高 AUPRC 值 0.506，对应 “GloVe + LSTM - Attention”。</figcaption>
</figure>

这个案例说明，表格 RAG 不是“把表格 markdown 化”就够了。真正困难的是同一张表内可能有多组实验、多级表头和重复标签，必须把单元格放回行列层级中理解。

## 局限与失败案例
论文也在附录中分析了当前多模态 RAG 的两个关键失败模式。

### 1. Text-centric retrieval bias
<figure class="figure">
  <img src="assets/papers/rag-anything/x6_cross_modal_noise.webp" alt="多模态 RAG 的文本中心检索偏置失败案例" loading="lazy" />
  <figcaption>图 6：跨模态噪声案例。即使问题明确要求从图像中找答案，系统仍可能优先检索主题相近的文本证据，导致视觉证据被噪声文本覆盖。</figcaption>
</figure>

多模态 RAG 很容易出现文本偏置：当文本证据和图像证据主题相近但粒度不同，检索器可能更偏向文本，因为文本与 query 的词面匹配更强。这会让模型看似找到了相关证据，实际上错过了真正需要的视觉信息。

### 2. Ambiguous table structure
<figure class="figure">
  <img src="assets/papers/rag-anything/x7_ambiguous_table.webp" alt="多模态 RAG 的歧义表格结构失败案例" loading="lazy" />
  <figcaption>图 7：歧义表格结构案例。当表格存在合并单元格、边界不清晰或非标准布局时，当前系统仍可能错误解析行列关系，说明 layout-aware parsing 仍是瓶颈。</figcaption>
</figure>

这说明 RAG-Anything 虽然把结构建模作为核心，但它仍依赖上游解析质量。如果 MinerU、OCR 或表格解析器不能正确识别布局，后续图构建也会继承错误。

## 我的理解与启发
### 1. 多模态 RAG 的关键不是“多塞几种 embedding”
这篇论文最有价值的启发是：多模态 RAG 需要先回答“知识单位是什么”。文本 RAG 的知识单位通常是 chunk；但在多模态文档里，知识单位可能是图中的一个 panel、表格里的一个 cell、公式里的一个变量，或图注与图像之间的引用关系。

### 2. 图结构适合表达文档中的“可定位关系”
GraphRAG 在文本场景里主要解决实体关系和多跳问题；RAG-Anything 把这个思路扩展到文档结构。对多模态文档来说，很多问题的本质不是语义相似，而是精确定位：哪个子图？哪一列？哪个年份？哪个指标？这类问题天然适合结构化导航。

### 3. 文本代理和原始视觉内容要分工
我很认同论文的检索-生成分工：检索阶段可以用文字描述和 embedding 来提高效率，但生成阶段最好恢复原始视觉证据。否则，系统只是“先把图转文字，再做文本 RAG”，仍然会丢掉视觉细节。

### 4. 长文档优势来自证据连接，而不是单纯上下文长度
GPT-4o-mini 直接长上下文输入并不强，说明把很多页塞给模型不是万能的。长文档 RAG 的关键是把分散证据压缩成可导航、可验证、可组合的结构。RAG-Anything 的双图索引正是在做这件事。

## 局限与待观察点
- **上游解析质量决定下限**：如果图片、表格、公式或版面结构解析错误，图构建会把错误结构固化进索引。
- **图构建成本不低**：用 MLLM 为非文本单元生成描述和实体摘要，再抽取关系，可能在大规模企业知识库中带来较高离线成本。
- **评估仍依赖 GPT-4o-mini 判断**：自动评估方便扩展，但可能引入 judge 偏差，尤其是数值型和细粒度视觉定位问题。
- **文本偏置尚未完全解决**：论文附录也指出，多模态检索仍可能偏向文本证据；需要更强的 query-aware modality routing。
- **动态图和增量更新值得关注**：真实知识库会持续更新，双图结构如何做增量构建、实体合并和冲突处理，论文没有深入展开。

## 结论
RAG-Anything 把多模态 RAG 从“把图片/表格转成文本后继续套文本 RAG”推进到“把不同模态都建模为结构化知识实体”。它的双图构建、图与向量并存索引、结构导航和语义检索融合，为长文档、多模态证据分散、复杂版面定位提供了一条很清晰的路线。对实际工程来说，这篇论文的启发是：**多模态 RAG 的核心资产不是更长 prompt，而是高质量的结构化多模态索引**。