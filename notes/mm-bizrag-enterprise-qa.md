---
id: "mm-bizrag-enterprise-qa"
title: "MM-BizRAG：面向企业问答的结构感知多模态 RAG"
category: "多模态 RAG"
date: "2026-06-11"
order: 9
readTime: "16 min"
tags: ["MM-BizRAG","Multimodal RAG","Enterprise QA","Document AI","Layout Parsing","FastRAGEval","ColPali","VisRAG"]
summary: "MM-BizRAG 反思了近年多模态 RAG 过度依赖页面截图和视觉 embedding 的趋势，提出按文档结构区分报告类纵向文档与幻灯片类横向文档：前者显式做版面解析、表格/图片 artifact 转换和占位符对齐，后者保留页面级整体语义。它在 SlideVQA、FinRAGBench-V 和内部企业数据上优于 ColPali、VisRAG 等视觉中心基线，最高提升 32 个百分点，并提出单次 LLM Judge 的 FastRAGEval 降低评估成本。"
---

## 一句话总结
MM-BizRAG 的核心观点是：**企业文档 RAG 不能简单把每页都当图片丢给多模态 embedding 和 VLM**。报告、财报、政策、技术文档这类纵向文档里，表格、图、段落、脚注和跨页上下文往往有明确结构；如果只依赖页面截图，模型很容易丢掉阅读顺序和 artifact 对齐关系。论文因此提出结构感知 ingestion：纵向文档走显式 layout-aware parsing，横向幻灯片走页面级整体表示，再在推理时把检索表示和生成上下文解耦组装，从而在不微调模型的前提下显著提升企业问答质量。

<div class="metric-grid">
  <div class="metric"><strong>32%</strong><span>相对视觉中心基线的最高提升点数</span></div>
  <div class="metric"><strong>1,908</strong><span>内部企业数据集问题数</span></div>
  <div class="metric"><strong>20,429</strong><span>内部企业数据集页面数</span></div>
  <div class="metric"><strong>0.808</strong><span>FastRAGEval 与人工评分 Pearson 相关</span></div>
</div>

## 论文信息
- **标题**：MM-BizRAG: Rethinking Multimodal Retrieval-Augmented Generation for General Purpose Enterprise Q&A
- **作者**：Hanoz Bhathena, Parin Rajesh Jhaveri, Rohan Mittal, Prateek Singh, Aymen Kallala, Rachneet Kaur, Yiqiao Jin, Zhen Zeng, Adwait Ratnaparkhi, Denis Kochedykov
- **机构**：JPMorgan Chase & Co.; Georgia Institute of Technology
- **版本**：arXiv:2606.04231，2026-06-02；ACL 2026 Industry Track
- **链接**：[arXiv](https://arxiv.org/abs/2606.04231) · [PDF](https://arxiv.org/pdf/2606.04231)

<figure class="figure">
  <img src="assets/papers/mm-bizrag/structure-aware-ingestion.webp" alt="MM-BizRAG 结构感知文档摄取流程" loading="lazy" />
  <figcaption>图 1：MM-BizRAG 的结构感知 ingestion。系统先判断文档是纵向报告类还是横向幻灯片类，再分别采用不同的解析、切块和表示策略。</figcaption>
</figure>

## 背景：为什么企业多模态 RAG 不能只靠页面截图？
近年的多模态文档 RAG 有一个明显趋势：减少复杂解析，直接把每一页渲染成图片，用视觉语言模型或多模态 embedding 做检索与生成。这类方法很简洁，也避免了传统 OCR / layout parser 的工程复杂度，但在企业文档中会暴露几个问题：

- **结构信息被隐式化**：表格、图表、段落和标题之间的顺序关系被压到一张图里，检索器未必能学会精确利用。
- **报告类文档文本密度高**：财报、合同、政策文档往往包含大量段落、跨页引用和密集表格，页面截图检索容易被视觉噪声稀释。
- **幻灯片和报告的阅读逻辑不同**：幻灯片通常一页就是一个整体语义单元，而报告更依赖自上而下的自然阅读顺序。
- **检索表示不等于生成上下文**：适合建索引的 chunk 可能很短、很干净，但适合生成的上下文需要恢复表格 markdown、原图、图片描述和周边文本。

MM-BizRAG 的切入点就是把这些差异显式建模：**不要用一种统一的页面图像策略处理所有文档，而是让文档结构决定 ingestion 和上下文组装方式**。

## 方法：结构感知的企业文档 RAG
### 1. 先按结构把文档分成纵向与横向
论文将文档分成两类：

- **纵向文档（Vertical）**：报告、财务文件、法律文档、政策、技术说明等，通常有稳定的阅读顺序和高文本密度。
- **横向文档（Horizontal）**：PPT / slide deck 等，页面内部是自由布局，文字、图表、表格共同构成单页语义。

这个分类可以由 LLM 根据前一两页图像判断，也可以由文件元数据确定。附录中作者构建了 517 个文档的分类测试集，分类器 precision 为 **100.00**，recall 为 **83.28**，F1 为 **90.87**。

### 2. 纵向文档：显式解析 text / table / picture
对于纵向文档，MM-BizRAG 使用 layout-aware parsing 提取：

- **Text Representation**：按自然阅读顺序串联文本块，并在表格和图片原位置插入唯一 placeholder。
- **Table Representation**：将表格转成 markdown，再由 LLM 生成逐行、结构化的文本描述。
- **Picture Representation**：用 VLM 为图、图表、流程图等生成描述，并过滤 logo、装饰图等无信息图片。
- **Page Images**：保留整页图片，便于后续需要视觉证据时组装上下文。

这种设计的关键是 **placeholder-based positional alignment**：表格和图片不是被抽离成孤立对象，而是保留与原文位置的对应关系。生成时可以把 markdown、图片和描述插回原来的文本位置。

<figure class="figure">
  <img src="assets/papers/mm-bizrag/chunk-types.webp" alt="MM-BizRAG 不同 chunk 类型处理方式" loading="lazy" />
  <figcaption>图 2：纵向文档中，文本 chunk、表格 chunk 和图片 chunk 会通过 placeholder 与父文本块对齐，最终恢复成 interleaved text-image-table 上下文。</figcaption>
</figure>

### 3. 横向文档：保留页面级整体语义
对于幻灯片，论文没有强行拆出每个元素。原因是 slide 的语义经常依赖整页布局：标题、图、表格、箭头、色块和注释共同表达一个观点。强拆成局部元素反而可能破坏语义。

因此横向文档的做法是：每页保留页面图片，并用 VLM 生成详细 slide-level description。这个描述包含页面文本、视觉元素、表格、图表及其相互关系。换句话说，横向文档更像 **page-level multimodal unit**。

### 4. 三种 MM-BizRAG 变体
论文设计了三个变体，用来隔离 ingestion 表示和 embedding 策略对最终 RAG 效果的影响。

<figure class="figure">
  <img src="assets/papers/mm-bizrag/variants.webp" alt="MM-BizRAG 三种检索表示与 embedding 变体" loading="lazy" />
  <figcaption>图 3：三种变体的差别主要在 chunk 粒度、文本/多模态 embedding 选择，以及 artifact 是在 ingestion 阶段合成还是推理阶段再组装。</figcaption>
</figure>

<div class="table-wrap">
  <table>
    <thead><tr><th>变体</th><th>核心思路</th><th>检索表示</th><th>适合场景</th></tr></thead>
    <tbody>
      <tr><td>TCTE</td><td>Token-level chunking + Text embedding</td><td>文本块、表格描述、图片描述；稀疏+稠密混合检索并 RRF 融合</td><td>生产推荐，效果接近最优但延迟更低</td></tr>
      <tr><td>PCMHE</td><td>Page-level chunking + Multimodal page hybrid embedding</td><td>页面文本 chunk 与页面图片 chunk 分别用多模态 embedding 检索，再融合排名</td><td>页面级视觉信号很强的场景</td></tr>
      <tr><td>TCMIE</td><td>Token-level chunking + Multimodal interleaved embedding</td><td>把文本、表格 markdown、图片 artifact 组合成多模态 chunk 后再 embedding</td><td>希望 ingestion 阶段就形成多模态单元的场景</td></tr>
    </tbody>
  </table>
</div>

作者最后推荐 **TCTE** 作为生产配置：它在 vertical documents 上的召回通常只比最优配置低 **1–3 个百分点**，但平均延迟约为 PCMHE 的一半。

## 推理：把检索表示和生成上下文解耦
MM-BizRAG 最值得借鉴的工程点，是 **inference-time multimodal assembly**。检索阶段可以用更轻量、更稳定的表示，例如文本 chunk、表格描述、图片描述；但生成阶段不只把这些检索 chunk 原样塞给模型，而是根据 placeholder 和元数据恢复更丰富的上下文。

<figure class="figure">
  <img src="assets/papers/mm-bizrag/inference-pipeline.webp" alt="MM-BizRAG 推理阶段检索、重排和答案生成流程" loading="lazy" />
  <figcaption>图 4：推理流程包含 query rewriting、混合检索、LLM list-wise reranking，以及把文本、表格、图片重新组装后送入多模态生成模型。</figcaption>
</figure>

以 TCTE 为例：

1. Query rewriter 根据对话历史改写问题。
2. Hybrid retrieval 同时做 dense embedding 检索和 BM25 检索，并用 RRF 融合。
3. 初筛取 70 个 dense chunk、100 个 BM25 chunk，融合后取 top 30。
4. LLM list-wise reranker 对 30 个 chunk 排序，选 top 20。
5. 如果检索到表格或图片描述，系统会找到其对应父文本 chunk。
6. 在父文本 chunk 的 placeholder 位置插入表格 markdown、图片、描述等 artifact。
7. 去重、按文档顺序重排，构造 interleaved multimodal prompt 给 GPT-4.1 生成答案。

<figure class="figure">
  <img src="assets/papers/mm-bizrag/mm-reader-view.webp" alt="MM-BizRAG 多模态阅读器视图" loading="lazy" />
  <figcaption>图 5：多模态生成上下文会把文本、表格、图片按原始位置重组，而不是只把检索命中的短文本片段交给模型。</figcaption>
</figure>

这个设计给我的启发是：**检索索引应该为召回效率服务，生成上下文应该为可读性、证据完整性和 grounding 服务**。两者目标不同，不应强行共用同一种表示。

## FastRAGEval：更便宜的细粒度生成召回评估
论文还提出了 FastRAGEval（FRE），用于评估长答案企业问答。传统 EM、BLEU、ROUGE 这类 token 级指标不适合长答案；RAGChecker 会先把答案拆成 atomic claims，再分别评估 precision / recall，但需要两次 LLM 调用。

FastRAGEval 的做法是在一次 LLM 调用中完成：

- 将 reference answer 拆成 atomic facts，检查生成答案覆盖了多少，得到 recall。
- 将 generated answer 拆成 atomic facts，检查是否被 reference 支持，得到 precision。
- 用 precision 和 recall 计算 F1。

<figure class="figure">
  <img src="assets/papers/mm-bizrag/fast-rag-eval.webp" alt="FastRAGEval 细粒度生成召回评估流程" loading="lazy" />
  <figcaption>图 6：FastRAGEval 在单次 LLM Judge 调用中完成事实拆解、precision、recall 和 F1 计算，相比 RAGChecker 降低成本和延迟。</figcaption>
</figure>

在 200 个人工标注样本上，FRE-Recall 与人工判断的相关性优于 RAGChecker：Pearson **0.808 vs. 0.748**，Spearman **0.808 vs. 0.736**，Kendall **0.808 vs. 0.725**；两位标注者 Cohen’s kappa 为 **0.966**。

## 数据与实验设置
论文评测了三个数据源：

<div class="table-wrap">
  <table>
    <thead><tr><th>数据集</th><th>规模</th><th>特点</th></tr></thead>
    <tbody>
      <tr><td>内部企业数据集</td><td>1,908 个问题、1,048 个文档、20,429 页</td><td>覆盖 PDF、PPTX、Word、HTML、TXT，包含金融、法律、技术、政策等 30 个业务域</td></tr>
      <tr><td>SlideVQA</td><td>1,652 个问题、300 个文档、60,000 页</td><td>面向 presentation-style 文档，测试横向 slide ingestion</td></tr>
      <tr><td>FinRAGBench-V</td><td>539 个问题、213 个 PDF、11,432 页</td><td>金融报告类纵向文档，覆盖 text、table、picture / chart 模态</td></tr>
    </tbody>
  </table>
</div>

主要基线包括：

- **Text-Only RAG**：OCR 后做句边界保留的 token chunking，使用同一 text embedding 模型。
- **ColPali**：视觉文档检索代表方法。
- **VisRAG**：基于页面图像的视觉 RAG pipeline。

模型方面，论文使用 `text-embedding-3-large` 作为文本 embedding，`cohere-embed-v4` 和 `nomic-multimodal-embed-3b` 作为多模态 embedding，Docling 做纵向文档 layout parsing，GPT-4.1 系列用于描述生成、query rewrite、rerank、answer generation 和评估。

## 实验结果：结构感知优于纯视觉中心路线
### 1. 总体结果
<div class="table-wrap">
  <table>
    <thead><tr><th>系统</th><th>SlideVQA FRE</th><th>FinRAGBench-V FRE</th><th>Internal FRE</th><th>平均延迟（Slide / Fin / Internal）</th></tr></thead>
    <tbody>
      <tr><td>Text-Only</td><td>67.8</td><td>60.3</td><td>83.7</td><td>5.2 / 6.4 / 7.6s</td></tr>
      <tr><td>ColPali</td><td>83.6</td><td>49.3</td><td>未评测：受模型权重访问与数据隐私限制</td><td>论文未报告</td></tr>
      <tr><td>VisRAG</td><td>78.8</td><td>46.0</td><td>未评测：受模型权重访问与数据隐私限制</td><td>论文未报告</td></tr>
      <tr><td>TCTE</td><td>87.3</td><td>80.2</td><td>88.1</td><td>34.3 / 11.9 / 11.1s</td></tr>
      <tr><td>PCMHE (Nomic)</td><td><strong>89.9</strong></td><td>79.6</td><td>87.6</td><td>28.4 / 21.2 / 18.0s</td></tr>
      <tr><td>PCMHE (Cohere)</td><td>89.1</td><td><strong>82.4</strong></td><td>87.8</td><td>28.1 / 22.7 / 22.0s</td></tr>
      <tr><td>TCMIE (Cohere)</td><td>88.2</td><td>76.9</td><td>88.0</td><td>29.5 / 11.3 / 11.3s</td></tr>
    </tbody>
  </table>
</div>

在 SlideVQA 上，MM-BizRAG 相比 ColPali 的 FRE-Recall 提升约 **3.6–6.3 个百分点**，相比 VisRAG 提升约 **8.4–11.1 个百分点**。这说明即使是幻灯片这种视觉中心方法本该擅长的场景，显式文本表示和上下文组装仍然有价值。

在 FinRAGBench-V 上差距更大：MM-BizRAG 相比 ColPali / VisRAG 在 RC 和 FRE 上提升超过 **25 个百分点**，最佳变体最高约 **32 个百分点**。这恰好支持论文核心假设：**纵向报告类文档不能完全依赖页面图像检索，layout-aware parsing 和 artifact transformation 非常关键**。

### 2. 分模态结果
<figure class="figure">
  <img src="assets/papers/mm-bizrag/modality-comparison.webp" alt="PCMHE 与 ColPali 在 FinRAGBench-V 上的分模态对比" loading="lazy" />
  <figcaption>图 7：在 FinRAGBench-V 上，PCMHE 相比 ColPali 在 text、table、chart 等模态上都有明显优势，尤其体现出结构感知解析对报告类文档的价值。</figcaption>
</figure>

内部数据集上，Text-Only 在纯文本问题上仍有竞争力，但在表格和图片问题上明显下降。MM-BizRAG 的优势在于更均衡：例如 picture 类问题中，Text-Only FRE 为 **68.96**，TCTE 为 **83.16**，PCMHE (Nomic) 为 **85.49**。

### 3. 延迟与生产取舍
论文指出，实验中的 pipeline 延迟有 **75–80%** 来自 GPT-4.1 生成阶段，并受 API rate limit 影响。对于纵向文档，PCMHE 的页面图片表示会让延迟接近 TCTE 的两倍，但 recall 只带来边际提升。因此作者推荐 TCTE：

- 效果通常距离最佳配置只有 **1–3 个百分点**。
- 纵向文档延迟显著低于 PCMHE。
- 使用文本 embedding + hybrid retrieval，更容易接入现有企业搜索系统。

## 我的理解与启发
### 1. 这篇论文反驳了“视觉文档 RAG 可以省掉 parsing”的过度乐观
ColPali、VisRAG 这类视觉中心路线很有价值，尤其适合 OCR 难、版面复杂或图片语义强的文档。但 MM-BizRAG 说明，在企业问答里，**显式结构仍然是质量和可控性的基础设施**。尤其是财报、合同、技术文档这类场景，准确恢复表格、图和段落的关系，比单纯获取页面视觉 embedding 更重要。

### 2. 检索和生成应该使用不同粒度的上下文
很多 RAG 系统默认“检索到什么，就把什么送给生成模型”。MM-BizRAG 的设计更细：检索阶段用适合召回的轻量表示，生成阶段再恢复原始 artifact 和位置关系。这对工程实践很有启发：索引不必承载所有信息，但元数据必须足够完整，才能在生成前重建证据上下文。

### 3. TCTE 是一个很现实的企业落地方案
从结果看，最 fancy 的多模态 embedding 不一定总是最适合生产。TCTE 仍以 text embedding 为核心，只是在 ingestion 和 assembly 阶段补足表格、图片和 slide 的多模态信息。这种路线更便宜、更易调试，也更容易与 BM25、权限控制、审计、引用链路结合。

### 4. FastRAGEval 的价值在于评估长答案召回
企业问答经常不是一句短答案，而是需要覆盖多个事实点的长答案。FRE 这种 atomic-fact recall 指标更贴近“答案有没有覆盖关键事实”。它不是完美评估，但相比 token overlap 更符合 RAG 系统的真实目标。

## 局限与待观察点
- **内部数据不可公开**：最能体现企业场景复杂性的 1,908 问内部数据集无法释放，复现性受限。
- **公开 benchmark 数量有限**：公共实验主要依赖 SlideVQA 和 FinRAGBench-V，且 FinRAGBench-V 只处理了 213 个英文 PDF 子集，而非完整 1,100+ 文档。
- **对比基线仍可扩展**：论文只比较 ColPali 和 VisRAG 两个开源视觉中心基线，未覆盖更多新近或商业文档智能系统。
- **成本分析仍偏高层**：虽然报告了平均延迟，但 ingestion 成本、LLM artifact 描述成本、索引膨胀、缓存策略等生产细节还可以更深入。
- **多语言与跨域泛化未充分验证**：企业知识库常包含多语言文档和跨地区监管材料，这部分仍需进一步评测。

## 结论
MM-BizRAG 给出的不是一个单点模型技巧，而是一套很工程化的企业多模态 RAG 方法论：**按文档结构决定解析策略，按检索目标决定索引表示，按生成目标重建多模态上下文**。它最重要的提醒是：页面级视觉 RAG 很优雅，但不能替代对企业文档结构的显式理解。对于需要处理财报、合同、PPT、政策、技术文档的企业 QA 系统，MM-BizRAG 的 TCTE 路线尤其值得作为强基线参考。