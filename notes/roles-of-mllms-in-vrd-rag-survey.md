---
id: "roles-of-mllms-in-vrd-rag-survey"
title: "VRD-RAG 综述：MLLM 在视觉富文档检索中的三种角色"
category: "多模态 RAG"
date: "2026-06-09"
order: 10
readTime: "16 min"
tags: ["RAG","MLLM","Multimodal Retrieval","VRD","Document AI","Survey"]
summary: "这篇 AACL-IJCNLP 2025 综述聚焦视觉富文档在 RAG 中的检索问题，将 MLLM 的作用归纳为三种角色：把视觉内容转成文本的 Captioner、统一多模态表征空间的 Embedder，以及直接编码整页图像的 End-to-End Representer，并系统比较了检索粒度、信息保真度、延迟、索引成本和可解释性之间的取舍。"
---

## 一句话总结
这篇综述讨论的是一个很实际的问题：**当知识库不再是干净的纯文本，而是 PDF、扫描件、PPT、报告、表格、图表和信息图时，RAG 应该怎么“检索”这些视觉富文档？** 论文认为，MLLM 正在把传统 OCR + 文本检索流程改造成更原生的多模态检索系统，并把已有工作归纳为三种角色：**Modality-Unifying Captioners**、**Multimodal Embedders** 和 **End-to-End Representers**。这三类方案没有绝对优劣，本质是在信息保真度、检索粒度、工程成本和在线延迟之间做选择。

<div class="metric-grid">
  <div class="metric"><strong>3 种角色</strong><span>Captioner / Embedder / Representer</span></div>
  <div class="metric"><strong>18 页综述</strong><span>AACL-IJCNLP 2025 Main Conference</span></div>
  <div class="metric"><strong>+27.4</strong><span>MLLM Embedder 在 MMEB overall 平均提升</span></div>
  <div class="metric"><strong>77.91</strong><span>VisRAG-Ret 在多 VQA 检索集上的平均 MRR@10</span></div>
</div>

## 论文信息
- **标题**：Roles of MLLMs in Visually Rich Document Retrieval for RAG: A Survey
- **作者**：Xiantao Zhang
- **机构**：Beihang University
- **会议/版本**：AACL-IJCNLP 2025 Main Conference；arXiv:2601.03262
- **链接**：[arXiv 摘要页](https://arxiv.org/abs/2601.03262) · [论文 PDF](https://arxiv.org/pdf/2601.03262)

## 背景：为什么 VRD-RAG 比普通 RAG 难？
普通 RAG 默认知识是文本：切 chunk、算 embedding、向量检索、rerank、喂给 LLM。视觉富文档（Visually Rich Documents, VRDs）打破了这个假设。PDF、报告、幻灯片、扫描表单、图表和信息图中的语义常常不是线性文本，而是由 **文字、版式、空间关系、图像、表格结构和视觉标注共同决定**。

这会带来几个典型问题：

- **OCR 脆弱**：扫描质量、字体、公式、表格和图表文字都会导致识别错误，错误还会继续传给检索和生成。
- **版式承载语义**：标题层级、左右栏、箭头、流程图、图例和表格单元格位置都可能改变含义。
- **证据分散**：一个答案可能同时依赖跨页段落、图表、脚注和表格中的数值。
- **文本代理会丢信息**：把图表压缩成一句 caption，通常会损失细粒度数值、关系和视觉对比。

因此，VRD-RAG 的核心不是“给图片加个 OCR”这么简单，而是要决定：**检索单元应该是文本片段、区域、表格、页面截图，还是跨页结构？索引里保存的是文本、向量、视觉 token，还是多种表征的组合？**

<figure class="figure">
  <img src="assets/papers/vrd-rag-mllm-survey/roles.webp" alt="MLLM 在视觉富文档 RAG 中的三种角色" loading="lazy" />
  <figcaption>图 1：论文提出的三角色框架。左侧 Captioner 将视觉元素转成文本代理；中间 Embedder 将文本和视觉内容映射到共享向量空间；右侧 Representer 直接把整页文档作为图像编码和检索。</figcaption>
</figure>

## 核心框架：MLLM 的三种角色
论文最有价值的地方，是没有把所有多模态 RAG 方法混在一起，而是按 MLLM 在 pipeline 中承担的角色来分类。

<div class="table-wrap">
  <table>
    <thead><tr><th>角色</th><th>核心做法</th><th>典型检索单元</th><th>适合场景</th><th>主要风险</th></tr></thead>
    <tbody>
      <tr><td>Modality-Unifying Captioner</td><td>把图、表、页面区域转成文本，再接入文本 RAG</td><td>caption、OCR 文本、Markdown/HTML、结构化文本</td><td>已有文本检索栈成熟，希望低成本支持多模态</td><td>视觉细节和结构关系被压缩，OCR/caption 错误级联</td></tr>
      <tr><td>Multimodal Embedder</td><td>把文本、图像、页面等映射到共享语义空间</td><td>文本向量、图片向量、融合向量</td><td>需要跨模态检索，查询和证据模态不一致</td><td>模型更大，编码延迟和索引存储显著增加</td></tr>
      <tr><td>End-to-End Representer</td><td>跳过 OCR/解析，直接对整页截图建模和检索</td><td>页面图像、整页视觉 token、late-interaction 表征</td><td>扫描件、复杂版式、OCR 不可靠的文档</td><td>粒度偏粗，定位具体证据和可解释性更难</td></tr>
    </tbody>
  </table>
</div>

## 角色一：Modality-Unifying Captioner
Captioner 路线的思路最工程化：**把非文本内容翻译成文本代理，然后复用成熟的文本 RAG 基础设施**。

在 VRD 中，这通常包括：

- 对页面或区域做 OCR，并保留标题、段落、表格、坐标等结构信息。
- 对图片、图表、流程图、UI 截图生成自然语言描述。
- 将表格、图表或页面转成 Markdown、HTML、JSON 等结构化文本。
- 将 caption 与原始文本一起送入文本 embedding、BM25、reranker 和 LLM reader。

这种方案的最大优点是 **兼容性强**。如果团队已经有稳定的文本 RAG 栈，只需要增加离线 captioning / textualization 层，就可以让 PDF、PPT 和图表进入同一套索引与召回流程。在线查询时，系统仍然是文本检索，因此延迟和调试方式都比较可控。

但它的根本问题是 **转换即压缩**。图表里的坐标轴、趋势、异常点、单位、表格单元格关系、流程图箭头方向，很难被一段自然语言完整保留。对于总结类任务，这种损失可能可以接受；但对于财务、法务、科研、医学等需要精确证据的场景，caption 的省略会直接变成幻觉或错误答案。

我的理解是，Captioner 路线适合做“把多模态文档纳入现有文本知识库”的第一步，但不能把 caption 当作原始证据的完全替代品。更稳妥的做法是：**caption 用于召回，原图/区域用于验证和 grounding**。

## 角色二：Multimodal Embedder
Embedder 路线不再强行把所有内容转成文本，而是让 MLLM 直接学习一个共享表征空间：文本查询、页面截图、图表、表格或图文混合内容都可以被编码成向量，然后通过相似度检索匹配。

这条路线的历史根源可以追溯到 CLIP、ALIGN、BLIP、SigLIP、ImageBind 等图文对齐模型。但论文强调，传统 VLM embedding 在文本密集型文档上有明显不足：VRD 往往既需要视觉理解，也需要强文本语义和细粒度阅读能力。因此，近期工作开始把更强的 MLLM 改造成 embedding 模型，例如 VLM2Vec、MM-Embed、E5-V、LLaVE、GME、LamRA、MoCa 等。

这类方法通常使用 contrastive learning，把语义相关的图文样本拉近，把负样本推远。相比 Captioner，它的好处是：

- **跨模态检索更自然**：文本问题可以直接召回相关图片、页面或图表。
- **保留视觉信号**：不必先把图像完全压缩成自然语言。
- **可替换性较好**：在工程上可以作为 embedding model 接入向量库。

实验上，论文汇总的 MMEB 结果显示，MLLM-based Embedder 相比传统多模态 embedding baseline 有明显提升：baseline overall 平均为 **36.1**，MLLM-based 平均为 **63.5**，overall 平均提升 **+27.4**；其中 VQA 子任务提升尤其大，达到 **+45.0**。

<div class="table-wrap">
  <table>
    <thead><tr><th>MMEB 维度</th><th>Baseline 平均</th><th>MLLM-based 平均</th><th>提升</th></tr></thead>
    <tbody>
      <tr><td>Classification</td><td>40.4</td><td>59.9</td><td>+19.5</td></tr>
      <tr><td>VQA</td><td>10.3</td><td>55.3</td><td>+45.0</td></tr>
      <tr><td>Retrieval</td><td>46.9</td><td>67.6</td><td>+20.7</td></tr>
      <tr><td>Grounding</td><td>52.2</td><td>80.2</td><td>+28.0</td></tr>
      <tr><td>Overall</td><td>36.1</td><td>63.5</td><td>+27.4</td></tr>
    </tbody>
  </table>
</div>

不过，Embedder 的代价也很明确：模型更大、向量维度更高、离线编码更慢、索引更重。论文提到 MM-Embed 的索引存储需求可能超过 CLIP 类模型 **5 倍以上**。所以这条路线的关键问题不是“效果会不会更好”，而是“单位成本下是否值得”。

## 角色三：End-to-End Representer
Representer 路线更加激进：**不要先 OCR，不要先做复杂版式解析，而是把整页文档当作图像输入，让模型直接学习页面级表示**。DSE、ColPali、VisRAG 是论文重点提到的代表。

这条路线的动机非常直接：在复杂文档里，传统 pipeline 的很多延迟和错误都来自 OCR、layout parsing、segment detection。既然现代 VLM 已经具备较强视觉阅读能力，那就让模型直接“看页面”。

论文附录中的延迟对比很能说明问题：

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>离线解析</th><th>离线编码</th><th>离线总计</th><th>在线编码</th><th>在线搜索</th><th>在线总计</th></tr></thead>
    <tbody>
      <tr><td>MiniCPM OCR pipeline</td><td>284 ms</td><td>28 ms</td><td>312 ms</td><td>28 ms</td><td>26 ms</td><td>54 ms</td></tr>
      <tr><td>VisRAG-Ret</td><td>—</td><td>121 ms</td><td>121 ms</td><td>28 ms</td><td>26 ms</td><td>54 ms</td></tr>
    </tbody>
  </table>
</div>

VisRAG-Ret 的单次页面编码更慢，但因为省掉了解析阶段，离线总耗时反而从 **312 ms** 降到 **121 ms**。这解释了为什么“端到端看页面”在工程上有吸引力：它不仅减少错误传播，也可能减少整条链路的总复杂度。

在检索效果上，论文汇总的多 VQA 数据集 MRR@10 结果中，End-to-End Representer 也很有竞争力：ColPali 平均 **76.54**，VisRAG-Ret 平均 **77.91**，高于 BM25 OCR 的 **65.23**，也略高于 MiniCPM OCR pipeline 的 **74.78** 和 NV-Embed-v2 OCR 的 **75.24**。

## 关键取舍：没有免费的最优解
这篇综述反复强调：三种角色不是线性替代关系，而是不同系统假设下的取舍。

### 1. 粒度 vs. 上下文
页面级 Representer 能保留全局版式和视觉上下文，但检索粒度粗。系统可能知道“答案在这一页”，却还需要二次定位具体段落、图表或单元格。

Captioner 和细粒度 Embedder 更容易做到段落、区域、表格级召回，但可能在切分时丢掉跨区域语义，例如图表标题、图例和正文描述之间的关系。

### 2. 保真度 vs. 可部署性
Captioner 最容易接入现有文本 RAG，但信息损失最大。Representer 保留原始视觉信号更多，但需要多模态索引、视觉 reranking 和 grounding 能力。Embedder 处在中间，适合跨模态召回，但成本通常更高。

### 3. 离线成本 vs. 在线体验
如果知识库更新很慢，可以接受昂贵的离线 captioning 和 embedding，以换取低在线延迟。如果文档持续更新，重 caption 和重建多模态索引的成本就会成为瓶颈。

<figure class="figure">
  <img src="assets/papers/vrd-rag-mllm-survey/latency.webp" alt="CLIP 与 UniEmb 的编码和向量搜索延迟对比" loading="lazy" />
  <figcaption>图 2：论文附录中的效率对比。MLLM-based embedder 通常带来更强表征能力，但编码延迟和搜索成本也更高，说明模型小型化、向量压缩和自适应检索非常关键。</figcaption>
</figure>

## 未来方向
论文最后提出了几个值得关注的方向，我认为都很贴近工程落地。

### 1. Adaptive retrieval units
未来系统不应该固定只按段落、区域或页面检索，而应根据任务动态选择粒度。粗粒度页面检索负责高召回，细粒度区域检索负责证据定位，必要时再做跨页聚合。

### 2. 更小、更快的 MLLM 检索模型
MLLM Embedder 和 Representer 的效果提升很明显，但成本也高。蒸馏、向量剪枝、Matryoshka 表征、多粒度 embedding、轻量化 VLM 都会成为关键。

### 3. 面向 grounding 的评测
传统文本 RAG 指标不足以评估 VRD-RAG。未来 benchmark 需要衡量：答案是否来自正确页面、正确图表、正确表格单元格；模型是否能跨页聚合证据；面对 paraphrase query 是否稳定；生成答案能否回指到可验证视觉证据。

## 我的理解与启发
这篇综述对做文档问答、企业知识库和技术资料检索很有启发。过去很多系统把 PDF 当作“可以 OCR 的文本容器”，但 VRD 的真实难点在于：**文档本身是一种二维甚至多模态的信息界面**。强行线性化会让 RAG 看起来简单，却把错误藏进了索引阶段。

如果从工程选型看，我会这样理解三类方案：

- **快速接入**：先用 Captioner，把图表和页面区域转成文本，接入现有 RAG；同时保留原图引用，避免 caption 成为唯一证据。
- **跨模态搜索**：当用户查询经常指向图、表、截图、页面布局时，引入 MLLM Embedder 或 VLM reranker。
- **复杂 PDF/扫描件**：OCR 明显不可靠时，优先尝试 ColPali / VisRAG 这类页面级 Representer，再补充细粒度 grounding。
- **高可信场景**：不要只追求 MRR 或 recall，需要把检索结果和最终答案绑定到页面区域、表格单元格或图表元素上。

## 局限与待观察点
论文也有一些天然限制。首先，它是一篇综述，很多性能和延迟数字来自不同论文与环境，硬件、实现和数据集差异会影响可比性。其次，目前 VRD-RAG 的评测仍偏 benchmark，离真实企业文档库还有差距，例如权限、版本、重复文档、扫描质量、图表规范不一致等问题。最后，用户体验层面的评估还不足：在真实产品里，答案是否正确只是底线，证据是否可读、是否能定位原文、是否支持追问和纠错同样关键。

## 结论
这篇综述的价值在于给 VRD-RAG 提供了一个清晰的系统地图：Captioner 让多模态内容进入文本 RAG，Embedder 让跨模态检索成为可能，Representer 则试图直接保留页面视觉结构。未来真正可用的视觉富文档 RAG 很可能不是三选一，而是 **页面级召回 + 区域级定位 + 多模态 rerank + 可验证 grounding** 的组合系统。