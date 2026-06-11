---
id: "qwen3-vl-embedding-reranker"
title: "Qwen3-VL-Embedding / Reranker：统一多模态检索与排序框架"
category: "多模态 RAG"
date: "2026-06-11"
order: 10
readTime: "14 min"
tags: ["Qwen3-VL","Multimodal Retrieval","Embedding","Reranker","MRL","QAT","MMEB-V2"]
summary: "Qwen3-VL-Embedding 与 Qwen3-VL-Reranker 基于 Qwen3-VL 构建统一多模态检索流水线，覆盖文本、图像、视觉文档和视频，在 MMEB-V2 上 8B Embedding 达到 77.8，总体排名领先，并通过 MRL 与 QAT 兼顾效果和部署成本。"
hero: "assets/papers/qwen3-vl-embedding/performance-comparison.png"
---

## 一句话总结
Qwen3-VL-Embedding / Reranker 的核心价值，是把 **VLM 的多模态理解能力** 转成可落地的检索系统能力：Embedding 负责把文本、图像、视觉文档、视频映射到统一向量空间做大规模召回，Reranker 再用 cross-encoder 对候选 query-document 对做细粒度相关性判断。论文的重点不只是模型效果 SOTA，而是给出了从数据合成、难负例挖掘、多阶段训练、reranker 蒸馏，到 MRL / QAT 部署优化的一整套多模态检索范式。

<div class="metric-grid">
  <div class="metric"><strong>77.8</strong><span>Qwen3-VL-Embedding-8B 在 MMEB-V2 总分</span></div>
  <div class="metric"><strong>78</strong><span>MMEB-V2 覆盖的数据集数量</span></div>
  <div class="metric"><strong>32K</strong><span>Embedding / Reranker 支持的上下文长度</span></div>
  <div class="metric"><strong>30+</strong><span>继承 Qwen3-VL 的多语言能力</span></div>
</div>

## 论文信息
- **标题**：Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking
- **作者**：Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, Jingren Zhou, Junyang Lin
- **机构**：Tongyi Lab, Alibaba Group
- **版本**：arXiv:2601.04720v2，2026-01-19
- **链接**：[arXiv](https://arxiv.org/abs/2601.04720) · [PDF](https://arxiv.org/pdf/2601.04720)

<figure class="figure">
  <img src="assets/papers/qwen3-vl-embedding/performance-comparison.png" alt="Qwen3-VL-Embedding 与 Reranker 性能总览" loading="lazy" />
  <figcaption>图 1：论文首页给出的性能总览。Qwen3-VL-Embedding / Reranker 面向图文、视频、视觉文档和文本检索统一建模。</figcaption>
</figure>

## 背景：为什么多模态检索需要新的统一框架？
传统检索系统通常围绕文本 embedding 设计，但真实业务里的内容已经远不止文本：商品图、截图、表格、PPT、扫描文档、短视频、长视频片段都可能成为 query 或 document。只做文本检索会遇到几个明显瓶颈：

- **模态割裂**：图像检索、视频检索、OCR 文档检索往往各自训练模型，系统复杂且难以共享语义空间。
- **视觉文档很难只靠 OCR**：发票、论文页面、幻灯片、网页截图中，版式、图表和文字共同决定语义。
- **视频检索需要时空理解**：不只是匹配一帧画面，还要理解动作、事件和 moment。
- **召回与排序目标不同**：embedding 适合大规模 ANN 召回，但 query-document 的细粒度相关性判断更适合 cross-encoder reranker。
- **部署成本不可忽略**：大规模检索要存储海量向量，维度和量化精度直接影响索引成本与延迟。

这篇论文的出发点很明确：基于 Qwen3-VL 这样强多模态基础模型，训练一套 **统一的多模态 embedding + reranker**，同时覆盖召回、排序、长上下文、多语言和低成本部署。

<figure class="figure">
  <img src="assets/papers/qwen3-vl-embedding/unified-space.png" alt="统一多模态表示空间示意图" loading="lazy" />
  <figcaption>图 2：Qwen3-VL-Embedding 将文本、图像、视觉文档和视频映射到统一语义空间，使跨模态内容可以直接比较相似度。</figcaption>
</figure>

## 方法：Embedding + Reranker 的双模块设计
### 1. Qwen3-VL-Embedding：bi-encoder 负责大规模召回
Embedding 模型采用 bi-encoder 架构。给定一段文本、图片、视觉文档、视频或它们的组合，模型按 Qwen3-VL 的对话格式输入，并在末尾追加 `PAD` token，取该 token 对应的最后一层 hidden state 作为 dense vector。

默认 instruction 是 `Represent the user's input.`，但模型也支持 instruction-aware 表示，即可以通过任务指令定义“什么叫相关”。这点对实际检索非常重要：同一张图片可以按颜色、物体、风格、OCR 文本或业务规则进行不同方式的匹配。

### 2. Qwen3-VL-Reranker：cross-encoder 负责精排
Reranker 使用 cross-encoder 架构，把 instruction、query 和 document 放到同一个上下文中，让模型通过 cross-attention 做细粒度交互。它最终预测 `yes` / `no`，表示 document 是否满足 query 与 instruction 定义的相关性。

推理时，相关性分数为：

\[
s = \mathrm{sigmoid}(\mathrm{logit}(\text{yes}) - \mathrm{logit}(\text{no}))
\]

这类设计比单纯 cosine similarity 更慢，但更适合放在召回后的 top-K 精排阶段。

<figure class="figure">
  <img src="assets/papers/qwen3-vl-embedding/architecture.png" alt="Qwen3-VL-Embedding 与 Qwen3-VL-Reranker 架构" loading="lazy" />
  <figcaption>图 3：Embedding 采用 bi-encoder 输出向量并计算 cosine similarity；Reranker 采用 cross-encoder，对 query-document pair 输出相关性分数。</figcaption>
</figure>

### 3. 模型规格：2B / 8B 两档
论文发布 2B 和 8B 两个尺寸，面向不同部署预算。

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>Size</th><th>Layers</th><th>上下文长度</th><th>Embedding 维度</th><th>MRL</th><th>QAT</th><th>Instruction-aware</th></tr></thead>
    <tbody>
      <tr><td>Qwen3-VL-Embedding</td><td>2B</td><td>28</td><td>32K</td><td>2048</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
      <tr><td>Qwen3-VL-Embedding</td><td>8B</td><td>36</td><td>32K</td><td>4096</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
      <tr><td>Qwen3-VL-Reranker</td><td>2B</td><td>28</td><td>32K</td><td>—</td><td>—</td><td>—</td><td>Yes</td></tr>
      <tr><td>Qwen3-VL-Reranker</td><td>8B</td><td>36</td><td>32K</td><td>—</td><td>—</td><td>—</td><td>Yes</td></tr>
    </tbody>
  </table>
</div>

## 数据：围绕多模态、多任务、多领域构造相关性数据
论文把每个子数据集定义成四元组 \((I, Q, C, R)\)：instruction 定义任务目标，query 和 corpus 可以是文本、图像、视频或多模态组合，relevance labels 则给出正负相关关系。

数据构建有两个关键点：

- **数据合成**：用 Qwen3-VL-32B 对图像和视频生成多任务标注，覆盖分类、QA、检索、moment retrieval 等任务。
- **正例精炼与难负例挖掘**：先用 embedding 模型召回 top-K，再过滤低质量正例，并选择与正例相近但不应相关的 hard negatives。

<figure class="figure">
  <img src="assets/papers/qwen3-vl-embedding/data-distribution.png" alt="数据合成种子池分布" loading="lazy" />
  <figcaption>图 4：数据合成种子池的分布。论文通过类别重平衡缓解公开与内部数据在模态、任务和领域上的不均衡。</figcaption>
</figure>

### 图像与视频任务
图像侧覆盖：

- **Image Classification**：图像 + 分类指令，对应类别标签。
- **Image QA**：图像 + grounded question，对应答案。
- **Image Retrieval**：文本 query 检索候选图像，query 可以从直接描述到抽象语义、组合逻辑、知识定位。

视频侧覆盖：

- **Video Classification**：动作、场景、事件、情绪/意图等类别。
- **Video QA**：事实识别、时间定位、主题推理、影视分析等。
- **Video Retrieval**：实体、动作、事件、情绪、教程等多粒度检索。
- **Moment Retrieval**：定位视频中的具体片段，并构造时间上明显不同的负例。

## 训练策略：三阶段把召回、精排和蒸馏串起来
论文最重要的工程设计之一，是多阶段训练流水线。它不是直接把所有数据混在一起训练，而是让模型能力和数据质量互相 bootstrapping。

<figure class="figure">
  <img src="assets/papers/qwen3-vl-embedding/training-pipeline.png" alt="Qwen3-VL-Embedding 与 Reranker 多阶段训练流水线" loading="lazy" />
  <figcaption>图 5：多阶段训练流水线。先用弱监督大规模数据预训练，再用高质量数据多任务训练，最后用 Reranker 蒸馏并做模型合并。</figcaption>
</figure>

### Stage 1：Contrastive Pre-training
第一阶段用大规模、多模态、多任务合成数据训练 embedding 模型。数据通过已有开源 embedding 模型挖掘，目标是让模型先获得基础的跨模态相关性理解能力，得到 `s0`。

### Stage 2：Multi-task Contrastive Learning + Reranker SFT
第二阶段混合 curated public datasets、内部高质量数据和采样合成数据。此时使用 Stage 1 得到的 `s0` 做更强的数据挖掘，再针对不同任务类型设计 contrastive objective，得到 `s1`。

同时，论文用 retrieval-specific 子集训练 Qwen3-VL-Reranker，覆盖 image retrieval、video retrieval、moment retrieval 和 visual document retrieval。

### Stage 3：Reranker Distillation + Model Merging
第三阶段用训练好的 reranker 给 query 和候选 document 生成细粒度 relevance logits，再让 embedding 模型对齐 reranker 的分布。这一步得到的 `s2` 在检索任务上提升明显，但 classification / QA 会略降。

因此最终又把 `s2` 与 `s1` 做模型合并，得到更均衡的 `s3`。

<div class="table-wrap">
  <table>
    <thead><tr><th>阶段</th><th>Image Overall</th><th>Video Overall</th><th>VisDoc Overall</th><th>All</th><th>解读</th></tr></thead>
    <tbody>
      <tr><td>s0</td><td>65.8</td><td>57.5</td><td>74.8</td><td>66.6</td><td>大规模弱监督预训练建立基础能力</td></tr>
      <tr><td>s1</td><td>74.8</td><td>60.3</td><td>77.1</td><td>72.1</td><td>高质量多任务训练带来整体提升</td></tr>
      <tr><td>s2</td><td>71.3</td><td>59.5</td><td>80.9</td><td>71.5</td><td>reranker 蒸馏强化检索，但部分任务下降</td></tr>
      <tr><td>s3</td><td><strong>75.0</strong></td><td><strong>61.9</strong></td><td>79.2</td><td><strong>73.2</strong></td><td>模型合并后最均衡</td></tr>
    </tbody>
  </table>
</div>

## 训练目标：为不同任务定制 loss
### 1. Retrieval：InfoNCE + hard negatives
检索数据使用 InfoNCE。正样本是 \((q_i, d_i^+)\)，负样本包括 hard negatives、in-batch queries 和 in-batch documents。论文还引入 mask 避免把潜在 false negative 强行推远。

Stage 2 中，作者进一步去掉 query-query 和 document-document 对比项，实验上更适合高质量多模态检索数据。

### 2. Classification：把类别标签当 document
分类任务也被转成对比学习：待分类实例是 query，类别标签是 document。负样本只来自同一 query 的错误标签，避免其他 batch label 引入 false negative。

### 3. STS：CoSent 保持相似度排序
语义文本相似度任务没有天然 query-document 方向，论文使用 CoSent loss，让 embedding cosine similarity 保持标注相似度的排序关系。

### 4. Distillation：让 embedding 学 reranker 的排序分布
蒸馏阶段，对每个 query 准备一个正例和多个负例，reranker 离线生成 relevance logits，embedding 模型在线计算 cosine similarity，并用 cross-entropy 对齐两者的候选分布。

## 部署友好性：MRL 与 QAT
大规模检索的瓶颈不只是模型推理，还包括向量存储、ANN 搜索延迟和低精度表示后的效果损失。论文因此把 **Matryoshka Representation Learning (MRL)** 和 **Quantization-Aware Training (QAT)** 纳入训练。

### MRL：同一个向量支持不同维度截断
MRL 的思想是：不仅在完整维度上训练，也在多个前缀维度上训练，使用户可以根据成本选择较低维度 embedding，而不需要重新训练模型。

### QAT：让向量对低精度更鲁棒
QAT 在训练时同时优化全精度和低精度向量，使模型生成的 embedding 对 int8 / binary 等量化格式更稳定。论文使用 LSQ，并通过 STE 传播量化操作的梯度。

<figure class="figure">
  <img src="assets/papers/qwen3-vl-embedding/mrl-qat-analysis.png" alt="MRL 与 QAT 对维度、量化、存储和延迟的影响" loading="lazy" />
  <figcaption>图 6：MRL 与 QAT 分析。维度降低会带来可控性能损失，但显著减少存储并提升检索速度；int8 基本保持效果，binary 损失更明显。</figcaption>
</figure>

论文给出的一个具体观察是：在文本检索任务中，将 embedding 维度从 1024 降到 512，只带来约 **1.4%** 性能下降，但可以减少 **50%** 存储并让检索速度约翻倍。

## 实验结果
### MMEB-V2：多模态 embedding 总体 SOTA
MMEB-V2 覆盖 Image、Video、Visual Document 三大域，共 78 个数据集。Qwen3-VL-Embedding-8B 总分达到 **77.8**，高于当时榜单上的开源与闭源模型。

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>Size</th><th>Image Overall</th><th>Video Overall</th><th>VisDoc Overall</th><th>All</th></tr></thead>
    <tbody>
      <tr><td>RzenEmbed</td><td>8B</td><td>75.9</td><td>55.7</td><td>81.3</td><td>72.9</td></tr>
      <tr><td>Seed-1.6-embedding-1215</td><td>—</td><td>78.0</td><td>67.7</td><td>82.2</td><td>76.9</td></tr>
      <tr><td>Qwen3-VL-Embedding-2B</td><td>2B</td><td>75.0</td><td>61.9</td><td>79.2</td><td>73.2</td></tr>
      <tr><td>Qwen3-VL-Embedding-8B</td><td>8B</td><td><strong>80.1</strong></td><td>67.1</td><td><strong>82.4</strong></td><td><strong>77.8</strong></td></tr>
    </tbody>
  </table>
</div>

论文中特别指出，Qwen3-VL-Embedding-8B 相比此前最佳开源模型有 **6.7%** 相对提升。值得注意的是，Video Overall 仍是相对更难的部分，即使 8B 达到 67.1，也明显低于 Image 和 VisDoc。

### Visual Document：Embedding 接近 ColPali 风格模型，Reranker 更强
视觉文档检索方面，Qwen3-VL-Embedding 的表现接近 ColPali-style 模型，而 Qwen3-VL-Reranker-8B 在 Avg 上达到 **80.3**，高于表中其他模型。

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>Size</th><th>VisRAG</th><th>VisDocOOD</th><th>Vidore-v3</th><th>JinaVDR</th><th>Avg</th></tr></thead>
    <tbody>
      <tr><td>tomoro-colqwen3-embed-8b</td><td>8B</td><td>90.2</td><td>76.8</td><td>61.6</td><td>79.2</td><td>77.7</td></tr>
      <tr><td>Qwen3-VL-Embedding-8B</td><td>8B</td><td>88.7</td><td>73.3</td><td>59.0</td><td>76.9</td><td>75.8</td></tr>
      <tr><td>Qwen3-VL-Ranker-2B</td><td>2B</td><td>90.2</td><td>72.5</td><td>60.8</td><td>80.9</td><td>76.7</td></tr>
      <tr><td>Qwen3-VL-Ranker-8B</td><td>8B</td><td><strong>91.2</strong></td><td>75.7</td><td><strong>66.7</strong></td><td><strong>83.6</strong></td><td><strong>80.3</strong></td></tr>
    </tbody>
  </table>
</div>

### 纯文本 MMTEB：多模态模型仍保持竞争力
Qwen3-VL-Embedding-8B 在 MTEB Multilingual 上 Mean(Task) 为 **67.9**，低于 Qwen3-Embedding-8B 的 70.6，但仍接近同尺寸文本 embedding 模型。这说明多模态适配并没有完全牺牲文本能力。

### Reranking：8B 相比 2B 明显提升
Reranker 使用 Qwen3-VL-Embedding-2B 先召回 top 100，再进行精排。结果显示 Reranker-8B 在多数任务上最好，整体比 Reranker-2B 高约 4.1 分。

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>MMEB-v2 Retrieval Avg</th><th>Image</th><th>Video</th><th>VisDoc</th><th>MMTEB Retrieval</th><th>JinaVDR</th><th>ViDoRe-v3</th></tr></thead>
    <tbody>
      <tr><td>Qwen3-VL-Embedding-2B</td><td>73.4</td><td>74.8</td><td>53.6</td><td>79.2</td><td>68.1</td><td>71.0</td><td>52.9</td></tr>
      <tr><td>Qwen3-VL-Reranker-2B</td><td>75.2</td><td>74.0</td><td>53.2</td><td>83.2</td><td>70.0</td><td>80.9</td><td>60.8</td></tr>
      <tr><td>Qwen3-VL-Reranker-8B</td><td><strong>79.2</strong></td><td><strong>78.2</strong></td><td><strong>61.0</strong></td><td><strong>85.8</strong></td><td><strong>74.9</strong></td><td><strong>83.6</strong></td><td><strong>66.7</strong></td></tr>
    </tbody>
  </table>
</div>

## Ablation：哪些设计最关键？
### 1. Reranker 蒸馏提升检索，但需要模型合并保持均衡
从阶段结果看，`s2` 的 VisDoc Overall 从 `s1` 的 77.1 提升到 80.9，但 Image Overall 从 74.8 降到 71.3。最终 `s3` 通过模型合并恢复 Image / Video 表现，并把 All 提到 73.2。这说明 reranker 蒸馏对 retrieval-centric 任务非常有效，但单独使用会改变能力分布。

### 2. MRL 是成本可调的关键
MRL 让同一个模型可以导出不同前缀维度的 embedding。对工程系统来说，这意味着可以按业务场景选择：高价值业务用高维度，高吞吐低成本业务用低维度，而不是维护多套模型。

### 3. Int8 比 binary 更稳
QAT 可以显著缓解低精度损失，但从分析图看，int8 更适合在效果和成本之间折中；binary 量化在维度较低时损失会明显放大，更适合极端成本敏感场景。

### 4. 视觉粒度存在边际收益递减
论文还分析了图片 token budget、视频帧数和视频空间分辨率对效果的影响。总体上资源增加会提升效果，但高到一定程度后收益变小，甚至可能因上下文过长带来轻微下降。

<figure class="figure">
  <img src="assets/papers/qwen3-vl-embedding/visual-granularity.png" alt="视觉粒度对不同任务性能的影响" loading="lazy" />
  <figcaption>图 7：视觉粒度消融。更多视觉 token / 帧通常提升效果，但存在明显边际收益递减，过长上下文还可能带来退化。</figcaption>
</figure>

## 我的理解与启发
这篇报告最值得关注的，不是“又训练了一个多模态 embedding 模型”，而是它把多模态检索系统拆成了几个工程上非常清晰的层次：

1. **统一表示层**：用 Qwen3-VL-Embedding 统一文本、图片、视频和视觉文档，减少多模型拼接带来的系统复杂度。
2. **精排判断层**：用 Qwen3-VL-Reranker 弥补 cosine similarity 的粗粒度问题，尤其适合视觉文档、复杂 query 和业务规则相关性。
3. **数据飞轮层**：embedding 负责挖掘候选，reranker 负责蒸馏更细的相关性，模型越强，后续数据清洗和难负例挖掘越强。
4. **部署压缩层**：MRL 和 QAT 把“效果最好”转化成“不同成本下都可用”，这对真正的大规模检索比单点榜单更重要。

对 RAG 系统来说，它的启发也很直接：未来的 RAG 不应只面向文本 chunk，而应该把网页截图、PDF 页面、表格、图片、视频片段都作为可检索对象。Embedding 负责高速召回，Reranker 负责可信精排，再把结果交给多模态 LLM 生成答案，这会是一条非常自然的多模态 RAG 路线。

## 局限与待观察点
- **报告没有公开完整训练数据规模**：论文描述了数据构造流程，但对总数据量、各阶段样本规模披露有限，复现成本仍然较高。
- **视频仍是相对短板**：MMEB-V2 中 Video Overall 低于 Image / VisDoc，说明视频语义与 moment retrieval 仍更难。
- **Reranker 成本更高**：cross-encoder 精排效果强，但吞吐和延迟不适合直接全库匹配，仍依赖 embedding 先召回。
- **业务相关性仍需领域适配**：instruction-aware 提供了入口，但在电商、办公文档、搜索广告等具体场景中，还需要领域数据微调或偏好校准。
- **榜单时间点需要注意**：论文声称截至 2026-01-08 在 MMEB-V2 排名领先；后续 leaderboard 变化需要再看最新结果。

## 结论
Qwen3-VL-Embedding / Reranker 给多模态检索提供了一套完整范式：用 VLM backbone 统一多模态理解，用 bi-encoder 做高效召回，用 cross-encoder 做精细排序，用 reranker 蒸馏反哺 embedding，再用 MRL 和 QAT 解决大规模部署成本。它的意义不只是 MMEB-V2 的 77.8 分，而是把多模态 RAG、视觉文档检索和视频检索所需的核心组件整理成了一条可工程化的 pipeline。