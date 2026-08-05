---
id: "graphdoc-graph-based-document-structure-analysis"
title: "GraphDoc：用关系图做文档结构分析"
category: "多模态 RAG"
date: "2026-07-30"
order: 1
readTime: "12 min"
tags: ["Document Layout Analysis", "Graph Structure", "Multimodal RAG", "Document Understanding", "Scene Graph", "ICLR"]
summary: "ICLR 2025 工作。传统文档布局分析（DLA）只检测元素类别和框，GraphDoc 把任务升级为图结构分析（gDSA）：同时预测元素、空间关系（上下左右）和逻辑关系（父子、顺序、引用）。基于 DocLayNet 构建 80K 页、413 万条关系标注的 GraphDoc 数据集，并提出即插即用的 DRGG 关系图生成头。实验显示 RoDLA+InternImage+DRGG 在 gDSA 上达到 mAPg@0.5=57.6，且 DLA 指标也从 80.5 提升到 81.5——结构关系预测反过来帮助了布局检测。"
---

## 一句话总结

传统文档布局分析只回答“页面上有什么”，GraphDoc 进一步追问“**这些元素之间是什么关系**”。它把文档页建模为关系图（节点是布局元素，边是空间/逻辑关系），构造了 80K 页、413 万条关系边的大型数据集，并设计了一个可插拔的关系图生成头 DRGG。**gDSA 任务不仅让模型理解文档结构，还能反哺布局检测精度**。

<div class="metric-grid">
  <div class="metric"><strong>80K</strong><span>文档页图像</span></div>
  <div class="metric"><strong>4.13M</strong><span>关系标注</span></div>
  <div class="metric"><strong>8</strong><span>种关系类型（4 空间 + 4 逻辑）</span></div>
  <div class="metric"><strong>57.6</strong><span>gDSA mAPg@0.5</span></div>
</div>

## 论文信息

- **标题**：Graph-based Document Structure Analysis
- **作者**：Yufan Chen, Ruiping Liu, Junwei Zheng, Di Wen, Kunyu Peng, Jiaming Zhang, Rainer Stiefelhagen（KIT CV:HCI Lab）
- **会议**：ICLR 2025
- **链接**：[arXiv:2502.02501](https://arxiv.org/abs/2502.02501) · [HTML](https://arxiv.org/html/2502.02501v1) · [项目主页](https://cvhci-research.github.io/GraphDoc/)
- **数据集与代码**：将公开于项目主页

## 背景：为什么需要图结构分析？

### 现有 DLA 的局限

文档布局分析（DLA）经过几年发展已经能较好地回答“页面上有什么元素”——标题、段落、表格、图片、页眉页脚等。但真实文档的理解远不止检测框和类别：

- **空间上**：哪些段落在同一栏？哪个图注属于哪张图？
- **逻辑上**：哪些段落属于同一章节？引用指向哪个图表？阅读顺序是什么？
- **结构上**：章节标题和正文如何嵌套？脚注和正文中的上标如何对应？

缺少这些关系，DLA 的输出只是一堆孤立框，难以直接支撑下游的文档问答、检索、摘要或重排。

### 从场景图到文档图

场景图生成（Scene Graph Generation）在图像理解中已经证明：把对象和关系显式建模成图，能显著提升视觉推理能力。GraphDoc 把这一思想迁移到文档领域，提出 **gDSA（graph-based Document Structure Analysis）**：要求模型同时输出元素检测和元素间关系图。

<figure class="figure">
  <img src="assets/papers/graphdoc/x1_overview.webp" alt="GraphDoc 数据集与任务概览" loading="lazy" />
  <figcaption>图 1：GraphDoc 同时支持 DLA（元素检测）和 gDSA（关系图预测）。右侧展示了 DRGG 在两个任务上的性能，说明统一建模结构关系能同时提升布局检测与关系推理。</figcaption>
</figure>

## 方法：GraphDoc 数据集与 DRGG

### 1. GraphDoc 数据集

GraphDoc 在 DocLayNet 的基础上增加了关系标注。DocLayNet 本身包含 80K 页、11 类布局元素（Caption、Footnote、Formula、List-item、Page-footer、Page-header、Picture、Section-header、Table、Text、Title）和精确的段落级边界框。GraphDoc 保留了这些 DLA 标注，并新增了两种关系图：

**空间关系图（Spatial Relations）**

用四个方向描述元素之间的相邻关系：

- **Up / Down / Left / Right**：只保留每个方向上最近的相邻元素，避免冗余。

这种设计适合 Manhattan Layout（网格状、不重叠的矩形区域），能覆盖科技文献、财报、法律条文等大多数规范文档。

**逻辑关系图（Logical Relations）**

从文档语义结构出发，定义四类关系：

<figure class="figure">
  <img src="assets/papers/graphdoc/x3_logical_relations.webp" alt="GraphDoc 中的四种逻辑关系" loading="lazy" />
  <figcaption>图 2：GraphDoc 的四种逻辑关系——Parent、Child、Sequence、Reference。它们捕捉章节层级、段落顺序、图文引用等结构化语义。</figcaption>
</figure>

- **Parent / Child**：章节标题与子标题、段落之间的从属关系。
- **Sequence**：同一父节点下子元素的自然阅读顺序。
- **Reference**：文本对图表、表格、脚注的引用关系。

### 2. 标注流程

关系标注采用规则系统 + 人工校验的混合方式：

1. **内容提取**：使用 pdfplumber 从 PDF 提取文本和位置；扫描版或加密 PDF 用 Tesseract OCR。
2. **空间关系提取**：基于 DocLayNet 不重叠的边界框，沿 x/y 轴扫描，记录每个元素在四个方向上最近的邻居。
3. **基础阅读顺序**：用 Recursive X-Y Cut 算法生成从左到右、从上到下的阅读顺序，并针对多栏布局做列优先调整。
4. **层级结构构建**：把元素分为四类——有直接结构关系的元素、非文本内容、无直接关联元素、仅引用元素。对前两类构建内部树。
5. **关系补全**：根据层级结构建立 Parent/Child，同父节点子元素按阅读顺序建立 Sequence，文本匹配建立 Reference（如脚注上标、图表引用）。

最终数据集规模如下：

<div class="table-wrap">
  <table>
    <thead><tr><th>数据集</th><th>年份</th><th>图像数</th><th>元素类别</th><th>关系类别</th><th>关系数</th><th>任务覆盖</th></tr></thead>
    <tbody>
      <tr><td>FUNSD</td><td>2019</td><td>199</td><td>4</td><td>1</td><td>7.4K</td><td>DLA, ROP, HSA</td></tr>
      <tr><td>ReadingBank</td><td>2021</td><td>500K</td><td>-</td><td>1</td><td>98.2M</td><td>ROP</td></tr>
      <tr><td>HRDoc</td><td>2023</td><td>66K</td><td>14</td><td>3</td><td>1.79M</td><td>DLA, HSA</td></tr>
      <tr><td>DocLayNet</td><td>2022</td><td>80K</td><td>11</td><td>-</td><td>-</td><td>DLA</td></tr>
      <tr><td><strong>GraphDoc</strong></td><td>2024</td><td>80K</td><td>11</td><td>8</td><td>4.13M</td><td>DLA, ROP, HSA, GSA</td></tr>
    </tbody>
  </table>
</div>

关系分布上，空间关系占比超过一半；逻辑关系中 Parent/Child 和 Sequence 居多，Reference 最少，存在长尾问题。

<figure class="figure">
  <img src="assets/papers/graphdoc/x4_relation_stats.webp" alt="GraphDoc 关系统计" loading="lazy" />
  <figcaption>图 3：GraphDoc 关系分布的弦图与热力图。Text 与 Picture、Section-header 与 Text 之间的交互最为密集。</figcaption>
</figure>

### 3. DRGG：Document Relation Graph Generator

DRGG 是一个即插即用的关系图生成头，可以接到任何基于 Encoder-Decoder 的目标检测器上（如 DETR、Deformable DETR、DINO、RoDLA）。

<figure class="figure">
  <img src="assets/papers/graphdoc/x5_architecture.webp" alt="DRGG 架构" loading="lazy" />
  <figcaption>图 4：DRGG 架构。在检测头之外新增 Relation Head，利用解码器各层的 object queries 和 object features 提取关系特征，最终预测 N×N 的关系图。</figcaption>
</figure>

DRGG 包含两个核心模块：

**关系特征提取器（Relation Feature Extractor）**

对每一解码器层 \(l\) 的 object queries \(X^0\) 和 object features \(X^l\)，分别通过两个独立的池化层和 MLP：

\[
D_1^l = \text{MLP}_p^1(P_1(X^l)), \quad D_2^l = \text{MLP}_p^2(P_2(X^l))
\]

然后上采样、再用 MLP 精炼，并与原始 object features 相加后分别沿不同轴扩展为二维，拼接得到最终关系特征 \(F^l \in \mathbb{R}^{N \times N \times 2d_{embed}}\)。这种设计同时捕捉直接（空间邻近）和间接（语义引用）关系。

**关系特征聚合（Relational Feature Aggregation）**

将各层的关系特征按可学习权重 \(\alpha^{(l)}\) 加权聚合，再通过关系预测器 \(\text{MLP}_g\) 输出 \(N \times N \times k\) 的关系图：

\[
G = \text{MLP}_g\left(\sum_{l=1}^{L} \alpha^{(l)} F^l\right)
\]

此外，DRGG 还引入了一个**辅助关系存在头（Auxiliary Relation Head）**，先用二分类判断“两个元素之间是否存在任何关系”，再与主关系分类器的输出逐元素相乘，抑制假阳性：

\[
G_{\text{final}} = G_{\text{pred}} \odot A_{\text{pred}}^{\otimes k}
\]

训练时使用匈牙利匹配（Hungarian Matching）将预测框与 GT 框对齐，再用 BCE 损失训练关系预测和辅助存在头：

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{bbox}} + \lambda \mathcal{L}_{\text{rel}} + \sigma \mathcal{L}_{\text{rel}_{\text{aux}}}
\]

## 评测指标

### DLA 任务

使用标准 COCO 风格的 mAP@[0.50:0.05:0.95]，即在不同 IoU 阈值下平均的平均精度。

### gDSA 任务

关系预测需要先匹配元素实例（IoU 阈值 0.5），再评估预测的三元组 \((subject, predicate, object)\)：

- **mRg@TR**：Mean Recall，强调召回，适合评估关系不平衡场景。
- **mAPg@TR**：Mean Average Precision，同时考虑精度和召回，\(T_R \in \{0.5, 0.75, 0.95\}\)。

## 实验结果

### 主实验

<div class="table-wrap">
  <table>
    <thead><tr><th>Backbone</th><th>Detector</th><th>Relation Head</th><th>DLA mAP@50:5:95</th><th>mRg@0.5</th><th>mAPg@0.5</th><th>mAPg@0.75</th><th>mAPg@0.95</th></tr></thead>
    <tbody>
      <tr><td>InternImage</td><td>RoDLA</td><td>-</td><td>80.5</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
      <tr><td>InternImage</td><td>DETR</td><td>DRGG</td><td>68.2</td><td>7.1</td><td>19.8</td><td>13.5</td><td>7.5</td></tr>
      <tr><td></td><td>Deformable DETR</td><td>DRGG</td><td>73.4</td><td>11.5</td><td>25.4</td><td>11.8</td><td>8.5</td></tr>
      <tr><td></td><td>DINO</td><td>DRGG</td><td>79.5</td><td>19.2</td><td>25.2</td><td>18.7</td><td>14.5</td></tr>
      <tr><td></td><td>RoDLA</td><td>DRGG</td><td><strong>81.5</strong></td><td><strong>30.7</strong></td><td><strong>57.6</strong></td><td><strong>56.3</strong></td><td><strong>46.5</strong></td></tr>
      <tr><td>ResNet</td><td>RoDLA</td><td>DRGG</td><td>71.0</td><td>13.8</td><td>45.8</td><td>17.6</td><td>13.3</td></tr>
      <tr><td>ResNeXt</td><td>RoDLA</td><td>DRGG</td><td>77.9</td><td>16.9</td><td>40.3</td><td>18.4</td><td>13.6</td></tr>
      <tr><td>Swin</td><td>RoDLA</td><td>DRGG</td><td>73.7</td><td>11.4</td><td>26.1</td><td>13.5</td><td>7.9</td></tr>
      <tr><td>InternImage</td><td>RoDLA</td><td>DRGG</td><td>81.5</td><td>30.7</td><td>57.6</td><td>56.3</td><td>46.5</td></tr>
    </tbody>
  </table>
</div>

关键结论：

- **RoDLA + InternImage + DRGG** 是最佳组合，DLA mAP 从 80.5 提升到 **81.5**，说明关系建模反过来帮助了元素检测。
- 不同检测器差异很大：RoDLA 作为检测器时 gDSA 性能远超 DETR/Deformable DETR/DINO，说明**关系预测高度依赖底层检测质量**。
- InternImage backbone 明显优于 ResNet/ResNeXt/Swin，尤其是 gDSA 任务。

### 不同关系类别的表现

<div class="table-wrap">
  <table>
    <thead><tr><th>Backbone</th><th>Detector</th><th>Up</th><th>Down</th><th>Left</th><th>Right</th><th>Parent</th><th>Child</th><th>Sequence</th><th>Reference</th></tr></thead>
    <tbody>
      <tr><td>InternImage</td><td>RoDLA</td><td><strong>49.0</strong></td><td><strong>49.0</strong></td><td><strong>99.0</strong></td><td><strong>99.0</strong></td><td><strong>45.5</strong></td><td><strong>45.5</strong></td><td><strong>56.4</strong></td><td>16.8</td></tr>
      <tr><td>ResNet</td><td>RoDLA</td><td>15.1</td><td>17.2</td><td>27.7</td><td>27.7</td><td>6.9</td><td>4.0</td><td>17.8</td><td>16.8</td></tr>
      <tr><td>ResNeXt</td><td>RoDLA</td><td>23.6</td><td>24.6</td><td>99.1</td><td>99.1</td><td>11.9</td><td>11.9</td><td>33.7</td><td>18.8</td></tr>
      <tr><td>Swin</td><td>RoDLA</td><td>18.8</td><td>19.8</td><td>33.7</td><td>99.0</td><td>3.9</td><td>3.8</td><td>23.5</td><td>5.6</td></tr>
    </tbody>
  </table>
</div>

空间关系（尤其 Left/Right）明显更容易学习；Reference 关系由于样本少、语义复杂，APg@0.5 只有 16.8。

### 不同文档领域

<div class="table-wrap">
  <table>
    <thead><tr><th>指标</th><th>Financial Reports</th><th>Scientific Articles</th><th>Laws and Regulations</th><th>Government Tenders</th><th>Manuals</th><th>Patents</th></tr></thead>
    <tbody>
      <tr><td>mRg@0.5</td><td>15.0</td><td>46.3</td><td>38.7</td><td>40.6</td><td>40.6</td><td>22.7</td></tr>
      <tr><td>mAPg@0.5</td><td>52.6</td><td>54.5</td><td><strong>63.2</strong></td><td>55.9</td><td>46.8</td><td>31.8</td></tr>
    </tbody>
  </table>
</div>

法律法规范式统一，表现最好；专利布局密集复杂，mAPg 仅 31.8。

### Ablation：关系特征提取器是否必要？

<div class="table-wrap">
  <table>
    <thead><tr><th>Relation Head</th><th>DLA mAP</th><th>mRg@0.5</th><th>mAPg@0.5</th><th>mAPg@0.75</th><th>mAPg@0.95</th></tr></thead>
    <tbody>
      <tr><td>DRGG</td><td><strong>81.5</strong></td><td><strong>30.7</strong></td><td><strong>57.6</strong></td><td><strong>56.3</strong></td><td><strong>46.5</strong></td></tr>
      <tr><td>linear layer</td><td>79.9</td><td>25.8</td><td>52.9</td><td>42.3</td><td>30.5</td></tr>
    </tbody>
  </table>
</div>

用简单线性层替换 DRGG 的关系特征提取器后，各项指标明显下降，说明**专门设计的双路关系特征提取与多层聚合是有价值的**。

<figure class="figure">
  <img src="assets/papers/graphdoc/x16_qualitative.webp" alt="DRGG 定性结果" loading="lazy" />
  <figcaption>图 5：DRGG 在 GraphDoc 验证集上的定性结果。模型能同时检测布局元素并预测它们之间的空间与逻辑关系。</figcaption>
</figure>

## 我的理解与启发

### 1. 文档理解需要“结构化表示”

RAG 场景中，我们经常把文档切成文本块后做嵌入。但复杂文档（论文、财报、手册）的布局信息会在这一步丢失。GraphDoc 提示我们：**如果把解析结果保留为图结构（段落-章节-图表引用），检索和问答都能利用更丰富的上下文**。例如，可以构造“问题 → 相关段落 → 所属章节 → 引用的图表”的推理路径。

### 2. 关系预测和元素检测可以互相促进

DRGG 让 DLA mAP 从 80.5 提升到 81.5，说明关系建模不是下游任务的额外负担，而是**检测任务的正则化和上下文补充**。未来在多模态 RAG 的文档解析器中，联合训练布局检测与结构关系可能会成为标准做法。

### 3. Reference 关系是最难也最有价值的

从实验看，Reference 的 AP 只有 16.8 左右，远低于空间关系。但 Reference 对 RAG 最实用——知道“某段文字引用了图 3”才能把图文一起送进 VLM 或生成器。这个方向还有很大提升空间。

### 4. 数据集构建的工程智慧

GraphDoc 没有完全从头标注，而是基于 DocLayNet 的边界框和 PDF 源文件，用规则系统 + OCR + 人工校验批量生成关系。这种**在已有标注上扩展新维度**的策略，对构建领域专属的文档结构数据集很有参考价值。

## 局限与待观察点

- **仅支持单页关系**：GraphDoc 只标注同一页内的关系，跨页引用、跨页章节层级未建模。
- **关系类别偏基础**：8 类关系对科技文献足够，但对表格结构、公式依赖、版面网格等更细粒度语义覆盖不足。
- **Reference 长尾**：Reference 样本最少且最难，当前方法难以处理复杂引用场景。
- **规则标注的泛化性**：虽然经过人工校验，但规则系统基于的 Manhattan/非 Manhattan 分类、X-Y Cut 等假设在极度自由的版式（海报、简历、杂志）上可能失效。
- **与 RAG 的衔接尚是间接**：gDSA 提供结构化输出，但如何把它有效用于 chunking、检索、重排仍需具体设计。

## 结论

GraphDoc 把文档理解从“检测孤立元素”推进到“预测元素关系图”，并提供了大规模数据集和可插拔的 DRGG 基线。它在 ICLR 2025 上为文档结构分析开辟了新任务方向，对多模态 RAG 的文档解析、结构化索引和图文关联推理都有直接启发。
