---
id: "colpali-efficient-document-retrieval-vlm"
title: "ColPali：用视觉语言模型重做文档检索"
category: "多模态 RAG"
date: "2026-06-11"
order: 10
readTime: "12 min"
tags: ["Document Retrieval","VLM","RAG","ColBERT","Late Interaction","ViDoRe"]
summary: "ColPali 把文档页当作图片直接编码，用 PaliGemma 生成多向量视觉表示，再通过 ColBERT 风格 late interaction 与查询匹配。它在 ViDoRe 上达到 81.3 nDCG@5，显著超过传统 OCR/Layout/Captioning 管线，同时把离线索引从 7.22s/page 降到 0.39s/page。"
---

## 一句话总结
ColPali 的核心观点很激进但很实用：**不要先把 PDF 解析成文本再检索，而是直接把文档页图片嵌入到视觉空间里检索**。它用 PaliGemma 处理页面图像，输出 ColBERT 风格的多向量表示，再用 late interaction 做 query-page 匹配，从而同时利用文字、版式、表格、图像和字体等视觉线索。

<div class="metric-grid">
  <div class="metric"><strong>81.3</strong><span>ViDoRe 平均 nDCG@5，显著高于最佳传统管线 67.0</span></div>
  <div class="metric"><strong>0.39s/page</strong><span>ColPali 离线页面索引延迟，传统 Unstructured+Captioning 为 7.22s/page</span></div>
  <div class="metric"><strong>30ms/query</strong><span>在线查询编码延迟，接近 BGE-M3 的 22ms/query</span></div>
  <div class="metric"><strong>118,695</strong><span>训练用 query-page pairs，包含公开数据与合成 PDF 问题</span></div>
</div>

## 论文信息
- **标题**：ColPali: Efficient Document Retrieval with Vision Language Models
- **作者**：Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo
- **机构**：Illuin Technology, Equall.ai, CentraleSupélec Paris-Saclay, ETH Zürich
- **会议**：ICLR 2025
- **链接**：[arXiv:2407.01449](https://arxiv.org/abs/2407.01449) · [资源主页](https://hf.co/vidore)

## 背景：为什么传统文档 RAG 管线很脆弱？
真实文档不是纯文本。PDF 页面里经常同时包含正文、标题、表格、图表、脚注、字体强调、空间排版和扫描噪声。传统 RAG 的文档摄入通常要经过一长串步骤：

- **OCR / PDF parser**：把页面转成文字。
- **Layout detection**：识别标题、段落、表格、图片等区域。
- **Chunking**：把抽取出的文本切成可检索片段。
- **Captioning**：对图表、图片再生成自然语言描述。
- **Text embedding**：最后才用文本嵌入模型建立索引。

问题在于，系统效果经常不取决于 embedding 模型有多强，而取决于前面的解析管线有没有漏掉视觉信息。表格结构、图中数值、页面布局和字体强调一旦在摄入阶段丢失，后续检索模型很难再补回来。

<figure class="figure">
  <img src="assets/papers/colpali/figure1_overview.webp" alt="ColPali 与标准文档检索管线对比" loading="lazy" />
  <figcaption>图 1：传统检索需要 OCR、layout detection、chunking 等复杂摄入流程；ColPali 直接把页面图片送入 VLM，并用 late interaction 与查询匹配。</figcaption>
</figure>

## Benchmark：ViDoRe 评测视觉丰富文档检索
论文首先提出 **ViDoRe（Visual Document Retrieval Benchmark）**，用于评估系统能否根据查询检索到正确的文档页。

ViDoRe 的价值在于它不是只测纯文本段落检索，而是覆盖更接近真实企业 RAG 的页面级检索：

- **模态多样**：文本、表格、图表、infographic、扫描文档。
- **领域多样**：医疗、能源、政府、金融、AI 科研、环境报告等。
- **语言多样**：英文与法文。
- **场景多样**：既包含学术 VQA 数据改造任务，也包含从公开 PDF 构建的实际检索任务。

<figure class="figure">
  <img src="assets/papers/colpali/table1_vidore.webp" alt="ViDoRe benchmark 数据集组成" loading="lazy" />
  <figcaption>图 2：ViDoRe 包含 DocVQA、InfoVQA、TAT-DQA、arXiVQA、TabFQuAD，以及能源、政府、医疗、AI、Shift Project 等实际文档任务。</figcaption>
</figure>

这个 benchmark 的设计也反过来说明了论文的核心判断：文档检索不能只评价文本 embedding，而要评价 **端到端摄入 + 表示 + 匹配** 的整体能力。

## 方法：ColPali 如何工作？
### 1. Retrieval in Vision Space
ColPali 不把 PDF 页面先解析成文本，而是把每一页渲染成图片，再输入 PaliGemma-3B。这样模型看到的是完整页面，天然保留：

- 文字内容与 OCR 线索；
- 表格、图表和图片区域；
- 标题层级、空间布局和相邻关系；
- 字体大小、颜色、强调等视觉提示。

这相当于把文档检索从 “text space” 改为 “vision space”：页面不再是被解析后的 chunk，而是一个可被 VLM 直接理解的视觉对象。

### 2. 多向量页面表示
ColPali 在 PaliGemma 输出 token embedding 后加一层投影层，把文本 token 和图像 token 都映射到维度 \(D=128\) 的公共向量空间。

关键不是把整页压成一个向量，而是保留每个 query token 和每个页面 patch/token 的多向量表示：

- 查询表示为 \(E_q \in \mathbb{R}^{N_q \times D}\)；
- 文档页表示为 \(E_d \in \mathbb{R}^{N_d \times D}\)。

这种设计比单向量 bi-encoder 更适合文档页，因为一页里可能同时有多个局部证据：表格某个单元格、图表坐标轴、标题区域、页脚注释等。

### 3. Late Interaction 匹配
ColPali 借鉴 ColBERT，用 late interaction 计算 query-page 分数。对每个 query token，找到它与页面所有向量中最相似的 patch/token，然后把这些最大相似度求和：

\[
LI(q,d)=\sum_{i=1}^{N_q}\max_{j \in [1,N_d]} \langle E_q^{(i)}, E_d^{(j)} \rangle
\]

这带来两个好处：

1. **文档页可以离线编码**：页面向量提前算好并索引，在线只需要编码 query。
2. **局部交互仍然丰富**：不像单向量检索那样把所有证据压扁，query token 可以分别对齐到页面中的不同区域。

### 4. 训练策略
训练集包含 **118,695 个 query-page pairs**：

- 63% 来自公开学术数据集训练集；
- 37% 来自 web-crawled PDF 页面，并用 Claude-3 Sonnet 生成伪问题；
- 训练集刻意保持英文，用来观察模型在法文任务上的 zero-shot 泛化。

模型训练 1 epoch，使用 LoRA 微调语言模型 transformer 层和最终投影层，采用 in-batch contrastive loss，把正确 query-page 对作为正样本，batch 内其他页面作为负样本。

## 实验结果
### 1. 主结果：ColPali 显著超过传统检索管线
在 ViDoRe 上，ColPali 的平均 nDCG@5 达到 **81.3**，明显超过传统强管线：

- Unstructured + OCR + BGE-M3：**66.1**
- Unstructured + Captioning + BGE-M3：**67.0**
- SigLIP vanilla：**51.4**
- BiSigLIP fine-tuning：**58.6**
- BiPali：**58.8**
- ColPali：**81.3**

<figure class="figure">
  <img src="assets/papers/colpali/table2_results.webp" alt="ColPali 在 ViDoRe 上的主实验结果" loading="lazy" />
  <figcaption>图 3：ColPali 在 ViDoRe 平均 nDCG@5 上达到 81.3，尤其在 infographic、arXiv 图表、法文表格等视觉复杂任务上提升明显。</figcaption>
</figure>

这组结果最重要的不是 “VLM 比文本模型强”，而是证明：**端到端保留页面视觉结构，比把视觉元素先翻译成文本再检索更有效**。

### 2. 性能提升来自三个因素
论文把提升拆成三步：

- **任务数据微调**：把 SigLIP 在文档检索数据上微调成 BiSigLIP，平均从 51.4 提升到 58.6。
- **引入 LLM 语义能力**：BiPali 用 PaliGemma 的语言模型处理图像 patch，法文任务上更好，但单向量表示整体仍有限。
- **多向量 + late interaction**：ColPali 相比 BiPali 从 58.8 提升到 81.3，说明页面级文档检索非常依赖细粒度局部匹配。

换句话说，ColPali 的关键贡献不是单独使用 PaliGemma，而是把 VLM 的页面理解能力和 ColBERT 的细粒度检索机制结合起来。

### 3. 效率：索引更快，在线查询仍可接受
传统高质量管线慢在离线摄入：layout detection、OCR、captioning 都很耗时。论文在 NVIDIA L4 上测得：

<div class="table-wrap">
  <table>
    <thead><tr><th>系统</th><th>主要步骤</th><th>离线索引延迟</th><th>在线查询编码</th></tr></thead>
    <tbody>
      <tr><td>Unstructured + Captioning</td><td>Layout + OCR + Captioning + Encoding</td><td><strong>7.22s/page</strong></td><td>约 22ms/query（BGE-M3）</td></tr>
      <tr><td>SigLIP</td><td>页面图片编码</td><td>0.12s/page</td><td>快速向量匹配</td></tr>
      <tr><td>ColPali</td><td>页面图片 VLM 编码</td><td><strong>0.39s/page</strong></td><td>约 30ms/query</td></tr>
    </tbody>
  </table>
</div>

ColPali 的在线查询比 BGE-M3 略慢，但离线索引比复杂解析管线快一个数量级以上。对于企业知识库、RAG 文档库、扫描 PDF 检索等场景，这个取舍很有吸引力。

### 4. 可解释性：能看到 query token 关注页面哪里
Late interaction 的另一个好处是可解释。因为每个 query token 都会找到页面中最相似的 patch，所以可以把相似度热力图叠加到原图上，观察模型到底在看哪里。

<figure class="figure">
  <img src="assets/papers/colpali/figure3_pooling_interpretability.webp" alt="ColPali token pooling 与可解释性热力图" loading="lazy" />
  <figcaption>图 4：左侧展示 token pooling 后性能退化很小；右侧展示 query token 在页面上的关注区域，例如 hour 对齐到图表中的 hours / hourly 文本和横轴。</figcaption>
</figure>

这对 RAG 产品很重要：检索系统不仅要返回页面，还最好能解释为什么这页相关。ColPali 的 patch-level 匹配为高亮证据区域提供了自然入口。

## Ablation：哪些设计最关键？
### 1. 多向量 late interaction 是核心
BiPali 同样使用 PaliGemma，但把页面压成单向量后平均只有 **58.8**。ColPali 保留多向量并进行 late interaction 后提升到 **81.3**。这说明视觉丰富文档的相关性往往分散在多个局部区域，单向量压缩会损失太多信息。

### 2. Patch 数量与模型能力存在取舍
把 PaliGemma 图像 patch 从 1024 减到 512 会导致明显性能下降（约 -24.8 nDCG@5），但显存占用更低。论文也尝试 Idefics2-8B，发现更强语言模型能用更少 image patches 得到更有效表示。

这说明文档检索不是简单地 “patch 越多越好”，而是取决于 VLM backbone、视觉 token 数量、存储成本和检索精度之间的平衡。

### 3. Token pooling 可以显著压缩存储
ColPali 的一个现实问题是存储开销：DocVQA 上 float16 页面 embedding 约 **257.5KB/page**，明显大于 BGE-M3 的 **8.60KB**。不过论文显示，用 token pooling 做聚类压缩时，pool factor=3 可以减少 **66.7%** 的向量数量，同时保持约 **97.8%** 的原始性能。

这意味着 ColPali 更适合作为高质量文档检索方案，但生产部署仍需要结合向量压缩、量化或分层召回。

### 4. 领域适配很直接
相比复杂的多步骤管线，ColPali 可以端到端微调到新任务。论文加入 1,552 条法文表格样本后，TabFQuAD 的 nDCG@5 提升 **+2.6**，Recall@1 提升 **+5**，且其他任务没有明显退化。

这对企业场景很有价值：如果某个业务文档模板固定，例如发票、财报、合同、能源报告，可以用少量 query-page 对进行专项适配。

## 我的理解与启发
ColPali 对 RAG 系统最大的启发，是把 “文档摄入” 从规则工程问题重新变成表示学习问题。

1. **不要过早丢弃视觉结构**：传统 OCR → chunk 的流程会把页面变成线性文本，但文档真正的语义经常藏在版式和图表中。ColPali 证明直接保留页面图像可能更稳。
2. **检索粒度应该贴合证据粒度**：页面级检索需要在一页内部找到局部证据，多向量表示比单向量更自然。
3. **RAG 的瓶颈常在 ingestion，而非 embedding 模型本身**：论文中 BM25 和 BGE-M3 在不同摄入管线下差距不如 OCR/captioning 是否保留视觉信息来得大。
4. **视觉检索可以和可解释性结合**：patch-level heatmap 可以支持证据高亮，帮助用户理解“为什么召回这页”。

如果做企业文档问答，ColPali 很适合作为 **视觉丰富 PDF 的第一阶段召回或重排器**。它未必替代所有文本索引，但可以补上 OCR、表格、图表和扫描件中的信息缺口。

## 局限与待观察点
- **存储开销较大**：多向量页面表示比传统文本 embedding 大很多，百万级页面库必须配合 pooling、量化或分层检索。
- **返回粒度是页面级**：页面召回后仍需要后处理来定位段落、表格单元格或具体答案区域。
- **依赖页面渲染质量**：扫描质量、分辨率、旋转、遮挡等因素会影响 VLM 编码效果。
- **线上 late interaction 有扩展挑战**：论文指出小规模 corpus 开销不大，但超大规模检索需要专门索引和近似搜索工程。
- **训练数据仍有限**：118K 规模相对不大，且训练集英文为主；更多语言、行业和低质量扫描文档仍需验证。

## 结论
ColPali 是文档 RAG 方向非常值得关注的工作：它没有继续堆 OCR、layout、captioning 等复杂规则管线，而是直接把文档页作为视觉对象来检索。通过 **PaliGemma 页面编码 + 多向量表示 + ColBERT late interaction**，ColPali 在 ViDoRe 上同时取得更高检索质量和更快离线索引速度。它的意义不只是提出一个模型，更是提示我们：面向真实 PDF 的检索系统，应该尽量保留文档的原生视觉结构。