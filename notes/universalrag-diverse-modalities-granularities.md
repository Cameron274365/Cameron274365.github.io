---
id: "universalrag-diverse-modalities-granularities"
title: "UniversalRAG：面向多模态、多粒度语料的通用检索增强生成"
category: "多模态 RAG"
date: "2026-06-11"
order: 2
readTime: "14 min"
tags: ["RAG", "Multimodal RAG", "UniversalRAG", "Routing", "Granularity", "Retrieval", "Qwen3-VL", "ACL 2026"]
summary: "UniversalRAG 提出一种 any-to-any RAG 框架：不再把所有文本、表格、图片、视频统一塞进一个 embedding 空间，而是先用路由器判断 query 需要哪些模态和粒度，再到对应的专用语料库中检索。论文在 10 个跨模态、多粒度 benchmark 上验证了该设计，显示出比单模态 RAG、统一 embedding RAG 和 MultiRAG 更稳定的平均效果。"
---

## 一句话总结
UniversalRAG 的核心观点是：**真实 RAG 系统不应该只有一个统一语料库和一个统一向量空间，而应该按 query 的信息需求动态选择“该查哪种模态、该查多细的粒度”**。论文发现，把文本、图片、视频等全部塞进统一 embedding 空间会产生 modality gap，检索结果容易偏向与 query 同模态的内容；因此它用 modality- and granularity-aware routing，把检索分发到段落、文档、表格、图片、视频 clip、完整视频或 no-retrieval 等不同路径。

<div class="metric-grid">
  <div class="metric"><strong>10 个</strong><span>跨文本、表格、图片、视频的评测数据集</span></div>
  <div class="metric"><strong>42.40</strong><span>Qwen3-VL-8B 下最佳 UniversalRAG 平均分</span></div>
  <div class="metric"><strong>95.28%</strong><span>Qwen3-VL-2B Router 的模态选择准确率</span></div>
  <div class="metric"><strong>44.82</strong><span>UniversalRAG R@5，高于 GME 22.16</span></div>
</div>

## 论文信息
- **标题**：UniversalRAG: Retrieval-Augmented Generation over Corpora of Diverse Modalities and Granularities
- **作者**：Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jinheon Baek, Sung Ju Hwang
- **机构**：KAIST, DeepAuto.ai
- **会议**：ACL 2026 Oral；arXiv v4, 2026-05-18
- **链接**：[arXiv:2504.20734](https://arxiv.org/abs/2504.20734) · [PDF](https://arxiv.org/pdf/2504.20734) · [项目主页](https://universalrag.github.io/) · [代码](https://github.com/wgcyeo/UniversalRAG)

<figure class="figure">
  <img src="assets/papers/universalrag/concept.webp" alt="UniversalRAG 与单模态、单粒度、统一语料 RAG 的对比" loading="lazy" />
  <figcaption>图 1：UniversalRAG 的问题设定。单模态 RAG 缺少跨模态知识，单粒度 RAG 无法适配查询复杂度，统一语料 RAG 又会受 modality gap 影响；UniversalRAG 用路由机制选择合适的模态和粒度。</figcaption>
</figure>

## 背景：为什么“一个向量库解决所有问题”不够？
传统 RAG 通常默认语料是纯文本：把文档切成 chunk，统一向量化，再检索 top-k。多模态 RAG 出现后，一个自然想法是：既然有 multimodal encoder，就把文本、表格、图片、视频都嵌入同一个空间，然后做统一检索。

论文认为这个做法在真实系统里有三类问题：

- **单模态限制**：只查文本时，无法回答需要图片空间信息、视频时序信息或表格结构信息的问题。
- **单粒度限制**：同样是文本，有的 query 只需要一个段落，有的需要完整文档；同样是视频，有的只需短 clip，有的需要完整视频上下文。
- **统一 embedding 的 modality gap**：多模态 embedding 虽然试图对齐语义，但实际仍倾向按模态聚类，导致检索偏向与 query 同模态的语料，而不是最相关的语料。

<figure class="figure">
  <img src="assets/papers/universalrag/embedding_space.webp" alt="统一 embedding 空间中的 modality gap" loading="lazy" />
  <figcaption>图 2：统一 embedding 空间的 t-SNE 可视化。不同模态仍明显分簇，说明“统一空间”并不等于真正消除了跨模态检索偏差。</figcaption>
</figure>

## 方法：UniversalRAG 的核心设计
### 1. Modality-Aware Retrieval：先选模态，再做专用检索
UniversalRAG 不把所有语料合并成一个大 corpus，而是保留多个模态专用 corpus：文本、表格、图片、视频等各自使用适合自己的检索器和 embedding 空间。

给定 query 后，路由器 \(\mathcal{R}\) 先预测需要哪些模态，得到 \(M_q\)，再只在这些目标模态对应的语料库中检索。这样做有两个好处：

- **避免跨模态直接比较带来的偏置**：不再让一条文本 query 在统一空间里和所有图片、视频、文本直接竞争。
- **方便扩展新模态**：新增音频、3D 或代码语料时，只需增加对应 corpus 和 routing option，不必重训整个统一 embedding 空间。

论文还用一个简化命题解释该设计：如果统一 embedding 的相似度中存在模态偏置项 \(\alpha\)，并且它强到足以压过真实相关性波动，那么 modality-aware routing 比统一检索更有机会命中真正需要的模态。

### 2. Granularity-Aware Retrieval：同一模态也要选粒度
UniversalRAG 进一步把每种模态拆成不同粒度：

- **None**：无需外部检索，例如常识或参数知识足够回答的问题。
- **Paragraph / Document**：短事实问题更适合段落，多跳或综合问题更适合文档。
- **Table**：保留结构化行列信息。
- **Image**：用于视觉实体、空间布局、图像证据。
- **Clip / Video**：短时局部事件适合 clip，长程问题可能需要完整视频。

这一步很关键，因为“检索到了正确模态”并不等于“检索到了合适上下文”。粒度太细可能缺少上下文，粒度太粗又会引入噪声、增加生成负担。

### 3. Router：训练式与免训练式两条路线
论文实现了两类路由器：

- **Training-based Router**：利用 benchmark 的任务属性自动构造标签，例如 NQ 对应 Paragraph，HotpotQA 对应 Document，HybridQA 对应 Paragraph + Table。随后用 Qwen3-VL-2B、InternVL3.5-1B、T5Gemma 2 270M 等模型训练多标签分类路由器。
- **Training-free Router**：直接 prompt GPT-5 或 Qwen3-VL-8B，让模型根据 query 判断需要哪些模态-粒度组合。

两者各有取舍：训练式路由器在 in-domain 上准确率高、成本低；免训练路由器在 OOD 场景更稳。论文最后还尝试了 confidence-based ensemble 和 majority voting，把两者互补起来。

## 数据与评测
论文构建了一个覆盖多模态、多粒度 RAG 的综合评测，包含 10 个 in-domain 数据集：

<div class="table-wrap">
  <table>
    <thead><tr><th>数据集</th><th>目标检索路径</th><th>查询数</th><th>语料规模</th></tr></thead>
    <tbody>
      <tr><td>MMLU</td><td>None</td><td>1,710</td><td>无需语料</td></tr>
      <tr><td>Natural Questions</td><td>Paragraph</td><td>2,000</td><td>850k paragraphs</td></tr>
      <tr><td>HotpotQA</td><td>Document</td><td>2,000</td><td>509k documents</td></tr>
      <tr><td>HybridQA</td><td>Paragraph + Table</td><td>2,000</td><td>15k tables</td></tr>
      <tr><td>MRAG-Bench</td><td>Image</td><td>1,353</td><td>6k images</td></tr>
      <tr><td>WebQA / InfoSeek</td><td>Paragraph + Image</td><td>各 2,000</td><td>20k mixed corpus</td></tr>
      <tr><td>LVBench</td><td>Clip / Video</td><td>777</td><td>89 videos，平均 3,865 秒</td></tr>
      <tr><td>VideoRAG-Wiki / Synth</td><td>Clip / Video</td><td>各 374</td><td>9k videos/clips，平均 378 秒</td></tr>
    </tbody>
  </table>
</div>

生成模型主要使用 Qwen3-VL-8B-Instruct、InternVL3.5-8B 和 Molmo2-4B。检索器则按模态分别选择：文本使用 Qwen3-Embedding-4B，视觉使用 VLM2Vec-V2，表格使用 row-level dense embedding。

## 实验结果
### 1. 平均性能：UniversalRAG 明显优于统一 embedding 和 MultiRAG
在 Qwen3-VL-8B-Instruct 作为生成模型时，UniversalRAG 的最佳平均分达到 **42.40**，非常接近 Oracle 的 **42.45**。对比之下，Naïve 为 **35.59**，MultiRAG 为 **34.07**，统一 embedding 方法 GME / VLM2Vec-V2 分别为 **33.88 / 33.31**。

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>平均分</th><th>特点</th></tr></thead>
    <tbody>
      <tr><td>Naïve</td><td>35.59</td><td>不检索，依赖模型参数知识</td></tr>
      <tr><td>最佳单模态 RAG</td><td>约 37.26</td><td>在对应任务有效，但跨任务不稳定</td></tr>
      <tr><td>GME / VLM2Vec-V2</td><td>33.88 / 33.31</td><td>统一 embedding 受 modality gap 影响</td></tr>
      <tr><td>MultiRAG</td><td>34.07</td><td>检索所有语料，容易引入无关噪声</td></tr>
      <tr><td><strong>UniversalRAG Qwen3-VL-2B Router</strong></td><td><strong>42.40</strong></td><td>模态与粒度路由后定向检索</td></tr>
      <tr><td>Oracle</td><td>42.45</td><td>理想路由上限</td></tr>
    </tbody>
  </table>
</div>

<figure class="figure">
  <img src="assets/papers/universalrag/barplot.webp" alt="不同 RAG 方法在多个 LVLM 上的平均结果" loading="lazy" />
  <figcaption>图 3：不同 RAG 方法的平均效果对比。UniversalRAG 在多个 LVLM 生成器上都保持更高平均表现，说明收益主要来自检索路由框架，而不只是某个生成模型。</figcaption>
</figure>

### 2. Cross-modal retrieval：多模态证据互补很重要
在 HybridQA 和 WebQA 这类需要跨模态证据的问题上，只路由到单一模态会丢信息。UniversalRAG 支持 multi-hot routing，可以同时检索 Paragraph + Table 或 Paragraph + Image。

例如在 Qwen3-VL-2B Router 下：

- HybridQA 从 uni-modal 的 **9.60 EM / 14.56 F1** 提升到 cross-modal 的 **11.05 EM / 16.23 F1**。
- WebQA 从 **67.93 ROUGE-L / 95.58 BERTScore** 提升到 **70.22 / 95.86**。

这说明 UniversalRAG 并不是简单“给 query 选一个最像的模态”，而是能把互补证据组合起来。

### 3. Modality routing：真正缓解了模态偏置
论文抽样 200 个 query 分析检索到的模态分布。VLM2Vec-V2 几乎全部检索文本，GME 也明显偏向文本；UniversalRAG 的检索分布更均衡，能按任务需要选择图片、视频、表格等语料。

<figure class="figure">
  <img src="assets/papers/universalrag/selection_rate.webp" alt="不同方法检索模态分布对比" loading="lazy" />
  <figcaption>图 4：检索模态分布对比。统一 embedding 方法容易偏向文本；UniversalRAG 的路由机制能更准确地选择目标模态。</figcaption>
</figure>

定量上，UniversalRAG Qwen3-VL-2B 的模态选择准确率达到 **95.28%**，R@5 达到 **44.82**；GME 的模态准确率只有 **36.27%**，R@5 为 **22.16**。这直接支持论文的主张：先选对 corpus，再做检索，比在统一大池子里硬搜更可靠。

### 4. Granularity：更多粒度通常能带来更合适的上下文
在训练免模型路由器上，论文比较了不同 granularity 数量。以 GPT-5 Router 为例，HotpotQA 从 1 个粒度的 **23.20 EM / 31.38 F1** 提升到 4 个粒度的 **24.70 / 33.25**；LVBench 也从 **31.92 Acc** 提升到 **32.85**。

不过提升不是完全单调，说明粒度选择本身存在 trade-off：更细的粒度能减少噪声，但也可能丢掉必要上下文；更粗的粒度能覆盖更多信息，但会增加无关内容。

### 5. 效率与路由器大小：路由开销可以被规模收益抵消
UniversalRAG 不是把所有语料都查一遍，而是只查相关 corpus。随着 corpus 规模增大，路由成本相对固定，而检索空间减少带来的收益会越来越明显。论文指出，在大规模语料，尤其超过 10M entries 后，UniversalRAG 的延迟优势会进一步扩大。

<figure class="figure">
  <img src="assets/papers/universalrag/latency.webp" alt="不同语料规模下的检索延迟" loading="lazy" />
  <figcaption>图 5：检索延迟随 corpus size 的变化。UniversalRAG 多了路由步骤，但通过缩小检索范围，在大规模语料下反而更具扩展性。</figcaption>
</figure>

同时，路由器不一定需要很大。论文显示 1B 级模型已经能达到约 **90%** 路由准确率，更大的模型可以进一步接近理想路由。

<figure class="figure">
  <img src="assets/papers/universalrag/router_size.webp" alt="路由器模型大小与准确率关系" loading="lazy" />
  <figcaption>图 6：路由器模型大小与准确率关系。路由准确率随模型规模提升，但小模型也具备实用价值。</figcaption>
</figure>

## Ablation：哪些设计最关键？
### Modality routing 是第一优先级
Unified embedding 方法的问题不是“检索器不够强”这么简单，而是检索空间本身存在模态偏置。UniversalRAG 的最大收益来自把跨模态比较拆成“先选模态、再在模态内检索”。

### Cross-modal routing 不能省
HybridQA、WebQA、InfoSeek 这类任务经常需要多源证据。只选一个模态会让系统在某些问题上天然信息不足。UniversalRAG 的 multi-hot routing 比单标签分类更符合真实用户 query。

### 粒度选择影响上下文质量
段落、文档、clip、video 并不是简单的大小差异，而是不同信息组织方式。RAG 系统如果只固定一种 chunk size，会在复杂 query 和简单 query 之间来回妥协。

### 训练式和免训练式路由器适合不同场景
训练式路由器 in-domain 准确率更高，例如 Qwen3-VL-2B 在 in-domain 的 router accuracy 为 **95.81%**；但 OOD 上会下降到 **71.29%**。GPT-5 training-free router 在 OOD 上达到 **77.38%**，说明通用推理能力对未知分布有帮助。

## 我的理解与启发
这篇论文最值得借鉴的是它把多模态 RAG 从“统一检索器能力问题”重新定义成 **查询意图路由 + 专用检索器编排问题**。

对实际工程系统来说，这个视角很重要：

1. **不要迷信统一向量空间**：跨模态 embedding 能做 alignment，但不代表能在生产级混合语料里稳定排序所有模态。
2. **Router 是 RAG 系统的控制平面**：检索器负责“怎么查”，路由器负责“查哪里、查多细、是否需要查”。
3. **多粒度索引比单一 chunk size 更接近真实需求**：用户 query 的范围不同，检索单元也应该不同。
4. **MultiRAG 不是简单地“查更多就更好”**：无关模态证据会污染生成上下文，尤其在 VLM 需要同时读图、读表、读长文本时更明显。
5. **可扩展性来自模块化**：新增模态时，不必推翻整个系统，只要增加 corpus、retriever 和 routing label。

如果要做面向企业知识库、科研论文库或视频资料库的通用 RAG，UniversalRAG 的设计可以作为一个非常实用的系统骨架：前面放一个轻量 router，后面挂多套专用索引，最后由统一 LVLM 进行证据融合与回答。

## 局限与待观察点
- **路由标签主要来自数据集归纳偏置**：论文自动把 benchmark 映射到目标模态/粒度，但真实 query 的最佳路由可能更复杂，需要人工或交互式标注验证。
- **粒度层级仍比较有限**：文本和视频主要是两级粒度，真实系统可能需要章节、页面、段落、句子、镜头、事件、对象轨迹等更多层次。
- **Router 错误会级联影响生成**：如果第一步路由错了，后续检索器再强也很难补救，因此需要置信度、fallback 和 ensemble 机制。
- **跨模态证据融合仍依赖生成模型能力**：检索到多模态证据只是第一步，LVLM 是否能正确读表、读图、读视频并合并推理仍是瓶颈。
- **评测仍以离线 benchmark 为主**：真实应用中的查询分布、语料更新、权限控制、延迟预算和用户反馈循环还需要进一步验证。

## 结论
UniversalRAG 给多模态 RAG 提供了一个清晰范式：**把语料按模态和粒度组织起来，用路由器选择最合适的检索路径，再交给 LVLM 融合证据生成答案**。它的贡献不只是多跑了几个 benchmark，而是指出统一 embedding RAG 在异构语料上存在结构性偏差，并用模块化 routing 方案给出了一条更适合真实系统扩展的路线。