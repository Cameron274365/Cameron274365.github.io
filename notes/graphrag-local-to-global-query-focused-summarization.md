---
id: "graphrag-local-to-global-query-focused-summarization"
title: "GraphRAG：从局部到全局的图增强查询聚焦摘要"
category: "多模态 RAG"
date: "2026-07-08"
order: 46
readTime: "15 min"
tags: ["RAG","GraphRAG","Knowledge Graph","Query-Focused Summarization","Community Detection","Sensemaking"]
summary: "GraphRAG 用 LLM 把语料抽成实体知识图谱，再用 Leiden 社区检测做层次划分并预生成社区摘要；面对针对整个语料的全局问题，通过 map-reduce 汇总所有社区的局部回答形成全局答案，在百万 token 级语料上的答案全面性与多样性显著超过传统向量 RAG。"
---

## 一句话总结

传统向量 RAG 擅长“检索具体事实”，但面对**“这个语料库的主要主题是什么”这类针对整个语料的全局问题**时会失效——因为这本质上是查询聚焦摘要（QFS）任务，而不是检索任务。GraphRAG 的核心思路是：**先用 LLM 把语料抽成实体-关系知识图谱，用社区检测把图切成层次化主题分区并预先生成社区摘要，再用 map-reduce 把所有相关社区的局部回答汇总成全局答案**。

<div class="metric-grid">
  <div class="metric"><strong>72–83%</strong><span>全局方法对向量 RAG 的全面性胜率（p&lt;.001）</span></div>
  <div class="metric"><strong>62–82%</strong><span>全局方法对向量 RAG 的多样性胜率</span></div>
  <div class="metric"><strong>9–43×</strong><span>根级社区摘要（C0）相比全文摘要的 token 成本降幅</span></div>
  <div class="metric"><strong>~1M</strong><span>评测语料量级（Podcast / News 两个数据集）</span></div>
</div>

## 论文信息

- **标题**：From Local to Global: A Graph RAG Approach to Query-Focused Summarization
- **作者**：Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson
- **机构**：Microsoft Research / Microsoft Strategic Missions and Technologies / Microsoft Office of the CTO
- **版本**：arXiv:2404.16130v2，2025-02-19 修订（v1 为 2024-04-24）
- **链接**：[arXiv](https://arxiv.org/abs/2404.16130) · [PDF](https://arxiv.org/pdf/2404.16130) · [代码](https://github.com/microsoft/graphrag)

## 背景：为什么向量 RAG 答不了“全局问题”？

RAG 的常规做法是把语料切成 chunk、向量化，按 query 检索 Top-K 相似片段再交给 LLM 生成答案。这对**显式事实检索**（“X 是什么时候发生的？”）很有效，但存在一个根本性盲区：

- **全局理解（sensemaking）问题失效**：像“数据集里有哪些主要主题？”这类问题，答案分散在整个语料中，没有任何单独的 chunk 能提供答案。向量检索只会返回若干局部片段，容易把“采样到的几条事实”伪装成“全局总结”。
- **传统 QFS 方法不可扩展**：查询聚焦摘要（Query-Focused Summarization）方法本可以处理这类问题，但无法扩展到 RAG 系统所索引的海量文本。

GraphRAG 的目标就是**兼取两者之长**：既能像 QFS 一样对整个语料做全局摘要，又能像 RAG 一样随语料规模扩展。它的关键洞察是——**用图结构 + 社区划分，把“对整个语料摘要”这个不可扩展的任务，拆解成对一批互斥、完备的社区分别摘要，再分治汇总**。

## 方法：索引期建图 + 查询期分治汇总

GraphRAG 分为**索引期（Indexing Time）**和**查询期（Query Time）**两大阶段。整体数据流如下：

<figure class="figure">
  <img src="assets/papers/graphrag/pipeline.webp" alt="GraphRAG 流程总览" loading="lazy" />
  <figcaption>图 1：GraphRAG 流程。索引期：源文档 → 文本块 → 实体与关系 → 知识图谱 → 图社区 → 社区摘要；查询期：社区摘要 → 社区答案 → 全局答案。每一步都用面向领域定制的 LLM prompt 完成抽取与摘要，社区检测采用 Leiden 算法。</figcaption>
</figure>

### 1. 源文档 → 文本块：chunk 大小的召回-成本权衡

首先把语料切成文本块交给 LLM 抽取。chunk 越长，抽取所需的 LLM 调用越少（成本低），但会出现“中间信息丢失”（lost in the middle）导致召回下降。论文实验发现：同一份数据，600-token chunk 抽出的实体引用数**几乎是 2400-token chunk 的两倍**。为此他们引入**自反思（gleaning）**机制（见后文启发部分），让大 chunk 也能保持召回。

### 2. 文本块 → 实体、关系与断言

对每个 chunk，LLM 被提示抽取三类元素并各自生成简短描述：

- **实体（Entities）**：人物、组织、地点等命名实体 + 描述；
- **关系（Relationships）**：明确相关实体之间的关系 + 描述；
- **断言（Claims）**：关于实体的可验证事实性陈述（日期、事件、交互等）。

这里的关键设计是**可领域定制**：通过替换 few-shot 示例，通用 prompt 可以适配科学、医学、法律等专业领域，无需改代码。抽取本质是一种**抽象式摘要**——关系和断言往往并未在原文中显式陈述，而是 LLM 归纳出来的。

### 3. 实体、关系 → 知识图谱

同一元素通常在多个文档中被反复抽出。这一步把这些实例**归并成图的节点和边**：实体描述被聚合摘要成节点；相同关系被聚合成边，重复次数作为**边权重**；断言同样聚合。论文用**精确字符串匹配**做实体消歧（简单但有效），并指出 GraphRAG 对重复实体天然鲁棒——因为重复项通常会被聚到同一社区一起摘要。

### 4. 知识图谱 → 图社区：Leiden 层次划分

用 **Leiden 社区检测**算法对图做**层次化递归划分**：在每个社区内递归检测子社区，直到不可再分的叶子社区。每一层都给出一个**互斥、完备（mutually exclusive, collectively exhaustive）**的节点划分——这正是能做“分治式全局摘要”的结构基础。

<figure class="figure">
  <img src="assets/papers/graphrag/community_level0.webp" alt="Leiden 社区检测 Level 0" loading="lazy" />
  <figcaption>图 4a：MultiHop-RAG 语料上的图社区（Level 0，根级社区，对应最大模块度的划分）。圆圈为实体节点，大小正比于度数，颜色代表社区。</figcaption>
</figure>

<figure class="figure">
  <img src="assets/papers/graphrag/community_level1.webp" alt="Leiden 社区检测 Level 1" loading="lazy" />
  <figcaption>图 4b：同一图的 Level 1 子社区，揭示了根级社区内部的更细结构。层次结构让不同粒度的问题可以在不同层级作答。</figcaption>
</figure>

### 5. 图社区 → 社区摘要

对每个社区生成摘要，自底向上构建：

- **叶子社区**：按“边的两端节点度数之和”从高到低排序（即整体显著性优先），依次把源节点、目标节点、边本身、相关断言的描述加入 LLM 上下文，直到达到 token 上限。
- **高层社区**：若所有元素摘要能装进上下文就直接摘要；否则按元素摘要 token 数从大到小，**用更短的子社区摘要替换更长的元素摘要**，直到装得下。

### 6. 社区摘要 → 社区答案 → 全局答案（map-reduce）

查询期给定用户问题，用某一社区层级的摘要分治作答：

1. **准备（Prepare）**：把社区摘要**随机打乱**并切成固定 token 大小的块，确保相关信息分散在各块而非集中在单个上下文里（避免丢失）。
2. **Map（社区答案）**：并行为每块生成中间答案，并让 LLM 给出 0–100 的**有用性打分**，得 0 分的答案被过滤。
3. **Reduce（全局答案）**：按有用性降序，把中间答案迭代拼进新上下文直到 token 上限，最终生成返回给用户的全局答案。

## 数据与评测

- **数据集**：两个约百万 token 级语料——**Podcast transcripts**（1669×600-token chunk，~1M tokens）与 **News articles**（3197×600-token chunk，~1.7M tokens）。
- **全局问题生成**：由于全局 sensemaking 问题没有标准答案，作者用 LLM **先生成用户画像 → 为每个用户生成任务 → 为每个（用户, 任务）生成需要理解整个语料的高层问题**。取 K=M=N=5，每个数据集 125 个测试问题。为公平，问题不直接从语料本身生成。
- **评价方式**：采用 **LLM-as-a-judge 的两两对比**（head-to-head），每次对比重复 5 次取平均。

评价用四个标准，前三个是目标指标，最后一个是“对照标准”：

- **全面性（Comprehensiveness）**：答案覆盖问题各方面的细节程度；
- **多样性（Diversity）**：答案提供的视角与洞见是否丰富多样；
- **赋能性（Empowerment）**：答案是否帮助读者理解并做出有据判断；
- **直接性（Directness）**（对照）：答案是否简洁、切中问题。它与全面性/多样性天然对立，用来验证结果可信度——没有任何方法应在四项上全赢。

## 实验结果

对比 6 个条件：GraphRAG 的四个社区层级 **C0（根级，最少）/ C1 / C2 / C3（最细，最多）**、直接对源文本做 map-reduce 摘要的 **TS**、以及向量 RAG 语义检索 **SS**。

**全局方法 vs 向量 RAG（SS）**：所有全局方法（C0–C3、TS）在**全面性和多样性**上显著超过 SS。全面性胜率 Podcast 72–83%、News 72–80%（p<.001）；多样性胜率 Podcast 75–82%（p<.001）、News 62–71%（p<.01）。作为对照，SS 在**直接性**上稳居第一——符合预期，佐证了结果的可信度。

**赋能性**结果喜忧参半。LLM 分析显示，能否提供**具体例子、引文和出处**是帮助用户形成判断的关键；调优元素抽取 prompt 以保留更多此类细节有望改善。

**GraphRAG vs 全文摘要（TS）**：C1–C3 在全面性和多样性上略优于 TS，但真正的杀手锏是**成本**。下表是各条件每次查询所需的上下文单元数与 token 数：

<table>
  <thead>
    <tr><th>数据集</th><th>指标</th><th>C0</th><th>C1</th><th>C2</th><th>C3</th><th>TS</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="2">Podcast</td><td>单元数</td><td>34</td><td>367</td><td>969</td><td>1310</td><td>1669</td></tr>
    <tr><td>Token 占最大比</td><td>2.6%</td><td>22.2%</td><td>55.8%</td><td>73.5%</td><td>100%</td></tr>
    <tr><td rowspan="2">News</td><td>单元数</td><td>55</td><td>555</td><td>1797</td><td>2142</td><td>3197</td></tr>
    <tr><td>Token 占最大比</td><td>2.3%</td><td>20.7%</td><td>57.4%</td><td>66.8%</td><td>100%</td></tr>
  </tbody>
</table>

**根级社区摘要 C0 只需全文摘要 2–3% 的 token，就能在全局问题上超过向量 RAG，并逼近其他全局方法的效果**——这是 GraphRAG 面向“同一语料反复查询”场景最实用的结论。

**实验 2（基于断言的客观验证）**：为了避免只依赖 LLM 主观打分，作者用 Claimify 从答案中抽取事实性断言（共 47,075 条，平均每个答案 31 条），定义两个客观指标：全面性 = 平均断言数；多样性 = 断言聚类后的平均簇数。结果与实验 1 一致——所有全局方法（C0–C3、TS）的断言数都显著多于 SS（如 News 上 C0 34.18 vs SS 25.23）。在非平局的对比中，LLM 主观判断与客观断言指标的一致率达全面性 78%、多样性 69–70%，表明主观评价有中等偏强的可信度。

## Ablation：哪些设计最关键？

- **社区层级的取舍**：C0–C3 之间**没有统计显著差异**，说明并不需要最细粒度的社区。因此 **C0（根级）成为性价比之王**：token 成本最低、效果不输更细层级。
- **图索引 vs 无图全文摘要（TS）**：更细层级（C1–C3）比 TS 略好，但差距不大；GraphRAG 相对 TS 的真正优势在于**大幅节省 token**（图索引一次、查询期复用）。
- **chunk 大小 + 自反思（gleaning）**：大 chunk 会漏抽实体。做法是抽完后把结果回喂 LLM，用 logit bias 强制 yes/no 判断“是否有遗漏”，若有则提示“上次遗漏了很多实体”让其补抽，可迭代多轮。这让大 chunk 也能保持召回，避免小 chunk 带来的高调用成本。

## 我的理解与启发

- **图的价值不在“可视化”，而在“可分治”**：GraphRAG 最本质的贡献不是画出漂亮的知识图，而是**用社区检测得到一组互斥完备的分区**，把“对整个语料摘要”这个不可扩展任务，转化为可并行、可 map-reduce 的分治问题。图结构是达成这一点的手段。
- **“全局问题”是 RAG 的真实盲区**：向量检索的假设是“答案存在于少数几个 chunk 中”。当问题需要跨整个语料归纳时，这个假设直接崩掉。GraphRAG 提醒我们：**先判断问题是检索型还是摘要型，再选架构**。
- **C0 的工程性价比极高**：对“同一语料被反复全局查询”的场景（企业知识库、情报分析），一次建图 + 根级社区摘要就能长期复用，边际查询成本极低。这是落地时最值得抄的点。
- **评测方法论也是贡献**：用 LLM 生成“用户→任务→问题”来构造无标准答案的全局问题集，再用 LLM-as-a-judge 做两两对比，并用 Directness 作为对照标准防止“全赢”幻觉——这套评测范式本身可复用到其他开放式生成任务。

## 局限与待观察点

- **评测仅限两个约 1M token 的语料**，跨领域、跨规模的泛化性仍需验证。
- **未系统比较幻觉率**：全局摘要若不能准确代表源数据，会对下游决策造成风险；论文建议配合 SelfCheckGPT 类方法并明确披露 AI 使用。
- **赋能性未见优势**：因为图索引在压缩过程中损失了具体引文/例子，这类需要“可追溯证据”的问题上 GraphRAG 未必比向量 RAG 强。
- **索引成本前置**：建图依赖大量 LLM 调用（Podcast 数据集索引耗时约 281 分钟），适合“建一次、查多次”，不适合一次性查询。

## 结论

GraphRAG 把**知识图谱生成**与**查询聚焦摘要**结合，为“对整个语料的全局 sensemaking”提供了一条可扩展的路径。在百万 token 级语料上，它在答案全面性与多样性上显著超过向量 RAG 基线，也优于无图的全文 map-reduce 摘要。对需要反复全局查询同一语料的场景，**根级社区摘要（C0）以极低的 token 成本，同时做到了超越向量 RAG、逼近其他全局方法**——这正是 GraphRAG 最具工程价值的结论。
