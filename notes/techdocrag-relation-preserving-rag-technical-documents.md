---
id: techdocrag-relation-preserving-rag-technical-documents
title: "TechDocRAG：面向技术文档的关系保持 RAG"
category: 多模态 RAG
date: 2026-05-06
order: 1
readTime: 10 min
tags: ["Multimodal RAG", "Technical Document QA", "Document Structure", "Evidence Grounding", "Graph Retrieval"]
summary: "TechDocRAG 提出一种面向技术文档（手册、工程规范、标准）的关系保持检索增强生成框架。它把每份文档解析为异构元素图（条款、段落、表格、图形、步骤、标题等），为每个元素同步维护标识符、语义摘要和原始证据三种视图，并按查询意图进行标识符召回 → 图扩展 → 摘要重排 → 原始证据打包的粗到细检索。在 MPMQA、DesignQA、MMLongBench-Doc、LongDocURL 四个基准共 7500+ QA 对上，TechDocRAG 比最强扁平基线平均提升 20.3 分，比最强非扁平基线提升 9.3 分；严格原始证据命中率从 0.510 提升到 0.942，查询延迟与标准混合检索相当。"
---

## 一句话总结

TechDocRAG 通过把技术文档中的**元素与元素间引用关系显式保留为图**，让检索不再返回孤立的文本块，而是返回关系完整、可追溯的证据子图，从而显著提升技术文档问答的准确性与可解释性。

## 论文信息

- **标题**：TechDocRAG: Relation-Preserving Retrieval-Augmented Generation (RAG) for Technical Documents
- **作者**：Seungjoon Lee、Myungryul Choi
- **机构**：Hanyang University（韩国汉阳大学）
- **发表**：*AI* 2026, 7(5), 161
- **DOI**：[10.3390/ai7050161](https://doi.org/10.3390/ai7050161)
- **关键词**：technical document question answering; retrieval-augmented generation; multimodal document understanding; evidence grounding; document structure analysis; technical standards

## 研究背景

技术文档（产品手册、工程规范、标准、维护指南）与普通开放域文本有两个核心差异：

1. **证据分散**：一个答案的证据往往分散在编号条款、表格、图示、图注、有序步骤中，而不是单一段落。
2. **强标识符依赖**：问题通常由精确标识符锚定，如条款号、参数名、图/表标签、版本号、修订标签。

传统 RAG 把这些元素扁平切分成独立 chunk，会破坏文档原本的结构和引用关系，导致检索到的上下文“语义相关但证据不完整”——例如只召回条款却漏掉该条款引用的表格或后续步骤。因此，技术文档 RAG 的关键不是检索更多段落，而是**保持使证据可解释的关系链**。

## 方法

### 1. 技术文档 = 异构元素图

TechDocRAG 将文档 $d$ 建模为异构元素图 $G_d=(V_d,E_d)$：

- **节点** $v$ 对应文档元素：条款（clause）、段落、表格、图形、图注、章节标题、步骤等。
- **节点属性**：$v=(\tau_v, r_v, k_v, s_v, m_v)$，分别表示元素类型、原始对象、技术标识符/关键词、语义摘要、元数据（页码、边界框、章节路径、版本、规范标签）。
- **边** $E_d$ 同时保留：
  - **结构边**：contains、precedes、same_section、step_next
  - **引用边**：clause_ref、table_ref、figure_ref、caption_of、same_identifier、version_of / supersedes

检索目标被形式化为选择关系完整且相关的证据子图：

$$
H_q^* = \arg\max_{H \subseteq G} \bigl[\text{Rel}(q,H) + \lambda\,\text{Conn}(H) + \gamma\,\text{Valid}(q,H) - \mu\,\text{Cost}(H)\bigr]
$$

### 2. 三种视图的数据库

每个元素同步存入三个对齐视图：

- **标识符索引 $I_{id}$**：BM25 稀疏索引，覆盖条款号、章节路径、参数名、命令、API 名、错误码、图/表标签、版本标签等。
- **摘要向量索引 $I_{sum}$**：all-MiniLM-L6-v2 语义摘要及其局部上下文。
- **原始证据库 $R$**：原始文本片段、结构化表格、图区域+图注、有序步骤段。
- **关系库 $G$**：元素图。

### 3. 查询分析与意图感知图扩展

查询首先被分解为：

$$
(k_q, e_q, z_q) = \text{Analyze}(q)
$$

其中 $k_q$ 为提取的标识符/关键词，$e_q$ 为语义表示，$z_q$ 为查询意图。意图决定图扩展策略 $\Omega(z_q)$。TechDocRAG 定义了 6 类意图：

<div class="table-wrap">
<table>
<thead>
<tr><th>查询意图</th><th>主要扩展关系</th><th>Max Hop</th><th>证据包组成</th><th>示例</th></tr>
</thead>
<tbody>
<tr><td>定义/要求查询</td><td>contains, same_identifier, clause_ref</td><td>1–2</td><td>匹配条款、所在段落、规范说明及关联表格</td><td>“What does parameter X mean in Section 4.2?”</td></tr>
<tr><td>流程/故障排查</td><td>step_next, contains, same_section</td><td>2</td><td>目标步骤、前后步骤、警告块、章节标题</td><td>“How do I recalibrate sensor Y?”</td></tr>
<tr><td>文本–表格推理</td><td>table_ref, caption_of, same_identifier</td><td>2</td><td>引用条款、相关行/单元格、表头路径、表注</td><td>“Which temperature range is allowed for mode A?”</td></tr>
<tr><td>文本–图形定位</td><td>figure_ref, caption_of, same_section</td><td>2</td><td>图裁剪、图注、引用段落</td><td>“What does Figure 3 indicate about the connector layout?”</td></tr>
<tr><td>交叉引用解析</td><td>clause_ref, table_ref, figure_ref</td><td>2</td><td>源条款 + 被引条款/表/图及其上下文</td><td>“In Section 5.2, what does Table 7 clarify?”</td></tr>
<tr><td>版本敏感查询</td><td>version_of, supersedes, same_identifier</td><td>1</td><td>当前条款、匹配的历史修订节点、版本元数据</td><td>“What changed for parameter Z in revision B?”</td></tr>
</tbody>
</table>
</div>

### 4. 粗到细检索与可溯源生成

在线检索分四步：

1. **标识符感知召回**（Identifier-aware recall）：BM25 + 标识符匹配 + 元数据匹配，Top-10。
2. **意图感知图扩展**：按 $\Omega(z_q)$ 在元素图上 2-hop 扩展。
3. **摘要级重排**：语义相似度 + 相关性分数 + 类型先验，Top-5。
4. **原始证据打包**：把摘要节点解析为关系完整的原始证据包（如条款+引用表格、图+图注+引用段落），在 2048 token + 10 视觉区域预算下打包，送入生成器。

生成同时输出**声明级溯源** $\Pi_q = \{(c_i, U_i)\}$，每个生成声明 $c_i$ 都对应一组原始证据对象。

<figure class="figure"><img src="assets/papers/techdocrag-relation-preserving-rag-technical-documents/fig1_architecture.webp" alt="TechDocRAG 整体架构" /><figcaption>图1：TechDocRAG 整体架构（离线元素图构建 + 在线检索与生成）。</figcaption></figure>

<figure class="figure"><img src="assets/papers/techdocrag-relation-preserving-rag-technical-documents/fig2_flowchart.webp" alt="TechDocRAG 查询时控制流程" /><figcaption>图2：查询时控制流程：标识符提取与意图预测决定关系扩展策略，随后进行摘要重排、原始证据打包与可溯源生成。</figcaption></figure>

## 实验

### 数据集与基线

- **MPMQA**：209 份产品手册，22k QA。
- **DesignQA**：工程规范 + CAD 图/工程图，4k QA。
- **MMLongBench-Doc**：130 份长 PDF，平均 49.4 页，1k QA。
- **LongDocURL**：33k+ 页，2k QA。

公平比较控制为 7500+ QA 子集，所有方法共享答案生成器 Gemini-3.1-Flash-Lite-Preview，上下文预算统一为 2048 token + 10 视觉区域。

基线包括 Dense Chunk RAG、Hybrid (Dense+BM25)、Self-RAG、CRAG、RAPTOR、GraphRAG、LightRAG、HippoRAG2、VisRAG。

### 主结果

<div class="table-wrap">
<table>
<thead>
<tr><th>方法</th><th>MPMQA ↑</th><th>DesignQA ↑</th><th>MMLong ↑</th><th>LongDoc ↑</th><th>Avg. Gain ↑</th></tr>
</thead>
<tbody>
<tr><td>Dense Chunk RAG</td><td>44.8</td><td>40.5</td><td>33.2</td><td>31.4</td><td>–</td></tr>
<tr><td>Hybrid (Dense+BM25)</td><td>48.2</td><td>44.1</td><td>36.5</td><td>34.2</td><td>+3.4</td></tr>
<tr><td>Self-RAG</td><td>51.8</td><td>47.5</td><td>39.8</td><td>37.5</td><td>+6.8</td></tr>
<tr><td>CRAG</td><td>52.4</td><td>48.8</td><td>40.5</td><td>38.2</td><td>+7.5</td></tr>
<tr><td>RAPTOR</td><td>50.5</td><td>46.2</td><td>41.4</td><td>38.8</td><td>+5.4</td></tr>
<tr><td>GraphRAG</td><td>54.8</td><td>50.5</td><td>44.8</td><td>42.4</td><td>+9.8</td></tr>
<tr><td>LightRAG</td><td>55.6</td><td>51.2</td><td>46.1</td><td>43.8</td><td>+10.7</td></tr>
<tr><td>HippoRAG2</td><td>56.5</td><td>52.8</td><td>47.5</td><td>45.2</td><td>+11.6</td></tr>
<tr><td>VisRAG</td><td>54.2</td><td>57.5</td><td>48.8</td><td>46.5</td><td>+12.6</td></tr>
<tr><td><strong>TechDocRAG (Ours)</strong></td><td><strong>68.5</strong></td><td><strong>62.2</strong></td><td><strong>58.4</strong></td><td><strong>55.2</strong></td><td><strong>+21.2</strong></td></tr>
</tbody>
</table>
</div>

TechDocRAG 在四个基准上全部最优，平均比最强扁平基线高 **20.3 分**，比最强非扁平基线高 **9.3 分**。

### 检索与溯源质量

<div class="table-wrap">
<table>
<thead>
<tr><th>方法</th><th>Recall@10 ↑</th><th>MRR ↑</th><th>REHR@10 ↑</th><th>SRTA ↑</th><th>ECR ↑</th><th>CSR ↑</th></tr>
</thead>
<tbody>
<tr><td>Dense Chunk RAG</td><td>42.5</td><td>0.28</td><td>0.452</td><td>0.412</td><td>0.382</td><td>0.425</td></tr>
<tr><td>Hybrid (Dense+BM25)</td><td>45.8</td><td>0.31</td><td>0.510</td><td>0.485</td><td>0.425</td><td>0.512</td></tr>
<tr><td>Self-RAG</td><td>48.2</td><td>0.34</td><td>0.542</td><td>0.512</td><td>0.451</td><td>0.545</td></tr>
<tr><td>CRAG</td><td>49.4</td><td>0.35</td><td>0.551</td><td>0.522</td><td>0.464</td><td>0.552</td></tr>
<tr><td>RAPTOR</td><td>52.1</td><td>0.38</td><td>0.582</td><td>0.574</td><td>0.505</td><td>0.578</td></tr>
<tr><td>GraphRAG</td><td>55.5</td><td>0.41</td><td>0.654</td><td>0.622</td><td>0.585</td><td>0.624</td></tr>
<tr><td>LightRAG</td><td>56.8</td><td>0.43</td><td>0.682</td><td>0.654</td><td>0.621</td><td>0.652</td></tr>
<tr><td>HippoRAG2</td><td>58.5</td><td>0.45</td><td>0.721</td><td>0.685</td><td>0.652</td><td>0.704</td></tr>
<tr><td>VisRAG</td><td>54.2</td><td>0.40</td><td>0.654</td><td>0.611</td><td>0.582</td><td>0.625</td></tr>
<tr><td><strong>TechDocRAG</strong></td><td><strong>69.2</strong></td><td><strong>0.56</strong></td><td><strong>0.942</strong></td><td><strong>0.914</strong></td><td><strong>0.852</strong></td><td><strong>0.942</strong></td></tr>
</tbody>
</table>
</div>

关键发现：

- **REHR（严格原始证据命中率）**从 0.510（Hybrid）提升到 **0.942**，说明 TechDocRAG 不仅找到语义相关内容，更频繁地召回精确证据节点。
- **SRTA / ECR / CSR** 的高企说明摘要-原始对齐与图连接确实保留了证据链。

### 按查询类型分解

<div class="table-wrap">
<table>
<thead>
<tr><th>方法</th><th>Clause</th><th>Param.</th><th>Procedure</th><th>Text–Table</th><th>Text–Figure</th><th>Cross-Ref</th><th>Version</th><th>Macro Avg.</th></tr>
</thead>
<tbody>
<tr><td>Hybrid Chunk RAG</td><td>52.1</td><td>48.4</td><td>35.2</td><td>31.4</td><td>28.5</td><td>25.1</td><td>22.4</td><td>34.7</td></tr>
<tr><td>Self-RAG</td><td>54.5</td><td>51.2</td><td>38.4</td><td>34.2</td><td>31.0</td><td>28.5</td><td>24.8</td><td>37.5</td></tr>
<tr><td>CRAG</td><td>55.2</td><td>52.5</td><td>40.1</td><td>35.8</td><td>32.4</td><td>30.2</td><td>26.5</td><td>38.9</td></tr>
<tr><td>RAPTOR</td><td>58.4</td><td>54.1</td><td>44.2</td><td>39.5</td><td>35.8</td><td>34.2</td><td>30.1</td><td>42.3</td></tr>
<tr><td>GraphRAG</td><td>61.2</td><td>58.5</td><td>48.5</td><td>44.2</td><td>40.1</td><td>38.5</td><td>34.2</td><td>46.4</td></tr>
<tr><td>LightRAG</td><td>58.8</td><td>56.4</td><td>46.1</td><td>42.5</td><td>38.2</td><td>36.4</td><td>32.5</td><td>44.4</td></tr>
<tr><td>HippoRAG2</td><td>62.5</td><td>60.1</td><td>51.2</td><td>47.5</td><td>43.4</td><td>41.2</td><td>37.5</td><td>49.1</td></tr>
<tr><td>VisRAG</td><td>55.4</td><td>53.2</td><td>42.1</td><td>49.5</td><td>52.4</td><td>32.1</td><td>28.4</td><td>44.7</td></tr>
<tr><td><strong>TechDocRAG</strong></td><td><strong>65.8</strong></td><td><strong>64.2</strong></td><td><strong>62.1</strong></td><td><strong>60.5</strong></td><td><strong>58.4</strong></td><td><strong>57.2</strong></td><td><strong>54.1</strong></td><td><strong>60.3</strong></td></tr>
</tbody>
</table>
</div>

提升在 **流程、文本–表格、文本–图形、交叉引用、版本敏感** 类查询上尤为显著，这些正是需要把多个文档元素作为关联证据进行推理的场景。

### 消融实验

<div class="table-wrap">
<table>
<thead>
<tr><th>变体</th><th>Main Score ↑</th><th>REHR@10 ↑</th><th>ECR ↑</th><th>CSR ↑</th><th>VCS ↑</th><th>POA ↑</th><th>Latency(s) ↓</th></tr>
</thead>
<tbody>
<tr><td>Full Model</td><td>62.5</td><td>0.94</td><td>0.85</td><td>0.82</td><td>0.78</td><td>0.75</td><td>0.42</td></tr>
<tr><td>w/o Relation Edges</td><td>51.2</td><td>0.88</td><td>0.42</td><td>0.65</td><td>0.74</td><td>0.48</td><td>0.35</td></tr>
<tr><td>w/o Identifier Recall</td><td>54.8</td><td>0.65</td><td>0.72</td><td>0.75</td><td>0.68</td><td>0.65</td><td>0.32</td></tr>
<tr><td>w/o Summary Routing</td><td>56.4</td><td>0.91</td><td>0.81</td><td>0.78</td><td>0.72</td><td>0.70</td><td>0.38</td></tr>
<tr><td>w/o Raw Bundling</td><td>58.2</td><td>0.92</td><td>0.83</td><td>0.45</td><td>0.75</td><td>0.72</td><td>0.36</td></tr>
<tr><td>w/o Intent-Aware Expansion</td><td>55.6</td><td>0.90</td><td>0.58</td><td>0.72</td><td>0.74</td><td>0.61</td><td>0.38</td></tr>
<tr><td>w/o Version Metadata</td><td>59.4</td><td>0.93</td><td>0.84</td><td>0.81</td><td>0.32</td><td>0.74</td><td>0.40</td></tr>
</tbody>
</table>
</div>

- 移除 **关系边**：ECR 从 0.85 掉到 0.42，影响最大。
- 移除 **标识符召回**：REHR 掉到 0.65，说明精确锚点是入口。
- 移除 **原始证据打包**：CSR 掉到 0.45，说明把碎片化证据组装成连贯证据包对生成质量至关重要。

### 资源开销与鲁棒性

<div class="table-wrap">
<table>
<thead>
<tr><th>方法</th><th>Offline Indexing Time(s) ↓</th><th>Index Size(MB) ↓</th><th>Avg. Query Latency(ms) ↓</th><th>Peak GPU Memory(MB) ↓</th></tr>
</thead>
<tbody>
<tr><td>Dense Chunk RAG</td><td>2.5</td><td>2.0</td><td>5.2</td><td>~150</td></tr>
<tr><td>Hybrid (Dense+BM25)</td><td>3.8</td><td>3.5</td><td>7.8</td><td>~180</td></tr>
<tr><td>HippoRAG2</td><td>12.4</td><td>8.2</td><td>45.6</td><td>~450</td></tr>
<tr><td><strong>TechDocRAG (Ours)</strong></td><td><strong>6.36</strong></td><td><strong>2.28</strong></td><td><strong>8.3</strong></td><td><strong>314.4</strong></td></tr>
</tbody>
</table>
</div>

<div class="table-wrap">
<table>
<thead>
<tr><th>Noise/Loss Level</th><th>REHR (ID Corruption) ↑</th><th>ECR (Edge Dropout) ↑</th></tr>
</thead>
<tbody>
<tr><td>0% (baseline)</td><td>1.0000</td><td>0.9875</td></tr>
<tr><td>5%</td><td>1.0000</td><td>0.9825</td></tr>
<tr><td>10%</td><td>1.0000</td><td>0.9725</td></tr>
<tr><td>20%</td><td>0.0700</td><td>0.9525</td></tr>
<tr><td>30%</td><td>–</td><td>0.9310</td></tr>
</tbody>
</table>
</div>

- 查询延迟接近标准 Hybrid 检索，远低于 HippoRAG2。
- **关系边随机缺失 30%** 时 ECR 仍保持 0.93，说明系统对关系丢失较鲁棒。
- **标识符损坏超过 10%** 时 REHR 急剧崩塌，说明第一阶段精确标识符召回依赖可靠的解析。

## 结论

TechDocRAG 的核心洞见是：**技术文档问答的瓶颈往往不是“找不到相关文本”，而是“找回来的证据不完整”**。通过把文档解析为异构元素图、为每个元素维护标识符/摘要/原始证据三种视图、并按意图进行关系保持的检索，TechDocRAG 在四个技术文档基准上取得了显著且一致的提升，同时保持了合理的查询延迟。

其局限性也提醒部署者：系统仍依赖高质量的 PDF/OCR 解析与标识符规范化；对隐式引用、非标准版式、跨版本歧义的处理仍是未来工作。
