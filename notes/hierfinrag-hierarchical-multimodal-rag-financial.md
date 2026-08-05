---
id: hierfinrag-hierarchical-multimodal-rag-financial
title: "HierFinRAG：面向金融文档理解的层次化多模态 RAG"
category: 多模态 RAG
date: 2026-02-10
order: 1
readTime: 9 min
tags: ["Multimodal RAG", "Financial Document Understanding", "Graph Neural Network", "Symbolic Reasoning", "Table-Text Fusion"]
summary: "HierFinRAG 是针对金融文档理解的层次化多模态 RAG 框架。它将文档解析为文档-章节-段落/表格-单元格的层级结构，用 Table-Text Graph Neural Network（TTGNN）显式建模表格单元格与文本之间的语义/结构依赖，再通过 Symbolic-Neural Fusion 根据查询类型把问题路由给神经网络生成器或符号计算器。在 FinQA 上取得 82.5% 的 Exact Match，比 GPT-4o Code Interpreter 基线高 6.5 个百分点；在 FinanceBench 上达到 74.0% 准确率，同时推理延迟比 agentic 方法低约 3.5 倍。"
---

## 一句话总结

HierFinRAG 把金融文档的**层级结构、表-文关系和符号计算**统一进 RAG 流程，证明针对特定领域精心设计的静态架构可以在准确率和效率上同时超过通用 agentic 方法。

## 论文信息

- **标题**：HierFinRAG—Hierarchical Multimodal RAG for Financial Document Understanding
- **作者**：Quang-Vinh Dang, Ngoc-Son-An Nguyen, Thi-Bich-Diem Vo
- **机构**：British University Vietnam、Industrial University of Ho Chi Minh City、GiaoHangNhanh
- **期刊**：*Informatics* 2026, Vol. 13, Issue 2, p. 30
- **DOI**：[10.3390/informatics13020030](https://doi.org/10.3390/informatics13020030)
- **链接**：[MDPI](https://www.mdpi.com/2227-9709/13/2/30) · [GitHub](https://github.com/vinhqdang/HierFinRAG---Hierarchical-Multimodal-RAG-for-Financial-Document-Understanding)

## 背景：为什么这个问题重要？

金融文档（年报、10-K、财报、风险因素页）有两个鲜明特点：

1. **叙事文本与结构化表格高度交织**：关键答案往往一半在文字描述、一半在表格数字里；
2. **对数值准确性要求极高**：增长率、利润率、同比变化等一旦算错，答案就不可用。

现有 RAG 系统通常把文本和表格分开处理，导致两类失败：

- **模态隔离**：检索器只召回文本或只召回表格，无法把“表格第 4 行第 2 列”和“文字解释”联系起来；
- **数值幻觉**：LLM 直接做算术容易出错，尤其是多步百分比、比率计算。

HierFinRAG 的判断是：与其让通用 agent 动态探索，不如把金融文档的**结构先验**（章节层级、表-文引用、数值关系）固化到 RAG 架构里。

## 方法：三层设计

<figure class="figure">
  <img src="assets/papers/hierfinrag-hierarchical-multimodal-rag-financial/Fig0_Architecture.webp" alt="HierFinRAG 整体架构" loading="lazy" />
  <figcaption>图 1：HierFinRAG 整体架构。Phase 1 把金融文档解析成层级结构并构建表-文异构图；Phase 2 通过 TTGNN 编码节点关系；Phase 3 由 Symbolic-Neural Fusion 根据查询类型选择神经生成或符号计算。</figcaption>
</figure>

### 1. 层次化文档解析

HierFinRAG 先把文档解析成一棵树：

- **Document** → **Section**（章节，带层级 level）
- **Section** → **Paragraph** / **Table**
- **Table** → **Cell**（行号、列号、是否表头、数值/文本）

这样每个单元格都带有它在原文中的结构坐标，检索和推理时可以直接引用“Row 4, Col 2”而不是模糊的文本片段。

### 2. Table-Text Graph Neural Network（TTGNN）

把解析后的元素建成**异构图**：

- 节点类型：Section、Paragraph、Table、Cell；
- 边关系：章节包含段落/表格、表格包含单元格、单元格与文本中提及它的句子相连、单元格之间按行列相邻。

TTGNN 在这个异构图上传播，显式地学习表格单元格与对应文本之间的语义和结构依赖。相比把表格 flatten 成文本再 embedding，这种做法保留了“行/列/表头”的拓扑信息，也让跨模态检索（从问题到单元格，或从单元格到解释性文字）更精确。

### 3. Symbolic-Neural Fusion

推理阶段，系统先用一个 **Reasoning Router** 判断查询该走哪条路：

- **Neural**：纯文本问答，直接由 LLM 生成；
- **Symbolic**：包含 calculate、sum、growth、ratio、percentage 等关键词，或检索结果以表格/单元格为主，走符号计算器；
- **Hybrid**：LLM 先抽出计算计划（操作 + 变量），符号计算器执行，最后 LLM 把结果组织成自然语言答案。

符号计算器支持加、减、除、百分比变化等常见财务运算，精度可控，避免 LLM 在数字上“胡说”。

## 数据与评测

- **FinQA**：金融问答数据集，需要联合文本和表格进行多步数值推理；评测指标为 **Exact Match（EM）**。
- **FinanceBench**：金融文档理解基准，覆盖表格阅读、指标计算、事实问答；评测指标为 **Accuracy**。
- **基线**：Vanilla RAG、GPT-4o with Code Interpreter、Graph RAG、ColPali 等。

## 实验结果

### 主结果（表 1）

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>FinQA EM</th><th>FinanceBench Acc</th></tr></thead>
    <tbody>
      <tr><td>Vanilla RAG</td><td>45.0</td><td>32.5</td></tr>
      <tr><td>GPT-4o + Code Interpreter</td><td>76.0</td><td>48.0</td></tr>
      <tr><td><strong>HierFinRAG</strong></td><td><strong>82.5</strong></td><td><strong>74.0</strong></td></tr>
    </tbody>
  </table>
</div>

- **FinQA 上 HierFinRAG 达到 82.5% EM，比 GPT-4o Code Interpreter 高 6.5 个百分点**，比 Vanilla RAG 高 37.5 个百分点。
- **FinanceBench 上优势更大**：74.0% vs 48.0%，提升 26.0 个百分点，说明表格-文本联合推理对金融文档理解至关重要。

<figure class="figure">
  <img src="assets/papers/hierfinrag-hierarchical-multimodal-rag-financial/Fig1_Main_Performance.webp" alt="主性能对比柱状图" loading="lazy" />
  <figcaption>图 2：HierFinRAG 与基线在 FinQA 和 FinanceBench 上的主性能对比。</figcaption>
</figure>

### 检索质量：Recall@k（表 2）

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>R@1</th><th>R@5</th><th>R@10</th><th>R@20</th></tr></thead>
    <tbody>
      <tr><td>Vanilla RAG</td><td>25</td><td>45</td><td>58</td><td>65</td></tr>
      <tr><td>Graph RAG</td><td>35</td><td>60</td><td>75</td><td>82</td></tr>
      <tr><td><strong>HierFinRAG</strong></td><td><strong>48</strong></td><td><strong>78</strong></td><td><strong>89</strong></td><td><strong>94</strong></td></tr>
    </tbody>
  </table>
</div>

- HierFinRAG 的 R@5 达到 78%，比 Graph RAG（60%）高 18 个百分点，比 Vanilla RAG（45%）高 33 个百分点。
- 这说明把表格单元格和文本作为异构图节点共同编码，能显著提升跨模态检索的召回。

<figure class="figure">
  <img src="assets/papers/hierfinrag-hierarchical-multimodal-rag-financial/Fig2_Retrieval_Recall.webp" alt="Recall@k 曲线" loading="lazy" />
  <figcaption>图 3：不同方法的 Recall@k 曲线。HierFinRAG 在各个 k 值下都显著优于 Graph RAG 和 Vanilla RAG。</figcaption>
</figure>

### 效率-准确率权衡（表 3）

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>平均延迟（秒/查询）</th><th>平均准确率（%）</th></tr></thead>
    <tbody>
      <tr><td>Vanilla RAG</td><td>2.5</td><td>38.75</td></tr>
      <tr><td>ColPali</td><td>8.0</td><td>55.0</td></tr>
      <tr><td>GPT-4o + Code Interpreter</td><td>15.0</td><td>62.0</td></tr>
      <tr><td><strong>HierFinRAG</strong></td><td><strong>4.2</strong></td><td><strong>78.25</strong></td></tr>
    </tbody>
  </table>
</div>

- HierFinRAG 的准确率最高，同时延迟仅 4.2 秒，比 GPT-4o Code Interpreter（15 秒）快约 **3.5 倍**。
- 这验证了作者的论断：一个针对金融领域精心设计的**静态流水线**，可以在特定问题上同时击败通用 agentic 方法的速度和效果。

<figure class="figure">
  <img src="assets/papers/hierfinrag-hierarchical-multimodal-rag-financial/Fig3_Efficiency_Tradeoff.webp" alt="效率-准确率散点图" loading="lazy" />
  <figcaption>图 4：效率-准确率散点图。HierFinRAG 位于左上角，兼顾高准确率与低延迟。</figcaption>
</figure>

### 错误分析

<div class="table-wrap">
  <table>
    <thead><tr><th>错误类型</th><th>占比</th></tr></thead>
    <tbody>
      <tr><td>检索错误（Retrieval）</td><td>40%</td></tr>
      <tr><td>推理错误（Reasoning）</td><td>30%</td></tr>
      <tr><td>生成错误（Generation）</td><td>20%</td></tr>
      <tr><td>跨引用错误（Cross-Reference）</td><td>10%</td></tr>
    </tbody>
  </table>
</div>

- 检索仍是最大错误来源（40%），说明即使有了 TTGNN，跨模态检索仍有提升空间；
- 跨引用错误（10%）指表格与文本之间的引用关系没对齐，是金融文档特有的难点。

## Ablation：各组件贡献（表 4）

<div class="table-wrap">
  <table>
    <thead><tr><th>配置</th><th>FinQA EM</th><th>FinanceBench Acc</th><th>Retrieval R@5</th></tr></thead>
    <tbody>
      <tr><td><strong>Full Model</strong></td><td><strong>82.5%</strong></td><td><strong>74.0%</strong></td><td><strong>78.0%</strong></td></tr>
      <tr><td>No Hierarchy</td><td>78.0%</td><td>69.5%</td><td>65.0%</td></tr>
      <tr><td>No Graph (TTGNN)</td><td>75.5%</td><td>66.0%</td><td>62.0%</td></tr>
      <tr><td>No Symbolic</td><td>70.0%</td><td>62.5%</td><td>78.0%</td></tr>
      <tr><td>No Reranking</td><td>79.2%</td><td>71.0%</td><td>72.0%</td></tr>
    </tbody>
  </table>
</div>

- **去掉 TTGNN** 对 FinanceBench 打击最大（-8.0%），因为该数据集高度依赖表格-文本关系；
- **去掉 Symbolic 计算** 对 FinQA 打击最大（-12.5%），说明多步数值推理离不开符号执行；
- **Hierarchy 和 Reranking** 各自贡献约 3–4.5 个百分点的 FinQA EM，说明层级结构和重排都是必要的。

### 成本分析（表 5）

<div class="table-wrap">
  <table>
    <thead><tr><th>数据集</th><th>平均输入 token</th><th>平均输出 token</th><th>1k 查询估算成本</th></tr></thead>
    <tbody>
      <tr><td>FinQA</td><td>1,250</td><td>150</td><td>$6.20</td></tr>
      <tr><td>FinanceBench</td><td>4,500</td><td>300</td><td>$21.50</td></tr>
    </tbody>
  </table>
</div>

- FinanceBench 的输入 token 远高于 FinQA，因为文档更长、表格更多；
- HierFinRAG 通过精确的图检索和符号计算，把需要交给 LLM 的上下文控制在较小范围内，从而控制成本。

## 我的理解与启发

- **领域专用架构可以比通用 agent 更高效**。HierFinRAG 没有使用 ReAct、Reflection 等动态 agent 机制，而是把金融文档的结构先验（章节、表格、数值关系）嵌入静态流水线，结果在速度和准确率上都超过了 GPT-4o Code Interpreter。这对其他垂直领域（法律、医疗、政务）是个重要信号：先建好领域结构，再谈 agent。
- **表-文异构图是金融 RAG 的关键表示**。把表格单元格当节点、把文本提及当边，比把表格拍平成文本更能保留“行列关系”和“跨引用”。TTGNN 的检索 R@5 提升 18–33 个百分点，说明图结构对跨模态检索非常有效。
- **符号计算不是可选项，而是刚需**。在 FinQA 上，去掉 symbolic 模块后 EM 从 82.5% 掉到 70.0%，下降 12.5 个百分点。任何涉及数字推理的 RAG 系统都应该把“提取数字 -> 选择运算符 -> 符号执行 -> 校验”作为一等公民，而不是指望 LLM 心算。
- **Routing 设计值得借鉴**。Neural / Symbolic / Hybrid 三档路由虽然实现简单，但切中了金融查询的真实分布：有的问题只需要读文字，有的必须算数字，有的需要先抽计划再算。后续可以把这个路由做成可学习的轻量级分类器。

## 局限与待观察点

1. **解析层依赖结构化输入**：论文代码中的 `JSONParser` 期望文档已经是 JSON 格式（章节、段落、表格、单元格）。如何从原始 PDF/HTML 年报自动、准确地得到这种结构，论文没有深入讨论。
2. **TTGNN 训练与规模**：代码实现是原型级别，真实金融文档可能有数百页、数千个单元格，图规模和训练稳定性需要验证。
3. **符号运算覆盖有限**：当前 SymbolicCalculator 只支持加减除、比率、百分比变化等简单操作。面对更复杂的财务公式（NPV、折现、多步归因），需要扩展运算符库。
4. **未在更大数据集上验证**：仅在 FinQA 和 FinanceBench 上评测，且 FinanceBench 的具体子集信息有限；对中文财报、非标准表格的泛化能力未知。
5. **数值提取仍依赖 LLM**：虽然计算是符号的，但“从表格/文本中抽取出正确数值”这一步仍由 LLM 完成，存在抽取错误风险。

## 结论

HierFinRAG 针对金融文档理解提出了一个清晰的垂直 RAG 范式：**层次化解析 → 表-文异构图 → 符号-神经融合**。它在 FinQA 和 FinanceBench 上同时取得了领先的准确率和较低的延迟，证明了把领域结构先验固化进 RAG 架构的价值。对于任何需要处理“文本 + 表格 + 数字”的垂直场景，HierFinRAG 的设计思路都直接可用。
