---
id: "vat-kg-visual-audio-text-knowledge-graph"
title: "VAT-KG：面向多模态 RAG 的知识密集型视觉-音频-文本知识图谱"
category: "多模态 RAG"
date: "2026-07-10"
order: 47
readTime: "14 min"
tags: ["RAG","Multimodal","Knowledge Graph","MMKG","Audio-Visual","MLLM"]
summary: "VAT-KG 是首个同时覆盖视觉、音频、文本三模态且概念中心（concept-centric）的知识密集型多模态知识图谱。它用一条四阶段过滤-对齐流水线从任意多模态语料自动构建 MMKG，每个三元组都链接到多模态数据并附带细粒度概念描述；配套的多模态 RAG 框架支持任意模态查询检索概念级知识，并用 Retrieval Checker 过滤误配结果，在 AQA/VQA/AVQA 任务上为 VideoLLaMA2 与 Qwen2.5-Omni 带来一致提升。"
---

## 一句话总结

多模态大模型（MLLM）的隐式知识有限，需要外部多模态知识图谱（MMKG）来做接地推理，但现有 MMKG 要么只覆盖图文两种模态、要么只做“实体-实体”连接而缺乏概念级细粒度描述。VAT-KG 的核心贡献是：**提出首个同时覆盖视觉/音频/文本、以概念为中心、且每个三元组都链接多模态数据与细粒度描述的知识图谱，并给出一条能从任意多模态语料自动构建它的严格过滤-对齐流水线**，再配一个支持任意模态查询、带检索校验模块的多模态 RAG 框架。

<div class="metric-grid">
  <div class="metric"><strong>4 模态</strong><span>文本/图像/音频/视频全覆盖（唯一同时做到概念中心的 MMKG）</span></div>
  <div class="metric"><strong>110,786</strong><span>严格四步过滤后保留的高对齐多模态样本</span></div>
  <div class="metric"><strong>+2.88 / +3.02</strong><span>VALOR(AVQA) 上 VideoLLaMA2 / Qwen2.5-Omni 的 M.J. 分数提升</span></div>
  <div class="metric"><strong>~98%</strong><span>人工评估下重述与三元组构建阶段的正确率</span></div>
</div>

## 论文信息

- **标题**：VAT-KG: Knowledge-Intensive Multimodal Knowledge Graph Dataset for Retrieval-Augmented Generation
- **作者**：Hyeongcheol Park, Jiyoung Seo, MinHyuk Jang, Hogun Park, Ha Dam Baek, Gyusam Chang, Hyeonsoo Im, Sangpil Kim
- **机构**：Korea University / Sungkyunkwan University / Hanwha Systems
- **版本**：arXiv:2506.21556v3，2025-09-26 修订（v1 为 2025-06-11）
- **链接**：[arXiv](https://arxiv.org/abs/2506.21556) · [PDF](https://arxiv.org/pdf/2506.21556) · [项目主页](https://vatkg.github.io/) · [数据/代码 (HuggingFace)](https://huggingface.co/vatkg)

## 背景：现有 MMKG 为什么撑不起多模态 RAG？

MMKG 把跨模态的显式知识结构化，用来补足 MLLM 的隐式知识、支撑更接地的 RAG 推理。但作者指出现有 MMKG 有两类系统性缺陷：

- **模态覆盖窄**：大多只有文本 + 图像。而近期的 MLLM（如 Qwen2.5-Omni、VideoLLaMA2）已经能联合理解视频与音频，图文 KG 无法为“需要同时听和看”的查询提供接地知识。
- **不是概念中心**：很多 MMKG（如 TIVA-KG）虽然模态多，但组织方式是“实体到实体”的连接，例如 `(quokka; IsA; mammal)`。这类三元组无法提供关于概念本身的**细粒度描述**，对回答知识密集问题帮助有限。少数概念中心的 KG（M2ConceptBase、VTKG）又退回到只支持图文两模态。

<figure class="figure">
  <img src="assets/papers/vat-kg/x1_comparison.webp" alt="VAT-KG 与现有 MMKG 对比" loading="lazy" />
  <figcaption>图 1：与现有 MMKG 的对比。(a) 覆盖多模态但非概念中心（只做实体连接）；(b) 概念中心但模态受限（仅图文）；(c) VAT-KG 同时做到概念中心 + 四模态覆盖，每个概念都链接细粒度描述。</figcaption>
</figure>

VAT-KG 的定位是**同时补齐这两块短板**：既是概念中心（每个 head/tail 概念都挂一段详细描述），又覆盖视觉-音频-文本，从而支持任意模态查询检索概念级知识。

## 方法一：从任意多模态语料自动构建 VAT-KG

VAT-KG 的构建是一条**四阶段流水线**，核心思想是先用严格过滤保证跨模态一致性，再用 LLM 做知识增强，最后把三元组与外部知识库的细粒度描述对齐。

<figure class="figure">
  <img src="assets/papers/vat-kg/x3_pipeline.webp" alt="VAT-KG 构建流水线总览" loading="lazy" />
  <figcaption>图 2：VAT-KG 构建流水线的四个阶段——① 多模态对齐过滤；② 知识密集型重述；③ 多模态三元组接地；④ 跨模态描述对齐。</figcaption>
</figure>

### 1. 多模态对齐过滤（Multimodal Alignment Filtering）

为了只保留视觉、音频、文本三者高度一致的样本，流水线依次做音频标注（Audio Tagging）、音频-文本相关性、视频-文本相关性过滤。其中 Audio-Text 过滤用 **CLAP** 计算音频与文本嵌入的余弦相似度，低于 0.2 直接丢弃。从 4 个来源（InternVid-FLT 的 10%、AudioCaps、AVQA、VALOR-32k）共约 116 万条样本，最终只留下 **110,786** 条，过滤相当激进——作者认为宁可小而精。

### 2. 知识密集型重述（Knowledge-Intensive Recaptioning）

原始 caption 往往只有粗粒度概念（如 “owl”）。这一步用 **DeepSeek-R1-Distill-Llama-70B**（用开源大模型替代昂贵的 GPT-4 API）结合 YouTube 元信息（标题、描述）把文本重写成知识密集的描述，把粗概念细化成更具体的概念（如 “desert tawny owl”）。

### 3. 多模态三元组接地（Multimodal Triplet Grounding）

同样用 DeepSeek-R1-70B，通过带 in-context 示例的提示从知识密集文本中抽取候选三元组，并选出**最能对齐当前多模态上下文**的那一个。这一步把纯文本三元组与视频/音频真正“接地”。

### 4. 跨模态描述对齐（Cross-Modal Description Alignment）

最后从外部知识库检索概念级的细粒度描述，并匹配到每个多模态三元组。这里用较轻的 **DeepSeek-R1-Distill-Llama-8B** 作为辅助知识库生成候选描述（因为它只是子知识库，选轻量模型降低推理成本）。

> **为什么这样设计**：整条流水线的关键取舍是“质量优先于规模”——通过严格过滤 + LLM 结构化任务（重述/抽取都是有据可依的改写，而非开放生成），把幻觉风险压到最低。人工评估显示阶段 2、3 的正确率分别达 98.84% 与 98.40%，最终三元组的概念准确率、内容捕获保真度、描述对齐度均在 95-97%。

## 方法二：面向任意模态查询的多模态 RAG 框架

由于现有 RAG 框架不联合考虑视频、音频、文本，作者针对 VAT-KG 的模态覆盖设计了配套 RAG 框架，支持 AQA、VQA、AVQA 三类任务，由三个组件组成。

<figure class="figure">
  <img src="assets/papers/vat-kg/x5_rag_framework.webp" alt="多模态 RAG 框架总览" loading="lazy" />
  <figcaption>图 3：多模态 RAG 框架。① 模态无关检索按嵌入相似度取回最多 5 条相关三元组；② Retrieval Checker 用同一基础模型的文本编码器过滤误配三元组；③ 用支持音视频理解的 MLLM 做增强生成。</figcaption>
</figure>

### 1. 模态无关检索（Modality-Agnostic Retrieval）

用合适的多模态基础模型编码查询（视频用 **ViCLIP**、音频用 **CLAP**）得到查询嵌入，然后在**与查询同模态**的三元组嵌入上做相似度检索，取回预设 L2 距离阈值内、最多 5 条相关三元组。嵌入空间用 **FAISS** 索引以加速。对 AVQA，直接把音频与视频嵌入拼接成联合查询嵌入，用同样的距离度量检索。

### 2. Retrieval Checker（检索校验）

这是提升质量的关键小模块：用同一多模态基础模型的**文本编码器**再校验一遍，把与查询语义不对齐的三元组过滤掉，避免误配知识反而加剧幻觉。

### 3. MLLM 增强生成

把输入问题连同检索到的 head/tail 概念及其描述一起喂给支持音视频联合理解的 MLLM，完成融合概念级知识的多模态 RAG 生成。

## 数据与评测

- **构建来源**：InternVid-FLT(10%)、AudioCaps、AVQA、VALOR-32k。
- **评测任务与基准**：AQA 用 AudioCaps-QA；VQA 用 VideoChatGPT (VCGPT)；AVQA 用 AVQA 与 VALOR。
- **基线 MLLM**：VideoLLaMA2、Qwen2.5-Omni（均支持音视频联合处理），另在附录评测 GPT-4o、Gemini-2.5。
- **对比 KG**：Wikidata5M（纯文本）、VTKG（图文概念中心）、M2ConceptBase（图文概念中心，主要基线）。
- **评测指标**：多数为开放式 QA，采用开源的 Model-as-Judge (M.J.) 打分，并辅以人工评估。

## 实验结果：多模态知识才真正有用

在整体性能上（Model-as-Judge 分数，越高越好），只有 VAT-KG 在所有任务上带来一致提升，而其他 KG 常常只有边际收益甚至**掉点**：

<table>
  <thead>
    <tr><th>基线</th><th>知识图谱</th><th>AudioCaps-QA (AQA)</th><th>VCGPT (VQA)</th><th>AVQA</th><th>VALOR (AVQA)</th></tr>
  </thead>
  <tbody>
    <tr><td>VideoLLaMA2</td><td>None</td><td>43.13</td><td>39.09</td><td>93.19</td><td>25.66</td></tr>
    <tr><td>VideoLLaMA2</td><td>Wikidata</td><td>43.58</td><td>38.58</td><td>92.70</td><td>26.43</td></tr>
    <tr><td>VideoLLaMA2</td><td>M2ConceptBase</td><td>42.19</td><td>39.31</td><td>92.92</td><td>25.93</td></tr>
    <tr><td>VideoLLaMA2</td><td><strong>VAT-KG</strong></td><td><strong>44.60</strong></td><td><strong>39.42</strong></td><td><strong>93.28</strong></td><td><strong>28.30</strong></td></tr>
    <tr><td>Qwen2.5-Omni</td><td>None</td><td>49.00</td><td>42.21</td><td>93.05</td><td>32.42</td></tr>
    <tr><td>Qwen2.5-Omni</td><td>M2ConceptBase</td><td>49.78</td><td>42.78</td><td>92.28</td><td>32.31</td></tr>
    <tr><td>Qwen2.5-Omni</td><td><strong>VAT-KG</strong></td><td><strong>51.30</strong></td><td><strong>43.50</strong></td><td><strong>93.07</strong></td><td><strong>35.44</strong></td></tr>
  </tbody>
</table>

- **AVQA 增益最明显**：VALOR 上 VideoLLaMA2 从 25.66 → 28.30、Qwen2.5-Omni 从 32.42 → 35.44。这类查询需要同时考虑音视频的时空特征，而其他图文 KG 往往**掉点**，说明现有大规模 KG/MMKG 的接地被限制在图文，难以支撑真实的音视频理解。
- **定性对比**：无 RAG 时模型会误认场景或幻觉（如把视频里的乐器认错）；用 M2ConceptBase 检索的知识常与查询语境错配、反而加剧幻觉；用 VAT-KG 后模型能准确识别“合成器在演奏音乐”等概念。
- **商用模型同样受益**：GPT-4o、Gemini-2.5-Flash 接入 VAT-KG 后在各基准一致提升（如 GPT-4o 在 VALOR 上 46.02 → 55.86）。
- **人工评估印证**：Qwen2.5-Omni + VAT-KG 在四个基准的人工评分全面领先（如 AVQA 68.78 → 79.00）。
- **统计显著性**：10 次独立实验的 t 检验 p 值普遍 < 0.10，VALOR 上更是 p=2.2e-14，说明提升并非评判器噪声。

## Ablation：哪些设计最关键？

- **多模态覆盖是核心价值**：在 VALOR(AVQA) 上比较仅音频/仅图像/仅视频/音频+视频四种检索策略，虽然每种单模态检索都优于无 RAG 基线，但**同时用音频与视频检索收益最大**——直接验证了 VAT-KG 覆盖多模态的意义。
- **Retrieval Checker 有效**：去掉校验模块后各任务均下降（如 VideoLLaMA2 在 VALOR 上 28.30 → 26.02），说明过滤误配知识对最终质量确有帮助。
- **知识密集诊断集**：在从 VALOR 采样的 10 个含挑战性概念（tiltrotor、locomotive、electric organ 等）的诊断集上，VAT-KG 带来大幅提升——GPT-4o 从 44 → 74、Qwen2.5-Omni 从 26 → 46，凸显其在知识密集场景的价值。

## 我的理解与启发

- **“概念中心 + 多模态接地”是本文最有价值的抽象**：相比传统 KG 的实体连接，把每个概念挂上一段细粒度描述、并让三元组真正绑定到具体音视频数据，才让 RAG 检索到的东西“可用”。这解释了为什么图文 KG 在音视频任务上反而掉点——检索到的知识没有音视频接地，等于噪声。
- **质量优先的过滤哲学值得借鉴**：从 116 万筛到 11 万、用 CLAP 相似度硬阈值卡掉弱对齐样本，牺牲规模换高对齐。配合“重述/抽取都是有据改写而非开放生成”，把 LLM 幻觉风险压到最低——这是构建可信数据集的实用范式。
- **Retrieval Checker 是低成本高回报的工程点**：只用同模型的文本编码器再过滤一遍就能稳定涨点，提示多模态 RAG 中“检索质量比检索数量更重要”。
- **全流程用开源蒸馏模型（DeepSeek-R1）替代 GPT-4 API**，在保证质量的同时大幅降低构建成本，对想复现 MMKG 构建的团队很友好。

## 局限与待观察点

- **依赖成对多模态语料**：构建必须建立在同时含视频、音频、文本 caption 的语料上，因此 VAT-KG 的多样性天然受限于底层数据集。
- **规模偏小**：11 万级样本相比其他多模态数据集偏小，作者也承认这一点并计划后续扩展。
- **多数基准并非知识密集设计**：常规 QA 基准上提升“看起来温和”，只有在专门构造的知识密集诊断集上才显出大幅优势——真实收益高度依赖任务是否真的需要外部知识。

## 结论

VAT-KG 通过“概念中心 + 视觉/音频/文本四模态覆盖 + 细粒度描述接地”，填补了现有 MMKG 在多模态 RAG 场景的空白，并给出了从任意多模态语料自动构建的可复现流水线。配套的多模态 RAG 框架（模态无关检索 + Retrieval Checker + MLLM 增强生成）在 AQA/VQA/AVQA 上为开源与商用 MLLM 带来一致、统计显著的提升，尤其在需要联合音视频理解的 AVQA 与知识密集场景中价值突出。
