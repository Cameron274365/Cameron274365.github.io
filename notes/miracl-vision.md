---
id: "miracl-vision"
title: "MIRACL-VISION：多语言视觉文档检索 Benchmark"
category: "多模态 RAG"
date: "2025-06-17"
order: 100
readTime: "12 分钟"
tags: ["多模态检索", "VLM", "Benchmark", "多语言", "RAG", "NVIDIA", "MIRACL", "文档检索"]
summary: "NVIDIA 发布的多语言视觉文档检索评测基准，覆盖 18 种语言，揭示当前 VLM 检索模型在多语言场景下比文本检索模型低 59.7% 的巨大差距，为多模态 RAG 的真实能力评估提供了关键标尺。"
---

## 一句话总结

MIRACL-VISION 是 NVIDIA 团队基于 MIRACL 文本检索数据集扩展的多语言视觉文档检索 Benchmark，覆盖 18 种语言（含阿拉伯语、日语、斯瓦希里语等低资源语言），通过巧妙的 hard-negative 筛选将语料库压缩 58 倍。评测揭示了一个关键发现：**当前 SOTA 视觉检索模型在多语言场景下，NDCG@10 比纯文本模型低 59.7%，甚至英语也低 12.1%**——视觉 RAG 的多语言能力远未成熟。

## 论文信息

- **标题**：MIRACL-VISION: A Large, multilingual, visual document retrieval benchmark
- **作者**：Radek Osmulski, Gabriel de Souza P. Moreira, Ronay Ak, Mengyao Xu, Benedikt Schifferer, Even Oldridge（全部来自 NVIDIA）
- **版本**：arXiv: 2505.11651，2025 年 5 月
- **链接**：[arXiv](https://arxiv.org/abs/2505.11651) | [HuggingFace](https://huggingface.co/datasets/nvidia/miracl-vision)
- **贡献**：NVIDIA 团队，全部作者贡献均等

## 背景：为什么这个问题重要？

### RAG 中检索的痛点

当前 RAG 系统的主流做法是**文本检索**：将文档解析、提取文本、分块、嵌入，然后进行 dense retrieval。但现实中的文档布局复杂——双栏排版、图文混排、表格和图表交织——这让文本提取变得极其困难。你需要：

- 文档布局检测模型来分割页面元素
- LLM 为图表生成自然语言 caption
- 结构感知的分块策略

这些构成了一个**非平凡的 ingestion pipeline**。

### 视觉检索的崛起与不足

近年来出现了**基于图像的文档检索**方案：直接把页面渲染为图片，用 VLM（如 ColPali、DSE-Qwen2、GME-Qwen2）来检索。这种方式利用 VLM 内置的 OCR 能力，省去了复杂的文本提取 pipeline。

但现有评测基准（ViDoRe、VDR-Multilingual）存在三个问题：
1. **语言覆盖有限**：ViDoRe 主要是英语+2 个法语数据集，VDR 仅 5 种欧洲语言
2. **问题多为合成生成**：用 VLM/LLM 生成问题，容易在问题中重复文档关键词，导致检索任务过于简单
3. **语料库太小**：每数据集仅 672-3000 篇文档，不足以构成真正的检索挑战

MIRACL-VISION 正是为填补这些空白而生。

## 方法：核心设计

### 整体思路

MIRACL-VISION 基于 MIRACL（多语言文本检索 Benchmark）扩展而来。MIRACL 拥有由母语者手工标注的 77k 高质量问题，覆盖 18 种语言。MIRACL-VISION 复用这些问题，将 ground truth 文本段落替换为**包含答案的文档页面截图**。

### 四步构建流程

**Step 1 — 仅保留每篇文章的第一段**

MIRACL 的语料来自 Wikipedia 文章，被切分成多个 chunk。例如英文 Wikipedia 有 570 万篇文章，MIRACL 英文有 3200 万个 chunk。为了将问题映射到页面图片，需要定位答案所在的 chunk 属于哪篇文章的哪一页。

作者采用了简化方案：**只保留每篇文章的第一个 chunk**。这样就能确保答案一定在文章第一页，不需要复杂的 chunk → 页面位置映射。

**Step 2 — 筛选可回答问题**

去除第一段后，部分问题不再有对应的 ground truth 文档。这一步过滤掉所有无法回答的问题。中间数据集称为 **MIRACL-1stParagraph**。

**Step 3 — 压缩语料库大小，保留 hard negatives**

这是最巧妙的设计。即使只保留第一段，某些语言的语料库仍然很大（英文 570 万篇）。对所有文档截图并评估是不现实的。

作者用 `multilingual-e5-large` 文本嵌入模型对所有问题和文档进行嵌入，计算余弦相似度，对每个问题只保留 top-k 最相似的文档：
- 英文：top-100
- 其他语言：top-50

这样只保留了对每个问题来说是**正样本或 hard negative** 的文档，其余"容易区分"的文档被丢弃。最终语料库平均每语言仅 **18,819 篇**，压缩了 58 倍，但检索难度并未降低。

**Step 4 — 生成图片和文本**

对每个文档，下载对应的 Wikipedia 文章，用 Playwright 渲染 HTML（去除侧边栏等无关元素），截取前 2048 像素，裁剪为 980×980 像素。同时提取 HTML 文本的前 12 句作为近似文本表示，构成 **MIRACL-VISION-text** 版本。

## 数据与评测

### 数据集规模

| 维度 | MIRACL（原始） | MIRACL-1stParagraph | MIRACL-VISION |
|------|:---:|:---:|:---:|
| 语言数 | 18 | 18 | 18 |
| 平均查询数/语言 | 750 | 439 | 439 |
| 平均文档数/语言 | 5.9M | 1.09M | **18,819** |

### 与现有 Benchmark 的对比

| 特性 | ViDoRe | VDR-Multilingual | **MIRACL-VISION** |
|------|:---:|:---:|:---:|
| 语言数 | 2 | 5 | **18** |
| 平均文档/数据集 | 672 | 3,000 | **18,500** |
| 文档选择 | 随机 | 随机 | **hard-negative 采样** |
| 模态 | 文本/图表/表格 | 文本/视觉 | 文本 |
| 问题生成 | 人工+合成 | 合成+人工审核 | **纯人工生成** |

MIRACL-VISION 的语料库是其他 Benchmark 的 **6 倍**大，且问题由母语者手工标注，避免了合成问题中关键词重复的问题。

## 实验结果

### 文本模型：MIRACL 各变体的 NDCG@10

| 模型 | MIRACL 原始 | 1stParagraph | 1stParagraph-Reduced | VISION-text |
|------|:---:|:---:|:---:|:---:|
| bge-m3 | 0.6776 | 0.8442 | 0.8468 | 0.7964 |
| multilingual-e5-large | 0.6512 | 0.8322 | 0.8323 | 0.7624 |
| Average | 0.6499 | 0.8271 | 0.8306 | 0.7798 |

关键洞察：
- 1stParagraph 版本比原始 MIRACL 分数更高，说明仅保留第一段后任务变简单了——可能是因为同一篇文章的其他 chunk 是更强的 distractor
- 1stParagraph 和 Reduced 版本分数几乎一致（0.8271 vs 0.8306），证明 **hard-negative 筛选方法有效，在压缩 58 倍的同时保持了检索难度**

### 视觉模型：跨 Benchmark 对比

| 模型 | VDR-Multilingual | ViDoRe | **MIRACL-VISION** |
|------|:---:|:---:|:---:|
| colqwen2-v1.0 | 0.9604 | 0.8969 | **0.4728** |
| gme-Qwen2-VL-2B | 0.9165 | 0.8878 | **0.5283** |
| dse-qwen2-2b-mrl-v1 | 0.8363 | 0.8416 | **0.4426** |

VDR-Multilingual 和 ViDoRe 几乎被饱和（NDCG@10 接近 0.9），而 MIRACL-VISION 上最好的模型也仅 0.5283。

### 核心发现：视觉 vs 文本的差距

| 语言 | 最佳文本模型 (bge-m3) | 最佳视觉模型 (gme-Qwen2) | **差距** |
|------|:---:|:---:|:---:|
| English | 0.7348 | 0.6784 | **-12.1%** |
| Chinese | 0.7458 | 0.6314 | **-15.3%** |
| Arabic | 0.8883 | 0.4888 | **-45.0%** |
| Hindi | 0.7581 | 0.3127 | **-58.7%** |
| Telugu | 0.9090 | 0.0893 | **-90.2%** |
| **Average** | **0.7964** | **0.5283** | **-33.7%** |

去掉 Telugu（视觉模型几乎完全失效）这个 outlier 后，文本模型仍比视觉模型高 **43%**。

更值得注意的是，**gte-multilingual-base 仅 305M 参数**，却比 1543M 参数的 Qwen2-VL 系列视觉模型表现更好。

## Ablation：哪些设计最关键？

### Hard-negative 筛选的有效性

对比 MIRACL-1stParagraph（1.09M 文档/语言）和 MIRACL-1stParagraph-Reduced（18,819 文档/语言），NDCG@10 几乎一致，证明压缩 58 倍不影响检索难度。

### 合成问题 vs 人工问题

VDR-Multilingual 和 ViDoRe 使用合成问题，导致了两个问题：
1. 问题中倾向于重复文档关键词，使检索变得 trivial
2. 部分 VQA 数据集的问题不适合检索（如 "What is the table number?" —— 看到图片才能回答，不适合作为检索 query）

MIRACL-VISION 使用 MIRACL 的母语者手工标注问题，更接近真实用户场景。

## 我的理解与启发

### 1. 视觉 RAG 的多语言能力是严重盲区

这篇论文最核心的贡献是**揭露了 VLM 检索模型在多语言场景下的巨大差距**。很多论文声称 VLM 具有零样本多语言能力，但在 MIRACL-VISION 上，最好的视觉模型也不如一个 305M 的纯文本模型。这提示我们：**在生产环境中部署视觉 RAG 时，多语言场景需要额外的文本 fallback 或混合检索策略**。

### 2. Hard-negative 采样的工程价值

通过文本嵌入模型预筛选 top-k 相似文档来压缩语料库的做法非常实用。在评估大规模检索系统时，可以借鉴这个思路来降低计算成本，同时不损害评估质量。

### 3. Benchmark 设计中的"合成问题"陷阱

使用 LLM 生成问题看似高效，但会引入系统性偏差：问题和文档共享关键词分布，导致检索过于简单。这提醒我们：**Benchmark 的难点不在于规模，而在于是否真实反映了用户行为**。

### 4. ViDoRe 和 VDR 已接近饱和

NDCG@10 接近 0.9-0.96，说明这些 Benchmark 的区分度在下降，社区需要 MIRACL-VISION 这样的新挑战来推动模型进步。

## 局限与待观察点

- **仅限文本模态**：MIRACL-VISION 的问题主要基于文本内容，不涵盖图表、表格、infographic 的检索。ViDoRe 和 VDR 在这方面更全面
- **仅保留第一段**：这是一个工程简化，可能导致部分问题类型（需要文章后半部分信息）被过滤掉
- **文本对比的公平性**：MIRACL-VISION-text 从 HTML 提取文本，质量高于 PDF 提取，因此视觉 vs 文本的差距可能被略微高估
- **缺少训练集**：目前只提供评估集，未来计划提供训练集以支持微调
- **Wikipedia 场景的局限性**：Wikipedia 文章结构相对规范，企业文档的布局更复杂，泛化性待验证

## 结论

MIRACL-VISION 是一个覆盖 18 种语言、设计精巧的多语言视觉文档检索 Benchmark。它通过复用 MIRACL 的高质量人工问题、巧妙的 hard-negative 筛选、以及 Wikipedia 页面截图生成，构建了一个**既有挑战性又计算高效**的评估基准。实验揭示了一个尖锐的事实：**当前 VLM 在多语言视觉检索上远不如纯文本检索**，差距高达 59.7%。这个 Benchmark 将帮助社区更准确地评估和推动多语言视觉 RAG 的进展。