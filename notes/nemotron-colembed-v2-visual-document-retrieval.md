---
id: nemotron-colembed-v2-visual-document-retrieval
title: "Nemotron ColEmbed V2：Late Interaction 视觉文档检索的 SOTA 模型家族"
category: 多模态 RAG
date: 2026-07-07
order: 4
readTime: 12 min
tags: ["Visual Document Retrieval", "Late Interaction", "ColBERT", "ColPali", "ViDoRe", "NVIDIA", "ECIR 2026"]
summary: "NVIDIA 发布 Nemotron ColEmbed V2 视觉文档检索模型家族（3B/4B/8B），通过双向注意力、Late Interaction、聚类采样、硬负样本挖掘、跨语言翻译和模型融合六大技术，在 ViDoRe V3 排行榜取得第一（NDCG@10 63.42）。同时深入讨论了 Late Interaction 的存储/延迟工程挑战和降维消融。"
---
## 一句话总结
Nemotron ColEmbed V2 证明了 **VLM + Late Interaction + 系统化工程优化** 是视觉文档检索的当前最优范式。8B 模型在 ViDoRe V3 排行榜排名第一（NDCG@10 63.42，领先第二名 3%），但论文最有价值的贡献不只是排名——它详细分析了 Late Interaction 在生产部署中的存储/延迟权衡，并给出了降维消融数据：将嵌入从 4096 维降至 128 维，存储减少 97% 而精度仅损失 4.6%。

<div class="metric-grid">
  <div class="metric"><strong>ViDoRe V3 #1</strong><span>8B 模型 NDCG@10 63.42，领先第二名 +3%</span></div>
  <div class="metric"><strong>3 个变体</strong><span>3B (Eagle 2)、4B (Qwen3-VL-4B)、8B (Qwen3-VL-8B)</span></div>
  <div class="metric"><strong>97% 存储缩减</strong><span>4096→128 维仅损失 4.6% 精度</span></div>
  <div class="metric"><strong>ECIR 2026</strong><span>Late Interaction Workshop</span></div>
</div>

## 论文信息
- **标题**：Nemotron ColEmbed V2: Top-Performing Late Interaction Embedding Models for Visual Document Retrieval
- **作者**：Gabriel de Souza P. Moreira, Ronay Ak, Mengyao Xu, Oliver Holworthy, Benedikt Schifferer 等（全部来自 NVIDIA）
- **会议**：Late Interaction Workshop @ ECIR 2026
- **版本**：arXiv:2602.03992v2, 2026-04
- **链接**：[arXiv](https://arxiv.org/abs/2602.03992) · [PDF](https://arxiv.org/pdf/2602.03992) · [HuggingFace: 3B](https://huggingface.co/nvidia/llama-nemoretriever-colembed-3b-v2) · [4B](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2) · [8B](https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2)

## 背景：为什么视觉文档检索需要 Late Interaction？

传统 RAG 系统依赖 OCR-centric 管线：先从 PDF/幻灯片中提取文本，再用文本嵌入模型检索。这种方式面临三个问题：

1. **布局信息丢失**：双栏排版、表格、图表等复杂文档的文本提取极其困难
2. **管线复杂**：需要文档布局检测 + OCR + caption 生成 + 分块 + 嵌入，每个环节都可能出错
3. **CLIP-style 全局嵌入不够**：整页压缩为单向量时，布局相关的细粒度信息被丢弃

VLM-based 视觉文档检索（如 ColPali）直接把文档页面渲染为图片，利用 VLM 内置的 OCR 和视觉理解能力，配合 Late Interaction（保留 token 级多向量）做检索。这种方式保留了视觉信息并大幅简化了管线。

<figure class="figure">
  <img src="assets/papers/nemotron-colembed-v2/fig1_biencoder.webp" alt="Bi-encoder 与 Late Interaction 对比" loading="lazy" />
  <figcaption>图 1a：Bi-encoder 架构。将查询和文档分别编码为单向量，通过余弦相似度比较。简单高效但压缩了 token 级信息。</figcaption>
</figure>

<figure class="figure">
  <img src="assets/papers/nemotron-colembed-v2/fig1_late_interaction.webp" alt="Late Interaction 架构" loading="lazy" />
  <figcaption>图 1b：Late Interaction 架构。保留所有 token 级嵌入，通过 MaxSim 操作做 token 间的细粒度交互，信息更丰富但存储和计算代价更大。</figcaption>
</figure>

## 方法：六大技术构建 SOTA

### 模型架构

| 模型 | 骨干 | 参数量 | 嵌入维度 | 平均每页 token 数 |
|------|------|:---:|:---:|:---:|
| llama-nemotron-colembed-vl-3b-v2 | Eagle 2 (SigLIP2 + Llama 3.2 3B) | 3.99B | 3072 | 2304 |
| nemotron-colembed-vl-4b-v2 | Qwen3-VL-4B-Instruct | 4.43B | 2560 | 773 |
| nemotron-colembed-vl-8b-v2 | Qwen3-VL-8B-Instruct | 8.14B | 4096 | 773 |

3B 模型使用动态图像切片（训练时 max_input_tiles=2，推理时 =8），每切片生成 256 个视觉 token。Qwen3-VL 系列采用 DeepStack 机制，将视觉编码器中间层 token 注入 LLM 多层。

<figure class="figure">
  <img src="assets/papers/nemotron-colembed-v2/fig2_architecture.webp" alt="3B 模型架构与 Late Interaction 评分" loading="lazy" />
  <figcaption>图 2：llama-nemotron-colembed-vl-3b-v2 架构。动态图像切片 → SigLIP2 编码为视觉 token → LLM（双向注意力）生成 token 级嵌入 → MaxSim Late Interaction 评分。</figcaption>
</figure>

### 1. 因果注意力 → 双向注意力
LLM/VLM 解码器默认使用因果注意力（只看左侧 token）。当用作嵌入模型（编码器）时，替换为**双向注意力**使每个 token 能整合完整上下文信息。论文报告这一改动带来了显著的性能提升（与 NV-Retriever 等先前工作一致）。

### 2. Hard-Negative Mining
用内部 Llama-Eagle 3B VLM 嵌入模型从语料库中挖掘 top-k 最相似的页面图片作为硬负样本。采用 NV-Retriever 的 **percentage-to-positive threshold** 方法（阈值 0.95）：只保留相似度 < 95% × 正样本相似度的负样本，创建 margin 以过滤假负样本。

### 3. 聚类数据采样 (Cluster-based Sampling)
公共训练数据集域分布不均衡。论文对正样本文档页面生成嵌入 → PCA 降至 50 维 → K-Means 聚类（gap statistics 选择最优 k=14）→ 每个聚类均匀采样。这保证了训练数据在域间的平衡性，防止对特定域过拟合。

### 4. 跨语言翻译
ViDoRe V3 的查询覆盖 6 种语言，而文档为英文/法文。用 Qwen3-235B-A22 将每个聚类中采样的查询翻译为多种语言，增强跨语言检索能力。

### 5. 两阶段训练（仅 3B 模型）
- **阶段一**：在纯文本三元组上训练，建立语义相似度基础
- **阶段二**：在图像检索语料上微调，实现跨模态对齐

4B/8B 模型因 Qwen3-VL 的强跨模态预训练基础，仅需单阶段图像对比学习。

### 6. 模型融合 (Model Merging)
对同一架构下用不同数据混合和超参数训练的多个模型做加权平均融合：
- 3B：8 个模型融合，精度提升 +0.8%
- 4B：4 个模型融合，精度提升 +1.0%
- 8B：4 个模型融合，精度提升 +1.5%

融合收益随模型规模增大而增大。

## 实验结果

### ViDoRe V3 排行榜（2026.02.03）

<div class="table-wrap">
<table>
<thead><tr><th>排名</th><th>模型</th><th>平均 NDCG@10</th><th>CompSci</th><th>Energy</th><th>FinanceEn</th><th>FinanceFr</th><th>HR</th></tr></thead>
<tbody>
<tr style="font-weight:bold"><td>1</td><td>nemotron-colembed-vl-8b-v2</td><td>63.42</td><td>79.30</td><td>69.82</td><td>67.29</td><td>51.54</td><td>66.32</td></tr>
<tr><td>2</td><td>tomoro-colqwen3-embed-8b</td><td>61.59</td><td>75.35</td><td>68.41</td><td>65.08</td><td>49.10</td><td>63.98</td></tr>
<tr style="font-weight:bold"><td>3</td><td>nemotron-colembed-vl-4b-v2</td><td>61.54</td><td>78.56</td><td>67.48</td><td>65.02</td><td>49.01</td><td>62.39</td></tr>
<tr><td>4</td><td>Ops-Colqwen3-4B</td><td>61.17</td><td>77.74</td><td>66.49</td><td>65.71</td><td>48.81</td><td>61.81</td></tr>
<tr style="font-weight:bold"><td>6</td><td>llama-nemotron-colembed-vl-3b-v2</td><td>59.79</td><td>77.09</td><td>64.88</td><td>64.23</td><td>44.41</td><td>62.28</td></tr>
<tr><td>7</td><td>jina-embeddings-v4</td><td>57.52</td><td>71.81</td><td>63.50</td><td>59.30</td><td>46.10</td><td>59.53</td></tr>
<tr><td>9</td><td>llama-nemoretriever-colembed-3b-v1</td><td>57.26</td><td>75.16</td><td>62.07</td><td>60.88</td><td>43.77</td><td>58.69</td></tr>
</tbody>
</table>
</div>

**关键发现**：
- 8B 模型在 10 个领域中 9 个排名第一或第二，**CompSci 领域达到 79.30**
- **FinanceFr 是最难的领域**（所有模型都在 44-52 之间），法语金融文档的跨语言检索极具挑战
- V1→V2 提升显著：3B 模型从 57.26（v1）提升到 59.79（v2），+2.53

### ViDoRe V1&V2 排行榜

8B 模型排名第二（NDCG@5 84.80），与第一名 Ops-Colqwen3-4B（84.87）仅差 0.07，但参数量是其两倍。3B/4B 也在 top-4。

### MIRACL-Vision 多语言检索

<div class="table-wrap">
<table>
<thead><tr><th>模型</th><th>规模</th><th>Average NDCG@10</th></tr></thead>
<tbody>
<tr><td>dse-qwen2-2b-mrl-v1</td><td>2B</td><td>0.4426</td></tr>
<tr><td>gme-Qwen2-VL-2B-Instruct</td><td>2B</td><td>0.5283</td></tr>
<tr><td>colqwen2-v1.0</td><td>2B</td><td>0.4728</td></tr>
<tr><td>llama-nemoretriever-colembed-3b-v1</td><td>3B</td><td>0.5841</td></tr>
<tr><td>llama-nemotron-colembed-vl-3b-v2</td><td>3B</td><td>0.6127</td></tr>
<tr><td>nemotron-colembed-vl-4b-v2</td><td>4B</td><td>0.6272</td></tr>
<tr style="font-weight:bold"><td>nemotron-colembed-vl-8b-v2</td><td>8B</td><td>0.6860</td></tr>
</tbody>
</table>
</div>

- 8B 模型在 18 种语言中 **14 种取得最高分**
- **低资源语言提升最大**：阿拉伯语从 0.4129 (colqwen2) 提升到 **0.7863**（+90%），泰语从 0.2389 提升到 **0.6699**（+180%）
- 跨语言翻译增强数据对低资源语言效果尤为显著

## Late Interaction 部署挑战：精度 vs 存储 vs 延迟

### 存储需求对比（100 万页文档）

<div class="table-wrap">
<table>
<thead><tr><th>模型</th><th>范式</th><th>参数量</th><th>嵌入维度</th><th>每页 token 数</th><th>存储 (GB)</th><th>ViDoRe V3</th></tr></thead>
<tbody>
<tr><td>nemotron-colembed-vl-8b-v2</td><td>Late Interaction</td><td>8.14B</td><td>4096</td><td>773</td><td>5,897.5</td><td>63.54</td></tr>
<tr><td>nemotron-colembed-vl-4b-v2</td><td>Late Interaction</td><td>4.43B</td><td>2560</td><td>773</td><td>3,686.0</td><td>61.42</td></tr>
<tr><td>llama-nemotron-colembed-vl-3b-v2</td><td>Late Interaction</td><td>3.99B</td><td>3072</td><td>2304</td><td>13,183.6</td><td>59.70</td></tr>
<tr><td>llama-nemoretriever-colembed-1b-v1</td><td>Late Interaction</td><td>2.15B</td><td>2048</td><td>2304</td><td>8,789.1</td><td>55.48</td></tr>
<tr><td>llama-nemotron-embed-vl-1b-v2</td><td>Bi-encoder</td><td>1.41B</td><td>2048</td><td>1</td><td>3.8</td><td>48.69</td></tr>
<tr><td>Bi-encoder + Reranker</td><td>两阶段</td><td>1.41+1.41B</td><td>2048</td><td>1</td><td>3.8</td><td>54.41</td></tr>
</tbody>
</table>
</div>

**核心洞察**：
- Late Interaction 比 Bi-encoder 需要 **2312× 更多存储**（同为 1B 骨干：8,789 GB vs 3.8 GB）
- 但 **Bi-encoder + Reranker 管线**（54.41）以极小存储代价接近 Late Interaction（55.48），延迟也更可控
- Eagle 2 平均每页生成 2304 个 token（远多于 Qwen3-VL 的 773），导致 3B 模型存储需求反而最大

### 嵌入降维消融

<div class="table-wrap">
<table>
<thead><tr><th>模型</th><th>嵌入维度</th><th>存储 (GB)</th><th>存储占比</th><th>ViDoRe V3</th><th>精度保持率</th></tr></thead>
<tbody>
<tr><td rowspan="3">nemotron-colembed-vl-8b-v2</td><td>4096</td><td>5,897.5</td><td>100%</td><td>62.29</td><td>100%</td></tr>
<tr><td>512</td><td>737.2</td><td>13%</td><td>59.81</td><td>96.0%</td></tr>
<tr><td>128</td><td>184.3</td><td>3%</td><td>59.40</td><td>95.4%</td></tr>
<tr><td rowspan="3">nemotron-colembed-vl-4b-v2</td><td>2560</td><td>3,686.0</td><td>100%</td><td>60.42</td><td>100%</td></tr>
<tr><td>512</td><td>737.2</td><td>20%</td><td>59.29</td><td>98.1%</td></tr>
<tr><td>128</td><td>184.3</td><td>5%</td><td>58.47</td><td>96.8%</td></tr>
</tbody>
</table>
</div>

**关键发现**：
- **512 维是最佳性价比**：存储减少 80-87%，精度仅损失 2-4%
- **128 维仍有 95%+ 精度**：但 184 GB/百万页 对生产环境可能仍偏高
- 4B 模型的降维鲁棒性优于 8B——可能是更低维嵌入空间更容易投影

## 我的理解与启发

1. **Late Interaction 不是万能的**。论文最有价值的部分是诚实地分析了 Late Interaction 的工程代价。对百万级文档库，5897 GB 的嵌入存储是不可接受的。**Bi-encoder + Reranker 是一个更实际的架构选择**，以 3.8 GB 存储达到接近的精度。

2. **VLM token 数差异对存储影响巨大**。Eagle 2 每页 2304 token vs Qwen3-VL 每页 773 token，这导致 3B 模型（Eagle 2 骨干）的存储需求是 8B 模型（Qwen3-VL 骨干）的 2.2 倍——尽管参数量只有其一半。选择 VLM 骨干时，视觉 token 效率是一个常被忽略的关键因素。

3. **聚类采样是处理不平衡数据的实用方案**。对文档检索训练数据做 K-Means 聚类 + 均匀采样，比简单的随机采样或按域加权更能保证泛化性。这个方法对任何不平衡的检索训练都适用。

4. **模型融合收益随规模增大而增大**。8 个 3B 模型融合 +0.8%，4 个 8B 模型融合 +1.5%。大模型的多样性更有价值——可能是因为大模型的参数空间更大，不同训练配置能探索到更多样的解。

5. **降维消融给出了清晰的工程指导**。512 维保留 96% 精度、128 维保留 95% 精度，这为实际部署提供了灵活的选择空间。对大规模语料库，可以索引时用 128 维做粗排，再用全维嵌入做精排。

## 局限与待观察点
- **ViDoRe 为主**：评测集中在 ViDoRe V1/V2/V3，缺少对更广泛文档类型（手写体、扫描件、非标准排版）的验证
- **推理延迟未量化**：论文讨论了存储权衡但未报告具体的检索延迟数据（MaxSim 在百万文档上的实际 QPS）
- **128 维在更大规模上的表现未知**：消融在 ViDoRe V3（约 5K 页面）上评测，百万级语料库上降维的影响可能更显著
- **Reranker 管线的公平对比不足**：Bi-encoder + Reranker 与 Late Interaction 的对比中，训练数据和图像编码器不同，无法完全归因于范式差异
- **跨语言能力依赖翻译增强**：低资源语言的性能提升来自 Qwen3-235B 翻译的训练数据增强，翻译质量本身可能成为瓶颈

## 结论
Nemotron ColEmbed V2 通过系统化的工程优化（双向注意力 + 聚类采样 + 硬负样本挖掘 + 跨语言增强 + 模型融合）将 VLM Late Interaction 视觉文档检索推到了 ViDoRe V3 排行榜第一。但论文更重要的贡献是**诚实地分析了 Late Interaction 在生产部署中的权衡**：它提供了更高的检索精度，但代价是 2000× 以上的存储开销和更高的推理延迟。对于实际部署，论文给出的降维消融（512 维保留 96% 精度）和 Bi-encoder + Reranker 替代方案提供了务实的工程选择。
