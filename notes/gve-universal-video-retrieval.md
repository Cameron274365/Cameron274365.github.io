---
id: "gve-universal-video-retrieval"
title: "GVE：面向通用视频检索的模态金字塔课程学习视频 Embedding"
category: "多模态 RAG"
date: "2026-07-01"
order: 10
readTime: "16 min"
tags: ["Video Retrieval", "Video Embedding", "Curriculum Learning", "Data Synthesis", "Benchmark", "MLLM", "CLIP"]
summary: "GVE 提出 Universal Video Retrieval 的统一框架：构建 UVRB（16 个数据集的诊断性评测），通过 V-SynFlow 合成 1.55M 高质量多任务训练数据，并设计 Modality Pyramid 课程学习算法训练 General Video Embedder。GVE-7B 在 UVRB 上达到 SOTA，比 Unite-7B 高 6.5%，且 3B 模型超越 7B baseline。"
---

## 一句话总结

GVE 的核心思路是：**评测-数据-训练三位一体**——先用 UVRB 诊断现有视频 embedding 模型的能力缺陷（发现 popular benchmarks 无法预测通用能力），再用 V-SynFlow 合成覆盖多任务的高质量训练数据，最后通过 Modality Pyramid 课程学习从简单到复杂逐步训练 General Video Embedder。GVE-7B 在 16 个数据集的 UVRB 上达到 SOTA，**3B 模型甚至超越 7B baseline**，说明课程学习和数据质量比单纯扩大模型更有效。

<div class="metric-grid">
  <div class="metric"><strong>0.573 R@1</strong><span>GVE-7B 在 UVRB 16 个数据集上的平均 Recall@1，SOTA 性能</span></div>
  <div class="metric"><strong>+6.5%</strong><span>相比 Unite-7B 的平均性能提升（尽管 Unite-7B 使用了 in-domain 数据）</span></div>
  <div class="metric"><strong>1.55M</strong><span>V-SynFlow 合成的高质量多任务训练数据对</span></div>
  <div class="metric"><strong>3B > 7B</strong><span>GVE-3B 超越 Unite-7B，证明课程学习比模型规模更有效</span></div>
</div>

## 论文信息
- **标题**：Towards Universal Video Retrieval: Generalizing Video Embedding via Synthesized Multimodal Pyramid Curriculum
- **作者**：Zhuoning Guo, Mingxin Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Xiaowen Chu
- **机构**：Alibaba Group (Tongyi Lab)
- **版本**：arXiv 2510.27571 (Oct 2025)
- **链接**：[arXiv](https://arxiv.org/abs/2510.27571) · [PDF](https://arxiv.org/pdf/2510.27571)

## 背景：为什么现有视频检索模型缺乏通用能力？

视频检索是搜索引擎和推荐系统的核心任务，需要从大规模视频中快速定位与查询相关的内容。当前视频 embedding 模型存在三个结构性问题：

1. **窄 benchmark 导致窄能力**：现有模型大多在 MSRVTT 等粗粒度 benchmark 上训练和评测，这些 benchmark 主要测试"主题匹配"能力（如"一只猫在玩球"），忽视了细粒度空间关系（如"红色杯子在蓝色杯子左边"）、时序动态（如"镜头从远景推到近景"）和长上下文理解。**模型在窄 benchmark 上的高分并不代表通用能力**。

2. **训练数据缺乏多任务覆盖**：现有训练数据主要是 video-text pairs with noisy captions（如 WebVid），缺乏对 composed retrieval（text+image query）、visual retrieval（image/video query）、partially relevant retrieval 等任务的覆盖。模型无法处理多样化的查询格式。

3. **能力间存在 trade-off**：论文发现 CLIP-based 模型在空间理解上强但时序理解弱（\(\rho_{S-CG}=0.99\) vs \(\rho_{T-CG}=0.46\)），而 MLLM-based 模型更均衡。这说明不同架构存在能力 trade-off，需要针对性的训练策略。

<figure class="figure">
  <img src="assets/papers/gve-universal-video-retrieval/fig1_overview.png" alt="GVE 框架总览" loading="lazy" />
  <figcaption>图 1：GVE 框架总览。论文提出 Universal Video Retrieval (UVR) 范式，通过评测-数据-训练三位一体的设计，训练能够处理多任务、多域、多粒度查询的通用视频 embedding 模型。</figcaption>
</figure>

## 方法：GVE 如何实现通用视频检索？

### 1. Universal Video Retrieval Benchmark (UVRB)

论文首先构建 UVRB 作为诊断性评测基准，包含 16 个数据集，覆盖 3 种任务类型和 6 种子领域：

<div class="table-wrap">
  <table>
    <thead><tr><th>任务类型</th><th>子领域</th><th>数据集</th></tr></thead>
    <tbody>
      <tr><td><strong>Textual (TXT)</strong></td><td>Coarse-grained (CG)</td><td>MSRVTT, DiDeMo, CRB-G</td></tr>
      <tr><td></td><td>Fine-grained Spatial (S)</td><td>CRB-S, VDC-O</td></tr>
      <tr><td></td><td>Fine-grained Temporal (T)</td><td>CRB-T, CMRB</td></tr>
      <tr><td></td><td>Partially Relevant (PR)</td><td>DREAM-E, LoVR-TH, PEV-K</td></tr>
      <tr><td></td><td>Long-context (LC)</td><td>LoVR-V, VDC-D</td></tr>
      <tr><td><strong>Composed (CMP)</strong></td><td>-</td><td>MS-TI, MS-TV</td></tr>
      <tr><td><strong>Visual (VIS)</strong></td><td>-</td><td>MSRVTT-V, MS-TV-V</td></tr>
    </tbody>
  </table>
</div>

评测指标主要使用 Recall@1 (R@1)，对于模糊查询任务（如 CMRB、LoVR-TH）使用 Recall@10。

<figure class="figure">
  <img src="assets/papers/gve-universal-video-retrieval/fig2_uvrb_radar.png" alt="UVRB 能力雷达图" loading="lazy" />
  <figcaption>图 2：UVRB 能力雷达图。展示不同模型在 16 个数据集和 9 种能力维度上的表现。GVE-7B 在所有维度上一致领先，尤其在 Composed 和 Temporal 任务上优势明显。</figcaption>
</figure>

### 2. V-SynFlow：多任务数据合成流程

现有训练数据（WebVid、InternVid 等）主要是 video-text pairs with noisy captions，缺乏多任务覆盖。V-SynFlow 分三步合成高质量多任务训练数据：

**Step 1: Multi-granular Quality Control**

对原始数据做三层过滤：
- **Annotation Rectification**：去除非描述性文本（如标题、标签）
- **Cross-Modal Consistency Filtering**：用预训练 embedder（如 GME-7B）过滤 video-text 相似度低于阈值的 pairs
- **Temporal Dynamics Filtering**：去除静态视频（帧间差异小）

**Step 2: Multi-dimensional Information Enrichment**

用 MLLM（如 Keye-VL-8B）为每个视频生成多视角 caption：
- 30% spatial（空间关系）、60% temporal（时序动态）、10% others
- 每个视频生成 5 个不同视角的 caption，增加数据多样性

**Step 3: Multimodal Task Extension**

从 video-text pairs 扩展出多种任务格式：
- **Frame-to-Video**：用 MLLM 描述帧与视频的关系，生成 composed query
- **Clip-to-Video**：用参考 clip 描述目标视频的时序变化
- **Text-to-Text**：从多 caption 中随机匹配生成 text retrieval pairs

最终合成 **1.55M** 高质量训练数据对，覆盖 text/image/composed 多种 query 格式。

<figure class="figure">
  <img src="assets/papers/gve-universal-video-retrieval/fig3_vsynflow.png" alt="V-SynFlow 数据合成流程" loading="lazy" />
  <figcaption>图 3：V-SynFlow 数据合成流程。三步流程：(1) 多层质量过滤；(2) 多视角 caption 生成；(3) 多任务扩展。最终产出 1.55M 高质量多任务训练数据。</figcaption>
</figure>

### 3. General Video Embedder (GVE)

GVE 基于 Qwen2.5-VL 构建，移除 autoregressive head，将多模态输入映射到统一 embedding 空间：

- **输入融合**：文本 token 与视觉 token（image/video frames）拼接，视觉 token 加入 absolute time encoding
- **Embedding 提取**：取最后一个 token 的 hidden state 作为 embedding，做 L2 normalization
- **训练**：使用 LoRA 微调 LLM 部分（q/k/v/up/down/gate projections），冻结 visual backbone

<figure class="figure">
  <img src="assets/papers/gve-universal-video-retrieval/fig4_gve_arch.png" alt="GVE 模型架构" loading="lazy" />
  <figcaption>图 4：GVE 模型架构。基于 Qwen2.5-VL，移除 autoregressive head，将文本和视觉 token 拼接后通过 LLM 处理，取最后 token 的 hidden state 作为 embedding。</figcaption>
</figure>

### 4. Modality Pyramid：课程学习算法

论文发现直接混合训练所有任务效果不佳——简单任务主导早期优化，复杂任务得不到充分训练。Modality Pyramid 设计了一个从简单到复杂的课程：

**能力层级设计**：

\[
\text{Visual (VIS)} \rightarrow \text{Textual (TXT)} \rightarrow \text{Composed (CMP)}
\]

直觉是：视觉匹配是最基础的能力（类似 image retrieval），文本匹配需要语言理解，composed retrieval 需要同时理解文本和视觉。

**Alignment-aware Dynamic Scheduling**：

每个 epoch 开始时，用 prober model \(\Psi_t\)（epoch 1 用 GME-7B，后续用前一 epoch 的 GVE checkpoint）评估每个任务的 alignment level：

\[
R_k^{(t)} = \frac{1}{N_p} \sum_i \cos(\Psi_t(x_i), \Psi_t(y_i))
\]

然后根据 alignment level 动态调整采样概率：

\[
P^{(t)}(k) \propto \exp(R_k^{(t)} / \sigma(t))
\]

其中温度 \(\sigma(t)\) 从 \(\sigma_{\min}=0.1\) 线性增加到 \(\sigma_{\max}=1.0\)。

**核心思想**：
- **早期**（低温）：优先采样 alignment level 高的任务（简单任务），建立基础能力
- **后期**（高温）：均匀采样所有任务，强化复杂能力
- **渐进式**：从 VIS → TXT → CMP 逐步建立能力栈

<figure class="figure">
  <img src="assets/papers/gve-universal-video-retrieval/fig5_modality_pyramid.png" alt="Modality Pyramid 课程学习" loading="lazy" />
  <figcaption>图 5：Modality Pyramid 课程学习。从简单到复杂的能力层级：Visual → Textual → Composed。简单任务为复杂任务奠定基础。</figcaption>
</figure>

**统一对比损失**：

使用 InfoNCE loss with hard negatives：

\[
\mathcal{L}_i^{(q \to v)} = -\log \frac{\exp(s_{q \to v^+} / \tau_l)}{\exp(s_{q \to v^+} / \tau_l) + \sum_{j=1}^{K} \exp(s_{q \to v_j^-} / \tau_l)}
\]

其中 hard negatives 由 prober model 从外部语料中预挖掘，增强判别能力。

## 数据与评测

**训练数据**：
- **公开数据**：VAST、InternVid-FLT、PE-Video、WebVid（过滤后）
- **合成数据**：V-SynFlow 生成的 1.55M 多任务数据对
- **总量**：3B 模型使用 12.55M instances，7B 模型使用 13.73M instances

**评测基准**：UVRB 包含 16 个数据集，覆盖：
- **任务类型**：Textual (TXT)、Composed (CMP)、Visual (VIS)
- **子领域**：Coarse-grained (CG)、Fine-grained Spatial (S)、Fine-grained Temporal (T)、Partially Relevant (PR)、Long-context (LC)

**Baseline 模型**：14 个模型，包括：
- **CLIP-based**：CLIP4Clip、ViCLIP、VideoCLIP-XL、LanguageBind、InternVideo2 (1B/6B)
- **MLLM-based**：GME (2B/7B)、Unite (2B/7B)、VLM2Vec-V2、BGE-VL、UniME-7B、B3-7B

**评测设置**：
- 统一使用 cosine similarity，无 re-ranking
- 每个视频均匀采样 8 帧，CLIP-based 使用 224×224，MLLM-based 限制每帧 ≤200 tokens
- 输入序列截断到 8192 tokens

## 实验结果

### 主结果：GVE 在 UVRB 上达到 SOTA

<div class="table-wrap">
  <table>
    <thead><tr><th>Model</th><th>Avg D</th><th>Avg A</th><th>TXT</th><th>CMP</th><th>VIS</th><th>CG</th><th>FG</th><th>LC</th><th>S</th><th>T</th><th>PR</th></tr></thead>
    <tbody>
      <tr><td>CLIP4Clip</td><td>0.416</td><td>0.451</td><td>0.487</td><td>0.223</td><td>0.714</td><td>0.501</td><td>0.437</td><td>0.689</td><td>0.769</td><td>0.321</td><td>0.221</td></tr>
      <tr><td>InternVideo2-6B</td><td>0.526</td><td>0.559</td><td>0.609</td><td>0.254</td><td>0.702</td><td>0.567</td><td>0.538</td><td>0.746</td><td>0.779</td><td>0.412</td><td>0.351</td></tr>
      <tr><td>GME-7B</td><td>0.548</td><td>0.581</td><td>0.629</td><td>0.279</td><td>0.667</td><td>0.589</td><td>0.559</td><td>0.773</td><td>0.801</td><td>0.438</td><td>0.382</td></tr>
      <tr><td>Unite-7B</td><td>0.508</td><td>0.527</td><td>0.609</td><td>0.254</td><td>0.702</td><td>0.567</td><td>0.538</td><td>0.746</td><td>0.779</td><td>0.412</td><td>0.351</td></tr>
      <tr style="font-weight:bold"><td>GVE-3B</td><td>0.544</td><td>0.571</td><td>0.619</td><td>0.304</td><td>0.647</td><td>0.552</td><td>0.541</td><td>0.764</td><td>0.816</td><td>0.430</td><td>0.377</td></tr>
      <tr style="font-weight:bold"><td>GVE-7B</td><td>0.573</td><td>0.600</td><td>0.657</td><td>0.312</td><td>0.657</td><td>0.587</td><td>0.570</td><td>0.814</td><td>0.821</td><td>0.469</td><td>0.419</td></tr>
    </tbody>
  </table>
</div>

**关键发现**：

1. **GVE-7B 达到 SOTA**：平均 R@1 达到 0.573（dataset avg）和 0.600（ability avg），比 Unite-7B 高 **+6.5%** 和 **+7.3%**。
2. **3B 超越 7B baseline**：GVE-3B（0.544/0.571）超越 Unite-7B（0.508/0.527），说明**课程学习和数据质量比模型规模更有效**。
3. **Composed 任务提升最显著**：CMP 任务从 Unite-7B 的 0.254 提升到 GVE-7B 的 0.312（+22.8%），说明 V-SynFlow 合成的 composed 数据非常有效。
4. **CLIP vs MLLM 性能分化**：CLIP4Clip 在 VIS 任务上最强（0.714），但在 TXT 和 CMP 上弱；MLLM-based 模型更均衡。

<figure class="figure">
  <img src="assets/papers/gve-universal-video-retrieval/fig6_data_scaling.png" alt="Data Scaling 效果" loading="lazy" />
  <figcaption>图 6：Data Scaling 效果。性能随数据量对数增长，GVE-3B 的 scaling efficiency 高于 GVE-7B（每 10x 数据提升 +7.4% vs +5.4%），但 GVE-7B 的绝对性能更高。</figcaption>
</figure>

### 四个关键发现

论文通过能力相关性分析发现四个重要规律：

**Finding 1: Partially Relevant Retrieval 是通用能力的最佳预测器**

MSRVTT（popular benchmark）与平均性能的相关性仅 \(\rho_{avg}=0.58\)，而 Partially Relevant (PR) 检索与平均性能的相关性高达 \(\rho_{avg}=0.97\)。**PR 是通用能力的最佳预测器**，建议未来评测纳入更多 PR benchmark。

<figure class="figure">
  <img src="assets/papers/gve-universal-video-retrieval/fig8_performance_correlation.png" alt="性能与能力维度相关性" loading="lazy" />
  <figcaption>图 8：平均性能与各能力维度/数据集的相关性。PR（Partially Relevant）与平均性能的相关性最高（0.97），而 MSRVTT 仅 0.58，说明 popular benchmark 无法预测通用能力。</figcaption>
</figure>

**Finding 2: 空间和时间理解是解耦的**

空间理解 (S) 和时间理解 (T) 的相关性仅 \(\rho=0.12\)，说明这两种能力是独立的。时间理解对细粒度任务更重要（\(\rho_{T-FG}=0.98\) vs \(\rho_{S-FG}=0.39\)）。

**Finding 3: CLIP-based 和 MLLM-based 模型存在能力分化**

- CLIP-based：空间理解强（\(\rho_{S-CG}=0.99\)），时间理解弱（\(\rho_{T-CG}=0.46\)），composed retrieval 与 visual retrieval 负相关（\(\rho_{CMP-VIS}=-0.71\)）
- MLLM-based：能力更均衡，PR-CG 相关性更高（0.98 vs 0.70），LC-T 正相关（0.64 vs -0.14）

**Finding 4: Scaling 对视觉感知提升有限**

87M 的 CLIP4Clip 在 VIS 任务上（0.714）超越 8B 的 Unite-7B（0.702），说明**模型规模对视觉感知提升有限**。视觉能力与整体检索性能的相关性仅 \(\rho_{AVG-VIS}=0.26\)。

<figure class="figure">
  <img src="assets/papers/gve-universal-video-retrieval/fig7_correlation.png" alt="能力相关性分析" loading="lazy" />
  <figcaption>图 7：能力相关性分析。CLIP-based 模型（上）在空间理解上强但时间理解弱；MLLM-based 模型（下）能力更均衡。PR 与平均性能相关性最高（0.97），是通用能力的最佳预测器。</figcaption>
</figure>

## Ablation：哪些设计最关键？

### V-SynFlow 和 Modality Pyramid 的贡献

<div class="table-wrap">
  <table>
    <thead><tr><th>Model</th><th>V-SynFlow</th><th>Pyramid</th><th>Avg D</th><th>Avg A</th><th>CMP</th></tr></thead>
    <tbody>
      <tr><td>GVE-i-3B</td><td>✗</td><td>✗</td><td>0.528</td><td>0.558</td><td>0.237</td></tr>
      <tr><td>GVE-s-3B</td><td>✓</td><td>✗</td><td>0.537</td><td>0.564</td><td>0.301</td></tr>
      <tr><td>GVE-3B</td><td>✓</td><td>✓</td><td>0.544</td><td>0.571</td><td>0.304</td></tr>
      <tr><td>GVE-i-7B</td><td>✗</td><td>✗</td><td>0.563</td><td>0.587</td><td>0.274</td></tr>
      <tr><td>GVE-s-7B</td><td>✓</td><td>✗</td><td>0.568</td><td>0.594</td><td>0.313</td></tr>
      <tr><td>GVE-7B</td><td>✓</td><td>✓</td><td>0.573</td><td>0.600</td><td>0.312</td></tr>
    </tbody>
  </table>
</div>

**消融结论**：

1. **V-SynFlow 对 Composed 任务贡献最大**：3B 模型的 CMP 从 0.237 提升到 0.301（+27%），说明合成的 composed 数据非常有效。
2. **Modality Pyramid 进一步提升**：在 V-SynFlow 基础上，Pyramid 再提升约 +0.7-0.8%，尤其在 TXT 任务上（0.648 → 0.657）。
3. **两者互补**：V-SynFlow 提供数据多样性，Pyramid 优化训练顺序，两者缺一不可。

## 我的理解与启发

1. **"评测-数据-训练"三位一体的设计哲学**：GVE 不是单纯提出一个新模型，而是构建了一个完整的生态系统——先定义什么是"通用能力"（UVRB），再合成覆盖这些能力的数据（V-SynFlow），最后设计匹配的训练策略（Modality Pyramid）。这种系统性思维值得学习。

2. **Partially Relevant Retrieval 是一个被忽视的重要场景**：论文发现 PR 与通用能力的相关性高达 0.97，远超 MSRVTT（0.58）。这解释了为什么很多模型在 MSRVTT 上高分但在实际应用中表现不佳——PR 才是真正考验模型理解能力的场景。

3. **课程学习的"金字塔"设计很巧妙**：从 VIS → TXT → CMP 的能力层级符合直觉——视觉匹配是最基础的，文本理解需要语言能力，composed retrieval 需要同时理解多模态。Alignment-aware scheduling 让模型在早期优先学习简单任务，后期均匀学习所有任务。

4. **3B 超越 7B 的启示**：在资源有限的场景下，**数据质量和训练策略比模型规模更重要**。GVE-3B 超越 Unite-7B 说明好的课程学习能让小模型达到甚至超越大模型的效果。

5. **CLIP vs MLLM 的能力分化值得关注**：CLIP 在视觉感知上强但语言理解弱，MLLM 更均衡但视觉感知略弱。这暗示未来可能需要结合两者优势——用 CLIP 做视觉 backbone + LLM 做语言理解。

## 局限与待观察点

- **UVRB 覆盖仍有盲区**：虽然 16 个数据集覆盖了多种任务和领域，但缺少医学、工业、监控等专业领域的数据。
- **V-SynFlow 依赖 MLLM 质量**：数据合成依赖 Keye-VL-8B 等 MLLM，caption 质量直接影响训练数据质量。
- **Modality Pyramid 的层级设计较简单**：仅 VIS → TXT → CMP 三层，可能需要更细粒度的课程（如 spatial → temporal → composed）。
- **缺少下游应用验证**：论文主要评测 retrieval 性能，但在下游应用（如 video QA、video captioning）上的效果有待验证。
- **Scaling law 仍在早期**：论文发现性能随数据量对数增长，但 scaling 效率在下降，可能需要探索更高效的训练策略。

## 结论

GVE 提出了面向通用视频检索的完整框架：**UVRB 定义通用能力、V-SynFlow 合成多任务数据、Modality Pyramid 优化训练顺序**。在 16 个数据集的 UVRB 上，GVE-7B 达到 SOTA（0.573 R@1），比 Unite-7B 高 6.5%，且 3B 模型超越 7B baseline。论文的四个关键发现（PR 是通用能力的最佳预测器、空间/时间理解解耦、CLIP vs MLLM 能力分化、scaling 对视觉感知提升有限）为未来视频检索研究提供了重要指导。
