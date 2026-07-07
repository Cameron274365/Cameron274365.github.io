---
id: e5-omni-explicit-cross-modal-alignment
title: "e5-omni：用显式对齐把 VLM 变成稳健的全模态嵌入模型"
category: 多模态 RAG
date: 2026-07-07
order: 2
readTime: 14 min
tags: ["Omni-modal Embedding", "Contrastive Learning", "Cross-modal Alignment", "Retrieval", "MMEB-V2", "AudioCaps"]
summary: "e5-omni 提出一套轻量级显式对齐方案，通过模态感知温度校准、可控负样本课程+DCL去偏、批量白化+协方差对齐三个插件式组件，将现成 VLM 转化为稳健的全模态嵌入模型，在 MMEB-V2 和 AudioCaps 上持续超越强基线。"
---
## 一句话总结
e5-omni 的核心贡献是指出当前全模态嵌入模型过度依赖 VLM 预训练带来的**隐式对齐**，导致相似度尺度不一致、负样本硬度失衡和排序不稳定三大问题；并提出一套**轻量级显式对齐 recipe**（模态感知温度 + 可控负样本课程 + 批量白化协方差对齐），在不修改骨干架构的前提下将现成 VLM 转化为稳健的全模态嵌入模型。

<div class="metric-grid">
  <div class="metric"><strong>66.4</strong><span>MMEB-V2 总分（7B），超越所有基线</span></div>
  <div class="metric"><strong>37.7</strong><span>AudioCaps Recall@1，显著优于 Omni-Embed-Nemotron</span></div>
  <div class="metric"><strong>3 个</strong><span>插件式对齐组件，无需修改骨干架构</span></div>
  <div class="metric"><strong>4+</strong><span>VLM 骨干验证可迁移性</span></div>
</div>

## 论文信息
- **标题**：e5-omni: Explicit Cross-modal Alignment for Omni-modal Embeddings
- **作者**：Haonan Chen, Sicheng Gao, Radu Timofte, Tetsuya Sakai, Zhicheng Dou
- **机构**：中国人民大学、维尔茨堡大学、早稻田大学
- **版本**：arXiv:2601.03666v2, 2026-01-09
- **链接**：[arXiv](https://arxiv.org/abs/2601.03666) · [PDF](https://arxiv.org/pdf/2601.03666) · [HuggingFace](https://huggingface.co/Haon-Chen/e5-omni-7B)

<figure class="figure">
  <img src="assets/papers/e5-omni/fig1_overview.webp" alt="e5-omni 隐式对齐 vs 显式对齐对比" loading="lazy" />
  <figcaption>图 1：(a) 隐式对齐导致模态依赖的锐度差异、负样本硬度失衡和排序不稳定；(b) e5-omni 通过三个轻量级模块进行显式对齐，校准跨模态相似度。</figcaption>
</figure>

## 背景：为什么隐式对齐不够？
现代信息检索越来越多地涉及多模态查询和结果（文本、图像、音频、视频）。近期工作开始从双模态嵌入扩展到全模态嵌入，但这些方法仍主要依赖 VLM 预训练带来的**隐式对齐**。论文指出这在全模态场景下会导致三个核心问题：

- **模态依赖的锐度 (Modality-dependent sharpness)**：单一全局温度参数会让某些模态组合的 logits 过于尖锐、另一些过于平坦，导致对比梯度不平衡
- **负样本硬度失衡 (Negative hardness imbalance)**：混合模态 batch 中负样本硬度分布不均，大量简单负样本在训练后期几乎不提供梯度信号，削弱了细粒度区分能力
- **排序不稳定 (Unstable ranking)**：不同模态的嵌入在一阶和二阶统计量上不匹配，使得相似度排序对微小分数变化敏感

这些问题在全模态设置下被放大，因为 mini-batch 天然混合了多种模态，加剧了尺度和几何的不匹配。

## 方法：三个轻量级显式对齐组件
### 1. 模态感知温度校准 (Modality-aware Temperature Calibration)
引入一个可训练的**逐模态缩放向量** \(\tau \in \mathbb{R}^{|\mathcal{M}_0|}\)。对于任意输入 \(x\)，根据其模态组成构造归一化指示权重 \(w(x)\)，计算实例温度 \(\tau(x) = w(x)^\top \tau\)。对于查询-目标对 \((q, p)\)，使用对称配对温度 \(\tau(q,p) = (\tau(q) + \tau(p))/2\)，校准后的相似度 logit 为：

\[\ell(q, p) = \frac{\text{sim}(e(q), e(p))}{\tau(q, p)}\]

直觉上，噪声更大/更模糊的模态会学到更大的温度值（使 logits 更平坦），从而稳定混合模态 batch 中的对比梯度。

**训练后学到的温度值**：T=0.0130, I=0.0127, A=0.0219, V=0.0223——音频和视频的温度明显更大，说明这些模态的相似度分布更模糊。

### 2. 可控负样本课程 + DCL 去偏
采用**分位数阈值**选择 in-batch 负样本，并在 warmup 后逐步提高难度。具体地，在训练步 \(t\) 使用阈值 \(\rho_t\)，只保留每行最难的 \((1-\rho_t)\) 比例负样本。

为防止硬负样本选择引入假负样本偏差，结合 **Debiased Contrastive Learning (DCL)**：

\[\mathcal{L}_{\text{DCL}} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(S_{ii})}{\exp(S_{ii}) + \tilde{N}_i}\]

其中 \(\tilde{N}_i = \max\left(\sum_{j \in \Omega_i \setminus \{i\}} \exp(S_{ij}) - \gamma_+ \exp(S_{ii}), \epsilon\right)\)，\(\gamma_+\) 为去偏系数。

课程调度从低 \(\rho\)（保留更多简单负样本）线性增长到高 \(\rho\)（聚焦困难负样本），让模型先学粗对齐再解决细粒度混淆。默认参数：\(\rho_{\text{init}}=0.1\)，\(\rho_{\text{final}}=0.5\)，warmup \(t_0=4000\) 步，\(\gamma_+=0.1\)。

### 3. 批量白化 + 协方差对齐
对 mini-batch 中的查询和目标嵌入应用**批量白化变换** \(W(\cdot)\)，然后添加 CORAL 风格的协方差正则化：

\[\mathcal{L}_{\text{coral}} = \frac{1}{4D^2} \|\text{Cov}(\hat{Q}) - \text{Cov}(\hat{P})\|_F^2\]

最终目标函数为 \(\mathcal{L} = \mathcal{L}_{\text{DCL}} + \lambda_{\text{coral}} \mathcal{L}_{\text{coral}}\)。这使共享嵌入空间中的二阶几何更加一致，稳定了跨模态相似度排序。\(\lambda_{\text{coral}}=0.05\)。

<figure class="figure">
  <img src="assets/papers/e5-omni/fig2_temperature.webp" alt="e5-omni 框架总览" loading="lazy" />
  <figcaption>图 2：e5-omni 框架。在 VLM 骨干上叠加三个轻量级组件：模态感知温度校准、可控负样本课程+DCL、批量白化+协方差对齐。</figcaption>
</figure>

## 数据与评测
### 评测基准
- **MMEB-V2**：大规模多模态嵌入基准，包含 9 个元任务、78 个子任务，覆盖 Image（36 任务）、Video（18 任务）、VisDoc（24 任务）。Image/Video 用 Hit@1，VisDoc 用 NDCG@5
- **AudioCaps**：约 4.4K 文本-音频对的检索基准，报告 Recall@1

### 训练设置
- 骨干网络：Qwen2.5-Omni-7B / 3B，LoRA 微调
- 混合训练数据：BGE-m3 文本对 + MMEB-V1/V2 图文对 + MSR-VTT 文视频对 + AudioCaps 文音频对
- 最大序列长度 512 tokens，1 epoch，lr=1e-4，8×H100，per-device batch=20，gradient accumulation=2

## 实验结果
### MMEB-V2：全面超越基线

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>规模</th><th>Image</th><th>Video</th><th>VisDoc</th><th>All</th></tr></thead>
    <tbody>
      <tr><td>ColPali-v1.3</td><td>3B</td><td>34.9</td><td>28.2</td><td>71.0</td><td>44.4</td></tr>
      <tr><td>VLM2Vec-v1</td><td>7B</td><td>65.5</td><td>33.7</td><td>46.4</td><td>52.3</td></tr>
      <tr><td>GME</td><td>7B</td><td>56.0</td><td>38.4</td><td>75.2</td><td>57.8</td></tr>
      <tr><td>CAFe</td><td>7B</td><td>67.6</td><td>42.4</td><td>63.9</td><td>60.6</td></tr>
      <tr><td>UME-R1</td><td>7B</td><td>71.3</td><td>47.5</td><td>67.1</td><td>64.5</td></tr>
      <tr><td>Tevatron-Omni</td><td>7B</td><td>37.1</td><td>35.1</td><td>74.5</td><td>48.1</td></tr>
      <tr><td>LCO-EMB</td><td>7B</td><td>44.0</td><td>38.2</td><td>69.8</td><td>50.6</td></tr>
      <tr><td>Omni-Embed-Nemotron</td><td>3B</td><td>43.7</td><td>36.9</td><td>74.2</td><td>51.5</td></tr>
      <tr><td>e5-omni w/o alignment</td><td>7B</td><td>69.0</td><td>41.8</td><td>74.7</td><td>64.4</td></tr>
      <tr style="font-weight:bold"><td>e5-omni-7B</td><td>7B</td><td>71.2</td><td>43.5</td><td>76.1</td><td>66.4</td></tr>
    </tbody>
  </table>
</div>

**关键观察**：
- e5-omni-7B 在所有分组上一致领先，总分 **66.4**，比无对齐版本提升 **+2.0**
- 对齐收益随模型规模增大：3B 提升 +1.1，7B 提升 +2.0，说明显式对齐与模型规模互补
- 在 VisDoc 长文档检索上也取得明显增益（76.1 vs 74.7），说明对齐对异构长视觉输入同样有效

### AudioCaps：音频检索显著提升

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>规模</th><th>Recall@1</th></tr></thead>
    <tbody>
      <tr><td>Tevatron-Omni</td><td>7B</td><td>34.0</td></tr>
      <tr><td>LCO-EMB</td><td>7B</td><td>24.2</td></tr>
      <tr><td>Omni-Embed-Nemotron</td><td>3B</td><td>20.5</td></tr>
      <tr><td>e5-omni-3B</td><td>3B</td><td>34.3</td></tr>
      <tr style="font-weight:bold"><td>e5-omni-7B</td><td>7B</td><td>37.7</td></tr>
    </tbody>
  </table>
</div>

e5-omni-7B 的 Recall@1 达到 **37.7**，显著优于所有基线（p < 0.05），说明对齐组件不仅适用于视觉-语言，也能改善音频嵌入的稳定性。

## Ablation：哪些设计最关键？

<div class="table-wrap">
  <table>
    <thead><tr><th>变体</th><th>MMEB-V2</th><th>AudioCaps</th></tr></thead>
    <tbody>
      <tr style="font-weight:bold"><td>e5-omni-7B (full)</td><td>66.4</td><td>37.7</td></tr>
      <tr><td>w/o Modality-aware Temp.</td><td>65.7 (-0.7)</td><td>36.6 (-1.1)</td></tr>
      <tr><td>w/o Curriculum Schedule</td><td>65.7 (-0.7)</td><td>36.7 (-1.0)</td></tr>
      <tr><td>w/o DCL (w/ Curriculum)</td><td>66.1 (-0.3)</td><td>37.0 (-0.7)</td></tr>
      <tr><td>w/o Whitening & CORAL</td><td>65.9 (-0.5)</td><td>36.3 (-1.4)</td></tr>
    </tbody>
  </table>
</div>

- **三个组件互补**：移除任何一个都会导致性能下降
- **模态感知温度**和**课程调度**贡献最大（各 -0.7 on MMEB-V2）
- **白化+协方差对齐**对 AudioCaps 影响最大（-1.4），说明二阶几何对齐对音频嵌入特别关键

### 嵌入空间诊断

<figure class="figure">
  <img src="assets/papers/e5-omni/fig3_pca.webp" alt="PCA 重叠可视化" loading="lazy" />
  <figcaption>图 3：VOC2007 上 PCA 重叠可视化。左：无对齐版本 centroid=0.530, covgap=0.085；右：e5-omni centroid=0.477, covgap=0.061。显式对齐使查询-目标分布更接近。</figcaption>
</figure>

<figure class="figure">
  <img src="assets/papers/e5-omni/fig4_covariance.webp" alt="协方差差异热力图" loading="lazy" />
  <figcaption>图 4：VOC2007 上协方差差异热力图（32D 随机投影）。e5-omni 的 covdiff Frobenius 范数从 7.14e-04 降至 5.79e-04，二阶统计量不匹配显著减少。</figcaption>
</figure>

### 超参数分析

<figure class="figure">
  <img src="assets/papers/e5-omni/fig5_hyperparam.webp" alt="超参数敏感性分析" loading="lazy" />
  <figcaption>图 5：e5-omni-7B 在不同训练设置下的 MMEB-V2 性能。温度初始化 \(\tau_0 \in [0.015, 0.03]\)、去偏系数 \(\gamma_+ \in [0.1, 0.2]\)、协方差权重 \(\lambda_{\text{coral}} \in [0.02, 0.1]\) 范围内均表现稳健。</figcaption>
</figure>

### 跨骨干迁移性

<div class="table-wrap">
  <table>
    <thead><tr><th>VLM 骨干</th><th>规模</th><th>w/o alignment</th><th>e5-omni</th><th>提升</th></tr></thead>
    <tbody>
      <tr><td>Qwen2.5-VL</td><td>3B</td><td>61.5</td><td>62.9</td><td>+1.4</td></tr>
      <tr><td>Qwen2-VL</td><td>2B</td><td>59.2</td><td>60.5</td><td>+1.3</td></tr>
      <tr><td>LLaVA-OV</td><td>7B</td><td>63.7</td><td>65.4</td><td>+1.7</td></tr>
    </tbody>
  </table>
</div>

e5-omni 在所有测试骨干上都取得一致提升，证明其作为**即插即用策略**的通用性。

## 我的理解与启发

1. **隐式对齐是全模态嵌入的瓶颈**：VLM 预训练的对齐信号是为生成任务优化的，不是为检索设计的。当模态数量增加时，隐式对齐的不足会被放大。e5-omni 的贡献在于把这个问题显式化并给出系统解法。

2. **三个组件对应三个层面的对齐**：温度校准解决一阶尺度问题，负样本课程解决优化动态问题，白化+协方差解决二阶几何问题。这种分层设计比单一技巧更全面。

3. **DCL 去偏在混合模态 batch 中不可或缺**：硬负样本选择在单模态场景下可能没问题，但在全模态 batch 中假负样本风险更高。DCL 提供了一个 principled 的修正机制。

4. **轻量级 = 实用**：不修改骨干架构意味着可以直接应用于任何新发布的 VLM，这对快速跟进最新模型非常重要。

5. **学到的温度值本身就有诊断价值**：音频（0.0219）和视频（0.0223）的温度远大于文本（0.0130）和图像（0.0127），定量证实了这些模态的嵌入分布确实更模糊，需要更平坦的对比梯度。

## 局限与待观察点
- **对齐范围限于检索几何**：不直接改善高层推理或组合理解能力，在多步推理任务上收益可能有限
- **依赖 batch 统计量**：白化和协方差对齐基于 mini-batch 估计，在小 batch 或高度不平衡的模态组成下可能不稳定
- **评测覆盖有限**：仅在 MMEB-V2 和 AudioCaps 上评测，缺少更多音频/视频领域和长程检索场景的验证
- **超参数敏感性**：虽然论文展示了合理的鲁棒范围，但在新的模态组合或数据分布下可能需要重新调优

## 结论
e5-omni 为全模态嵌入提供了一个清晰范式：**不要依赖 VLM 预训练的隐式对齐，而是用轻量级显式对齐组件主动校准跨模态相似度尺度、优化动态和嵌入几何**。它的三个插件式组件（模态感知温度、可控负样本课程+DCL、批量白化+协方差对齐）在不修改骨干的前提下带来一致且显著的性能提升，并且可迁移到多种 VLM 骨干。对于构建全模态 RAG 系统来说，e5-omni 提供了一个实用的嵌入层解决方案。
