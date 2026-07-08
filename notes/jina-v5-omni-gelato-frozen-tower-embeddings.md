---
id: jina-v5-omni-gelato-frozen-tower-embeddings
title: "jina-embeddings-v5-omni：只训 0.35% 参数的冻结塔式全模态嵌入"
category: 多模态 RAG
date: 2026-07-08
order: 5
readTime: 14 min
tags: ["Multimodal Embedding", "Frozen Tower", "GELATO", "Jina Embeddings", "Projector Training", "Matryoshka", "ViDoRe"]
summary: "Jina 提出 GELATO：冻结文本嵌入骨干和视觉/音频编码器，仅训练每模态一个线性投影层加少量分隔符嵌入（占总参数 0.35%），将纯文本嵌入模型扩展为支持文本/图像/音频/视频的全模态嵌入模型。1.57B 的 Small 变体在四模态平均上逼近甚至超越 3-7B 的全参数联合训练模型，文本嵌入与原始纯文本模型逐比特一致。"
---

## 一句话总结

GELATO 的核心洞察是：**如果文本嵌入骨干已经足够好，那么只需要一个线性投影层就能把预训练的视觉/音频编码器\"桥接\"进同一语义空间**——无需联合重训、无需修改骨干，文本嵌入逐比特不变。1.57B 的 jina-embeddings-v5-omni-small 在 MIEB（图像）+ MMEB-Video + MAEB（音频）+ MMTEB（文本）四模态平均上达到 54.04，仅低于 8.93B 的 LCO-Embedding-Omni-7B（54.43），是 **2B 以下参数量级的最强全模态嵌入模型**。

<div class="metric-grid">
  <div class="metric"><strong>0.35%</strong><span>可训练参数占比（仅投影层 + 分隔符嵌入）</span></div>
  <div class="metric"><strong>1.57B → 54.04</strong><span>Small 四模态平均，逼近 8.93B LCO-7B (54.43)</span></div>
  <div class="metric"><strong>79.25</strong><span>ViDoRe 文档检索，0.92B 活跃参数超越 LCO-3B (78.24)</span></div>
  <div class="metric"><strong>逐比特一致</strong><span>文本嵌入与 Jina Embeddings v5 Text 完全相同</span></div>
</div>

## 论文信息

- **标题**：jina-embeddings-v5-omni: Geometry-preserving Embeddings via Locked Aligned Towers
- **作者**：Florian Hönicke, Michael Günther, Andreas Koukounas, Mohammad Kalim Akram, Scott Martens, Saba Sturua, Han Xiao
- **机构**：Jina by Elastic
- **版本**：arXiv:2605.08384v3, 2026-06-05 修订
- **链接**：[arXiv](https://arxiv.org/abs/2605.08384) · [PDF](https://arxiv.org/pdf/2605.08384) · [HuggingFace Collection](https://huggingface.co/collections/jinaai/jina-embeddings-v5-omni)

## 背景：全模态嵌入的两种路线

当前构建全模态嵌入模型有两条主流路线：

1. **联合训练路线**（LCO-Embedding, E5-Omni, Omni-Embed-Nemotron）：从头或在 VLM 基础上联合训练/对齐多个模态编码器。效果好但训练成本高，且可能改变文本嵌入空间的几何结构。
2. **CLIP-style 路线**（LanguageBind, Jina CLIP v2）：独立训练每个模态的对比编码器。但模态间存在明显的\"模态间隙\"（modality gap），不同模态的嵌入占据共享空间的不同区域。

GELATO 提出**第三条路线**：选一个已经训好的纯文本嵌入模型作为冻结骨干，接入同样冻结的视觉/音频编码器，**仅训练连接两者的线性投影层**。这与 LiT（Locked-image Text Tuning）的冻结塔思路一脉相承，但首次扩展到了图像 + 视频 + 音频三模态同时支持，且文本嵌入完全不变。

<figure class="figure">
  <img src="assets/papers/jina-v5-omni/fig1_performance_vs_params.webp" alt="多模态嵌入性能 vs 参数量" loading="lazy" />
  <figcaption>图 1：六个开源全模态嵌入模型在 MIEB/MMEB-Video/MAEB/MMTEB 四模态平均得分 vs 参数量。jina-v5-omni-small (1.57B) 以极小参数量达到与 LCO-7B (8.93B) 几乎持平的性能，位于效率前沿的最左上角。</figcaption>
</figure>

## 方法：GELATO 架构与训练

### 整体架构

<figure class="figure">
  <img src="assets/papers/jina-v5-omni/fig2_architecture.webp" alt="jina-embeddings-v5-omni 架构" loading="lazy" />
  <figcaption>图 2：jina-embeddings-v5-omni-small 架构。冻结的视觉编码器（Qwen3.5 ViT）和音频编码器（Qwen2.5-Omni Whisper）通过可训练的投影层注入冻结的 Jina Embeddings v5 Text 骨干。任务级 LoRA 适配器（检索/分类/聚类/文本匹配）在投影层之上切换。</figcaption>
</figure>

### 两个模型规模

| 模型 | 文本骨干 | 视觉编码器 | 总参数 | 文本隐藏维度 |
|------|---------|-----------|:---:|:---:|
| jina-v5-omni-nano | jina-v5-text-nano (0.24B) | Qwen3.5-0.8B ViT | 0.95B | 768 |
| jina-v5-omni-small | jina-v5-text-small (0.67B) | Qwen3.5-2B ViT | 1.57B | 1024 |

两个模型共享同一个音频编码器（Qwen2.5-Omni 的 Whisper-large-v3 变体，输出 1280 维）。

### 投影层设计

**视觉投影**：复用 Qwen3.5 的视觉投影器结构（LayerNorm → 2×2 空间合并 → fc_vision_1 → GELU → fc_vision_2），但**仅替换最后一层 fc_vision_2**（从原来的 4096→2048 或 3072→1024 改为 4096→1024 或 3072→768），随机初始化并训练。LayerNorm、空间合并和 fc_vision_1 保持冻结。

**音频投影**：一个单层全连接层 fc_audio，将 1280 维音频编码器输出投影到文本隐藏维度（1024 或 768），随机初始化并训练。

**可训练参数集**：fc_vision_2 + fc_audio + 模态分隔符嵌入（vision_start/end, audio_start/end）。对 Small 模型，视觉投影 4.20M 参数 + 音频投影 1.31M 参数，仅占总参数的 **0.35%**。

### 输入序列构造

非文本模态用特殊分隔符包裹：图像为 `<|vision_start|> <|image_pad|>×N <|vision_end|>`，音频为 `<|audio_start|> <|audio_pad|>×K <|audio_end|>`。视频是逐帧拼接的视觉段，如有音轨则音频段在前。占位符位置被投影后的编码器特征覆盖，不作为独立嵌入学习。

### 训练：InfoNCE + Matryoshka

使用双向 in-batch InfoNCE 对比学习损失，结合 **Matryoshka Representation Learning**：

\[
\mathcal{L} = \sum_{k \in \mathcal{K}} \mathcal{L}_{\text{NCE}}^{(k)}
\]

其中 \(\mathcal{K}_{\text{Small}} = \{32, 64, 128, 256, 512, 768, 1024\}\)，\(\mathcal{K}_{\text{Nano}} = \{32, 64, 128, 256, 512, 768\}\)。温度 \(\tau = 0.02\)，全局 batch 128。

每个任务（检索、分类、聚类、文本匹配）独立训练投影层和分隔符嵌入，文本骨干和 LoRA 适配器始终冻结。

<figure class="figure">
  <img src="assets/papers/jina-v5-omni/fig3_training_data.webp" alt="训练数据分布" loading="lazy" />
  <figcaption>图 3：训练 token 的语义类型分布。图像以自然照片（35.5%）和医学图像（30.3%）为主；音频以音乐（55.0%）和环境声（25.5%）为主。文档 OCR 和图表类数据占比适中。</figcaption>
</figure>

## 实验结果

### 四模态总体对比

<div class="table-wrap">
<table>
<thead><tr><th>模型</th><th>参数 (B)</th><th>文本</th><th>图像</th><th>视频</th><th>音频</th><th>平均</th></tr></thead>
<tbody>
<tr><td>jina-v5-omni-nano</td><td>0.95</td><td>65.52</td><td>47.87</td><td>26.87</td><td>49.69</td><td>47.49</td></tr>
<tr><td>LanguageBind</td><td>1.14</td><td>27.34</td><td>47.80</td><td>48.06</td><td>20.08</td><td>35.82</td></tr>
<tr style="font-weight:bold"><td>jina-v5-omni-small</td><td>1.57</td><td>67.00</td><td>58.00</td><td>41.20</td><td>49.96</td><td>54.04</td></tr>
<tr><td>Omni-Embed-Nemotron-3B</td><td>4.70</td><td>47.64</td><td>44.47</td><td>24.46</td><td>48.27</td><td>41.21</td></tr>
<tr><td>LCO-Embedding-Omni-3B</td><td>4.70</td><td>57.55</td><td>58.42</td><td>46.84</td><td>52.51</td><td>53.83</td></tr>
<tr><td>LCO-Embedding-Omni-7B</td><td>8.93</td><td>59.31</td><td>58.64</td><td>47.41</td><td>52.37</td><td>54.43</td></tr>
</tbody>
</table>
</div>

**关键发现**：
- **Small (1.57B) 四模态平均 54.04**，仅低于 LCO-7B (54.43)，超过 LCO-3B (53.83)，且参数量仅为 LCO-7B 的 18%
- **文本嵌入最强**（67.00），因为直接继承了 Jina v5 Text 的冻结骨干
- **图像和音频有竞争力**，但**视频是明显弱项**（41.20 vs LCO-3B 的 46.84、LanguageBind 的 48.06）
- Nano (0.95B) 以不到 1B 参数仍达到 47.49，超过 LanguageBind (1.14B, 35.82) 和 Nemotron-3B (4.70B, 41.21)

### 图像任务细分

<div class="table-wrap">
<table>
<thead><tr><th>任务</th><th>#Tasks</th><th>Nano</th><th>Small</th><th>最强开源基线</th></tr></thead>
<tbody>
<tr><td>图像分类</td><td>45</td><td>53.26</td><td>68.99</td><td>64.30 (LCO-3B)</td></tr>
<tr><td>图像聚类</td><td>5</td><td>72.28</td><td>86.01</td><td>83.24 (LCO-3B)</td></tr>
<tr><td>视觉 STS</td><td>7</td><td>75.92</td><td>81.74</td><td>79.62 (LCO-3B)</td></tr>
<tr><td>检索</td><td>45</td><td>30.66</td><td>37.95</td><td>46.29 (LCO-3B)</td></tr>
<tr><td>文档检索</td><td>10</td><td>79.25</td><td>79.25</td><td>85.64 (Nemotron-3B)</td></tr>
</tbody>
</table>
</div>

- **分类、聚类、STS 上 Small 超过 LCO-3B**，说明冻结塔的视觉特征质量高
- **通用检索仍然落后**（37.95 vs 46.29），投影层可能丢失了检索所需的细粒度跨模态对齐信号
- **文档检索与 Nano 持平 (79.25)**，以 0.31B 活跃参数超越 LCO-3B (78.24)

### 文档检索（ViDoRe）

<div class="table-wrap">
<table>
<thead><tr><th>模型</th><th>活跃参数 (B)</th><th>文档检索</th></tr></thead>
<tbody>
<tr><td>jina-v5-omni-nano</td><td>0.31</td><td>79.25</td></tr>
<tr><td>jina-v5-omni-small</td><td>0.92</td><td>79.25</td></tr>
<tr><td>LCO-Embedding-Omni-3B</td><td>4.07</td><td>78.24</td></tr>
<tr><td>LCO-Embedding-Omni-7B</td><td>8.93</td><td>80.32</td></tr>
<tr><td>Omni-Embed-Nemotron-3B</td><td>4.70</td><td>85.64</td></tr>
</tbody>
</table>
</div>

Nano 仅用 0.31B 活跃参数就达到 79.25，超过 13 倍参数量的 LCO-3B。ViDoRe 上 Nemotron-3B (85.64) 仍是所有模型中最强的。

### 音频任务

MAEB 总体得分 Small 49.96 / Nano 49.69，接近 LCO-7B (52.37) 但仍有差距。**音频聚类是最大短板**（Small 6.13 vs 专业模型 clap-htsat-fused 22.74），因为单个线性投影层丢弃了音频聚类所需的模态内方差信息。

### 跨语言表现

jina-v5-omni-small 在非英语图像-语言检索和音频检索上**相对基线均值表现强劲**，尤其是多语言图像检索（MIEB Multilingual 63.75 vs LCO-3B 的 69.04 差距小于英语检索差距），说明 Qwen3.5 视觉编码器的多语言能力通过投影层有效迁移。

## 模态几何分析

<figure class="figure">
  <img src="assets/papers/jina-v5-omni/fig4_umap_geometry.webp" alt="UMAP 嵌入空间几何" loading="lazy" />
  <figcaption>图 4：六个模型在 80 个 MSR-VTT 视频片段（每片段 4 个模态嵌入）上的 2D UMAP。LanguageBind 的冻结独立塔产生了经典的\"模态间隙\"（四团分离）；包括 GELATO 在内的统一解码器模型产生交错几何——即使塔是冻结的。</figcaption>
</figure>

MS-COCO 图文跨模态检索（Karpathy split, 2000 图 × 5 caption）：

| 模型 | 参数 | 质心 L2 | R@1 图→文 | R@1 文→图 |
|------|:---:|:---:|:---:|:---:|
| LCO-7B | 8.93B | 0.46 | **74.0** | **63.6** |
| LCO-3B | 3.7B | 0.43 | 71.6 | 58.0 |
| **jina-v5-omni-small** | **1.57B** | 0.71 | 68.0 | 57.0 |
| jina-v5-omni-nano | 0.95B | 0.54 | 36.6 | 27.7 |
| Omni-Embed-Nemotron-3B | 4.7B | 0.92 | 23.1 | 1.4 |

Small 以 1.57B 参数达到 68.0/57.0，与 LCO-3B (3.7B) 在文→图方向上统计无显著差异。注意 Omni-Embed-Nemotron-3B (4.7B) 虽然参数更多，但在跨模态检索上严重退化（文→图仅 1.4%）——参数量不决定跨模态对齐质量。

## Ablation：哪些设计最关键？

<figure class="figure">
  <img src="assets/papers/jina-v5-omni/fig7_8_9_ablations.webp" alt="消融实验结果" loading="lazy" />
  <figcaption>图 7-9：消融实验。上：视觉投影层消融（CIRR-IT2I + NIGHTS-I2I），仅训 fc_vision_2 已接近饱和，解冻编码器仅带来 +0.001。中：音频投影层消融（8-task MAEB），解冻音频编码器的两阶段方案可提升 +0.022。下：Matryoshka 前缀截断跨模态测试，Small 在 128 维仍保持大部分检索精度。</figcaption>
</figure>

### 可训练参数范围

- **视觉**：仅训 fc_vision_2（配置 I）已接近最优。解冻 fc_vision_1（II）、解冻整个 ViT（III）、或两阶段训练（IV/V）几乎无提升（+0.001）。这说明 Qwen3.5 的视觉编码器和 fc_vision_1 已经提供了足够好的语言对齐特征。
- **音频**：仅训 fc_audio（配置 I）是基线。解冻音频编码器做两阶段训练（III）可带来 **+0.022** 的 MAEB 提升——音频投影是当前最弱的桥接路径，有明确的改进空间。

### Matryoshka 截断

训练时对多个前缀维度施加 InfoNCE 损失，使得截断到低维时性能平滑退化。Small 模型从 1024 维截到 128 维，图像检索 nDCG@10 仅从 ~38 降到 ~33，文本从 ~67 降到 ~63，展现了良好的维度-性能权衡。

### 训练效率

| 设置 | 训练范围 | 更新参数 | 每步耗时 | 峰值显存 | 15k 步总时间 |
|------|---------|:---:|:---:|:---:|:---:|
| Small 视觉 | 仅投影 | 4.20M | 0.413s | 7.52 GiB | 103 min |
| Small 视觉 | 全量 | 920.6M | 0.752s | 12.96 GiB | 188 min |
| Small 音频 | 仅投影 | 1.31M | 0.617s | 6.06 GiB | 154 min |
| Small 音频 | 全量 | 1232.1M | 1.989s | 19.53 GiB | 497 min |

投影训练比全量训练**视觉快 1.8×、音频快 3.2-3.9×**，峰值显存也显著更低。

## 我的理解与启发

1. **\"冻结塔 + 线性桥\"是一种被低估的工程范式**。0.35% 参数训练达到联合训练 95%+ 的性能，这对资源有限的团队意义重大。核心前提是文本骨干和模态编码器都已经被充分语言对齐——GELATO 不能替代预训练，它是预训练的\"最后一公里\"。

2. **文本嵌入逐比特不变是生产部署的杀手特性**。企业级 RAG 系统的文本索引已经建好，换嵌入模型意味着全量重建索引。GELATO 允许在不触碰现有文本索引的情况下\"加模态\"，这是一个巨大的工程优势。

3. **视频是 GELATO 的阿喀琉斯之踵**。逐帧拼接的方式缺乏时序建模能力，导致视频任务明显落后。论文也承认这一点但未提出解决方案。对视频检索有高要求的场景仍需专用视频嵌入模型。

4. **音频聚类短板揭示了线性投影的信息瓶颈**。单一线性层无法保留音频聚类所需的模态内方差（6.13 vs 专业模型 22.74）。这与消融实验中音频编码器解冻带来的 +0.022 提升一致——音频桥接需要更多可训练参数。

5. **编码器选择比训练策略更重要**。消融显示解冻编码器几乎无提升，说明 Qwen3.5 ViT 和 Qwen2.5-Omni 音频编码器本身的语言对齐质量已经足够好。GELATO 的上限取决于编码器的选择，而不是投影层的设计。

## 局限与待观察点

- **视频性能显著落后**：是四个模态中唯一明显输给基线的，论文未给出改进方案
- **音频聚类几乎不可用**（6.13 vs 22.74），线性投影的表达能力不足以保留模态内结构
- **跨模态检索仍落后联合训练模型**：MS-COCO 图→文 R@1 差 LCO-7B 约 6%，差距在音频上更大（11-15%）
- **编码器选择未充分探索**：论文仅测试了 Qwen3.5 ViT + Qwen2.5-Omni 音频编码器这一组合
- **仅在 Jina v5 Text 上验证**：GELATO 是否适用于其他文本嵌入骨干（如 GTE、E5-Mistral）尚不清楚，虽然 Nano 和 Small 使用了不同的骨干架构提供了一些暗示
- **投影层独立训练**：每个任务的投影层单独训练，未探索多任务联合训练是否更好

## 结论

GELATO 提出了一种极轻量级的全模态嵌入扩展方案：冻结文本骨干和模态编码器，仅训练连接两者的线性投影层（0.35% 参数），就能将纯文本嵌入模型扩展为支持文本/图像/音频/视频的全模态模型。jina-embeddings-v5-omni-small (1.57B) 在四模态平均上达到 54.04，以 18% 的参数量逼近 8.93B 的 LCO-7B (54.43)。**对\"已有文本索引、需要追加多模态能力\"的生产场景，GELATO 是当前最具工程可行性的方案**——文本嵌入逐比特不变、训练成本极低、部署只需加载额外几 MB 的投影层权重。
