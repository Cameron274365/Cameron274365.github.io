---
id: "roma-real-time-omni-multimodal-assistant"
title: "ROMA：统一主动与被动交互的实时全模态助手"
category: "论文阅读"
date: "2026-06-09"
order: 10
readTime: "14 min"
tags: ["Omni-Modal","Streaming Video","Audio-Visual","Proactive Response","TMRoPE","Speak Head","Benchmark"]
summary: "ROMA 面向实时音视频流交互，把被动问答和主动响应统一到同一个 omni-multimodal assistant 中：用一秒级多模态单元对齐音频与视频，用 chunked TMRoPE 维护连续时间线，并用轻量 speak head 专门判断何时开口。"
hero: "assets/papers/roma/x1.png"
---

## 一句话总结
ROMA 的核心价值是把实时音视频助手从“用户问了再回答”推进到 **既能被动 QA，也能主动监控、主动提醒、实时解说** 的统一交互范式。它用一秒级 multimodal unit 解决音频稠密、视频稀疏的粒度错配；用 chunked TMRoPE 在流式输入中保持跨模态时间对齐；再用独立的 speak head 判断“什么时候该说话”，避免把响应时机和文本生成混在同一个 LM head 里。

<div class="metric-grid">
  <div class="metric"><strong>676,731</strong><span>流式训练样本总量</span></div>
  <div class="metric"><strong>0.3697s</strong><span>平均处理一个 1 秒 multimodal unit</span></div>
  <div class="metric"><strong>53.7</strong><span>QVHighlights mAP</span></div>
  <div class="metric"><strong>35.21</strong><span>YouCook2 实时解说 F1</span></div>
</div>

## 论文信息
- **标题**：ROMA: Real-time Omni-Multimodal Assistant with Interactive Streaming Understanding
- **作者**：Xueyun Tian, Wei Li, Bingbing Xu, Heng Dong, Yuanzhuo Wang, Huawei Shen
- **机构**：CAS Key Laboratory of AI Safety, Institute of Computing Technology, CAS；University of Chinese Academy of Sciences；Tsinghua University
- **版本**：arXiv 2026
- **链接**：[arXiv:2601.10323](https://arxiv.org/abs/2601.10323) · [PDF](https://arxiv.org/pdf/2601.10323) · [项目页](https://eureka-maggie.github.io/ROMA_show/)

<figure class="figure">
  <img src="assets/papers/roma/x1.png" alt="ROMA 支持主动提醒、实时解说和被动问答" loading="lazy" />
  <figcaption>图 1：ROMA 的目标是统一 proactive 与 reactive 两类交互。模型既能在事件发生时主动提醒，也能持续生成实时解说，还能在用户提问后基于音视频上下文回答。</figcaption>
</figure>

## 背景：为什么实时全模态助手仍然困难？
现有 omni-multimodal LLM 已经能统一处理语音、视觉和文本，但真正的流式音视频理解仍然有明显断层：

- **能力形态割裂**：一些工作偏语音对话，缺少视觉感知；一些工作处理视频流，但通常忽略同步音频，或者只支持提醒、解说、问答中的某一种任务。
- **交互范式割裂**：reactive QA 是“用户问了再答”，proactive monitoring 是“持续观察，条件满足才开口”。两者对时机、上下文和生成策略的要求不同。
- **音视频粒度错配**：音频是连续稠密信号，视频帧是离散稀疏采样；如果没有统一时间线，模型很难稳定对齐“某个声音”和“某个画面”。
- **响应时机不可混入生成任务**：让 LM head 同时负责“说什么”和“何时说”，容易受到文本生成偏置干扰；而实时助手最关键的往往是该不该说、何时说。

因此，ROMA 关注的不是单纯提高长视频 QA 分数，而是把在线音视频流建模为一个连续交互系统：模型只能看到 stream prefix，需要在每个时间步判断是否响应，并在响应时生成合适内容。

## 方法：ROMA 的三个核心设计
<figure class="figure">
  <img src="assets/papers/roma/x2.png" alt="ROMA 模型架构" loading="lazy" />
  <figcaption>图 2：ROMA 将连续输入切成对齐的 multimodal units；speak head 判断响应时机，超过阈值后再激活 LM head 生成文本。</figcaption>
</figure>

### 1. 一秒级 multimodal unit：把音频和视频放到同一时间格
ROMA 将连续音频切成 **1 秒 intervals**，并与对应视频帧组成 temporally aligned multimodal units。每个 unit 会按时间顺序送入 LLM backbone，让模型在流式过程中逐步积累上下文。

这个设计的直觉很清楚：音频和视频天然同步，但表示粒度不同。ROMA 不试图把整段音视频一次性扔给模型，而是先把它们整理成统一时间格，再让模型像处理时间序列一样持续消费。

### 2. Chunked TMRoPE：让流式输入保留连续时间线
ROMA 改造 Qwen2.5-Omni 的 Time-aligned Multimodal RoPE，用于 chunked audio-video streams：

- 每个 1 秒 unit 内，视觉 token 与音频 token 共享时间边界。
- 视频 token 在该 unit 内共享常量 temporal ID，因为多帧视觉输入在编码阶段被聚合成融合表示。
- 音频 token 保留更细粒度的时间 ID，约 **40ms** 一个位置，以保留声音细节。
- 后续 unit 从前一个 unit 的最大 position ID 继续累加，形成全局递增时间线。

<figure class="figure">
  <img src="assets/papers/roma/x3.png" alt="Chunked TMRoPE" loading="lazy" />
  <figcaption>图 3：Chunked TMRoPE 将离散 unit 串成连续时间轴，并在每个 unit 内对齐视觉和音频 token 的边界。</figcaption>
</figure>

我的理解是，Chunked TMRoPE 解决的是流式系统里非常基础但关键的问题：**每次只来一小段输入，但位置编码不能把这些小段看成互不相关的独立样本**。它让模型既能增量处理，又保留“这是同一条音视频时间线”的归纳偏置。

### 3. Speak Head：把“何时说话”从“说什么”中解耦
ROMA 在 LLM backbone 上并联一个轻量 **two-layer MLP speak head**。每处理完一个 1 秒 multimodal unit，speak head 根据当前 stream prefix 输出一个二分类概率：

- 概率超过阈值：触发响应，LM head 开始生成内容。
- 概率未超过阈值：保持沉默，继续消费后续流。

speak head 的输入不是只取最后一层，而是学习聚合最后 \(K\) 层 hidden states，论文默认 \(K=4\)。这样做的意义是：响应时机既需要高层语义判断，也需要稳定的时间线信号；多层聚合比单层更鲁棒。

这个设计相比 silence token 更干净：silence token 把控制决策伪装成普通文本生成，容易被语言分布影响；speak head 则把触发建模为显式概率，便于阈值控制、窗口平滑和时序评测。

## 数据与训练：先适配流式格式，再学习何时响应
<figure class="figure">
  <img src="assets/papers/roma/x4.png" alt="ROMA 流式数据集" loading="lazy" />
  <figcaption>图 4：ROMA 构建了包含 proactive alert、real-time narration 和 reactive QA 的流式数据集，总计 676,731 个样本。</figcaption>
</figure>

ROMA 的训练数据覆盖两大类、三类子任务：

- **Online Proactive：27K**
  - 来源包括 DiDeMo、OOPS、Charades-STA。
  - 改写成 “Alert me when [event] happens” 形式，训练模型持续监控并在事件发生时提醒。
- **Online Narration：109K**
  - 来源包括 MMDuetIT、COIN、YouCook2、ActivityNet。
  - 训练模型只在片段转换处生成简洁 narration，而不是每秒密集输出。
- **Reactive QA：540K**
  - 来源包括 InternVid、CogStream、EgoPlan-IT、AVQA、TimeChat-Online、YouCook2、ViSpeak-Instruct 等。
  - 覆盖过去事件、时间顺序、未来推理和音视频理解，并将文本 query 合成为语音以适配 audio instruction。

### 两阶段微调
ROMA 把训练拆成两个阶段，而不是直接混合所有任务端到端训练：

1. **Stage 1：Streaming Template Alignment**
   - 使用 reactive QA 数据，把离线音视频理解迁移到流式 multimodal unit 输入格式。
   - 目标仍是标准 autoregressive LM loss，重点是消除 offline template 到 streaming template 的分布偏移。
2. **Stage 2：Time-Aware Decision Making**
   - 激活 speak head，把每个 unit 的响应时机建模为二分类。
   - 使用 weighted BCE 处理正样本稀疏问题，默认 \(w_{pos}=3\)。
   - 混入少量 Stage 1 QA 数据并保留 LM loss，避免模型只优化触发时机而损伤生成质量。

推理时，ROMA 以 **2 FPS** 采样视频，单帧最大像素数 65,536，并维护 persistent KV cache。论文报告平均处理一个 1 秒 unit 需要 **0.3697 秒**；同时限制每个 segment 最多生成 25 tokens，用近似流水线方式模拟实时响应。

## 评测：统一 proactive 与 reactive 两类协议
论文认为现有 benchmark 的任务边界和协议比较碎片化，因此把评测重组为两类：

- **Proactive interaction**：用户在开头给出指令，模型持续监控流并自主决定什么时候响应。
  - Event-driven alert：检测事件并及时提醒。
  - Real-time narration：在视频推进中增量生成描述。
- **Reactive interaction**：用户提问后，模型基于已累积上下文回答。
  - OVO-Bench、StreamingBench 使用文本 query 以公平比较 streaming VideoLLM。
  - Video-MME、EgoSchema 使用语音 query，评估更接近真实对话的全模态能力。

## 实验结果
### Proactive Alert：时间定位和动态触发都显著领先
在静态 temporal grounding 上，ROMA 在 QVHighlights 达到 **53.7 mAP / 53.0 HIT@1**，明显高于 MMDuet 的 31.3 / 49.6；在 Charades-STA 上达到 **44.3 / 19.9 R@0.5/0.7**。

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>QVHighlights mAP / HIT@1</th><th>Charades-STA R@0.5 / 0.7</th></tr></thead>
    <tbody>
      <tr><td>TimeChat</td><td>14.5 / 23.9</td><td>32.2 / 13.4</td></tr>
      <tr><td>VTG-LLM</td><td>16.5 / 33.5</td><td>33.8 / 15.7</td></tr>
      <tr><td>MMDuet</td><td>31.3 / 49.6</td><td>42.4 / 18.0</td></tr>
      <tr><td>ROMA</td><td><strong>53.7 / 53.0</strong></td><td><strong>44.3 / 19.9</strong></td></tr>
    </tbody>
  </table>
</div>

在动态流式决策上，ROMA 在 PA、PO、REC 三类任务上取得最好结果：**PA 37.50、PO 53.60、REC 33.81**。CRR 上 Dispider 更高，但 ROMA 仍显著优于 VideoLLM-online 和 MMDuet。

### Real-Time Narration：触发更准，内容质量也更好
ROMA 在 YouCook2 实时解说上达到 **35.21 F1**，超过 TimeChat、VTG-LLM、VideoLLM-online 和 MMDuet；在 OVO-Bench SSR 上达到 **14.54 F1**。更重要的是，它在 GPT-4o 评分上也最高，说明不是只触发得准，生成内容的连贯性、对齐性和简洁性也更好。

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>YouCook2 F1 / GPT</th><th>OVO-Bench SSR F1 / GPT</th></tr></thead>
    <tbody>
      <tr><td>VideoLLM-online</td><td>18.82 / 0.17</td><td>10.24 / 0.18</td></tr>
      <tr><td>MMDuet</td><td>17.81 / 0.23</td><td>9.02 / 0.31</td></tr>
      <tr><td>ROMA</td><td><strong>35.21 / 0.39</strong></td><td><strong>14.54 / 0.42</strong></td></tr>
    </tbody>
  </table>
</div>

<figure class="figure">
  <img src="assets/papers/roma/x6.png" alt="ROMA 实时解说案例" loading="lazy" />
  <figcaption>图 5：实时解说案例。ROMA 需要在片段转换或关键动作出现时开口，并把之前生成的内容继续作为后续上下文。</figcaption>
</figure>

### Reactive QA：保持竞争力，并受益于音视频联合建模
在 OVO-Bench 上，ROMA 在 Real-time Visual Perception 与 Backward Tracing 多个子项取得最好结果，例如 OCR 63.09、ATR 68.10、FPD 69.31、EPM 55.89、HLD 23.66。

在 StreamingBench 上，ROMA 在 Real-Time Visual Understanding、Omni-Source Understanding 和 Contextual Understanding 中整体领先，尤其在 ATP 82.05、PR 82.41、MA 58.80、SQA 44.47 等子任务上表现突出。

在全模态 QA 上，ROMA 使用语音问题和联合音视频输入，在 Video-MME 无字幕设置达到 **33.30**，超过 Qwen2.5-Omni、VITA-1.5 和 MiniCPM-o；EgoSchema 为 **55.40**，略低于 Qwen2.5-Omni 的 58.40，但仍有竞争力。

## Ablation：哪些设计最关键？
### 两阶段训练优于直接混合训练
直接把所有数据混合并使用 Stage 2 目标训练，会让动态决策和 narration 明显下降。例如 REC 从 33.81 降到 13.13，OVO-Bench SSR F1 从 14.54 降到 8.88。这说明流式能力不是简单多任务混合就能学好，先适配格式、再学习时机更稳。

### Speak Head 是 proactive 能力的关键
去掉 speak head、改用 silence token 后，动态触发表现大幅下降：PA 从 37.50 降到 12.50，PO 从 53.60 降到 12.00，REC 从 33.81 降到 6.46；YouCook2 F1 也从 35.21 降到 9.25。

这组结果非常说明问题：**响应时机不是普通文本生成的副产品，而是需要独立建模的在线决策变量**。

### 多层 hidden state 聚合比只用最后一层更稳
当 speak head 只使用最后一层 \(K=1\) 时，时间定位和动态触发普遍下降。例如 QVHighlights mAP 从 53.7 降到 46.4，REC 从 33.81 降到 24.32。说明最后几层聚合能提供更稳定的时序触发信号。

### \(w_{pos}\) 影响主动任务，但对普通 QA 较不敏感
由于主动响应的正样本时刻很稀疏，weighted BCE 的正样本权重会明显影响 trigger 行为。论文默认 \(w_{pos}=3\)，在主动提醒和实时解说之间取得较平衡表现；而 reactive QA 对这个超参相对不敏感。

## 我的理解与启发
ROMA 最值得借鉴的地方，是它把实时多模态助手拆成了两个互补问题：

1. **状态表示问题**：如何把持续到来的音频、视频和查询放到统一时间轴上？ROMA 用 multimodal unit + chunked TMRoPE 给出了简洁答案。
2. **行动时机问题**：模型什么时候应该从观察者变成说话者？ROMA 用 speak head 把这个问题显式建模，避免生成头承担不该承担的控制逻辑。

这对产品化实时助手很有启发。真正的 always-on AI 不只是“能理解视频”，还要能判断“现在是否值得打断用户”。如果没有独立的响应时机模块，系统很容易变成两种极端：要么频繁打扰，要么关键时刻沉默。

<figure class="figure">
  <img src="assets/papers/roma/x5.png" alt="ROMA 主动提醒案例" loading="lazy" />
  <figcaption>图 6：主动提醒案例。模型需要持续观察 stream prefix，在目标事件真正发生时触发，而不能依赖未来上下文。</figcaption>
</figure>

## 局限与待观察点
- **仍依赖强基座模型能力**：ROMA 建立在 Qwen2.5-Omni 等强 omni foundation model 之上，方法迁移到较弱模型或端侧模型时可能不稳定。
- **实时性仍是近似流水线**：论文用 25 tokens/segment 和 pipelined setup 近似实时交互，真实低延迟语音对话还需要更细的系统工程。
- **阈值与正样本权重需要调参**：speak head 输出概率虽然可控，但不同任务对误报、漏报的容忍度不同，部署时需要场景化校准。
- **主动响应的评测仍在演进**：论文已经整合 12 个 benchmark，但 proactive assistant 的用户体验还涉及打断成本、响应频率、安全性和个性化偏好，这些很难完全由现有指标覆盖。
- **音频查询合成可能有分布差异**：部分 reactive QA 使用 TTS 合成语音问题，和真实用户语音的口音、停顿、噪声仍有差距。

## 结论
ROMA 给实时全模态助手提供了一套清晰范式：用一秒级音视频单元和 chunked TMRoPE 建立流式时间对齐，用 speak head 专门学习响应时机，再通过两阶段训练把离线理解能力迁移到在线主动交互。它的贡献不只是若干 benchmark 的提升，而是把“实时看、实时听、该说时再说”的产品形态，拆解成了可训练、可评测、可迭代的模型结构。