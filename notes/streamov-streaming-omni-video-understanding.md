---
id: "streamov-streaming-omni-video-understanding"
title: "StreamOV：基于证据记忆与响应触发的流式全模态视频理解"
category: "Streaming VLM"
date: "2026-06-08"
order: 10
readTime: "13 min"
tags: ["VLM","Omni-Modal","Streaming Video","Audio-Visual","Memory","Proactive Response","Benchmark"]
summary: "StreamOV 面向流式全模态视频理解：它用多模态证据引导的长短期记忆在固定预算下保留关键音视频上下文，并用 MLLM 隐状态触发器决定何时回答。论文同时提出 SOVBench，评测在线、多轮、音视频理解和主动响应能力。"
---

## 一句话总结
StreamOV 的核心是把“看视频回答问题”推进到更真实的 **流式全模态交互**：模型需要持续接收视觉与音频、在有限上下文里保留关键证据，并在证据充分时主动回答、证据不足时保持沉默。论文提出的方案包含三块：多模态证据构造、长短期证据记忆，以及基于 MLLM 早期隐状态的轻量响应触发器。

<div class="metric-grid">
  <div class="metric"><strong>81.6%</strong><span>SOVBench-O 音视频上下文 Avg.</span></div>
  <div class="metric"><strong>83.8%</strong><span>加入 QA context 后 SOVBench-O Avg.</span></div>
  <div class="metric"><strong>90.5</strong><span>SOVBench-T 响应触发 F1</span></div>
  <div class="metric"><strong>1,739</strong><span>SOVBench-O 多轮 QA turns</span></div>
</div>

## 论文信息
- **标题**：StreamOV: Streaming Omni-Video Understanding via Evidence-Guided Memory and Response Triggering
- **作者**：Ming Xie, Zizheng Huang, Xudong Tan, Chao Wang, Xiangyu Zeng, Wenxiao Wu, Tao Chen, Limin Wang, Yanwei Fu
- **机构**：Shanghai Innovation Institute, Fudan University, Nanjing University, Shanghai Artificial Intelligence Laboratory, Huazhong University of Science and Technology
- **版本**：arXiv 2026
- **链接**：[arXiv:2605.25621](https://arxiv.org/abs/2605.25621) · [PDF](https://arxiv.org/pdf/2605.25621)

<figure class="figure">
  <img src="assets/papers/streamov/x1.webp" alt="StreamOV 与 SOVBench 概览" loading="lazy" />
  <figcaption>图 1：StreamOV 同时解决流式音视频记忆和响应时机判断。相比 silence-token 或外部 router，StreamOV 使用紧凑多模态记忆和轻量 hidden-state trigger。</figcaption>
</figure>

## 背景：为什么 streaming omni-video 更难？
之前很多流式视频理解工作主要关注视觉流：怎么采样帧、压缩视觉 token、维护视频记忆。但真实场景里，视频往往同时包含画面、语音、环境音和用户多轮意图，因此更接近 **omni-video understanding**。

论文指出当前方法主要有两类缺口：

- **上下文会无限增长**：离线模型可以一次性看完整视频；流式模型只能边来边处理，不能无上限堆叠所有音视频 token。
- **模型不知道何时该说话**：真实助手不应每个时间步都输出，也不能在证据没出现时胡答；它需要判断 Respond 还是 Wait。
- **现有主动响应机制不够理想**：silence token 会把非语义控制符混进生成分布；外部 router 又可能缺少主 MLLM 的多模态推理能力。
- **评测不匹配真实交互**：很多 benchmark 是离线、单轮 QA，不能检验多轮上下文、音视频联合证据、主动等待和故意沉默。

因此，StreamOV 把任务定义为：模型在连续到来的音视频流中维护一个有界记忆状态 \(M_t\)，并基于当前查询 \(q_t\) 生成回答或选择等待。

## SOVBench：评测在线多轮全模态交互
论文提出 SOVBench，包含两个互补部分：

- **SOVBench-O**：评测在线多轮音视频理解，覆盖 Real-Time、Recall 和 Proactive 三类交互。
- **SOVBench-T**：把“是否该回答”建模成二分类任务，检验模型在证据出现时及时响应、证据缺失时保持沉默的能力。

<figure class="figure">
  <img src="assets/papers/streamov/x2.webp" alt="SOVBench-O 数据统计" loading="lazy" />
  <figcaption>图 2：SOVBench-O 的类别分布、局部 QA group 时长和完整 streaming session 时长统计。</figcaption>
</figure>

### 数据构建方式
SOVBench 以 FineVideo 为主要数据源。FineVideo 提供密集元数据，包括全局描述、场景叙事、局部活动、道具交互和带时间戳 ASR。作者先用强 LLM 从五个维度筛选视频：visual dynamism、narrative coherence、information density、audio-visual alignment 和 reasoning value；随后把视觉事件、音频信号和转写文本整理成严格时间线，再用 Gemini 生成多轮 QA，最后人工验证时间正确性和跨模态 grounding。

### 规模与任务类型
SOVBench-O 包含 **172 个 streaming sessions**、**1,739 个 QA turns** 和 **969 个 temporally coherent dialogue groups**。每个 group 平均 1.79 轮，平均覆盖 15.46 秒视频内容；其中 601 个 group 是多轮交互，说明评测重点不是单次识别，而是连续上下文理解。

三类问题分别对应：

- **Real-Time**：回答正在发生或刚发生的音视频事件。
- **Recall**：回忆较早出现过的视觉、语音或声音证据。
- **Proactive**：用户显式问题被省略或证据尚未出现时，模型需要从上下文判断意图与响应时机。

SOVBench-T 进一步包含 **226 个触发样本**，其中 120 个正样本要求响应，106 个负样本要求保持沉默，接近均衡。

## 方法：StreamOV 的三个核心设计
<figure class="figure">
  <img src="assets/papers/streamov/x3.webp" alt="StreamOV 方法框架" loading="lazy" />
  <figcaption>图 3：StreamOV 从连续音视频流构造多模态证据，更新长短期记忆，再将记忆输入冻结 MLLM；轻量触发器基于早期 hidden state 决定 Respond 或 Wait。</figcaption>
</figure>

### 1. Multimodal Evidence Construction：既看变化，也看相关性
StreamOV 首先为每个时间窗口计算多模态证据分数。这里的关键是同时考虑两类信号：

- **Query-agnostic dynamics**：不依赖具体问题，捕捉流本身的变化。
  - visual change：相邻帧差异，代表画面变化。
  - audio saliency：波形峰值，代表显著声音。
  - audio-visual co-burst：视觉和音频同时爆发，代表跨模态事件。
- **Query-aware relevance**：依赖当前 query，衡量片段和问题的语义相关性。
  - visual semantic score：视觉片段与 query 的匹配。
  - audio semantic score：音频片段与 query 的匹配。

论文还特别处理了音频语义匹配容易受噪声影响的问题：只有当音频本身显著，或存在音视频同步变化时，才强化音频-query 相关性。这个 gating 很实用，因为环境噪声、背景音乐和无关语音很容易误导音频检索。

随后，StreamOV 将证据路由到三类互斥分支：

- **visual-only evidence**：主要由视觉提供的信息。
- **audio-only evidence**：主要由声音或语音提供的信息。
- **audio-visual aligned evidence**：视觉和音频共同支持的信息。

我的理解是，这一步像是在给流式视频做“在线证据标注”：它不只是选重要帧，而是在判断某段证据到底来自画面、声音，还是两者的同步关系。

### 2. Long-Short Term Memory：近期密集，历史稀疏
证据构造后，StreamOV 用长短期记忆维护有界上下文：

- **短期记忆**：从当前时间窗口中保留 Top-\(K_S\) 的密集观察，用来覆盖最近正在发生的事件。
- **长期记忆**：从历史缓存和即将滑出短期窗口的观察中选择 Top-\(K_L\) 高价值证据，用来保留长程信息。

最终记忆 \(M_t\) 是短期记忆和长期记忆的并集，去重后按时间顺序序列化，再以交错的 visual-audio 格式输入基础模型。

这个设计的取舍很清晰：**近期信息需要高密度，历史信息需要高选择性**。如果只保留近期，模型无法回答 Recall；如果长期记忆不筛选，固定预算很快被低价值片段填满。

### 3. Hidden-state-driven Trigger：让 MLLM 自己判断要不要说
主动响应是这篇论文最有意思的部分。作者没有让模型显式生成 `<silence>`，也没有训练一个外部小 router，而是把 MLLM 自身看作 trigger。

具体做法是：当模型看到当前记忆 \(M_t\) 和查询 \(q_t\) 后，提取最后一层 decoder 的早期 hidden states。论文最终实现里只使用 **prefilling 后最后输入位置的 hidden state**，再通过一个轻量 Cross-Attention Trigger 和分类头输出二分类结果：

- **Respond**：证据足够，继续完整自回归生成回答。
- **Wait**：证据不足，立刻截断生成，节省计算并避免幻觉。

这个方案的优点是响应决策仍然来自主 MLLM 的内部语义状态，不需要把“沉默”当作普通 token 学习，也不依赖能力较弱的外部路由模型。

## 实验结果
### SOVBench：在线多轮音视频理解显著提升
在 SOVBench-O 上，StreamOV 在预算受限的在线设置下达到 **81.6% Avg.**；加入 QA context 后达到 **83.8% Avg.**。相比 Qwen3-Omni-30B 的 73.7% / 79.9%，分别提升 **7.9%** 和 **3.9%**。

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>Audio</th><th>输入设置</th><th>SOVBench-O Avg.</th><th>+ QA context Avg.</th><th>SOVBench-T F1</th></tr></thead>
    <tbody>
      <tr><td>Qwen2.5-VL-7B</td><td>✗</td><td>1 FPS</td><td>46.8</td><td>50.6</td><td>Offline</td></tr>
      <tr><td>Qwen2.5-Omni-7B</td><td>✓</td><td>1 FPS</td><td>63.9</td><td>65.5</td><td>Offline</td></tr>
      <tr><td>Qwen3-Omni-30B-A3B</td><td>✓</td><td>1 FPS</td><td>73.7</td><td>79.9</td><td>Offline</td></tr>
      <tr><td>ROMA-7B</td><td>✓</td><td>1 FPS</td><td>60.4</td><td>61.5</td><td>52.2</td></tr>
      <tr><td>StreamOV</td><td>✓</td><td>64 frames budget</td><td><strong>81.6</strong></td><td><strong>83.8</strong></td><td><strong>90.5</strong></td></tr>
    </tbody>
  </table>
</div>

更细看三类任务，StreamOV 在 Proactive 上提升尤其明显：在 audio-visual context 设置下达到 **78.6%**，而 Qwen3-Omni-30B 是 **64.8%**。这说明 evidence memory 和 trigger 不只是提升普通 QA，也增强了模型对“什么时候该响应”的判断。

### StreamingBench：音视频和纯视觉都有效
在 StreamingBench 的 Audio-Visual QA 子集上，StreamOV 平均分达到 **68.6**，超过 Qwen3-Omni-30B 的 **61.0**，也大幅超过 ROMA-7B 的 **46.1**。

更有意思的是，StreamOV 在 Visual-Only QA 子集上也达到 **86.2**，超过 Qwen3-Omni-30B 的 **77.9**。这说明证据路由不是简单依赖音频捷径，而是能更好地筛选和保留视觉证据。

### Offline benchmarks：流式方法也能改善离线音视频理解
在 Video-Holmes 和 Daily-Omni 上，StreamOV 也有竞争力：

- **Video-Holmes Avg.**：StreamOV 53.1，高于 Qwen3-Omni-30B 的 52.8。
- **Daily-Omni Avg.**：StreamOV 69.3，高于 Qwen3-Omni-30B 的 67.8。

这说明 StreamOV 的证据选择并不只适用于在线 streaming；在离线任务中，保留高价值音视频证据同样有帮助。

### OVO-Bench 与 Video-MME：长视频和视觉在线能力
附录中，StreamOV 在 OVO-Bench 达到 **64.0 Overall**，相比 Qwen3-Omni-30B 的 60.3 提升 3.7%，相比 StreamForest 的 55.6 提升 8.4%。

在 Video-MME 无字幕设置下，StreamOV 达到 **73.5 overall** 和 **63.4 long-video**，优于 Qwen3-Omni-30B 的 68.5 / 58.1。这进一步支持论文的核心观点：固定预算下的证据记忆能保留长视频中的关键多模态信息。

## Ablation：哪些设计最关键？
### 证据构造和长期记忆是互补的
主文消融显示：从 73.7% baseline 出发，加入 Query-Aware 特征提升到 **80.0%**，加入 Query-Agnostic 特征提升到 **78.0%**，两者结合达到 **80.4%**，再加入 Long-Memory 达到 **81.6%**。

<div class="table-wrap">
  <table>
    <thead><tr><th>Query-Aware</th><th>Query-Agnostic</th><th>Long-Memory</th><th>Avg. Acc.</th></tr></thead>
    <tbody>
      <tr><td>✗</td><td>✗</td><td>✗</td><td>73.7</td></tr>
      <tr><td>✓</td><td>✗</td><td>✗</td><td>80.0</td></tr>
      <tr><td>✗</td><td>✓</td><td>✗</td><td>78.0</td></tr>
      <tr><td>✓</td><td>✓</td><td>✗</td><td>80.4</td></tr>
      <tr><td>✓</td><td>✓</td><td>✓</td><td><strong>81.6</strong></td></tr>
    </tbody>
  </table>
</div>

我的理解是：Query-Aware 更像“按问题找证据”，适合即时相关片段；Query-Agnostic 更像“不要错过显著事件”，适合捕捉用户尚未明确问到的变化；Long-Memory 则负责让这些证据跨时间保留下来。

### Trigger 只用 prefilling hidden state 已经足够
触发器消融中，Qwen3-Omni + Trigger 的 F1 是 81.4；StreamOV + Trigger 使用一个 prefilling hidden state 时达到 **90.5**；增加第一个 decode hidden state 只提升到 **90.7**。

这说明额外解码带来的收益很小，而只用 prefilling hidden state 可以在生成前立即判断 Respond / Wait，更适合低延迟流式系统。

## 案例分析：更贴合当前音视频证据
<figure class="figure">
  <img src="assets/papers/streamov/x4.webp" alt="StreamOV 流式音视频与视觉理解案例" loading="lazy" />
  <figcaption>图 4：在火箭发射、游戏等案例中，StreamOV 相比 Qwen3-Omni 更能对齐当前时间段的证据，避免把未来或上一段内容误当作答案。</figcaption>
</figure>

<figure class="figure">
  <img src="assets/papers/streamov/x5.webp" alt="StreamOV 不同模态需求案例" loading="lazy" />
  <figcaption>图 5：StreamOV 能根据问题需求使用视觉、音频或音视频联合证据，例如用视觉定位人物位置、用音频捕捉发音信息。</figcaption>
</figure>

案例部分说明了一个容易被平均分掩盖的问题：在线模型最怕 **时间错位**。如果模型把上一段、下一段或全局印象拿来回答当前问题，就算语义上“看起来合理”，在流式交互里也是错误的。StreamOV 的证据记忆通过时间顺序和长短期筛选，能更好地把回答绑定到当前可见、可听的证据上。

## 我的理解与启发
我觉得这篇论文最有价值的地方在于，它把流式多模态助手拆成了三个很工程化的问题：

1. **什么值得记？** 不是所有帧和音频都值得保留，要结合视觉变化、音频显著性、跨模态同步和 query 相关性。
2. **记忆怎么分层？** 最近信息需要密集保留，历史信息需要稀疏筛选；视觉、音频和音视频对齐证据也最好分开管理。
3. **什么时候该回答？** 主动助手不仅要会回答，还要会等待；用 MLLM hidden state 做触发比 silence token 和外部 router 更贴近模型自身理解状态。

和 StreamChat、StreamingVLM、Flash-VStream 这类工作相比，StreamOV 更强调 **音视频联合证据** 和 **响应时机判断**。它不是只把视频流压缩成记忆，也不是只解决实时推理效率，而是把“是否有足够证据可以说话”纳入了模型系统设计。

对产品形态也很有启发：如果要做实时陪伴、会议助手、直播助手、机器人或车载 Agent，系统不能每秒都输出，也不能只等用户显式提问；真正关键的是维护一个可解释、可预算的多模态状态，并在合适时间主动开口。

## 局限与待观察点
- **记忆更新仍偏启发式**：证据分数依赖预定义指标和 Top-K 选择，未来可以探索更自适应或 agentic 的记忆策略。
- **触发动作仍是二分类**：当前只判断 Respond / Wait，但真实交互可能需要追问、部分回答、延迟回答或主动寻证。
- **SOVBench 仍是有限 curated benchmark**：虽然覆盖多轮、音频和主动响应，但距离开放世界噪声流、长时直播和真实用户行为仍有差距。
- **计算资源较高**：实验基于 Qwen3-Omni-30B-A3B，并在 8×H100 80G 上运行；低成本部署还需要更多压缩和系统优化。
- **触发器依赖基础 MLLM 能力**：附录也提到，当音视频证据极稀疏时，trigger 可能受底座模型理解能力限制。

## 结论
StreamOV 给流式全模态视频理解提供了一条清晰路线：先把连续音视频流转化为 visual-only、audio-only 和 audio-visual-aligned 三类证据；再用短期密集、长期稀疏的有界记忆保留关键上下文；最后用 MLLM 早期 hidden state 判断是否该回答。它的贡献不只是 SOVBench 上的性能提升，更重要的是把 **记什么、何时说、何时沉默** 变成了可评测、可实现的系统问题。