---
id: "proact-vl-proactive-videollm-real-time-ai-companions"
title: "Proact-VL：面向实时 AI Companion 的主动式 VideoLLM"
category: "Streaming VLM"
date: "2026-06-05"
order: 10
readTime: "13 min"
tags: ["VideoLLM","Proactive AI","Streaming","AI Companion","Game Commentary"]
summary: "Proact-VL 把实时视频理解、主动响应时机判断和短句生成统一到一个流式框架中，并用 Live Gaming Dataset/Benchmark 评估游戏解说、协同解说和实时指导三类 AI Companion 场景。"
hero: "assets/papers/proact-vl/page-1.png"
---

## 一句话总结
Proact-VL 关注的不是“看懂视频后回答问题”，而是 **AI Companion 什么时候该说、说多久、说什么**。论文把游戏解说和玩家指导抽象成可自动评测的实时交互场景：每秒处理视频 chunk，用 `<|FLAG|>` 的隐藏状态判断是否发言，再生成适合一秒播放的短文本；同时通过转移加权分类损失、稳定性正则和滑窗 KV cache，让模型在长时间流式输入下保持低延迟、少打断和较高内容质量。

<div class="metric-grid">
  <div class="metric"><strong>561h</strong><span>Live Gaming Dataset 视频规模</span></div>
  <div class="metric"><strong>12</strong><span>覆盖游戏标题数量</span></div>
  <div class="metric"><strong>64.87</strong><span>Live Gaming Benchmark Overall F1</span></div>
  <div class="metric"><strong>10-15 FPS</strong><span>论文估计的实际可处理视频流吞吐</span></div>
</div>

## 论文信息
- **标题**：Proact-VL: A Proactive VideoLLM for Real-Time AI Companions
- **作者**：Weicai Yan, Yuhong Dai, Qi Ran, Haodong Li, Wang Lin, Tao Jin, Xing Xie, Hao Liao, Jianxun Lian
- **会议/版本**：ICML 2026，arXiv v3，2026-05-24
- **链接**：[arXiv:2603.03447](https://arxiv.org/abs/2603.03447) · [PDF](https://arxiv.org/pdf/2603.03447) · [Homepage](https://proact-vl.github.io)

<figure class="figure">
  <img src="assets/papers/proact-vl/page-1.png" alt="Proact-VL 论文首页与应用概览" loading="lazy" />
  <figcaption>图 1：Proact-VL 面向两类典型 AI Companion：实时游戏解说和玩家指导。核心挑战是既要理解环境，又要决定是否主动开口。</figcaption>
</figure>

## 背景：为什么主动式实时陪伴很难？
这篇论文的切入点很清晰：真正像人的 AI Companion 不只是“能回答”，还必须控制交互节奏。

- **低延迟**：视频是连续流，模型不能等长片段结束后再离线总结。
- **主动时机**：一直说会打扰用户，长期沉默又没有陪伴感，所以模型要学会何时开口。
- **内容长度与密度**：触发后不能生成一大段离线答案，而要输出短、连续、可播放的实时反馈。
- **多人协同**：在 co-commentary 场景中，AI 还要避免打断其他解说者，保持社会性 turn-taking。

已有方法通常偏向两端：proactive 模型会判断何时响应，但一旦触发就生成完整长答案，延迟和粒度都偏粗；real-time 模型能低延迟持续输出，但缺少明确的说话控制，容易话太多。Proact-VL 试图把两者统一起来。

## 数据与评测：Live Gaming Dataset / Benchmark
论文选择游戏场景不是偶然的。游戏直播天然有连续视觉变化、实时事件、解说节奏、协作对话和目标导向指导，而且可以通过时间戳与文本质量进行相对自动化评测。

### 数据规模与场景
- **Live Gaming Dataset**：561 小时英文高质量游戏视频，覆盖 12 个热门游戏标题。
- **三类设置**：Solo Commentary、Co-Commentary、User Guidance。
- **训练集**：约 128,000 samples，包含 10 个游戏域，并融合 LiveCC、Ego4D 等数据。
- **Live Gaming Benchmark**：3,014 个 clip，其中 2,640 个 in-domain 测试样本，另有 Ego4D 和 Black Myth: Wukong 用于 common/general 评测。
- **Live Gaming Benchmark-Streaming**：10 个完整视频，时长从 30 分钟到 2 小时，用于评估长时间流式稳定性。

### 数据处理流程
评论员数据和指导数据走两条处理链路：

- **Commentary**：用 WhisperX-large-v3 做 ASR、说话人识别和时间戳对齐；用 Qwen3-Omni-Flash 标注停顿、笑声、拉长音等副语言信息；再用 DeepSeek-V3.2-Exp 做游戏术语纠错、文本清洗和领域化润色。
- **Guidance**：先把 gameplay 切成 5 分钟片段，用 Qwen3-VL-Plus 找潜在玩家问题和时间区间，再由 GPT-4.1 改写成简洁、教练式、行动导向的指导。
- **Persona Enrichment**：从解说文本中抽取 tone、vocabulary、rhythm/pacing 等 persona，让模型在不同游戏和角色中保持更一致的风格。

<figure class="figure">
  <img src="assets/papers/proact-vl/page-3.png" alt="Live Gaming Dataset 与数据处理流程" loading="lazy" />
  <figcaption>图 2：Live Gaming Dataset 覆盖多种游戏类型；数据管线分别处理评论员语音、玩家指导、persona 与时间对齐。</figcaption>
</figure>

## 方法：Proact-VL 如何实现“边看边决定是否说”？
### 1. 每秒 chunk-wise 输入输出
Proact-VL 把连续视频离散成固定时长 chunk，论文中使用 **1 秒** 作为基本时间步。每个时间步输入三类信息：

- **当前视频 chunk** `V_t`：这一秒看到的画面。
- **可选用户问题** `Q_t`：用户即时提出的目标或问题。
- **环境上下文** `B_t`：上一秒评论、其他助手发言、历史摘要等。

模型输出同样是一段与当前时间步对齐的短 utterance `U_t`。如果需要更长表达，不是在一个时间点一次性生成长答案，而是跨多个连续 chunk 自然延续。这个设计很适合实时 TTS 或字幕系统：每秒只需要产生足够短的一小段内容。

### 2. `<|FLAG|>` 决策 token + response head
Proact-VL 不把“沉默”当作一个普通生成 token，而是在 user message 末尾插入一个无语义的特殊 token `<|FLAG|>`：

1. 模型先做一次 priming forward。
2. 取 `<|FLAG|>` 对应隐藏状态。
3. 送入轻量 MLP response head，得到说话概率 `p_t`。
4. 若 `p_t ≥ τ`，追加 assistant prefix 并生成短句；否则输出 silence placeholder。

这个设计的好处是 **决策与文本生成解耦**。如果直接生成 `<|SILENCE|>`，沉默概率会受 temperature、top-p 等解码参数影响，而且大量 silent assistant 样本容易让模型退化。response head 则更容易调阈值，也更适合线上控制发言密度。

<figure class="figure">
  <img src="assets/papers/proact-vl/page-4.png" alt="Proact-VL 方法结构与主动响应机制" loading="lazy" />
  <figcaption>图 3：Proact-VL 在每秒输入视频、query 和 history，通过 FLAG hidden state 判断是否触发响应；触发后生成短 clip-level 文本，否则保持沉默。</figcaption>
</figure>

### 3. 训练目标：既学“说什么”，也学“何时说”
训练包含两个互补目标：

- **主语言建模损失** `L_main`：监督模型生成什么内容。
- **响应损失** `L_resp`：监督模型何时说话。

响应损失不是简单逐秒 BCE。论文指出关键不只是 silence/response 类别不均衡，而是 **状态转移远少于状态保持**：大多数时间模型要么持续沉默，要么持续说话，真正难学的是从沉默切到说话、或从说话切回沉默。

因此 Proact-VL 引入两部分：

- **Transition-smoothed classification loss**：当 `y_t ≠ y_{t-1}` 时给更高权重，论文中 transition:persistence 约为 1:5，所以设置 `γ=5`。
- **Stability regularization**：一方面约束连续同状态下的概率不要抖动，另一方面让平均说话率接近人类标注的说话率。

这使 response head 学到的不是离散点判断，而是更平滑、可控的说话行为。

### 4. Infinite inference：双缓存滑窗 + reverse RoPE
为了支持长时间视频流，Proact-VL 使用 dual-cache sliding-window KV cache：

- **system cache**：保留初始 system prompt、角色和任务设定。
- **streaming cache**：保存持续到来的 user/assistant token。

当上下文预算接近上限时，系统会删除 streaming cache 中最旧的 20%，保留近期交互。同时，为避免位置编码在长时间运行后漂移到训练分布外，论文对 KV cache 施加 **reverse-RoPE**，把剩余 token 的有效位置重新对齐到较小坐标系中。

这点和 StreamingVLM 的 contiguous RoPE 思路很接近：长时间流式系统的关键不是无限堆上下文，而是维持一个稳定、可复用、位置合理的在线状态。

## 实验结果
### Live Gaming Benchmark：文本质量和主动时机都领先
在主 benchmark 上，Proact-VL 同时对比了三类模型：

- **闭源离线模型**：GPT-4o、Gemini 2.5 Pro。
- **Proactive 模型**：VideoLLM-online、MMDuet、LiveStar。
- **Real-time 模型**：LiveCC、StreamingVLM。

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>Overall CC ↑</th><th>Overall LiveU ↑</th><th>Overall FinalQ ↑</th><th>说明</th></tr></thead>
    <tbody>
      <tr><td>GPT-4o</td><td>39.42</td><td>4.62</td><td>4.80</td><td>离线强模型，但不是专门流式主动交互</td></tr>
      <tr><td>LiveCC-7B-Instruct</td><td>28.35</td><td>4.90</td><td>3.96</td><td>实时输出较强，但缺少主动触发控制</td></tr>
      <tr><td>StreamingVLM</td><td>14.89</td><td>3.49</td><td>2.65</td><td>更关注无限视频流理解，非游戏陪伴场景最优</td></tr>
      <tr><td>Proact-VL</td><td><strong>49.23</strong></td><td><strong>6.52</strong></td><td><strong>5.03</strong></td><td>文本质量和实时可用性整体领先</td></tr>
    </tbody>
  </table>
</div>

主动响应指标上，Proact-VL 的整体 F1 达到 **64.87**，高于 LiveCC-7B-Instruct 的 56.05 和 StreamingVLM 的 50.67；整体 TimeDiff 为 **1.71**，说明触发时间与人类标注更接近。尤其在 Co-Commentary 中，Proact-VL F1 达到 **77.44**，显示它对多人解说中的插话时机有较强建模能力。

### Common / General：能迁移到 Ego4D 和未见游戏
论文还在 common/general 子集上测试 Ego4D 和 Black Myth: Wukong。Proact-VL 在 Ego4D 上取得 **63.43 CC、7.21 LiveU、5.42 FinalQ**，在未见游戏 Black Myth: Wukong 上取得 **55.21 CC、6.22 LiveU、5.24 FinalQ**。这说明它不是只记住训练游戏，而是学到了一定通用的“实时讲解/指导”模式。

### Streaming：长时间运行保持稳定
在 Live Gaming Benchmark-Streaming 上，论文按 10、20、30、40、50 分钟以及 overall 报告结果。Proact-VL 相对 StreamingVLM 的 Streaming Commentary win rate 随时间保持稳定，overall 达到 **82.03**；主动响应 F1 overall 为 **69.23**，TimeDiff overall 为 **0.81**。

这部分对产品很关键：AI Companion 不是运行几十秒，而是要在直播、游戏、教育或陪伴场景里持续在线。Proact-VL 的结果说明，主动触发机制和滑窗缓存没有随着时间明显崩掉。

<figure class="figure">
  <img src="assets/papers/proact-vl/page-5.png" alt="Proact-VL 实验结果表格" loading="lazy" />
  <figcaption>图 4：主实验结果覆盖文本质量、响应质量、common/general 评测和长时间 streaming 评测。</figcaption>
</figure>

## Ablation：哪些设计最关键？
### 响应损失必须同时有分类和稳定性
训练损失消融显示，单独使用响应分类或稳定性正则都不够。完整版本达到 **CC 50.91、TimeDiff 3.41、F1 60.08**；如果缺少稳定性正则，F1 会大幅下降到 11.03，TimeDiff 上升到 18.50。这里说明 response head 最大的风险不是“不会开口”，而是节奏抖动、乱触发或过度沉默。

### 阈值控制触发覆盖率和内容一致性的权衡
响应阈值 `τ` 是一个非常产品化的旋钮：阈值越高，模型越保守，F1 通常下降，但 CC 可能上升；阈值越低，覆盖更多事件，但容易过度触发。论文认为 **0.3-0.5** 是比较稳定的折中区间。

### 窗口大小影响内容质量，但对 F1 影响较小
上下文窗口从 2048 增大到 16K/24K，通常会提升 CC，因为模型能看到更多近期上下文；但 F1 变化相对平稳，说明“何时说”的能力更多来自 response head 和训练目标，而不是单纯更长上下文。

## 效率与部署信号
论文的 streaming inference 表明 Proact-VL 的端到端 chunk 时间大约在 **0.36-0.43 秒**，峰值显存约 **16-17GB**。在 364 tokens/frame、0.3 秒生成预算的假设下，系统预计能稳定处理 **10-15 FPS** 视频流。

这组数字的意义是：Proact-VL 的目标不是离线最高精度，而是让“视频输入—是否说话—短句生成”形成可持续的在线循环。对实际应用来说，这比一次性长答案更接近直播助手、游戏陪练、实时教育和机器人交互的形态。

## 我的理解与启发
这篇论文最值得借鉴的是它把 AI Companion 拆成了三个可控模块：

1. **连续感知**：每秒 chunk-wise 输入，持续吸收环境变化。
2. **主动决策**：用 response head 明确建模“该不该说”。
3. **短句生成**：把输出限制在实时可播放的 clip-level 片段。

这比传统 VQA 更像一个在线系统。很多多模态 Agent 的问题不在于“模型是否知道答案”，而在于 **什么时候介入才有帮助**。Proact-VL 给出的答案是：把介入时机作为一等公民训练和评测，而不是让生成模型隐式学会沉默。

它也提示了一个工程方向：未来实时多模态产品可能需要同时暴露多个控制旋钮，例如响应阈值、说话率、最大句长、打断策略、persona、窗口大小。模型本身负责感知和生成，但产品体验很大程度取决于这些在线控制策略。

## 局限与待观察点
- **场景仍偏游戏**：游戏适合构造 benchmark，但迁移到真实陪伴、会议、教育、机器人等场景还需要验证。
- **评测依赖 LLM-as-a-Judge**：论文做了 judge 稳定性和人类对齐分析，但绝对分数仍可能存在模型偏好。
- **OCR 与数值推理仍是短板**：failure case 中模型把 0.1K 金币差距误说成 2K lead，说明 HUD 小字识别和轻量计算仍不稳。
- **复杂界面容易触发空泛重复**：信息密集时，模型可能进入“想说但不知道说什么”的模式，输出重复感叹词。
- **安全与个性化控制仍不足**：AI Companion 越像人，越需要可解释的打断策略、风格边界和内容安全机制。

## 结论
Proact-VL 的价值在于把实时 VideoLLM 从“低延迟生成”推进到“主动陪伴交互”：它不仅看视频，也学习什么时候开口、何时沉默、如何用短句持续参与。对实时多模态 Agent 来说，这篇论文给了一个很清楚的系统范式：**chunk-wise streaming + explicit response head + stability-aware training + windowed KV cache**，让模型在长时间视频流中更像一个可控、低延迟、有节奏感的陪伴者。