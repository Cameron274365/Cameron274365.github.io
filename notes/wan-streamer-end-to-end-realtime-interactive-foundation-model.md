---
id: "wan-streamer-end-to-end-realtime-interactive-foundation-model"
title: "Wan-Streamer v0.1：端到端实时全双工音视频交互基础模型"
category: "Streaming VLM"
date: "2026-06-29"
order: 10
readTime: "12 min"
tags: ["Real-time Interaction", "Full-Duplex", "Streaming VLM", "Audio-Visual Generation", "Thinker-Performer", "Flow Matching", "Block-Causal Attention", "Digital Human"]
summary: "Wan-Streamer 是一个从底层为实时交互设计的全双工基础模型，将语言、音频、视频统一为单 Transformer 中的因果流，不依赖外部 ASR/TTS/Avatar 模块。模型端延迟约 200ms，含网络延迟约 550ms，支持 25fps 同步音视频输出，是目前首个端到端覆盖感知、推理、语音生成与视觉生成的实时交互模型。"
---

## 一句话总结

Wan-Streamer 的核心突破在于：**将感知、语言推理、语音生成、视觉生成、响应时机和轮次管理全部学在一个因果 Transformer 中**，而不是拼装 VAD→ASR→LLM→TTS→Avatar 的级联管线。这使得模型在 160ms 的 streaming unit 内同时消费用户音视频输入并产出同步的语音+视频响应，模型端信号到信号延迟仅约 200ms。

<div class="metric-grid">
  <div class="metric"><strong>~200ms</strong><span>模型端信号到信号响应延迟（含编码、状态更新、latent 生成、解码）</span></div>
  <div class="metric"><strong>~550ms</strong><span>含 350ms 双向网络延迟的总交互延迟，支持亚秒级音视频通信</span></div>
  <div class="metric"><strong>160ms / 25fps</strong><span>最小流式单元 160ms，视频输出 25 帧/秒</span></div>
  <div class="metric"><strong>单模型全栈</strong><span>无需外部 VAD、ASR、LLM、TTS 或视频生成模块</span></div>
</div>

## 论文信息

- **标题**：Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models
- **作者**：Wan Team, Alibaba Group（核心贡献者：Lianghua Huang, Zhi-Fan Wu, Wei Wang 等）
- **机构**：Alibaba Group
- **版本**：arXiv 2606.25041v2 (Jun 2026)
- **链接**：[arXiv](https://arxiv.org/abs/2606.25041) · [PDF](https://arxiv.org/pdf/2606.25041) · [项目主页](https://wan-streamer.com/)

## 背景：为什么实时音视频交互需要端到端模型？

当前实时对话系统（如 GPT-4o Realtime、豆包语音、Gemini Live 等）虽然已能实现亚秒级语音交互，但它们普遍存在两个结构性问题：

1. **非对称或级联架构**：系统由 VAD、ASR、语言模型、TTS、音视频生成等独立模块拼接而成。每个模块边界都引入等待时间，且识别与同步误差逐级累积。即使接口的输入输出看起来是多模态的，文本往往作为中间表示连接各模块，response timing、turn management、身份保持等难以作为一个整体行为来学习。

2. **缺乏真正的视觉全双工**：现有语音全双工模型（如 Moshi、Seeduplex）不生成视觉 agent；而现有 Avatar/数字人系统（如 MIDAS、LiveTalk）通常依赖外部对话和语音模块，其发布的运行指标不包括完整对话路径的延迟。用户说话时 agent 应当表现出可见的倾听行为，agent 响应时也应当持续感知用户的音视频反馈以支持打断和适应。

Wan-Streamer 的设计哲学是：**实时交互不是多模态理解与多模态生成的简单合并，而是一个固有的全双工过程**——streamability 是建模约束而非部署优化。

<figure class="figure">
  <img src="assets/papers/wan-streamer/fig1_overview.webp" alt="Wan-Streamer 架构总览" loading="lazy" />
  <figcaption>图 1：Wan-Streamer 总览。语言、音频、视频作为输入和输出在单个 Transformer 中以 block-causal attention 进行增量流式生成。用户音视频经因果编码器处理后，模型同步产出 Agent 文本、音频和视频，每个 160ms 的 streaming frame 都包含完整的感知-推理-生成循环。</figcaption>
</figure>

## 方法：Wan-Streamer 如何实现端到端流式交互？

### 1. 统一因果流表示

Wan-Streamer 将交互建模为连续的因果流。在第 $k$ 个流式单元中，用户观测 $u_k = (u_k^t, u_k^a, u_k^v)$ 和 agent 响应 $y_k = (y_k^t, y_k^a, y_k^v)$ 共同更新交互历史：

$$p_\theta(y_{1:K} | u_{1:K}) = \prod_{k=1}^{K} p_\theta(y_k^t, y_k^a, y_k^v \mid u_{\le k}^t, u_{\le k}^a, u_{\le k}^v, y_{<k}^t, y_{<k}^a, y_{<k}^v)$$

- **文本**响应为离散 token，使用 cross-entropy loss 做 next-token prediction
- **音频和视频**响应在连续 latent space 中，使用 conditional flow matching 联合生成
- 生成的 clean latent 直接追加到历史上下文中，供后续流式单元使用

关键设计：整个栈从底层就为因果性设计——严格的因果 audio/video VAE、因果编码器、因果解码器、temporal-causal Transformer + block-causal attention。

### 2. 三阶段训练

**阶段一：独立任务预训练**
- 从语言模型初始化，在多模态理解（image/audio/video understanding、ASR、TTS）和生成（image/audio/video/joint audio-visual generation）任务上混合训练
- 目标：使感知、语言推理和 latent generation 在一个序列模型中对齐

**阶段二：端到端交互训练**
- 在双工交互数据上训练，用户 text/audio/video 和 agent text/audio/video 交织在同一因果流中
- 模型必须学会：从当前观测更新状态、生成同步的语言+音频+视频响应、将 clean latent 提交回历史
- Response timing、active listening、interruption handling、long-context consistency 都在推理时使用的因果格式下学习

**阶段三：低延迟蒸馏**
- 将使用 CFG + 多步 flow-matching solver 的强教师蒸馏为高效学生模型
- 采用 **rolling distillation** + self-forcing 策略：学生在连续流式单元上 rollout 并用自己生成的历史训练，减少 train-test mismatch

### 3. Thinker-Performer 推理部署

虽然训练时是单个端到端模型，推理时部署为分离的 thinker-performer 管线以最大化硬件利用率和 overlap：

<div class="table-wrap">
  <table>
    <thead><tr><th>组件</th><th>职责</th></tr></thead>
    <tbody>
      <tr><td><strong>Thinker</strong> (GPU 0)</td><td>因果音视频编码、语言预测与 KV-cache 状态更新、因果音视频解码（输出前一步的 latent）</td></tr>
      <tr><td><strong>Performer</strong> (GPU 1)</td><td>仅运行 flow-matching solver 生成下一步的 clean AV latent</td></tr>
    </tbody>
  </table>
</div>

<figure class="figure">
  <img src="assets/papers/wan-streamer/fig2_thinker_performer.webp" alt="Thinker-Performer 流式推理调度" loading="lazy" />
  <figcaption>图 2：Thinker-Performer 重叠调度。在 unit k，Thinker 编码当前用户观测、更新 KV-cache、解码前一步的响应 latent 并立即发射。Performer 接收 KV slice 并运行 flow-matching solver 生成下一步的 clean 音视频 latent。两部分通过 KV/latent 通信实现 overlap，Performer 时间+通信开销 < 160ms 即可实时运行。</figcaption>
</figure>

**关键洞察**：
- Thinker 不运行昂贵的 flow-matching solver，Performer 不运行解码器
- 当前帧感知+状态更新、前帧输出解码、下帧 latent 去噪、KV/latent 通信在相邻 streaming unit 间 overlap
- 通过 KV-cache 交换保持统一的因果状态——Performer 接收 KV slice 后追加到自己的全历史 cache 中
- 结合 CUDA graph capture、编译优化和高效 kernel，实现约 200ms 模型端延迟

## 数据与评测

**训练数据**：广泛的混合数据，包括：
- 理解类：image/audio/video understanding、text dialogue、ASR、TTS、audio dialogue
- 生成类：image/audio/video generation、joint audio-visual generation
- 交互类：end-to-end duplex interaction data（text/audio/video 同时出现在输入和输出侧）

**评测方式**：论文主要评测延迟指标而非传统 benchmark 分数，因为作为实时交互模型，最关键的是 signal-to-signal 响应延迟。

## 实验结果

### 延迟对比：语音/全模态交互系统

<div class="table-wrap">
<table>
<thead>
<tr><th>系统</th><th>交互类型</th><th>用户可见响应延迟</th><th>关键差异</th></tr>
</thead>
<tbody>
<tr><td>Doubao Realtime Voice</td><td>speech-to-speech</td><td>~1s overall</td><td>纯语音产品，无视觉 agent 输出</td></tr>
<tr><td>GPT-4o Realtime API</td><td>speech + audio/vision in</td><td>~500ms API TTFB, ~800ms target</td><td>混合了模型响应、API TTFB、endpointing 和网络</td></tr>
<tr><td>Hume EVI 3</td><td>speech-to-speech</td><td>0.9–1.4s (web benchmark)</td><td>无视觉输出流</td></tr>
<tr><td>Gemini Live API</td><td>speech-to-speech</td><td>1.2–3.6s (API benchmark)</td><td>无模型端分解</td></tr>
<tr><td>Moshi</td><td>speech-to-speech</td><td>160ms 理论 / 200ms 实际</td><td>原生全双工语音，无视觉 agent</td></tr>
<tr><td>Qwen3.5-Omni</td><td>AV-text in, speech/text out</td><td>first-packet: 235/426ms</td><td>无同步视觉 avatar 生成</td></tr>
<tr><td><strong>Wan-Streamer</strong></td><td><strong>text/AV in & out</strong></td><td><strong>~550ms total (含 350ms 网络)</strong></td><td><strong>单模型端到端，25fps 视频输出</strong></td></tr>
</tbody>
</table>
</div>

**关键发现**：
- **Wan-Streamer 是唯一同时覆盖文本 I/O、用户音视频感知、同步语音和视觉 agent 输出的系统**。其他系统要么是纯语音（Moshi、Doubao），要么接受音视频输入但不关闭视觉输出循环（Qwen3.5-Omni、MiniCPM-o）。
- 模型端 ~200ms 延迟与 Moshi 的 ~200ms 相当，但 Wan-Streamer 额外生成了同步的 25fps 视频响应。
- 论文特别强调延迟指标的可比性：很多系统报告的 first-packet、first-token、API TTFB 等指标与用户感知到的远程交互延迟不直接等价。

### 数字人/视觉 Agent 运行时对比

<div class="table-wrap">
<table>
<thead>
<tr><th>系统</th><th>视觉交互范围</th><th>报告运行时</th><th>与 Wan-Streamer 的核心差异</th></tr>
</thead>
<tbody>
<tr><td>VASA-1</td><td>音频驱动 talking face</td><td>40 FPS, 170ms 前置延迟</td><td>纯渲染器，无对话推理或用户视觉感知</td></tr>
<tr><td>LiveTalk</td><td>多模态交互 Avatar 视频</td><td>24.82 FPS, 0.33s first-frame</td><td>使用 Qwen3-Omni 做语音推理，视频延迟独立</td></tr>
<tr><td>Hallo-Live</td><td>文本驱动 joint AV avatar</td><td>20.38 FPS, 0.94s 延迟</td><td>文本驱动，不持续感知用户音视频</td></tr>
<tr><td>X-Streamer</td><td>从 portrait 开始的开放视频聊天</td><td>25 FPS (双 A100)</td><td>绝对响应延迟未公开</td></tr>
<tr><td><strong>Wan-Streamer</strong></td><td><strong>text/AV 感知对话 + 同步 speech/video</strong></td><td><strong>25 FPS, ~550ms total, ~200ms model</strong></td><td><strong>单因果 Transformer 学习全部行为</strong></td></tr>
</tbody>
</table>
</div>

### 自然交互行为

Wan-Streamer 不仅追求低延迟，还展现出多种自然交互特性：

- **Idle 状态**：agent 在不说话时不会变成静态画面，而是维持身份、注视、姿态、呼吸和微妙的面部运动
- **Listening 状态**：模型能产生与用户语音和视觉线索时序耦合的非语言反馈（注视转移、点头、微表情、姿态变化）
- **打断处理**：全双工行为从交织的交互数据中学习，而非手工规则。推理时模型在生成自己响应的同时持续消费用户音视频，可以在用户自然打断时停止、缩短或重新引导语音
- **主动发言**：当输入流中出现显著的视觉事件、对象、表情或用户动作时，模型可以主动发起相关评论或提问

## 我的理解与启发

1. **"Streamability as modeling constraint" 是核心设计哲学**。不是先训练一个离线多模态模型再做流式适配，而是从 VAE、编码器、解码器到 Transformer 全部为因果性设计。这个思路对任何想做实时系统的团队都值得参考——事后工程化很难恢复真正的低延迟全双工行为。

2. **Thinker-Performer 分离是单模型约束下的工程最优解**。训练时保持一个统一模型，推理时把"理解+状态更新"和"latent 生成"分到两块 GPU 上做 overlap。这比级联管线更优的地方在于：KV-cache 交换保持了统一的交互状态，而非在模块边界丢失信息。

3. **端到端全栈的优势在自然性而非纯延迟**。纯语音系统可以报告更低的延迟数字，但它们不生成同步视觉响应。Avatar 渲染系统可以跑到 40 FPS，但依赖外部对话模块。Wan-Streamer 的价值在于把所有行为（说话、倾听、打断、主动发言）学为一个整体——这使得 lip motion、facial dynamics 和 prosody 天然同步，而非事后对齐。

4. **v0.1 的 192p 输出分辨率是明显的局限**。论文坦承这只是 proof of concept，更高分辨率的扩展留给未来工作。对于实际产品应用，视觉质量仍是关键瓶颈。

5. **缺乏定量质量评测**。论文主要对比延迟指标，但对生成音视频的质量（语音自然度、视频保真度、lip-sync 精度等）没有提供 MOS 或 benchmark 分数。这对于评估实际可用性是一个遗憾。

## 局限与待观察点

- **输出分辨率低**：当前仅验证了 192p 输出，距离实际产品所需的 720p/1080p 还有明显差距
- **缺少质量评测**：没有 FID、FVD、MOS 等音视频质量指标，无法与其他 Avatar 系统做公平的质量对比
- **训练数据和模型细节披露有限**：作为技术报告（v0.1），数据规模、模型参数量、训练算力等关键信息未公开
- **泛化性未知**：论文展示的场景偏 casual conversation，在任务导向对话、多语言、嘈杂环境等场景下的表现有待验证
- **长期一致性**：虽然 rolling distillation 缓解了 long-horizon degradation，但在超长会话（>10 分钟）中的身份保持和一致性仍需观察
- **硬件需求**：推理需要两块 GPU（thinker + performer），对于端侧部署不友好

## 结论

Wan-Streamer v0.1 是首个端到端的实时全双工音视频交互基础模型，将感知、语言推理、语音生成和视觉生成统一在单个因果 Transformer 中。其核心贡献不在于任何一个模块的技术创新，而在于证明了一个从底层为 streamability 设计的全栈模型可以同时实现约 200ms 模型端延迟和自然的交互行为（倾听、打断、主动发言）。虽然 v0.1 在视觉质量和定量评测上还有明显不足，但它指明了一个方向：真正的实时多模态交互不应该是模块拼接的产物，而应该从一开始就被设计为一个原生的全双工系统。
