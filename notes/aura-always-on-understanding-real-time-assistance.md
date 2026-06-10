---
id: "aura-always-on-understanding-real-time-assistance"
title: "AURA：面向视频流的常开理解与实时助手"
category: "Streaming VLM"
date: "2026-06-08"
order: 20
readTime: "11 min"
tags: ["VideoLLM","Streaming","Real-Time QA","Proactive Assistant","Qwen3-VL","vLLM"]
summary: "AURA 将流式视频理解从被动问答推进到常开式实时助手：统一模型持续观察视频流，在需要时回答或主动提醒，并通过双滑窗上下文、流式数据构造、静默-发言平衡训练和 vLLM 推理优化实现长时稳定交互。"
hero: "assets/papers/aura/x3.png"
---

## 一句话总结
AURA 的核心价值在于把 VideoLLM 做成一个 **Always-On 的视频流助手**：它不是等用户上传完整视频后离线分析，也不是只做连续 caption，而是持续接收视频流，在每个时间片判断“该沉默还是该响应”，从而同时支持实时问答、延迟式主动回答和多次响应的长期监控任务。

<div class="metric-grid">
  <div class="metric"><strong>73.1%</strong><span>StreamingBench 总体准确率</span></div>
  <div class="metric"><strong>65.3%</strong><span>OVO-Bench 总体准确率</span></div>
  <div class="metric"><strong>25.4%</strong><span>OmniMMI 总体准确率</span></div>
  <div class="metric"><strong>~312 ms</strong><span>端到端首句响应延迟估计</span></div>
</div>

## 论文信息
- **标题**：AURA: Always-On Understanding and Real-Time Assistance via Video Streams
- **作者**：Xudong Lu, Yang Bo, Jinpeng Chen, Shuhan Li, Xintong Guo, Huankang Guan, Fang Liu, Dunyuan Xu, Peiwen Sun, Heyang Sun, Rui Liu, Hongsheng Li
- **机构**：Huawei Research, CUHK MMLab
- **版本**：arXiv 2026
- **链接**：[arXiv:2604.04184](https://arxiv.org/abs/2604.04184) · [GitHub](https://github.com/aurateam2026/AURA) · [项目主页](https://aurateam2026.github.io/) · [Hugging Face](https://huggingface.co/aurateam/AURA/)

## 背景：为什么需要 Always-On 视频助手？
多数 VideoLLM 仍然偏离线：先缓存完整视频或片段，再进行理解。这适合事后分析，但不适合实时助手、机器人、看护、交通和监控等场景，因为这些任务需要系统 **持续观察、及时响应，并且能在没有明确用户问题时主动提醒**。

现有 streaming VideoLLM 主要有两类不足：

- **解耦式触发-响应架构不稳定**：小触发模型决定是否调用主模型，但触发模型和主模型不共享完整上下文，容易出现触发不准或响应不一致。
- **统一式架构能力偏窄**：很多方法更接近实时 caption 或解说，不能很好支持开放式问答、延迟响应和长时间交互。

AURA 的目标是把这些能力统一到一个 VideoLLM 内：模型既要理解画面，也要学会什么时候不说话、什么时候立即回答、什么时候等未来证据出现后再回答。

## 方法：AURA 的四个核心设计
### 1. 交互式视频流上下文管理
AURA 将视频流按小时间片组织成对话格式。每个视频 chunk 对应一条 user message：

- 如果此时用户提问，就把问题文本和当前视频 chunk 放在同一条 user message 中。
- 如果没有用户提问，就只放视频 chunk。
- 每条 user message 后都跟一条 assistant message；需要响应时输出文本，不需要响应时输出特殊 token `<|silent|>`。

这个格式把“观察”和“对话”统一成同一个序列，使模型可以在每个时间点进行决策：继续沉默，还是产生响应。

### 2. 双滑窗：近期视觉 + 历史问答
视频流是无界的，直接把所有帧和对话放进上下文会导致显存、延迟和位置长度失控。AURA 使用双滑窗策略：

- **视频滑窗**：只保留最近 N 秒视频，论文默认 N=30。视觉 token 密度高，且多数实时任务依赖近期画面。
- **QA 滑窗**：保留最近 M 组问答，论文默认 M=10。文本 token 更省，而且承载用户意图、历史目标和已发生事件。

当问答落到视频窗口之外时，AURA 会丢弃旧视频 chunk 和 `<|silent|>`，只保留有效文本问答。这一点很实用：旧画面成本高，但旧问题和回答仍可能决定当前任务该如何继续。

<figure class="figure">
  <img src="assets/papers/aura/x3.png" alt="AURA 交互式视频流上下文管理机制" loading="lazy" />
  <figcaption>图 1：AURA 的交互式视频流上下文管理。系统同时维护近期视频窗口和历史 QA 窗口：视频负责当前视觉证据，文本 QA 负责保留用户意图与长期交互状态。</figcaption>
</figure>

### 3. 三类流式问答模式
AURA 明确区分三种交互模式：

- **Real-Time QA**：用户问完后，模型基于当前或历史观察立即回答。
- **Proactive QA**：用户提出一个监控目标后，模型先保持沉默，直到未来证据出现再给出一次回答。
- **Multi-Response QA**：用户提出持续性任务后，模型在事件演化过程中多次响应，不需要用户反复追问。

这三个模式让 AURA 更接近真实助手。例如“我刚才关灯了吗？”属于实时或回溯问答；“水烧开了提醒我”属于主动响应；“看到异常就告诉我”则可能需要多次响应。

<figure class="figure">
  <img src="assets/papers/aura/x4.png" alt="AURA 支持的三类流式 QA 交互" loading="lazy" />
  <figcaption>图 2：AURA 定义三类流式交互：Real-Time QA 立即回答，Proactive QA 等未来证据出现后回答，Multi-Response QA 则在持续事件中多次响应。</figcaption>
</figure>

### 4. 静默-发言平衡训练
流式视频中，大多数时间模型都应该沉默。如果直接训练，`<|silent|>` 会远多于有效回答，模型容易学成“过度沉默”。AURA 使用 Silent-Speech Balanced Loss：

- 只监督所有 silent assistant message 和当前样本最后一个非 silent answer。
- 对 silent token 按样本内 silent 数量进行降权，使沉默和有效发言的训练贡献更均衡。

这个设计解决的是流式助手的核心难题：模型不仅要知道“答什么”，还要知道“什么时候不答”。

## 数据：从粗到细构造流式监督
AURA 构建了一个 Coarse-to-Fine Streaming Data Engine，覆盖五个阶段：

<figure class="figure">
  <img src="assets/papers/aura/x5.png" alt="AURA Coarse-to-Fine Streaming Data Engine" loading="lazy" />
  <figcaption>图 3：AURA 的 Coarse-to-Fine Streaming Data Engine。它从视频准备、QA 合成、QA 精修、流式结构化到质量验证，逐步把普通视频转化为可训练流式交互数据。</figcaption>
</figure>

1. **Video Preparation**：从公开互联网来源收集视频，覆盖体育、vlog、纪录片、百科、影视、课程、游戏、动画等类别，并统一到 2 FPS。
2. **QA Synthesis**：先做场景分割和场景描述，再合成带时间戳的 Real-Time、Proactive 和 Multi-Response QA。
3. **QA Refinement**：增加问题难度、改写问题表达，提升数据多样性。
4. **Streaming Structuring**：把带时间戳的 QA 展开成符合流式上下文管理机制的训练样本。
5. **Quality Verification**：检查答案是否被当前可见视频窗口和保留问答历史支持，过滤幻觉样本。

训练数据规模约为 **115K streaming video QA 样本**，约 **1.04B tokens**；再混合约 **59K 内部离线视频 QA 样本**，总计约 **174K 样本 / 1.2B tokens**。模型基于 `Qwen3-VL-8B-Instruct`，训练时冻结视觉编码器和 connector，仅微调 LLM 部分。

<figure class="figure">
  <img src="assets/papers/aura/x7.png" alt="AURA 训练数据分布" loading="lazy" />
  <figcaption>图 4：训练数据分布。左侧展示 QA 类型分布，右侧展示视频领域分布，说明 AURA 的训练样本同时覆盖多种交互形式和开放域视频内容。</figcaption>
</figure>

## 推理系统：从模型到可交互 Demo
GitHub 仓库释放的不只是权重，还包括一个完整实时系统。整体链路包括：

<figure class="figure">
  <img src="assets/papers/aura/x6.png" alt="AURA 端到端实时推理系统" loading="lazy" />
  <figcaption>图 5：AURA 的端到端实时推理系统。浏览器采集视频和语音，ASR 转写语音，AURA 执行多模态推理，非静默响应再交给 streaming TTS 输出语音。</figcaption>
</figure>

- **视频输入**：浏览器端采集摄像头画面，后端按 chunk 送入模型。
- **语音输入**：ASR 将用户语音转为文本，并和对应时间点的视频 chunk 组合。
- **多模态推理**：AURA 基于当前视频窗口和历史 QA 判断是否输出文本。
- **语音输出**：非静默响应交给 streaming TTS，形成语音反馈。

工程上，AURA 采用 vLLM 推理，并利用 prefix KV cache 复用、滑窗裁剪、异步 ASR/TTS、多进程资源隔离等优化。论文报告 demo 在 **2 FPS** 下运行，使用两个 80G 加速器；端到端延迟估计约为 **312.2 ms**，其中 ASR 约 84.2 ms、TTFT 约 75.0 ms、TTS 首 chunk 约 93.0 ms。

一个重要细节是：如果每来一个新视频 chunk 都 FIFO 删除最旧 chunk，prompt 前缀会频繁变化，prefix cache 难以复用。AURA 允许视频窗口在 N 基础上多浮动 N′，达到 N+N′ 后一次性删除 N′ 个旧 chunk。这样在接下来一段时间内 prefix 更稳定，可以持续复用 KV cache。

## 实验结果
AURA 在三个流式视频理解 benchmark 上都取得了强结果：

<div class="table-wrap">
  <table>
    <thead><tr><th>Benchmark</th><th>AURA</th><th>对比结论</th></tr></thead>
    <tbody>
      <tr><td>StreamingBench</td><td><strong>73.1%</strong></td><td>超过 MiniCPM-o-4.5 10.4 个百分点，也超过 Gemini-1.5-Pro 6.0 个百分点</td></tr>
      <tr><td>OVO-Bench</td><td><strong>65.3%</strong></td><td>超过最佳开源基线 ViSpeak 4.2 个百分点，并高于 Gemini-1.5-Pro 2.3 个百分点</td></tr>
      <tr><td>OmniMMI</td><td><strong>25.4%</strong></td><td>总体准确率超过开源和闭源模型，并具备 Proactive Alerting 能力</td></tr>
    </tbody>
  </table>
</div>

在 StreamingBench 上，AURA 不只是总体分高，而是在 Real-Time Visual Understanding、Omni-Source Understanding、Contextual Understanding 三个高层类别都表现领先。论文也指出，MiniCPM-o-4.5 虽然支持 full-duplex live-streaming，但在长于两分钟的视频流里容易沉默或生成无关响应，这恰好说明长时稳定上下文管理的重要性。

<figure class="figure">
  <img src="assets/papers/aura/x8.png" alt="AURA 推理性能与 TTFT 对比" loading="lazy" />
  <figcaption>图 6：AURA 的推理性能对比。关闭滑窗会让计算 token 数随视频流增长；关闭 prefix caching 会让 TTFT 持续偏高；AURA 同时使用滑窗和 KV 复用来维持低延迟。</figcaption>
</figure>

## 我的理解与启发
AURA 最值得关注的地方不是单个 benchmark 分数，而是它把“实时视频助手”拆成了一个完整系统问题：

- **交互协议**：用 `<|silent|>` 把“沉默”显式建模，而不是把每一帧都变成 caption。
- **上下文策略**：视觉和文本采用不同保留时间尺度，符合多模态信息密度差异。
- **数据闭环**：数据构造直接服务 Real-Time、Proactive、Multi-Response 三类真实交互。
- **推理优化**：滑窗不是只为省 token，还要兼顾 prefix cache 复用，降低在线延迟。

这对产品设计也有启发：一个常开式 AI 助手如果频繁打断用户会很烦，如果一直沉默又没有价值。因此“何时响应”本身就是能力的一部分，需要在数据、训练和系统层面一起设计。

## 局限与待观察点
- **硬件成本仍高**：论文 demo 使用两个 80G 加速器，离低成本部署或端侧常开仍有距离。
- **主动性需要安全边界**：主动提醒很有用，但真实场景中还需要控制误报、隐私和打扰频率。
- **评测仍在发展中**：StreamingBench、OVO-Bench、OmniMMI 已覆盖不少流式能力，但对长期生活场景、机器人和复杂多用户交互的覆盖还有限。
- **长时记忆仍偏窗口化**：双滑窗能稳定运行，但对跨小时、跨天的长期偏好和事件记忆，还需要外部记忆或检索机制补充。

## 结论
AURA 给出了一个清晰方向：未来的视频流 AI 不应只是“看完再答”的离线模型，也不应只是持续生成 caption 的旁白模型，而应该是能持续观察、选择性沉默、及时回答和主动提醒的交互式助手。它把上下文管理、数据构造、训练目标和实时推理框架整合起来，是 VideoLLM 从 benchmark 能力走向真实在线交互的重要一步。
