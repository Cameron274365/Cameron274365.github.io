---
id: "videollm-online-streaming-video"
title: "VideoLLM-online：面向流式视频的在线大语言模型"
category: "Streaming VLM"
date: "2026-06-11"
order: 10
readTime: "15 min"
tags: ["VideoLLM", "Streaming Video", "Online Assistant", "Temporal Alignment", "KV Cache", "Ego4D", "COIN"]
summary: "VideoLLM-online 提出 LIVE（Learning-In-Video-strEam）框架，把传统离线视频理解改造成可持续接收视频流、按时间点主动回答或保持沉默的在线 VideoLLM。核心设计包括 Streaming EOS 训练目标、从离线标注生成流式对话数据，以及连续 KV cache 与编码/解码并行的推理流水线。"
hero: "assets/papers/videollm-online/x1.webp"
---

## 一句话总结
VideoLLM-online 解决的是一个很实际的问题：**视频大模型不能只看用户手动截取的短片段，而应该像 AR 眼镜助手一样持续观看视频流，在合适的时间主动回答，不该说话时保持沉默**。论文提出的 LIVE 框架把训练目标、数据构造和推理流水线一起改造，使模型具备时间对齐、长上下文和实时响应能力；在单张 A100 上处理 5 分钟视频流时可达到 **10 FPS 以上**，显存低于 **20GB**。

<div class="metric-grid">
  <div class="metric"><strong>10+ FPS</strong><span>单张 A100 上 5 分钟视频流平均速度</span></div>
  <div class="metric"><strong>&lt;20GB</strong><span>5 分钟 Ego4D 流式对话显存开销</span></div>
  <div class="metric"><strong>2 FPS</strong><span>训练默认视频采样帧率</span></div>
  <div class="metric"><strong>150</strong><span>过去/当前/未来事件问题模板数量</span></div>
</div>

## 论文信息
- **标题**：VideoLLM-online: Online Video Large Language Model for Streaming Video
- **作者**：Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, Mike Zheng Shou
- **机构**：Show Lab, National University of Singapore；Reality Labs Research, Meta
- **会议/版本**：CVPR 2024；arXiv:2406.11816，arXiv 版本升级到 Llama-3
- **链接**：[arXiv 摘要页](https://arxiv.org/abs/2406.11816) · [论文 PDF](https://arxiv.org/pdf/2406.11816) · [项目主页](https://showlab.github.io/videollm-online/)

<figure class="figure">
  <img src="assets/papers/videollm-online/x1.webp" alt="VideoLLM-online 应用于第一视角视频流的示例" loading="lazy" />
  <figcaption>图 1：VideoLLM-online 面向连续视频流工作，而不是离线短视频问答。模型可以在需要时回答，在画面无变化或无需响应时跳过输出。</figcaption>
</figure>

## 背景：为什么离线 VideoLLM 不够用？
现有 VideoLLM 多数默认输入是一个已经截好的视频片段：用户先选片段，再问问题，模型再生成回答。这种范式适合离线理解，但不适合“always-on”的实时助手。

论文把流式视频助手拆成三个关键要求：

- **时间对齐**：用户可能问“牛排该翻面时提醒我”，模型必须持续扫描新帧，不能等完整视频结束后再回答。
- **长上下文**：总结、计划和追问都依赖历史视觉与语言上下文，但完整保留所有帧会迅速耗尽 LLM 上下文窗口和显存。
- **实时性**：模型必须跟上视频输入速度。如果每一帧都走完整多轮对话和长文本解码，就会产生严重延迟和跳帧。

这也是论文批评 per-frame dialogue 的原因：把每一帧都包装成一次用户输入，看似自然，实际会引入大量重复 prompt、冗余 EOS 或无意义回答，既慢又占上下文。

<figure class="figure">
  <img src="assets/papers/videollm-online/x2.webp" alt="VideoLLM-online 流式视频叙述示例" loading="lazy" />
  <figcaption>图 2：模型在流式视频叙述任务中表现出时间对齐能力：输出跟随当前画面变化，而不是对整段视频做离线总结。</figcaption>
</figure>

## 方法：LIVE 框架的三个核心设计
LIVE（Learning-In-Video-strEam）不是单一模型结构，而是一套面向在线视频流的训练、数据和推理框架。

### 1. Streaming EOS：学习“什么时候该说话”
论文最关键的设计是 **Streaming EOS prediction**。模型不是每帧都生成文本，而是在每个视频帧 token 上判断：

- 如果当前时刻不需要回答，就预测 EOS，并且 **不把这个 EOS 追加到上下文**。
- 如果当前时刻需要回答，就正常进行语言建模，生成 assistant response。

这个设计和普通 next-token prediction 不完全一样。EOS 在这里更像一个“保持沉默/跳过当前帧”的控制信号，而不是对话历史里的真实 token。这样做有两个好处：

- **减少上下文污染**：不把大量“无回答”轮次写入历史，避免上下文窗口被无意义 token 占满。
- **提升实时性**：大多数帧只需要一次轻量判断，不必触发完整语言解码。

训练损失由两部分组成：标准 LM loss 负责学习该回答时怎么回答，Streaming loss 负责学习不该回答时预测 EOS。论文发现默认交叉熵就足够，不需要 Focal Loss 或 OHEM 专门处理 EOS 类别不平衡。

### 2. 把离线标注转换成流式对话数据
真正的流式视频对话数据很少。LIVE 采用两类数据构造方式：

- **天然在线标注**：Ego4D Narration 原本就是标注员边看第一视角视频边写实时 narration，适合转成流式叙述任务。
- **离线标注转在线对话**：对于 COIN 等只有时间段标注的数据，论文用 LLM 把时间轴标注转换成用户-助手对话。

转换流程大致是：先准备关于过去、当前、未来事件的问题模板；再读取视频时间轴标注；把状态变化的关键时间点作为理想响应时刻；最后让 LLM 根据问题和标注生成这些时刻的回答。训练时还会随机插入用户问题，构造更接近真实使用的流式多轮交互。

<figure class="figure">
  <img src="assets/papers/videollm-online/x3.webp" alt="LIVE 将离线视频标注转换为流式对话数据" loading="lazy" />
  <figcaption>图 3：LIVE 的数据生成流程。通过问题模板和时间轴标注，LLM 可以把离线视频数据转换为带时间点的流式对话监督信号。</figcaption>
</figure>

### 3. 连续 KV cache + 编码/解码并行
推理阶段，视频帧持续输入。LIVE 使用连续 KV cache 保存已经处理过的上下文，避免每一帧都重新拼接历史并重复计算。

此外，论文注意到视觉编码器比 LLM 小很多：CLIP / SigLIP 编码一帧比 7B/8B LLM 解码文本快得多。为了避免 LLM 解码阻塞视频帧编码，LIVE 把两者并行：

- 视觉编码器持续编码新帧，并把 frame tokens 放进 FIFO 队列。
- LLM 完成上一轮解码后，从队列中取下一批 frame tokens 继续处理。
- 即使回答生成较慢，也不必让视频编码停下来等待。

<figure class="figure">
  <img src="assets/papers/videollm-online/x5.webp" alt="VideoLLM-online 推理流水线" loading="lazy" />
  <figcaption>图 4：LIVE 推理阶段维护连续 KV cache，并并行化视频帧编码与语言解码，从而减少延迟和跳帧。</figcaption>
</figure>

## 模型结构与训练设置
VideoLLM-online 的模型结构相对朴素，类似 LLaVA：

- **视觉编码器**：CLIP ViT-L 或 SigLIP ViT-L，从视频帧提取视觉 embedding。
- **投影层**：2-layer MLP projector，把视觉 embedding 映射到 LLM token 空间。
- **语言模型**：Llama-2-7B-Chat 或 Llama-3-8B-Instruct。
- **微调方式**：对 LLM 所有线性层使用 LoRA，rank 128，scaling factor 256。

论文中主要比较两个版本：

<div class="table-wrap">
  <table>
    <thead><tr><th>版本</th><th>视觉编码器</th><th>LLM</th><th>每帧 token</th><th>定位</th></tr></thead>
    <tbody>
      <tr><td>VideoLLM-online-7B-v1</td><td>OpenCLIP-ViT-L-224</td><td>Llama-2-7B-Chat</td><td>1 个 CLS token</td><td>更高效，论文主实验默认设置</td></tr>
      <tr><td>VideoLLM-online-8B-v1+</td><td>SigLIP-ViT-L-384</td><td>Llama-3-8B-Instruct</td><td>1 + 3×3 = 10 个 token</td><td>更强视觉细节，demo 使用</td></tr>
    </tbody>
  </table>
</div>

需要注意的是，论文为了效率在主实验中使用每帧 1 个 token，这使模型能在 4096 上下文窗口内处理约半小时视频；但代价是空间细节理解较弱。

<figure class="figure">
  <img src="assets/papers/videollm-online/x4.webp" alt="VideoLLM-online 训练方法" loading="lazy" />
  <figcaption>图 5：训练时把视频帧、用户问题和助手回答按时间顺序组织成输入序列。LM loss 学习回答内容，Streaming loss 学习不该回答时保持沉默。</figcaption>
</figure>

## 数据与评测
论文使用 COIN 和 Ego4D 构建多种评测设置：

- **Ego4D Narration Stream**：模拟人类标注员边看视频边做实时 narration，用于评估在线叙述。
- **COIN + Ego4D Stream**：把 COIN 的步骤标注转换成实时提醒、动作总结和未来动作预测。
- **COIN Benchmarks**：包括 step recognition、task recognition、next forecasting、procedure forecasting 等离线任务。
- **Ego4D LTA**：长期动作预测，要求预测未来 20 个动作的 verb / noun / action。

在线流式任务使用三个指标：

- **LM-PPL / LG-Match**：衡量语言建模质量。
- **TimeDiff**：模型响应时间与理想响应时间的平均差异，越低越好。
- **Fluency**：连续成功预测 token 的比例，综合反映语言质量和时间对齐。

论文也明确指出，这些指标主要适合简单 narration 任务，对复杂自由对话仍不完美。这一点很重要：流式 VideoLLM 的评测标准本身仍未成熟。

## 实验结果
### 流式训练目标显著优于 per-frame dialogue
在 Ego4D Narration Stream 验证集上，Streaming Dialogue 在语言质量、时间对齐和训练成本之间取得最好平衡：

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>训练目标</th><th>LM-PPL ↓</th><th>TimeDiff ↓</th><th>Fluency ↑</th><th>训练 token</th><th>训练成本</th></tr></thead>
    <tbody>
      <tr><td>No Training</td><td>—</td><td>498.5</td><td>6.50</td><td>0.1%</td><td>n/a</td><td>n/a</td></tr>
      <tr><td>Interleaved Dialogue</td><td>Language Modeling</td><td>2.45</td><td>6.47</td><td>11.1%</td><td>1694</td><td>12h</td></tr>
      <tr><td>Per-frame Dialogue</td><td>LM + EOS turns</td><td>3.34</td><td>2.52</td><td>37.7%</td><td>6737</td><td>22h</td></tr>
      <tr><td>Streaming Dialogue</td><td>LM + Streaming EOS</td><td><strong>2.43</strong></td><td><strong>2.32</strong></td><td><strong>42.6%</strong></td><td>1694</td><td>12h</td></tr>
    </tbody>
  </table>
</div>

这里最关键的不是 LM-PPL 的小幅变化，而是 TimeDiff 和 Fluency：LIVE 能更准确地在合适时间回答，同时不牺牲语言建模能力。

### 推理效率优势明显
论文在 5 分钟 Ego4D narration stream 上比较了生成显存和速度：

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>显存 ↓</th><th>FPS ↑</th></tr></thead>
    <tbody>
      <tr><td>Interleaved</td><td>34.4G</td><td>1.5</td></tr>
      <tr><td>Per-frame Streaming</td><td>24.9G</td><td>7.5</td></tr>
      <tr><td>Streaming</td><td><strong>18.2G</strong></td><td><strong>13.5</strong></td></tr>
    </tbody>
  </table>
</div>

这说明 Streaming EOS 的价值不只是“回答更准”，更重要的是减少无意义 token 和 KV cache 膨胀，让在线系统真正有机会跟上视频流。

### 离线视频任务也有竞争力
虽然论文主打在线场景，VideoLLM-online 在传统离线任务上也表现不错。

在 COIN benchmark 上，VideoLLM-online-8B-v1+ 达到：

- **Step recognition**：63.1
- **Task recognition**：92.7
- **Next step forecasting**：49.1
- **Procedure forecasting**：49.8
- **Procedure forecasting with goal**：54.1

在 Ego4D LTA 上，VideoLLM-online-8B-v1+ 的 ED@Z=20 为：

- **Verb**：0.689
- **Noun**：0.671
- **Action**：0.884

它不是所有方法中绝对最优，但作为端到端文本生成模型，已经优于相近的 VideoLLM baseline，并且不依赖复杂级联流程。

### Llama-3 与更多视觉 token 带来提升
不同模型变体在 Ego4D Narration Stream 上的表现如下：

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>LG-Match ↑</th><th>TimeDiff ↓</th><th>Fluency ↑</th></tr></thead>
    <tbody>
      <tr><td>VideoLLM-online-7B-v1</td><td>42.3%</td><td>2.25</td><td>42.6%</td></tr>
      <tr><td>VideoLLM-online-8B-v1</td><td>48.3%</td><td>2.05</td><td>45.2%</td></tr>
      <tr><td>VideoLLM-online-8B-v1+</td><td><strong>49.0%</strong></td><td><strong>2.05</strong></td><td><strong>45.3%</strong></td></tr>
    </tbody>
  </table>
</div>

Llama-3 明显提升语言质量；更多视觉 token 对在线指标提升有限，但有助于 demo 中的视觉细节理解。

<figure class="figure">
  <img src="assets/papers/videollm-online/x6.webp" alt="GPT-4V per-frame prompting 对流式视频的局限" loading="lazy" />
  <figcaption>图 6：附录中对 GPT-4V per-frame prompting 的分析。即使强 prompt 限制输出，逐帧对话仍会消耗大量 token 和时间，并可能出现不稳定回答。</figcaption>
</figure>

## Ablation：哪些设计最关键？
### Streaming EOS 是核心
Interleaved Dialogue 倾向于每帧都输出语言，时间对齐差；Per-frame Dialogue 虽然能学会更多“静默”，但大量 prompt 和 EOS turns 让训练 token 从 1694 增到 6737，训练成本也从 12h 增到 22h。Streaming EOS 在不追加无意义 token 的情况下监督“何时回答”，因此更适合流式场景。

### 标准 CE loss 已经足够
论文比较了 Standard CE、OHEM 和 Focal Loss。结果显示 Standard CE 的 LM-PPL、TimeDiff 和 Fluency 都最好或接近最好，说明这里不需要复杂类别不平衡处理。更重要的是目标设计本身，而不是 loss trick。

### 流式推理工程不可忽略
连续 KV cache 和编码/解码并行使模型能从 1.5 FPS / 34.4G 的 interleaved baseline，提升到 13.5 FPS / 18.2G。对在线助手来说，这种系统优化与模型指标同等重要。

## 我的理解与启发
这篇论文最有价值的点，是把“实时视频助手”从一个离线视频 QA 问题改写成 **在线事件触发式语言建模**：模型不是一直说，而是持续观察、维护上下文、判断时机，然后在关键时刻生成语言。

我觉得它对后续 Streaming VLM / Always-on Assistant 有三个启发：

1. **沉默是一种能力**：实时助手不该把每一帧都变成回答。能判断“不需要说话”，本身就是在线智能的关键能力。
2. **数据格式决定模型行为**：如果训练数据都是离线 clip QA，模型自然不会学会时间对齐和主动响应。LIVE 的离线标注转流式对话，是很务实的数据工程方案。
3. **实时性需要训练和系统一起设计**：只靠更快推理框架不够；训练目标必须减少冗余输出，推理流水线也必须避免视觉编码和语言解码互相阻塞。

和后来的 StreamingVLM 等工作相比，VideoLLM-online 更早强调“何时回答/何时静默”这个问题；它不只是长视频记忆，而是更偏向在线助手交互协议。

## 局限与待观察点
论文自己也提到两个主要限制：

- **高质量流式对话数据不足**：很多训练对话来自现有视频数据集和 LLM 合成，无法覆盖真实用户复杂、多变的需求。小数据集上还可能过拟合。
- **空间理解能力较弱**：主实验为了效率每帧只用 1 个 token，适合长时间流式处理，但会牺牲细粒度视觉定位和空间推理。

此外，我认为还有几个工程问题值得继续观察：

- **评测仍偏简单 narration**：LM-PPL、TimeDiff、Fluency 对真实自由对话、错误提醒、主动帮助是否足够，还需要更贴近产品的 benchmark。
- **KV cache 长期增长问题**：论文使用连续 KV cache 提速，但对超长运行场景下的缓存压缩、遗忘策略和长期记忆没有展开。
- **主动打扰边界**：always-on 助手什么时候应该主动说话，什么时候应该避免打扰用户，除了技术指标还涉及 UX 和安全策略。

## 结论
VideoLLM-online 是流式视频大模型方向的一篇基础工作。它的核心贡献不是模型结构多复杂，而是提出了一个清晰范式：用 **Streaming EOS** 学会何时回答，用 **离线标注转流式对话** 解决数据稀缺，用 **连续 KV cache 与编码/解码并行** 支撑实时推理。对于想构建 AR 眼镜助手、实时陪伴助手或长时间在线视频理解系统的人来说，这篇论文提供了非常实用的起点。