---
id: "streamchat-memory-enhanced-streaming-video-understanding"
title: "StreamChat：用层级记忆实现流式视频理解与多轮交互"
category: "Streaming VLM"
date: "2026-06-04"
order: 10
readTime: "13 min"
tags: ["VLM","Streaming Video","Memory","RAG","Multi-round Interaction","Benchmark"]
summary: "StreamChat 是一个 training-free 的流式视频理解框架，通过长短期视觉记忆、对话记忆和并行系统调度，让 Video-LLM 能在在线场景中实时处理长视频并支持多轮问答。论文同时提出 StreamBench，用 306 个视频、24.8 小时内容和 1.8K QA 评测真实流式交互能力。"
hero: "assets/papers/streamchat/x4.webp"
---

## 一句话总结
StreamChat 的核心贡献是把长视频理解从“离线把若干帧塞进上下文”改造成一个 **持续写入、分层压缩、按需检索、低延迟回答** 的在线系统：视频流被不断写入短期记忆和长程树状记忆，历史问答被写入对话记忆；当用户在任意时刻提问时，系统从视觉记忆和对话记忆里检索相关信息，再交给 Video-LLM 生成回答。论文还提出 StreamBench，专门评测流式、多轮、长短期记忆和复杂推理能力。

<div class="metric-grid">
  <div class="metric"><strong>64.7%</strong><span>StreamChat Slow 在 StreamBench 上的在线准确率</span></div>
  <div class="metric"><strong>32 FPS</strong><span>Fast 版本视频处理速度，显著快于既有 streaming 方法</span></div>
  <div class="metric"><strong>&lt; 0.9s</strong><span>请求到开始生成的响应延迟 RPD</span></div>
  <div class="metric"><strong>1.8K</strong><span>StreamBench 高质量多轮 QA 数量</span></div>
</div>

## 论文信息
- **标题**：Streaming Video Understanding and Multi-round Interaction with Memory-enhanced Knowledge
- **作者**：Haomiao Xiong, Zongxin Yang, Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Jiawen Zhu, Huchuan Lu
- **机构**：Dalian University of Technology, Harvard University
- **会议**：ICLR 2025
- **链接**：[arXiv:2501.13468](https://arxiv.org/abs/2501.13468) · [项目代码](https://github.com/hmxiong/StreamChat)

## 背景：为什么在线长视频对话更难？
传统 Video-LLM 通常把视频当作一个离线片段：先采样若干帧，再和问题一起输入模型。这类方法在短视频单轮问答里可行，但到了真实在线场景会遇到几个问题：

- **视频长度不断增长**：完整保留视觉 token 会带来显存、延迟和上下文长度瓶颈。
- **问题会在视频过程中出现**：用户不一定等视频结束才问，系统必须只依赖当前和过去的信息回答。
- **多轮对话需要记住历史**：当前问题可能依赖前几轮问答，而不仅依赖画面。
- **关键事件可能很短暂**：如果只做均匀采样，短暂出现的对象或动作很容易被错过。

<figure class="figure">
  <img src="assets/papers/streamchat/x1.webp" alt="StreamChat 与既有 Video-LLM 的性能对比" loading="lazy" />
  <figcaption>图 1：StreamChat 在准确率和响应速度上相对既有 Video-LLM 更均衡，目标是面向真实在线流式交互。</figcaption>
</figure>

论文认为，评测也存在问题：很多已有 benchmark 视频短、问题单轮、离线输入，无法反映真实的人机交互。因此作者同时提出方法 StreamChat 和评测集 StreamBench。

## StreamBench：更接近真实在线交互的评测
StreamBench 由 **306 个视频** 构成，总时长 **24.8 小时**，平均每个视频约 **4.5 分钟**，包含 **1.8K 高质量 QA**。视频覆盖四大来源：egocentric videos、web videos、working videos 和 movies，并进一步细分为 16 个子类。

<figure class="figure">
  <img src="assets/papers/streamchat/x3.webp" alt="StreamBench benchmark overview" loading="lazy" />
  <figcaption>图 2：StreamBench 覆盖 4 个视频大类和 16 个子类，并刻意构造多轮交互与不同类型问题。</figcaption>
</figure>

### 六类问题设计
StreamBench 的关键不是“多收集一些视频”，而是把在线交互拆成六类能力：

- **Object Search（OS）**：找短暂出现的对象，目标出现少于 5 秒，且距离提问超过 30 秒。
- **Long-term Memory Search（LM）**：回忆较早发生的事件，事件结束到提问间隔超过 1 分钟。
- **Short-term Memory Search（SM）**：回答最近刚发生的事件，事件结束到提问少于 20 秒。
- **Conversational Interaction（CI）**：当前问题依赖历史对话，要求模型检索前几轮问答。
- **Knowledge-based QA（KG）**：结合视频中出现的对象或事件，回答背景知识问题。
- **Simple Factual（SF）**：视频开始 30 秒内的简单事实问题，用于测试基础短期观察能力。

这个设计很有价值：它把“长视频理解”具体拆成了对象搜索、长期记忆、近期记忆、对话记忆和常识融合，而不是只给一个平均准确率。

## 方法：StreamChat 的三个核心模块
StreamChat 建立在 LongVA 之上，本身是 **training-free** 的系统框架。它主要由三个并行组件构成：Selective Frame Stacking、Memory Formation 和 Contextual Summarization。

<figure class="figure">
  <img src="assets/papers/streamchat/x4.webp" alt="StreamChat framework overview" loading="lazy" />
  <figcaption>图 3：StreamChat 总览。系统把视频处理、记忆更新和用户问答拆成独立线程，从而避免视频越长缓冲区越大的问题。</figcaption>
</figure>

### 1. Selective Frame Stacking：只编码有信息增量的帧
视频流里大量相邻帧是冗余的。StreamChat 使用 Lucas-Kanade Optical Flow 估计相邻帧运动强度，只有当运动向量幅度超过阈值时，才把该帧编码为视觉 embedding 并放入 vision buffer。

这个策略的作用是把“每秒多少帧”转换成“每秒多少有意义的视觉变化”。阈值越高，速度越快，但也越可能丢掉细节；阈值越低，保留更多信息，但延迟和计算会上升。

### 2. Hierarchical Memory Storage：长短期视觉记忆 + 对话记忆
论文最核心的设计是层级记忆系统，包含三部分：

- **Short-term Memory**：保留最近的视觉 embedding，模拟人类短时记忆，用于回答刚发生的事件。
- **Long-term Memory Tree**：把较长历史视频分 chunk，每个 chunk 生成压缩视觉特征和文本线索，再继续按时间聚合成树状结构，用于长期检索。
- **Dialogue Memory**：每轮问答结束后，把问答对编码成向量并写入记忆，用 FAISS 等方式支持后续对话检索。

<figure class="figure">
  <img src="assets/papers/streamchat/x5.webp" alt="StreamChat hierarchical memory storage" loading="lazy" />
  <figcaption>图 4：层级记忆结构。长程记忆把视频历史组织成树，短期记忆保留最近视觉信息，对话记忆保存历史问答。</figcaption>
</figure>

我的理解是，StreamChat 的记忆设计其实是一个多模态 RAG：文本线索负责定位，压缩视觉 token 负责补充视觉证据，对话向量负责恢复交互上下文。它没有把所有历史都塞进 LLM，而是把历史变成可检索状态。

### 3. Contextual Summarization：按问题检索相关上下文再回答
当用户问题到来时，系统会：

1. 在 long memory tree 中计算问题和文本线索的相似度，沿着最相关路径检索视觉记忆。
2. 合并短期视觉记忆，避免忽略刚刚发生的事件。
3. 在 dialogue memory 中检索相关历史问答。
4. 把检索到的视觉与文本上下文一起输入 Video-LLM 生成回答。

这种方式比固定窗口更灵活：问题如果问很久之前的事件，就走长期记忆；如果问刚发生什么，就依赖短期记忆；如果问“刚才你说的那个工具在哪里”，就需要对话记忆。

## 系统调度：为什么能做到实时？
StreamChat 把系统拆成三个线程并行运行：

- **Selective frame stacking thread**：持续筛选并编码有效帧。
- **Memory formation thread**：把视觉 buffer 更新为短期记忆和长期记忆树。
- **Contextual summarization thread**：用户提问时独立检索并生成回答。

这种解耦很关键。如果视频处理和问答在同一条阻塞链路中，视频越长，用户等待越久；而 StreamChat 把视频流持续整理成 memory，问题到来时只需要检索相关状态，因此请求处理延迟可以控制在 **0.85s-0.90s** 左右。

## 实验结果
### StreamBench 在线评测
论文报告了 Slow、Base、Fast 三种配置，分别在准确率和速度之间做取舍。

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>FPS</th><th>Score</th><th>Acc.</th><th>Coh.</th><th>RPD</th></tr></thead>
    <tbody>
      <tr><td>Video-online</td><td>5</td><td>3.11</td><td>56.4</td><td>1.94</td><td>1.07s</td></tr>
      <tr><td>Flash-VStream</td><td>1</td><td>2.89</td><td>52.1</td><td>2.21</td><td>4.15s</td></tr>
      <tr><td>StreamChat Slow</td><td>15</td><td><strong>3.48</strong></td><td><strong>64.7</strong></td><td><strong>1.76</strong></td><td>0.90s</td></tr>
      <tr><td>StreamChat Base</td><td>20</td><td>3.42</td><td>63.8</td><td>1.79</td><td>0.89s</td></tr>
      <tr><td>StreamChat Fast</td><td><strong>32</strong></td><td>3.28</td><td>61.7</td><td>1.81</td><td><strong>0.85s</strong></td></tr>
    </tbody>
  </table>
</div>

关键结论：

- Slow 相比 Video-online 准确率提升 **8.3%**，score 提升 **0.37**。
- Fast 达到 **32 FPS**，仍比 Video-online 准确率高 **5.3%**。
- 三个版本 RPD 都约 **0.9 秒**，说明并行系统调度确实降低了交互延迟。

### 六类任务上的表现
层级记忆对不同任务的收益很符合预期：Slow 相比 Video-online 在 OS、LM、SM、CI 上分别提升 **10.3% / 5.1% / 4.9% / 5.8%**。也就是说，它最明显地改善了需要视觉记忆或对话记忆的任务。

### 离线 benchmark 上也有增益
虽然 StreamChat 面向在线场景，但在 MSRVTT-QA、ActivityNet-QA、NExT-QA、MSVD-QA 等离线 benchmark 上也有效。Base 模型平均准确率达到 **50.6%**，比基础模型 LongVA 的 **48.1%** 提高 **2.5%**。

论文中特别提到：在 ActivityNet 上，加入长程记忆后比 Flash-VStream 高 **12.8%**，也比较强的离线方法 LLaVA-Hound 高 **1.4%**；在 NExT-QA 上，StreamChat 比 LongVA 高 **5.1%**。

<figure class="figure">
  <img src="assets/papers/streamchat/x6.webp" alt="StreamChat inference case study" loading="lazy" />
  <figcaption>图 5：案例分析展示了 StreamChat 如何沿长期记忆树检索相关视觉信息，同时从对话记忆中找回历史问答。</figcaption>
</figure>

## Ablation：哪些设计最关键？
### 1. 三类记忆分别服务不同任务
消融显示，不同记忆模块的收益高度对应任务类型：

- 加入 **dialogue memory**，CI 任务提升约 **4.1%**。
- 加入 **long-term memory**，LM 任务提升约 **6.2%**。
- 加入 **short-term memory**，SM 任务提升约 **3.2%**。
- 长期记忆和短期记忆同时使用时，平均准确率继续提升 **0.9%**。

这说明三类记忆不是简单堆叠，而是在不同时间尺度上互补。

### 2. 速度阈值存在准确率-延迟权衡
光流阈值提高会让系统处理更快，但当速度过高时，模型会丢掉更多原始视觉信息。论文观察到，处理速度从较低水平提升到 32 FPS 后会趋于饱和，而继续追求更快会使性能从约 **64.0%** 下降到 **60.7%**。

### 3. 长程记忆树容量不能无限增大
chunk length、group size 和 clustering goal 都会影响长期记忆树：

- chunk length 从 15 到 30 时准确率从 **61.2%** 提升到 **64.0%**，但到 40 反而略降到 **63.1%**，延迟也从 **0.84s** 增至 **1.26s**。
- group size 从 2 到 12 可提升准确率 **62.0% → 63.9%**，但 RPD 也从 **0.76s** 增至 **1.02s**。
- clustering goal 从 3 到 10 可提升准确率 **59.4% → 64.0%**，但显存从 **20GB** 增至 **56GB**。

<figure class="figure">
  <img src="assets/papers/streamchat/x7.webp" alt="StreamChat memory parameter analysis" loading="lazy" />
  <figcaption>图 6：记忆参数消融。更高的信息保留通常带来更高准确率，但也会增加延迟和显存。</figcaption>
</figure>

## 我的理解与启发
我觉得这篇论文最有价值的地方不是某个单独算法，而是它把在线多模态助手拆成了一个清晰的系统问题：

1. **视频流需要持续整理成状态**：不能等用户提问时再从头处理全量视频。
2. **记忆要分时间尺度**：近期画面、长期事件和历史对话应该用不同结构管理。
3. **检索比全量上下文更工程化**：文本线索 + 视觉 token 的树状检索，是把长视频压缩成可查询知识库的一种方式。
4. **评测必须模拟交互**：StreamBench 的 OS/LM/SM/CI/KG/SF 六类任务，比传统单轮 VQA 更能暴露在线系统的问题。

和 Flash-VStream、StreamingVLM 相比，StreamChat 更强调“多轮交互”和“记忆检索”：它不是只解决无限视频的缓存问题，也不是只做固定窗口压缩，而是让视频历史和对话历史都成为可检索的 memory。

## 局限与待观察点
- **检索算法仍偏基础**：作者也承认当前主要依赖相似度匹配，复杂问题下可能检索错路径。
- **树状视觉记忆有显存压力**：clustering goal 增大能提升准确率，但显存可能从 20GB 增到 56GB，长时运行时需要更强压缩策略。
- **benchmark 仍是构造式交互**：StreamBench 已比传统评测真实很多，但距离开放世界机器人、监控、直播助手仍有差距。
- **training-free 是优点也是限制**：部署成本低，但没有通过端到端训练让模型真正学会使用这种记忆结构。
- **对音频和语音交互覆盖有限**：真实流式场景往往同时有画面、语音、环境音和用户打断，这篇论文主要聚焦视觉和文本问答。

## 结论
StreamChat 给在线长视频助手提供了一条务实路线：用光流筛选降低视觉冗余，用短期记忆覆盖最近事件，用长期记忆树保存过去视频，用对话记忆支持多轮上下文，再通过并行调度把响应延迟压到 0.9 秒以内。它的意义不只是 StreamBench 上的 **64.7%** 准确率和 **32 FPS** 速度，更重要的是展示了一个可部署的流式 Video-LLM 系统应该如何组织记忆、检索和调度。
