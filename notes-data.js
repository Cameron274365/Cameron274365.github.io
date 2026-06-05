    const NOTES = [
      {
        id: "streamchat-memory-enhanced-streaming-video-understanding",
        title: "StreamChat：用层级记忆实现流式视频理解与多轮交互",
        category: "论文阅读",
        date: "2026-06-04",
        readTime: "13 min",
        tags: ["VLM", "Streaming Video", "Memory", "RAG", "Multi-round Interaction", "Benchmark"],
        summary: "StreamChat 是一个 training-free 的流式视频理解框架，通过长短期视觉记忆、对话记忆和并行系统调度，让 Video-LLM 能在在线场景中实时处理长视频并支持多轮问答。论文同时提出 StreamBench，用 306 个视频、24.8 小时内容和 1.8K QA 评测真实流式交互能力。",
        hero: "assets/papers/streamchat/x4.png",
        content: `
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
  <img src="assets/papers/streamchat/x1.png" alt="StreamChat 与既有 Video-LLM 的性能对比" loading="lazy" />
  <figcaption>图 1：StreamChat 在准确率和响应速度上相对既有 Video-LLM 更均衡，目标是面向真实在线流式交互。</figcaption>
</figure>

论文认为，评测也存在问题：很多已有 benchmark 视频短、问题单轮、离线输入，无法反映真实的人机交互。因此作者同时提出方法 StreamChat 和评测集 StreamBench。

## StreamBench：更接近真实在线交互的评测
StreamBench 由 **306 个视频** 构成，总时长 **24.8 小时**，平均每个视频约 **4.5 分钟**，包含 **1.8K 高质量 QA**。视频覆盖四大来源：egocentric videos、web videos、working videos 和 movies，并进一步细分为 16 个子类。

<figure class="figure">
  <img src="assets/papers/streamchat/x3.png" alt="StreamBench benchmark overview" loading="lazy" />
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
  <img src="assets/papers/streamchat/x4.png" alt="StreamChat framework overview" loading="lazy" />
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
  <img src="assets/papers/streamchat/x5.png" alt="StreamChat hierarchical memory storage" loading="lazy" />
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
  <img src="assets/papers/streamchat/x6.png" alt="StreamChat inference case study" loading="lazy" />
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
  <img src="assets/papers/streamchat/x7.png" alt="StreamChat memory parameter analysis" loading="lazy" />
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
        `
      },
      {
        id: "flash-vstream-efficient-real-time-long-video-streams",
        title: "Flash-VStream：面向长视频流的高效实时理解",
        category: "论文阅读",
        date: "2026-06-04",
        readTime: "11 min",
        tags: ["VLM", "Long Video", "Video Stream", "Memory", "Real-Time", "Qwen2-VL"],
        summary: "Flash-VStream 用异步双进程框架和固定大小的 Flash Memory，把长视频流处理与问题回答解耦：CSM 聚合长程时序信息，DAM 按信息密度取回关键帧细节，在 11520 个视频 token 预算内实现一秒级响应，并在 EgoSchema、MLVU、LVBench、MVBench、Video-MME 上取得强结果。",
        hero: "assets/papers/flash-vstream/x1_framework.png",
        content: `
## 一句话总结
Flash-VStream 的核心思路是：不要把长视频当作一个越来越长的短视频输入，而是把视频流持续压缩到一个固定大小、可实时查询的 **Flash Memory**。它用低容量的 **Context Synopsis Memory（CSM）** 建模长程时序与信息密度，再用高容量的 **Detail Augmentation Memory（DAM）** 按 CSM 找到的关键位置补充空间细节，最终让 VLM 在长视频流上兼顾实时响应、长程理解和较高精度。

<div class="metric-grid">
  <div class="metric"><strong>&lt; 1s</strong><span>论文定义的实时响应标准：问题到来后首 token 一秒内生成</span></div>
  <div class="metric"><strong>11520</strong><span>默认 Flash Memory 对应的视频 token 预算</span></div>
  <div class="metric"><strong>68.2</strong><span>EgoSchema 准确率，相比 Qwen2-VL-online 提升 4.2</span></div>
  <div class="metric"><strong>53%</strong><span>相比 Qwen2-VL 官方长上下文设置节省的视频 token</span></div>
</div>

## 论文信息
- **标题**：Flash-VStream: Efficient Real-Time Understanding for Long Video Streams
- **作者**：Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, Xiaojie Jin
- **机构**：Tsinghua University, Tsinghua Shenzhen International Graduate School, Beijing Jiaotong University, ByteDance Inc.
- **会议**：ICCV 2025
- **链接**：[CVF Open Access PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Flash-VStream_Efficient_Real-Time_Understanding_for_Long_Video_Streams_ICCV_2025_paper.pdf) · [项目代码](https://github.com/IVGSZ/Flash-VStream)

## 背景：为什么长视频流理解难？
长视频理解的主要矛盾不是“模型能不能看视频”，而是 **能不能在视频持续增长时仍然低延迟、低显存、可交互**。

- **离线长视频 VLM** 通常把更多帧或更多 token 一次性塞进模型，准确率可能不错，但问题到来时需要处理大量上下文，很难做到实时响应。
- **在线视频 VLM** 能边看边处理，但如果只保留局部窗口，就容易丢失早期关键线索；如果保留太多历史，又会拖慢推理。
- **真实应用** 如多模态助手、机器人、安防监控，需要模型持续接收视频流，并在用户随时提问时快速回答。

论文把实时 VLM 定义为：当用户提问时，模型能在 **1 秒内生成首个回答 token**。这比普通离线 VQA 更接近产品交互场景。

<figure class="figure">
  <img src="assets/papers/flash-vstream/x1_framework.png" alt="Flash-VStream 双进程框架和 Flash Memory 总览" loading="lazy" />
  <figcaption>图 1：Flash-VStream 的双进程框架。Frame Handler 持续编码新帧并更新 Flash Memory；Question Handler 在问题到来时异步读取当前记忆并快速生成答案。</figcaption>
</figure>

## 方法：核心设计
### 1. 双进程架构：把看视频和答问题解耦
Flash-VStream 由两个并行过程组成：

- **Frame Handler**：持续读取视频流，对新帧做视觉编码，并更新共享的 Flash Memory。
- **Question Handler**：像服务端一样等待用户问题；一旦问题到来，直接基于当前 Flash Memory 调用 LLM 生成回答。

这个设计的关键价值是把视觉编码成本从“提问时”提前摊到“视频流处理过程中”。用户提问时不需要重新扫完整视频，因此延迟更稳定。

### 2. Flash Memory：固定大小的长视频状态
Flash Memory 不是简单的均匀采样，也不是无限增长的 KV cache，而是由两类互补记忆交错组成：

- **Context Synopsis Memory（CSM）**：用低分辨率特征做 K-means 聚类，聚合语义相近的历史帧。每个 CSM 项是一个簇中心，代表一段上下文的“摘要”。簇大小也隐式表达了视频中的信息密度分布。
- **Detail Augmentation Memory（DAM）**：CSM 能保留长期时序，但会损失空间细节。DAM 根据 CSM 中最大的若干簇，去特征库中检索最接近簇中心的高分辨率关键帧特征，补充细粒度视觉信息。

我的理解是：CSM 负责“知道视频长期发生过什么”，DAM 负责“在关键位置看清楚细节”。二者组合，比单纯压缩或单纯采样更适合长视频 VQA。

<figure class="figure">
  <img src="assets/papers/flash-vstream/x3_memory_config.png" alt="Flash Memory 默认配置" loading="lazy" />
  <figcaption>图 2：Flash Memory 默认配置。CSM 使用 120 个输入帧、60 个低分辨率特征；DAM 使用 60 个输入帧、30 个高分辨率特征，最终共 11520 个 LLM 视频 token。</figcaption>
</figure>

### 3. CSM：用聚类表达长程信息密度
CSM 的更新方式是：当新帧进入后，把新帧的低分辨率特征加入已有 CSM，再用 K-means 保持固定数量的簇中心。这样，即使视频越来越长，CSM 的大小也不会增长。

更重要的是，簇本身还提供了“哪些片段更典型/更密集”的线索。DAM 正是利用这个分布来选关键帧，而不是盲目均匀采样。

### 4. DAM：从关键簇取回高分辨率细节
DAM 会维护高分辨率 Feature Bank，并按 CSM 中簇大小排序，选择 top-k 最大簇对应的关键帧。论文默认使用 **Feature-Centric retrieval**：找到距离簇中心最近的帧作为代表帧，再取其高分辨率特征。

这解决了 CSM 的一个天然问题：低分辨率聚合适合压缩长程信息，但对于 OCR、物体细节、场景布局等问题不够精确；DAM 则把高分辨率预算集中花在最有代表性的关键帧上。

### 5. AM-RoPE：给压缩记忆分配合理位置
Flash Memory 里既有 CSM 的“簇中心”，也有 DAM 的“关键帧”。它们不是原始连续帧，因此普通 M-RoPE 的位置编号并不完全适用。

论文提出 **Adaptive Multimodal RoPE（AM-RoPE）**：

- DAM token 使用对应关键帧的时间位置。
- CSM token 使用簇内帧位置的平均值。
- 空间位置根据 DAM/CSM 的不同分辨率做相应映射。

这样模型看到的不是无序压缩特征，而是带有相对时间与空间位置的结构化记忆。

## 数据与评测
### 训练设置
Flash-VStream 基于 **Qwen2-VL-7B** 初始化，视觉编码器冻结，对 projector 和 LLM 的线性层做 LoRA instruction tuning。训练数据采用 LLaVA-Video 的 9K 子集，包含 captioning、open-ended VQA 和 multiple-choice VQA。

### 评测设置
论文在在线设置下评测：视频以 **1 fps** 输入，逐帧更新 Flash Memory，问题在视频帧流结束时提出。主要报告多选 VQA 准确率。

覆盖的 benchmark 包括：

- **EgoSchema**：第一人称长视频行为理解。
- **MLVU**：多任务长视频理解。
- **LVBench**：极长视频理解。
- **MVBench**：强调时序理解能力。
- **Video-MME**：综合视频理解，视频长度从 11 秒到 1 小时。

## 实验结果
### 1. 主结果：同等实时 token 预算下全面优于 Qwen2-VL-online
论文设置了一个公平的在线基线 **Qwen2-VL-online**：把 Qwen2-VL 的最大视频 token 限制到 11520，与 Flash-VStream 的 Flash Memory 成本相同。

在这个同成本条件下，Flash-VStream 在所有报告 benchmark 上都优于 Qwen2-VL-online：

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>Max NVtokens</th><th>EgoSchema</th><th>MLVUdev</th><th>LVBench</th><th>MVBench</th><th>Video-MME w/o</th><th>Video-MME w/</th></tr></thead>
    <tbody>
      <tr><td>Qwen2-VL-online</td><td>11520</td><td>64.0</td><td>62.9</td><td>39.8</td><td>63.3</td><td>59.4</td><td>65.1</td></tr>
      <tr><td>Flash-VStream</td><td>11520</td><td><strong>68.2</strong></td><td><strong>66.3</strong></td><td><strong>42.0</strong></td><td><strong>65.4</strong></td><td><strong>61.2</strong></td><td><strong>67.0</strong></td></tr>
    </tbody>
  </table>
</div>

<figure class="figure">
  <img src="assets/papers/flash-vstream/x5_main_results.png" alt="Flash-VStream 在多个视频理解 benchmark 上的主结果" loading="lazy" />
  <figcaption>图 3：主结果表。Flash-VStream 在 11520 视频 token 成本下达到 EgoSchema 68.2、MLVUdev 66.3、LVBench 42.0、MVBench 65.4、Video-MME 61.2/67.0。</figcaption>
</figure>

### 2. 效率：一秒实时阈值下保持更好准确率
论文的速度测试显示，7B 模型要满足一秒首 token 响应，视频 token 数应控制在约 **12000** 以内。许多长视频模型依赖更大的 token 预算，例如 LongVILA、LongVA、Qwen2-VL 官方设置等，因此延迟超过一秒。

Flash-VStream 的优势不是单纯减少 token，而是通过 CSM + DAM 让固定 token 预算更有信息量：同样 11520 个 token，它比均匀截断/在线版本更准确。

<figure class="figure">
  <img src="assets/papers/flash-vstream/x2_latency.png" alt="Flash-VStream 响应延迟与准确率对比" loading="lazy" />
  <figcaption>图 4：响应延迟、准确率和推理成本关系。超过约 12000 个视频 token 后难以满足实时要求；Flash-VStream 在实时区间取得更好的准确率。</figcaption>
</figure>

### 3. Video-MME：长视频段尤其受益
在 Video-MME 上，Flash-VStream 相比 Qwen2-VL-online：

- 无字幕平均分从 59.4 提升到 **61.2**。
- 有字幕平均分从 65.1 提升到 **67.0**。
- 有字幕 Long 子集从 57.9 提升到 **61.4**。

这说明 Flash Memory 对长视频段更有帮助：它不是只保留最近窗口，而是在固定容量内保留跨时间的代表性信息和关键细节。

## Ablation：哪些设计最关键？
### 1. CSM 和 DAM 都有贡献，CSM 更关键
默认 Flash Memory（CSM + DAM）平均准确率为 **64.9**。移除或替换组件后会下降：

<div class="table-wrap">
  <table>
    <thead><tr><th>设置</th><th>NVtokens</th><th>MVBench</th><th>EgoSchema</th><th>Video-MME w/o</th><th>Average</th></tr></thead>
    <tbody>
      <tr><td>CSM + DAM</td><td>11520</td><td><strong>65.4</strong></td><td><strong>68.2</strong></td><td><strong>61.2</strong></td><td><strong>64.9</strong></td></tr>
      <tr><td>CSM + Uniform DAM</td><td>11520</td><td>64.3</td><td>67.8</td><td>60.6</td><td>64.2</td></tr>
      <tr><td>CSM only</td><td>3840</td><td>64.0</td><td>66.8</td><td>60.1</td><td>63.6</td></tr>
      <tr><td>Uniform CSM only</td><td>3840</td><td>62.4</td><td>63.4</td><td>59.0</td><td>61.6</td></tr>
    </tbody>
  </table>
</div>

论文结论是：CSM 相比均匀采样带来约 **2.0%** 平均提升，DAM 带来约 **0.7%** 平均提升。我的理解是，长视频任务首先需要“别丢掉长期结构”，因此 CSM 的贡献更大；DAM 则进一步补足细节。

<figure class="figure">
  <img src="assets/papers/flash-vstream/x6_ablation.png" alt="Flash Memory 组件消融实验" loading="lazy" />
  <figcaption>图 5：Flash Memory 组件消融。CSM 和 DAM 的组合最优；只用均匀采样或移除关键组件都会降低平均准确率。</figcaption>
</figure>

### 2. 记忆容量分配：约三分之一给 CSM 最合适
在固定总 token 预算下，论文网格搜索 CSM/DAM 容量比例和空间池化比例。结果显示：当 pool ratio = 4，且约 **1/3 容量给 CSM、2/3 容量给 DAM** 时，EgoSchema 与 MVBench 表现最好。

这很符合直觉：CSM 太少会丢长期时序，DAM 太少会缺空间细节；二者不是替代关系，而是互补关系。

<figure class="figure">
  <img src="assets/papers/flash-vstream/x4_capacity.png" alt="CSM 容量比例与空间池化比例消融" loading="lazy" />
  <figcaption>图 6：容量分配消融。固定实时 token 预算下，约 1/3 CSM + 2/3 DAM、pool ratio = 4 是较优配置。</figcaption>
</figure>

### 3. K-means 和 Feature-Centric retrieval 是有效选择
CSM 的聚类策略中，K-means 在 EgoSchema 和 MVBench 上优于 DBScan、GMM、Neighbor Merge、Neighbor Drop 和均匀采样。DAM 的检索策略中，Feature-Centric + top-k largest clusters 的组合最稳。

这说明论文不是简单提出“有个 memory 就行”，而是验证了 memory 的更新策略、检索策略和容量分配都会影响最终效果。

## 我的理解与启发
这篇论文和 StreamingVLM 都在解决长视频流问题，但侧重点不同：StreamingVLM 更强调训练-推理一致的流式 KV cache；Flash-VStream 更强调 **在线压缩视频状态**，把视觉流持续写入一个可查询的固定大小记忆。

我觉得它最有启发的地方有三点：

1. **把实时性作为一等约束**：论文明确用一秒首 token 作为工程标准，而不是只报离线准确率。
2. **记忆不是越大越好，而是要有信息密度建模**：CSM 的簇大小让模型知道哪些片段更有代表性，再用 DAM 把细节预算投向这些片段。
3. **异步系统设计很重要**：在真实产品里，视频编码、状态更新和用户问答本来就是不同节奏的任务。双进程框架比“问题来了再处理全量视频”更接近可部署形态。

<figure class="figure">
  <img src="assets/papers/flash-vstream/x7_case.png" alt="Flash Memory 可视化与问答案例" loading="lazy" />
  <figcaption>图 7：Flash Memory 可视化与案例。左侧 PCA 显示 CSM/DAM 对特征空间进行了压缩与整合；右侧案例展示了模型在 OCR、空间理解和复杂推理问题上的表现。</figcaption>
</figure>

## 局限与待观察点
- **训练数据规模较小**：只使用 LLaVA-Video 9K 子集做 LoRA，说明方法有效，但也留下了更大规模训练下是否继续提升的问题。
- **问题触发位置偏离真实连续交互**：评测通常在视频结束后提问，而真实场景中用户可能在任意时刻插入多轮问题。
- **Feature Bank 可能转移成本**：论文提到高分辨率特征库可 offload 到磁盘避免内存溢出，但磁盘/IO 成本在实际部署中需要进一步量化。
- **固定记忆可能漏掉罕见关键事件**：K-means 与簇大小偏向代表性片段，极少出现但对问题关键的事件是否会被稳定保留，仍值得观察。
- **多模态覆盖有限**：论文主要处理视觉流和文本问题，音频、ASR、实时语音交互等信息没有成为核心设计的一部分。

## 结论
Flash-VStream 给长视频流理解提供了一条很务实的路线：用双进程架构降低在线交互延迟，用 CSM 建模长程时序和信息密度，用 DAM 补充关键帧空间细节，再用 AM-RoPE 让压缩记忆保留合理的位置结构。它的价值不只是 benchmark 提升，而是把长视频 VLM 从“离线大上下文推理”推进到“持续运行、固定预算、随时可问”的实时系统形态。
        `
      },
      {
        id: "streamingvlm-real-time-infinite-video",
        title: "StreamingVLM：面向无限视频流的实时理解",
        category: "论文阅读",
        date: "2026-06-03",
        readTime: "12 min",
        tags: ["VLM", "Streaming", "KV Cache", "RoPE", "Video Understanding"],
        summary: "StreamingVLM 通过训练-推理一致的流式框架，让 VLM 在近无限视频上保持低延迟、低显存和稳定理解能力。论文构建了 Inf-Streams-Train / Eval，并在长达 2 小时以上的视频流评测中取得强结果。",
        hero: "assets/papers/streamingvlm/x1.png",
        content: `
## 一句话总结
StreamingVLM 的核心不是单点技巧，而是把 **训练方式、推理缓存结构、位置编码和数据集** 对齐成一套流式系统：训练时用短且重叠的视频块做全注意力 SFT；推理时保留 attention sink、近期文本窗口和近期视觉窗口，并复用 KV cache；再用 contiguous RoPE 避免位置编号无限增长。

<div class="metric-grid">
  <div class="metric"><strong>66.18%</strong><span>Inf-Streams-Eval 上对 GPT-4o mini 的胜率</span></div>
  <div class="metric"><strong>8 FPS</strong><span>单张 NVIDIA H100 上保持实时流式理解</span></div>
  <div class="metric"><strong>2.12h</strong><span>Inf-Streams-Eval 视频平均长度</span></div>
  <div class="metric"><strong>+5.96</strong><span>OVOBench Realtime 相对基座模型提升</span></div>
</div>

## 论文信息
- **标题**：StreamingVLM: Real-Time Understanding for Infinite Video Streams
- **作者**：Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Yao Lu, Song Han
- **机构**：MIT, NVIDIA
- **会议**：ICLR 2026
- **链接**：[arXiv:2510.09608](https://arxiv.org/abs/2510.09608) · [项目代码](https://github.com/mit-han-lab/streaming-vlm)

## 背景：为什么长视频流很难？
现有 VLM 面对实时视频流时通常遇到三类问题：

- **Full Attention 成本失控**：视频越长，注意力开销近似按 O(T²) 增长，显存和延迟都不可接受，并且超过训练长度后性能下降。
- **无重叠 Sliding Window 破坏连贯性**：每个窗口像独立片段，历史语义容易断裂；窗口过长又会造成延迟。
- **有重叠 Sliding Window 重算严重**：能保留部分上下文，但大量重复计算导致实时性差。

<figure class="figure">
  <img src="assets/papers/streamingvlm/x1.png" alt="StreamingVLM 与 full attention、sliding window 方法对比" loading="lazy" />
  <figcaption>图 1：StreamingVLM 与既有方案对比。它通过 Sliding Window + Reuse KV 复用 attention sink、短视觉窗口和长文本窗口，从而兼顾历史、低延迟与稳定性。</figcaption>
</figure>

## 方法：训练-推理一致的流式框架
### 1. Streaming-aware KV Cache
推理时，模型维护一个紧凑 KV cache，主要保留三部分：

- **Attention sink tokens**：系统提示和早期关键文本 token，用于稳定注意力。
- **Long text window**：较长的近期文本窗口，保留对话和解说历史。
- **Short vision window**：较短的近期视觉 token 窗口，跟踪正在发生的动作。

这种非对称策略很符合视频流特点：视觉画面变化快，旧视觉 token 可以先淘汰；文本承载长期语义、人物、比分和上下文，需要保留更久。

<figure class="figure">
  <img src="assets/papers/streamingvlm/x3.png" alt="StreamingVLM 推理结构" loading="lazy" />
  <figcaption>图 3：推理时保留 512 个 attention-sink token、512 个近期文本 token，以及覆盖 16 秒的近期视觉 token。</figcaption>
</figure>

### 2. Contiguous RoPE
如果简单丢弃旧 token，剩余 token 的 RoPE 位置编号会持续增长，最后落到训练分布之外。StreamingVLM 在淘汰 token 后将位置编号左移，让保留 token 和新 token 的位置保持连续且有界。

对 Qwen-VL 这类使用 3D 位置编码的模型，论文也把 contiguous RoPE 扩展到视觉 token 的时间、高度、宽度维度。

### 3. Overlapped-chunk Full-attention SFT
训练阶段不直接训练超长视频，而是把长视频切成长度 W 的短 chunk，相邻 chunk 有 O 帧重叠，并在每个 chunk 内使用 full attention。

这个设计很关键：虽然训练时看的是短片段，但重叠区域让训练样本近似模拟推理时的 “sink + 近期文本 + 近期视觉” 注意力模式，从而实现 **训练-推理一致**。

<figure class="figure">
  <img src="assets/papers/streamingvlm/x4.png" alt="StreamingVLM 训练策略" loading="lazy" />
  <figcaption>图 4：训练时使用重叠 chunk 内的全注意力，模拟推理时保留 attention sink 和近期窗口的有效注意力模式。</figcaption>
</figure>

## 数据：Inf-Streams-Train 与 Inf-Streams-Eval
论文围绕体育解说构建数据，因为体育赛事天然具备长视频、实时动作、强时序和密集解说的特点。

### 数据清洗流程
- 收集篮球、足球、美式橄榄球、冰球、棒球五类比赛视频。
- 使用 WhisperX 获取带时间戳 ASR。
- 将视频切成 120 秒片段，用 GPT-5 判断每句解说应该 keep、edit 还是 delete。
- 原始 ASR 中约 46.32% 保留，37.89% 编辑，15.79% 删除。

<figure class="figure">
  <img src="assets/papers/streamingvlm/x5.png" alt="数据构建流程" loading="lazy" />
  <figcaption>图 5：数据管线从五类体育比赛中采集视频和 ASR，经 GPT 清洗后构建 SFT 数据、评测集和高质量 annealing 数据。</figcaption>
</figure>

### 训练与评测数据
- **SFT 数据**：以 W=24s、O=12s 切分，构成 525K streaming samples。
- **高质量 annealing 数据**：保留实时场上动作占比超过 80% 的片段，最终 14,786 条。
- **Inf-Streams-Eval**：20 场完整比赛，平均 2.12 小时，按 100 秒片段评估，使用 GPT-5 基于参考解说进行 pairwise vote。

## 实验结果
### Captioning：长视频解说显著领先
StreamingVLM 在 Inf-Streams-Eval 的 infinite 模式下，对 GPT-4o mini chunk 模式达到 **66.18% win rate**，对 LiveCC chunk 模式达到 **87.81% win rate**，对 LiveCC infinite 模式达到 **99.12% win rate**。

<figure class="figure">
  <img src="assets/papers/streamingvlm/x6.png" alt="既有 VLM 在跨 chunk 连贯性与训练长度之间的权衡问题" loading="lazy" />
  <figcaption>图 6：既有 VLM 很难同时兼顾跨 chunk 连贯性和训练长度限制；短 chunk 容易失去上下文，长 chunk 又会逼近甚至超过训练长度。</figcaption>
</figure>

<div class="table-wrap">
  <table>
    <thead><tr><th>模型 A</th><th>对 GPT-4o mini</th><th>对 LiveCC chunk</th><th>对 LiveCC infinite</th><th>说明</th></tr></thead>
    <tbody>
      <tr><td>Qwen2.5-VL-7B-Instruct</td><td>0.01</td><td>20.44</td><td>95.97</td><td>基座模型缺少连续解说能力</td></tr>
      <tr><td>LiveCC-7B-Instruct chunk</td><td>15.73</td><td>—</td><td>—</td><td>短 chunk 尚可，但长程连贯性不足</td></tr>
      <tr><td>StreamingVLM infinite</td><td><strong>66.18</strong></td><td><strong>87.81</strong></td><td><strong>99.12</strong></td><td>保持长程历史和实时视觉感知</td></tr>
    </tbody>
  </table>
</div>

<figure class="figure">
  <img src="assets/papers/streamingvlm/x2.png" alt="既有 VLM 的问题与 StreamingVLM 的改进" loading="lazy" />
  <figcaption>图 2：没有 SFT 时模型难以跨轮次连贯生成；full attention 很快超过训练长度并带来高延迟；普通 sliding window 难以保留足够上下文。</figcaption>
</figure>

### VQA：没有专门 VQA 微调也有提升
StreamingVLM 并未针对 VQA 再做特殊微调，但在多个 VQA benchmark 上仍优于基座模型，说明流式 SFT 和数据本身改善了通用视频理解能力。

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>MVBench</th><th>VideoMME</th><th>LongVideoBench</th><th>OVOBench Realtime</th></tr></thead>
    <tbody>
      <tr><td>Qwen2.5-VL-7B-Instruct</td><td>67.34</td><td>65.10</td><td>54.70</td><td>56.00</td></tr>
      <tr><td>StreamingVLM</td><td><strong>69.16</strong></td><td>65.10</td><td><strong>59.00</strong></td><td><strong>61.96</strong></td></tr>
    </tbody>
  </table>
</div>

### Efficiency：延迟稳定，支持实时
Full attention 会随视频变长不断变慢甚至 OOM；普通 sliding window 会周期性重建上下文或重复计算。StreamingVLM 由于固定缓存长度并复用 KV，per-token latency 基本稳定，可在单张 H100 上以 8 FPS 支持实时解说。

<figure class="figure">
  <img src="assets/papers/streamingvlm/x7.png" alt="不同推理方式的延迟对比" loading="lazy" />
  <figcaption>图 7：StreamingVLM 的 per-token latency 随视频长度保持稳定；虚线表示实时阈值。</figcaption>
</figure>

<figure class="figure">
  <img src="assets/papers/streamingvlm/x8.png" alt="StreamingVLM 随时间推移的稳定性" loading="lazy" />
  <figcaption>图 8：附录稳定性实验。测试视频按 20% 间隔分成五段，StreamingVLM 在后续片段中没有明显退化，说明复用 KV 的流式方案能够维持长时稳定性。</figcaption>
</figure>

## Ablation：哪些设计最关键？
### Contiguous RoPE
Native RoPE 在无限流上性能明显下降，因为位置编号不断外推；contiguous RoPE 让位置保持在训练分布内，Inf-Streams-Eval 对 GPT-4o mini 的胜率从 25.09 提升到 66.18。

### 视觉窗口与文本窗口
- 0 秒视觉窗口明显下降，说明近期视觉上下文对连续动作非常重要。
- 16 秒视觉窗口是较优点：足够覆盖近期动作，同时不会显著增加计算。
- 文本 token 的 sink 和 window 保留策略对长程连贯性很重要。

### 数据和训练策略
仅使用 Live-WhisperX-526K 的效果有限；加入 Inf-Streams-Train 后，长视频流式能力大幅提升；再加入高质量 annealing 数据后，解说质量和 VQA 表现继续增强。非重叠训练策略也弱于 overlapped strategy，证明 **数据本身不够，训练方式也必须匹配推理形态**。

## 我的理解与启发
这篇论文最值得借鉴的是系统设计视角：它没有把 “无限视频理解” 简化成单个长上下文技巧，而是同时处理了四个约束：

1. **计算约束**：通过 KV 复用和固定窗口避免重复计算。
2. **记忆约束**：文本和视觉使用不同保留策略，符合多模态信息的时间尺度差异。
3. **分布约束**：contiguous RoPE 避免位置编码超出训练分布。
4. **监督约束**：重叠 chunk 训练让短上下文 SFT 学到流式推理模式。

对实际产品也有启发：如果要做实时助手、机器人或自动驾驶场景的多模态 Agent，不能只追求模型输入更长；更重要的是设计稳定、低延迟、可持续运行的在线状态表示。

## 局限与待观察点
- 主要数据来自体育解说，迁移到开放世界监控、机器人或车载场景仍需验证。
- GPT-5 作为评审器带来可扩展性，但也可能引入 judge 偏差。
- 论文强调 8 FPS 与单 H100，端侧或低成本部署还需要进一步压缩。
- 模型长期输出中的事实一致性、安全性和可控性仍是产品化挑战。

## 结论
StreamingVLM 给出了一个务实路线：用 **attention sink + 文本长窗口 + 视觉短窗口 + KV 复用 + contiguous RoPE + overlapped SFT**，把已有 VLM 改造成可实时处理近无限视频流的系统。它的贡献不仅是性能指标，更是把训练、推理和数据闭环对齐的工程范式。
        `
      },
      {
        id: "how-to-maintain-this-blog",
        title: "如何维护这个静态笔记博客",
        category: "站点管理",
        date: "2026-06-03",
        readTime: "4 min",
        tags: ["Static Site", "Notes", "Workflow"],
        summary: "这个站点不依赖构建工具、不需要后端。新增文章时，在 index.html 的 NOTES 数组里追加对象即可，页面会自动生成列表、分类、搜索和文章目录。",
        content: `
## 结构说明
这个博客是一个单文件静态站点，适合 GitHub Pages：

- **数据层**：页面底部的 \`NOTES\` 数组。
- **交互层**：搜索、分类筛选、文章切换、目录生成和主题切换都由原生 JavaScript 完成。
- **资源层**：图片放在 \`assets/\` 目录，文章里使用相对路径引用。

## 新增笔记步骤
1. 在 \`NOTES\` 数组顶部或底部追加一条对象。
2. 填写 \`id\`、\`title\`、\`category\`、\`date\`、\`readTime\`、\`tags\`、\`summary\`、\`content\`。
3. 如果有图片，放到 \`assets/你的主题/\` 目录，并在正文中用 \`<figure class="figure">...\</figure>\` 引用。
4. 提交并推送到 GitHub，GitHub Pages 会自动更新。

## 内容建议
- 标题尽量清晰，便于搜索。
- 每篇文章保留“一句话总结”和“我的理解”。
- 论文笔记建议记录：问题背景、核心方法、实验结论、局限、个人启发。
- 对长期知识库来说，标签比时间更重要，建议保持标签稳定。
        `
      }
    ];
