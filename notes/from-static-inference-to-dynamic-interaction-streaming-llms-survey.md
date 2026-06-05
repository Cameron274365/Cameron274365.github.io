---
id: "from-static-inference-to-dynamic-interaction-streaming-llms-survey"
title: "Streaming LLM 综述：从静态推理到动态交互"
category: "论文阅读"
date: "2026-06-05"
order: 10
readTime: "18 min"
tags: ["LLM","Streaming","Survey","Real-Time Interaction","KV Cache","Multimodal"]
summary: "这篇 ACL 2026 Findings 综述把 Streaming LLM 从一个容易混淆的术语拆成三层：output-streaming、sequential-streaming 和 concurrent-streaming，并从生成机制、增量编码、上下文管理、架构适配、交互策略和应用任务几个维度梳理了实时智能的发展路线。"
hero: "assets/papers/streaming-llm-survey/top.png"
---

## 一句话总结
这篇综述的核心价值不是提出一个新模型，而是给 “Streaming LLM” 这个正在快速扩张但概念混乱的方向建立了一套清晰坐标系：**只边生成不等于真正 streaming；能边读输入也只是中间阶段；真正困难的是模型在输入持续到来时还能同时决定何时读、何时写、何时等待、何时结束。** 论文据此把 Streaming LLM 分为 output-streaming、sequential-streaming 和 concurrent-streaming 三层，并把相关技术整理成从低延迟生成、动态输入处理到全双工实时交互的研究路线图。

<div class="metric-grid">
  <div class="metric"><strong>3 类范式</strong><span>Output / Sequential / Concurrent Streaming LLM</span></div>
  <div class="metric"><strong>2 条主线</strong><span>技术层：如何构建；应用层：如何落地</span></div>
  <div class="metric"><strong>4 类架构适配</strong><span>Re-encoded、Concatenated、Interleaved、Grouped streaming</span></div>
  <div class="metric"><strong>ACL 2026</strong><span>Findings，聚焦 Streaming LLM 的系统综述</span></div>
</div>

## 论文信息
- **标题**：From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models
- **作者**：Junlong Tong, Zilong Wang, YuJie Ren, Peiran Yin, Hao Wu, Wei Zhang, Xiaoyu Shen
- **机构**：Shanghai Jiao Tong University；Institute of Digital Twin, Eastern Institute of Technology, Ningbo
- **会议**：ACL 2026 Findings
- **链接**：[arXiv:2603.04592](https://arxiv.org/abs/2603.04592) · [论文 PDF](https://arxiv.org/pdf/2603.04592) · [Awesome-Streaming-LLMs](https://github.com/EIT-NLP/Awesome-Streaming-LLMs)

## 为什么需要这篇综述？
Streaming LLM 这个词在不同论文里经常指向不同东西。有人说的是 **token-by-token 输出**，有人说的是 **增量输入处理**，也有人说的是 GPT-4o 式的 **实时双向交互**。这些方向都和 streaming 有关，但技术难点完全不同。

论文认为，标准 LLM 的默认范式是 “read-at-once”：先拿到完整输入，prefill 形成上下文，再 autoregressive decode。这个范式适合静态 benchmark，但不适合真实世界里持续变化的信号：

- **语音** 是连续到来的，系统不能等一句话完全结束才理解。
- **视频** 是长时间甚至无限流，模型不能无限保存所有帧。
- **机器人或数字人** 需要一边感知、一边思考、一边说话或行动。
- **工具型 Agent** 可能需要在输入未完全结束时提前检索、调用工具、生成中间状态。

因此，Streaming LLM 的问题不只是 “输出快一点”，而是要解决 **动态数据流里的实时决策**：模型在每个时刻能看到多少输入？要不要继续等？是否可以开始输出？输出后新输入又来了怎么办？

<figure class="figure">
  <img src="assets/papers/streaming-llm-survey/top.png" alt="Streaming LLM 三类范式" loading="lazy" />
  <figcaption>图 1：论文提出的三类 Streaming LLM。左侧 output-streaming 是静态读入后流式输出；中间 sequential-streaming 是流式读入后再生成；右侧 concurrent-streaming 是输入和输出同时展开。</figcaption>
</figure>

## 统一定义：用数据流和交互并发性定义 Streaming LLM
论文把模型过程抽象为条件分布 `P(Y|X)`，其中输入流为 `X = (x1, ..., xM)`，输出流为 `Y = (y1, ..., yN)`。关键是引入一个决策函数 `phi(t)`：在生成第 `t` 个输出 token 时，模型到底能看到输入流的哪一部分。

这个定义很有用，因为它把 “streaming” 从直觉描述变成了一个关于 **可见输入前缀** 和 **读写时序** 的问题。

### 1. Output-streaming LLM：完整读入之后逐步输出
这类模型在生成开始前已经看到了完整输入，因此 `phi(t) = M`。它们的 streaming 只体现在输出端，也就是常见的 token-by-token 展示。

典型例子包括大多数聊天 LLM：用户输入完整 prompt 后，模型逐 token 返回答案。它能改善用户感知延迟，但本质上仍是静态输入推理。

### 2. Sequential-streaming LLM：输入增量到来，但生成仍发生在读完之后
这类模型能够逐步处理输入流，避免每次新数据到来都重算完整上下文。它适合长视频、长音频、持续日志等输入越来越长的场景。

但它通常仍是 “先读再写”：系统在输入流被处理到某个阶段后，基于累积上下文回答问题。比如长视频一直被编码进记忆，用户提问时模型读取当前记忆生成答案。

### 3. Concurrent-streaming LLM：边读边写的动态交互
这是最严格也最接近实时智能的形式。输入和输出同时展开，模型生成第 `t` 个输出时只能看到部分输入，并且新的输入可能在输出过程中继续到来。

这类系统需要解决：

- **什么时候继续读输入？**
- **什么时候开始写输出？**
- **已经生成的输出是否能被后续输入修正或打断？**
- **输入 token 和输出 token 如何共享或隔离注意力与位置编码？**

我的理解是，concurrent-streaming 才是从 “模型推理服务” 走向 “实时交互主体” 的关键一步。

## 总体分类框架
论文的 taxonomy 可以理解成一条能力升级路径：

<div class="table-wrap">
  <table>
    <thead><tr><th>范式</th><th>输入处理</th><th>输出生成</th><th>核心问题</th><th>典型技术</th></tr></thead>
    <tbody>
      <tr><td>Output-streaming</td><td>静态完整输入</td><td>流式输出</td><td>如何更快生成、更省显存</td><td>token-wise / block-wise / refinement generation，speculative decoding，KV 压缩</td></tr>
      <tr><td>Sequential-streaming</td><td>流式输入，增量编码</td><td>读完或查询时生成</td><td>如何避免重算，如何管理长期上下文</td><td>incremental encoding，memory retention，attention-aware eviction</td></tr>
      <tr><td>Concurrent-streaming</td><td>流式输入</td><td>同时流式输出</td><td>如何处理读写并发、位置冲突和交互策略</td><td>re-encoded / concatenated / interleaved / grouped streaming，rule/SFT/RL policy</td></tr>
    </tbody>
  </table>
</div>

<figure class="figure">
  <img src="assets/papers/streaming-llm-survey/main.png" alt="Streaming LLM 分类框架与核心挑战" loading="lazy" />
  <figcaption>图 2：论文的技术分类总览。Output-streaming 关注生成机制和生成效率；sequential-streaming 关注增量编码和上下文管理；concurrent-streaming 进一步引入架构适配和交互策略。</figcaption>
</figure>

## 第一层：Output-streaming LLM
Output-streaming 是最常见、也是最容易被误认为 “Streaming LLM 全部含义” 的范式。它处理的是 **完整输入已知时如何逐步显露输出**。

### 1. Streaming generation mechanism：输出如何逐步形成？
论文把生成机制分为三类。

#### Token-wise：逐 token 自回归
这是标准 LLM 的主流方式。模型每次生成一个 token，再把它加入上下文继续生成。优点是简单、稳定、可控；缺点是序列依赖强，天然存在串行延迟。

在多模态生成里，很多系统也会把非文本模态对齐到 token 空间，然后沿用自回归生成，例如语音 token、视频 token、图像 token 等。

#### Block-wise：按块生成
Block-wise 方法把生成单位从单 token 扩展到多个 token，试图减少串行深度。

- **Semi-autoregressive**：一个 block 内多个 token 并行预测，block 与 block 之间仍保持自回归关系。
- **Block-diffusion**：每次对一个 block 做扩散式去噪或修正，再按 block 顺序输出。

这类方法的核心取舍是：牺牲一部分严格的 token 级依赖，换取更高并行度。

#### Refinement-based：从粗到细逐步修正
这类方法不是简单往后追加 token，而是逐步提高整个输出的质量或分辨率。

- **Multi-scale**：例如先生成低分辨率或粗粒度结构，再逐步细化。
- **Global-diffusion**：从噪声或粗略序列开始，对整体进行多轮 denoising。

它更适合图像、视频、音频等可以自然粗到细呈现的模态，也给文本生成提供了不同于纯自回归的 streaming 可能性。

### 2. Efficient streaming generation：如何降低生成延迟和内存？
论文把效率优化收束到两个方向。

#### Decoding path acceleration：缩短执行路径
- **Token-path acceleration**：典型是 speculative decoding，用小 draft model 先并行提出多个候选 token，再由大模型验证接受。
- **Layer-depth acceleration**：根据 token 难度动态跳过层或提前退出，减少每个 token 的计算深度。

#### Memory efficiency：控制 KV cache 增长
自回归生成时 KV cache 会随生成长度线性增长。Streaming 场景下，如果输出很长，显存压力会持续上升。代表性思路包括：

- 保留 attention sink 和最近窗口。
- 根据 token 重要性动态驱逐 KV。
- 对保留的 KV 做量化、低秩近似或压缩。

这一层的重点是 **更快地写**，但还没有真正解决动态输入。

## 第二层：Sequential-streaming LLM
Sequential-streaming 开始处理真正的动态输入。它的关键问题从 “生成快不快” 转向 “输入持续增长时，模型还能不能稳定、低成本地维护上下文”。

### 1. Incremental encoding：如何只处理新来的部分？
增量编码要求新输入到来时，只编码新增部分，历史表示不因为未来输入而改变。否则每来一点数据都重算全部上下文，成本会迅速爆炸。

论文把增量编码分成两类。

#### Atomic encoding：天然有原子单位
文本 token 是最典型的 atomic unit。一个 token 的表示可以在它到来时确定，不需要等未来 token。

视频帧也可以作为预定义单位：每一帧独立进入视觉编码器，之后再做时间聚合。它的好处是简单在线，问题是帧级单位未必对应语义边界。

#### Fragmented encoding：人为切分连续信号
语音波形、视频像素流并没有天然离散边界，因此需要切分。

- **Fixed-interval partitioning**：按固定时间窗口切片，效率高、实现简单，但可能切断词、句子或动作。
- **Semantic-driven partitioning**：根据语音词边界、视频镜头切换、场景变化等语义信号切分，更连贯但成本更高。

这里的核心矛盾是：固定切分适合系统，语义切分适合理解。

### 2. Streaming context management：长期上下文如何保存？
当输入流越来越长，系统无法无限保存所有 token 或帧。论文把上下文管理拆成三个问题：

- **保留什么信息？**
- **如何存储和更新？**
- **如何高效访问？**

#### Memory retention：选择、淘汰与压缩历史
一种路线是选择重要内容，丢掉不重要内容。例如基于 saliency、recency、task relevance 选择关键片段。

另一种路线不是直接丢弃，而是做 token merging 或 memory consolidation，把多个历史状态合并成更紧凑的表示。这样能保留粗粒度长期信息，但会损失细节。

这也解释了很多长视频流论文的设计：短期保留细节，长期保留摘要，按任务需要再取回关键帧。

#### KV cache 与 attention 管理
KV 管理更偏模型内部状态。由于 attention 决定哪些历史状态会被访问，KV cache 的存储策略和 attention pattern 是耦合的。

- **Attention-aware eviction**：只保留模型可能关注的重要历史，如最近窗口、attention sink、heavy hitter token。
- **Representation compression**：对保留下来的 KV 做低比特量化、低秩近似或其他压缩。

Sequential-streaming 的目标是 **持续地读**，并在有限预算下维护一个不断更新的世界状态。

## 第三层：Concurrent-streaming LLM
Concurrent-streaming 是论文最强调的难点，因为它要求模型同时读输入和写输出。这比 “输入流式处理” 更复杂：模型生成过程中，新输入可能插入进来，读写顺序不再是标准 LLM 预训练时的单一因果链。

### 1. 架构冲突：Attention contention 和 Position-ID conflict
论文指出，把 batch-oriented LLM 改造成 concurrent-streaming LLM 会出现两个结构性冲突。

- **Attention contention**：新输入插入时，它和历史输出之间的因果依赖关系变得模糊。新输入应该看见历史输出吗？历史输出显然不能看见未来输入，但系统实现中如果不小心会造成依赖泄漏。
- **Position-ID conflict**：输入流和输出流异步增长时，新输入 token 和正在生成的输出 token 可能竞争同一位置编号，导致位置编码不再清晰。

<figure class="figure">
  <img src="assets/papers/streaming-llm-survey/conflict.png" alt="Concurrent streaming 中的注意力和位置冲突" loading="lazy" />
  <figcaption>图 3：Concurrent-streaming 的结构冲突。输入和输出同时展开后，模型需要重新定义注意力依赖和位置编号，否则会出现 attention contention 与 position-ID conflict。</figcaption>
</figure>

### 2. 四类架构适配方法
论文将 concurrent-streaming 的架构适配分为四类。

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>基本思路</th><th>优点</th><th>问题</th></tr></thead>
    <tbody>
      <tr><td>Re-encoded streaming</td><td>新输入到来后重编码全部历史 cache</td><td>最接近标准 batch attention，依赖关系清晰</td><td>计算成本高，不适合长上下文实时场景</td></tr>
      <tr><td>Concatenated streaming</td><td>把新输入和历史输出拼成复合序列共同输入</td><td>统一注意力和位置顺序</td><td>内存和延迟随上下文增长，通常需要架构改造和重训</td></tr>
      <tr><td>Interleaved streaming</td><td>按时间顺序交错排列输入 token 和输出 token</td><td>符合真实交互时间线，效率和连续性较平衡</td><td>需要同步机制，避免依赖泄漏</td></tr>
      <tr><td>Grouped streaming</td><td>输入和输出分组，维护独立注意力和位置空间</td><td>消除位置冲突，可提高并行性</td><td>需要设计组间信息流，训练和实现更复杂</td></tr>
    </tbody>
  </table>
</div>

我的理解是，这四类方法其实是在回答同一个问题：**输入流和输出流到底应该共享一条时间轴，还是各自有时间轴再通过受控通道交互？** Interleaved 倾向于共享时间轴，Grouped 倾向于隔离后再通信。

### 3. Interaction policy：何时读，何时写？
仅有架构还不够，模型还要决定交互节奏。论文把交互策略分为 rule-based、SFT-based 和 RL-based。

<figure class="figure">
  <img src="assets/papers/streaming-llm-survey/interaction.png" alt="Concurrent streaming 中的读写决策" loading="lazy" />
  <figcaption>图 4：交互策略决定模型在实时流中何时读取输入、何时生成输出。好的策略需要在低延迟和高质量之间动态平衡。</figcaption>
</figure>

#### Rule-based policy：规则驱动
最典型的是 Wait-k：先等待 k 个输入单位，再开始输出，并保持固定滞后。

规则方法简单、可控、可解释，但不理解输入难度。如果当前片段很简单，它可能等太久；如果当前片段很复杂，它可能输出太早。

也有 adaptive thresholding 方法，基于模型置信度、attention 或其他实时统计量决定是否继续读输入。

#### SFT-based policy：监督学习读写时机
这类方法用标注数据训练模型显式预测交互动作。

- **In-context prediction**：让 LLM 在正常生成中输出特殊控制 token，比如 `WAIT` 或 `EOS`，把交互决策融入语言建模。
- **Auxiliary decision**：额外训练一个轻量分类器，单独判断当前应该 read 还是 write。

前者更统一，后者更可控。区别在于是否把控制策略和语言分布绑定在一起。

#### RL-based policy：把交互看成序列决策
RL 方法把 read/write 当成动作，把质量和延迟设计成 reward，让模型学习非固定的交互模式。

它的优势是能直接优化 “回答质量 vs 响应延迟” 的权衡，尤其适合 proactive video interaction、实时同传、具身智能等场景。但训练更难，reward 设计也更敏感。

## 应用任务：Streaming LLM 能落到哪些场景？
论文刻意不把 output-streaming 当成主要应用讨论，因为它已经是大多数 LLM 的通用属性。应用部分重点放在 streaming input 和 concurrent interaction。

### 1. Sequential streaming tasks：长输入流理解
这类任务的共同点是输入可能很长或无界，但系统通常在某个查询点生成答案。

- **Streaming long video understanding**：视频持续输入，模型维护记忆；用户提问时基于当前记忆回答。
- **Streaming dialogue / logs / documents**：对不断增长的文本流做持续压缩和检索。
- **Streaming speech understanding**：语音持续到来，模型增量编码并更新语义状态。

这些任务的关键技术通常是增量编码和上下文管理，而不是复杂的读写同步。

### 2. Concurrent streaming tasks：实时双向交互
论文进一步按处理深度把 concurrent streaming 任务分为两类。

#### Perception-level：直接从输入映射到输出
这是 `X → Y` 形式，模型主要做实时感知和输出。

- **实时翻译 / 同传**：边听边翻译。
- **ASR / TTS**：边接收语音或文本边生成对应模态。
- **实时视频描述**：边看视频边生成 caption。
- **Streaming QA**：用户或环境输入未完全停止时，模型就开始回答。

这类任务强调低延迟和时序对齐。

#### Cognition-level：引入中间状态再输出
这是 `X → Z → Y` 形式，模型不只是直接映射，还要维护 latent state，用于推理、规划或工具调用。

代表方向包括：

- **Streaming reasoning**：输入持续变化时，模型持续维护思考状态。
- **Streaming tool usage / RAG**：输入过程中就触发检索、工具调用或外部动作。
- **Embodied / VLA 场景**：机器人一边感知环境，一边规划行动，一边输出语言或控制信号。

我觉得 cognition-level 是更接近 Agent 的部分：系统不只是反应快，而是要在连续世界中维护自己的内部状态。

## 未来方向：从流式模型到流式智能
论文把未来方向分成技术层和应用层。

### 技术层方向
#### 1. 更高效的 Streaming LLM
延迟和显存仍是核心瓶颈。未来需要同时优化：

- 增量编码的重算成本。
- 解码加速。
- 长期上下文和 KV cache 的在线预算管理。

这里的重点不是离线吞吐，而是在严格实时约束下做动态资源分配。

#### 2. 新的 concurrent streaming 范式
现有方法主要是 interleaved 和 grouped。论文认为还可以探索 semi-autoregressive 或 block-wise generation 与 concurrent interaction 的结合。

这很有意思：如果输出不是一个 token 一个 token 写，而是按 block 或多尺度写，交互策略也会改变。模型可能不再只是 wait/read/write，而是 read、draft、revise、commit。

#### 3. Proactive interaction policy
未来模型不应只被动等待用户问，而要能主动判断什么时候介入。

例如视频助手看到危险动作时主动提醒，会议助手在检测到决策点时主动总结，机器人在环境变化时主动调整计划。

#### 4. 可解释性
实时交互系统的行为动态还很难解释。模型为什么此刻说话？为什么等待？为什么忽略某段输入？这些问题在产品化时非常重要。

### 应用层方向
#### 1. 扩展 streaming modality
当前主要集中在文本、语音、基础视频。未来会走向更复杂的 omni-modal continuous streams，例如视频、音频、传感器、动作控制同时流动。

#### 2. 扩展 concurrency level
论文提出一个很好的层级想象：

- **两层并发**：listen-while-speaking，read-while-thinking。
- **三层并发**：perceiving、reasoning、generation 同时进行。
- **四层并发**：perceiving、reasoning、tool-using、generation 同时进行。

这本质上是把 LLM 从单线程对话模型推向多线程实时系统。

#### 3. 扩展 streaming tasks
应用会从简单被动响应转向主动交互和长期参与。模型需要长期记忆、主动干预、持续理解和动态规划，最终接近论文所说的 brain-like streaming intelligence。

## 和已有方向的区别
论文在附录里特别区分了 Streaming LLM 和几个相邻方向，这部分对理解边界很重要。

### Streaming LLM vs Efficient LLM
Efficient LLM 问的是：如何让模型在给定完整上下文时跑得更快、更省？

Streaming LLM 进一步问的是：当输入还在持续变化、输出也要实时生成时，模型如何边读边想边说？

也就是说，效率技术是 Streaming LLM 的基础组件，但不是完整定义。

### Streaming LLM vs Multimodal LLM
Multimodal LLM 关注 “信息是什么”：图像、语音、视频如何对齐到语言模型。

Streaming LLM 更关注 “信息如何随时间流动”：输入什么时候可见，输出什么时候生成，不同流之间如何同步。

一个多模态模型可以不是 streaming；一个 streaming 架构也可以处理文本、语音、视频等不同模态。

### Streaming LLM vs Long-context LLM
Long-context LLM 关注扩展固定上下文窗口，处理很长但通常有界的输入。

Streaming LLM 关注无界、实时、持续增长的数据流。它不只是把窗口变大，而是要在线维护状态、控制延迟，并处理读写并发。

这也是我觉得这篇综述最有价值的地方：它把 long context、efficient inference、multimodal interaction 这些技术重新放到 **动态数据流** 的框架里理解。

## 我的理解与启发
### 1. “Streaming” 应该按读写关系拆开看
过去看到 streaming，我经常会先想到流式输出。但这篇综述提醒我们，真正重要的是区分：

- 输出是不是流式？
- 输入是不是流式？
- 输入和输出是不是并发？
- 模型是否能学习读写时机？

这比简单说 “某模型支持 streaming” 更准确。

### 2. Concurrent-streaming 是 Agent 产品化的关键能力
很多 Agent demo 仍然是回合制：用户说完，模型想完，再调用工具，再回答。但真实助手、机器人、会议系统、视频陪伴系统都不是这样。它们需要在环境变化中持续更新状态，并在合适时机插话或行动。

因此，Streaming LLM 不只是推理优化方向，也是 Agent 从离线规划走向实时交互的基础。

### 3. 架构和策略必须一起设计
只解决 KV cache 或上下文压缩，只能让模型看得更久；只解决输出加速，只能让模型说得更快。实时交互还需要处理输入输出的结构关系和读写策略。

这也是论文把 architecture adaptation 和 interaction policy 并列放在 concurrent-streaming 下的原因。

### 4. 未来系统可能更像异步操作系统
如果模型同时进行 perceiving、reasoning、tool-using 和 generation，它就不再像一个单次函数调用，而更像一个有调度器、状态管理、外部 IO 和优先级控制的异步系统。

这对工程实现也有启发：未来的实时 LLM 应用可能需要显式设计事件循环、状态缓存、打断机制、工具调用队列和安全策略。

## 局限与待观察点
- **综述偏概念框架，不做统一实验比较**：论文主要梳理定义、范式和代表方法，没有提供跨系统的标准化 benchmark。
- **系统实现细节展开有限**：对部署中的并发调度、网络延迟、端侧资源、错误恢复等工程问题讨论不多。
- **术语仍可能继续演化**：Streaming LLM 方向发展很快，尤其是 omni-modal 和 agentic streaming，未来可能出现新范式。
- **交互策略评估仍缺统一标准**：质量、延迟、打断体验、主动性、安全性很难用单一指标衡量。
- **多流智能仍处早期**：论文展望了多层并发和多模态连续流，但多数现有工作还停留在两流或特定任务场景。

## 结论
这篇综述适合作为 Streaming LLM 方向的入门地图。它最重要的贡献是把 “streaming” 从模糊术语拆解成三个递进层次：**output-streaming 解决流式生成，sequential-streaming 解决动态输入处理，concurrent-streaming 解决实时双向交互。** 沿着这条路径，LLM 的角色也从静态推理器变成持续运行的交互系统。对于实时多模态助手、视频理解、语音同传、机器人和 Agent 来说，这个框架很有参考价值。