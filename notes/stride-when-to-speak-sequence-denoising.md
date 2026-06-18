---
id: "stride-when-to-speak-sequence-denoising"
title: "STRIDE：用序列去噪解决流式视频中的何时开口"
category: "Streaming VLM"
date: "2026-06-08"
order: 10
readTime: "12 min"
tags: ["Video-LLM","Streaming","Proactive AI","Masked Diffusion","When-to-Speak","Temporal Grounding"]
summary: "STRIDE 将流式视频理解里的 when-to-speak 问题从逐帧二分类改写为滑动窗口上的结构化序列建模，用轻量 masked diffusion 激活模块迭代去噪激活序列，从而生成更连续、更稳定的触发片段。"
hero: "assets/papers/stride/fig1.webp"
---

## 一句话总结
STRIDE 的核心贡献是把流式视频里的 **何时开口（when-to-speak）** 从“每帧独立判断要不要响应”改成“在一个时间窗口内预测连续激活片段”。它用轻量级 masked diffusion 模块对激活序列做迭代去噪，让模型在视频边到达边更新触发区间，从而减少边界抖动、过早响应和延迟响应。

<div class="metric-grid">
  <div class="metric"><strong>59.07</strong><span>Qwen3-VL-8B + STRIDE 在 OVO-Bench overall 得分</span></div>
  <div class="metric"><strong>59.70</strong><span>OVO-Bench Forward Active Responding 得分</span></div>
  <div class="metric"><strong>32.6</strong><span>ET-Bench 激活平均 F1，高于 Baseline-AR 的 24.3</span></div>
  <div class="metric"><strong>113ms</strong><span>K=8 时的激活模块在线开销，约 5.2GB VRAM</span></div>
</div>

## 论文信息
- **标题**：STRIDE: When to Speak Meets Sequence Denoising for Streaming Video Understanding
- **作者**：Junho Kim, Hosu Lee, James M. Rehg, Minsu Kim, Yong Man Ro
- **机构**：UIUC, KAIST, Google DeepMind
- **版本**：arXiv 2026
- **链接**：[arXiv:2603.27593](https://arxiv.org/abs/2603.27593) · [项目页](https://interlive-team.github.io/STRIDE/) · [代码](https://github.com/interlive-team/STRIDE) · [模型](https://huggingface.co/interlive)

## 背景：为什么 “when-to-speak” 很难？
过去很多 Video-LLM 主要解决的是离线理解：给模型一段完整视频，再让它回答问题。但真实应用更像一个持续运行的多模态助手：视频帧在线到达，模型不仅要知道 **说什么**，还要决定 **什么时候说**。

这带来三个关键难点：

- **部分可观测**：当前时刻只能看到已经到达的视频，不能偷看未来帧。
- **触发边界模糊**：事件开始、持续和结束通常不是单帧决定，而是一个连续时间段。
- **逐帧二分类容易抖动**：如果每帧独立输出 0/1，就容易出现 0→1→0→1 的 flickering，导致过早触发、重复触发或错过最佳响应时机。

STRIDE 的判断是：主动触发不是一个 point-wise classification 问题，而是一个 **span-structured sequence modeling** 问题。也就是说，模型应该预测一段连续激活区间，而不是孤立判断某一帧是否该说话。

<figure class="figure">
  <img src="assets/papers/stride/fig1.webp" alt="STRIDE 两阶段流式视频理解框架" loading="lazy" />
  <figcaption>图 1：STRIDE 总览。轻量 Activation Model 持续监听视频流，并在滑动窗口内用 masked diffusion 预测连续激活片段；触发后再把累积视觉上下文交给下游 Video-LLM 生成回复。</figcaption>
</figure>

## 方法：核心设计
### 1. 两阶段架构：把“何时说”和“说什么”解耦
STRIDE 采用两阶段 streaming framework：

- **Activation Model**：轻量前端模块，持续处理新到达的视频帧，判断是否应该触发响应。
- **Downstream Video-LLM**：只有在触发时才接收从最近一次 query 开始累积的视觉上下文，并生成具体回答。

这个设计的好处是模块化：STRIDE 不要求重训下游 Video-LLM，而是作为一个 plug-and-play 的触发器，给 Gemma3、InternVL3、Qwen3-VL 等不同后端提供更合适的响应时机。

### 2. Span-level activation：把触发建模为窗口序列
在时刻 T，STRIDE 维护长度为 W 的激活窗口：

\[
\mathbf{a}_T = [a_{T-W}, \ldots, a_T] \in \{0,1\}^W
\]

其中 1 表示当前时间位置处于可响应的 active 区间，0 表示 inactive。这样，模型可以显式学习三个转移模式：

- **0→1 onset**：什么时候开始具备足够证据，可以开口。
- **1→1 persistence**：事件持续期间保持稳定激活。
- **1→0 offset**：什么时候事件结束，不该继续触发。

这比单帧二分类更贴近视频事件的真实形态：很多“该不该回答”的判断依赖一个短时间段内的上下文，而不是单独一帧。

### 3. Masked diffusion：用迭代去噪预测激活片段
STRIDE 将激活序列看作离散 token 序列，token 取值为 0、1 或 `[M]`。训练和推理都围绕 masked diffusion 展开：

- **训练时**：随机遮住激活序列的一部分，让模型根据视频、query 和未遮住的激活状态恢复真实 0/1。
- **推理时**：从带 mask 的激活窗口开始，分 K 步逐渐填充高置信位置，再反复细化边界位置。

论文默认使用 **K=8** 个 denoising steps。由于输出空间只有 0/1，去噪过程收敛很快，约 100ms 级别就能达到较好精度。

<figure class="figure">
  <img src="assets/papers/stride/fig2.webp" alt="STRIDE 激活建模和推理过程" loading="lazy" />
  <figcaption>图 2：训练阶段使用 sequence duplication 和三类结构化 masking；推理阶段滑动窗口随新帧前移，保留高置信历史状态，重新 mask 低置信位置并继续迭代去噪。</figcaption>
</figure>

### 4. 三类结构化 masking：逼模型学习边界
标准 masked diffusion 通常独立随机 mask token，但这不适合激活序列。因为激活区域是连续片段，如果只随机遮少数点，模型可能靠局部插值就猜出来，而不需要真正理解视频事件边界。

STRIDE 设计了三类 corruption：

- **Boundary-Anchored Span Masking**：遮住覆盖激活边界的连续块，迫使模型判断 onset / offset。
- **Span Unmasking**：从全 mask 序列中露出一段连续区域，模拟推理时先确定高置信片段、再细化边界的过程。
- **Full Masking**：整个激活序列全 mask，用于训练 cold-start 情况，让模型仅从视频上下文估计整体激活布局。

消融结果显示，三者结合时 ET-Bench 平均 F1 达到 **32.6**；如果只用独立 mask，平均 F1 只有 **7.2**。

### 5. Sequence duplication：不改 causal backbone，也获得窗口级上下文
STRIDE 的 activation backbone 初始化自 Qwen3-VL-2B 这类 causal 模型。问题在于：masked diffusion 需要双向上下文，而 causal attention 默认只能看左侧。

论文用一个很巧的输入重参数化解决：把激活序列复制成 `[a, a']`。前一个 `a` 作为 conditioning prefix，后一个 `a'` 作为预测目标。由于 `a` 位于 `a'` 左侧，`a'` 中每个 token 都可以在 causal attention 下看到完整窗口信息。

这避免了修改底层 attention mask，也保留了直接复用预训练 VLM 的便利性。

### 6. Selective re-masking：新帧到来后修正旧判断
在 streaming inference 中，窗口每来一帧就前移一格。旧判断不能无条件继承，因为新视觉证据可能改变此前的边界判断。STRIDE 使用置信度阈值 \(\tau\)：

- 如果 carried-forward 的激活状态置信度超过 \(\tau\)，则保留。
- 如果置信度不足，则重新变成 `[M]`，和新帧位置一起进入下一轮 denoising。

论文默认 \(\tau=0.75\)，触发阈值 span ratio \(\gamma=1\)，视频以 **1 FPS** 进入激活模型。

## 数据与评测
### 训练数据
论文从多个带时间标注的视频理解数据集中构造激活训练样本，覆盖：

- dense video captioning
- temporal activity detection
- grounded video QA
- sequential step recognition
- moment localization

核心处理方式是把时间标注转换为二值激活序列：标注事件内部帧为 active，其他帧为 inactive。对于多事件或过程性任务，还会避免模型对已经发生过的事件重复触发。

### 评测集
论文主要使用三个互补 benchmark：

- **OVO-Bench**：在线视频理解，包含 backward tracing、real-time visual perception 和 forward active responding，重点考察是否能等到足够证据再回答。
- **StreamingBench**：覆盖 real-time visual understanding、omni-source understanding、contextual understanding，其中 Proactive Output 任务直接考察响应时机。
- **ET-Bench**：用于单独评估 activation model 的时间定位质量，任务包括 TVG、EPM、TAL、DVC、SLC，指标为 F1。

## 实验结果
### 1. OVO-Bench：主动响应能力提升明显
在 OVO-Bench 上，STRIDE 接到不同下游模型后都能带来提升。以 Qwen3-VL-8B 为例，overall 从 **51.77** 提升到 **59.07**，Forward Active Responding 从 **46.30** 提升到 **59.70**。

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>Real-Time Visual Perception</th><th>Backward Tracing</th><th>Forward Active Responding</th><th>Overall</th></tr></thead>
    <tbody>
      <tr><td>Qwen3-VL-8B</td><td>65.00</td><td>44.00</td><td>46.30</td><td>51.77</td></tr>
      <tr><td>Qwen3-VL-8B + Baseline-AR</td><td>69.35</td><td>46.33</td><td>42.73</td><td>52.81</td></tr>
      <tr><td><strong>Qwen3-VL-8B + STRIDE</strong></td><td><strong>69.68</strong></td><td><strong>47.83</strong></td><td><strong>59.70</strong></td><td><strong>59.07</strong></td></tr>
    </tbody>
  </table>
</div>

这里最值得注意的是 Forward Active Responding：它不是简单“看见就答”，而是要求模型在证据不足时等待。STRIDE 的 span-level 激活更适合这种场景。

### 2. StreamingBench：Proactive Output 也受益
在 StreamingBench 上，Qwen3-VL-8B + STRIDE overall 达到 **59.29**，高于 Qwen3-VL-8B 原始在线推理的 **46.84**，也高于 Baseline-AR 的 **57.12**。

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>Real-time Visual Understanding</th><th>Omni-Source Understanding</th><th>Contextual Understanding</th><th>Proactive Output</th><th>Overall</th></tr></thead>
    <tbody>
      <tr><td>Qwen3-VL-8B</td><td>60.88</td><td>30.40</td><td>28.20</td><td>32.40</td><td>46.84</td></tr>
      <tr><td>Qwen3-VL-8B + Baseline-AR</td><td>73.79</td><td>36.20</td><td>36.38</td><td>24.30</td><td>57.12</td></tr>
      <tr><td><strong>Qwen3-VL-8B + STRIDE</strong></td><td><strong>74.24</strong></td><td><strong>41.30</strong></td><td><strong>39.90</strong></td><td><strong>42.80</strong></td><td><strong>59.29</strong></td></tr>
    </tbody>
  </table>
</div>

Baseline-AR 在整体上已有提升，但 Proactive Output 只有 24.30，说明逐帧触发虽然能帮助部分任务，却不一定能学好“等到合适时机再说”。STRIDE 在 PO 上达到 **42.80**，更能体现结构化激活的价值。

### 3. ET-Bench：激活边界定位更准
为了剥离下游 Video-LLM 的影响，论文在 ET-Bench 上单独评估 activation model。STRIDE 只有 2B 参数，但平均 F1 达到 **32.6**，高于 Baseline-AR 的 **24.3**。其中 TVG 从 **35.7** 大幅提升到 **62.8**。

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>参数</th><th>TVG F1</th><th>EPM F1</th><th>TAL F1</th><th>DVC F1</th><th>SLC F1</th><th>Avg</th></tr></thead>
    <tbody>
      <tr><td>Baseline-AR</td><td>2B</td><td>35.7</td><td>2.5</td><td>21.2</td><td><strong>39.6</strong></td><td>22.6</td><td>24.3</td></tr>
      <tr><td><strong>STRIDE</strong></td><td>2B</td><td><strong>62.8</strong></td><td><strong>10.7</strong></td><td><strong>24.6</strong></td><td>36.5</td><td><strong>28.5</strong></td><td><strong>32.6</strong></td></tr>
    </tbody>
  </table>
</div>

这说明 STRIDE 的主要收益确实来自更好的时间边界建模，而不仅是下游模型更强。

<figure class="figure">
  <img src="assets/papers/stride/fig3.webp" alt="STRIDE 和 Baseline-AR 在事件边界附近的激活抖动对比" loading="lazy" />
  <figcaption>图 3：ET-Bench TVG 上的事件边界转移频率。Baseline-AR 在事件开始和结束附近频繁 0/1 抖动；STRIDE 的激活片段更平滑，边界更稳定。</figcaption>
</figure>

## Ablation：哪些设计最关键？
### 1. 结构化 masking 是核心
消融显示，masking 策略决定了模型是否能真正学习 span 结构：

<div class="table-wrap">
  <table>
    <thead><tr><th>Masking 策略</th><th>TVG F1</th><th>DVC F1</th><th>SLC F1</th><th>Avg</th></tr></thead>
    <tbody>
      <tr><td>Independent only</td><td>8.5</td><td>8.8</td><td>9.2</td><td>7.2</td></tr>
      <tr><td>Span only</td><td>30.6</td><td>25.4</td><td>20.6</td><td>21.1</td></tr>
      <tr><td>Span + Full</td><td>36.8</td><td>24.0</td><td>21.3</td><td>23.0</td></tr>
      <tr><td><strong>Span + Full + Span Unmasking</strong></td><td><strong>62.8</strong></td><td><strong>36.5</strong></td><td><strong>28.5</strong></td><td><strong>32.6</strong></td></tr>
    </tbody>
  </table>
</div>

我的理解是：independent mask 让模型学到的是“补点”，structured mask 才让模型学到“找边界”。

### 2. Sequence duplication 带来完整窗口上下文
去掉 sequence duplication 后，ET-Bench 平均 F1 从 **32.6** 降到 **22.9**，DVC 从 **36.5** 降到 **19.9**。这说明即使模型底层是 causal backbone，触发判断仍然需要窗口级上下文；只看左侧很难稳定恢复完整激活片段。

### 3. Selective re-masking 避免错误继承
如果只 mask 新加入的最后一个位置，旧判断永远固定，平均 F1 只有 **22.6**。加入 selective re-masking 后升到 **32.6**。这说明 streaming 场景下历史判断必须允许被新证据修正，否则早期边界错误会持续传播。

<figure class="figure">
  <img src="assets/papers/stride/fig5.webp" alt="STRIDE 对 retention threshold tau 的敏感性分析" loading="lazy" />
  <figcaption>图 4：\(\tau\) 敏感性分析。无条件继承旧判断（\(\tau=0\)）效果最差；多数任务在 \(\tau \in [0.75, 0.85]\) 附近表现最好，论文默认取 \(\tau=0.75\)。</figcaption>
</figure>

## 效率：触发器开销小，但能省掉大量无效生成
STRIDE 的 activation model 使用 Qwen3-VL-2B 初始化，在单张 H100 上评估。论文报告：

- base activation state 占用约 **5.2GB VRAM**。
- 每个 denoising step 约 **12ms**，额外 **10MB**。
- append 新帧约 **20ms**，额外 **30MB**。
- K=8 时总触发判断约 **113ms**。
- 如果无需触发，下游 Video-LLM 的生成可以被跳过，相比每帧都调用大模型可节省约 **91%** 时间。

<figure class="figure">
  <img src="assets/papers/stride/fig4.webp" alt="STRIDE denoising step 数和延迟精度权衡" loading="lazy" />
  <figcaption>图 5：denoising step K 的延迟-精度权衡。K 增大能提升激活 F1，但很快趋于饱和；论文选择 K=8，在精度和实时开销之间折中。</figcaption>
</figure>

## 我的理解与启发
这篇论文的价值不在于“又做了一个视频 benchmark 提升”，而在于它把实时多模态交互里一个经常被忽略的问题拆了出来：**模型什么时候应该主动说话？**

我觉得有三点特别值得借鉴：

1. **触发不是分类点，而是时间段**：真实视频事件天然有持续区间，工程上也更需要稳定的触发片段，而不是高频抖动的单点概率。
2. **轻量 gating 是实时系统的关键组件**：不应该让大 Video-LLM 每帧都做完整理解和生成；前端 activation model 可以先决定是否值得唤醒后端。
3. **允许修正历史判断很重要**：streaming 系统天然处在 partial observability 下，随着新帧到来，旧边界判断可能需要调整。Selective re-masking 是一个很实用的机制。

从产品视角看，STRIDE 更接近一个“多模态唤醒器”：它不直接解决所有视觉推理问题，而是负责在合适的时间把上下文交给更强的模型。这种分层设计对实时助手、车载系统、可穿戴设备和机器人都很有参考价值。

## 局限与待观察点
- **依赖时间标注数据**：训练需要将事件区间转成激活序列，开放场景下高质量时间标注成本不低。
- **激活和回答仍是两阶段解耦**：触发器只决定何时唤醒，回答质量依赖下游 Video-LLM；触发最优不等于最终回答最优。
- **1 FPS 设置可能不足以覆盖高频动作**：对体育、驾驶或机器人操作等快速场景，更高帧率下的延迟和显存表现还需要验证。
- **\(\gamma=1\) 的触发规则较简单**：不同产品可能需要更复杂的触发策略，例如置信度、用户打断、历史对话状态和风险级别共同决定。
- **评测仍偏 benchmark 化**：真实主动助手需要考虑用户体验，比如误唤醒成本、沉默成本、重复提示和多轮交互节奏。

## 结论
STRIDE 提供了一条很清晰的路线：把 proactive streaming video understanding 拆成 “activation + response generation”，并把 activation 建模为滑动窗口上的结构化序列去噪。相比逐帧二分类，它更能捕捉事件边界和持续区间，因此在 OVO-Bench、StreamingBench 和 ET-Bench 上都带来更稳定的 when-to-speak 表现。对构建实时多模态 Agent 来说，这篇论文强调的不是“输入更长视频”，而是“在合适的时间唤醒合适的模型”。
