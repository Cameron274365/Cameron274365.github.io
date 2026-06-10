---
id: "streamingvlm-real-time-infinite-video"
title: "StreamingVLM：面向无限视频流的实时理解"
category: "Streaming VLM"
date: "2026-06-03"
order: 30
readTime: "12 min"
tags: ["VLM","Streaming","KV Cache","RoPE","Video Understanding"]
summary: "StreamingVLM 通过训练-推理一致的流式框架，让 VLM 在近无限视频上保持低延迟、低显存和稳定理解能力。论文构建了 Inf-Streams-Train / Eval，并在长达 2 小时以上的视频流评测中取得强结果。"
hero: "assets/papers/streamingvlm/x1.png"
---

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
