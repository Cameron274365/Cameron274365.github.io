---
id: "simple-stream-baseline-streaming-video-understanding"
title: "SimpleStream：流式视频理解中的强简单基线"
category: "Streaming VLM"
date: "2026-06-08"
order: 10
readTime: "11 min"
tags: ["VLM","Streaming Video","Simple Baseline","OVO-Bench","StreamingBench"]
summary: "SimpleStream 质疑流式视频理解中越来越复杂的记忆机制：只把最近 N 帧输入现成 VLM，就能在 OVO-Bench 和 StreamingBench 上匹配甚至超过多个已发表流式方法。论文的关键价值不在新架构，而在提醒后续工作必须先跑强 recency baseline，并拆开评估实时感知与长期记忆。"
---

## 一句话总结
SimpleStream 的结论很直接：在当前流式视频理解 benchmark 上，**强 VLM backbone + 未压缩的最近几帧** 已经是非常强的系统。论文用一个几乎没有工程复杂度的 recent-window baseline 证明，复杂记忆、检索和压缩模块如果不能在同协议下明确超过它，就很难说明自己真的带来了流式能力进步。

<div class="metric-grid">
  <div class="metric"><strong>4 帧</strong><span>SimpleStream 最强默认窗口，仅使用最近 4 帧</span></div>
  <div class="metric"><strong>67.70%</strong><span>Qwen3-VL-8B + 4f 在 OVO-Bench 的 Avg.</span></div>
  <div class="metric"><strong>80.59%</strong><span>Qwen3-VL-8B + 4f 在 StreamingBench RTVU</span></div>
  <div class="metric"><strong>13 个</strong><span>对比离线与在线视频 LLM 基线</span></div>
</div>

## 论文信息
- **标题**：A Simple Baseline for Streaming Video Understanding
- **作者**：Yujiao Shen, Shulin Tian, Jingkang Yang, Ziwei Liu
- **机构**：S-Lab, Nanyang Technological University
- **版本**：arXiv 2026-04-02
- **链接**：[arXiv:2604.02317](https://arxiv.org/abs/2604.02317) · [项目主页](https://simple-stream.github.io/) · [代码](https://github.com/EvolvingLMMs-Lab/SimpleStream)

## 背景：为什么这篇文章值得看？
流式视频理解这两年有一个明显趋势：为了处理长视频流，方法越来越依赖复杂的历史管理机制，比如外部记忆、检索、KV cache 压缩、层级记忆或 latent memory。

这些设计的共同假设是：**强流式理解必须依赖强历史记忆**。但论文指出，很多 benchmark 里的问题并不总是纯长期记忆问题；大量题目仍然高度依赖当前画面、短期动作、OCR、物体和属性识别。如果把最新视觉证据压缩、稀释或混入噪声历史，反而可能伤害实时感知。

SimpleStream 的意义就在这里：它不是提出一个更复杂的新模块，而是问了一个更基础的问题：**只看最近几帧，到底已经能做到多强？**

<figure class="figure">
  <img src="assets/papers/simple-stream/figure1_baseline_teaser.webp" alt="SimpleStream 方法概览与感知记忆对比" loading="lazy" />
  <figcaption>图 1：SimpleStream 只把最近 N 帧和问题输入现成 VLM。右侧展示它在 OVO-Bench 的 perception-memory 坐标上已经处于很强的位置。</figcaption>
</figure>

## 方法：SimpleStream 到底有多简单？
### 1. 问题建模：因果、预算受限的上下文管理
论文把 streaming QA 形式化为：在查询时刻 \(t\)，模型只能看到视频前缀，不能访问未来帧；同时推理时的显存、attention token 和计算预算都是有限的。因此关键不是“能不能把整个视频塞进去”，而是如何从历史中构造一个有界工作上下文 \(C_t\)。

很多已有方法会扩展或重组 \(C_t\)：

- **外部记忆**：维护事件树、固定大小 memory slot 或全局摘要。
- **检索式历史**：把过去表征保存起来，在问题到来时按查询取回。
- **压缩式方法**：压缩 KV、视觉 token 或 latent state。
- **训练式流式模型**：通过专门监督或位置设计学习在线生成。

SimpleStream 刻意不做这些事情，只构造一个最小对照组。

### 2. 核心公式：只取最近 N 帧
给定视频帧序列 \(f_i\) 和查询 \(q_t\)，SimpleStream 的输入是：

\[
\textsc{SimpleStream}(t)=\mathrm{VLM}(\{f_{t-N+1},\ldots,f_t\}, q_t)
\]

也就是说，它只保留最近 \(N\) 帧，窗口外的帧直接丢弃。它没有额外训练、没有 memory bank、没有 retrieval、没有视觉压缩、没有 KV 历史管理，只是一个 inference-time input policy。

这种极简设定反而让它成为很干净的基线：如果复杂模块在相同 backbone、相同协议下赢不了它，就不能把提升归因于“更强的流式记忆”。

<figure class="figure">
  <img src="assets/papers/simple-stream/figure2_landscape.webp" alt="流式视频理解方法谱系" loading="lazy" />
  <figcaption>图 2：论文把已有流式视频理解方法按 external memory、retrieval、compression、latent memory 等机制分类，并把 SimpleStream 放在只依赖 recent context 的极简位置。</figcaption>
</figure>

## 数据与评测
论文主要使用两个 benchmark：

- **OVO-Bench**：包含 1,640 个问题，覆盖 memory recall、real-time perception、future-oriented reasoning。论文重点评测 Backward Tracing 和 Real-Time Visual Perception 两类，因为它们最能体现记忆与实时感知的张力。
- **StreamingBench**：使用官方 real-time visual understanding subset，共 2,500 个问题，覆盖 10 类任务，用来验证趋势是否能迁移到另一个实时理解 benchmark。

对比对象包括 6 个离线视频 LLM 和 7 个代表性在线/流式视频 LLM。SimpleStream 使用两个开源 backbone：Qwen2.5-VL-7B-Instruct 和 Qwen3-VL-8B-Instruct；默认以 1 fps 采样，只喂最近 \(N\in\{2,4,8\}\) 帧。

## 实验结果
### 1. 主结果：4 帧 SimpleStream 超过多个复杂流式模型
在 OVO-Bench 上，最强的 SimpleStream 配置是 **Qwen3-VL-8B + 4f**，Avg. 达到 **67.70%**，超过此前最强 published streaming method HERMES 的 **59.20%**，领先 **8.5 pp**。

在 StreamingBench 上，同一配置达到 **80.59%**，也超过 HERMES 的 **79.44%**。这说明 SimpleStream 不是只在单一 benchmark 上偶然有效，而是在两个主流流式视频评测里都很强。

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>帧设置</th><th>StreamingBench</th><th>OVO Real-Time</th><th>OVO Backward</th><th>OVO Avg.</th></tr></thead>
    <tbody>
      <tr><td>StreamForest-7B</td><td>1 fps</td><td>77.26</td><td>61.2</td><td>52.0</td><td>56.60</td></tr>
      <tr><td>HERMES-7B</td><td>1 fps / 4K tokens</td><td>79.44</td><td>69.0</td><td>49.4</td><td>59.20</td></tr>
      <tr><td>SimpleStream Qwen2.5-VL-7B + 4f</td><td>4 帧</td><td>78.47</td><td>78.4</td><td>51.9</td><td>65.13</td></tr>
      <tr><td><strong>SimpleStream Qwen3-VL-8B + 4f</strong></td><td><strong>4 帧</strong></td><td><strong>80.59</strong></td><td><strong>81.4</strong></td><td><strong>54.0</strong></td><td><strong>67.70</strong></td></tr>
    </tbody>
  </table>
</div>

一个很关键的细节是，SimpleStream 的优势主要来自 Real-Time Visual Perception：Qwen3-VL-8B + 4f 在该类上达到 **81.4%**，而 HERMES 是 **69.0%**。这支持了论文的核心判断：当前 benchmark 中，“保留清晰最近画面”比“注入复杂历史”更容易带来高分。

### 2. 窗口大小：更多上下文不一定更好
论文做了 recent-window ablation，测试 \(N=2,4,8,16\)。结果并不是窗口越大越好：

- 从 2 帧到 4 帧，Overall 从 **66.4** 提升到 **67.7**，Real-Time 从 **79.3** 提升到 **81.4**。
- 到 8 帧时，Overall 降到 **67.4**，Real-Time 降到 **79.9**。
- 到 16 帧时，Overall 继续降到 **67.1**，Real-Time 降到 **77.9**。

这说明短期上下文确实有帮助，但超过某个点后，额外帧会带来冗余、注意力稀释或视觉干扰。

<figure class="figure">
  <img src="assets/papers/simple-stream/figure3_window_size_accuracy.webp" alt="SimpleStream 窗口大小消融" loading="lazy" />
  <figcaption>图 3：窗口大小消融。4 帧在受控设置下取得最高 Real-Time accuracy，继续扩大窗口并不会单调提升整体表现。</figcaption>
</figure>

### 3. 模型规模：最优窗口依赖 backbone，而不是简单随规模增大
论文进一步在 Qwen2.5-VL 与 Qwen3-VL 多个尺寸上测试 2/4/8/16 帧。结论更细：更大的模型有时能利用更长窗口，但不存在“模型越大，最优窗口必然越长”的规律。

例如：

- Qwen2.5-VL-7B 在 4 帧达到最佳 Avg. **65.13**。
- Qwen2.5-VL-72B 在 16 帧达到最佳 Avg. **70.76**。
- Qwen3-VL-32B 在 8 帧达到最佳 Avg. **74.09**。
- Qwen3-VL-30B-A3B 反而在 4 帧达到最佳 Avg. **73.28**。

因此，窗口长度不是一个可以固定照搬的超参，而是和 backbone 的视觉编码、上下文处理能力、训练分布和 benchmark 能力结构共同相关。

<figure class="figure">
  <img src="assets/papers/simple-stream/figure4_optimal_frames_by_scale.webp" alt="不同模型规模下的最优窗口" loading="lazy" />
  <figcaption>图 4：模型规模消融。许多 checkpoint 在 4 帧附近达到最优，但也有更大模型偏好 8 或 16 帧，说明最优 recent window 与 backbone family 和模型容量有关。</figcaption>
</figure>

### 4. 效率：低显存、低 TTFT，不依赖历史状态
SimpleStream 的另一个优点是状态不会随视频流增长。因为它只保留固定数量的最近帧，所以 peak GPU memory 基本保持平坦。

TTFT 方面，SimpleStream-4f 在 16/64/256 observed frames 下分别是 **35/33/38 ms**，只慢于 HERMES 的 **27/29/29 ms**，明显快于大多数流式方法。对于一个没有专门 memory module 的 baseline 来说，这个延迟已经非常有竞争力。

<figure class="figure">
  <img src="assets/papers/simple-stream/figure5_vram_scaling_sota.webp" alt="峰值显存随观察帧数变化" loading="lazy" />
  <figcaption>图 5：SimpleStream-4f 只保留固定 recent window，因此 peak GPU memory 随 observed frames 增长基本保持平坦。</figcaption>
</figure>

## Ablation：记忆真的带来收益了吗？
### Visual-RAG：历史检索提升部分记忆题，但伤害实时感知
论文用 Visual-RAG 测试“加入历史是否更好”：离线建立 CLIP 历史 chunk 索引，推理时取 top-5 相似历史片段附加到 recent frames 后面。

结果非常有代表性：

<div class="table-wrap">
  <table>
    <thead><tr><th>能力/任务</th><th>Base</th><th>+V-RAG</th><th>变化</th></tr></thead>
    <tbody>
      <tr><td>EPM：Episodic Memory</td><td>52.5</td><td>59.6</td><td><strong>+7.1</strong></td></tr>
      <tr><td>ASI：Action Sequence Identification</td><td>58.8</td><td>64.9</td><td><strong>+6.1</strong></td></tr>
      <tr><td>HLD：Hallucination Detection</td><td>45.7</td><td>33.3</td><td><strong>-12.4</strong></td></tr>
      <tr><td>OCR：Optical Character Recognition</td><td>94.0</td><td>85.9</td><td><strong>-8.1</strong></td></tr>
      <tr><td>ACR：Action Recognition</td><td>78.9</td><td>71.6</td><td><strong>-7.3</strong></td></tr>
      <tr><td>OJR：Object Recognition</td><td>81.5</td><td>72.3</td><td><strong>-9.2</strong></td></tr>
      <tr><td>Overall Acc.</td><td>66.0</td><td>63.7</td><td><strong>-2.3</strong></td></tr>
    </tbody>
  </table>
</div>

这组数字很好地说明了论文的核心 trade-off：历史确实能提升一些记忆任务，但代价是多个实时感知任务明显下降，最终 overall 反而变差。

### Perception-memory trade-off
论文进一步定义了两个指标：

- \(\Delta P\)：相对 SimpleStream 的 Real-Time average 变化，衡量实时感知成本。
- \(\Delta M\)：EPM 与 ASI 的均值变化，衡量更接近 episodic recall 的记忆收益。

结果显示，很多外部 memory / retrieval / long-context 方法能带来正向 memory gain，但几乎都付出了 perception cost。例如 StreamForest 的 \(\Delta M=+8.9\)，但 \(\Delta P=-13.8\)；HERMES 的 \(\Delta M=+2.4\)，但 \(\Delta P=-6.0\)。

<figure class="figure">
  <img src="assets/papers/simple-stream/figure6_perception_memory_tradeoff.webp" alt="感知成本与记忆收益权衡" loading="lazy" />
  <figcaption>图 6：相对 SimpleStream Qwen2.5-VL + 2f 的 perception cost 与 memory gain。许多方法能提升记忆相关指标，但同时显著损失实时感知。</figcaption>
</figure>

## 我的理解与启发
这篇文章最有价值的地方不是“4 帧就够了”这个表层结论，而是重新校准了流式视频理解的研究基线。

1. **强 baseline 是研究复杂模块的前提**：如果一个 memory module 只和弱基线比，很容易把 backbone 能力、prompt 差异或评测偏置误认为方法贡献。SimpleStream 提醒我们，复杂设计必须先超过 matched recency baseline。
2. **实时感知和长期记忆需要拆开看**：单一 macro-average 会掩盖能力取舍。一个方法可能记忆更强，但实时画面识别更差；如果 benchmark 更偏实时感知，它就会被整体分数惩罚。
3. **历史不是越多越好，而是要按需取用**：未来更合理的方向可能是 recent-first, history-on-demand。默认保护最新画面质量，只在当前证据不足时检索或注入历史。
4. **压缩与检索要避免污染当前证据**：很多多模态 Agent 场景里，最近几秒的信息往往最可靠。历史模块如果占用过多 token 或引入噪声，就会稀释模型对当前画面的注意力。

对产品化实时助手也有启发：先把近期感知链路做稳，未必比堆复杂长期记忆弱；而长期记忆模块应该被设计成可解释、可控、按需触发，而不是默认全量注入。

## 局限与待观察点
- **依赖强 backbone**：论文主要基于 Qwen2.5-VL 和 Qwen3-VL，结论是否能泛化到更多 VLM 家族仍需验证。
- **不是长期记忆解决方案**：SimpleStream 本身会直接丢弃窗口外历史，因此不能解决真正依赖远距离事件的任务。
- **benchmark 可能偏向实时感知**：论文自己也指出 OVO-Bench 的 macro-average 对 Real-Time track 更敏感，因此高分不等价于解决 long-horizon memory。
- **4 帧不是通用最优值**：不同模型规模和 backbone family 的最优窗口不同，实际系统仍需要按任务和模型调参。
- **Visual-RAG 实现只是一个 probe**：历史检索失败不代表所有记忆机制都无效，但它证明“加历史”需要被更严格地量化。

## 结论
SimpleStream 是一篇很适合作为研究校准点的论文：它用极简 recent-window baseline 证明，当前流式视频理解 benchmark 上的很多收益可能来自强 backbone 和清晰最近视觉证据，而不是复杂记忆机制本身。后续方法如果要主张“更强流式理解”，至少应该在相同协议下超过 SimpleStream，并明确报告实时感知、长期记忆、幻觉鲁棒性和效率之间的取舍。