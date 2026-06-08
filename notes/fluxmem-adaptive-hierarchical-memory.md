---
id: "fluxmem-adaptive-hierarchical-memory"
title: "FluxMem：面向流式视频理解的自适应层级记忆"
category: "论文阅读"
date: "2026-06-08"
order: 20
readTime: "11 min"
tags: ["VLM","Streaming Video","Memory","Token Compression","Training-free","OVO-Bench","StreamingBench"]
summary: "FluxMem 是一个免训练的流式视频记忆框架，通过短期、中期、长期三层视觉记忆，以及 TAS 与 SDC 两个自适应压缩模块，在保持因果在线处理的同时降低视觉 token、延迟和显存开销。"
hero: "assets/papers/fluxmem/teaserfigure.png"
---

## 一句话总结
FluxMem 的核心思想是：**不要把所有历史视频 token 平等保存**。它把视觉记忆拆成短期、中期、长期三层：近期画面完整保留，中期画面用时间相邻变化筛选，长期画面再做空间区域合并；同时用 Otsu 自适应阈值根据场景动态自动决定压缩强度。因此，它不需要重新训练模型，就能让 Qwen2.5-VL-7B 这类 MLLM 更适合实时流式视频理解。

<div class="metric-grid">
  <div class="metric"><strong>76.4</strong><span>StreamingBench real-time 分数</span></div>
  <div class="metric"><strong>67.2</strong><span>OVO-Bench real-time 分数</span></div>
  <div class="metric"><strong>-69.9%</strong><span>OVO-Bench 推理延迟下降</span></div>
  <div class="metric"><strong>-34.5%</strong><span>OVO-Bench 峰值显存下降</span></div>
</div>

## 论文信息
- **标题**：FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding
- **作者**：Yiweng Xie, Bo He, Junke Wang, Xiangyu Zheng, Ziyi Ye, Zuxuan Wu
- **机构**：Fudan University, Shanghai Innovation Institute, Shanghai Key Laboratory of Multimodal Embodied AI, University of Maryland, College Park
- **会议/版本**：CVPR 2026，arXiv:2603.02096
- **链接**：[arXiv](https://arxiv.org/abs/2603.02096) · [PDF](https://arxiv.org/pdf/2603.02096) · [Code](https://github.com/YiwengXie/FluxMem)

<figure class="figure">
  <img src="assets/papers/fluxmem/teaserfigure.png" alt="FluxMem 论文总览图" loading="lazy" />
  <figcaption>图 1：FluxMem 是一个渐进式压缩视觉 token 的自适应层级记忆框架；论文也展示了它在长视频推理中相对其他方法的显存优势。</figcaption>
</figure>

## 背景：为什么流式视频需要新的记忆机制？
传统视频 MLLM 多数面向离线场景：完整视频已经给定，模型可以一次性采样帧并回答问题。但机器人、自动驾驶、智能眼镜等真实场景更像持续到来的视觉流，模型需要一边接收画面，一边保留历史上下文，并在用户提问或场景变化时及时回答。

这里的关键矛盾是：

- **视觉 token 冗余很高**：相邻帧中背景、静态物体和重复区域会产生大量相似 token。
- **历史不能简单丢弃**：很多问题依赖过去事件，尤其是历史检索、动作变化、未来预测和空间关系。
- **近期信息更重要**：当前 query 通常更依赖刚刚发生的画面，近期细节不能被过度压缩。
- **固定压缩比例不稳**：静态场景和快速运动场景的信息密度不同，手调阈值或固定 drop ratio 很难泛化。

FluxMem 的定位就是在这些约束之间做折中：它不是训练一个新的在线模型，而是作为 **training-free、plug-and-play 的视觉记忆模块**，在 token 进入 LLM 之前先做因果、分层、自适应压缩。

## 方法：Adaptive Hierarchical Memory
<figure class="figure">
  <img src="assets/papers/fluxmem/framework.png" alt="FluxMem 自适应层级记忆框架" loading="lazy" />
  <figcaption>图 2：FluxMem 的短期、中期、长期记忆级联结构。新帧先进入短期记忆；短期溢出后经 TAS 筛选进入中期；中期溢出后经 SDC 合并进入长期。</figcaption>
</figure>

### 1. 三层记忆：按时间远近决定压缩强度
FluxMem 将视觉上下文组织成三类记忆：

- **Short-term Memory**：保存最近帧的完整视觉 token，不做压缩。它负责当前 query 的即时感知和细粒度 grounding。
- **Mid-term Memory**：保存从短期记忆溢出的帧，但只保留相邻帧之间变化明显的 token。它关注“发生了什么变化”。
- **Long-term Memory**：保存更早的历史 token，并进一步把空间上相似、相邻的区域合并成 anchor。它关注“长期背景和关键事件的紧凑表示”。

这个设计符合视频流的时间尺度：越近的画面对回答越关键，越远的历史越应该压缩；但长期记忆仍要保留全局上下文，不能像普通 sliding window 一样直接丢掉。

### 2. TAS：Temporal Adjacency Selection
TAS 用于短期记忆到中期记忆的过渡。直觉上，如果某个空间位置的 token 和前后相邻帧都很相似，它很可能只是静态背景或重复信息；如果它和前一帧或后一帧差异明显，就更可能对应动作、物体变化或重要事件。

论文对每个 token 计算它与相邻帧 3×3 局部邻域的最小余弦距离，分别得到 backward score 和 forward score。只要 token 相对过去或未来任一方向足够“新”，就被保留下来进入中期记忆。

这个 3×3 邻域比较很实用：它不需要光流，也能容忍轻微运动、摄像机抖动和局部错位。

### 3. SDC：Spatial Domain Consolidation
SDC 用于中期记忆到长期记忆的过渡。经过 TAS 后，长期候选 token 已经更偏向动态区域，但空间上仍可能存在大量相邻重复 token。SDC 在原始 3×3 空间邻域内建立稀疏图，把距离足够近的 token 连接起来，再用 union-find 找连通分量，最后用每个连通分量的均值 anchor 替代整组 token。

换句话说，TAS 解决的是 **时间冗余**，SDC 解决的是 **空间冗余**。两者级联后，模型既能保留变化事件，也能把长期历史变成更紧凑的视觉摘要。

### 4. Otsu 自适应阈值：不手调压缩率
很多 token pruning/merging 方法依赖固定比例或手工阈值，这在流式视频里很脆弱：运动激烈时需要保留更多 token，静态场景则应该更激进压缩。

FluxMem 用 Otsu 方法从当前帧的相似度/距离分布中自动寻找阈值。对于 TAS，它根据时间相邻 token 的距离分布决定哪些 token 代表变化；对于 SDC，它根据空间邻域 token 的距离分布决定哪些区域可以合并。

这个设计的好处是：压缩强度来自视频自身统计，而不是外部超参数。论文强调它能在高动态场景提高阈值、在静态场景降低阈值，从而更稳地处理不同视频内容。

### 5. Proactive Response Triggering
FluxMem 还复用了 TAS 的 backward score 做主动响应触发。当某一帧中超过自适应阈值的 token 占比足够高，说明画面发生明显变化，系统可以触发模型输出。

这点很有工程意义：主动响应不需要额外训练一个触发器，也不需要额外视觉模块，而是直接复用记忆压缩时已经计算出的变化统计。

## 数据与评测
论文在 Qwen2.5-VL-7B 上实现 FluxMem，并覆盖在线和离线两类视频理解任务。

### 在线评测
- **OVO-Bench**：面向 streaming video 的时间戳理解，覆盖历史检索、实时感知和主动响应等能力。
- **StreamingBench real-time**：评估实时视觉、上下文和多源信息理解，论文重点报告 real-time 子任务。

### 离线评测
- **VideoMME without subtitles**：综合视频理解评测。
- **MLVU**：长视频多任务理解，覆盖分钟到小时级视频。
- **LongVideoBench**：长上下文视频语言推理。

### 实现设置
在线场景下，论文以 1 fps 模拟实时输入，每帧最多 256 个视觉 token，每个视频最多 256 帧；短期记忆长度为 8 帧，中期记忆长度为 64 帧，剩余帧进入长期记忆。离线场景同样使用 1 fps，但每帧 token 减到 64，最大序列长度为 1024 帧。

## 实验结果
### 在线视频理解：免训练也能超过多种在线模型
在实时任务上，FluxMem 相比 Qwen2.5-VL 基线有稳定提升：

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>OVO-Bench real-time</th><th>OVO-Bench overall</th><th>StreamingBench real-time</th></tr></thead>
    <tbody>
      <tr><td>Qwen2.5-VL-7B</td><td>63.3</td><td>49.8</td><td>73.9</td></tr>
      <tr><td>FluxMem</td><td><strong>67.2</strong></td><td><strong>53.3</strong></td><td><strong>76.4</strong></td></tr>
      <tr><td>提升</td><td>+3.9</td><td>+3.5</td><td>+2.5</td></tr>
    </tbody>
  </table>
</div>

更细的任务上，FluxMem 在需要短期线索和稳定上下文的任务中收益明显，例如 StreamingBench 的 Prospective Reasoning 提升 +6.5、Spatial Understanding 提升 +3.3；OVO-Bench 的 Action Recognition 提升 +6.4、Object Recognition 提升 +5.9。

### 离线长视频理解：压缩 token 的同时提升精度
虽然 FluxMem 是为 streaming 场景设计的，但它在离线长视频上也有效：

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>VideoMME</th><th>MLVU</th><th>LongVideoBench</th></tr></thead>
    <tbody>
      <tr><td>Qwen2.5-VL-7B</td><td>63.3</td><td>67.9</td><td>60.7</td></tr>
      <tr><td>FluxMem</td><td><strong>65.3</strong></td><td><strong>73.1</strong></td><td><strong>61.1</strong></td></tr>
      <tr><td>提升</td><td>+2.0</td><td>+5.2</td><td>+0.4</td></tr>
    </tbody>
  </table>
</div>

其中 MLVU 的提升最显著，说明层级记忆不仅是加速技巧，也能帮助模型从长视频中保留更有用的全局信息。

### Efficiency：部署成本明显降低
FluxMem 的一个关键卖点是效率。论文为了隔离记忆机制的贡献，预先提取视觉特征后统计 LLM 侧成本：

<div class="table-wrap">
  <table>
    <thead><tr><th>数据集</th><th>方法</th><th>Latency</th><th>Peak GPU Memory</th><th>性能</th></tr></thead>
    <tbody>
      <tr><td>MLVU</td><td>Baseline</td><td>3614 ms</td><td>41.3 GB</td><td>67.9</td></tr>
      <tr><td>MLVU</td><td>FluxMem</td><td><strong>2014 ms (-44.3%)</strong></td><td><strong>28.4 GB (-31.2%)</strong></td><td><strong>73.1</strong></td></tr>
      <tr><td>OVO-Bench</td><td>Baseline</td><td>2701 ms</td><td>35.8 GB</td><td>49.8</td></tr>
      <tr><td>OVO-Bench</td><td>FluxMem</td><td><strong>812 ms (-69.9%)</strong></td><td><strong>23.5 GB (-34.5%)</strong></td><td><strong>53.3</strong></td></tr>
    </tbody>
  </table>
</div>

在线更新的额外开销也很小：每帧总计约 4.1 ms，其中 TAS 1.3 ms，SDC 2.4 ms，其他部分 0.4 ms。

## Ablation：哪些设计最关键？
### 1. 层级记忆比单一记忆更稳
论文分别启用短期、中期、长期记忆，结论很清晰：

- **只用短期记忆**：不压缩，冗余大，长程信息利用不足。
- **只用长期记忆**：压缩强，但容易丢失细粒度近期线索。
- **中期 + 长期**：MLVU 上达到 73.1，说明 TAS 和 SDC 的互补性很强。
- **短期 + 中期 + 长期**：整体平均分最高，token drop ratio 约 64.3%，在准确率和效率之间最均衡。

### 2. FluxMem 的压缩策略优于常见 token reduction
<figure class="figure">
  <img src="assets/papers/fluxmem/ablation_method.png" alt="不同记忆压缩方法的消融对比" loading="lazy" />
  <figcaption>图 3：在 MLVU 上比较不同 token reduction 方法。FluxMem 在 50%–70% 这类实用压缩区间表现更好，即使 drop ratio 到 85% 仍保持较强性能。</figcaption>
</figure>

论文对比了 FIFO、Uniform、Random、DTD 等策略。所有方法大致都有一个共同趋势：适度丢 token 会提升效果，丢太多则下降。但 FluxMem 在 50%–70% 的实用压缩区间持续领先，在约 64% drop ratio 下 MLVU 达到 73.1；即便到 85% drop ratio，仍有 70.1。

这说明它不是简单“少看一些帧”，而是在更合理地选择哪些视觉信息应该进入长期上下文。

### 3. 自适应阈值比固定阈值更省 token
<figure class="figure">
  <img src="assets/papers/fluxmem/ablation_mid.png" alt="中期记忆自适应阈值消融" loading="lazy" />
  <figcaption>图 4：中期记忆中，自适应阈值能以更高 drop ratio 达到与最优固定阈值相当或更好的性能，说明它更能适配不同视频动态。</figcaption>
</figure>

<figure class="figure">
  <img src="assets/papers/fluxmem/ablation_long.png" alt="长期记忆自适应阈值消融" loading="lazy" />
  <figcaption>图 5：长期记忆中，自适应阈值在保持性能的同时实现更强压缩，反映出 SDC 对空间冗余的稳定建模能力。</figcaption>
</figure>

在中期记忆里，最优固定阈值在 29.4% drop ratio 下达到 65.5，而自适应阈值在 42.8% drop ratio 下达到 65.6。长期记忆里，最优固定阈值在 63.8% compression 下达到 65.6，自适应阈值则在 84.5% drop ratio 下保持 65.4。

这组实验支持了论文的核心主张：**阈值应该来自当前视频统计，而不是手工固定规则**。

### 4. Training-free 与 SFT 可以叠加
FluxMem 本身不需要训练，但论文也测试了小规模 SFT 后的效果：

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>OVO-Bench overall</th><th>StreamingBench real-time</th></tr></thead>
    <tbody>
      <tr><td>Qwen2.5-VL</td><td>49.8</td><td>73.9</td></tr>
      <tr><td>+ FluxMem</td><td>53.3</td><td>76.4</td></tr>
      <tr><td>+ FluxMem + SFT</td><td><strong>61.4</strong></td><td><strong>76.7</strong></td></tr>
    </tbody>
  </table>
</div>

这说明 FluxMem 既可以作为即插即用的推理时增强，也可以作为后续在线视频 SFT 的基础记忆结构。

## 我的理解与启发
FluxMem 最有价值的地方不是某个单独公式，而是把“视频记忆”拆成了更符合工程直觉的多层缓存系统：

1. **近期不要省**：短期画面直接保留，保证实时 grounding。
2. **中期看变化**：相邻帧差异决定哪些 token 进入中期。
3. **长期做摘要**：空间相似区域合并成 anchor，降低长期上下文成本。
4. **压缩率随场景变**：用 Otsu 从 token 分布自动找阈值，减少手调。
5. **触发器复用统计量**：场景变化检测和记忆压缩共享计算，工程上很经济。

这对实时多模态 Agent 很有启发：长视频理解并不一定要依赖更长上下文窗口或更大模型，很多时候需要的是一个合理的在线状态表示。FluxMem 本质上是在回答：如果只能保留有限视觉记忆，应该按什么层级、什么尺度、什么阈值保留？

## 局限与待观察点
- **依赖视觉 token 相似度质量**：如果视觉编码器的 token 表征不稳定，TAS/SDC 的距离度量也会受影响。
- **阈值自适应不是任务自适应**：Otsu 能根据视频统计调整压缩，但不知道用户 query 的具体需求；对于查询相关性很强的任务，query-guided memory 可能仍有价值。
- **主动响应触发仍有超参数**：场景切换比例阈值 gamma 仍需要设置，不同应用对“何时说话”的容忍度不同。
- **评测主要基于 1 fps 设置**：更高帧率、更长时间运行和真实机器人/智能眼镜部署中的表现还需要进一步验证。
- **空间合并可能损失细节**：对 OCR、小目标、精细位置关系等任务，过强 SDC 可能带来信息损失。

## 结论
FluxMem 给流式视频理解提供了一条非常务实的路线：在不训练新模型的前提下，通过 **短期完整保留、中期 TAS 时间筛选、长期 SDC 空间合并、Otsu 自适应阈值**，把高冗余视觉流压缩成更紧凑、更适合 LLM 使用的层级记忆。它的意义不只是提升 benchmark，更是证明了合理的在线记忆设计可以同时改善准确率、延迟和显存。