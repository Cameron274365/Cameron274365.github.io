---
id: "hermes-kv-cache-hierarchical-memory"
title: "HERMES：把 KV Cache 当作层次化记忆的流式视频理解框架"
category: "Streaming VLM"
date: "2026-06-08"
order: 10
readTime: "13 min"
tags: ["VLM","Streaming Video","KV Cache","Memory","Token Compression","Real-time"]
summary: "HERMES 将 MLLM 的 KV cache 解释为分层视频记忆：浅层保留近期感知，中层承担工作记忆，深层存储长期语义锚点。它无需训练，通过层次化缓存管理、跨层记忆平滑和位置重索引，在流式视频理解中实现低延迟、低显存和稳定准确率。"
---

## 一句话总结
HERMES 的核心观点是：**KV cache 不只是推理加速缓存，也可以被当作模型内部的层次化视频记忆来管理**。论文通过注意力机制分析发现，MLLM 不同 decoder 层天然承担不同时间粒度的记忆功能；据此设计训练无关的 KV cache 压缩与重用策略，使模型在连续视频流中保持固定显存、实时响应和较稳定的理解能力。

<div class="metric-grid">
  <div class="metric"><strong>Training-free</strong><span>无需重新训练，作为即插即用缓存管理方法接入现有 MLLM</span></div>
  <div class="metric"><strong>10×</strong><span>相对先前 SOTA 的 TTFT 加速</span></div>
  <div class="metric"><strong>68%</strong><span>最多减少的视频 token 比例</span></div>
  <div class="metric"><strong>+11.4%</strong><span>流式数据集上相对基础模型的最高准确率提升</span></div>
</div>

## 论文信息
- **标题**：HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding
- **作者**：Haowei Zhang, Shudong Yang, Jinlan Fu, See-Kiong Ng, Xipeng Qiu
- **会议/版本**：ACL 2026 Main；arXiv v4，2026-05-07
- **链接**：[arXiv:2601.14724](https://arxiv.org/abs/2601.14724) · [PDF](https://arxiv.org/pdf/2601.14724)
- **代码**：arXiv 页面未显示官方代码链接

<figure class="figure">
  <img src="assets/papers/hermes/x1_teaser.webp" alt="HERMES 方法、注意力分析与效率结果总览" loading="lazy" />
  <figcaption>图 1：HERMES 总览。左侧展示训练无关的层次化 KV cache 管理；中间是不同层注意力偏好的机制分析；右侧显示在流式输入中保持低 TTFT 与稳定 GPU 显存。</figcaption>
</figure>

## 背景：为什么流式视频理解难？
离线视频理解通常可以一次性采样视频帧，再把视觉 token 和问题一起输入模型。但真实流式场景更复杂：视频不断到达，用户问题可能在任意时刻出现，系统还要同时满足三类约束。

- **理解要稳定**：不能因为视频越来越长，就遗忘早期关键事件或在长程问题上退化。
- **响应要实时**：用户提问后不能再临时检索、重编码或重新 prefill 大量历史内容。
- **显存要可控**：视频流理论上无限长，缓存不能随时间线性增长。

已有方法大致分两类：外部记忆方法会把历史视频存成 caption、patch 或数据库条目，问题到来时再检索；内部缓存方法则直接管理 KV cache。前者灵活但查询延迟高，后者更接近实时系统需求。HERMES 选择后者，并进一步提出：**不同 transformer 层里的 KV cache 本身就对应不同粒度的记忆**。

## 方法：KV Cache as Hierarchical Memory
### 1. 机制发现：不同层天然存不同粒度的记忆
论文先在 LLaVA-OV 上做注意力可视化：从 VideoMME 的短、中、长视频中各采样 100 个 video-question pair，以 0.5 fps 流式输入；每帧编码为 196 个视觉 token，每个 chunk 8 帧，并在每层维护 6K 视频 token 的 FIFO KV cache。

观察结果很关键：

- **浅层是 Sensory Memory**：例如 layer 0 强烈关注最近视觉 token，注意力随时间距离快速衰减，像短暂的感知缓冲区。
- **深层是 Long-term Memory**：例如 layer 26 的注意力不再偏向最近 token，而是呈现稀疏、周期性的峰值；峰值间隔正好约为 196 token，对应每帧的 frame-level anchor token。
- **中层是 Working Memory**：例如 layer 8 处于过渡状态，一方面保留近期偏好，另一方面开始形成类似深层的帧级语义锚点。

这个发现使缓存管理不再只是“保留最近 token”或“按全局注意力排序”，而是可以按层设计不同策略。

<figure class="figure">
  <img src="assets/papers/hermes/x2_framework.webp" alt="HERMES 层次化 KV cache 管理框架" loading="lazy" />
  <figcaption>图 2：HERMES 架构。视频流按 chunk 进入模型，KV cache 在不同层采用不同的重要性估计与压缩策略；用户提问时直接复用已有紧凑缓存，无需额外检索或外部记忆计算。</figcaption>
</figure>

### 2. Hierarchical KV Cache Management
HERMES 将 decoder 层划分为三段：前 10% 为浅层，中间 60% 为中层，后 30% 为深层。每个视频 token 在第 \(l\) 层都有一个重要性分数 \(S_i^l\)，用于决定是否保留。

- **浅层：按遗忘曲线保留近期 token**  
  浅层有强 recency bias，因此用类似 Ebbinghaus 遗忘曲线的指数衰减建模重要性：越新的 token 越重要。

- **深层：按注意力保留长期语义锚点**  
  深层关注稀疏的 frame-level anchor token，因此直接用 guidance prompt 对视频 token 的注意力权重估计长期重要性。这里使用通用伪查询，是为了在真实用户问题尚未到来时，也能提前完成缓存压缩。

- **中层：在近期性和注意力之间插值**  
  中层从感知记忆过渡到长期记忆，因此使用层相关权重 \(\omega^l\) 在 recency score 和 attention score 之间插值；层越深，越依赖注意力而不是单纯近期性。

### 3. Cross-Layer Memory Smoothing
层次化策略带来一个新问题：同一个物理缓存位置上的 token，可能在不同层被不同步地保留或淘汰，导致跨层语义不一致。HERMES 用跨层记忆平滑缓解这个问题：第 \(l\) 层的 token 分数不仅看本层 \(S_i^l\)，还混合下一层 \(S_i^{l+1}\) 的信号。

直观理解是：**深层长期记忆会反向提醒浅层和中层哪些 token 可能具有更长期价值**。之后每层根据平滑后的分数做 Top-K 选择，使每层维持固定预算 \(|M|\)。

此外，HERMES 不直接丢弃所有被淘汰 token，而是在深层把它们聚合成 summary token，作为紧凑长期记忆保留在 KV cache 中。

### 4. Position Re-Indexing
流式视频会让位置编号不断增长，超过模型训练时见过的范围，进而破坏 RoPE/M-RoPE 的位置分布。HERMES 提供两种重索引策略：

- **Lazy Re-Indexing**：位置编号接近模型上限时才重排。它保留近期 token 的原始位置，更适合流式视频理解，开销也更低。
- **Eager Re-Indexing**：每次压缩都重排到连续区间 \([0, |M|)\)。它更稳定地维护长程视觉语义，但频繁重排成本更高，更适合离线视频。

论文在流式评测中默认使用 lazy re-indexing，在离线评测中默认使用 eager re-indexing。

## 数据与评测设置
论文同时评估流式和离线视频理解。

### 流式评测
- **StreamingBench**：实时理解子集，平均 10.1 分钟，500 个视频，2,500 个多选 QA。
- **OVO-Bench**：包含 backward tracing 和 real-time 两类子集，平均 5.9/8.8 分钟。
- **RVS-Ego / RVS-Movie**：开放式流式问答，分别约 60 分钟和 30 分钟视频，使用 GPT-3.5-turbo-0125 评估答案 accuracy 和 1–5 分 score。

### 离线评测
- **MVBench**：短视频多选评测，平均 16 秒。
- **Egoschema**：长视频多选评测，平均 3 分钟。
- **VideoMME**：更长视频评测，平均 17 分钟，不使用字幕。

### 模型与实现
论文验证了多个模型族与规模，包括 LLaVA-OneVision、Qwen2.5-VL，以及扩展实验中的 Qwen3-VL。视频按 chunk 流式进入，每个 chunk 默认 16 帧；缓存超过预算后触发压缩。效率测试使用单张 A800 80GB GPU，FP16 混合精度，生成采用 greedy decoding。

## 实验结果
### 1. 流式多选任务：HERMES 在多个模型上稳定提升
在 StreamingBench 和 OVO-Bench 上，HERMES 明显优于基础模型和其他 training-free 方法。

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>设置</th><th>StreamingBench RT</th><th>OVO RT</th><th>OVO BW</th><th>OVO Avg.</th></tr></thead>
    <tbody>
      <tr><td>LLaVA-OV-7B</td><td>64 frames</td><td>71.34</td><td>63.06</td><td>43.64</td><td>53.35</td></tr>
      <tr><td>+ HERMES</td><td>4K tokens</td><td><strong>73.23</strong></td><td><strong>66.34</strong></td><td><strong>50.20</strong></td><td><strong>58.27</strong></td></tr>
      <tr><td>Qwen2.5-VL-7B</td><td>1 fps</td><td>73.31</td><td>59.90</td><td>44.65</td><td>52.28</td></tr>
      <tr><td>+ HERMES</td><td>4K tokens</td><td><strong>79.44</strong></td><td><strong>68.98</strong></td><td><strong>49.43</strong></td><td><strong>59.21</strong></td></tr>
      <tr><td>Qwen2.5-VL-32B</td><td>1 fps</td><td>74.27</td><td>64.40</td><td>50.33</td><td>57.37</td></tr>
      <tr><td>+ HERMES</td><td>6K tokens</td><td><strong>80.20</strong></td><td><strong>71.93</strong></td><td><strong>57.71</strong></td><td><strong>64.82</strong></td></tr>
    </tbody>
  </table>
</div>

一个值得注意的点是，HERMES 在 4K token 预算下通常已经很强。对于流式视频，关键不是无限保留所有历史，而是用固定预算保留多粒度信息。

### 2. 开放式 RVS：细粒度时空理解更好
在 RVS-Ego 和 RVS-Movie 上，HERMES 也超过多数 training-free 方法。以 LLaVA-OV-7B 为例，基础模型在 RVS-Ego 上 accuracy 为 56.2，HERMES 6K 达到 60.3；RVS-Movie 从 43.0 提升到 54.4。论文指出，相对统一采样 64 帧的基础模型，HERMES 在流式数据集上最高带来 **11.4%** 的准确率提升。

<figure class="figure">
  <img src="assets/papers/hermes/x5_temporal_case.webp" alt="HERMES 细粒度时间理解案例" loading="lazy" />
  <figcaption>图 3：RVS temporal case。HERMES 相比基础模型能更好利用历史缓存中的细粒度时间信息，回答涉及事件顺序和长期上下文的问题。</figcaption>
</figure>

### 3. 离线任务：压缩后仍保持竞争力
在离线 benchmark 上，HERMES 在有限 token 预算下仍能达到与基础模型相当或更好的表现。LLaVA-OV-7B + HERMES 4K 在 VideoMME Avg. 上达到 58.85，高于基础模型 57.67；Egoschema 达到 60.29，略高于基础模型 59.93；MVBench 则基本持平。

这说明 HERMES 不只是为流式场景定制的工程缓存策略，也能作为长视频 token 压缩方法迁移到离线理解。

### 4. 效率：用户提问时无需额外检索
HERMES 最大的工程价值在 TTFT 和显存稳定性。由于视频流到达时已经持续压缩并维护紧凑 KV cache，用户提问时不需要临时检索外部存储，也不需要重新 prefill 历史视频。

<figure class="figure">
  <img src="assets/papers/hermes/x3_efficiency.webp" alt="HERMES GPU 显存和 TTFT 效率对比" loading="lazy" />
  <figcaption>图 4：效率对比。HERMES 在输入帧数增加时保持稳定 GPU 显存和低 TTFT；在 256 帧设置下，相比 LiveVLM 峰值显存降低约 1.04×，相比 StreamingTOM 的 TTFT 快约 10×。</figcaption>
</figure>

论文进一步报告，在 4K token 预算下，chunk size 为 8 时，从 16 到 512 帧 GPU 显存基本维持在 16.54–16.66 GB，TTFT 约 27–28 ms，TPOT 约 24 ms；chunk size 为 16 时，显存约 17.46–17.66 GB，TTFT 也稳定在约 27–29 ms。

## Ablation：哪些设计最关键？
### 总缓存预算
LLaVA-OV-7B 的预算实验显示，HERMES 在 4K token 后基本稳定；流式数据集对较小预算更宽容，而长离线数据集在低于 4K 时明显退化。这符合直觉：离线长视频问题可能依赖更广范围的信息，而流式实时问题通常更看重最近和局部历史。

<figure class="figure">
  <img src="assets/papers/hermes/x4_budget_llava.webp" alt="HERMES 在 LLaVA-OV 上的缓存预算消融" loading="lazy" />
  <figcaption>图 5：总缓存预算消融。4K 左右成为较重要的稳定点；更低预算会损害长视频理解，尤其是 VideoMME 这类长离线任务。</figcaption>
</figure>

### 层间预算分配
论文还测试了浅层、中层、深层不同预算权重。整体性能差异不大，说明 HERMES 对具体分配不太敏感；但给深层更多预算时，VideoMME long subset 的结果更好，支持“深层承担长期记忆”的解释。

### Cross-Layer Memory Smoothing
去掉跨层平滑会降低 VideoMME 表现。默认 \(\lambda_{deep}=0.4, \lambda_{mid}=0.3, \lambda_{shallow}=0.1\) 时，VideoMME Avg. 达到 58.44，高于无平滑的 54.74。这说明层与层之间的记忆一致性不是细节，而是稳定长视频理解的重要因素。

### Summary Token
深层加入 summary token 后，VideoMME long 从 47.78 提升到 49.11，Avg. 从 57.96 提升到 58.44。它的作用类似把被淘汰 token 的长期残差信息压缩进一个紧凑槽位，避免“硬删除”。

### Position Re-Indexing
流式场景中 lazy re-indexing 更好：StreamingBench RT 达到 72.63，高于 eager 的 72.30；OVO Avg. 也从 56.06 提升到 56.94。离线场景则相反，eager re-indexing 在 VideoMME Avg. 达到 58.44，高于 lazy 的 54.93。这个差异说明位置策略要按任务形态选择，而不是一刀切。

## 我的理解与启发
HERMES 最有价值的地方，不只是提出了一个新的缓存压缩算法，而是给出了一个很有解释力的系统视角：**缓存管理应该尊重模型内部层级的功能分工**。

1. **从“token 重要性”走向“记忆层级”**  
   许多压缩方法默认所有层共享同一套重要性标准，但 HERMES 说明浅层、中层、深层对视频 token 的使用方式不同。以后做长上下文或流式 Agent，也许应该更多考虑“分层状态管理”。

2. **训练无关方法仍然可以有机制依据**  
   HERMES 没有重新训练模型，但不是简单启发式裁剪，而是先分析注意力机制，再映射到缓存策略。这种路径适合快速适配已有 MLLM。

3. **实时交互的关键是 query 前完成状态维护**  
   很多外部记忆方案在用户提问时才检索、组装、prefill，上线体验容易被 TTFT 卡住。HERMES 把压缩和状态更新放在视频流到达阶段，用户提问时直接使用 compact KV cache，更符合实时助手形态。

4. **位置编码是流式系统的隐形约束**  
   只压缩 token 不够，剩余 token 的位置分布也要稳定。lazy/eager re-indexing 的差异提醒我们：流式系统里，RoPE/M-RoPE 不是实现细节，而会直接影响长期稳定性。

## 局限与待观察点
- **没有验证超大模型**：论文说明受计算资源限制，未在 Qwen2.5-VL-72B 等 72B 级模型上实验。
- **与其他 training-free 技术的组合尚未探索**：HERMES 可能还能与其他压缩、检索或缓存策略叠加，但论文没有系统验证。
- **guidance prompt 仍是近似查询**：用户真实问题不可预测，用通用伪查询提前估计深层 token 重要性很实用，但在高度任务相关的问题上可能仍会漏掉关键细节。
- **summary token 的信息瓶颈需要更多分析**：聚合被淘汰 token 能改善长期记忆，但不同视频类型、不同事件密度下该聚合是否稳定，还值得进一步研究。
- **主要验证集中在视觉问答**：音频、语音、多轮主动交互等更完整的实时助手场景还没有成为核心实验对象。

## 结论
HERMES 把 KV cache 重新定义为 MLLM 的层次化视频记忆：浅层维护近期感知，中层融合工作上下文，深层保留长期语义锚点。通过层次化缓存管理、跨层记忆平滑和位置重索引，它在不训练模型、不依赖外部检索的前提下，实现了固定显存、低 TTFT 和较强的流式视频理解能力。对实时多模态 Agent 来说，这篇论文的启发是：**与其只追求更长上下文，不如把在线状态设计成可持续维护、可解释、低延迟的内部记忆系统**。