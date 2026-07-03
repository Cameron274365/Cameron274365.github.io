---
id: "flash-vstream-efficient-real-time-long-video-streams"
title: "Flash-VStream：面向长视频流的高效实时理解"
category: "Streaming VLM"
date: "2026-06-04"
order: 20
readTime: "11 min"
tags: ["VLM","Long Video","Video Stream","Memory","Real-Time","Qwen2-VL"]
summary: "Flash-VStream 用异步双进程框架和固定大小的 Flash Memory，把长视频流处理与问题回答解耦：CSM 聚合长程时序信息，DAM 按信息密度取回关键帧细节，在 11520 个视频 token 预算内实现一秒级响应，并在 EgoSchema、MLVU、LVBench、MVBench、Video-MME 上取得强结果。"
---

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
  <img src="assets/papers/flash-vstream/x1_framework.webp" alt="Flash-VStream 双进程框架和 Flash Memory 总览" loading="lazy" />
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
  <img src="assets/papers/flash-vstream/x3_memory_config.webp" alt="Flash Memory 默认配置" loading="lazy" />
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
  <img src="assets/papers/flash-vstream/x5_main_results.webp" alt="Flash-VStream 在多个视频理解 benchmark 上的主结果" loading="lazy" />
  <figcaption>图 3：主结果表。Flash-VStream 在 11520 视频 token 成本下达到 EgoSchema 68.2、MLVUdev 66.3、LVBench 42.0、MVBench 65.4、Video-MME 61.2/67.0。</figcaption>
</figure>

### 2. 效率：一秒实时阈值下保持更好准确率
论文的速度测试显示，7B 模型要满足一秒首 token 响应，视频 token 数应控制在约 **12000** 以内。许多长视频模型依赖更大的 token 预算，例如 LongVILA、LongVA、Qwen2-VL 官方设置等，因此延迟超过一秒。

Flash-VStream 的优势不是单纯减少 token，而是通过 CSM + DAM 让固定 token 预算更有信息量：同样 11520 个 token，它比均匀截断/在线版本更准确。

<figure class="figure">
  <img src="assets/papers/flash-vstream/x2_latency.webp" alt="Flash-VStream 响应延迟与准确率对比" loading="lazy" />
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
  <img src="assets/papers/flash-vstream/x6_ablation.webp" alt="Flash Memory 组件消融实验" loading="lazy" />
  <figcaption>图 5：Flash Memory 组件消融。CSM 和 DAM 的组合最优；只用均匀采样或移除关键组件都会降低平均准确率。</figcaption>
</figure>

### 2. 记忆容量分配：约三分之一给 CSM 最合适
在固定总 token 预算下，论文网格搜索 CSM/DAM 容量比例和空间池化比例。结果显示：当 pool ratio = 4，且约 **1/3 容量给 CSM、2/3 容量给 DAM** 时，EgoSchema 与 MVBench 表现最好。

这很符合直觉：CSM 太少会丢长期时序，DAM 太少会缺空间细节；二者不是替代关系，而是互补关系。

<figure class="figure">
  <img src="assets/papers/flash-vstream/x4_capacity.webp" alt="CSM 容量比例与空间池化比例消融" loading="lazy" />
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
  <img src="assets/papers/flash-vstream/x7_case.webp" alt="Flash Memory 可视化与问答案例" loading="lazy" />
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
