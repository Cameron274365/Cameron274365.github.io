---
id: light-omni-reflex-over-reasoning-agentic-video
title: "Light-Omni：用反射取代推理的智能体长视频理解"
category: 多模态 RAG
date: 2026-07-06
order: 1
readTime: 10 min
tags: ["Agentic Video Understanding", "Long-Term Memory", "Multimodal LLM", "Reflexive Agent", "Memory Consolidation"]
summary: "Light-Omni 提出一种面向长时程多模态视频理解的记忆增强智能体框架。它通过双上下文状态——全局状态 Sg（非参数化的分层 episodic 记忆脚本）和潜状态 Sl（由可学习 soft prompt 生成的检索嵌入与动作控制信号）——把传统侦探式迭代推理压缩为单次前向传播的反射式响应。实验表明，Light-Omni 在 VideoMME-long 和 LVBench 上平均准确率达 58.0%，相比 M3-Agent 提升 2.4%，同时实现 12.1 倍加速与 2.6 倍显存降低；作为记忆模块接入 Qwen2.5-VL-7B、Qwen3-VL-8B、Gemini-2.0-Flash 后也能稳定提升性能并降低延迟。"
---

## 一句话总结

Light-Omni 用**全局上下文 + 潜状态检索嵌入**把智能体视频理解从“反复推理找证据”变成“一次前向反射式响应”，在准确率、延迟和显存上同时击败了迭代式 agent 基线。

## 论文信息

- **标题**：Light-Omni: Reflex over Reasoning in Agentic Video Understanding with Long-Term Memory
- **作者**：Chang Nie、Jiaju Wei、Junlan Feng、Chaoyou Fu、Caifeng Shan
- **机构**：Nanjing University
- **发表**：arXiv:2607.05511 [cs.CV]，2026-07-06
- **链接**：[arXiv](https://arxiv.org/abs/2607.05511) · [HTML](https://arxiv.org/html/2607.05511v1)

## 背景：为什么“推理”成了瓶颈？

现有的智能体视频理解系统通常采用**侦探式工作流**：多轮推理拆解查询、改写检索意图、反复调用外部工具（搜索、ASR、grounding）来聚合证据。这种方式虽然准确，但带来三个问题：

1. **延迟高**：每次查询都要经过多次模型调用和工具交互；
2. **显存大**：长视频全部特征或 KV cache 都要保留；
3. **依赖外部 API**：系统脆弱且成本高。

作者认为，这种重推理本质上是在补偿两件事的缺失：

- **缺乏显式的全局上下文**：没有持续、整体的过去观测表示；
- **查询与记忆分布不对齐**：检索 embedding 和查询语义不在同一空间，需要反复改写/迭代来 bridge gap。

## 方法：双上下文状态 + 反射式响应

Light-Omni 的整体架构可以概括为：

<figure class="figure"><img src="assets/papers/light-omni-reflex-over-reasoning-agentic-video/x2.webp" alt="Light-Omni 整体架构" loading="lazy" /><figcaption>图 2：Light-Omni 整体架构。上半部分为实时反射（Real-time Reflex）：根据全局状态 Sg 和当前输入生成潜状态 Sl，用于动作控制和检索；下半部分为睡眠时巩固（Sleep-time Consolidation）：通过 CRUD 和分层聚合异步维护长期记忆并刷新 Sg。</figcaption></figure>

### 1. 多模态长期记忆

记忆 $\mathcal{M}$ 被拆成三部分：

- **User Profile** $\mathcal{M}_p$：用户头像、偏好、性格等长期属性；
- **Semantic Memory** $\mathcal{M}_s$：以文本 key-value 形式存储的抽象事实、概念和关系；
- **Episodic Memory** $\mathcal{M}_e$：按时间顺序记录的事件脚本，包含时间戳、视觉场景描述、听觉线索和助手回复。

对于每个时间步 $t$ 的多模态输入 $\mathcal{I}^t = \{V^t, A^t, T^t\}$，系统直接生成当前片段的 topic（语义事实 + 事件脚本）。在线场景下，记忆构建在**睡眠时异步**完成，不阻塞实时交互。

### 2. 全局状态 Sg：非参数化的“大脑”

$\mathcal{S}_g$ 不是模型参数，而是从 episodic 记忆中**分层合并**出来的一个紧凑脚本：

- 当某层的节点数达到 $k+1$（默认 $k=8$）时，将最老的 $k$ 个节点合并成上一层的一个节点；
- 最近细节保留在底层，远期事件逐层抽象；
- 这样既保证时序连续性，又把上下文窗口控制在固定大小。

### 3. 潜状态 Sl：一次前向同时做动作和检索

在 $\mathcal{S}_g$ 的基础上，模型接收当前输入和一组**可学习 soft prompt** $\boldsymbol{P}_{soft}$，通过单次前向得到隐藏特征 $\boldsymbol{H}^t$，再用两个独立头解码：

- **动作头** $\boldsymbol{a}_{act}^t$：通过 Bernoulli 采样决定执行哪些动作（如 `search`、`speech`）；
- **检索嵌入头** $\boldsymbol{z}_{ret}^t$：把隐藏状态投影到检索空间，用于从记忆中召回相关条目。

关键设计是：$\boldsymbol{z}_{ret}^t$ 与原查询 embedding **逐元素相加**，直接在连续潜空间里修正查询表示，使其与记忆分布对齐，从而省去显式的 query rewrite。

### 4. 训练策略

- **记忆与生成**：用标准 Next Token Prediction 训练对应 LoRA adapter；
- **反应（动作 + 检索）**：用混合目标训练：

$$
\mathcal{L}_2 = \sum_{c \in \mathcal{C}} \mathcal{L}_{CE}(\boldsymbol{p}_c, \boldsymbol{y}_c) - \lambda \log \frac{\exp(\boldsymbol{z}_{ret} \cdot \boldsymbol{k}^+ / \tau)}{\sum_{j=1}^{B} \exp(\boldsymbol{z}_{ret} \cdot \boldsymbol{k}_j / \tau)}
$$

其中 $\mathcal{C}=\{\mathtt{search}, \mathtt{speech}\}$，第二项是对比检索对齐损失。作者通过 **Multi-LoRA** 把 memorization、generation、reaction 三个能力解耦，避免优化冲突。

<figure class="figure"><img src="assets/papers/light-omni-reflex-over-reasoning-agentic-video/x1.webp" alt="范式对比与整体性能" loading="lazy" /><figcaption>图 1：左：侦探式迭代推理 vs. Light-Omni 的反射式单次前向；右：Light-Omni 在长视频基准上性能强劲，同时保持接近恒定的延迟。</figcaption></figure>

## 实验

### 数据集与基线

- **VideoMME-long**：小时级长视频理解；
- **LVBench**：长视频感知、视听事件理解与推理；
- **HippoVlog**、**OVO-Bench**：在线/真实交互场景。

基线包括：

- 通用 MLLM：GPT-4o、Gemini-2.0-Flash、Qwen2.5-Omni-7B、Qwen2.5-VL-7B/72B、Qwen3-VL-8B；
- 记忆增强方法：Naive RAG、RAG-Rewrite、MovieChat、HippoMM；
- 推理型 agent：Ego-R1、M3-Agent、WorldMM-8B。

默认 backbone 为 **Qwen2.5-Omni-7B**，检索器为 **Qwen3-Embedding-0.6B**，默认取 12 条 semantic memory 和 4 条 episodic memory。

### 主结果

<div class="table-wrap">
<table>
<thead>
<tr><th rowspan="2">Model</th><th colspan="3">VideoMME-long ↑/↓</th><th colspan="3">LVBench ↑/↓</th><th rowspan="2">Avg Acc. ↑</th></tr>
<tr><th>Acc.</th><th>Latency(s)</th><th>Mem(GB)</th><th>Acc.</th><th>Latency(s)</th><th>Mem(GB)</th></tr>
</thead>
<tbody>
<tr><td>GPT-4o</td><td>65.3</td><td>–</td><td>–</td><td>30.8</td><td>–</td><td>–</td><td>48.1</td></tr>
<tr><td>Gemini-2.0-Flash</td><td>63.0</td><td>–</td><td>–</td><td>48.6</td><td>–</td><td>–</td><td>55.8</td></tr>
<tr><td>Qwen2.5-Omni-7B</td><td>55.3</td><td>48.8</td><td>81.7</td><td>41.6</td><td>51.9</td><td>80.0</td><td>48.5</td></tr>
<tr><td>Qwen2.5-VL-7B</td><td>55.4</td><td>51.7</td><td>113.2</td><td>43.7</td><td>48.8</td><td>119.0</td><td>49.6</td></tr>
<tr><td>Qwen2.5-VL-72B</td><td>63.4</td><td>–</td><td>OOM</td><td>48.8</td><td>–</td><td>OOM</td><td>56.1</td></tr>
<tr><td>Qwen3-VL-8B</td><td>61.0</td><td>14.9</td><td>29.9</td><td>48.3</td><td>15.1</td><td>29.9</td><td>54.7</td></tr>
<tr><td>Naive RAG</td><td>58.4</td><td>7.3</td><td>27.6</td><td>46.9</td><td>6.9</td><td>27.7</td><td>52.7</td></tr>
<tr><td>RAG-Rewrite</td><td>60.8</td><td>8.5</td><td>27.7</td><td>47.6</td><td>7.4</td><td>27.9</td><td>54.2</td></tr>
<tr><td>MovieChat</td><td>19.4</td><td>–</td><td>30.7</td><td>22.5</td><td>–</td><td>31.1</td><td>21.0</td></tr>
<tr><td>HippoMM</td><td>41.6</td><td>16.1</td><td>36.3</td><td>38.2</td><td>14.3</td><td>34.8</td><td>39.9</td></tr>
<tr><td>M3-Agent</td><td>61.8</td><td>25.5</td><td>62.4</td><td>49.3</td><td>33.1</td><td>62.2</td><td>55.6</td></tr>
<tr><td><strong>Light-Omni (Ours)</strong></td><td><strong>66.1</strong></td><td><strong>2.2</strong></td><td><strong>24.0</strong></td><td><strong>49.9</strong></td><td><strong>2.6</strong></td><td><strong>24.2</strong></td><td><strong>58.0</strong></td></tr>
</tbody>
</table>
</div>

与基线 Qwen2.5-Omni-7B 相比，Light-Omni 获得 **+9.5% 准确率、20.5 倍加速、3.3 倍显存降低**；与推理型 agent M3-Agent 相比，获得 **+2.4% 准确率、12.1 倍加速、2.6 倍显存降低**。

### 作为通用记忆模块的迁移能力

<div class="table-wrap">
<table>
<thead>
<tr><th>Backbone + Memory</th><th>VideoMME-long Acc.</th><th>VideoMME Latency(s)</th><th>VideoMME Mem(GB)</th><th>LVBench Acc.</th><th>LVBench Latency(s)</th><th>LVBench Mem(GB)</th><th>Avg Acc.</th></tr>
</thead>
<tbody>
<tr><td>Qwen2.5-VL-7B</td><td>55.4</td><td>51.7</td><td>113.2</td><td>43.7</td><td>48.8</td><td>119.0</td><td>49.6</td></tr>
<tr><td>+ Light-Omni</td><td>59.6 (+4.2)</td><td>8.5 (6.1×)</td><td>46.7 (2.4×)</td><td>49.3 (+5.6)</td><td>6.8 (7.2×)</td><td>46.9 (2.5×)</td><td>54.5 (+4.9)</td></tr>
<tr><td>Qwen3-VL-8B</td><td>61.0</td><td>14.9</td><td>29.9</td><td>48.3</td><td>15.1</td><td>29.9</td><td>54.7</td></tr>
<tr><td>+ Light-Omni</td><td>65.1 (+4.1)</td><td>7.0 (2.1×)</td><td>46.3</td><td>49.3 (+1.0)</td><td>6.9 (2.2×)</td><td>46.9</td><td>57.2 (+2.5)</td></tr>
<tr><td>Gemini-2.0-Flash</td><td>63.0</td><td>–</td><td>–</td><td>48.6</td><td>–</td><td>–</td><td>55.8</td></tr>
<tr><td>+ Light-Omni</td><td>69.0 (+6.0)</td><td>–</td><td>–</td><td>50.1 (+1.5)</td><td>–</td><td>–</td><td>59.6 (+3.8)</td></tr>
</tbody>
</table>
</div>

把 Light-Omni 作为记忆系统接入现有 MLLM 后，三个不同规模的模型在准确率和效率上都有提升，说明它是一个可插拔的通用记忆模块。

### 与推理型 Agent 的效率对比

<div class="table-wrap">
<table>
<thead>
<tr><th>Model</th><th>Backbone</th><th colspan="3">Interactive Response (s)</th><th>Calls</th><th>Memorization Time (s)</th><th>Calls</th><th>Acc.</th></tr>
<tr><th></th><th></th><th>Retr.</th><th>Resp.</th><th>Total</th><th></th><th></th><th></th><th></th></tr>
</thead>
<tbody>
<tr><td>M3-Agent</td><td>Qwen3-32B</td><td>16.09</td><td>9.82</td><td>25.91</td><td>2.5</td><td>3312.78</td><td>81.2</td><td>61.8</td></tr>
<tr><td>WorldMM</td><td>Qwen3-VL-8B</td><td>–</td><td>–</td><td>≥20.00</td><td>≤5.0</td><td>–</td><td>–</td><td>66.0</td></tr>
<tr><td><strong>Light-Omni</strong></td><td>Qwen2.5-Omni-7B</td><td>0.76</td><td>1.52</td><td><strong>2.28</strong></td><td>1.0</td><td>1753.30</td><td>153.5</td><td><strong>66.1</strong></td></tr>
</tbody>
</table>
</div>

Light-Omni 的每次交互总延迟仅 2.28 秒，不到 M3-Agent 的 1/10；离线记忆处理速度也快于 M3-Agent。

<figure class="figure"><img src="assets/papers/light-omni-reflex-over-reasoning-agentic-video/x4.webp" alt="与推理型 agent 的对比" loading="lazy" /><figcaption>图 4：Light-Omni 与推理型视频 agent 的对比。在三个数据集上，Light-Omni 综合性能最优，且比 WorldMM-8B 快 8 倍以上。</figcaption></figure>

### 双状态机制分析

**全局状态 Sg 的有效性（表 3）**

<div class="table-wrap">
<table>
<thead>
<tr><th>Global State Strategy</th><th>Acc. (%)</th><th>△</th></tr>
</thead>
<tbody>
<tr><td>Vanilla（仅检索结果，无全局上下文）</td><td>63.11</td><td>–</td></tr>
<tr><td>Uniform（均匀采样帧）</td><td>64.11</td><td>+1.00</td></tr>
<tr><td>STM（短期记忆）</td><td>63.67</td><td>+0.56</td></tr>
<tr><td><strong>Sg（Ours，分层合并）</strong></td><td><strong>66.10</strong></td><td><strong>+2.99</strong></td></tr>
</tbody>
</table>
</div>

**潜状态 Sl 的有效性（表 4）**

<div class="table-wrap">
<table>
<thead>
<tr><th>Retrieval Representation</th><th>Acc. (%)</th><th>△</th></tr>
</thead>
<tbody>
<tr><td>Text Emb.（标准 RAG）</td><td>49.26</td><td>–</td></tr>
<tr><td>Latent Emb.（只用潜变量）</td><td>46.48</td><td>−2.78</td></tr>
<tr><td>Soft Prompt（把潜变量当 soft prompt）</td><td>43.45</td><td>−5.81</td></tr>
<tr><td><strong>Sl（Ours，潜变量与文本 embedding 相加）</strong></td><td><strong>49.90</strong></td><td><strong>+0.64</strong></td></tr>
</tbody>
</table>
</div>

关键发现：

- 单独用潜变量做检索反而下降，说明生成预训练模型直接产出检索 embedding 并不容易；
- Light-Omni 的相加融合方式既保留了文本 embedding 的稳定性，又用潜变量修正了查询分布。

### 噪声下的鲁棒性

<figure class="figure"><img src="assets/papers/light-omni-reflex-over-reasoning-agentic-video/x5.webp" alt="噪声查询下的鲁棒性" loading="lazy" /><figcaption>图 5：在 LVBench 上注入文本噪声和音频噪声后的鲁棒性对比。Light-Omni 的下降最小。</figcaption></figure>

<figure class="figure"><img src="assets/papers/light-omni-reflex-over-reasoning-agentic-video/x6.webp" alt="查询-记忆相似度矩阵" loading="lazy" /><figcaption>图 6：256 个噪声查询与真实记忆条目的相似度矩阵。Light-Omni 的对角线更锐利，SNR 更高（1.352 vs. 1.201），说明检索对齐更精准。</figcaption></figure>

### 检索粒度与可扩展性

<figure class="figure"><img src="assets/papers/light-omni-reflex-over-reasoning-agentic-video/x7.webp" alt="检索粒度影响" loading="lazy" /><figcaption>图 7：检索数量对性能的影响。Semantic memory 在 Ns≈12 时趋于稳定；Episodic memory 在 LVBench 上 Ne=4 后反而引入噪声，因此默认 Ns=12、Ne=4。</figcaption></figure>

<figure class="figure"><img src="assets/papers/light-omni-reflex-over-reasoning-agentic-video/x8.webp" alt="长时程可扩展性" loading="lazy" /><figcaption>图 8：极端长时程下的可扩展性。分层合并使全局状态的话题数保持近恒定，即使模拟 6 个月连续交互也仅为一小时会话的约 3 倍。</figcaption></figure>

<figure class="figure"><img src="assets/papers/light-omni-reflex-over-reasoning-agentic-video/x9.webp" alt="延迟随视频长度变化" loading="lazy" /><figcaption>图 9：不同视频时长下的响应延迟。Light-Omni 平均 2.37 秒，2 小时视频仍低于 6 秒，接近 O(1) 复杂度。</figcaption></figure>

## 我的理解与启发

- **“反射”是对“推理”的降维打击**：当系统拥有与记忆对齐的查询表示和全局上下文后，很多原本需要多轮推理才能回答的问题，可以压缩成单次前向。这对延迟敏感的产品（实时视频助手、可穿戴设备）非常重要。
- **潜空间对齐比 query rewrite 更根本**：RAG-Rewrite 等显式改写方法本质上是在离散文本空间做 bridge；Light-Omni 通过训练让 backbone 直接输出检索 embedding，在连续空间里一次性对齐，效率更高。
- **记忆组织需要分层与时间衰减**：Sg 的分层合并同时保留近期细节和远期摘要，是对人类记忆“时间衰减”的工程化实现。这比均匀采样或纯短期记忆更适合长视频。
- **Multi-LoRA 是能力解耦的实用方案**：memorization、generation、reaction 三个任务优化目标不同，用独立 LoRA 分别训练再动态切换，既避免冲突，又不需要维护三个完整模型。

## 局限与待观察点

1. **动作空间有限**：目前只支持 `search` 和 `speech`，更复杂的工具调用（grounding、API 调用）尚未验证；
2. **记忆构建依赖异步离线**：在线场景下睡眠时巩固如何保证实时一致性、失败恢复，论文未深入；
3. **训练数据为合成**：43k 训练样本由公开长视频合成，真实多轮交互分布可能不同；
4. **潜状态 Sl 的提升幅度有限**：在 LVBench 上仅比 Text Emb. 高 0.64%，说明对齐还有提升空间；
5. **backbone 依赖 Qwen-Omni 系列**：对非 Omni 架构的 MLLM 是否同样有效需要更多验证。

## 结论

Light-Omni 提出了一种从“推理型 agent”向“反射型 agent”转变的范式。通过双上下文状态——非参数全局状态 $\mathcal{S}_g$ 提供长期上下文、参数化潜状态 $\mathcal{S}_l$ 实现单次前向的检索与动作控制——它在多个长视频基准上同时实现了更高的准确率、更低的延迟和更小的显存占用。对于需要实时响应的长时程多模态交互系统，这一设计思路具有很强的参考价值。
