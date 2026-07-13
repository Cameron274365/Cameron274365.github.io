---
id: "omnirag-agent-agentic-omnimodal-long-audio-video-qa"
title: "OmniRAG-Agent：低资源长音视频问答的智能体式全模态推理"
category: "多模态 RAG"
date: "2026-07-13"
order: 48
readTime: "13 min"
tags: ["RAG","Omnimodal","Agent","Audio-Video QA","GRPO","Reinforcement Learning"]
summary: "OmniRAG-Agent 面向低资源约束下的长音视频问答，让 OmniLLM 先把长视频压缩、再从外部图像库与 ASR 音频库中按需检索短片段作为细粒度证据；通过多轮智能体循环（规划-检索-合并证据-决定停止）拼接散落线索，并用带双重约束奖励的 GRPO 端到端联合优化工具使用与答案质量。在 OmniVideoBench、WorldSense、Daily-Omni 三个基准上于 15GB 显存的低资源设置下一致超越基线，且可即插即用于 Qwen2.5/3-Omni 等多种骨干。"
---

## 一句话总结

长音视频问答的线索往往散落在几分钟里、还得"看到的"和"听到的"结合才能答对，但对整段视频做密集编码在算力和显存上都太贵。OmniRAG-Agent 的核心思路是：**把长视频先降采样压缩喂给 OmniLLM，再让它像智能体一样多轮地从外部图像库/音频库检索短相关片段作为细粒度证据，并用 GRPO 强化学习端到端联合优化"何时检索、检索什么、何时停止、怎么回答"**——从而在低资源（15GB 显存）下把长达 30 分钟的音视频问答做好。

<div class="metric-grid">
  <div class="metric"><strong>15GB</strong><span>显存部署成本（对比 Qwen2.5-Omni 的 60GB）</span></div>
  <div class="metric"><strong>30min</strong><span>支持的最大音视频输入长度</span></div>
  <div class="metric"><strong>3 基准</strong><span>OmniVideoBench / WorldSense / Daily-Omni 一致提升</span></div>
  <div class="metric"><strong>即插即用</strong><span>支持 Qwen2.5-Omni、Qwen3-Omni 等多种骨干</span></div>
</div>

## 论文信息

- **标题**：OmniRAG-Agent: Agentic Omnimodal Reasoning for Low-Resource Long Audio-Video Question Answering
- **作者**：Yifan Zhu, Xinyu Mu, Tao Feng, Zhonghong Ou, Yuning Gong, Haoran Luo
- **机构**：北京邮电大学 / 南洋理工大学 / 清华大学 / 新加坡国立大学 / 北京信息科技大学
- **版本**：arXiv:2602.03707v4，2026-03-30 修订（v1 为 2026-02-03）
- **链接**：[arXiv](https://arxiv.org/abs/2602.03707) · [PDF](https://arxiv.org/pdf/2602.03707)

## 背景：长音视频问答的四个痛点

OmniLLM 已经能统一处理文本、图像、音频、视频，但在**低资源、长时长**场景下，现有全模态 QA 系统存在四个问题：

- **密集编码太贵**：对每一帧和每段音频都密集编码，算力与显存开销快速爆炸。
- **细粒度检索弱**：长视频降采样后丢失细节，模型抓不住"小而关键"的证据。
- **缺乏主动规划**：端到端模型一次性看完就答，无法针对问题主动去"找线索"。
- **没有端到端优化**：API 拼接式系统各模块割裂，无法整体调优。

<figure class="figure">
  <img src="assets/papers/omnirag-agent/x2_comparison.webp" alt="长音视频 QA 不同方法对比" loading="lazy" />
  <figcaption>图 1：低资源约束下长音视频 QA 的几类方法对比——端到端 OmniLLM、API 式 OmniAgent、LLM 微调优化，以及本文带图像-音频片段检索的低资源智能体训练框架。</figcaption>
</figure>

现有两类主流做法各有短板：**API 式 Omni 系统**调用多个专用模型逐步分析、再拼接结果，部署 token 成本高；**微调式 Omni 系统**用 RL 优化 OmniLLM 更擅长抓细节，但没有引入外部检索。OmniRAG-Agent 的定位是**兼取检索 + 智能体 + 强化学习三者之长**，且部署成本更低。

## 方法：RAG 环境 + 多轮智能体 + GRPO

OmniRAG-Agent 由三部分组成：多模态 RAG 框架、多轮推理过程、基于 RL 的训练方法。

<figure class="figure">
  <img src="assets/papers/omnirag-agent/x3_framework.webp" alt="OmniRAG-Agent 框架总览" loading="lazy" />
  <figcaption>图 2：OmniRAG-Agent 框架总览。OmniLLM 与一个多模态检索环境交互，通过多轮检索回答长音视频问题；图像库与音频库是即插即用的。</figcaption>
</figure>

### 1. 多模态 RAG：从原视频建库、按需检索片段

由于 OmniLLM 的视频输入长度有限，先用**时间降采样** \(\tilde{X}=\mathcal{D}_t(X)\) 把原始流 \(X\) 压缩成 \(t\) 秒的短流喂给模型。但降采样丢了细粒度信息，于是从原始 \(X\) 构建两个外部库来"找回"线索：

- **图像库 \(B^{img}\)**：以固定间隔 \(\Delta\) 秒从 \(X\) 采样帧图像，每张图带时间戳元数据。
- **音频库 \(B^{aud}\)**：对音轨跑 ASR，得到带时间戳的转写片段序列，用文本索引。

**检索时**：模型在每一轮生成自然语言检索 query。图像检索用 **CLIP** 文本/图像编码器算相似度，取 top-\(K_{img}\) 帧；音频检索把 query 与 ASR 文本编码后算相似度，取 top-\(K_{aud}\) 段。检索结果的元数据提供了"最可能含关键证据的时间戳"，等于用检索把散落在长视频里的线索定位回来。

### 2. 多轮推理：规划-检索-合并-停止的智能体循环

把"图像库+音频库+检索工具"视为**环境 \(L\)**，OmniLLM 作为**智能体 \(\mathcal{M}\)**，遵循随机策略 \(\pi_\theta\)。每一轮的动作 \(a_t=(z_t, q_t, c_t)\) 包含三部分：

- \(z_t\)：**规划轨迹**——想清楚下一步该找什么；
- \(q_t\)：**检索 query**——发给图像/音频检索工具；
- \(c_t\)：**继续/停止决策**——是否还要再检索一轮。

每轮把 query 与返回证据追加进交互历史 \(H_t = H_{t-1}\oplus(q_t, E_t)\)。由于历史会随轮数增长，作者额外维护一个**紧凑状态表示 \(F(h_t)\)**，用时间摘要算子把历史压缩，避免上下文爆炸。当模型决定停止或达到最大轮数时，基于完整历史输出最终答案。这种"多轮取证"正是为了应对长视频里线索分散、需要从不同音视觉角度拼凑的复杂问题。

### 3. RL 优化：双重约束奖励 + GRPO

用 **GRPO（Group Relative Policy Optimization）** 做端到端强化学习，奖励由两部分门控组合：

- **格式奖励 \(R_{fmt}\)**：强制模型遵循结构化的工具调用输出格式，按匹配的格式标签数量给分（上限 1.0）。
- **性能奖励 \(R_{perf}\)**：答案与 ground-truth 精确匹配则为 1，否则 0。
- **门控组合**：只有当格式奖励达到阈值（\(R_{fmt}\ge 0.5\)）时才计入性能奖励，否则只给格式分。这保证模型先学会"规规矩矩地调用工具"，再学"答对"。

GRPO 在一个 batch 内对多条采样轨迹的奖励做标准化得到优势 \(\hat{A}^{(i)}\)，配合裁剪的策略比和 KL 正则更新策略——省去了额外的 value network，适合低资源训练。

## 数据与评测

- **基准**：OmniVideoBench（细粒度能力子集）、WorldSense、Daily-Omni，均要求对长音视频流推理、关键证据散落在时间轴上。
- **骨干模型**：闭源 GPT-5.1、Gemini 2.0/2.5-Flash；开源 Qwen2.5-Omni-3B/7B、Qwen3-Omni-30B。
- **消融变体**：Base → +RAG（单步检索）→ +RAG+Agent（多轮规划工具调用）→ +RAG+Agent+RL（GRPO 优化），逐组件叠加以隔离贡献。
- **指标**：按各基准官方协议报告各能力子集准确率及平均分。

## 实验结果

在三个基准上，逐组件叠加带来一致提升（OmniVideoBench 平均分为例）：

<table>
  <thead>
    <tr><th>骨干</th><th>配置</th><th>OmniVideoBench Avg</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="4">Qwen2.5-Omni-3B</td><td>Base</td><td>23.05</td></tr>
    <tr><td>+RAG</td><td>24.61</td></tr>
    <tr><td>+RAG+Agent</td><td>26.95</td></tr>
    <tr><td>+RAG+Agent+RL</td><td><strong>27.34</strong></td></tr>
    <tr><td rowspan="4">Qwen2.5-Omni-7B</td><td>Base</td><td>25.00</td></tr>
    <tr><td>+RAG</td><td>27.73</td></tr>
    <tr><td>+RAG+Agent</td><td>28.52</td></tr>
    <tr><td>+RAG+Agent+RL</td><td><strong>29.69</strong></td></tr>
    <tr><td rowspan="2">Gemini 2.5-Flash（闭源）</td><td>Base</td><td>28.52</td></tr>
    <tr><td>+RAG+Agent</td><td><strong>35.16</strong></td></tr>
  </tbody>
</table>

- **主结果（RQ1）**：在同一低资源设置下，OmniRAG-Agent 在三基准上一致提升，收益覆盖 Reasoning、Perception、Temporal 等多个能力维度。闭源模型（Gemini 2.5-Flash）叠加 RAG+Agent 后 OmniVideoBench 从 28.52 → 35.16。
- **泛化（RQ2）**：在 OmniVideoBench 上训练可直接迁移到 WorldSense / Daily-Omni 并明显超过零样本，说明学到的"检索-推理"行为不绑定单一数据风格；小骨干收益更明显，大骨干呈边际递减。
- **可迁移性（RQ3）**：插到 5 种代表性 OmniLLM 上都能把性能曲线整体外扩，尤其在依赖细粒度接地与跨模态证据的能力（推理、时序理解、文本相关）上提升最大。
- **预算分析（RQ5）**：检索片段数量适中时性价比最高（够准又不会让上下文过长）；多轮检索与 RL 都能持续推高分数。

## Ablation：哪些设计最关键？

- **RAG 是第一个明显增益点**：能拉进短而相关的片段，通常带来第一波提升。
- **Agent 循环再加一层**：对依赖多轮推理与检索的能力尤其有效。
- **GRPO 锦上添花**：对开源模型带来虽小但确定的提升，且能端到端改善证据选择与最终作答。

## 我的理解与启发

- **"降采样喂主干 + 外部库按需检索"是低资源长视频的关键取舍**：不追求把所有帧都塞进上下文，而是把细粒度信息外置成可检索的库，用检索把线索"召回"，直接把显存从 60GB 压到 15GB。这是长视频理解里非常务实的工程范式。
- **把 RAG 变成"多轮智能体动作"很自然**：检索 query、规划、停止决策统一到一个策略里，用 RL 联合优化，避免了 RAG 与推理割裂——这正是 API 拼接式系统缺的"端到端"。
- **门控奖励设计值得借鉴**：先卡格式再给答对分，能有效防止模型为了拿答案分而破坏工具调用格式，是训练工具使用 agent 的实用技巧。
- **音频用 ASR 文本索引而非音频嵌入**：把音频检索转化为对语音内容的自然语言检索，简单且能复用文本编码器，降低了多模态检索的实现门槛。

## 局限与待观察点

- **依赖检索质量**：检索不准会直接拖累最终答案。
- **多轮误差累积**：迭代交互可能逐轮放大错误。
- **迭代交互带来额外计算**：多轮取证虽省显存，但增加了推理轮次开销。
- 作者提出未来方向：更鲁棒的停止/验证策略、扩展到更广的多模态场景、改善向真实域的迁移。

## 结论

OmniRAG-Agent 通过让 OmniLLM 与图像库/音频库多轮交互检索细粒度证据，配合 GRPO 端到端优化工具使用与作答，在低资源约束下把长达 30 分钟的音视频问答做出了一致、可迁移的提升。三条核心结论：(i) 紧预算下也能检索到细粒度证据；(ii) 多步智能体规划有助于跨长输入收集并核验散落线索；(iii) 强化学习能进一步端到端改善证据选择与最终答案。
