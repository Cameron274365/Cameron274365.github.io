---
id: "qwen3.5-omni-technical-report"
title: "Qwen3.5-Omni：混合注意力 MoE 的全模态实时交互大模型"
category: "全模态模型"
date: "2026-07-08"
order: 47
readTime: "16 min"
tags: ["Omni-modal","Qwen","Thinker-Talker","MoE","Speech Generation","Streaming","ARIA"]
summary: "Qwen3.5-Omni 把 Thinker-Talker 架构升级为混合注意力 MoE，支持 256k 上下文、10 小时音频与 400 秒 720P 视频；提出 ARIA 动态对齐文本与语音单元解决流式合成不稳定问题，在 215 个音频/音视频子任务上取得 SOTA，关键音频任务超过 Gemini-3.1 Pro，并涌现出音视频 Vibe Coding 新能力。"
---

## 一句话总结

Qwen3.5-Omni 是 Qwen-Omni 家族的最新一代**全模态端到端大模型**：它延续 Thinker-Talker 架构，但把 Thinker 和 Talker 双双升级为**混合注意力 MoE（Hybrid-Attention MoE）**以支撑高效长序列推理，并针对流式语音合成不稳定的顽疾提出 **ARIA**——一种在流式解码时动态对齐文本与语音单元的技术。结果是：文本/视觉能力不退化，音频与音视频能力在 215 个子任务上取得 SOTA，关键音频任务超过 Gemini-3.1 Pro。

<div class="metric-grid">
  <div class="metric"><strong>256k</strong><span>上下文长度，支持 10 小时音频 / 400 秒 720P 视频（1 FPS）</span></div>
  <div class="metric"><strong>215</strong><span>音频与音视频子任务/榜单取得 SOTA</span></div>
  <div class="metric"><strong>235ms</strong><span>Flash 音频输入首包延迟（Plus 435ms）</span></div>
  <div class="metric"><strong>113 / 36</strong><span>语音识别 / 语音合成支持的语言与方言数</span></div>
</div>

## 论文信息

- **标题**：Qwen3.5-Omni Technical Report
- **作者**：Qwen Team（阿里巴巴通义千问团队）
- **版本**：arXiv:2604.15804v2，2026-04-21 修订（v1 为 2026-04-17）
- **链接**：[arXiv](https://arxiv.org/abs/2604.15804) · [PDF](https://arxiv.org/pdf/2604.15804)
- **模型规格**：Plus / Flash 两个变体，均为 instruct 模型，256k 上下文

<figure class="figure">
  <img src="assets/papers/qwen3.5-omni/teaser.webp" alt="Qwen3.5-Omni 能力总览" loading="lazy" />
  <figcaption>图 1：Qwen3.5-Omni 是统一的端到端模型，可处理文本、音频、图像、视频，并实时生成文本或语音。Plus 侧重原生多模态、SOTA 性能、详细音视频字幕、多语言；Realtime 侧重语音控制、WebSearch 工具、语音克隆、语义打断等实时交互能力。</figcaption>
</figure>

## 背景：相较 Qwen3-Omni 的五大升级

Qwen3.5-Omni 建立在 Qwen2.5-Omni 的 Thinker-Talker 架构之上，相对上一代 Qwen3-Omni 有五个关键升级：

1. **混合注意力 MoE**：Thinker 和 Talker 都采用 Hybrid-Attention MoE，兼顾容量与推理效率；
2. **256k 长上下文**：支持 10+ 小时音频、400+ 秒 720P 音视频（1 FPS）；
3. **多码本 codec 表示**：单帧即时合成语音；
4. **ARIA**：流式解码时动态对齐文本与语音单元，显著提升自然度与鲁棒性；
5. **多语言大幅扩展**：语音识别覆盖 113 种语言/方言，语音合成覆盖 36 种。

关键前提是——**这些多模态能力没有以牺牲文本/视觉能力为代价**。与同尺寸的纯 Qwen 单模型相比，Qwen3.5-Omni 在文本和视觉模态上保持了 SOTA，不出现退化。

## 方法：Thinker-Talker + 混合 MoE + 流式语音

<figure class="figure">
  <img src="assets/papers/qwen3.5-omni/architecture.webp" alt="Qwen3.5-Omni 整体架构" loading="lazy" />
  <figcaption>图 2：Qwen3.5-Omni 总体架构。Thinker 负责文本生成；Talker 直接接收 Thinker 的中间层高层表示，自回归预测多码本序列生成流式语音 token。每个解码步由 MTP 模块输出当前帧的残差码本，再由 Code2Wav 渲染器逐帧增量合成波形，实现帧级流式生成。</figcaption>
</figure>

### 1. 双脑分工：Thinker 想、Talker 说

- **Thinker**：通过 Vision Encoder 和 AuT（音频编码器）接收视觉与音频信号，音视频输入交错编码并**插入显式时间戳**以增强时序感知（对长视频/音视频尤为关键）。它是理解与推理的核心，支持 256k token 输入。
- **Talker**：负责上下文相关的语音生成，条件建立在多模态输入 + Thinker 的文本输出之上。它采用 Qwen3-Omni 引入的 RVQ 语音表示大幅提升推理效率，并能根据对话上下文动态调节韵律、音量、情感等声学属性。

### 2. 混合 MoE + GDN：为长音视频序列加速

Thinker 和 Talker 都基于 Qwen3.5 的 Hybrid MoE 架构。除了 MoE 本身的效率优势，它包含 **Gated Delta Net（GDN）**模块，对长音视频序列建模特别有效，能**显著降低长上下文推理的 KV-cache I/O 开销**，从而提升生成吞吐并支持更高并发。

配合沿用自 Qwen2.5/3-Omni 的**分块预填充（chunked prefilling）**机制——音视频编码器沿时间维度分块输出——可大幅降低 Thinker 和 Talker 的首 token 时延（TTFT）。最终端到端首包延迟：音频输入 Plus 435ms / Flash 235ms，视频输入 Plus 651ms / Flash 426ms。

### 3. AuT 音频编码器：从头训练的 6.25Hz 通用音频表示

<figure class="figure">
  <img src="assets/papers/qwen3.5-omni/aut.webp" alt="AuT 音频编码器结构" loading="lazy" />
  <figcaption>图 3：AuT 概览。采用 attention-encoder-decoder 结构从头训练。FBank 特征经 4 个 Conv2D 块下采样 16 倍，再送入 32 层自注意力，得到 6.25Hz token 率的音频 token；解码器为 8 层自注意力 + 交叉注意力。</figcaption>
</figure>

AuT 消耗了 Qwen3-ASR 生成的 **4000 万小时音频-文本对**数据训练，相比 Qwen3-Omni 编码器加入了 20+ 种语言的更多多语言数据（中/英/多语言比例约 3.5:3.5:3）。它采用**动态注意力窗口训练机制**，兼顾实时预填充缓存下的推理与离线音频理解任务的性能平衡。

### 4. ARIA：解决流式语音合成的不稳定

流式 TTS 的核心痛点是——**文本 tokenizer 和语音 tokenizer 的编码效率不匹配**，导致跳字、错误发音、数字念错等问题。Qwen3-Omni 用的是双轨（dual-track）Talker 输入设计，而 Qwen3.5-Omni 改用 **ARIA**：在交错文本与语音单元之前，**动态对齐**两者。这缓解了 tokenization 速率不匹配带来的不稳定，在几乎不增加延迟的前提下显著提升对话语音的稳定性与韵律。

## 训练：三阶段预训练 + 分模块后训练

**预训练（三阶段）**：数据涵盖 image-text、video-text、audio-text、video-audio、video-audio-text、纯文本等多模态语料。

1. **编码器对齐（S1）**：锁定 LLM 参数，先训各自 adapter 再训 Vision Encoder 和 AuT 编码器；
2. **通用阶段（S2）**：解冻全部参数，用约 **4 万亿 token**（文本 0.92T / 音频 1.99T / 图像 0.95T / 视频 0.14T / 音视频 0.29T），序列长 32,768；
3. **长上下文阶段（S3）**：序列长扩到 262,144，提高长音频/长视频比例。

**后训练**分 Thinker 与 Talker 两条线：

- **Thinker（三阶段）**：① 专家蒸馏——先为文本/视觉/音频分别训练领域专家 teacher（SFT+RL），再蒸馏进统一模型；② **On-Policy 蒸馏（OPD）**——把文本输入下的高质量回答作为音频输入回答的蒸馏目标，缩小"音频问答不如文本问答"的差距；③ **交互对齐 RL**——针对多轮对话中的语言乱切换、人设不一致、长上下文指令遵循退化等问题构造多轮轨迹并设计奖励。
- **Talker（四阶段）**：通用阶段（2000 万+ 小时多语言语音）→ 长上下文阶段（扩到 64k，用 Captioner 降幻觉）→ RL 阶段（DPO + 规则奖励 + GSPO）→ 说话人微调。

## 实验结果

### 文本与视觉：不退化

与纯文本模型 Qwen3.5-Plus-Instruct 相比，Qwen3.5-Omni-Plus 在知识、指令遵循、长上下文、STEM、推理、Agent 各维度**基本持平**，指令遵循甚至略优（IFBench 52.6 vs 52.6 基线 51.1）。视觉上与 Qwen3.5-Plus-Instruct 相当，且在长短视频理解上更强（VideoMME 81.9、MLVU 86.8、MVBench 79.0 均超基线）。

### 音频：多项超越 Gemini-3.1 Pro

<table>
  <thead>
    <tr><th>基准</th><th>Gemini-3.1 Pro</th><th>Omni-Flash</th><th>Omni-Plus</th></tr>
  </thead>
  <tbody>
    <tr><td>MMAU（音频理解）</td><td>81.1</td><td>80.4</td><td><strong>82.2</strong></td></tr>
    <tr><td>MMSU</td><td>81.3</td><td>72.2</td><td><strong>82.8</strong></td></tr>
    <tr><td>RUL-MuchoMusic（音乐）</td><td>59.6</td><td>60.5</td><td><strong>72.4</strong></td></tr>
    <tr><td>VoiceBench（对话）</td><td>88.9</td><td>87.8</td><td><strong>93.1</strong></td></tr>
    <tr><td>Fleurs ASR（WER↓）</td><td>7.32</td><td>10.75</td><td><strong>6.55</strong></td></tr>
    <tr><td>LibriSpeech clean（WER↓）</td><td>3.36</td><td>1.30</td><td><strong>1.11</strong></td></tr>
  </tbody>
</table>

在音频理解、推理、识别、翻译、对话上，Omni-Plus 全面超过或持平 Gemini-3.1 Pro；S2TT 与 ASR 更是**跨多语言/方言/歌声转写一致领先**（如 Kespeech WER 从 Gemini 的 23.67 降到 3.46）。

### 音视频：对齐 Gemini-3.1 Pro 级

DailyOmni 取得 SOTA（84.6），Qualcomm IVD 真实交互场景大幅超过 Gemini（68.5 vs 66.2），字幕（OmniCloze 64.8）与工具使用（OmniGAIA 57.2%）也表现强劲。

### 语音生成：零样本 / 多语言 / 跨语言全面领先

- **零样本 TTS**：SEED-TTS test-en WER 低至 **1.26**，超过 CosyVoice3、MiniMax-Speech 等；
- **多语言**：29 种语言中 **22 种取得最低 WER**，且说话人相似度（音色克隆保真度）多数领先；
- **跨语言语音克隆**：12 个方向中 10 个 SOTA，zh→ko 错误率从 CosyVoice3 的 14.4 降到 4.03（相对下降约 72%）。

### 涌现能力：音视频 Vibe Coding

论文报告了全模态模型的一个**新涌现能力**——**Audio-Visual Vibe Coding**：模型能直接根据音视频指令进行编程。这暗示全模态训练可能带来单模态训练难以获得的跨模态泛化。

## 我的理解与启发

- **"不退化"是全模态模型最难也最有说服力的指标**：很多多模态模型是以牺牲纯文本能力换来的。Qwen3.5-Omni 强调与同尺寸纯文本/视觉模型持平，OPD（把文本回答蒸馏到音频输入）是缩小模态间质量差距的关键工程手段，值得借鉴。
- **ARIA 抓住了流式 TTS 的真问题**：文本与语音 tokenizer 的速率不匹配是跳字/错读的根因。用"动态对齐 + 交错"替代双轨输入，是一个针对性强、低延迟代价的设计。
- **Hybrid MoE + GDN 是长音视频推理的效率抓手**：256k 上下文、10 小时音频的可行性，本质靠降低 KV-cache I/O。对做长视频/长音频实时系统的工程师，这是核心可复用点。
- **分块预填充 + 帧级流式合成**：从编码器分块到 MTP 逐帧残差码本再到 Code2Wav 增量渲染，整条链路都为"首包延迟"服务——这是实时语音交互产品化的完整工程范式。

## 局限与待观察点

- **技术报告性质**：作者仅署名 Qwen Team，未披露完整参数规模、精确训练成本与消融，很多结论以榜单结果呈现，缺少对单项设计（如 ARIA、GDN）的独立消融量化。
- **Vibe Coding 仅为观察性报告**：这个"涌现能力"缺乏系统评测，泛化边界不明。
- **对比基线单一**：音频/音视频主要对比 Gemini-3.1 Pro，与其他开源全模态模型的横向对比较少。
- **Flash 与 Plus 差距明显**：Flash 在知识/推理类任务上明显弱于 Plus（如 HMMT 59.0 vs 84.4），实时性与能力仍需按场景权衡。

## 结论

Qwen3.5-Omni 把全模态大模型往"**原生 omni agent**"又推进了一步：混合注意力 MoE 支撑起 256k 长上下文与超长音视频，ARIA 解决了流式语音合成的稳定性顽疾，三阶段/四阶段的蒸馏与 RL 后训练保证了跨模态一致的高质量交互。它在音频、音视频、语音生成上全面达到或超过 Gemini-3.1 Pro 级别，同时不牺牲文本与视觉能力——对追求"一个模型统一感知、推理、行动、发声"的实时多模态系统而言，这是一份工程完成度很高的路线参考。
