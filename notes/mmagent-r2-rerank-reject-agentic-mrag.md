---
id: mmagent-r2-rerank-reject-agentic-mrag
title: "MMAgent-R²：给 Agentic 多模态 RAG 装上「视觉重排 + 主动拒绝」的内部校验"
category: 多模态 RAG
date: 2026-07-21
order: 1
readTime: 14 min
tags: ["Agentic RAG", "Multimodal RAG", "KB-VQA", "Visual Reranking", "Active Rejection", "GRPO", "Process Reward", "ECCV 2026"]
summary: "MMAgent-R² 针对 KB-VQA 中「检索到一堆视觉相似但事实不匹配的干扰项、且后处理被锁死在固定候选集里」的问题，提出把视觉重排（直接比对 query 图与候选图）和主动拒绝（不自信就丢弃并检索下一批）作为 agent 的内部校验动作。用含步骤级校验奖励的复合奖励 + GRPO 联合优化检索、校验、答案生成。在 InfoSeek/E-VQA/MMhops 上达 SOTA，E-VQA +7.2、MMhops Bridging +13.2。ECCV 2026 接收。"
---

## 一句话总结

MMAgent-R² 的核心洞察是：**现有多模态 RAG 用全局视觉特征去匹配候选实体，当知识库里有大量长得很像的实体时，检索器根本分不清，候选集会被"视觉相似但事实错误"的干扰项塞满；而后续的去噪、过滤又都被锁死在这个固定候选集里，检索一旦失败，错误必然传导到最终答案**。解法是把两个动作变成 agent 的**内部校验机制**：① **视觉重排（Visual Reranking）**——直接拿 query 图和候选图逐一比对，抓住文本描述之外的判别性细节，在相似候选里精确锁定目标；② **主动拒绝（Active Rejection）**——当当前批次没有自信匹配时，丢弃并触发检索下一批候选，突破固定候选池的限制。整套流程用带步骤级校验奖励的复合奖励函数，通过 GRPO 端到端联合优化"外部检索 + 内部校验 + 答案生成"。

<div class="metric-grid">
  <div class="metric"><strong>+7.2 / +2.2</strong><span>E-VQA 上 overall / Single-Hop 相比次优方法的提升（8B）</span></div>
  <div class="metric"><strong>+13.2 / +10.4</strong><span>MMhops Bridging / Comparison 提升，超越 Gemini-2.5-Pro</span></div>
  <div class="metric"><strong>15.4% → 29.5%</strong><span>E-VQA 上实体识别准确率（+14.1），检索越难增益越大</span></div>
  <div class="metric"><strong>仅靠 RL</strong><span>无需 SFT 冷启动 / 外部 critic，也超越多阶段训练的 ReAG</span></div>
</div>

## 论文信息

- **标题**：MMAgent-R²: Learning to Rerank and Reject for Agentic mRAG
- **作者**：Tao Zhang, Ziqi Zhang, Zongyang Ma, Yuxin Yang, Bing Li, Chunfeng Yuan, Kang Rong, Fengyun Rao, Jing Lyu, Weiming Hu
- **机构**：中科院自动化所（CASIA）、中国科学院大学、微信视觉（腾讯）等
- **版本**：arXiv:2607.07383v1，2026-07-08（ECCV 2026 接收）
- **链接**：[arXiv](https://arxiv.org/abs/2607.07383) · [PDF](https://arxiv.org/pdf/2607.07383) · [HTML](https://arxiv.org/html/2607.07383v1)

## 背景：固定候选集是错误传导的根源

基于知识的视觉问答（KB-VQA）要求模型先识别图中的视觉实体，再关联外部百科知识回答属性问题（例如"这座建筑建于何年？"需先精确识别是哪座建筑）。主流多模态 RAG 走的是 **"Retrieve-then-Postprocess"（先检索再后处理）** 范式：检索阶段用全局视觉特征从大规模图文百科里召回相似候选，后处理阶段再从一个**小而固定**的候选集里过滤出匹配项。

<figure class="figure">
  <img src="assets/papers/mmagent-r2/fig1_comparison.webp" alt="Retrieve-then-Postprocess 与 MMAgent-R² 的对比" loading="lazy" />
  <figcaption>图 1：(a) 固定候选集范式——用全局视觉特征检索，模型容易被视觉相似但不匹配的条目误导；(b) MMAgent-R² 把 query 图与候选图（①–⑧）比对来重排，并在没有自信匹配时拒绝当前批、检索更多候选，从而正确识别实体并作答。</figcaption>
</figure>

这个范式有两个结构性缺陷：

1. **检索器分不清相似实体**：全局特征匹配对"细粒度判别"无能为力，候选集里塞满视觉相似的干扰项。E-VQA 的知识库有约 200 万 Wikipedia 页面，检索器 Recall@1 只有 **15.4%**。
2. **错误被锁死并传导**：后处理只能在这个失败的固定候选集里打转，检索的错误无法纠正，直接传到最终答案。

## 方法：把校验变成 agent 的动作

<figure class="figure">
  <img src="assets/papers/mmagent-r2/fig2_framework.webp" alt="MMAgent-R² 整体框架" loading="lazy" />
  <figcaption>图 2：MMAgent-R² 整体框架。给定多图输入和问题，VLM agent 生成交错的"先推理后行动（Reason-then-Act）"轨迹，每一步从「外部检索 / 内部校验 / 答案生成」动作空间里选择并执行；用带校验奖励的复合奖励做 RL 优化，校验奖励为每个重排/拒绝决策提供密集的步骤级监督。</figcaption>
</figure>

### 1. 动作空间：三类动作（§3.2.1）

把 VLM 建模为由策略 π_θ 驱动的 agent，在多轮循环内自主生成"思考 + 动作"序列，动作空间 𝒜 分三类：

- **外部检索（External Retrieval）**
  - `SearchImage(j)`：编码第 j 张 query 图，从视觉知识库返回 Top-K 候选（图 + 实体名）。**关键：支持动态候选扩展**——对同一 query 图连续调用会返回不重叠的后续批次，突破初始 Top-K 的限制。
  - `SearchText({qᵢ})`：并行提交 m 个文本查询，各自从文本知识库返回 Top-K 段落并拼接，一轮内聚合多维事实。
- **内部校验（Internal Verification）**——本文核心
  - `Rerank(eₖ)`：模型通过视觉特征比对确认候选批中实体 eₖ 匹配 query 图，输出 eₖ 并写入上下文，作为后续推理的确定前提。
  - `Reject()`：当前批次无匹配时执行，触发动态候选扩展、环境给出下一批；每张 query 图设最大拒绝次数 Rej_max 以平衡搜索深度与效率。
- **答案生成（Answer Generation）**
  - `Answer(y)`：综合已积累的证据链生成最终答案，终止循环。

两者协同：**重排负责在相似结果里"精确锁定"（precision），拒绝负责在检索失败时"动态扩边界"（candidate coverage）**，从互补维度提升实体识别。

### 2. Agentic 推理循环（§3.2.2）

多轮"推理-动作-反馈"范式：agent 基于历史 ℋ_{t-1} 自主推理并选下一个动作（不走固定流程）；环境执行后返回观测 oₜ（图像批或文本），连同动作命令追加进历史，形成持续更新的多模态证据链。循环在输出 `Answer` 或达到最大轮数 T_max 时终止；每张 query 图达到 Rej_max 后，环境给出边界提示引导平衡。agent 甚至可以直接用参数化知识回答。

### 3. Agentic RL 训练与复合奖励（§3.3）

用 GRPO 训练。对每个样本采样 G 条完整多轮 rollout，复合奖励为：

\[R = R_{\text{outcome}} + R_{\text{format}} + R_{\text{ver}}\]

- **结果奖励 R_outcome**：答案匹配 ground-truth 得 1，否则 0。
- **格式奖励 R_format**：每轮输出是否严格遵守动作语法，保证结构化交互稳定。
- **校验奖励 R_ver**（本文关键）：对所有 query 图的每个校验决策（Rerank / Reject）给步骤级监督，逐步奖励求和。每步奖励 ρ 定义为：

\[\rho(a,\mathcal{B},e^*)=\begin{cases}\delta & a=\texttt{Reject} \text{ 且 } e^*\notin\mathcal{B}\\ 1 & a=\texttt{Rerank}(e^*)\\ 0 & \text{otherwise}\end{cases}\]

即：候选批里**有**正确实体时正确重排给 1；候选批里**没有**正确实体时正确拒绝给 δ（δ 为拒绝奖励权重）。这种密集监督补足了稀疏的最终答案信号，**显式鼓励模型在候选不确定时选择"拒绝"而非"瞎猜"**。

- **环境响应掩码（§3.3.2）**：多轮 rollout 时，把所有环境返回的 token（候选图、文本段落、确认消息、边界提示）都 mask 掉，策略梯度只更新模型自己生成的"推理 + 动作" token，确保模型对自身决策负责。

## 实验结论

**数据集**：InfoSeek（100K 页知识库，Unseen-Q/Unseen-E 两子集，谐波平均）、E-VQA（2M 页知识库、单跳+两跳、BEM 分数）、MMhops（多图多跳，Bridging 单图链式 / Comparison 多图对比，需 3–4 步推理）。两个变体：MMAgent-R²-7B（Qwen2.5-VL-7B）、MMAgent-R²-8B（Qwen3-VL-8B），检索器 EVA-CLIP-8B。

**主结果**：两个变体在三个 benchmark 上都超越所有对比方法（Direct Answer / Multimodal RAG / Agentic mRAG 三类基线）。

| Benchmark | 指标 | MMAgent-R² | 相比次优 |
|-----------|------|-----------|---------|
| E-VQA | Single-Hop / All | 55.9 / 54.2 (8B) | +2.2 / +7.2 |
| InfoSeek | All | 50.2 (7B) | +3.0 |
| MMhops | Bridging / Comparison | 67.2 / 39.8 (8B) | +13.2 / +10.4 |

几个关键结论：

- **检索越难，视觉重排越关键**：E-VQA 检索器 Recall@1 仅 15.4%，MMAgent-R² 仍拿到 SOTA，证明当初始检索不可靠时视觉重排是救命稻草。
- **仅靠 RL 胜过多阶段训练**：InfoSeek 上超越 ReAG（后者需 SFT 冷启动 + 外部 critic + RL 多阶段），而 MMAgent-R² 只用 RL。
- **视觉校验不可或缺**：并发的 agentic 方法 MMhops-R1（无视觉校验）在 InfoSeek 上只有 33.2%。
- **多跳泛化强**：MMhops 上大幅超越闭源 Gemini-2.5-Pro（54.0 / 29.4）。

**实体识别提升（表 4）**：MMAgent-R² 的实体识别准确率全面超越检索器 Recall@1——E-VQA 从 15.4% 提到 29.5%（+14.1），InfoSeek 从 53.7% 提到 63.6%（+9.9）。但与 R@15 上界仍有差距，说明"在大量相似实体里精确区分"仍是难题。

<figure class="figure">
  <img src="assets/papers/mmagent-r2/fig3_qualitative.webp" alt="MMAgent-R² 定性示例" loading="lazy" />
  <figcaption>图 3：定性示例。上：E-VQA 中几只飞蛾候选颜色翅形都相似、仅细微斑纹不同，MMAgent-R² 通过视觉比对锁定目标，而无重排的变体选了错误 top-1；下：InfoSeek 中正确建筑不在首批候选里，模型主动拒绝并在后续检索里找到目标。</figcaption>
</figure>

**消融（在 InfoSeek 26K 样本上，8B）**：

- **重排 + 拒绝（表 5）**：都去掉 = 用检索器 top-1 作答（Ident. 53.7%，VQA 40.2%）；加视觉重排 → 57.5% / 44.3%（+3.8/+4.1，主驱动力）；再加拒绝 → 58.6% / 45.5%（拒绝靠扩大候选覆盖再补一刀）。
- **候选批大小 K（表 6）**：K=3 覆盖不足、拒绝频繁（Avg. Rej. 0.63）；K=7 覆盖广但 token 成本高、收益递减；**K=5 最佳平衡**。
- **最大拒绝次数（表 7）**：0→2 稳步提升（44.3%→45.5%），3 时反降（冗余候选拉长上下文、损害推理）。平均拒绝次数远低于上限（Rej_max=2 时仅 0.59），说明模型学会了"按需拒绝"而非无脑扩张。
- **拒绝奖励权重 δ（表 8）**：δ=0.2 峰值（45.5%）；0.1 太弱、拒绝机制没用起来；0.3 太强、触发不必要拒绝、拖累准确率。

## 局限

1. **受限于检索器召回上界**：性能仍被 retriever 的 recall ceiling 卡住（R@15 上界与实际仍有差距）。
2. **候选图注入的 token 开销大**：把大量候选图塞进推理上下文带来额外 token 成本。
3. **未与检索器联合优化**：未来可探索压缩候选图信息、以及与检索器的联合优化。

## 个人理解与启发

- **这篇和上一篇 HiPRAG 是同一思路在不同模态上的呼应**：都在说"agentic RAG 光靠结果奖励不够，要给中间的检索/校验决策以步骤级监督"。HiPRAG 管的是文本 QA 里"该不该搜"，MMAgent-R² 管的是视觉 KB-VQA 里"这个候选到底对不对、要不要换一批"。**过程奖励 + RL 正在成为 agentic RAG 的标配范式**。
- **最漂亮的一点是把"拒绝"提升为一等公民动作**：传统 RAG 的候选集是死的，MMAgent-R² 让"拒绝"直接触发候选扩展，把固定候选池变成可动态探索的空间。而 ρ 里给正确拒绝一个 δ（<1）的奖励，既鼓励"没匹配就换"、又通过 δ<1 避免模型滥用拒绝——这个"不确定时宁可拒绝也不瞎猜"的价值取向定义得很干净。
- **视觉重排 = 让 VLM 做它最擅长的细粒度感知，而不是让 CLIP 全局特征硬扛**：检索器负责粗召回，VLM 负责逐图比对抓判别性细节。这种"检索器管覆盖、VLM 管精度"的分工，对任何"相似实体难区分"的检索场景都有借鉴意义。
- 一个务实的顾虑：把多批候选图塞进上下文，token 开销和推理延迟都不小（消融里 K 和 Rej_max 增大都会因上下文变长而反噬）。论文自己也把"压缩候选图信息 + 与检索器联合优化"列为未来方向——这恰恰是这套"往上下文里灌图"范式能否实用落地的关键瓶颈。
