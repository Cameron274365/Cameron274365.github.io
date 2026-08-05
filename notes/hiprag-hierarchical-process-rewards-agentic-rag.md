---
id: hiprag-hierarchical-process-rewards-agentic-rag
title: "HiPRAG：用分层过程奖励治好检索智能体的「过度搜索」与「搜索不足」"
category: 多模态 RAG
date: 2026-07-14
order: 1
readTime: 14 min
tags: ["Agentic RAG", "Reinforcement Learning", "Process Reward", "Search Agent", "PPO", "GRPO", "Over-search", "ICLR 2026"]
summary: "HiPRAG 针对 agentic RAG 中普遍存在的「过度搜索（明明已知却还去检索）」和「搜索不足（该查不查导致幻觉）」问题，提出把细粒度、知识感知的过程奖励引入 RL 训练：把推理轨迹拆成可解析的步骤，在线检测每一步搜索决策是否最优，再按最优步骤比例给一个分层 bonus（在结果奖励+格式奖励之上）。在 7 个 QA benchmark 上，3B/7B 平均准确率达 65.4%/67.2%，同时把过度搜索率从 27%+ 压到 2.3%。ICLR 2026 接收。"
---

## 一句话总结

HiPRAG 的核心洞察是：**训练检索智能体时只用「最终答案对不对」这种结果奖励（outcome reward）太粗糙，管不住中间的搜索行为**——模型会一边过度搜索（已经知道的还去查，浪费开销）、一边搜索不足（该查的不查，直接编造导致幻觉）。解法是把推理轨迹拆成一步步可用规则解析的离散步骤，在 RL rollout 时**在线判断每一步的搜索决策是否"最优"**（该搜的搜了、不该搜的没搜），然后在常规的结果奖励+格式奖励之上，按"最优步骤占比"追加一个**分层过程奖励 bonus**。关键设计是这个 bonus 被"门控"——只有当答案正确且格式正确时才发放，从而避免为了省搜索而牺牲准确率。

<div class="metric-grid">
  <div class="metric"><strong>65.4% / 67.2%</strong><span>3B / 7B 在 7 个 QA benchmark 上的平均 CEM 准确率，超越所有基线</span></div>
  <div class="metric"><strong>27%+ → 2.3%</strong><span>过度搜索率（OSR）大幅下降</span></div>
  <div class="metric"><strong>3B > 7B baseline</strong><span>HiPRAG-3B (64.4%) 超越常规奖励训练的 7B (61.2%)</span></div>
  <div class="metric"><strong>ICLR 2026</strong><span>跨 PPO/GRPO、Qwen2.5/Llama-3.2、base/instruct 均有效</span></div>
</div>

## 论文信息

- **标题**：HiPRAG: Hierarchical Process Rewards for Efficient Agentic Retrieval Augmented Generation
- **作者**：Peilin Wu, Mian Zhang, Kun Wan, Wentian Zhao, Kaiyu He, Xinya Du, Zhiyu Chen
- **机构**：The University of Texas at Dallas、Adobe Inc.
- **版本**：arXiv:2510.07794（v1 2025-10-09；v2 2026-04-11，ICLR 2026 接收）
- **链接**：[arXiv](https://arxiv.org/abs/2510.07794) · [PDF](https://arxiv.org/pdf/2510.07794) · [HTML](https://arxiv.org/html/2510.07794v2)

## 背景：结果奖励管不住搜索行为

Agentic RAG 让 LLM 在推理过程中自主决定"何时该调用检索工具"，弥补参数化知识的不足。但论文指出两类普遍存在的次优搜索行为：

- **过度搜索（over-search）**：检索本已掌握的信息，带来不必要的延迟和开销，还可能被无关结果干扰导致答错；
- **搜索不足（under-search）**：该检索时不检索，凭参数化知识硬答，导致幻觉或事实错误。

现有 RL 训练方法（如 Search-R1、R1-Searcher、β-GRPO）主要依赖**结果奖励**——只看最终答案对不对。这种信号太粗，缺乏对"搜索决策本身是否合理"的细粒度控制，因此难以同时优化准确率和效率。论文给出的证据很直接：baseline 的过度搜索率超过 27%。

HiPRAG 的主张是：**要优化的是推理过程本身，而不只是最终结果**。

## 方法：三个组件

<figure class="figure">
  <img src="assets/papers/hiprag/fig1_workflow.webp" alt="HiPRAG 训练流程总览" loading="lazy" />
  <figcaption>图 1：HiPRAG 训练流程总览。策略模型生成多步推理轨迹，每一步被在线评估以检测次优搜索行为；最终的分层奖励把"步骤最优性"的过程 bonus 与"答案正确性 + 格式规范"的奖励组合起来。</figcaption>
</figure>

### 1. 把推理轨迹拆成可解析的步骤（§3.1）

重新设计了一套显式结构化的输出格式，用 XML 标签把轨迹切成离散的、可用规则解析的步骤。每一步要么是**搜索步（search step, sᴿ）**，要么是**非搜索步（non-search step, sᴺᴿ，即纯内部推理）**。通过两条途径保证模型遵守 schema：① 系统提示里给出显式指令和 few-shot 示例；② RL 用格式奖励激励模型持续产出可解析的轨迹。

### 2. 在线检测次优搜索（§3.2）

轨迹切分后，就能在 RL 训练阶段对每一步做高效检查：

- **过度搜索检测（Over-search）**：对每个搜索步，把该步"不使用检索结果、仅靠模型自身参数化知识"重新生成一遍（re-generation），再用外部 LLM 裁判判断——如果不搜也能正确得到该步结论，说明这次检索是多余的，判为过度搜索。为提速，re-generation 可以在 rollout 阶段批量单独执行，再交给 LLM 裁判。实现中用 **gpt-4.1-mini** 做过度搜索裁判。
- **搜索不足检测（Under-search）**：对每个非搜索步 sᴺᴿ=(rᵢ, oᵢ)，用外部裁判核验其推理 rᵢ 和结论 oᵢ 的事实与逻辑正确性。若内容错误，说明该步本应检索却没检索、产生了幻觉，判为搜索不足。实现中用 **gpt-5-mini** 做裁判。

两个检测可并发执行以提升速度。

### 3. 分层过程奖励（§3.3）

设 A(T)∈{0,1} 表示最终答案是否正确（用 Cover Exact Match），F(T)∈{0,1} 表示轨迹是否符合格式；N(T) 是轨迹步数，N_corr(T) 是"最优步"数量（搜索步里非过度搜索的 + 非搜索步里非搜索不足的）。奖励定义为：

\[R(T) = A(T)(1-\lambda_f) + \lambda_f F(T) + \lambda_p\, A(T)\,F(T)\,\frac{N_{\text{corr}}(T)}{N(T)}\]

其中 λ_f 是格式权重、λ_p 是过程 bonus 系数。这个式子的巧妙之处：

- 当 λ_p=0 时，它**退化为**先前工作常用的"结果 + 格式"奖励；
- 过程 bonus 项被 A(T)·F(T) **门控**——只有答案对且格式对时才发放。此时奖励变为 R(T) = 1 + λ_p·(N_corr/N)。

这种分层结构保证智能体**先**学会产出格式规范、答案正确的轨迹，**只有达成这个首要目标后**，才因推理路径的效率与有效性拿到额外 bonus。这避免了"为省搜索而过度抑制检索"的陷阱，同时引导模型形成对自身知识边界的更细腻认知。主实验取 λ_f=0.2、λ_p=0.4。

## 实验设置

- **数据与指标**：仿照 Search-R1，训练集用 NQ + HotpotQA 的官方训练集（覆盖单跳与多跳）。评测用 7 个 QA 数据集：NQ、PopQA、TriviaQA、2WikiMultiHopQA、Bamboogle、HotpotQA、Musique。主指标是 **Cover Exact Match (CEM)**（判断标准答案串是否出现在生成答案中，比严格 EM 更适配现代 LLM 的冗长输出）；效率指标是 **过度搜索率 OSR** 和 **搜索不足率 USR**。
- **模型与训练**：主实验用 Qwen2.5-(3B/7B)-Instruct，另用 Llama-3.2-3B-Instruct 和 Qwen2.5-3B 验证泛化性。RL 主用 PPO，另试 GRPO（group size 5）。检索环境沿用 Search-R1：2018 Wikipedia dump + E5-base 检索器，每次搜索返回 top-3 段落。4×A100 80GB，训练 400 步。

## 实验结论

**主结果（表 1）**：HiPRAG 在 3B 和 7B 上都超越所有基线。HiPRAG-7B 平均 CEM 67.2%，明显高于次优基线 R1-Searcher++（62.2%）；HiPRAG-3B 达 65.4%。

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>平均 CEM</th></tr></thead>
    <tbody>
      <tr><td>Standard RAG</td><td>45.3</td></tr>
      <tr><td>Search-o1</td><td>43.9</td></tr>
      <tr><td>Search-R1</td><td>60.3</td></tr>
      <tr><td>R1-Searcher++</td><td>62.1</td></tr>
      <tr><td>β-GRPO</td><td>62.5</td></tr>
      <tr><td><strong>HiPRAG-3B</strong></td><td><strong>65.4</strong></td></tr>
      <tr><td><strong>HiPRAG-7B</strong></td><td><strong>67.2</strong></td></tr>
    </tbody>
  </table>
</div>

**效率与泛化（表 2）**：process bonus（相对 λ_p=0 的 baseline）在几乎所有配置上都同时提升准确率、降低 OSR 与 USR。最佳配置 Qwen2.5-7B + GRPO 达到 67.2% CEM，OSR 低至 2.3%、USR 32.6%。

<figure class="figure">
  <img src="assets/papers/hiprag/fig2a_reward_curves.webp" alt="PPO 与 GRPO 的奖励曲线" loading="lazy" />
  <figcaption>图 2(a)：Qwen2.5-3B-Instruct 用 PPO / GRPO + HiPRAG 训练的奖励曲线。GRPO 收敛更快、峰值更高，但稳定性略逊；PPO 更稳、常能跑完全程不崩。</figcaption>
</figure>

几个值得记的发现：

- **小模型追平大模型**：HiPRAG 的 Qwen2.5-3B + GRPO（64.4%）不仅超过外部 7B 基线 R1-Searcher++（62.2%），还超过用常规奖励训练的 7B（61.2%）——**优化过程比单纯堆模型规模更有效**。
- **RL 算法**：GRPO 无 critic、样本更高效、收敛快、峰值高（7B 上 67.2% vs PPO 64.5%，且 OSR 2.3% vs 6.2%），代价是稳定性差；PPO 更稳定。
- **模型家族**：Llama-3B 初期更依赖参数化知识（非搜索步多、USR 更高）；训练后 Qwen-3B 以更低的次优率达到高准确率。基座本身的倾向会影响最终搜索效率。
- **指令微调**：instruct 模型初期奖励更高（更会遵循格式），但 base 模型最终能追平，甚至 CEM 略高（64.5% vs 64.1%）、OSR 更低（3.2% vs 4.9%），可能因为它更"纯粹"地从 RL 目标学习、没有指令微调引入的偏差。

<figure class="figure">
  <img src="assets/papers/hiprag/fig2b_search_ratio.webp" alt="搜索步占比随训练变化" loading="lazy" />
  <figcaption>图 2(b)：Qwen2.5-3B 与 Llama-3.2-3B 训练过程中搜索步在全部推理步里的占比曲线。不同家族的初始搜索倾向差异明显。</figcaption>
</figure>

**消融（表 2 下半 + §5.3）**：

- **输出格式**：把 HiPRAG 的可解析格式套用到 Search-R1 / β-GRPO（记为 -step*），性能基本持平甚至略升，证明**增益来自过程奖励机制本身，而非格式改动的副作用**。
- **过程 bonus 系数 λ_p**：0.4 最优（64.1% CEM）；0.2 太小、接近纯结果奖励（59.6%，OSR/USR 偏高）；0.6 过分追求"步骤纯度"反而牺牲答案正确率（62.5%）。
- **只惩罚一种**：只惩过度搜索 → 模型变得不敢搜，USR 飙到 52.7%、CEM 仅 58.8%；只惩搜索不足 → USR 大降到 16.9% 但 OSR 微升到 6.6%、CEM 63.3%（说明防幻觉比提效率更关键）。**两者同时惩罚才最好（64.1%）**。

## 局限

1. **依赖外部 LLM 裁判**：过度/搜索不足检测都要调用 gpt-4.1-mini / gpt-5-mini 做在线判断，训练成本和对裁判质量的依赖不可忽视。
2. **USR 仍偏高**：搜索不足率即便优化后仍在 30% 上下，说明"何时必须检索"的判断还有较大改进空间。
3. **任务范围**：实验集中在知识密集型 QA（单跳/多跳），未覆盖更开放的工具调用或多模态检索场景。

## 个人理解与启发

- **这是一篇把"过程监督（process reward）"思想干净地落到 agentic RAG 上的工作**。相比 PRM 在数学推理里给每步打分，HiPRAG 的过程奖励非常具体、可规则化——就盯"这一步该不该搜"这一个决策，还给了 over/under-search 两个可在线检测的算子。信号定义得越具体，RL 越容易学到想要的行为。
- **门控设计（A·F 才发 bonus）是全篇最值得借鉴的一点**：过程奖励最怕"reward hacking"——模型为了拿过程分而牺牲主目标。HiPRAG 用严格的层级门控把"先答对、再谈效率"钉死，消融里 λ_p=0.6 的退化也印证了这个平衡的必要性。任何想加辅助奖励的 RL 系统都可以照搬这个"门控 + 小系数"的思路。
- **"不搜也能答对 = 这次搜索多余"这个过度搜索判据很朴素但有效**：本质是在线探测模型的知识边界。可迁移到更一般的工具调用场景——判断"这次工具调用是否必要"，都可以用"去掉工具结果重新生成 + 裁判比对"的范式。
- 一个开销上的顾虑：训练时每个搜索步都要 re-generation + LLM 裁判，rollout 成本相当高。论文靠批量生成缓解，但如果换成更长的轨迹或更贵的裁判，这套流水线的性价比还需要重新算账。这也呼应了它 USR 仍偏高、检测质量受裁判能力限制的局限。
