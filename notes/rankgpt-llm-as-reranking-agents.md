---
id: rankgpt-llm-as-reranking-agents
title: "RankGPT：让 LLM 直接生成排列，把 ChatGPT 的排序能力蒸馏进 440M 小模型"
category: 多模态 RAG
date: 2026-07-24
order: 1
readTime: 14 min
tags: ["Listwise Reranking", "LLM", "Zero-shot", "Permutation Distillation", "RankNet", "BEIR", "TREC", "EMNLP 2023"]
summary: "RankGPT 是 listwise LLM 重排的奠基之作。它指出此前用 LLM 做 IR 的两种提示方式（查询生成 QG、相关性生成 RG）都要依赖输出的 log 概率且效果有限，提出「指令式排列生成」（PG）——把一组候选段落带编号一起喂给 LLM，直接输出 [2] > [3] > [1] 这样的排列；再用滑动窗口突破上下文长度限制。零样本 GPT-4 在 TREC/BEIR/Mr.TyDi 上分别超过有监督 SOTA 2.7/2.3/2.7 nDCG。为解决成本问题，又提出「排列蒸馏」：用 ChatGPT 对 10K MS MARCO 查询产出的排列作为标签、以 RankNet 损失训练小模型，蒸馏出的 435M DeBERTa 在 BEIR 上反超 3B 的 monoT5，并超过教师 ChatGPT 本身。"
---

## 一句话总结

RankGPT 回答了两个问题：**（RQ1）ChatGPT 能做段落重排吗？（RQ2）怎么把这种能力搬进一个小模型？**

针对 RQ1，作者发现此前把 LLM 用于 IR 的两条路子都不好：**查询生成（QG）** 要靠 LLM 输出的 log 概率反推"这段话生成该查询的可能性"，**相关性生成（RG）** 让 LLM 逐条判"相关/不相关"——两者都是 pointwise、都依赖 log-prob（很多 API 拿不到），且效果有限。于是提出 **指令式排列生成（Instructional Permutation Generation, PG）**：把一组候选段落各带一个编号 `[1] [2] [3]…` 一起塞进 prompt，让 LLM **直接输出降序排列** `[2] > [3] > [1] > …`，全程不产生中间相关性分数。再配一个**滑动窗口**策略（窗口 20、步长 10、从后往前扫）绕开上下文长度限制，让 100 个候选也能排。结果：零样本 GPT-4 在 TREC、BEIR、Mr.TyDi 上分别比有监督 SOTA（monoT5-3B）高 2.7、2.3、2.7 nDCG。为排除数据污染质疑，还新建了 **NovelEval** 测试集（GPT-4 发布后才出现的 21 个新问题）。

针对 RQ2，提出 **排列蒸馏（Permutation Distillation）**：从 MS MARCO 采 10K 查询、每个用 BM25 召回 20 个候选，把 **ChatGPT 输出的排列直接当训练目标**（不做一致性检查、不操作 log 概率），用 RankNet 成对损失训学生模型。蒸馏出的 435M DeBERTa 在 BEIR 上达到 53.03 nDCG，**超过 3B 的 monoT5（51.36），也超过它的老师 ChatGPT（49.37）**。

<div class="metric-grid">
  <div class="metric"><strong>+2.7 / +2.3 / +2.7</strong><span>零样本 GPT-4 相对有监督 SOTA 在 TREC / BEIR / Mr.TyDi 的 nDCG 提升</span></div>
  <div class="metric"><strong>53.03 vs 51.36</strong><span>蒸馏后 435M DeBERTa 与 3B monoT5 的 BEIR 平均 nDCG@10</span></div>
  <div class="metric"><strong>90.45</strong><span>GPT-4 在无污染 NovelEval 上的 nDCG@10（monoT5-3B 为 84.62）</span></div>
  <div class="metric"><strong>10 vs 100</strong><span>排列生成 vs pointwise 方法重排 100 个段落所需的 API 请求数</span></div>
</div>

## 论文信息

- **标题**：Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents
- **作者**：Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, Zhaochun Ren
- **机构**：山东大学、百度、莱顿大学（Leiden University）
- **会议**：**EMNLP 2023**（arXiv v1 2023-04-19，v3 2024-12-28）
- **链接**：[arXiv](https://arxiv.org/abs/2304.09542) · [PDF](https://arxiv.org/pdf/2304.09542) · [HTML](https://arxiv.org/html/2304.09542v3)

## 背景：为什么这个问题重要？

IR 系统由检索、重排等子模块串成，传统方法**重度依赖人工标注**（MS MARCO 那种），人力成本高且泛化性弱。LLM 有强大的零样本泛化能力，自然想用上——但当时的工作基本都在用 LLM **生成内容**（生成查询、生成伪文档），而不是**直接对一组段落做相关性排序**。

核心矛盾在于：**重排的目标和 LLM 的预训练目标差异很大**。重排要求模型理解用户需求、**全局比较**一组段落、再按相关性排序；而 LLM 的预训练是 next-token prediction。怎么把后者的能力引导到前者，是当时未被回答的问题。

<figure class="figure">
  <img src="assets/papers/rankgpt/fig1_overall_results.webp" alt="ChatGPT 与 GPT-4 在重排基准上的总体结果" loading="lazy" />
  <figcaption>图 1：ChatGPT 和 GPT-4（零样本）在段落重排基准（TREC、BEIR、Mr.TyDi）上的平均结果，与 BM25 及此前最佳有监督系统（如 monoT5）对比。</figcaption>
</figure>

## 方法一：指令式排列生成（PG）

<figure class="figure">
  <img src="assets/papers/rankgpt/fig2_three_instructions.webp" alt="三种零样本重排指令方式" loading="lazy" />
  <figcaption>图 2：三种零样本段落重排指令。灰块为输入、黄块为输出。(a) 查询生成 QG：依赖 LLM 的 log 概率，让模型基于段落生成查询；(b) 相关性生成 RG：让 LLM 输出相关性判断；(c) 排列生成 PG：直接生成一组段落的排序列表。</figcaption>
</figure>

**做法**（图 2c）：把 M 个候选段落各配一个唯一编号 `[1] [2] [3]…` 一起送进 LLM，指令要求按与查询的相关性**降序输出编号排列**，格式如 `[2] > [3] > [1]`。**不产生中间相关性分数**，排序就是输出本身。

**为什么 PG 比 QG/RG 好？**作者给了两条解释，都很有说服力：

1. **PG 让 LLM 显式地相互比较多个段落**，因此能辨别段落间的细微差异——这体现在 PG 的 top-1 准确率显著高于其他方法（表 4 里 curie-001 的 nDCG@1：PG 66.67 vs QG 50.78 vs RG 39.53）；
2. **读多个段落让 LLM 对查询和段落获得更全面的理解**，段落之间可能有互补信息，反过来提升排序能力。

这两点正是 **listwise 相对 pointwise 的本质优势**：候选之间可见、可比。

### 滑动窗口策略

<figure class="figure">
  <img src="assets/papers/rankgpt/fig3_sliding_window.webp" alt="滑动窗口重排示意" loading="lazy" />
  <figcaption>图 3：用窗口大小 4、步长 2 的滑动窗口重排 8 个段落。蓝色为前两个窗口、黄色为最后一个窗口。窗口按<strong>从后往前</strong>的顺序应用，也就是说前一个窗口里排在前面的 2 个段落会参与下一个窗口的重排。</figcaption>
</figure>

上下文长度装不下 100 个段落，所以用滑动窗口：**窗口 20、步长 10、从列表尾部往头部扫**。每个窗口重排后，靠前的段落会带入下一个（更靠前的）窗口继续参与比较，从而把好的候选逐步"冒泡"到全局前列。重排 100 个段落只需 **10 次 API 调用**（对比 pointwise 方法要 100 次）。

附录 G 分析了窗口大小：窗口 20/步长 10 的 nDCG@10 最好（67.05），窗口 40 在 nDCG@1/@5 上更好（78.30/71.32）——**窗口越大模型的"排序视野"越宽，但长上下文和大量条目的处理难度也越大**。

## 方法二：排列蒸馏（Permutation Distillation）

动机很实际：ChatGPT/GPT-4 **太贵、太慢、生成不稳定**——GPT-4 重排一个查询要 $0.596、32 秒。作者的判断是"**LLM 的能力对重排任务而言是冗余的**"，可以专门化蒸馏进小模型。

**关键设计**：直接把 **ChatGPT 输出的排列当作训练目标**，不引入任何归纳偏置（不做一致性检查、不操作 log 概率）——这是与 InPars、Sachan 等前作的本质区别。

- **数据**：MS MARCO 采 10K 查询，每个用 BM25 召回 20 个候选，让 ChatGPT 用 PG 排出顺序。
- **损失**：RankNet 成对损失。$M=20$ 个候选可构造 $M(M-1)/2 = 190$ 个偏序对：

\\[
\mathcal{L}_{\text{RankNet}} = \sum_{i=1}^{M}\sum_{j=1}^{M} \mathbb{1}_{r_i < r_j} \log\left(1 + \exp(s_j - s_i)\right)
\\]

- **学生架构**两种：**BERT 式** cross-encoder（DeBERTa-v3-Large，`[CLS]` 出相关性分数）和 **GPT 式**（LLaMA-7B，用相关性生成指令，取相关性 token 的生成概率当分数）。

## 数据与评测

- **TREC-DL**：DL19（43 查询）、DL20（54 查询）
- **BEIR** 八个子集：Covid、NFCorpus、Touche、DBPedia、SciFact、Signal、News、Robust04
- **Mr.TyDi**：十种低资源语言（阿拉伯、孟加拉、芬兰、印尼、日、韩、俄、斯瓦希里、泰卢固、泰），各取测试集前 100 条
- **NovelEval-2306**（本文新建）：21 个 GPT-4 发布后才出现的问题（如"2023 年金棕榈奖是哪部电影？"——戛纳 2023-05-27 才颁），用 Google 搜 20 个候选、人工标 0/1/2 三级相关性。**目的是排除数据污染，验证 LLM 能否排序它压根没学过的知识**。
- 统一重排 BM25 的 top-100（pyserini），指标 nDCG@{1,5,10}。

## 实验结果

### TREC 与 BEIR（表 1，nDCG@10）

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>DL19</th><th>DL20</th><th>BEIR (Avg)</th></tr></thead>
    <tbody>
      <tr><td>BM25</td><td>50.58</td><td>47.96</td><td>43.42</td></tr>
      <tr><td>monoBERT (340M，有监督)</td><td>70.50</td><td>67.28</td><td>47.16</td></tr>
      <tr><td>monoT5 (3B，有监督 SOTA)</td><td>71.83</td><td>68.89</td><td>51.36</td></tr>
      <tr><td>Cohere Rerank-v2</td><td>73.22</td><td>67.08</td><td>49.45</td></tr>
      <tr><td>UPR (FLAN-T5-XL，无监督)</td><td>53.85</td><td>56.02</td><td>42.99</td></tr>
      <tr><td>gpt-3.5-turbo + PG</td><td>65.80</td><td>62.91</td><td>49.37</td></tr>
      <tr><td><strong>gpt-4 + PG</strong></td><td><strong>75.59</strong></td><td><strong>70.56</strong></td><td><strong>53.68</strong></td></tr>
    </tbody>
  </table>
</div>

- **GPT-4 全面超越有监督 SOTA**：相比 monoT5-3B，TREC/BEIR 平均分别 +2.7 / +2.3 nDCG@10。
- **ChatGPT 在 BEIR 上就已超过大多数有监督基线**（49.37 vs monoBERT 47.16）。
- **成本技巧**：BEIR 上先用 ChatGPT 排，再只用 GPT-4 重排 top-30——效果好且**成本只有纯 GPT-4 的 1/5**（$0.098 vs $0.596 每查询）。

### Mr.TyDi 多语言（表 2）

GPT-4 平均 62.93 nDCG@10，超过有监督 mmarcoCE（60.28）约 2.65。但**在孟加拉、泰卢固、泰语等极低资源语言上反而输给 mmarcoCE**——作者归因于 GPT-4 在这些语言上的语言建模能力较弱，且低资源语言文本消耗更多 token，导致段落被过度截断。

### NovelEval：无污染场景（表 3）

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>nDCG@1</th><th>nDCG@5</th><th>nDCG@10</th></tr></thead>
    <tbody>
      <tr><td>BM25</td><td>33.33</td><td>45.96</td><td>55.77</td></tr>
      <tr><td>monoBERT (340M)</td><td>78.57</td><td>70.65</td><td>77.27</td></tr>
      <tr><td>monoT5 (3B)</td><td>83.33</td><td>78.38</td><td>84.62</td></tr>
      <tr><td>gpt-3.5-turbo</td><td>76.19</td><td>74.15</td><td>75.71</td></tr>
      <tr><td><strong>gpt-4</strong></td><td><strong>85.71</strong></td><td><strong>87.49</strong></td><td><strong>90.45</strong></td></tr>
    </tbody>
  </table>
</div>

GPT-4 在完全没学过的新知识上仍显著超过 monoT5-3B（90.45 vs 84.62），**说明 LLM 的重排能力来自真正的理解和推理，不是记忆泄漏**。这个实验设计在 2023 年很有前瞻性。

### 排列蒸馏结果（表 7）

<div class="table-wrap">
  <table>
    <thead><tr><th>训练标签</th><th>模型</th><th>DL19</th><th>DL20</th><th>BEIR (Avg)</th></tr></thead>
    <tbody>
      <tr><td>—</td><td>BM25</td><td>50.58</td><td>47.96</td><td>43.42</td></tr>
      <tr><td>—</td><td>ChatGPT（教师）</td><td>65.80</td><td>62.91</td><td>49.37</td></tr>
      <tr><td>MS MARCO</td><td>monoT5 (3B)</td><td>71.83</td><td>68.89</td><td>51.36</td></tr>
      <tr><td>MS MARCO</td><td>DeBERTa-Large</td><td>68.89</td><td>61.38</td><td>42.64</td></tr>
      <tr><td>MS MARCO</td><td>LLaMA-7B</td><td>69.24</td><td>58.97</td><td>47.71</td></tr>
      <tr><td><strong>ChatGPT 排列</strong></td><td><strong>DeBERTa-Large (435M)</strong></td><td>70.66</td><td>67.15</td><td><strong>53.03</strong></td></tr>
      <tr><td>ChatGPT 排列</td><td>LLaMA-7B</td><td>71.78</td><td>66.89</td><td>51.68</td></tr>
    </tbody>
  </table>
</div>

三个关键发现：

1. **排列蒸馏 > 同架构的有监督训练**（DeBERTa 在 BEIR 上 53.03 vs 42.64，差距巨大）。作者的解释是 **ChatGPT 的相关性判断比 MS MARCO 标注更全面**——MS MARCO 是浅池标注，很多真正相关的段落被标成负例（引用 Arabzadeh et al. 2021 的"shallow pooling"问题）。
2. **435M 学生在 BEIR 上超过 3B 的 monoT5**（53.03 vs 51.36），参数量少 7 倍。
3. **学生超过了它的老师 ChatGPT**（53.03 vs 49.37）——原因是**小模型的排序稳定性远好于 ChatGPT**（见下面的稳定性分析）。

<figure class="figure">
  <img src="assets/papers/rankgpt/fig4_scaling.webp" alt="模型规模与数据规模的 scaling 实验" loading="lazy" />
  <figcaption>图 4：Scaling 实验。虚线为基线（GPT-4、monoT5、monoBERT、ChatGPT）；绿实线为排列蒸馏得到的 DeBERTa，灰实线为在 MS MARCO 上有监督学习的 DeBERTa。对比不同模型规模（70M–435M）和训练数据量（500–10K）下在 TREC 与 BEIR 上的表现。</figcaption>
</figure>

Scaling 的三点观察：**① 蒸馏模型在所有规模/数据量设置下都稳定超过有监督对照，BEIR 上尤为明显；② 只用 1K 训练查询，蒸馏出的 DeBERTa 就已超过 monoT5-3B；③ 增加模型参数带来的收益大于增加训练数据**。而有监督模型的表现随规模变化很不稳定，作者归因于 MS MARCO 标签噪声导致的过拟合。

## Ablation：哪些设计最关键？

### 初始顺序极其关键（表 5）

<div class="table-wrap">
  <table>
    <thead><tr><th>配置（gpt-3.5-turbo + PG）</th><th>nDCG@1</th><th>nDCG@5</th><th>nDCG@10</th></tr></thead>
    <tbody>
      <tr><td>标准（BM25 初始顺序）</td><td><strong>82.17</strong></td><td><strong>71.15</strong></td><td>65.80</td></tr>
      <tr><td>随机初始顺序</td><td>26.36</td><td>25.32</td><td>25.17</td></tr>
      <tr><td>反转 BM25 顺序</td><td>36.43</td><td>31.79</td><td>32.77</td></tr>
      <tr><td>重排 2 轮</td><td>78.29</td><td>69.37</td><td>66.62</td></tr>
      <tr><td>重排 3 轮</td><td>78.29</td><td>69.74</td><td>66.97</td></tr>
      <tr><td>GPT-4 重排 top-30</td><td>80.23</td><td>76.70</td><td><strong>73.64</strong></td></tr>
    </tbody>
  </table>
</div>

**把初始顺序打乱，nDCG@10 从 65.80 崩到 25.17**（比 BM25 的 50.58 还差一半）——这是全文最触目惊心的数字。说明 PG **严重依赖一阶段检索给出的良好起点**，滑动窗口只是在好的初始顺序上做局部精修，不具备"从混乱中重建全局排序"的能力。这也是作者在 Limitations 里坦承的核心弱点。

**多轮重排的效果是矛盾的**：nDCG@10 略升（65.80 → 66.97），但 nDCG@1 反而掉了近 3.9（82.17 → 78.29）——多扫几遍会扰乱头部。

### 稳定性分析（附录 F，表 10）

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>重复编号 ↓</th><th>漏掉编号 ↓</th><th>拒绝排序</th><th>RBO ↑（两次排序一致性）</th></tr></thead>
    <tbody>
      <tr><td>text-davinci-003</td><td>0</td><td>280</td><td>0</td><td>72.30</td></tr>
      <tr><td>gpt-3.5-turbo</td><td>14</td><td>153</td><td>7</td><td>81.49</td></tr>
      <tr><td><strong>gpt-4</strong></td><td>0</td><td><strong>1</strong></td><td>11</td><td><strong>82.08</strong></td></tr>
    </tbody>
  </table>
</div>

这张表非常有价值——它量化了**生成式排序的固有脆弱性**：编号重复、编号漏掉（davinci 漏了 280 次！实现里只能把漏掉的段落按原顺序补到末尾）、模型拒绝排序（"这些段落都不相关…"）、两次排序不一致。GPT-4 显著更稳定（只漏 1 个），这正是它效果最好的重要原因；而 **ChatGPT 的不稳定性也解释了为什么蒸馏出的小模型能反超老师**。

### 其他 LLM（表 6，重排 top-20）

专有模型里 GPT-4 最强（nDCG@10 65.68）；Claude-2（55.91）和 Google text-bison（58.67）落后于 ChatGPT。**开源模型差距巨大**：FLAN-T5-XXL 50.26、ChatGLM-6B 50.58、Vicuna-13B 49.08——基本等于没重排（BM25 是 50.58）。作者归因于"生成 20 个段落的排列"对当时的开源模型太难。

## 我的理解与启发

- **这篇是 listwise LLM 重排的源头，理解它才能理解后面一整条技术线**。它确立的两个范式今天还在用：**① 编号 + 直接生成排列**（"[2] > [3] > [1]"），**② 用强 LLM 的排列做蒸馏标签 + RankNet 损失**。我笔记里的 [ZipRerank](/#/note/ziprerank-efficient-listwise-multimodal-reranking) 几乎是 RankGPT 在多模态长文档上的直系后代：同样是编号标识符、同样用 RankNet 督导、同样用强模型（GPT-5）蒸馏——**连"学生反超弱教师"这个现象都被复现了**（RankGPT: DeBERTa 53.03 > ChatGPT 49.37；ZipRerank: 学生 > GPT-5-nano）。
- **"学生超过老师"的机制值得单独记住：不是学生更聪明，而是学生更稳定**。表 10 显示 ChatGPT 会重复编号、漏 153 次编号、拒绝排序——这些格式故障每次都在损失精度。小模型输出的是确定性分数，不存在这类问题。**这也解释了 ZipRerank 单 token 解码为什么是对的方向**：把"生成一个排列字符串"换成"读一次 logit"，从根上消灭了重复/漏项/格式错乱。RankGPT 时代只能靠 GPT-4 的强稳定性硬扛，后来的工作用架构设计绕开了它。
- **初始顺序敏感性（65.80 → 25.17）是 listwise 重排至今未彻底解决的软肋**。它意味着重排器不是独立的质量提升器，而是**依附于一阶段召回质量的放大器**。滑动窗口本质是局部冒泡排序，缺乏全局视野。这个观察至今有效——也解释了为什么后续工作（包括 MCMR 那篇）都强调一阶段召回质量的重要性。
- **"LLM 的能力对重排任务而言是冗余的"这个判断非常有洞察力**，而且被反复验证：RankGPT 用 435M 打败 3B，miniReranker 用早退证明重排信号在中间层就收敛，ZipRerank 用单 token 解码证明不需要自回归。**重排是一个判别任务，被硬塞进生成模型的壳子里；后续所有效率工作本质上都在把这层壳子拆掉。**
- **NovelEval 的设计思路值得借鉴到任何 LLM 评测**：与其争论"模型是不是背过测试集"，不如直接建一个模型发布之后才出现的测试集。低成本（21 个问题）、高说服力。

## 局限与待观察点

1. **主要结论建立在闭源模型上**：ChatGPT/GPT-4 不开源、行为会随版本漂移；作者也测了 FLAN-T5、ChatGLM-6B、Vicuna-13B，但**效果基本等于没重排**，"如何让开源模型做好 PG"在当时是未解问题（这个问题后来才被 RankVicuna/RankZephyr 等工作推进）。
2. **上限受一阶段召回限制**，且**对初始顺序极度敏感**——作者在 Limitations 里明确点出这是最需要后续探索的方向。
3. **成本高昂**：GPT-4 重排一个查询 $0.596、约 32 秒（10 次 API 调用 × 3.2s）；整个 GPT-4 实验花了 **$556**。排列蒸馏正是对这个问题的回应。
4. **低资源语言上不敌有监督模型**：token 消耗更多导致段落被截断，暴露了"把多语言文本塞进有限上下文"的结构性问题。
5. **滑动窗口的超参没有普适最优**：窗口 20 的 nDCG@10 最好，窗口 40 的 nDCG@1/@5 最好，需按指标取舍。

## 结论

RankGPT 用一个极简的想法（**给候选编号，让 LLM 直接输出排列**）证明了零样本 LLM 能在段落重排上超过精心训练的有监督系统，并用 NovelEval 排除了数据污染的质疑；又用**排列蒸馏**把这种昂贵能力压进 435M 的小模型，性能反超 3B 基线甚至教师本身。它同时给出了范式（PG + 滑动窗口）、诊断（初始顺序敏感、生成不稳定）和落地路径（蒸馏），是 2023 年之后所有 listwise LLM 重排工作的共同起点。
