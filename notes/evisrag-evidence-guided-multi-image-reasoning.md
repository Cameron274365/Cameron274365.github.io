---
id: "evisrag-evidence-guided-multi-image-reasoning"
title: "EVisRAG：证据引导的多图像推理与奖励作用域强化学习"
category: "多模态 RAG"
date: "2026-06-18"
order: 20
readTime: "15 min"
tags: ["Visual RAG","Multi-Image Reasoning","RS-GRPO","VLM","Evidence Grounding","Reinforcement Learning"]
summary: "EVisRAG 提出 observe → record evidence → reason → answer 四阶段视觉 RAG 推理范式，并引入 RS-GRPO（Reward-Scoped GRPO）将感知奖励和推导奖励分别绑定到对应 token 作用域，在 5 个 VQA benchmark 上平均 F1 超过 Qwen2.5-VL-7B backbone 27 个百分点。"
hero: "assets/papers/evisrag/page3_framework.webp"
---

## 一句话总结
EVisRAG 的核心思路是让 VLM 像侦探一样工作：**先逐页观察检索到的图像并记录每页证据，再基于汇聚的证据进行推理得出答案**。配套的 RS-GRPO 算法将不同奖励信号精确绑定到对应的 token 区间（感知奖励 → observe/evidence 区间，推导奖励 → reason/answer 区间），解决了混合奖励导致的信用分配模糊问题，在 7B 模型上超越了 32B baseline。

<div class="metric-grid">
  <div class="metric"><strong>+27% F1</strong><span>相比 Qwen2.5-VL-7B backbone 的平均 F1 提升</span></div>
  <div class="metric"><strong>75.01 Acc</strong><span>5 个 VQA benchmark 平均准确率，超过 OpenVLThinker 约 10 个百分点</span></div>
  <div class="metric"><strong>7B > 32B</strong><span>7B 参数的 EVisRAG 超过 Qwen2.5-VL-32B（75.01 vs 71.43 Acc）</span></div>
  <div class="metric"><strong>~100s</strong><span>推理延迟约 100s，与最强 baseline 的 90-95s 接近，无显著额外开销</span></div>
</div>

## 论文信息
- **标题**：VisRAG 2.0: Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation
- **作者**：Yubo Sun, Chunyi Peng, Yukun Yan, Shi Yu, Zhenghao Liu, Chi Chen, Zhiyuan Liu, Maosong Sun
- **机构**：Peking University, Northeastern University, Tsinghua University
- **版本**：arXiv 2510.09733 (Oct 2025)
- **链接**：[arXiv](https://arxiv.org/abs/2510.09733) · [PDF](https://arxiv.org/pdf/2510.09733) · [代码](https://github.com/OpenBMB/VisRAG)

## 背景：为什么现有 VRAG 在多图像场景下表现不佳？

Visual RAG (VRAG) 将文档页快照作为检索单元，让 VLM 直接从图像中读取证据，避免了 OCR/Captioning 管线对视觉信息的丢失。但在 **多图像推理** 场景下，现有方法存在三个关键问题：

1. **跨图像证据感知不稳定**：当前 VRAG 方法（如 R1-Router、MMSearch-R1）大多移植文本 RAG 范式到视觉模态，缺乏对多图场景的跨图像 grounding、layout-aware reading 和区域级注意力。模型在多图上下文中容易遗漏或误解关键证据。

2. **感知与推理混合奖励导致信用模糊**：现有 VLRM 训练策略（如 VLM-R1、Mixed-R1）用同一个答案正确性奖励同时优化感知和推理，忽略了两种信号的作用范围和目标差异。感知奖励应作用于观察/记录区间，推理奖励应作用于推导区间——混合使用会模糊 credit assignment，导致训练不稳定。

3. **外部 agent/工具增加架构复杂度**：一些工作（如 VRAG-RL、ViDoRAG）引入视觉感知动作空间或多 agent 框架来改善感知，但增加了架构复杂度和计算成本，难以端到端训练和后续重构。

<figure class="figure">
  <img src="assets/papers/evisrag/page1_fig1.webp" alt="EVisRAG 与 normal VLRM 对比" loading="lazy" />
  <figcaption>图 1：Normal VLRM 直接对检索图像进行端到端推理，容易在多图场景中遗漏证据；EVisRAG 先逐页观察并记录每页证据，再基于证据推理，产出更准确的答案。</figcaption>
</figure>

## 方法：EVisRAG 如何实现证据引导的多图像推理？

### 1. 四阶段证据引导推理范式

EVisRAG 的输出被特殊 token 分割为四个作用域（scope）：

- **Observe**（`<observe>...</observe>`）：逐页观察检索到的图像，生成每页的粗粒度描述，识别页面包含哪些内容。
- **Record Evidence**（`<evidence>...</evidence>`）：为每张图像记录与问题相关的证据，格式为 `[i]: evidence text`；若图像无相关信息则记录 `[i]: no relevant information`。
- **Reason**（`<think>...</think>`）：基于已记录证据进行侦探式推理——交叉验证不同页面的信息、检验假设、排查矛盾。
- **Answer**（`<answer>...</answer>`）：输出最终答案。

整个流程为：\(q, D \xrightarrow{\text{VisRAG-Ret}} D_R = \{d_i\}_{i=1}^{k} \xrightarrow{\text{Observe}} r_{\text{observe}} \xrightarrow{\text{Evidence}} r_{\text{evidence}} \xrightarrow{\text{Reason}} r_{\text{reason}} \xrightarrow{\text{Answer}} a\)

### 2. RS-GRPO：奖励作用域绑定的强化学习

RS-GRPO 是论文的核心算法贡献。它定义了三种细粒度奖励，并将每种奖励绑定到其对应的 token 作用域：

| 奖励 | 作用 | 绑定的 token 作用域 |
|------|------|---------------------|
| **Format Reward** \(R_{\text{format}}\) | 强制模型遵守四阶段输出格式 | 所有 token |
| **Perception Reward** \(R_{\text{perception}}\) | 检查每页证据区域是否被正确定位和摘要 | Observe + Evidence 区间 |
| **Derivation Reward** \(R_{\text{derivation}}\) | 评估最终答案是否正确 | Reason + Answer 区间 |

形式化地，位置 \(t\) 的 reward-scope 映射为：

\[
M(t) = \begin{cases} \{R_{\text{perception}}, R_{\text{format}}\} & t \in T_o \cup T_e \\ \{R_{\text{derivation}}, R_{\text{format}}\} & t \in T_r \cup T_a \end{cases}
\]

每个 token 的聚合奖励为其作用域内奖励通道的均值：

\[
\bar{R}^i_t = \frac{1}{|M(t)|} \sum_{m \in M(t)} R^{(m),i}_t
\]

然后使用标准 GRPO 的组内归一化计算 token-level advantage：

\[
\hat{A}^i_t = \frac{\bar{R}^i_t - \text{mean}(\{\bar{R}^1_t, \ldots, \bar{R}^G_t\})}{\text{std}(\{\bar{R}^1_t, \ldots, \bar{R}^G_t\})}
\]

这个设计的关键洞察是：**感知奖励不应该作用于推理区间（否则会干扰推理学习），推理奖励也不应该作用于观察区间（否则会模糊感知学习）**。

<figure class="figure">
  <img src="assets/papers/evisrag/page3_framework.webp" alt="EVisRAG 整体框架与 RS-GRPO" loading="lazy" />
  <figcaption>图 2：EVisRAG 框架。Stage 1 为 SFT 冷启动（Long CoT cold start），Stage 2 为 RS-GRPO 策略模型更新。模型输出四个 token scope，三种细粒度奖励分别绑定到对应 scope 的 token 上，scope 内奖励均值化后做组归一化得到 token advantages。</figcaption>
</figure>

### 3. 感知奖励设计

感知奖励评估模型是否从每张图像中提取了有用的视觉信息：

\[
R_{\text{perception}} = \frac{\sum_{i=1}^{n} r_i}{n}
\]

其中，对包含问题相关信息的图像，\(r_i\) 为模型预测证据与 gold evidence 的 F1 分数；对不相关图像，如果模型正确标记为 "no relevant information" 则 \(r_i = 1\)，否则 \(r_i = 0\)。Gold evidence 由更大的 VLM（Qwen2.5-VL-72B）预先生成。

### 4. 两阶段训练流程

- **Stage 1 — SFT 冷启动**：使用 60K 高质量推理轨迹进行监督微调。轨迹由 Qwen2.5-VL-72B 和 Qwen2.5-VL-7B 生成候选 chain-of-thought，只保留产生正确答案的轨迹。训练 1 epoch，lr=5e-7。
- **Stage 2 — RS-GRPO**：使用 4K 精选样本（包含 curriculum learning 策略从简单到复杂排列，并额外加入 400 条 MMLongBench 多跳难题）。训练 4 epoch，lr=1e-6。

## 数据与评测

论文在 **5 个 VQA benchmark** 上评测，涵盖单跳和多跳场景：

<div class="table-wrap">
  <table>
    <thead><tr><th>数据集</th><th>问题数</th><th>描述</th><th>充分上下文比例</th></tr></thead>
    <tbody>
      <tr><td>ChartQA</td><td>1,250</td><td>图表视觉与逻辑推理</td><td>59.36%</td></tr>
      <tr><td>InfoVQA</td><td>718</td><td>信息图问答</td><td>92.90%</td></tr>
      <tr><td>DocVQA</td><td>591</td><td>文档视觉问答（OOD）</td><td>83.59%</td></tr>
      <tr><td>SlideVQA</td><td>556</td><td>多幻灯片问答（OOD）</td><td>89.93%</td></tr>
      <tr><td>ViDoSeek</td><td>1,142</td><td>多文档检索与推理（OOD）</td><td>84.24%</td></tr>
    </tbody>
  </table>
</div>

其中 ChartQA 和 InfoVQA 为 In-Distribution 测试集（训练数据来自这两个数据集），DocVQA、SlideVQA、ViDoSeek 为 Out-of-Distribution 测试集。每个问题使用 VisRAG-Ret 检索 top-3 图像作为上下文。

评测指标区分 **sufficient context**（检索图像包含充分证据时用原始参考答案）和 **insufficient context**（检索不充分时模型应输出 "insufficient to answer"），报告全局 Accuracy 和 F1 Score。

## 实验结果

### 主结果：EVisRAG 在所有 benchmark 上一致领先

<div class="table-wrap">
  <table>
    <thead><tr><th>方法</th><th>ChartQA F1</th><th>InfoVQA F1</th><th>DocVQA F1</th><th>SlideVQA F1</th><th>ViDoSeek F1</th><th>平均 Acc</th><th>平均 F1</th></tr></thead>
    <tbody>
      <tr><td>Qwen2.5-VL-7B</td><td>52.80</td><td>54.61</td><td>56.03</td><td>46.11</td><td>42.48</td><td>55.50</td><td>50.41</td></tr>
      <tr><td>Qwen2.5-VL-32B</td><td>60.58</td><td>66.06</td><td>73.78</td><td>58.65</td><td>52.78</td><td>71.43</td><td>62.37</td></tr>
      <tr><td>OpenVLThinker</td><td>62.72</td><td>70.51</td><td>72.51</td><td>72.63</td><td>57.27</td><td>65.27</td><td>67.13</td></tr>
      <tr><td>MMSearch-R1</td><td>59.89</td><td>57.71</td><td>60.82</td><td>60.97</td><td>54.34</td><td>58.50</td><td>58.75</td></tr>
      <tr><td>VRAG-RL</td><td>10.03</td><td>12.21</td><td>22.39</td><td>15.37</td><td>18.14</td><td>60.55</td><td>15.63</td></tr>
      <tr><td>R1-Router</td><td>15.53</td><td>15.17</td><td>25.25</td><td>17.21</td><td>12.53</td><td>63.46</td><td>17.14</td></tr>
      <tr style="font-weight:bold"><td>EVisRAG</td><td>76.60</td><td>79.80</td><td>86.82</td><td>80.28</td><td>65.78</td><td>75.01</td><td>77.86</td></tr>
    </tbody>
  </table>
</div>

关键观察：

- EVisRAG 平均 Acc 75.01、F1 77.86，超过 backbone Qwen2.5-VL-7B 约 **+27% F1**，超过最强 VLRM baseline OpenVLThinker 约 **+10.73% F1**。
- 7B 参数的 EVisRAG **超过 32B 的 Qwen2.5-VL-32B**（75.01 vs 71.43 Acc），说明证据引导推理范式比单纯增大模型参数更有效。
- 其他 VRAG 方法（VRAG-RL、R1-Router）的 F1 分数极低（15-17%），尽管 Accuracy 尚可——说明它们缺乏对 "insufficient to answer" 场景的准确判断，EVisRAG 在这一点上远优于它们。

<figure class="figure">
  <img src="assets/papers/evisrag/page7_results.webp" alt="主实验结果与消融实验" loading="lazy" />
  <figcaption>图 3：Table 1（上）为总体性能对比，Table 2（下）为消融实验。EVisRAG 在所有 benchmark 上一致领先，消融实验逐步移除 Perception、Perception Reward、RS-GRPO 后性能依次下降。</figcaption>
</figure>

## Ablation：哪些设计最关键？

<div class="table-wrap">
  <table>
    <thead><tr><th>变体</th><th>ChartQA F1</th><th>InfoVQA F1</th><th>DocVQA F1</th><th>SlideVQA F1</th><th>ViDoSeek F1</th><th>平均 F1</th></tr></thead>
    <tbody>
      <tr style="font-weight:bold"><td>EVisRAG (full)</td><td>76.60</td><td>79.80</td><td>86.82</td><td>80.28</td><td>65.78</td><td>77.86</td></tr>
      <tr><td>w/o Perception（纯 think-then-answer）</td><td>68.33</td><td>77.41</td><td>79.62</td><td>79.09</td><td>58.84</td><td>72.65</td></tr>
      <tr><td>w/o Perception Reward（仅答案奖励）</td><td>74.66</td><td>76.44</td><td>85.60</td><td>78.84</td><td>65.69</td><td>76.25</td></tr>
      <tr><td>w/o RS-GRPO（标准 GRPO）</td><td>75.56</td><td>79.77</td><td>85.18</td><td>79.02</td><td>65.61</td><td>77.02</td></tr>
    </tbody>
  </table>
</div>

逐步消融的结论非常清晰：

1. **证据引导推理范式**（vs w/o Perception）：去掉 observe + evidence 阶段后平均 F1 下降 **5.21**，说明显式的逐页证据收集对多图推理至关重要。
2. **感知奖励**（vs w/o Perception Reward）：去掉感知奖励后平均 F1 再降 **1.61**，说明仅靠答案正确性信号不足以驱动精确的视觉感知。
3. **RS-GRPO**（vs w/o RS-GRPO）：把标准 GRPO（所有奖励求和作用于所有 token）替换为 RS-GRPO 后平均 F1 提升 **0.84**，说明 reward scope 绑定让信用分配更清晰、训练更稳定。

### 视觉注意力分析

论文手动标注了 100+ 案例的证据区域，计算 **visual evidence attention ratio**（注意力落在证据框内的比例）。

<figure class="figure">
  <img src="assets/papers/evisrag/page9_attention.webp" alt="视觉证据注意力分析与证据密度对比" loading="lazy" />
  <figcaption>图 4：(a) 准确率 vs 证据区域注意力比例——EVisRAG 达到最高注意力比例和最高准确率，两者呈正相关。(b) 可视化注意力热力图——EVisRAG 能精确聚焦到包含证据的 top bar 区域。</figcaption>
</figure>

关键发现：**模型对证据区域的注意力比例与回答准确率呈正相关**。EVisRAG 的注意力比例显著高于所有 baseline，说明 RS-GRPO 确实让模型学会了 "看对地方"。

### 证据密度鲁棒性

论文测试了 top-1 到 top-5 不同检索数量下的证据密度变化。随着检索图像增多，总证据 token 数增加但证据密度急剧下降。EVisRAG 在所有密度水平下一致优于 baseline，尤其在 DocVQA 上性能保持稳定，说明其 **抗幻觉能力更强**。

## 我的理解与启发

1. **"先记录再推理" 是一个简单但强大的范式**：EVisRAG 的四阶段流程本质上是在强制模型做显式的 evidence extraction，而不是让模型在长 chain-of-thought 中同时完成感知和推理。这对多图场景尤其重要——当上下文包含多张图像时，模型很容易在长序列中 "忘记" 或混淆不同页面的信息。

2. **Reward Scope 是 RL 训练中信用分配的精细工具**：RS-GRPO 的核心思想很朴素——不同的奖励信号应该作用于它们真正关心的 token。这个设计思路可以推广到很多其他场景：例如在 agent 训练中，工具调用奖励应只作用于工具调用 token，规划奖励应只作用于规划 token。

3. **感知奖励的 "no relevant information" 设计很巧妙**：允许模型显式标记无关图像为 "no relevant information"，并给予正确标记以奖励，这让模型学会了 **选择性注意**——不是每张图都值得详细描述。这比要求模型对所有图像都产出描述要高效得多。

4. **7B 超越 32B 的意义**：在 RAG 场景中，推理范式和训练策略可能比模型参数量更重要。EVisRAG 证明了好的证据引导机制能让小模型做出比大模型更准确的推理，这对部署成本敏感的场景非常有价值。

5. **与 ColPali 的互补性**：ColPali 解决的是 "如何检索到正确的文档页"，EVisRAG 解决的是 "检索到之后如何正确理解和推理"。两者可以组合成一个更完整的视觉 RAG 系统。

## 局限与待观察点

- **依赖检索质量**：EVisRAG 使用 VisRAG-Ret 检索 top-3 图像，检索质量直接影响最终效果。论文也评测了 insufficient context 场景，但改进检索本身不在本文范围内。
- **训练数据来自 ChartQA 和 InfoVQA**：虽然 OOD 泛化表现不错，但训练数据的多样性仍有限。更多文档类型（如手写文档、低质量扫描件）的效果有待验证。
- **推理延迟仍有优化空间**：四阶段输出比直接回答更长，虽然论文显示推理时间约 100s（vs baseline 的 90-95s），但在大规模生产环境中可能需要进一步优化。
- **感知奖励依赖大模型生成 gold evidence**：使用 Qwen2.5-VL-72B 生成 gold evidence 作为感知奖励的参照标准，这引入了对大模型质量的依赖，且增加了数据构建成本。
- **未涉及检索-推理联合优化**：EVisRAG 将检索和推理视为分离的阶段，未来可以探索模型在推理过程中动态触发检索的机制。

## 结论

EVisRAG 在 Visual RAG 方向提出了两个关键贡献：一是 **证据引导的四阶段推理范式**（observe → record → reason → answer），让 VLM 像侦探一样逐页收集证据后再推理；二是 **RS-GRPO 算法**，通过将感知奖励和推导奖励分别绑定到对应 token 作用域，解决了混合奖励导致的信用分配模糊问题。在 5 个 VQA benchmark 上，EVisRAG 以 7B 参数取得了平均 +27% F1 的提升，甚至超过 32B 模型，证明了推理范式设计和训练策略优化在视觉 RAG 场景中比单纯扩大模型参数更有效。
