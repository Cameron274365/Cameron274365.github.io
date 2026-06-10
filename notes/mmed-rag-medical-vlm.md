---
id: "mmed-rag-medical-vlm"
title: "MMed-RAG：面向医疗视觉语言模型的多模态 RAG"
category: "论文阅读"
date: "2026-06-10"
order: 20
readTime: "13 min"
tags: ["Medical VLM","Multimodal RAG","RAG-PT","Retrieval","Alignment","ICLR 2025"]
summary: "MMed-RAG 面向医疗视觉语言模型的事实性幻觉问题，提出领域感知检索、自适应检索上下文选择和 RAG-based Preference Fine-Tuning，在放射、眼科、病理五个数据集上显著提升医疗 VQA 与报告生成的事实准确性。"
hero: "assets/papers/mmed-rag/overview.png"
---

## 一句话总结
MMed-RAG 的核心贡献是把医疗多模态 RAG 从“检索几条相似报告塞给模型”升级成一套 **可泛化、可控噪、可对齐** 的系统：先识别医学影像所属领域并选择对应检索器，再根据相似度断崖自适应截断检索上下文，最后用偏好微调让 Med-LVLM 学会什么时候相信图像、什么时候利用检索、什么时候忽略错误证据。

<div class="metric-grid">
  <div class="metric"><strong>43.8%</strong><span>五个医疗数据集上的平均事实准确性提升</span></div>
  <div class="metric"><strong>18.5%</strong><span>医疗 VQA 相对原始 Med-LVLM 的提升</span></div>
  <div class="metric"><strong>69.1%</strong><span>报告生成任务相对原始 Med-LVLM 的提升</span></div>
  <div class="metric"><strong>5 datasets</strong><span>覆盖放射、眼科、病理三类医学影像</span></div>
</div>

## 论文信息
- **标题**：MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models
- **作者**：Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Weijia Shi, Sheng Wang, Linjun Zhang, James Zou, Huaxiu Yao
- **机构**：UNC-Chapel Hill, Brown University, Carnegie Mellon University, Rutgers University, University of Washington, Stanford University
- **会议**：ICLR 2025
- **链接**：[arXiv:2410.13085](https://arxiv.org/abs/2410.13085) · [PDF](https://arxiv.org/pdf/2410.13085) · [代码/数据](https://github.com/richard-peng-xia/MMed-RAG)

<figure class="figure">
  <img src="assets/papers/mmed-rag/overview.png" alt="MMed-RAG 方法总览" loading="lazy" />
  <figcaption>图 1：MMed-RAG 总览。系统由领域感知检索、自适应上下文选择和 RAG-based Preference Fine-Tuning 三部分组成。</figcaption>
</figure>

## 背景：为什么医疗多模态 RAG 不能直接套通用做法？
医疗 VLM 的问题不是“回答不够丰富”，而是 **错误回答的代价极高**。当模型在 X-ray、眼底图、病理切片等场景中幻觉出不存在的病灶，可能直接影响诊断和治疗决策。

论文指出，Fine-tuning 和 RAG 都能缓解幻觉，但在医疗场景各有明显限制：

- **Fine-tuning 数据受限**：高质量医学标注数据稀缺，且训练分布和真实部署分布常有 gap。
- **固定 RAG 不够泛化**：不同医学影像的视觉形态差异巨大，单一检索器很难同时适配放射、眼科和病理。
- **检索上下文会引入新噪声**：检索太少，证据不足；检索太多，低相关报告会污染生成。
- **RAG 会破坏模型对齐**：模型可能绕过图像、直接复制检索结果；也可能被错误检索干扰，把原本正确的答案改错。

因此，MMed-RAG 关注的不是简单“加 RAG”，而是解决医疗多模态 RAG 中的三个关键问题：**检索域是否匹配、证据数量是否合适、模型是否正确使用证据**。

## 方法：MMed-RAG 的三个核心设计
### 1. Domain-Aware Retrieval：按医学影像领域选择检索器
MMed-RAG 首先用一个领域识别模块判断输入图像属于哪个医学域，例如 radiology、ophthalmology 或 pathology。论文用少量带领域标签的医学图像微调 BiomedCLIP，使其具备领域感知能力。

得到领域标签后，系统不再把所有医学数据混在一个库里检索，而是路由到对应领域的多模态检索器。每个领域检索器用图文对进行对比学习：图像编码器和文本编码器分别生成 embedding，通过 CLIP 风格的 contrastive loss 拉近同一样本的图文表示、推远不同样本。

这个设计的直觉很明确：**胸片和病理切片的相似性标准不是一回事**。把所有模态混在一起训练通用检索器，看似更统一，但容易让跨域视觉差异淹没真正的医学语义。

### 2. Adaptive Retrieved Context Selection：根据相似度断崖决定取几条证据
很多 RAG 系统使用固定 top-k，例如每次都取 5 条或 10 条。但在医疗图像中，不同样本的检索质量分布差异很大：有些图像前几条都很相关，有些图像只有第一条可信，后面迅速变成噪声。

<figure class="figure">
  <img src="assets/papers/mmed-rag/context-selection.png" alt="检索上下文数量与相似度关系" loading="lazy" />
  <figcaption>图 2：检索结果的相似度常在某个位置出现明显下降。MMed-RAG 利用相邻相似度比值检测这个断点，从而动态截断低质量上下文。</figcaption>
</figure>

MMed-RAG 的做法是观察连续检索结果的相似度比值：如果第 i 条和第 i+1 条之间出现明显下降，就认为后续上下文相关性不足，应截断。论文用 \(u_i = \log(S_i / S_{i+1})\) 表示相邻相似度差异，当 \(u_i\) 超过阈值 \(\gamma\) 时停止保留后续上下文。

这相当于给医疗 RAG 加了一个简单但有效的“证据质量闸门”：**不是检索越多越好，而是只保留相似度曲线断崖之前的证据**。

### 3. RAG-based Preference Fine-Tuning：让模型学会正确使用检索证据
论文最关键的部分是 RAG-PT。作者发现，引入检索后，Med-LVLM 会出现两类 misalignment：

- **Cross-modality misalignment**：模型忽视输入图像，直接根据检索报告作答。论文在 Harvard-FairVLMed 上用噪声图像替换原图后，发现加入 RAG 后仍有 **55.08%** 样本回答正确，说明模型可能只是在复制检索证据。
- **Overall alignment misalignment**：模型原本能答对，但加入错误检索后反而被带偏。论文报告这类 over-reliance 比例达到 **43.31%**。

<figure class="figure">
  <img src="assets/papers/mmed-rag/noisy-image.png" alt="交叉模态对齐中使用的噪声图像示例" loading="lazy" />
  <figcaption>图 3：用于检测 cross-modality misalignment 的噪声图像。若模型在图像被强噪声替换后仍回答正确，说明它可能过度依赖检索上下文。</figcaption>
</figure>

RAG-PT 构造三类偏好样本来训练模型：

1. **图像优先样本**：当检索信息和噪声图像同时存在时，模型仍答对，说明它可能只看检索。偏好训练会鼓励模型不要忽略真实图像。
2. **检索有益样本**：原图单独答错，但加入正确检索后答对，训练模型在不确定时合理利用检索证据。
3. **检索干扰样本**：原图单独答对，但加入错误检索后答错，训练模型学会抵抗错误检索干扰。

这三类样本最后合并，用 DPO 风格目标进行偏好优化。我的理解是，RAG-PT 实际上给 Med-LVLM 学了一个“证据仲裁策略”：图像、问题、检索报告三者冲突时，模型应该如何分配信任。

## 数据与评测
论文覆盖五个医疗视觉语言数据集，涉及三类医学影像：

- **Radiology**：MIMIC-CXR、IU-Xray，主要是胸部 X-ray 与放射报告。
- **Ophthalmology**：Harvard-FairVLMed，眼底图及相关文本。
- **Pathology**：PMC-OA pathology subset、Quilt-1M，病理图像与图文对。

VQA 数据由医学报告经 GPT-4 生成 yes/no 问答，并进行自检与人工过滤。报告生成任务使用 BLEU、ROUGE-L、METEOR 评估；VQA 使用 Accuracy、F1、AUROC。

<div class="table-wrap">
  <table>
    <thead><tr><th>任务</th><th>数据域</th><th>训练检索器数据</th><th>RAG-PT 总样本</th></tr></thead>
    <tbody>
      <tr><td>VQA</td><td>Ophthalmology</td><td>7,000</td><td>3,247</td></tr>
      <tr><td>VQA</td><td>Radiology</td><td>4,034</td><td>4,836</td></tr>
      <tr><td>VQA</td><td>Pathology</td><td>5,000</td><td>1,990</td></tr>
      <tr><td>Report Generation</td><td>Ophthalmology</td><td>7,000</td><td>3,247</td></tr>
      <tr><td>Report Generation</td><td>Radiology</td><td>4,034</td><td>4,836</td></tr>
    </tbody>
  </table>
</div>

## 实验结果
### 医疗 VQA：多数数据集上超过解码方法和 RAG baseline
在 LLaVA-Med-1.5 7B 上，MMed-RAG 在五个 VQA 数据集上整体最优。几个代表性结果如下：

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>IU-Xray Acc</th><th>MIMIC-CXR Acc</th><th>Harvard-FairVLMed Acc</th><th>Quilt-1M Acc</th><th>PMC-OA Acc</th></tr></thead>
    <tbody>
      <tr><td>LLaVA-Med-1.5</td><td>75.47</td><td>75.79</td><td>63.03</td><td>62.80</td><td>59.28</td></tr>
      <tr><td>+ FactMM-RAG</td><td>84.51</td><td>77.58</td><td>83.67</td><td>69.25</td><td>60.49</td></tr>
      <tr><td>+ RULE</td><td>87.84</td><td><strong>83.92</strong></td><td>87.12</td><td>68.97</td><td>61.41</td></tr>
      <tr><td>MMed-RAG</td><td><strong>89.54</strong></td><td>83.57</td><td><strong>87.94</strong></td><td><strong>72.95</strong></td><td><strong>64.54</strong></td></tr>
    </tbody>
  </table>
</div>

论文总结，MMed-RAG 在医疗 VQA 上相对原始 Med-LVLM 提升 **18.5%**，相对解码方法提升 **11.5%**，相对已有 RAG 方法仍提升 **2.8%**。

### 报告生成：开放式生成更依赖检索对齐
报告生成任务上提升更明显，因为开放式医学报告更依赖外部证据，也更容易被错误证据带偏。

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>IU-Xray BLEU</th><th>IU-Xray METEOR</th><th>MIMIC-CXR BLEU</th><th>Harvard-FairVLMed BLEU</th><th>Harvard-FairVLMed METEOR</th></tr></thead>
    <tbody>
      <tr><td>LLaVA-Med-1.5</td><td>9.64</td><td>8.21</td><td>12.11</td><td>18.11</td><td>10.75</td></tr>
      <tr><td>+ FactMM-RAG</td><td>14.70</td><td>15.92</td><td>18.71</td><td>20.82</td><td>15.31</td></tr>
      <tr><td>+ RULE</td><td>27.53</td><td>27.99</td><td>18.61</td><td>22.35</td><td>17.74</td></tr>
      <tr><td>MMed-RAG</td><td><strong>31.38</strong></td><td><strong>32.43</strong></td><td><strong>23.25</strong></td><td><strong>24.82</strong></td><td><strong>19.85</strong></td></tr>
    </tbody>
  </table>
</div>

相对原始 Med-LVLM，MMed-RAG 在报告生成上提升 **69.1%**；相对已有 RAG baseline 仍提升 **16.1%**。这说明 RAG-PT 对长文本、开放式医学生成尤其重要。

### 与其他 Med-LVLM 对比：跨领域泛化更强
论文还比较了 Med-Flamingo、MedVInT、RadFM、miniGPT-Med 等模型。按领域平均结果看，MMed-RAG 在 radiology、ophthalmology、pathology 上分别达到 **56.94 / 56.38 / 54.10**，明显高于其他 7B 级医学 VLM。

这组结果强调了论文的“versatile”主张：不是只在单个医学域做一个专用模型，而是用领域路由 + 对齐训练把 RAG 扩展到多类医学影像。

## Ablation：哪些设计最关键？
论文在 IU-Xray 和 Harvard-FairVLMed 上逐步加入模块，结果很清楚：DR、RCS、RAG-PT 都有贡献，其中 RAG-PT 带来的增益最大。

<div class="table-wrap">
  <table>
    <thead><tr><th>模型</th><th>IU-Xray VQA</th><th>IU-Xray RG</th><th>FairVLMed VQA</th><th>FairVLMed RG</th></tr></thead>
    <tbody>
      <tr><td>LLaVA-Med-1.5</td><td>68.99</td><td>10.04</td><td>66.63</td><td>13.41</td></tr>
      <tr><td>+ DR</td><td>77.12</td><td>13.23</td><td>72.69</td><td>15.89</td></tr>
      <tr><td>+ RCS</td><td>79.56</td><td>17.92</td><td>75.74</td><td>17.22</td></tr>
      <tr><td>+ RAG-PT</td><td><strong>85.80</strong></td><td><strong>29.80</strong></td><td><strong>87.18</strong></td><td><strong>20.42</strong></td></tr>
    </tbody>
  </table>
</div>

偏好数据的组合消融也很有启发：单独使用任一类 RAG-PT 数据都有效，但三类组合最好。完整的 RAG-PT 1+2+3 在 IU-Xray VQA / RG 上达到 **85.58 / 29.69**，在 Harvard-FairVLMed VQA / RG 上达到 **87.02 / 20.31**。

### 检索器质量：领域化检索器显著更强
论文报告不同 retriever 的 Recall@1 / Recall@5：CLIP 只有 **3.91 / 7.88**，BiomedCLIP 为 **15.7 / 23.8**，而 MMed-RAG 的领域检索器达到 **45.6 / 71.8**。这说明在医疗多模态 RAG 中，检索器本身不是可替换的小组件，而是事实性提升的基础。

## 对齐分析：RAG-PT 是否真的减少误用检索？
论文用 Copy-Reference Rate 和 Over-Reliance Rate 衡量 RAG 引入的错误来源。经过 MMed-RAG 后：

- **Copy-Reference Rate** 降到 **28.19%**；
- **Over-Reliance Rate** 从 **43.31%** 降到 **8.38%**。

<figure class="figure">
  <img src="assets/papers/mmed-rag/alignment-rates.png" alt="MMed-RAG 对 Copy-Reference 和 Over-Reliance 的改善" loading="lazy" />
  <figcaption>图 4：RAG-PT 显著降低直接复制检索内容和被错误检索干扰的比例。</figcaption>
</figure>

注意力可视化也支持这一点：原始 Med-LVLM 加入 RAG 后更关注文本检索结果，而 RAG-PT 后模型对图像 token 的关注增强，对干扰性检索上下文的关注下降。

<figure class="figure">
  <img src="assets/papers/mmed-rag/attention-map.png" alt="加入 RAG-PT 前后的注意力对比" loading="lazy" />
  <figcaption>图 5：注意力图显示，RAG-PT 后模型更重视视觉信息，并降低对错误检索内容的过度依赖。</figcaption>
</figure>

## 我的理解与启发
### 1. 医疗 RAG 的核心是“证据治理”而不是“上下文拼接”
MMed-RAG 最值得借鉴的是它把 RAG 拆成可治理的三层：检索域治理、检索数量治理、证据使用治理。很多 RAG 系统只优化 recall，但医疗场景更需要控制错误证据的传播路径。

### 2. RAG 会引入新的 alignment 问题
这篇论文提醒我们：RAG 不是天然更安全。检索内容看起来像证据，但模型可能把它当成比图像更可信的捷径。尤其在医疗、多模态、长报告生成中，错误检索可能比没有检索更危险。

### 3. 偏好数据可以围绕“失败模式”构造
RAG-PT 的偏好样本并不是泛泛收集好/坏回答，而是围绕三类具体失败模式构造：忽略图像、不会利用检索、过度相信检索。这种按失败模式设计偏好数据的方法，比单纯做 SFT 更有工程启发。

### 4. 多领域系统应优先考虑 routing，而不是强行统一
医学图像的域差异极大，论文最终选择领域识别 + 专家检索器，而不是一个大而全的通用检索器。对于企业知识库、工业视觉、跨行业多模态系统也类似：先做可靠 routing，再做域内高质量检索，往往比过早追求统一 embedding 更稳。

## 局限与待观察点
- **仍依赖领域识别准确性**：论文中领域识别准确率很高，但面对更细粒度或未知医学模态时，routing 失败会影响整体效果。
- **新增医学域需要训练检索器**：系统可扩展，但不是零成本扩展；每个新领域都需要足够的图文对训练检索器。
- **VQA 问答形式偏 yes/no**：这有利于事实性分析，但可能低估真实临床问答的复杂性。
- **报告生成指标仍是文本相似度**：BLEU、ROUGE-L、METEOR 无法完全衡量临床事实正确性，还需要更强的医学事实核查指标。
- **RAG-PT 需要构造偏好数据**：虽然比全量重训轻量，但仍需要额外训练流程和失败样本挖掘。
- **真实临床部署仍需谨慎**：论文证明了事实性提升，但医疗系统还需要可追溯引用、置信度校准、医生审核和隐私合规。

## 结论
MMed-RAG 是一篇很适合理解“医疗多模态 RAG 如何真正落地”的论文。它没有停留在检索增强的表层，而是系统处理了多领域检索、动态证据选择和 RAG 后对齐三个核心问题。对我来说，它最重要的启发是：在高风险场景里，RAG 的价值不只是提供更多知识，而是建立一套让模型 **正确选择、正确使用、必要时拒绝错误证据** 的机制。