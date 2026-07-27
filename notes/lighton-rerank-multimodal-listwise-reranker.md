---
id: "lighton-rerank-multimodal-listwise-reranker"
title: "LightOn-rerank：一个 LoRA 用 listwise 同时重排文本与文档图像"
category: "多模态 RAG"
date: "2026-07-23"
order: 1
readTime: "16 min"
tags: ["Reranker", "Multimodal RAG", "Listwise", "ViDoRe", "ColPali", "LoRA", "Visual Document Retrieval"]
summary: "LightOn-rerank 是一个 Qwen3.5-2B 的 LoRA，用生成式 listwise 同时重排文本段落和文档页面图像，在 ViDoRe V3 上达到 62.66 NDCG@10（比 ColQwen2.5 一阶段 +7 分），领先所有开源 2B 级多模态 reranker。这篇 field notes 最有价值的不是榜单，而是几个反直觉发现：真正带来重排质量的是「跨文档注意力」（四个候选在同一次前向里互相比较），而非 listwise loss 本身；文本 reranker 的提速技巧（tournament、first-token readout）在视觉上全部失效；2B 下最大开销是 ViT 编码器而非 decode；最省的提速杠杆是缩小候选池而非换打分方式。"
---

## 一句话总结

开源的多模态 reranker 非常稀缺——处理 PDF/网页/文本混合的检索管线，要么串两个单模态 reranker，要么用更大的模型硬扛服务成本。LightOn-rerank 想做一个「小而统一」的方案：**一个 Qwen3.5-2B 的 LoRA，用生成式 listwise 同时重排文本段落和文档页面图像**。它在 ViDoRe V3 上 62.66 NDCG@10（比 ColQwen2.5 一阶段 +7.1 分），领先所有开源 2B 级多模态 reranker，文本 BEIR 上也保持竞争力。但这篇 field notes 真正的价值在于那些「没料到」的工程发现——它们比榜单更重要。

<div class="metric-grid">
  <div class="metric"><strong>62.66</strong><span>ViDoRe V3 NDCG@10（2B listwise，比一阶段 +7.1）</span></div>
  <div class="metric"><strong>+10.8</strong><span>仅靠「跨文档注意力」带来的 NDCG 摆动</span></div>
  <div class="metric"><strong>46%</strong><span>每窗口 GPU 时间花在 ViT 图像编码器（最大头）</span></div>
  <div class="metric"><strong>85% / 5×</strong><span>只重排 top-20 保留 85% 增益、窗口数减少 5 倍</span></div>
</div>

## 博客信息

- **标题**：One Adapter, Both Modalities: Field Notes from Building and Serving a Multimodal Reranker
- **作者**：Amélie Chatelain、Ishrat Jahan Ananya（LightOn AI）
- **发布**：2026-07-16（HuggingFace Community Article）
- **模型**：LightOn-rerank，Qwen3.5-2B LoRA（前作是 MonoQwen2-VL-v0.1，业界首个开源视觉 reranker）
- **链接**：[博客原文](https://huggingface.co/blog/lightonai/lighton-rerank) · [模型合集](https://huggingface.co/lightonai)

> 说明：本文是技术博客（field notes）而非论文，核心内容以数据表格呈现，无方法/架构示意图，故下文忠实复现关键表格。

## 背景：两种重排方式，以及要挤进的赛道

reranker 拿一阶段检索器（BM25、稠密向量）的 top 候选，用更强的模型重新排序。全文的核心对立就是两种重排方式：

- **Pointwise（PW，逐条打分）**：模型一次看一个 (query, document) 对，输出相关性分数（score = logit("Yes") − logit("No")），每个候选**孤立打分**再排序。天然可并行、候选之间无交互。MonoQwen2-VL、jina-reranker-m0、Qwen3-VL-Reranker 都是这类。
- **Generative Listwise（LW，生成式列表）**：模型在一个 prompt 里同时看 query 和多个文档，**以文本生成一个排名**（如 `[2] > [4] > [1] > [3]`）。因为所有文档在同一次前向里，它们会**互相注意（attend）**——模型靠「互相比较」而非「对绝对相关性打分」来排序。100 个候选放不进一个 prompt，部署时用宽度 4、步长 2 的**滑动窗口**滑过并合并。

同赛道的三个参照点：**Qwen3-VL-Reranker-2B/8B**（数百万样本多阶段训练 + pointwise 打分头）、**jina-reranker-m0**（Qwen2-VL-2B + pointwise MLP 头）、**RankNexus**（listwise，但靠 GPT-4/Claude 级教师做 cross-encoder 蒸馏）。

## 方法：训练配方

- **基座**：Qwen3.5-2B，原生用共享 decoder + ViT 图像编码器同时处理文本和图像。
- **数据**：文本 107k 组（NQ + TriviaQA + MS MARCO，每例 4 元组 `[pos, neg_0, neg_1, neg_2]`）+ 视觉 106k 组（ColPali 训练集，Nomic 挖的负样本），共 **213k 组，微批内 50/50 混合**。
- **损失**：ListMLE，只在排列 token 上算交叉熵。硬负样本挖掘管线自带分数，因此**黄金排列（pos > neg_0 > neg_1 > neg_2）可直接从元数据构造**——这其实也是一种蒸馏，但蒸的是「挖掘器给出的排名」，而非 cross-encoder 的分数。
- **配置**：LoRA rank 32、rsLoRA、AdamW、lr 5e-5、1 epoch（419 步）、图像训练时 resize 到 512×512（每页约 256 image token，四页刚好塞进 2048 视觉预算）。

与三个参照点的差异：像 RankNexus 一样 listwise，但**不用 LLM 教师**；像小号 RankZephyr 一样生成式，但**多模态且文本-视觉联合**；且**没有 pointwise 打分头**——原因正是全文的核心发现。

## 核心发现：真正起作用的是「跨文档注意力」

作者用一句话点题：模型从不在真空里回答「这篇文档相关吗」，而是回答「这篇比其它几篇更好吗」。**带来重排质量的是候选之间的互相比较，而非 listwise loss 或 prompt 格式本身**。这个教训他们学了两次：

**贵的版本（训练侧）**：早期用 listwise loss（四文档一组、对黄金排名做 softmax 交叉熵），但**每个文档仍各自独立前向打分**。文本上看似进步，ViDoRe V3 上却倒退——51.86 overall，比 pointwise 基线（59.87）低约 8 分，甚至比它要重排的一阶段（55.60）还低 3.7 分，16 个 split 全输给 pointwise。同样的语料、同样的挖掘管线、更「信息量大」的训练信号，结果更差。

改成**把四个文档喂进同一个 prompt**（生成一个排列而非四个独立分数）后，符号翻转：**62.66，比 pointwise 高 2.8**——仅「共享上下文」这一项就带来 **10.8 NDCG 的摆动**。

**便宜的版本（服务侧）**：把训练好的 listwise 模型拿去 pointwise 服务（逐条打分、排序），想省掉滑动窗口。结果同样掉分：

<table>
  <thead>
    <tr><th>方式</th><th>NDCG@5</th><th>NDCG@10</th></tr>
  </thead>
  <tbody>
    <tr><td>Listwise 滑动窗口（生产）</td><td>80.77</td><td>82.26</td></tr>
    <tr><td>同一模型，pointwise 读出</td><td>76.41</td><td>78.24</td></tr>
    <tr><td>Δ（切成 pointwise 损失）</td><td>−4.36</td><td>−4.02</td></tr>
    <tr><td>一阶段（ColQwen2.5，不重排）</td><td>73.83</td><td>76.98</td></tr>
  </tbody>
</table>

两个教训是同一个：**listwise 模型的力量在于文档之间相对打分——不管在训练时（独立打分）还是服务时（pointwise 读出）拿掉这个比较，质量就随之流失。**

## 结果

### ViDoRe V3（ColQwen2.5 一阶段，rerank top-100，NDCG@10）

覆盖 8 个领域（金融、计算机、HR、能源、工业、制药、物理）、6 种语言，评测 EN/FR 子集：

<table>
  <thead>
    <tr><th>模型</th><th>规模</th><th>EN</th><th>FR</th><th>Overall</th></tr>
  </thead>
  <tbody>
    <tr><td>一阶段 ColQwen2.5（不重排）</td><td>—</td><td>57.58</td><td>53.63</td><td>55.60</td></tr>
    <tr><td>MonoQwen2-VL-v0.1</td><td>2B</td><td>59.30</td><td>56.22</td><td>57.76</td></tr>
    <tr><td>Qwen3-VL-Reranker-2B</td><td>2B</td><td>60.90</td><td>57.46</td><td>59.18</td></tr>
    <tr><td>jina-reranker-m0</td><td>2B</td><td>60.41</td><td>58.38</td><td>59.40</td></tr>
    <tr><td>本文 pointwise</td><td>2B</td><td>60.98</td><td>58.76</td><td>59.87</td></tr>
    <tr><td><strong>本文 listwise</strong></td><td>2B</td><td><strong>64.01</strong></td><td><strong>61.31</strong></td><td><strong>62.66</strong></td></tr>
  </tbody>
</table>

pointwise 版本已超过前作和两个开源竞品；listwise 再加 2.8 分，领先同规模所有开源模型、比一阶段高 7.1 分，且对自家 pointwise 基线赢 16 个 split 中的 13 个。值得强调：竞品用数百万样本或大模型蒸馏，而本文只在 213k 组上跑了 1 个 epoch。

**换一阶段验证非过拟合**：把一阶段换成 Qwen3-VL-Embedding-2B 后排序不变（本文 59.74 > Qwen3-VL-Reranker-2B 58.06 > jina-m0 58.01），说明不是与 ColQwen2.5 配对的假象。

**跨语言迁移（意外收获）**：训练数据全是英文（NQ/TriviaQA/MS MARCO/ColPali EN），adapter 从没见过法语样本，但法语子集只比英文低约 3 分（且金融_fr、物理、能源三域 FR 反超 EN）。Qwen 基座原生多语言，LoRA 把重排信号编码在了一个「已经会法语」的表示空间里。全程无 OCR，模型直接读页面像素。

### BEIR 文本（BM25 一阶段，rerank top-100）

视觉是主打，但同一 checkpoint 在纯文本上也得站得住。结论**反转**：文本上 jina-reranker-m0 领先（clean mean 51.97），本文 listwise（48.12）与 Qwen3-VL-Reranker-2B（49.08）大致打平。有趣的是**文本上本文的 pointwise 变体（49.13）反而略强于 listwise**——恰与视觉相反（视觉上 listwise 在每个规模都赢）。

## 缩放：0.8B / 2B / 4B × pointwise / listwise

<table>
  <thead>
    <tr><th>基座</th><th>Pointwise (V3@10)</th><th>Listwise (V3@10)</th><th>Δ (LW−PW)</th></tr>
  </thead>
  <tbody>
    <tr><td>Qwen3.5-0.8B</td><td>48.20</td><td>58.25</td><td>+10.0</td></tr>
    <tr><td>Qwen3.5-2B</td><td>59.87</td><td>62.66</td><td>+2.8</td></tr>
    <tr><td>Qwen3.5-4B</td><td>59.80</td><td>64.69</td><td>+4.9</td></tr>
  </tbody>
</table>

- **listwise 随规模持续涨，pointwise 早早饱和**：pointwise 从 2B→4B 几乎不动（−0.1），因为「一次只判一个文档相关与否」的决策很快到顶；listwise 把多出来的容量花在更丰富的跨文档比较上。
- **0.8B 下 pointwise 甚至低于一阶段**（48.20 < 55.60）——欠容量模型孤立打分会把排序搞得更糟；正是跨文档注意力让 0.8B listwise 仍可用（+10.0）。
- **4B listwise 达到 64.69**，在同一协议下超过官方 Qwen3-VL-Reranker-8B（64.23）——用一半参数、213k 组数据追平并反超 2 倍参数的模型。
- 附带细节：0.8B 下「2 epoch 优于 1 epoch」，与 2B 的「1 epoch 最好」相反——小模型欠容量，第二遍仍在提取信号而非记忆。

## 为什么文本 reranker 的提速技巧全失效了

**① Tournament scheduling（锦标赛淘汰）**：文本里的标准替代方案，分组取胜者晋级。视觉上惨败——smoke split 上 NDCG 从 72.62 掉到 45.47（−27.15），只换来 1.2× 提速。原因：淘汰制不宽容，一旦某文档早轮出局就再无机会；强文本 reranker 早轮很少犯错，但**从强一阶段（ColQwen2.5）来的困难视觉候选早轮噪声很大**——两张视觉相似但语义不同的页会来回换位，每次换错就永久踢掉一个相关页。滑动窗口让所有文档留在场上更久、能「冒泡」回来。

**② First-token readout（首 token 读出）**：文本里用首个生成位的 logits 直接诱导排序、跳过 decode。这里放弃了，因为**天花板只有约 1.6×**——瓶颈根本不在 decode。

## 多模态推理的时间到底花在哪（最意外的发现）

作者原以为 decode 是瓶颈（文本 listwise 确实如此），但四张页面图像一进来，天平就翻了。2B listwise 单窗口 GPU 时间剖析：

<table>
  <thead>
    <tr><th>阶段</th><th>中位 ms</th><th>占比</th></tr>
  </thead>
  <tbody>
    <tr><td>ViT 图像编码器（4 张图）</td><td>427.0</td><td><strong>46%</strong></td></tr>
    <tr><td>LLM prefill</td><td>147.0</td><td>16%</td></tr>
    <tr><td>LLM decode（17 步）</td><td>348.3</td><td>38%</td></tr>
    <tr><td>单窗口合计（GPU）</td><td>921.8</td><td>100%</td></tr>
  </tbody>
</table>

**最大单项是 ViT 编码器，占近一半**——编码四张输入图比解码整个 18 token 的排列还贵。这也解释了为什么 first-token readout 无意义：删掉 decode 最多省 1.6×，图像编码的账照付。

**分辨率的取舍**：模型只在 512²（256 token/页）训练，但推理时**提高分辨率能恢复质量**——16M 像素默认档比训练分辨率高 5.6 分，但 1M 像素以上收益迅速拉平（最后 6× 的 ViT 算力只买 0.8 分）。**1M 像素是甜点**。一个「预算驱动的幸运」：可以用低分辨率训练、靠推理时提分辨率把质量拿回来，省掉高分辨率训练的开销。

**跨网格的剖析**：ViT 成本在 2B/4B 视觉塔内是平的（427ms，都是 24 层），0.8B 减半（12 层）；per-token decode 从 0.8B 到 2B 平（~20ms），4B 才跳到 ~28ms。所以「ViT 吃掉一半墙钟」只在 (2B, listwise) 这个点严格成立——pointwise 在所有规模都是 ViT 主导，0.8B/4B listwise 则是 decode 主导。

## 最省的提速杠杆：少重排候选

比起调分辨率和模型大小，最大的旋钮其实是**到底重排多少个候选**：

<table>
  <thead>
    <tr><th>重排深度</th><th>窗口/query</th><th>NDCG@10</th><th>Δ vs K=100</th><th>保留的重排增益</th></tr>
  </thead>
  <tbody>
    <tr><td>K=100（全池）</td><td>49</td><td>62.66</td><td>—</td><td>100%</td></tr>
    <tr><td>K=20</td><td>9</td><td>61.62</td><td>−1.0 (−1.7%)</td><td>85%</td></tr>
    <tr><td>K=10</td><td>4</td><td>59.67</td><td>−3.0 (−4.8%)</td><td>58%</td></tr>
    <tr><td>一阶段（不重排）</td><td>0</td><td>55.60</td><td></td><td></td></tr>
  </tbody>
</table>

硬天花板是 first-stage recall@K：重排 K 个候选无法捞出一阶段排在 K 之后的文档。**top-20 砍 5.4× 窗口保留 85% 增益，top-10 砍 12× 保留 58%**，且任何深度都不低于一阶段。一阶段越强、截断越便宜。生产建议：**别为提速换打分方式，而是缩小候选池**，深度按自家一阶段的 recall 曲线定。

## 我的理解与启发

- **「比较」是 listwise reranker 的灵魂，不能在任何环节丢掉**。训练时独立打分、或服务时 pointwise 读出，都会把跨文档注意力这个真正的信号丢掉。这提醒我们：listwise 不是「换个 loss」，而是要保证候选在同一次前向里能互相看见。
- **视觉重排的瓶颈与文本完全不同**。从文本迁来的提速直觉（decode 是瓶颈、tournament 淘汰、first-token 读出）在多模态下几乎全错——真正的大头是 ViT 图像编码。做多模态系统一定要在**自己的操作点上真实剖析前向**，而非套用文本经验。
- **「缩池子不是缩模型」是极实用的工程结论**。强一阶段下 top-20 近乎免费地拿到 85% 增益，比换更小模型或降分辨率都划算，且模型始终 in-distribution。
- **低分辨率训练 + 高分辨率推理**是省成本的巧招：256 token/页训练，靠推理提分辨率恢复质量，避免高分辨率训练的显存/时间代价。
- **listwise 才是随规模持续受益的方向**：pointwise 早饱和，这对「买更大模型是否值得」是很清晰的指引——4B listwise 用一半参数超过 8B pointwise 竞品。

## 局限与待观察点

- **文本上不占优**：同一 checkpoint 在 BEIR 上被 jina-m0 明显甩开，且自家 pointwise 反而比 listwise 强，说明「listwise 全面更好」只在视觉成立。
- **监督来源未直接消融**：排名来自挖掘器的 rank 蒸馏，可能带入挖掘器偏差；作者承认还没做「打乱负样本 / 仅正负偏序」的干净消融（评论区已认领）。
- **1M 像素训练未验证**：512² 训练是预算所限，作者认为在 1M 像素训练是自然的下一步，但会把训练序列长度乘 3。
- **服务成本仍高**：2B listwise 每 query 约 45.7s（单调用口径，生产批处理可降 4–8×），4B listwise 更慢（77s）；延迟敏感场景需在 pointwise（快）与 listwise（准）间权衡。

## 结论

LightOn-rerank 用一个 2B LoRA、生成式 listwise，统一重排文本与文档图像，在 ViDoRe V3 上领先同规模开源模型。但它最有价值的是一组反直觉的 field notes：**重排质量来自「跨文档注意力」而非 listwise loss；文本提速技巧在视觉上失效；ViT 编码器才是 2B 下的最大开销；提速应缩候选池而非换打分方式**。作者给同类项目的四条建议也很干脆——先钉死评测协议、2B 级多模态 reranker 选生成式 listwise、别把 listwise 模型 pointwise 服务、在自己的操作点上剖析前向。
