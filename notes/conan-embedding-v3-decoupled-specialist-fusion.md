---
id: conan-embedding-v3-decoupled-specialist-fusion
title: "Conan-embedding-v3：用「解耦-融合-修复」把模态专家拼成全模态嵌入"
category: 多模态 RAG
date: 2026-07-14
order: 1
readTime: 15 min
tags: ["Omni-modal Embedding", "Model Merging", "Task Arithmetic", "Projector Drift", "Audio Retrieval", "MMEB", "MAEB", "Qwen3-VL"]
summary: "Conan-embedding-v3 提出 decouple–fuse–recover 框架：先独立训练图像/视频/文档/音频四个模态专家，再用任务向量在参数空间融合成单一骨干（Decoupled Specialist Fusion）。融合会让嫁接的音频投影层与新骨干失配（Projector Drift），论文用 Projector Recovery（冻结骨干只微调投影层）+ 均衡复习修复。最终单骨干模型在 MMEB 达 74.9、在 30 任务 MAEB 音频套件达 55.61，超越现有全模态嵌入 SOTA。"
---

## 一句话总结

Conan-embedding-v3 的核心思路是：**与其用一次大规模混合模态训练把所有模态硬塞进一个模型（会产生 seesaw 跷跷板效应），不如把能力获取和模型融合解耦**——先让图像、视频、文档、音频各自独立训练成"专家"，再用任务向量算术（Task Arithmetic）在参数空间把它们拼成一个骨干。论文进一步发现了一个关键失效模式 **Projector Drift**：音频是通过外挂编码器+投影层"嫁接"进来的，融合改变了骨干表示空间后，投影层仍校准在旧的音频专家骨干上，导致音频检索崩溃（哪怕音频模块原封不动地复制过来）。修复方案是 **Projector Recovery**（冻结骨干、只全参微调那 19M 的投影层）+ **均衡复习**，最终得到一个同时胜任视觉与音频检索的单一模型。

<div class="metric-grid">
  <div class="metric"><strong>74.96</strong><span>MMEB 总分（图像/视频/文档加权平均），逼近视觉专用 Qwen3-VL-Embedding-8B</span></div>
  <div class="metric"><strong>55.61</strong><span>30 任务 MAEB 音频套件，超越所有现有全模态嵌入模型</span></div>
  <div class="metric"><strong>32.68 → 55.82</strong><span>音频经 Projector Recovery 从崩溃恢复（MAEB）</span></div>
  <div class="metric"><strong>19M</strong><span>音频投影层参数量——修复阶段只需微调这一小部分</span></div>
</div>

## 论文信息

- **标题**：Conan-embedding-v3: Fusing Modality-Specific Models for Omni-Modal Embedding
- **作者**：Shiyu Li, Zhiyuan Hu, Yifan Wang, Peiming Li, Zheng Wei, Yang Tang
- **机构**：腾讯（Tencent）、清华大学
- **版本**：arXiv:2606.09331v1，2026-06-08
- **链接**：[arXiv](https://arxiv.org/abs/2606.09331) · [PDF](https://arxiv.org/pdf/2606.09331) · [HTML](https://arxiv.org/html/2606.09331v1)

## 背景：为什么全模态嵌入这么难训？

全模态检索（omni-modal retrieval）的理想是：文本、图像、视频、文档、音频都映射到同一个嵌入空间，用内积相似度就能跨模态检索。但真正把这样一个统一检索器训出来很难，因为这些模态在**数据分布、架构、优化动态**上都差异巨大。

最直接的做法是把所有模态数据混在一起微调一个模型，但这会产生**跨模态优化冲突（cross-modal optimization conflict）**，也就是"跷跷板效应（seesaw effect）"：提升一个模态往往会拖垮另一个。论文的消融实验很直观地证明了这一点——直接联合训练 I+V+D+A 四模态时，视频分数从解耦专家的 67.2 暴跌到 43.0（见下表消融）。

即使用 PCGrad 这类梯度手术（gradient surgery）方法在联合训练里投影掉冲突梯度，也只是略微缓解，视频和文档仍分别比独立专家低 9.5% 和 11.6%。这说明**联合训练的根本问题在于共享一个优化器和敏感的任务配比，靠梯度技巧治标不治本**。

## 方法：decouple–fuse–recover 三阶段框架

<figure class="figure">
  <img src="assets/papers/conan-embedding-v3/fig2_overview.webp" alt="Conan-embedding-v3 三阶段框架总览" loading="lazy" />
  <figcaption>图 2：Conan-embedding-v3 总览。Stage 1 从共享初始化 θ₀ 用模态特定 LoRA 训练各专家（音频专家额外嫁接音频编码器+投影层）；Stage 2 融合共享骨干的任务向量、直接复制音频模块，此时诱发 Projector Drift；Stage 3 先做 Projector Recovery（只微调投影层）再做均衡复习，得到最终全模态检索器。</figcaption>
</figure>

统一的训练目标是标准的 InfoNCE 对比损失（温度 τ=0.02），有双向配对时取 query→target 与 target→query 的平均。

### 1. 解耦专家训练（Decoupled Specialist Training）

从同一个基座 θ₀（**Qwen3-VL-8B**，原生支持文本+视觉）出发，对每个模态组 m ∈ {图像 I, 视频 V, 文档 D, 音频 A} 只在其**各自的数据分布**上独立训练一个专家 θ_m = Train(θ₀, 𝒟_m)。

- 图像、视频、文档三个专家复用基座原生的视觉-语言通路；
- 音频专家则额外**嫁接**一条外部音频通路（从 Qwen3-Omni-30B-A3B 抽取音频编码器 + 一个两层 MLP 投影层，把 2048 维音频特征映射到骨干的 4096 维隐空间，约 19M 参数），并新增 `<|audio_start|>`、`<|audio_end|>`、`<|audio_pad|>`、`<|AUDIO|>` 特殊 token，投影后的音频特征替换 `<|AUDIO|>` 的 token 嵌入。

训练时对语言骨干、视觉编码器、音频编码器、音频投影层的所有线性层都加 LoRA，训完把 LoRA 合并进 dense 权重，得到可做参数空间运算的完整 checkpoint。这样每个 checkpoint 都携带一个相对"干净"的单模态任务向量。

### 2. 任务向量融合（Task-Vector Fusion）

对共享参数 𝒦_shared，定义每个专家的任务向量 Δ_m[k] = θ_m[k] − θ₀[k]，然后用固定系数线性组合成统一骨干：

\[\theta_{\text{merge}}[k] = \theta_0[k] + \sum_{m \in \{I,V,D,A\}} \alpha_m \Delta_m[k]\]

主模型采用 **multiway-C 系数**：α_I = α_V = α_D = 0.3，α_A = 0.5（在与 MMEB/MAEB 评测集不相交的诊断验证集上网格搜索得到，用于平衡视觉保留与音频贡献）。

对于**基座里根本不存在的音频专属模块**（音频编码器、投影层、专用 tokenizer 条目），无法构造任务向量，直接从音频专家原样复制：θ_merge[k] = θ_A[k]。

这样融合出的 checkpoint 在**结构上是全模态的**——具备处理四种模态所需的全部通路。但结构兼容不等于表示对齐，这正是下一步要解决的问题。

### 3. Projector Drift：融合带来的接口失配

<figure class="figure">
  <img src="assets/papers/conan-embedding-v3/fig1_projector_drift.webp" alt="不同 Transformer 深度下的 Projector Drift" loading="lazy" />
  <figcaption>图 1：Projector Drift 随 Transformer 深度加剧。半圆展示第 0、17、35 层音频 token 的方向（相对音频专家蓝色基准）。直接融合（红）在深层逐渐偏离，而 Projector Recovery（绿）保持贴近专家，恢复了 AudioCaps 性能。</figcaption>
</figure>

问题的本质：**任务向量融合改变了骨干所期望的表示空间**。音频投影层 P_A 是在音频专家骨干 B_A 上训练的，它的作用是把音频编码器输出转成与 B_A 兼容的隐状态。但融合后，同一个投影层要面对一个不同的骨干 B_merge（融合了其它专家的任务向量），于是投影层仍校准在旧骨干上、推理却用了新骨干——这个投影层-骨干失配就是 **Projector Drift**。

论文用两组诊断分析揭示成因：

- **任务向量几何（图 4）**：音频任务向量与所有视觉更新几乎正交（cos ≤ 0.001），意味着视觉更新对音频几乎没有方向性支持；同时音频的全局更新范数最大（‖τ_A‖₂ ≈ 42.3），且这种大幅位移贯穿整个网络深度、在深层达到峰值（>14.0）。因为投影层在融合时是静止的，这种逐层漂移会破坏投影层-骨干接口。
- **输出空间邻域（图 5）**：在 AudioCaps 上做 t-SNE，音频专家能把配对的音频-文本嵌入交织在共享邻域里（R@1=92%），而直接融合把两个流形拉开（R@1 掉到 56%），说明这不是随机噪声而是**系统性的流形偏移**。

<figure class="figure">
  <img src="assets/papers/conan-embedding-v3/fig4_task_vector_geometry.webp" alt="四个模态任务向量的几何特性" loading="lazy" />
  <figcaption>图 4：四个模态任务向量的几何。(a) 成对余弦相似度显示音频与视觉更新近乎正交（cos ≤ 0.001）；(b) 音频的全局更新范数最大（≈42.3）；(c) 音频更新在整个骨干上都很大，并在深层达到峰值。</figcaption>
</figure>

### 4. Projector Recovery + 均衡复习

修复分两个连续阶段：

1. **Projector Recovery（仅投影层调优）**：冻结融合后的语言骨干和音频编码器，只对那 19M 参数的音频投影层做全参微调（在音频-文本检索对上）。因为骨干冻结，视觉/文档能力不会被音频梯度污染。这一步就把音频从崩溃的 32.68 恢复到 55.82（MAEB），同时**完全保留了融合前的视觉分数**。
2. **均衡复习（Balanced Rehearsal）**：从投影恢复后的 checkpoint 出发，用 LoRA 在骨干和视觉编码器上做一个轻量复习阶段，混合音频/图像/视频/文档四种检索数据，学习率更低（1e-5）、更新预算保守（约 2000 步）。这一步在保住音频的同时把视觉分数拉回 77.2 / 65.1 / 79.0。

论文的假设是：这个轻量复习阶段作用于已经专业化的权重上，更像是对已学表示的**温和全局协调**，而非容易冲突的能力获取过程，因此能缓解跨模态跷跷板。

## 实验结论

**基座与评测**：基座 Qwen3-VL-8B，约 50M 检索样本训练。视觉用 MMEB-V2（78 任务，覆盖图像 36 / 视频 18 / 文档 24），音频用 MAEB（30 任务，覆盖语音、音乐、环境声、音频-文本匹配）。

**主结果**：单一骨干同时达到 MMEB 74.96 与 MAEB 55.61。加入音频通路带来的视觉回退很小（相比视觉专用的 Qwen3-VL-Embedding-8B 的 77.82），但在 MAEB 上**超越所有现有全模态嵌入模型**（如 LCO-Embedding-Omni-7B 的 53.54、jina-v5-omni-small 的 50.41）。

**逐组件消融（表 3）**——每一步都不可或缺：

| 变体 | 说明 | Image | Video | VisDoc | MAEB |
|------|------|-------|-------|--------|------|
| Joint Training (I+V+D) | 无解耦/融合，仅视觉 | 80.3 | 67.2 | 82.0 | – |
| Joint Training (I+V+D+A) | 无解耦，全模态联合 | 69.2 | 43.0 | 74.9 | 48.01 |
| Direct Fusion | 融合但不修复 | 68.5 | 56.7 | 68.9 | **32.68** |
| + Audio-Only Recovery | 无均衡复习 | 68.5 | 56.7 | 68.9 | 55.82 |
| **Conan-embedding-v3** | 完整 decouple–fuse–recover | **77.2** | **65.1** | **79.0** | **55.61** |

<figure class="figure">
  <img src="assets/papers/conan-embedding-v3/fig5_tsne.webp" alt="Projector Drift 的输出空间视图" loading="lazy" />
  <figcaption>图 5：诊断音频-文本检索对的输出空间视图。音频专家（左）对齐配对嵌入（R@1=92%）；直接融合（中）分离了音频与文本流形（R@1=56%）；Projector Recovery（右）缩小了分离、恢复检索性能（R@1=91%）。</figcaption>
</figure>

**关于恢复策略（表 4）**：比起"骨干+投影层 LoRA"（音频略好 56.39 但改变了视觉权衡），"只调投影层"因冻结骨干而完整保住视觉分数；第二阶段的均衡复习（vs 一阶段联合恢复）取得最佳权衡。

**关于融合权重（图 3 / 附录 C）**：把音频系数 α_A 从 0.3 提到 0.9，音频稳步变好但视觉严重退化（Image MMEB 跌到 62.5%），α_A=0.5 是恢复前的最佳平衡点。这说明**光调系数无法同时平衡两个模态**，必须有融合后的修复阶段。

<figure class="figure">
  <img src="assets/papers/conan-embedding-v3/fig3_fusion_weights.webp" alt="恢复前的融合权重消融" loading="lazy" />
  <figcaption>图 3：恢复前的融合权重消融。增大音频任务向量系数 α_A 提升音频能力却损害视觉检索，展现了两者内在的冲突。</figcaption>
</figure>

**关于融合方法（附录 B）**：TIES-Merging、DARE 与 Task Arithmetic 在恢复前都同样受 Projector Drift 困扰（MAEB ≤ 32.68），证明 **Projector Drift 是任何参数空间融合的普遍后果**，与剪枝/丢弃/符号协调机制无关；因此恢复阶段是所有融合范式都必需的。Task Arithmetic 因视觉保留最好被选为默认。

**效率（附录 D）**：全流程在 A100(80G) 上约两周，但解耦让各专家可在不同节点**并行**训练，消除了联合训练的串行瓶颈；新增模态只需单独训一个专家 + 轻量恢复，模块化程度高。

## 局限

1. **Projector Recovery 是修复而非根治**：Projector Drift 被减小但没有完全消除。
2. **恢复配置靠人工经验搜索**：系数与流程是定向手工搜索得到的，缺乏自动化流程。
3. **多投影层交互未知**：随着更多模态通过额外投影层加入，多个 projector-drift 效应可能相互作用，本文没有研究超过四模态的情况。

## 个人理解与启发

- **这篇的价值在于把"模型合并（model merging）"这套原本主要在单模态、同构架构里研究的技术，落到了异构多模态嵌入这个更棘手的场景**，并诚实地暴露了它的失效模式（Projector Drift）。相比 jina-v5-omni 的"冻结塔+只训投影层"或 e5-omni 的"显式对齐 recipe"，Conan 走的是"先分头训练再参数空间拼装"的另一条路线，工程上的最大好处是**模态可并行、可增量扩展**。
- **Projector Drift 是一个很干净、可复现的洞察**：嫁接模态的投影层校准在旧骨干上，一旦骨干被融合改变就失配。这提示我们，凡是"外挂编码器+投影层"接进 LLM 的做法，只要动了骨干（合并、继续训练、量化等），都要警惕投影层是否需要重新校准。任务向量近乎正交 + 范数巨大，正好解释了为什么音频这条最"外来"的通路受伤最重。
- **"冻结骨干只修接口"是个可迁移的低成本套路**：只微调 19M 投影层就能把音频从 32.68 救回 55.82 且不伤视觉，这种"最小干预修复"的思路在很多"融合后能力回退"的场景里都值得先试。
- 一个值得追问的点：论文自己也承认这是"用额外训练/融合阶段换取对模态能力的可控性"。当模态继续增多、多个投影层的漂移开始相互耦合时，这套流水线的复杂度和调参成本会不会反噬掉它相对联合训练的效率优势？这也是作者留给未来的 end-to-end 全模态训练方向。
