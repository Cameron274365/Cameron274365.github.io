---
id: "mm-rag-all-modalities"
title: "MM-RAG 全模态综述：从输入输出组合看万物皆可 RAG"
category: "多模态 RAG"
date: "2026-06-10"
order: 4
readTime: "20 min"
tags: ["RAG", "Multimodal", "MM-RAG", "Survey", "Retrieval", "Generation", "Evaluation", "Modality Combination"]
summary: "这篇 TechRxiv 综述从输入模态和输出模态的组合出发，系统梳理几乎所有可用于多模态 RAG 的研究空间。它的核心价值不是只总结图文 RAG，而是把文本、图像、音频、视频、代码、表格、知识图谱、3D 等模态放到同一张 taxonomy 中，指出 54 种潜在组合里只有 18 种已有较充分研究，并进一步拆解 MM-RAG 的四阶段工作流、训练策略、评估基准和未来方向。"
---

## 一句话总结
这篇综述的最大价值是把 **MM-RAG 从“图文检索增强”扩展为“任意输入模态到任意输出模态的检索增强系统设计问题”**。作者不是只按检索器、生成器或应用领域分类，而是先枚举输入/输出模态组合，再回到系统工作流、训练和评估方法，帮助我们看到：当前研究主要集中在少数图文任务上，而音频、视频、代码、表格、知识图谱、3D 等组合仍有大量空白。

<div class="metric-grid">
  <div class="metric"><strong>54 种</strong><span>潜在输入-输出模态组合</span></div>
  <div class="metric"><strong>18 种</strong><span>已有研究覆盖的组合</span></div>
  <div class="metric"><strong>4 阶段</strong><span>Pre-retrieval / Retrieval / Augmentation / Generation</span></div>
  <div class="metric"><strong>8 类模态</strong><span>文本、图像、音频、视频、代码、表格、KG、3D 等</span></div>
</div>

## 论文信息
- **标题**：A Comprehensive Survey on Multimodal RAG: All Combinations of Modalities as Input and Output
- **作者**：Rui Zhang, Chen Liu, Yixin Su, Ruixuan Li, Xuanjing Huang, Xuelong Li, Philip S. Yu
- **机构**：Huazhong University of Science and Technology, Fudan University, Institute of Artificial Intelligence (TeleAI) of China Telecom, University of Illinois at Chicago
- **版本**：TechRxiv v2, 2025-11
- **链接**：[TechRxiv DOI](https://doi.org/10.36227/techrxiv.176341513.38473003/v2) · [项目主页 / Awesome-MM-RAG](https://github.com/INTREBID/Awesome-MM-RAG)

<figure class="figure">
  <img src="assets/papers/mm-rag-all-modalities/definition.webp" alt="MM-RAG 数学定义示意" loading="lazy" />
  <figcaption>图 1：论文对 MM-RAG 的形式化定义。系统由知识库、检索器和生成器组成：检索器从多模态知识库中取回 top-k 相关文档，生成器再基于 query 与检索证据生成目标输出。</figcaption>
</figure>

## 问题定义：MM-RAG 到底是什么？
论文把 MM-RAG 形式化为三个核心组件：

- **知识库 \(D\)**：由文本、图像、代码、音频、视频、3D 数据、结构化数据等多模态文档组成。这里的 “document” 是广义概念，可以是一段文本、一张图、一个视频片段、一段代码或其他可检索对象。
- **检索器 \(R\)**：给定用户 query \(q\)，从知识库中检索 top-k 相关文档 \(Z=R(q,D)\)。相关性可以由相似度函数、稀疏检索、密集向量检索或任务特定检索器计算。
- **生成器 \(G\)**：基于 query 和检索证据生成输出 \(y=G(q,Z)\)。输出不一定是文本，也可以是图像、音频、视频、代码或 3D。

论文进一步将目标写成：在给定度量 \(M\) 下，让生成结果 \(y\) 与 ground truth \(y^*\) 的差异最小。由于多模态任务的 ground truth 往往不唯一，实际评估通常拆成两类 proxy：**检索质量指标**（Precision@k、Recall@k、nDCG 等）和 **生成质量指标**（文本 BLEU/ROUGE，图像 FID/IS 等）。

## 背景：为什么需要“全组合”视角？
过去谈多模态 RAG，很多时候默认是 **Text + Image → Text**：用户用文本问问题，系统从图文文档或图片中检索证据，最后生成文本答案。这当然重要，但它只是 MM-RAG 的一个小切片。

现实中的信息形态远比图文复杂：

- **视频** 不只是很多帧图片，还包含动作、时序、事件边界和 ASR/OCR 辅助信息。
- **音频** 不只是转写文本，还包含语气、说话人、环境声和音乐结构。
- **表格** 与 **知识图谱** 更强调结构化关系、约束和可验证推理。
- **代码** 需要语法、依赖、仓库上下文和可执行性。
- **3D 对象** 涉及空间结构、视角、材质和几何一致性。

如果仍然用“图像检索 + 文本生成”的范式去概括全部 MM-RAG，就会遮蔽大量新问题。作者因此提出从 **输入模态 × 输出模态** 出发重画地图：先问“哪些模态能作为 query？哪些模态能作为 retrieved knowledge？最终输出可以是什么？”，再看每个组合中已经有哪些方法、任务和空白。

<figure class="figure">
  <img src="assets/papers/mm-rag-all-modalities/taxonomy.webp" alt="MM-RAG 输入输出模态组合分类" loading="lazy" />
  <figcaption>图 2：论文按输入和输出模态组合组织 MM-RAG 研究。这个 taxonomy 的重点是暴露研究版图：54 种潜在组合中只有 18 种已有研究覆盖，许多跨模态生成与结构化输出仍待探索。</figcaption>
</figure>

## 核心贡献：这篇综述和常见 MM-RAG 综述有什么不同？
### 1. 不是只看图文，而是枚举输入-输出组合
已有综述往往围绕视觉问答、文档理解、医疗图像或视频 QA 等单一方向展开。这篇文章的切入点更底层：**将 MM-RAG 看成一种可组合系统**。

例如：

- **Image → Text**：检索增强图像描述、视觉问答、医学报告生成。
- **Text → Image**：检索增强文生图，用外部视觉样例约束实体、风格或构图。
- **Video + Text → Text**：长视频 QA、视频语料检索、事件级问答。
- **Text → Audio / Audio → Text**：音频生成、音频 caption、语音问答和音频知识检索。
- **Code → Text / Text → Code**：代码摘要、仓库级代码生成、检索增强代码补全。
- **Text / Table / KG → Text**：表格问答、知识库问答、图检索增强推理。
- **Text → 3D**：检索增强 3D 生成、动作或几何结构生成。

这种分类的好处是：它把“已经很热的任务”和“几乎没人做但可能很有用的任务”放在同一张表里，研究空白会非常明显。

### 2. 强调 MM-RAG 是系统工程，而不是单个检索器
论文认为一个可用的 MM-RAG 系统至少包含四个阶段：预检索、检索、增强、生成。每个阶段都和模态强相关，不能简单复用文本 RAG 的经验。

<figure class="figure">
  <img src="assets/papers/mm-rag-all-modalities/workflow.webp" alt="MM-RAG 四阶段工作流" loading="lazy" />
  <figcaption>图 3：MM-RAG 的典型工作流。预检索负责知识库和查询准备，检索负责找出相关多模态证据，增强负责重排序、选择、压缩、去噪和融合证据，生成负责输出目标模态结果。</figcaption>
</figure>

### 3. 把训练、评估和应用也纳入同一框架
很多 RAG 文章只讨论 pipeline，但真正落地时，训练和评估同样关键：检索器是否需要微调？生成器是否冻结？如何评估视觉证据是否被正确使用？生成的图片、音频、视频又该如何度量？这篇综述专门把训练策略和评估 benchmark 单独整理出来，避免 MM-RAG 只停留在概念层。

## 方法框架：MM-RAG 的四阶段工作流
### 阶段一：Pre-retrieval，先把知识变成可检索对象
预检索阶段处理的是“检索之前”的准备工作。文本 RAG 中它通常对应清洗、切块、向量化；但在 MM-RAG 中，预检索复杂很多。

论文把 pre-retrieval 拆成两部分：**knowledge base preparation** 和 **query preparation**。

知识库准备又包含：

- **数据预处理**：清洗、切块、附加 metadata。视频可转成文本描述、帧、OCR/ASR 片段；3D 数据可转成图像作为检索 key；代码可按语法结构自然切块。
- **知识库组织**：论文总结了四类组织方式：uni-modal embedding、pair-wise storage、unified embedding、graph construction。
- **Uni-modal embedding**：各模态分别编码、分别建库，适合模态关系较松的场景，但可能破坏跨模态语义关系。
- **Pair-wise storage**：利用图文、音文等配对数据构建 key-value 检索，适合 caption、audio generation 等任务，但依赖大量预构造配对数据。
- **Unified embedding**：用 VLM/MLLM 将多模态数据映射到统一语义空间，例如 VisRAG、M3DocRAG 用页面截图直接编码文档，避免 OCR/解析损失。
- **Graph construction**：把对象、实体、关系、时间线、函数调用等组织成图，适合多跳推理和复杂问答；图像可用 scene graph，视频可用时空关系，代码可用 property graph。

查询准备包括五种技术：query expansion、query rewriting、query transformation、query dropout 和 multi-query。它们分别解决 query 信息不足、表达不匹配、模态不对齐、检索视角单一和复杂问题拆解等问题。

我理解这一阶段的核心取舍是：**越早把多模态信息转成文本，系统越容易搭建；但越可能丢失多模态 RAG 相比文本 RAG 的真正价值**。

### 阶段二：Retrieval，从多模态知识库中找到证据
检索阶段的目标是根据用户 query 找到相关证据。不同模态的检索难点不同：

- **文本检索**：BM25、dense retriever、hybrid retrieval 仍然是基础，但需要处理由视觉/音频/视频转换来的 noisy text。
- **图像检索**：通常依赖 CLIP/BLIP/MLLM embedding，难点是细粒度区域、属性、关系和 OCR 信息。
- **视频检索**：除了语义相似，还要考虑时间定位、事件边界和跨片段连续性。
- **音频检索**：可以走 ASR 文本，也可以直接使用音频表征；后者更能保留非语音信息。
- **表格/KG 检索**：需要结构化约束、实体链接和关系路径，不只是向量相似度。
- **代码检索**：仓库级依赖、API 调用、类型信息和执行上下文都可能影响相关性。

论文从三个维度整理 retrieval：

- **Retriever 类型**：sparse retriever 适合文本/代码这类有“词”的模态；dense retriever 通过 DPR、Contriever、CLIP、CLAP 等模型支持跨模态检索；代码、图像生成等任务还会使用 AST/edit distance/Jaccard/beam search 等任务特定 retriever。
- **Retrieval strategy**：hybrid retrieval 组合稀疏、密集、结构化查询等信号；hierarchical retrieval 先检索文档再检索片段，适合长文档和多层结构。
- **Retrieval timing**：single retrieval 最简单但不适合复杂问题；iterative retrieval 在检索和生成之间多轮交替；adaptive retrieval 让模型判断何时继续检索、何时生成，以平衡准确性和延迟。

因此 MM-RAG 的检索并不是“换一个 multimodal embedding 就结束”。更合理的系统往往需要 **分模态索引 + 跨模态重排序 + 任务感知过滤**。

### 阶段三：Augmentation，把证据组织成模型能用的上下文
检索到证据后，最大问题是：如何把异构证据喂给生成器？

论文把 augmentation 具体拆成五类：

1. **Context reranking**：用更强模型或规则重新评估候选证据，例如 CLIP 图文相似度、MLLM reranker、编辑距离或 reverse re-ranking。
2. **Context selection**：根据分数断崖、重复度或任务规则删除无关/冗余证据，避免模型复制检索结果或被重复上下文稀释注意力。
3. **Context compression**：把长文本、多模态 embedding 或检索文档压缩成更紧凑的表示，例如 soft prompt、Perceiver 压缩或摘要。
4. **Noise injection**：训练时主动加入无关或误导证据，提高模型面对 imperfect retrieval 的鲁棒性，避免过度依赖检索内容。
5. **Context fusion**：把 query 与检索证据融合，可分为 concatenation、Fusion-in-Encoder、Fusion-in-Decoder 和 latent aggregation 等方式。

这比“把检索结果拼进 prompt”细得多。真正的 MM-RAG augmentation 既要处理多模态格式差异，也要处理相关性、冗余、压缩、噪声和融合位置。

<figure class="figure">
  <img src="assets/papers/mm-rag-all-modalities/workflow-tree.webp" alt="MM-RAG 工作流细分树" loading="lazy" />
  <figcaption>图 4：论文进一步展开四阶段工作流下的常见技术组件。它提醒我们，MM-RAG 的每个阶段都可以继续细分，并且每个细分节点都可能需要模态特定优化。</figcaption>
</figure>

### 阶段四：Generation，输出也可以是多模态
很多 RAG 系统默认输出文本，但这篇综述强调：在 MM-RAG 中，输出也可以是图像、音频、视频、代码、表格甚至 3D。

这带来新的挑战：

- **Text 输出**：需要保证答案 faithfully grounded in retrieved evidence。
- **Image 输出**：检索到的图像可能用于实体一致性、风格参考、布局约束或细节补全。
- **Audio 输出**：检索到的声音样例可用于音色、场景声、风格或语义约束。
- **Video 输出**：既要保留视觉一致性，又要处理时间连续性和动作逻辑。
- **Code 输出**：生成结果不仅要语义正确，还要可运行、可集成、符合仓库上下文。
- **Table/KG 输出**：生成内容需要结构合法、关系一致、便于后续查询。

这也是全组合 taxonomy 的意义：一旦输出不再只是文本，RAG 的“增强”就不只是提供事实，而是变成 **对生成空间的检索约束和结构引导**。

## 训练策略：冻结还是可训练？
论文把 MM-RAG 训练策略概括为两大类：**parameter-frozen** 和 **parameter-trainable**。

<figure class="figure">
  <img src="assets/papers/mm-rag-all-modalities/training.webp" alt="MM-RAG 训练策略" loading="lazy" />
  <figcaption>图 5：MM-RAG 的训练策略概览。论文将训练划分为 parameter-frozen 与 parameter-trainable 两类；后者又可细分为训练检索器、训练生成器或训练中间层。</figcaption>
</figure>

### Parameter-frozen：更适合快速搭建和工程落地
冻结策略通常不改动核心 LLM/MLLM 参数，而是通过以下方式适配：

- 设计 query rewriting、routing 或 prompt 模板。
- 使用现成 embedding 模型构建多模态索引。
- 训练轻量投影层或 adapter。
- 通过 reranker、filter、planner 进行检索后优化。

它的优势是成本低、风险小、容易接入商业闭源模型；缺点是系统能力强依赖已有模型的模态理解能力，遇到复杂跨模态对齐时天花板明显。

### Parameter-trainable：更适合强耦合跨模态任务
可训练策略会更新检索器、融合模块、生成器或其中一部分，使系统学习特定任务中的检索-生成配合。

它适合以下场景：

- 领域数据高度专业，如医学影像、遥感、法律文档。
- 检索证据与生成输出之间存在复杂对齐关系。
- 输出是图像、音频、视频等非文本模态，需要把检索内容转化为生成约束。
- 需要优化特定指标，如 factuality、citation、visual grounding、temporal localization。

但它也更依赖高质量训练数据和评估闭环。对于许多仍处早期的模态组合，数据可能比模型结构更稀缺。

<figure class="figure">
  <img src="assets/papers/mm-rag-all-modalities/datasets.webp" alt="MM-RAG 常用训练数据集" loading="lazy" />
  <figcaption>图 6：论文按输入-输出模态组合整理了常用训练数据集。可以看到，图文和文本任务数据最丰富，而音频、视频、3D、代码等组合仍然依赖更少、更分散的数据源。</figcaption>
</figure>

## 评估：MM-RAG 不能只看答案对不对
MM-RAG 的评估至少要同时看两个核心组件：**retriever** 和 **generator**。

<figure class="figure">
  <img src="assets/papers/mm-rag-all-modalities/evaluation.webp" alt="MM-RAG 评估指标和基准" loading="lazy" />
  <figcaption>图 7：论文整理了 MM-RAG 的评估指标和 benchmark。评估不应只看最终答案，还要看检索相关性、检索质量、答案相关性、答案正确性和答案忠实性。</figcaption>
</figure>

### Retriever 评估
检索器不仅要找“语义相似”的内容，还要找“对当前生成有用”的证据。常见指标包括 Recall、Precision、MRR、NDCG，但在多模态场景中还需要关注：

- 检索证据是否覆盖了正确模态？
- 是否找到了关键时间片段、图像区域或表格单元？
- 跨模态检索是否存在模态偏置，例如文本证据总是压过视觉证据？
- 检索到的证据能否被生成器真正利用？

### Generator 评估
生成器评估更复杂，因为输出可能是多模态的：

- 文本答案要看 factuality、faithfulness、citation accuracy。
- 图像生成要看实体一致性、视觉质量、参考图利用程度。
- 视频生成要看时间一致性、动作合理性和检索约束遵循情况。
- 代码生成要看可编译、可运行、测试通过和仓库集成。
- 表格/KG 生成要看结构合法性、关系一致性和可查询性。

论文还指出，现有 MM-RAG benchmark 仍高度集中在 **Text + Image → Text**，尤其是 VQA 和视觉知识问答。代表性基准包括 WebQA、OK-VQA、A-OKVQA、MRAG-BENCH、Visual-RAG、M²RAG、MRAMG-Bench、Dyn-VQA、CogBench 等；复杂文档方向有 OMG-QA、PDF-MVQA、Real-MM-RAG；文本 RAG 中的 RAGAS、ARES、RGB 也可以通过替换为 MLLM 的方式迁移到 MM-RAG 评估。

我认为这部分最值得借鉴的是：**MM-RAG 评估必须拆开看检索质量和生成质量**。最终答案错了，可能是检索没找到，也可能是找到了但生成器没用；最终答案对了，也不代表检索证据真的被利用。

## 应用场景：论文重点讨论了哪些落地方向？
论文把 MM-RAG 的应用总结为四类高潜领域：

- **Healthcare and Medicine**：医学影像分析、临床决策支持和药物发现。DeltaNet 基于医学图像和报告生成诊断报告；MMed-RAG 与 FactMM-RAG 用检索增强提升 Med-LVLM 在放射、眼科、病理报告中的 factuality；IRDIFF 用检索增强 3D diffusion 生成更符合约束的分子；EMERGE 利用电子病历中的时间序列信号和临床笔记支持风险预测。
- **Finance**：围绕 K 线图、财报、合同、票据等金融多模态数据构建知识库，支持可解释投资、风控和股票走势预测。
- **Traffic**：在自动驾驶中结合视觉感知与知识推理。RealGen 通过检索示例组合生成新驾驶训练场景；RAG-Driver 检索专家驾驶示范，用于解释驾驶行为、给出理由和预测控制信号。
- **Education**：整合教材、题库、教学视频、课件图表等多模态教育资源。VL-RAG 为教学场景定制答案库和图片库，使自动化教学能在保留学科材料的同时降低人工维护成本。

这些应用的共同点是：知识不只存在于文本里，而且答案往往需要 **跨模态证据、领域约束和可解释引用**。

## Future Directions：论文认为接下来该做什么？
论文最后给出六个未来方向：

1. **探索未覆盖的输入-输出组合**：54 种组合中只有 18 种已有研究，很多空白组合可能对应新应用。
2. **提升跨模态对齐能力**：文本-图像相对成熟，但 3D、视频、音频、结构化数据之间的复杂对齐仍然薄弱。
3. **增强鲁棒性与安全性**：多模态噪声、误导图片、deepfake 视频、跨数据源知识冲突、corpus poisoning 和 prompt attack 会同时影响检索与生成。
4. **迁移文本 RAG 的成熟方法**：Modular RAG、Parametric RAG、Agentic RAG 等文本 RAG 思路值得扩展到多模态系统。
5. **建设更全面 benchmark**：尤其需要覆盖文本和图像之外的模态，并建立多维度、标准化的评估协议。
6. **结合个性化、Agent、具身智能和专家系统**：MM-RAG 可以与用户画像、工具调用、真实世界感知行动和知识验证机制结合，推动工业落地。

## 我的理解与启发
### 1. “全组合”视角更适合作为研究选题地图
这篇综述最适合用来找方向。它不是告诉你某个单点方法最强，而是告诉你哪些区域已经拥挤、哪些区域仍然空白。

比如 Text + Image → Text 已经有大量 benchmark 和方法；但 Text + Video → Video、Text + Audio → Audio、Table/KG → Text、Code → Code 等方向仍有很多系统设计空间。对研究来说，这种地图能帮助避免在过饱和方向上重复堆模块。

### 2. MM-RAG 的核心不是“多模态输入”，而是“跨模态证据流”
很多系统只是允许用户上传图片，但检索、增强和生成仍然是文本中心的。这更像 “text RAG with image caption”，不一定是真正的 MM-RAG。

更强的系统应该让证据在原始模态中被保留和验证：视频证据要保留时间片段，图像证据要保留区域和 OCR，表格证据要保留行列结构，代码证据要保留依赖关系。

### 3. 输出模态会反过来决定检索策略
如果输出是文本，检索证据主要用于事实 grounding；如果输出是图像，检索证据可能用于风格、布局和实体参考；如果输出是代码，检索证据需要包含 API、类型和调用上下文。

所以 MM-RAG 不应该只问“query 是什么模态”，还要问“最终要生成什么模态”。输出模态决定了检索证据的粒度和组织方式。

### 4. 工程上应优先做可观测、可拆解的 pipeline
由于 MM-RAG 涉及多个模态和多个阶段，端到端黑盒系统很难调试。更实用的做法是把 pipeline 拆开观测：query routing、retrieval recall、rerank quality、context packing、generation grounding、citation/attribution 都应该能单独评估。

## 局限与待观察点
- **论文宽度很大，深度必然有限**：全组合 taxonomy 能提供地图，但每个具体方向仍需要阅读专门论文。
- **组合数量并不等于研究价值**：某些空白组合可能只是需求弱或评估困难，不一定都值得投入。
- **公开 benchmark 仍偏少**：许多输出为图像、音频、视频、3D 的 RAG 系统还缺少统一评测协议。
- **多模态证据利用难验证**：模型可能看似利用了图片或视频，实际仍依赖文本先验，需要更强 grounding 评估。
- **系统成本会快速上升**：跨模态索引、重排序、MLLM 推理和多模态生成都可能带来显著延迟与成本。
## 结论
这篇综述适合当作 MM-RAG 研究和工程选型的“全局地图”。它提醒我们，多模态 RAG 不只是把图片塞进文本 RAG，也不只是为视觉问答加检索；更完整的问题是：**任意输入模态如何检索、组织并约束任意输出模态的生成**。从这个角度看，当前研究只覆盖了潜在空间的一部分，真正有价值的后续工作可能来自那些尚未被系统探索、但具备明确业务场景和评估闭环的模态组合。