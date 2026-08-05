---
id: "visual-rag-generation-speedup-notes"
title: "视觉 RAG 生成提速调研笔记"
category: "多模态 RAG"
date: "2026-07-13"
order: 49
readTime: "18 min"
tags: ["Multimodal RAG", "Visual RAG", "VLM", "ColPali", "Inference Optimization", "Engineering"]
summary: "面向视觉 RAG 生成端（VLM 一次处理多页图片）的提速调研：定位 prefill 视觉 token 数为主要瓶颈，从降分辨率、减少送图张数、混合文本-图片生成、工程加速、自托管深度优化五条路径给出可落地方案；深入梳理 patch 级证据定位（ColPali 热力图 / HPC-ColPali 压缩 / VRAG-RL 等 agentic 路线）并回答"两条线路分工""ColQwen 能否当 reranker""线路 2 作用时机"三个关键问答。"
---

> 背景：某视觉 RAG 项目中，生成模型（VLM）需要一次处理多张页面图片，
> 生成速度较慢。本文整理优化方向调研、相关研究综述及关键问答。
> 整理日期：2026-07

---

## 1. 瓶颈定位

当前视觉生成链路（检索服务的视觉生成入口、主服务的多模态消息构建函数）的耗时构成：

<div class="table-wrap">
  <table>
    <thead><tr><th>阶段</th><th>现状</th><th>问题</th></tr></thead>
    <tbody>
      <tr><td>图片准备</td><td>top_k=5 页，每页 144 DPI JPEG（A4 约 1240×1754 px），逐张读盘转 base64</td><td>主服务循环内逐张 <code>await</code>，串行</td></tr>
      <tr><td>请求上传</td><td>5 张 base64 内联，请求体可达数 MB</td><td>上传本身有秒级开销</td></tr>
      <tr><td>VLM prefill</td><td>Qwen-VL 按 28×28 patch 切 token，一页约 2500~2800 视觉 token，5 页 1.3 万+ token</td><td><strong>最大头</strong>，prefill 时间近似随视觉 token 数线性增长</td></tr>
      <tr><td>decode</td><td>生成答案</td><td>受 max_tokens 影响</td></tr>
    </tbody>
  </table>
</div>

核心思路只有两条：**少送图、送小图**；其余是工程加速。

---

## 2. 五个优化方向

### 方向一：减少每张图的 token（成本最低，收益最大）

1. **生成端降分辨率**：144 DPI 是为嵌入/存档保真设的，生成时把送 VLM 的图缩到
   长边 ~1024px，单页视觉 token 从 ~2700 降到 ~1000，5 页 prefill 直接砍 60%+。
   不用动索引：在视觉生成入口 / 图片编码函数里加 resize
   （VLM 适配层已有现成压缩工具函数，目前只用在嵌入端），
   或摄取时额外落一份"生成用缩略副本"。
2. **JPEG 质量降到 75~80**：对 VLM 识别几乎无影响，进一步压上传体积。

### 方向二：减少送图张数

1. **自适应 top-k / 分数截断**：按分数落差截断，top1 与后续差距大时只送 1~2 页。
2. **视觉 reranker 精排后裁剪**：单级召回场景下引入精排。用 ColPali 系
   late-interaction 模型（本地已有 Ops-Colqwen3-4B、nemotron-colembed-vl-4b-v2
   权重）或 MonoQwen 类轻量 VLM reranker，召回 10 页精排后只送前 2~3 页。
3. **区域裁剪（研究向）**：检索侧换 ColPali 类 patch 级模型后，用 MaxSim 热力图
   定位证据区域，只裁相关区域送 VLM（详见第 3 节）。

### 方向三：混合文本-图片生成（架构级）

- 摄取时离线做每页 OCR/caption（Celery 异步任务，不占查询延迟）；
- 查询时只有 top-1~2 页送原图，其余页送文本摘要；
- 进一步可做**路由**：先用纯文本（OCR）+ LLM 快速回答，置信度不足或问题涉及
  图表/版式时才走 VLM 图片通道。
- 与文本+图片双通道架构天然契合——文本通道为视觉通道提供降级燃料。

### 方向四：工程与服务层加速

1. **流式输出**：检索服务的视觉生成入口若非流式，改 `stream=True` 后 TTFT 体验立刻改善；
2. **图片准备并行化**：主服务的消息构建函数若逐张 `await asyncio.to_thread(...)`，
   改 `asyncio.gather` 并行读盘+编码；
3. **用 URL 代替 base64**：DashScope 兼容接口支持传 OSS 签名 URL，省掉数 MB 请求体上传；
4. **换更快的模型档位**：qwen-vl-flash 等小杯模型 prefill 明显更快，可按问题复杂度分级调用；
5. **限制 max_tokens + 提示词要求简洁作答**，压 decode 时间。

### 方向五：自托管场景的深度优化（长期）

- **vLLM Automatic Prefix Caching**：对多模态输入做哈希复用图片 KV cache。热点文档页
  被反复检索时，图片放消息前部、按 doc_id+page 稳定排序，同页第二次出现可跳过
  prefill（社区实测 TTFT 480ms→110ms 量级）；
- **视觉 token 剪枝/压缩**：SparseVLM、TopV、LUVC 等 training-free 方法约 2× 推理加速，
  精度几乎无损，适合文档页大量空白/冗余 patch 的输入；
- **map-reduce 并行生成**：每页并发调一次 VLM 出 per-page 答案再汇总，墙钟时间 ≈
  单页耗时 + 汇总；代价是总 token 费用上升、跨页推理变弱。

### 建议落地顺序

1. **立刻做**（几行改动）：生成端降分辨率 + 检索服务流式输出 + 主服务并行编码 → 预计生成延迟降一半以上；
2. **短期**：自适应 top-k / 分数截断，OSS URL 直传；
3. **中期**：视觉 reranker（复用已有 colembed 权重）精排裁剪 + 摄取时离线 caption 的混合生成；
4. **长期**：自托管 vLLM + prefix cache、patch 级证据裁剪。

---

## 3. 深入：patch 级证据定位与区域裁剪（Evidence Cropping）

### 3.1 核心原理

ColPali 系多向量模型把页面切成 patch（约 28×28 px 一个视觉 token），每个 patch 有
独立向量，查询每个 token 与所有 patch 做 MaxSim 打分求和。

关键点：**MaxSim 打分过程天然产生"查询 token × 页面 patch"的相似度矩阵**，
叠加回原图就是证据热力图——哪个区域导致这页被召回一目了然，且是检索时顺带产生、
几乎零额外成本。单向量模型（如当前 qwen3-vl-embedding 整页单向量）没有此能力，
所以这条路要求检索/精排侧先引入 late-interaction 模型。

用法：热力图阈值化 → 连通区域 → 外接框（加 margin）→ 只把裁剪区域送 VLM。
一页 2500+ 视觉 token 可压到几百，且送进去的全是相关内容。

### 3.2 相关研究（三条线）

**线路 1：ColPali 可解释性热力图（基础，已工程可用）**

- ColPali 原论文（arXiv 2407.01449）展示了 similarity map；官方仓库
  illuin-tech/colpali 内置 `interpretability` 模块，可直接生成每个查询词的 patch 热力图；
- *Needle in a Patched Haystack*（AAAI AIES 2025）评测此类 saliency map 定位质量：
  大体可靠，但对细粒度小目标（单个数字）定位有噪声 → 裁剪框要放宽。

**线路 2：基于显著性的 token 压缩（training-free）**

- **HPC-ColPali**（arXiv 2506.21601）：对 patch 向量做层次化压缩 + 注意力显著性剪枝，
  只保留 top-p% 显著 patch，大幅削减视觉 token 且检索精度几乎不掉；
- 通用范式：检索热力图 → 选显著区域 → 裁剪/抽取 → 生成，无需训练任何模型。

**线路 3：让 VLM 自己"选区-裁剪-放大"（agentic/RL，前沿）**

- **VRAG-RL**（阿里通义，arXiv 2505.22019）：定义视觉感知动作空间
  （select / crop / scale），RL 训练 VLM 多轮由粗到细收集信息，该子方向引用最多；
- **ViDoRAG**：多智能体动态迭代推理，agent 决定是否细看某区域；
- **UniDoc-RL**（arXiv 2604.14967）、**MDocRAG-RL**（ACL 2026 Findings）：
  "检索→重排→区域选择→生成"整链统一 RL 端到端优化；
- **FVPMR**（CVPR 2026）：不裁剪，直接在原图上画框/高亮（fine-grained visual prompt）
  引导 VLM 注意力，规避裁剪丢全局上下文的风险；
- RAG 之外的通用 zoom-in 工作：V*/SEAL、Zoom Eye、DeepEyes、Chain-of-Focus
  （树搜索式放大，需多轮 VLM 调用，延迟不一定划算）。

### 3.3 落地形态与权衡

最务实路线：**线路 1+2 的 training-free 组合**——

1. 精排侧用 ColQwen 类多向量模型；
2. 用 similarity map 对命中页生成热力图，阈值化取证据框；
3. 送 VLM 采用 **"全页低清缩略图 + 证据区域高清裁剪"**（类似 UReader/DocOwl 的
   global-local 策略），兼顾全局版式与局部细节。

<div class="table-wrap">
  <table>
    <thead><tr><th>风险</th><th>说明</th><th>缓解</th></tr></thead>
    <tbody>
      <tr><td>丢全局上下文</td><td>跨栏表格、需整页版式的问题受损</td><td>附一张低清全页缩略图</td></tr>
      <tr><td>定位噪声</td><td>对小目标定位不精确</td><td>裁剪框加大 margin，按连通域合并</td></tr>
      <tr><td>检索侧改造成本</td><td>需引入多向量检索/精排</td><td>先只在 rerank 阶段用，主检索不动</td></tr>
      <tr><td>多向量存储膨胀</td><td>每页 ~1000 patch 向量</td><td>HPC-ColPali 式量化/池化，或 rerank-only 不落库</td></tr>
    </tbody>
  </table>
</div>

低侵入切入点：主检索保持单向量 pgvector，**只在精排阶段用 ColQwen 对 top-10
重打分，顺带产出热力图用于裁剪**——一次前向同时解决"送哪几页"和"每页送哪块"。

精读顺序建议：ColPali 原文（similarity map）→ HPC-ColPali（压缩思路）→ VRAG-RL（agentic 上限）。

---

## 4. 关键问答

### Q1：线路 1 和线路 2 有什么区别？

两者用同一个信号（patch 级相似度），但所处层面和目的不同：

<div class="table-wrap">
  <table>
    <thead><tr><th></th><th>线路 1：可解释性热力图</th><th>线路 2：显著性 token 压缩</th></tr></thead>
    <tbody>
      <tr><td>本质</td><td><strong>产生定位信号</strong>的基础能力</td><td><strong>消费定位信号</strong>的效率机制</td></tr>
      <tr><td>作用层面</td><td>像素层：热力图叠回原图，可视化/裁剪图片</td><td>向量层：对 patch embedding 剪枝、量化、池化</td></tr>
      <tr><td>输出</td><td>热力图 / 证据裁剪框</td><td>更小的向量集合</td></tr>
      <tr><td>改变什么</td><td>不改 pipeline，只提供&quot;证据在哪&quot;</td><td>改变索引结构和检索计算量</td></tr>
      <tr><td>典型用途</td><td>裁证据区域送 VLM、高亮出处、调试检索</td><td>解决多向量&quot;每页 1000+ 向量&quot;的存储/算力膨胀</td></tr>
    </tbody>
  </table>
</div>

一句话：线路 1 回答"**证据在页面哪个位置**"（决定送 VLM 什么图），
线路 2 回答"**哪些 patch 向量值得保留**"（降低多向量索引成本）。

### Q2：ColQwen 能当 reranker 吗？

**可以，且是它非常自然的用法**。late-interaction 介于双塔和 cross-encoder 之间：
打分时 query 与文档有 token 级交互（精度接近 cross-encoder），文档侧表示可预先算好。

两种工作模式：

- **模式 A：摄取时预存 patch 向量（推荐）**——入库时对每页额外算多向量表示存下
  （可配合池化/量化，~1000 patch 可压到 100 以内）；查询时只对 query 做一次前向，
  与预存向量做纯矩阵 MaxSim，**毫秒级、无在线图片推理**。代价是索引膨胀和摄取变慢，
  但可放本地文件/单独表，不动现有 pgvector HNSW 索引。
- **模式 B：查询时在线算页面向量**——零存储，但 top-10 页在线 GPU 推理要秒级，
  在"给生成提速"的场景里本末倒置，一般不推荐。

对比 **MonoQwen2-VL** 类 pointwise VLM reranker（query+页面图一起喂 VLM 判相关性）：
精度上限更高，但每个候选页过一次完整 VLM，延迟和成本远高于 MaxSim 打分。

注意点：MaxSim 是求和分、不在 0~1 区间，诸如 `score_threshold=0.3` 这类固定阈值的过滤逻辑不能直接
复用，需改为按分数落差截断或归一化后再设阈值。

### Q3：线路 2 具体作用于什么时候？

**作用在"入库时"（离线一次性），收益体现在"查询时"。**

多向量的代价：

- **存储**：每页 ~1000 patch × 128 维 float16 ≈ 256 KB/页；1 万页 → ~2.5 GB
  （单向量方案仅 ~50 MB，差 50 倍）；
- **打分**：MaxSim = query_token 数 × patch 数的矩阵运算，
  20 token × 1000 patch × 100 候选页 = 200 万次内积。

流水线位置：

```
【入库时 · 离线】
渲染页面 → ColQwen 前向 → ~1000 patch 向量
                              │
                      ┌───────▼────────┐
                      │   ★ 线路 2 ★   │  ← 作用点
                      │ 剪枝/池化/量化  │
                      └───────┬────────┘
                              ▼
                    存入索引（压缩后的向量）

【查询时 · 在线】
query 编码 → 主检索 top-k → MaxSim 精排 ← 收益体现（打分矩阵变小）
                                │
                                ▼
                        热力图 → ★ 线路 1 ★ 裁剪 → 送 VLM
```

三种压缩手段（可叠加：1000 → 剪枝 400 → 池化 ~120 → 量化存储）：

1. **显著性剪枝**：用模型注意力权重删"废 patch"（空白边距、纯色背景），
   HPC-ColPali 可砍 ~60% patch 精度几乎不掉；判断只看页面本身、查询无关，可放心离线做；
2. **token 池化**：语义相近的相邻 patch 层次聚类合并（Answer.AI 推广），
   池化 3~4 倍精度损失通常 <2%，实现最简单；
3. **量化**：K-means 码本（向量→1 字节簇编号）或二值化，只省存储不减 patch 数。

与线路 1 的本质分工：

<div class="table-wrap">
  <table>
    <thead><tr><th></th><th>线路 2</th><th>线路 1</th></tr></thead>
    <tbody>
      <tr><td>时机</td><td>入库时（离线，一次性）</td><td>查询时（在线，每次不同）</td></tr>
      <tr><td>是否知道查询</td><td>❌ 查询无关，删&quot;对谁都没用&quot;的 patch</td><td>✅ 查询相关，定位&quot;对这个问题有用&quot;的区域</td></tr>
      <tr><td>操作对象</td><td>向量（embedding）</td><td>图片（像素）</td></tr>
      <tr><td>优化目标</td><td>索引存储 + 精排打分速度</td><td>送 VLM 的图片 token 数 → 生成速度</td></tr>
    </tbody>
  </table>
</div>

务实结论：只为生成端提速的话线路 2 非必需；它是采纳"ColQwen 精排 + 热力图裁剪"后
应对存储膨胀的配套手段。小规模场景（万页以下）可先不做——top-10 精排预存向量
不压缩也只有几十 MB，先跑通"精排 + 裁剪"，等万页级或精排延迟成瓶颈再补池化/量化
（改动点只在摄取管线，不影响查询侧）。

**线路 1 决定要不要用多向量模型，线路 2 决定用了之后养不养得起。**

---

## 5. 参考文献 / 链接

<div class="table-wrap">
  <table>
    <thead><tr><th>主题</th><th>出处</th></tr></thead>
    <tbody>
      <tr><td>VisRAG：页面图为单元的 VLM 检索生成</td><td>arXiv 2410.10594</td></tr>
      <tr><td>ColPali：late-interaction 视觉文档检索</td><td>arXiv 2407.01449；github.com/illuin-tech/colpali</td></tr>
      <tr><td>HPC-ColPali：层次化 patch 压缩</td><td>arXiv 2506.21601</td></tr>
      <tr><td>VRAG-RL：视觉感知动作空间 + RL</td><td>arXiv 2505.22019</td></tr>
      <tr><td>UniDoc-RL：粗到细统一 RL 视觉 RAG</td><td>arXiv 2604.14967</td></tr>
      <tr><td>MDocRAG-RL</td><td>ACL 2026 Findings</td></tr>
      <tr><td>FVPMR：细粒度视觉提示</td><td>CVPR 2026</td></tr>
      <tr><td>saliency map 质量评测</td><td>Needle in a Patched Haystack, AAAI AIES 2025</td></tr>
      <tr><td>视觉 token 剪枝</td><td>SparseVLM (ICML 2025)、TopV (CVPR 2025)、LUVC (arXiv 2512.09010)</td></tr>
      <tr><td>vLLM 多模态 prefix caching</td><td>docs.vllm.ai（Automatic Prefix Caching / MM Processing）</td></tr>
      <tr><td>多模态 RAG 工程实践</td><td>Microsoft ISE Blog: Multimodal RAG with Vision</td></tr>
    </tbody>
  </table>
</div>
