---
name: LaTeX 公式渲染问题
description: 网页中 LaTeX 公式如 \(V_t\)、\(p_t \geq \tau\) 无法正确显示，需要检查前端 Markdown 渲染器配置。
type: insight
tags: [latex, formula, frontend]
scope: project
createdAt: 2026-06-08T01:51:32.148Z
source: reflection
convertedTo: null
---

用户指出论文笔记中的 LaTeX 数学公式（如 V_t, p_t ≥ τ）在网页上无法正常渲染。这可能是由于前端 Markdown 解析器未启用 MathJax 或 KaTeX 支持导致的。
**Why:** 用户明确指出了这个技术问题，说明它影响了文档的可读性。
**How to apply:** 在处理学术论文笔记时，应确保前端环境支持 LaTeX 数学公式的渲染，或者提供替代的文字描述方式。
