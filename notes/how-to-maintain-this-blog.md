---
id: "how-to-maintain-this-blog"
title: "如何维护这个静态笔记博客"
category: "站点管理"
date: "2026-06-03"
order: 40
readTime: "4 min"
tags: ["Static Site","Notes","Workflow"]
summary: "这个站点不依赖框架和后端。新增文章时，在 notes/ 目录添加 Markdown 文件，再运行 node scripts/build-notes.js 生成 notes-data.js，页面会自动生成列表、分类、搜索和文章目录。"
hero: ""
---

## 结构说明
这个博客是一个纯静态站点，适合 GitHub Pages：

- **内容层**：每篇笔记维护为 `notes/*.md`，使用 frontmatter 描述元信息，正文使用 Markdown 和少量 HTML。
- **数据层**：`scripts/build-notes.js` 会读取 `notes/*.md` 并生成 `notes-data.js`，前端页面直接读取 `NOTES` 数组。
- **交互层**：搜索、分类筛选、文章切换、目录生成和主题切换都由 `app.js` 中的原生 JavaScript 完成。
- **资源层**：图片放在 `assets/` 目录，文章里使用相对路径引用。

## 新增笔记步骤
1. 在 `notes/` 目录新增一篇 Markdown 文件。
2. 在文件顶部填写 frontmatter：`id`、`title`、`category`、`date`、`order`、`readTime`、`tags`、`summary`、`hero`。
3. 在 frontmatter 下方编写正文内容。
4. 如果有图片，放到 `assets/你的主题/` 目录，并在正文中用 `<figure class="figure">...</figure>` 引用。
5. 运行 `node scripts/build-notes.js` 重新生成 `notes-data.js`。
6. 提交并推送到 GitHub，GitHub Pages 会自动更新。

## 内容建议
- 标题尽量清晰，便于搜索。
- 每篇文章保留“一句话总结”和“我的理解”。
- 论文笔记建议记录：问题背景、核心方法、实验结论、局限、个人启发。
- 对长期知识库来说，标签比时间更重要，建议保持标签稳定。
