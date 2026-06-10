    const state = {
      activeCategory: "全部",
      query: "",
      activeNoteId: new URLSearchParams(location.search).get("note") || NOTES[0].id,
      collapsedGroups: new Set()
    };

    const elements = {
      themeToggle: document.querySelector("#themeToggle"),
      searchInput: document.querySelector("#searchInput"),
      categoryList: document.querySelector("#categoryList"),
      noteList: document.querySelector("#noteList"),
      noteCount: document.querySelector("#noteCount"),
      stats: document.querySelector("#stats"),
      articleHeader: document.querySelector("#articleHeader"),
      articleBody: document.querySelector("#articleBody"),
      tocList: document.querySelector("#tocList")
    };

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function slugify(text) {
      return text
        .toLowerCase()
        .trim()
        .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
        .replace(/^-+|-+$/g, "") || "section";
    }

    function inlineMarkdown(text) {
      return escapeHtml(text)
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
    }

    function renderMarkdown(markdown) {
      const lines = markdown.trim().split("\n");
      const html = [];
      let paragraph = [];
      let listOpen = false;
      let passthrough = false;

      function closeParagraph() {
        if (paragraph.length) {
          html.push(`<p>${inlineMarkdown(paragraph.join(" ").trim())}</p>`);
          paragraph = [];
        }
      }

      function closeList() {
        if (listOpen) {
          html.push("</ul>");
          listOpen = false;
        }
      }

      for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        const trimmed = line.trim();

        if (trimmed.startsWith("<") || passthrough) {
          closeParagraph();
          closeList();
          html.push(line);
          passthrough = !(trimmed.endsWith(">") && !trimmed.includes("<table") && !trimmed.includes("<figure") && !trimmed.includes("<div"));
          if (trimmed.includes("</table>") || trimmed.includes("</figure>") || trimmed.includes("</div>")) {
            passthrough = false;
          }
          continue;
        }

        if (!trimmed) {
          closeParagraph();
          closeList();
          continue;
        }

        const headingMatch = trimmed.match(/^(#{2,4})\s+(.+)$/);
        if (headingMatch) {
          closeParagraph();
          closeList();
          const level = headingMatch[1].length;
          const headingText = inlineMarkdown(headingMatch[2]);
          const id = slugify(headingMatch[2]);
          html.push(`<h${level} id="${id}">${headingText}</h${level}>`);
          continue;
        }

        if (trimmed.startsWith("- ")) {
          closeParagraph();
          if (!listOpen) {
            html.push("<ul>");
            listOpen = true;
          }
          html.push(`<li>${inlineMarkdown(trimmed.slice(2))}</li>`);
          continue;
        }

        if (trimmed.startsWith("> ")) {
          closeParagraph();
          closeList();
          html.push(`<blockquote>${inlineMarkdown(trimmed.slice(2))}</blockquote>`);
          continue;
        }

        paragraph.push(trimmed);
      }

      closeParagraph();
      closeList();
      return html.join("\n");
    }

    function getCategories() {
      return ["全部", ...Array.from(new Set(NOTES.map(note => note.category)))];
    }

    function getFilteredNotes() {
      const normalizedQuery = state.query.trim().toLowerCase();
      return NOTES.filter(note => {
        const categoryMatched = state.activeCategory === "全部" || note.category === state.activeCategory;
        const searchable = [note.title, note.summary, note.category, note.tags.join(" "), note.content].join(" ").toLowerCase();
        return categoryMatched && (!normalizedQuery || searchable.includes(normalizedQuery));
      });
    }

    function renderStats() {
      const categories = new Set(NOTES.map(note => note.category));
      const tags = new Set(NOTES.flatMap(note => note.tags));
      elements.stats.innerHTML = `
        <div class="stat"><strong>${NOTES.length}</strong><span>篇笔记</span></div>
        <div class="stat"><strong>${categories.size}</strong><span>个分类</span></div>
        <div class="stat"><strong>${tags.size}</strong><span>个标签</span></div>
      `;
    }

    function renderCategories() {
      elements.categoryList.innerHTML = getCategories().map(category => `
        <button class="chip ${category === state.activeCategory ? "active" : ""}" type="button" data-category="${escapeHtml(category)}">${escapeHtml(category)}</button>
      `).join("");
    }

    function groupNotesByCategory(notes) {
      const groups = new Map();
      for (const note of notes) {
        const category = note.category;
        if (!groups.has(category)) {
          groups.set(category, []);
        }
        groups.get(category).push(note);
      }
      return groups;
    }

    function renderNoteList() {
      const filteredNotes = getFilteredNotes();
      elements.noteCount.textContent = `${filteredNotes.length}/${NOTES.length}`;

      if (!filteredNotes.length) {
        elements.noteList.innerHTML = '<div class="empty-state">没有匹配的笔记</div>';
        return;
      }

      const groups = groupNotesByCategory(filteredNotes);
      const groupsHtml = [];

      for (const [category, notes] of groups) {
        const isExpanded = !state.collapsedGroups || !state.collapsedGroups.has(category);
        const cardsHtml = notes.map(note => `
          <button class="note-card ${note.id === state.activeNoteId ? "active" : ""}" type="button" data-note-id="${note.id}">
            <div class="meta-row"><span>${escapeHtml(note.date)}</span><span>·</span><span>${escapeHtml(note.readTime)}</span></div>
            <h3>${escapeHtml(note.title)}</h3>
            <p>${escapeHtml(note.summary)}</p>
            <div class="tag-row">${note.tags.slice(0, 3).map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>
          </button>
        `).join("");

        groupsHtml.push(`
          <div class="note-group">
            <button class="note-group-header ${isExpanded ? "expanded" : ""}" type="button" data-group-category="${escapeHtml(category)}">
              <span class="note-group-chevron">${isExpanded ? "▾" : "▸"}</span>
              <span class="note-group-title">${escapeHtml(category)}</span>
              <span class="note-group-count">${notes.length}</span>
            </button>
            <div class="note-group-body ${isExpanded ? "" : "collapsed"}">${cardsHtml}</div>
          </div>
        `);
      }

      elements.noteList.innerHTML = groupsHtml.join("");
    }

    function markMathSources(rootElement) {
      const mathPattern = /(\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\]|\$\$[\s\S]+?\$\$|(?<!\$)\$[^$\n]+?\$(?!\$))/g;
      const textNodes = [];
      const walker = document.createTreeWalker(rootElement, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
          const parentElement = node.parentElement;
          if (!parentElement || parentElement.closest("code, pre, script, style, textarea, [data-tex]")) {
            return NodeFilter.FILTER_REJECT;
          }

          return mathPattern.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
        }
      });

      while (walker.nextNode()) {
        textNodes.push(walker.currentNode);
      }

      for (const textNode of textNodes) {
        const fragment = document.createDocumentFragment();
        const text = textNode.nodeValue;
        let lastIndex = 0;
        mathPattern.lastIndex = 0;

        for (const match of text.matchAll(mathPattern)) {
          if (match.index > lastIndex) {
            fragment.append(document.createTextNode(text.slice(lastIndex, match.index)));
          }

          const mathSource = match[0];
          const mathElement = document.createElement("span");
          mathElement.className = "math-source";
          mathElement.dataset.tex = mathSource;
          mathElement.textContent = mathSource;
          fragment.append(mathElement);
          lastIndex = match.index + mathSource.length;
        }

        if (lastIndex < text.length) {
          fragment.append(document.createTextNode(text.slice(lastIndex)));
        }

        textNode.replaceWith(fragment);
      }
    }

    function renderMath() {
      markMathSources(elements.articleBody);

      if (!window.MathJax || !window.MathJax.typesetPromise) {
        return;
      }

      window.MathJax.typesetPromise([elements.articleBody]).catch(error => {
        console.warn("MathJax render failed", error);
      });
    }

    function writeClipboardContent(event, htmlContent, plainTextContent) {
      event.clipboardData.setData("text/html", htmlContent);
      event.clipboardData.setData("text/plain", plainTextContent);
      event.preventDefault();
    }

    function createHtmlFromFragment(fragment) {
      const container = document.createElement("div");
      container.append(fragment.cloneNode(true));
      return container.innerHTML;
    }

    function bindCopyMathSources() {
      elements.articleBody.addEventListener("copy", event => {
        const selection = window.getSelection();
        if (!selection || selection.rangeCount === 0) {
          return;
        }

        const range = selection.getRangeAt(0);
        const startElement = range.startContainer.nodeType === Node.ELEMENT_NODE
          ? range.startContainer
          : range.startContainer.parentElement;
        const selectedMathElement = startElement && startElement.closest("[data-tex]");

        if (selectedMathElement && selectedMathElement.dataset.tex) {
          const mathSource = selectedMathElement.dataset.tex;
          writeClipboardContent(event, escapeHtml(mathSource), mathSource);
          return;
        }

        const selectedContent = range.cloneContents();
        let containsMath = false;
        selectedContent.querySelectorAll("[data-tex]").forEach(mathElement => {
          containsMath = true;
          mathElement.textContent = mathElement.dataset.tex;
        });

        if (!containsMath) {
          return;
        }

        writeClipboardContent(event, createHtmlFromFragment(selectedContent), selectedContent.textContent);
      });
    }

    let imageObserver = null;

    function setupImageLazyLoad() {
      if (imageObserver) {
        imageObserver.disconnect();
      }

      const lazyImages = elements.articleBody.querySelectorAll("img[data-src]");
      if (!lazyImages.length) return;

      imageObserver = new IntersectionObserver((entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;
            img.removeAttribute("data-src");
            imageObserver.unobserve(img);
          }
        }
      }, { rootMargin: "200px 0px" });

      for (const img of lazyImages) {
        imageObserver.observe(img);
      }
    }

    function deferImages(html) {
      return html.replace(/<img\s([^>]*?)src="([^"]+)"([^>]*?)>/g, (match, before, src, after) => {
        const cleanBefore = before.replace(/loading="[^"]*"\s*/g, "");
        const cleanAfter = after.replace(/loading="[^"]*"\s*/g, "");
        return `<img ${cleanBefore}data-src="${src}" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"${cleanAfter}>`;
      });
    }

    function renderArticle() {
      const filteredNotes = getFilteredNotes();
      let note = NOTES.find(item => item.id === state.activeNoteId);
      if (!filteredNotes.some(item => item.id === state.activeNoteId)) {
        note = filteredNotes[0] || NOTES[0];
        state.activeNoteId = note.id;
      }

      history.replaceState(null, "", `?note=${encodeURIComponent(note.id)}`);
      document.title = `${note.title} · 屿佳的笔记博客`;

      elements.articleHeader.innerHTML = `
        <div class="meta-row"><span>${escapeHtml(note.category)}</span><span>·</span><span>${escapeHtml(note.date)}</span><span>·</span><span>${escapeHtml(note.readTime)}</span></div>
        <h1>${escapeHtml(note.title)}</h1>
        <p class="article-summary">${escapeHtml(note.summary)}</p>
        <div class="tag-row">${note.tags.map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>
      `;
      elements.articleBody.innerHTML = deferImages(renderMarkdown(note.content));
      setupImageLazyLoad();
      renderMath();
      renderToc();
    }

    function renderToc() {
      const headings = elements.articleBody.querySelectorAll("h2, h3, h4");
      if (!headings.length) {
        elements.tocList.innerHTML = '<li class="empty-state">暂无目录</li>';
        return;
      }

      elements.tocList.innerHTML = Array.from(headings).map(heading => `
        <li><a class="toc-${heading.tagName.toLowerCase()}" href="#${heading.id}">${heading.textContent}</a></li>
      `).join("");
    }

    function render() {
      renderStats();
      renderCategories();
      renderNoteList();
      renderArticle();
    }

    function bindEvents() {
      elements.searchInput.addEventListener("input", event => {
        state.query = event.target.value;
        renderNoteList();
        renderArticle();
      });

      elements.categoryList.addEventListener("click", event => {
        const button = event.target.closest("[data-category]");
        if (!button) return;
        state.activeCategory = button.dataset.category;
        renderCategories();
        renderNoteList();
        renderArticle();
      });

      elements.noteList.addEventListener("click", event => {
        const groupHeader = event.target.closest("[data-group-category]");
        if (groupHeader) {
          const category = groupHeader.dataset.groupCategory;
          if (state.collapsedGroups.has(category)) {
            state.collapsedGroups.delete(category);
          } else {
            state.collapsedGroups.add(category);
          }
          renderNoteList();
          return;
        }

        const button = event.target.closest("[data-note-id]");
        if (!button) return;
        state.activeNoteId = button.dataset.noteId;
        renderNoteList();
        renderArticle();
        window.scrollTo({ top: document.querySelector(".layout").offsetTop - 80, behavior: "smooth" });
      });

      elements.themeToggle.addEventListener("click", () => {
        const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
        document.documentElement.dataset.theme = nextTheme;
        localStorage.setItem("notes-theme", nextTheme);
        elements.themeToggle.textContent = nextTheme === "dark" ? "浅色" : "深色";
      });
    }

    function initTheme() {
      const savedTheme = localStorage.getItem("notes-theme");
      const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      const theme = savedTheme || (prefersDark ? "dark" : "light");
      document.documentElement.dataset.theme = theme;
      elements.themeToggle.textContent = theme === "dark" ? "浅色" : "深色";
    }

    initTheme();
    bindEvents();
    bindCopyMathSources();
    render();
  
