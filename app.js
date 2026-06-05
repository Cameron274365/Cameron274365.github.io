    const state = {
      activeCategory: "全部",
      query: "",
      activeNoteId: new URLSearchParams(location.search).get("note") || NOTES[0].id
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
      return text
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

        const headingMatch = trimmed.match(/^(#{2,3})\s+(.+)$/);
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

    function renderNoteList() {
      const filteredNotes = getFilteredNotes();
      elements.noteCount.textContent = `${filteredNotes.length}/${NOTES.length}`;

      if (!filteredNotes.length) {
        elements.noteList.innerHTML = '<div class="empty-state">没有匹配的笔记</div>';
        return;
      }

      elements.noteList.innerHTML = filteredNotes.map(note => `
        <button class="note-card ${note.id === state.activeNoteId ? "active" : ""}" type="button" data-note-id="${note.id}">
          <div class="meta-row"><span>${escapeHtml(note.category)}</span><span>·</span><span>${escapeHtml(note.date)}</span><span>·</span><span>${escapeHtml(note.readTime)}</span></div>
          <h3>${escapeHtml(note.title)}</h3>
          <p>${escapeHtml(note.summary)}</p>
          <div class="tag-row">${note.tags.slice(0, 3).map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>
        </button>
      `).join("");
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
      elements.articleBody.innerHTML = renderMarkdown(note.content);
      renderToc();
    }

    function renderToc() {
      const headings = elements.articleBody.querySelectorAll("h2, h3");
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
    render();
  
