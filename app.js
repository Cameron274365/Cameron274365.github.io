    const state = {
      activeCategory: "全部",
      query: "",
      activeNoteId: new URLSearchParams(location.search).get("note") || NOTES[0].id,
      collapsedGroups: new Set(),
      sidebarOpen: false,
      tocOverlayOpen: false
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
      articleHeroBanner: document.querySelector("#articleHeroBanner"),
      tocListDesktop: document.querySelector("#tocListDesktop"),
      tocListMobile: document.querySelector("#tocListMobile"),
      sidebar: document.querySelector("#sidebar"),
      sidebarToggle: document.querySelector("#sidebarToggle"),
      sidebarBackdrop: document.querySelector("#sidebarBackdrop"),
      tocFab: document.querySelector("#tocFab"),
      tocOverlay: document.querySelector("#tocOverlay"),
      tocOverlayClose: document.querySelector("#tocOverlayClose")
    };

    /* ===== Utility ===== */

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    const usedSlugIds = new Map();

    function slugify(text) {
      const base = text
        .toLowerCase()
        .trim()
        .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
        .replace(/^-+|-+$/g, "") || "section";
      if (usedSlugIds.has(base)) {
        const count = usedSlugIds.get(base) + 1;
        usedSlugIds.set(base, count);
        return `${base}-${count}`;
      }
      usedSlugIds.set(base, 0);
      return base;
    }

    function debounce(fn, delay) {
      let timer;
      return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
      };
    }

    /* ===== Markdown Rendering ===== */

    function inlineMarkdown(text) {
      return escapeHtml(text)
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<figure class="figure"><img src="$2" alt="$1" /><figcaption>$1</figcaption></figure>')
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
    }

    function renderMarkdown(markdown) {
      const lines = markdown.trim().split("\n");
      const html = [];
      let paragraph = [];
      const listStack = [];
      let passthrough = false;
      let inCodeBlock = false;
      let codeLang = "";
      let codeLines = [];

      function closeParagraph() {
        if (paragraph.length) {
          html.push(`<p>${inlineMarkdown(paragraph.join(" ").trim())}</p>`);
          paragraph = [];
        }
      }

      function closeAllLists() {
        while (listStack.length) {
          html.push(`</${listStack.pop()}>`);
        }
      }

      function closeCodeBlock() {
        if (inCodeBlock) {
          const langAttr = codeLang ? ` class="language-${escapeHtml(codeLang)}"` : "";
          html.push(`<pre><code${langAttr}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
          inCodeBlock = false;
          codeLang = "";
          codeLines = [];
        }
      }

      for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        const trimmed = line.trim();

        if (inCodeBlock) {
          if (trimmed.startsWith("```")) {
            closeCodeBlock();
          } else {
            codeLines.push(rawLine);
          }
          continue;
        }

        if (trimmed.startsWith("```")) {
          closeParagraph();
          closeAllLists();
          inCodeBlock = true;
          codeLang = trimmed.slice(3).trim();
          continue;
        }

        if (trimmed.startsWith("<") || passthrough) {
          closeParagraph();
          closeAllLists();
          html.push(line);
          passthrough = !(trimmed.endsWith(">") && !trimmed.includes("<table") && !trimmed.includes("<figure") && !trimmed.includes("<div"));
          if (trimmed.includes("</table>") || trimmed.includes("</figure>") || trimmed.includes("</div>")) {
            passthrough = false;
          }
          continue;
        }

        if (!trimmed) {
          closeParagraph();
          closeAllLists();
          continue;
        }

        if (/^[-*_]{3,}$/.test(trimmed)) {
          closeParagraph();
          closeAllLists();
          html.push("<hr />");
          continue;
        }

        const headingMatch = trimmed.match(/^(#{2,4})\s+(.+)$/);
        if (headingMatch) {
          closeParagraph();
          closeAllLists();
          const level = headingMatch[1].length;
          const headingText = inlineMarkdown(headingMatch[2]);
          const id = slugify(headingMatch[2]);
          html.push(`<h${level} id="${id}">${headingText}</h${level}>`);
          continue;
        }

        const indent = line.search(/\S/);
        const listLevel = Math.min(Math.floor(indent / 2), 3);

        const ulMatch = trimmed.match(/^[-*+]\s+(.+)$/);
        if (ulMatch) {
          closeParagraph();
          while (listStack.length > listLevel + 1) {
            html.push(`</${listStack.pop()}>`);
          }
          if (listStack.length <= listLevel) {
            while (listStack.length <= listLevel) {
              html.push("<ul>");
              listStack.push("ul");
            }
          }
          html.push(`<li>${inlineMarkdown(ulMatch[1])}</li>`);
          continue;
        }

        const olMatch = trimmed.match(/^\d+\.\s+(.+)$/);
        if (olMatch) {
          closeParagraph();
          while (listStack.length > listLevel + 1) {
            html.push(`</${listStack.pop()}>`);
          }
          if (listStack.length <= listLevel) {
            while (listStack.length <= listLevel) {
              html.push("<ol>");
              listStack.push("ol");
            }
          }
          html.push(`<li>${inlineMarkdown(olMatch[1])}</li>`);
          continue;
        }

        if (trimmed.startsWith("> ")) {
          closeParagraph();
          closeAllLists();
          html.push(`<blockquote>${inlineMarkdown(trimmed.slice(2))}</blockquote>`);
          continue;
        }

        paragraph.push(trimmed);
      }

      closeParagraph();
      closeAllLists();
      closeCodeBlock();
      return html.join("\n");
    }

    /* ===== Image Lazy Loading ===== */
    /* Using native browser loading="lazy" instead of custom IntersectionObserver.
       More reliable and simpler — WebP images are already small. */

    /* ===== MathJax ===== */

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
      if (!window.MathJax || !window.MathJax.typesetPromise) return;
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
        if (!selection || selection.rangeCount === 0) return;

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

        if (!containsMath) return;
        writeClipboardContent(event, createHtmlFromFragment(selectedContent), selectedContent.textContent);
      });
    }

    /* ===== Data Helpers ===== */

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

    /* ===== Render Functions ===== */

    function renderStats() {
      const categories = new Set(NOTES.map(note => note.category));
      const tags = new Set(NOTES.flatMap(note => note.tags));
      elements.stats.innerHTML = `
        <span>${NOTES.length} 篇笔记</span>
        <span>·</span>
        <span>${categories.size} 个分类</span>
        <span>·</span>
        <span>${tags.size} 个标签</span>
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

      const groups = groupNotesByCategory(filteredNotes);
      const groupsHtml = [];

      for (const [category, notes] of groups) {
        const isExpanded = !state.collapsedGroups.has(category);
        const cardsHtml = notes.map(note => `
          <button class="note-card ${note.id === state.activeNoteId ? "active" : ""}" type="button" data-note-id="${note.id}">
            <h3>${escapeHtml(note.title)}</h3>
            <div class="card-meta">${escapeHtml(note.category)} · ${escapeHtml(note.date)}</div>
            <div class="card-summary">${escapeHtml(note.summary)}</div>
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

    function renderArticle() {
      const filteredNotes = getFilteredNotes();
      let note = NOTES.find(item => item.id === state.activeNoteId);
      if (!filteredNotes.some(item => item.id === state.activeNoteId)) {
        note = filteredNotes[0] || NOTES[0];
        state.activeNoteId = note.id;
      }

      history.replaceState(null, "", `?note=${encodeURIComponent(note.id)}`);
      document.title = `${note.title} · 屿佳的笔记博客`;

      // Hero banner
      if (note.hero) {
        elements.articleHeroBanner.innerHTML = `<img src="${escapeHtml(note.hero)}" alt="${escapeHtml(note.title)}" loading="eager" />`;
      } else {
        elements.articleHeroBanner.innerHTML = "";
      }

      // Merged meta header
      elements.articleHeader.innerHTML = `
        <div class="meta-row">
          <span class="chip active">${escapeHtml(note.category)}</span>
          <span>${escapeHtml(note.date)}</span>
          <span>·</span>
          <span>${escapeHtml(note.readTime)}</span>
        </div>
        <h1>${escapeHtml(note.title)}</h1>
        <div class="tag-row">${note.tags.map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>
      `;

      usedSlugIds.clear();
      elements.articleBody.innerHTML = renderMarkdown(note.content);
      renderMath();
      renderToc();
      initScrollSpy();
    }

    /* ===== TOC ===== */

    function renderToc() {
      const headings = elements.articleBody.querySelectorAll("h2, h3, h4");
      if (!headings.length) {
        const emptyHtml = '<li class="empty-state">暂无目录</li>';
        elements.tocListDesktop.innerHTML = emptyHtml;
        elements.tocListMobile.innerHTML = emptyHtml;
        return;
      }

      const sections = [];
      let currentSection = null;

      for (const heading of headings) {
        if (heading.tagName === "H2") {
          currentSection = { heading, children: [] };
          sections.push(currentSection);
        } else if (currentSection) {
          currentSection.children.push(heading);
        } else {
          sections.push({ heading, children: [] });
        }
      }

      const tocHtml = sections.map(section => {
        const sectionHeading = section.heading;
        const hasChildren = section.children.length > 0;

        if (!hasChildren) {
          return `<li><a class="toc-${sectionHeading.tagName.toLowerCase()}" href="#${sectionHeading.id}">${sectionHeading.textContent}</a></li>`;
        }

        const childrenHtml = section.children.map(child =>
          `<li><a class="toc-${child.tagName.toLowerCase()}" href="#${child.id}">${child.textContent}</a></li>`
        ).join("");

        return `
          <li class="toc-section">
            <div class="toc-section-header">
              <a class="toc-h2" href="#${sectionHeading.id}">${sectionHeading.textContent}</a>
              <button class="toc-toggle" type="button" aria-label="展开/折叠">▾</button>
            </div>
            <ul class="toc-children">${childrenHtml}</ul>
          </li>`;
      }).join("");

      elements.tocListDesktop.innerHTML = tocHtml;
      elements.tocListMobile.innerHTML = tocHtml;
    }

    function bindTocToggleFor(listElement) {
      listElement.addEventListener("click", event => {
        const toggle = event.target.closest(".toc-toggle");
        if (!toggle) return;
        event.preventDefault();
        const section = toggle.closest(".toc-section");
        const children = section.querySelector(".toc-children");
        const isCollapsed = children.classList.toggle("collapsed");
        toggle.textContent = isCollapsed ? "▸" : "▾";
      });
    }

    function handleTocLinkClick(event) {
      const link = event.target.closest("a[href^='#']");
      if (!link) return;
      event.preventDefault();
      const targetId = link.getAttribute("href").slice(1);
      const target = document.getElementById(targetId);
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }

    /* ===== Scroll Spy ===== */

    let spyObserver = null;

    function initScrollSpy() {
      if (spyObserver) spyObserver.disconnect();

      const headings = elements.articleBody.querySelectorAll("h2[id], h3[id], h4[id]");
      if (!headings.length) return;

      const allTocLinks = document.querySelectorAll(".toc-list a[href^='#']");
      allTocLinks.forEach(a => a.classList.remove("active"));

      let currentActiveId = null;

      spyObserver = new IntersectionObserver(entries => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            currentActiveId = entry.target.id;
            break;
          }
        }

        if (!currentActiveId) return;

        [elements.tocListDesktop, elements.tocListMobile].forEach(list => {
          if (!list) return;
          list.querySelectorAll("a").forEach(a => {
            a.classList.toggle("active", a.getAttribute("href") === `#${currentActiveId}`);
          });
        });

        const activeLink = elements.tocListDesktop?.querySelector("a.active");
        if (activeLink) {
          activeLink.scrollIntoView({ block: "nearest", behavior: "smooth" });
        }
      }, {
        rootMargin: "-80px 0px -70% 0px",
        threshold: 0
      });

      headings.forEach(h => spyObserver.observe(h));
    }

    /* ===== Sidebar Drawer ===== */

    function toggleSidebar(open) {
      state.sidebarOpen = open;
      elements.sidebar.classList.toggle("open", open);
      elements.sidebarBackdrop.classList.toggle("visible", open);
      document.body.style.overflow = open ? "hidden" : "";
      if (open) {
        const activeCard = elements.sidebar.querySelector(".note-card.active");
        if (activeCard) {
          activeCard.scrollIntoView({ block: "nearest" });
        }
      }
    }

    /* ===== TOC Overlay (Mobile) ===== */

    function toggleTocOverlay(open) {
      state.tocOverlayOpen = open;
      elements.tocOverlay.classList.toggle("open", open);
    }

    /* ===== Main Render ===== */

    function render() {
      renderStats();
      renderCategories();
      renderNoteList();
      renderArticle();
    }

    /* ===== Theme ===== */

    function initTheme() {
      const savedTheme = localStorage.getItem("notes-theme");
      const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      const theme = savedTheme || (prefersDark ? "dark" : "light");
      document.documentElement.dataset.theme = theme;
    }

    /* ===== Event Binding ===== */

    function bindEvents() {
      // Search with debounce
      elements.searchInput.addEventListener("input", debounce(event => {
        state.query = event.target.value;
        renderNoteList();
        renderArticle();
      }, 200));

      // Category filter
      elements.categoryList.addEventListener("click", event => {
        const button = event.target.closest("[data-category]");
        if (!button) return;
        state.activeCategory = button.dataset.category;
        renderCategories();
        renderNoteList();
        renderArticle();
      });

      // Note list: group toggle + note selection
      elements.noteList.addEventListener("click", event => {
        const groupHeader = event.target.closest(".note-group-header");
        if (groupHeader) {
          const category = groupHeader.dataset.groupCategory;
          const group = groupHeader.closest(".note-group");
          const body = group.querySelector(".note-group-body");
          const chevron = groupHeader.querySelector(".note-group-chevron");

          if (state.collapsedGroups.has(category)) {
            state.collapsedGroups.delete(category);
            body.classList.remove("collapsed");
            groupHeader.classList.add("expanded");
            chevron.textContent = "▾";
          } else {
            state.collapsedGroups.add(category);
            body.classList.add("collapsed");
            groupHeader.classList.remove("expanded");
            chevron.textContent = "▸";
          }
          return;
        }

        const button = event.target.closest("[data-note-id]");
        if (!button) return;
        state.activeNoteId = button.dataset.noteId;
        renderNoteList();
        renderArticle();

        // Close sidebar on mobile after selection
        if (window.innerWidth <= 1200) {
          toggleSidebar(false);
        }
      });

      // Theme toggle
      elements.themeToggle.addEventListener("click", () => {
        const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
        document.documentElement.dataset.theme = nextTheme;
        localStorage.setItem("notes-theme", nextTheme);
      });

      // Sidebar drawer
      elements.sidebarToggle.addEventListener("click", () => toggleSidebar(true));
      elements.sidebarBackdrop.addEventListener("click", () => toggleSidebar(false));

      // Mobile TOC
      elements.tocFab.addEventListener("click", () => toggleTocOverlay(!state.tocOverlayOpen));
      elements.tocOverlayClose.addEventListener("click", () => toggleTocOverlay(false));

      // TOC link clicks (smooth scroll)
      elements.tocListDesktop.addEventListener("click", handleTocLinkClick);
      elements.tocListMobile.addEventListener("click", event => {
        handleTocLinkClick(event);
        toggleTocOverlay(false);
      });

      // Escape key closes overlays
      document.addEventListener("keydown", event => {
        if (event.key === "Escape") {
          if (state.sidebarOpen) toggleSidebar(false);
          if (state.tocOverlayOpen) toggleTocOverlay(false);
        }
      });
    }

    /* ===== Init ===== */

    initTheme();
    bindEvents();
    bindCopyMathSources();
    bindTocToggleFor(elements.tocListDesktop);
    bindTocToggleFor(elements.tocListMobile);
    render();
