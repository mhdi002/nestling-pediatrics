/**
 * Nestling — parental SPA
 * API_BASE is relative so Docker / reverse-proxy can serve /api alongside /web
 */
(function () {
  "use strict";

  const API_BASE = "/api";
  const STORAGE_CHILD = "nestling_active_child";
  const STORAGE_SESSION = "nestling_chat_session";
  const STORAGE_LANG = "nestling_lang";

  const ASQ_AGES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 27, 30, 33, 36, 42, 48, 54, 60];

  const state = {
    children: [],
    activeChild: null,
    chatSessionId: localStorage.getItem(STORAGE_SESSION) || null,
    quiz: null,
    lang: localStorage.getItem(STORAGE_LANG) || "en",
  };
  // normalize after helpers exist — set below after function defs via boot

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  function t(key, vars) {
    const pack = (window.NESTLING_I18N && window.NESTLING_I18N[state.lang]) || {};
    let s =
      pack[key] != null
        ? pack[key]
        : (window.NESTLING_I18N && window.NESTLING_I18N.en[key]) || key;
    if (vars && typeof s === "string") {
      Object.keys(vars).forEach((k) => {
        s = s.replace(new RegExp(`\\{${k}\\}`, "g"), String(vars[k]));
      });
    }
    return s;
  }

  function applyI18n() {
    const pack = (window.NESTLING_I18N && window.NESTLING_I18N[state.lang]) || window.NESTLING_I18N.en;
    document.documentElement.lang = pack.lang || state.lang;
    document.documentElement.dir = pack.dir || "ltr";
    document.title = pack.brand || "Nestling";
    $$("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (key && pack[key] != null) el.textContent = pack[key];
    });
    $$("[data-i18n-placeholder]").forEach((el) => {
      const key = el.getAttribute("data-i18n-placeholder");
      if (key && pack[key] != null) el.setAttribute("placeholder", pack[key]);
    });
    $$("[data-i18n-title]").forEach((el) => {
      const key = el.getAttribute("data-i18n-title");
      if (key && pack[key] != null) {
        el.setAttribute("title", pack[key]);
        if (el.hasAttribute("aria-label")) el.setAttribute("aria-label", pack[key]);
      }
    });
    const toggle = $("#lang-toggle");
    if (toggle) toggle.textContent = pack.langToggle || "فارسی";
  }

  function setLang(lang) {
    state.lang = lang === "fa" ? "fa" : "en";
    localStorage.setItem(STORAGE_LANG, state.lang);
    applyI18n();
    // Refresh chat welcome in the new language
    const thread = $("#chat-thread");
    if (thread && thread.dataset.ready) {
      const first = thread.querySelector(".bubble.assistant");
      if (first && thread.querySelectorAll(".bubble.user").length === 0) {
        first.textContent = t("chatWelcome");
      }
    }
    loadChildren().catch(() => {});
  }

  /* —— Utils —— */
  function loadJson(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch {
      return fallback;
    }
  }

  function saveActiveChild(child) {
    const normalized = normalizeChild(child);
    state.activeChild = normalized;
    if (normalized) localStorage.setItem(STORAGE_CHILD, JSON.stringify(normalized));
    else localStorage.removeItem(STORAGE_CHILD);
    renderChildChip();
  }

  function normalizeChild(raw) {
    if (!raw) return null;
    // API create returns { child_id, child }; list items are flat rows
    const c = raw.child && typeof raw.child === "object" ? { ...raw.child } : { ...raw };
    const id = c.child_id || c.id || raw.child_id || raw.id;
    if (id) c.child_id = id;
    return c.name ? c : null;
  }

  function renderChildChip() {
    const chip = $("#active-child-chip");
    if (state.activeChild && state.activeChild.name) {
      chip.hidden = false;
      const ga = state.activeChild.gestational_age_weeks;
      const mat =
        ga != null ? (Number(ga) < 37 ? t("preterm") : t("term")) : "";
      chip.textContent = mat
        ? `${state.activeChild.name} · ${mat}`
        : state.activeChild.name;
      chip.title = t("activeChildTitle", { name: state.activeChild.name });
    } else {
      chip.hidden = true;
      chip.textContent = "";
    }
  }

  async function loadChildDossier(childId) {
    const panel = $("#child-dossier");
    const body = $("#child-dossier-body");
    if (!panel || !body || !childId) {
      if (panel) panel.hidden = true;
      return;
    }
    try {
      const data = await api(`/children/${encodeURIComponent(childId)}/dossier`);
      const p = data.profile || {};
      const growth = data.growth || [];
      const screens = data.screenings || [];
      const overlays = data.overlays || [];
      const growthHtml = growth.length
        ? `<ul class="dossier-list">${growth
            .slice()
            .reverse()
            .slice(0, 8)
            .map(
              (g) =>
                `<li><strong>${escapeHtml(g.measure)}</strong> ${escapeHtml(String(g.value))} @ ${escapeHtml(
                  String(g.weeks)
                )}w · centile ${escapeHtml(String(g.centile != null ? Number(g.centile).toFixed(1) : "—"))}</li>`
            )
            .join("")}</ul>`
        : `<p class="muted">${escapeHtml(t("noGrowthYet"))}</p>`;
      const screenHtml = screens.length
        ? `<ul class="dossier-list">${screens
            .slice(-5)
            .map(
              (s) =>
                `<li><strong>${escapeHtml(s.instrument || "")}</strong> — ${escapeHtml(
                  (s.result && s.result.summary) || ""
                )}</li>`
            )
            .join("")}</ul>`
        : `<p class="muted">${escapeHtml(t("noScreensYet"))}</p>`;
      const chartsHtml = overlays.length
        ? `<div class="dossier-charts">${overlays
            .map(
              (o) =>
                `<a href="${escapeHtml(o.url)}" target="_blank" rel="noopener" title="${escapeHtml(
                  o.measure || o.filename || "chart"
                )}"><img class="overlay-img" src="${escapeHtml(
                  o.url
                )}" alt="${escapeHtml(o.measure || t("chartOverlayAlt"))}" /><span class="muted">${escapeHtml(
                  o.measure || ""
                )}</span></a>`
            )
            .join("")}</div>`
        : "";
      body.innerHTML = `
        <p class="dossier-summary"><strong>${escapeHtml(p.name || "")}</strong>
          · ${escapeHtml(String(p.sex || ""))}
          · ${escapeHtml(t("gaLabel"))} ${escapeHtml(String(p.gestational_age_weeks ?? "—"))}w
          · ${escapeHtml(data.maturity || "")}</p>
        <h4>${escapeHtml(t("growthCheck"))}</h4>
        ${growthHtml}
        <h4>${escapeHtml(t("screeningTitle"))}</h4>
        ${screenHtml}
        ${
          chartsHtml
            ? `<h4>${escapeHtml(t("chartsSaved"))}</h4>${chartsHtml}`
            : ""
        }
      `;
      panel.hidden = false;
    } catch (err) {
      body.innerHTML = `<p class="muted">${escapeHtml(err.message)}</p>`;
      panel.hidden = false;
    }
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function toast(message, type = "info") {
    const el = $("#toast");
    el.textContent = message;
    el.classList.toggle("error", type === "error");
    el.hidden = false;
    el.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => {
      el.classList.remove("show");
      setTimeout(() => {
        el.hidden = true;
      }, 350);
    }, 3200);
  }

  function setLoading(btn, loading) {
    if (!btn) return;
    btn.disabled = loading;
    btn.classList.toggle("loading", loading);
    const spin = $(".spinner", btn);
    if (spin) spin.hidden = !loading;
  }

  async function api(path, options = {}) {
    const opts = {
      headers: { Accept: "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}) },
      ...options,
    };
    if (opts.body && typeof opts.body === "object") {
      opts.body = JSON.stringify(opts.body);
    }
    let res;
    try {
      res = await fetch(`${API_BASE}${path}`, opts);
    } catch (err) {
      const e = new Error(t("serverUnreachable"));
      e.cause = err;
      throw e;
    }
    let data = null;
    const text = await res.text();
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = { raw: text };
      }
    }
    if (!res.ok) {
      const msg =
        (data && (data.detail || data.message || data.error)) ||
        `Request failed (${res.status})`;
      const e = new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      e.status = res.status;
      e.data = data;
      throw e;
    }
    return data;
  }

  function overlaySrc(data) {
    if (!data) return null;
    if (data.overlay_url) return data.overlay_url;
    if (data.image_url) return data.image_url;
    if (data.overlay_image) return data.overlay_image;
    if (data.overlay_filename) {
      return `/api/overlays/${encodeURIComponent(data.overlay_filename)}`;
    }
    // API often returns bare filename in `overlay`
    if (data.overlay && !/[\\/]/.test(data.overlay)) {
      return `/api/overlays/${encodeURIComponent(data.overlay)}`;
    }
    const path = data.overlay_path || data.overlay;
    if (path) {
      return `/api/overlays/${encodeURIComponent(String(path).split(/[/\\]/).pop())}`;
    }
    return null;
  }

  /* —— Routing —— */
  const routes = {
    "/": "view-home",
    "/child": "view-child",
    "/chat": "view-chat",
    "/growth": "view-growth",
    "/screening": "view-screening",
  };

  function currentPath() {
    const hash = location.hash.replace(/^#/, "") || "/";
    const path = hash.split("?")[0];
    return path.startsWith("/") ? path : `/${path}`;
  }

  function navigate() {
    const path = currentPath();
    const id = routes[path] || routes["/"];
    $$(".view").forEach((v) => {
      v.hidden = v.id !== id;
    });
    const topbar = $("#topbar");
    topbar.style.opacity = path === "/" ? "0.92" : "1";

    if (path === "/child") loadChildren();
    if (path === "/chat") initChat();
    if (path === "/growth") {
      fillChildSelects();
      syncGrowthSexFromChild();
    }
    if (path === "/screening") {
      fillChildSelects();
      if (!state.quiz) showScreeningPicker();
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  /* —— Home health —— */
  async function checkHealth() {
    const el = $("#health-status");
    try {
      await api("/health");
      el.textContent = t("ready");
      el.className = "home-foot ok";
    } catch {
      el.textContent = t("offline");
      el.className = "home-foot bad";
    }
  }

  /* —— Children —— */
  function fillChildSelects() {
    const opts = [`<option value="">${escapeHtml(t("none"))}</option>`]
      .concat(
        state.children.map((c) => {
          const id = c.child_id || c.id;
          const selected =
            state.activeChild && (state.activeChild.child_id || state.activeChild.id) === id
              ? " selected"
              : "";
          return `<option value="${escapeHtml(id)}"${selected}>${escapeHtml(c.name)}</option>`;
        })
      )
      .join("");
    const g = $("#growth-child");
    const s = $("#screening-child");
    if (g) g.innerHTML = opts;
    if (s) s.innerHTML = opts;
  }

  async function loadChildren() {
    const list = $("#children-list");
    try {
      const data = await api("/children");
      state.children = Array.isArray(data) ? data : data.children || [];
      fillChildSelects();
      if (!state.children.length) {
        list.innerHTML = `<p class="muted">${escapeHtml(t("noChildrenYet"))}</p>`;
        return;
      }
      const activeId = state.activeChild && (state.activeChild.child_id || state.activeChild.id);
      list.innerHTML = state.children
        .map((c) => {
          const id = c.child_id || c.id;
          const gaNum = c.gestational_age_weeks;
          const maturity =
            gaNum != null ? (Number(gaNum) < 37 ? t("preterm") : t("term")) : "";
          const ga = gaNum != null ? `${t("gaLabel")} ${gaNum}w · ${maturity}` : "";
          const sex = c.sex ? String(c.sex) : "";
          const meta = [sex, ga].filter(Boolean).join(" · ");
          return `<button type="button" class="child-item${id === activeId ? " active" : ""}" data-id="${escapeHtml(id)}" role="listitem">
            <span>${escapeHtml(c.name)}</span>
            <span class="meta">${escapeHtml(meta)}</span>
          </button>`;
        })
        .join("");
      // Deduplicate visual noise: keep latest 12 for display if many demo rows
      const items = $$(".child-item", list);
      if (items.length > 12) {
        items.slice(0, items.length - 12).forEach((el) => el.remove());
      }
      $$(".child-item", list).forEach((btn) => {
        btn.addEventListener("click", async () => {
          const child = state.children.find((c) => (c.child_id || c.id) === btn.dataset.id);
          saveActiveChild(child);
          // Rebind chat session to this child
          state.chatSessionId = null;
          localStorage.removeItem(STORAGE_SESSION);
          const thread = $("#chat-thread");
          if (thread) {
            thread.dataset.ready = "";
            thread.innerHTML = "";
          }
          await loadChildDossier(child.child_id || child.id);
          fillChildSelects();
          $$(".child-item", list).forEach((b) =>
            b.classList.toggle("active", b.dataset.id === (child.child_id || child.id))
          );
          toast(t("childSelected", { name: child.name }));
        });
      });
      if (activeId) await loadChildDossier(activeId);
      else {
        const panel = $("#child-dossier");
        if (panel) panel.hidden = true;
      }
    } catch (err) {
      list.innerHTML = `<p class="muted">${escapeHtml(err.message)}</p>`;
    }
  }

  function wireChildForm() {
    $("#child-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const form = e.target;
      const btn = $("#child-submit");
      const fd = new FormData(form);
      const body = {
        name: String(fd.get("name") || "").trim(),
        sex: fd.get("sex"),
        gestational_age_weeks: Number(fd.get("gestational_age_weeks")),
      };
      setLoading(btn, true);
      try {
        const created = await api("/children", { method: "POST", body });
        saveActiveChild(created);
        form.reset();
        toast(t("childSaved"));
        await loadChildren();
        const id = state.activeChild && state.activeChild.child_id;
        if (id) await loadChildDossier(id);
      } catch (err) {
        toast(err.message, "error");
      } finally {
        setLoading(btn, false);
      }
    });
  }

  /* —— Chat —— */
  async function ensureChatSession() {
    if (state.chatSessionId) return state.chatSessionId;
    const childId =
      (state.activeChild && (state.activeChild.child_id || state.activeChild.id)) || undefined;
    const data = await api("/sessions", {
      method: "POST",
      body: childId ? { child_id: childId } : {},
    });
    state.chatSessionId = data.session_id;
    localStorage.setItem(STORAGE_SESSION, data.session_id);
    return state.chatSessionId;
  }

  async function startNewChat() {
    state.chatSessionId = null;
    localStorage.removeItem(STORAGE_SESSION);
    const thread = $("#chat-thread");
    if (thread) {
      thread.dataset.ready = "";
      thread.innerHTML = "";
    }
    const panel = $("#chat-history-panel");
    if (panel) panel.hidden = true;
    await ensureChatSession();
    initChat();
    toast(t("newChatStarted"), "ok");
  }

  async function loadChatHistoryList() {
    const panel = $("#chat-history-panel");
    const list = $("#chat-history-list");
    if (!panel || !list) return;
    panel.hidden = false;
    list.innerHTML = `<p class="muted">${escapeHtml(t("loading"))}</p>`;
    const childId =
      (state.activeChild && (state.activeChild.child_id || state.activeChild.id)) || "";
    const q = childId ? `?child_id=${encodeURIComponent(childId)}&limit=30` : "?limit=30";
    const data = await api(`/sessions${q}`);
    const sessions = data.sessions || [];
    if (!sessions.length) {
      list.innerHTML = `<p class="muted">${escapeHtml(t("noChatHistory"))}</p>`;
      return;
    }
    list.innerHTML = sessions
      .map((s) => {
        const title = (s.title || s.preview || s.session_id || "").slice(0, 80);
        const meta = `${s.message_count || 0} · ${(s.updated_at || "").slice(0, 16)}`;
        return `<button type="button" class="history-item" data-sid="${escapeHtml(
          s.session_id
        )}"><strong>${escapeHtml(title || t("chatFallbackTitle"))}</strong><span class="muted">${escapeHtml(
          meta
        )}</span></button>`;
      })
      .join("");
    list.querySelectorAll(".history-item").forEach((btn) => {
      btn.addEventListener("click", () => openChatSession(btn.getAttribute("data-sid")));
    });
  }

  async function openChatSession(sessionId) {
    if (!sessionId) return;
    const data = await api(`/sessions/${encodeURIComponent(sessionId)}`);
    state.chatSessionId = sessionId;
    localStorage.setItem(STORAGE_SESSION, sessionId);
    const thread = $("#chat-thread");
    thread.dataset.ready = "1";
    thread.innerHTML = "";
    const hist = data.history || [];
    if (!hist.length) {
      appendBubble("assistant", t("chatWelcome"));
    } else {
      hist.forEach((m) => {
        if (m.role === "user" || m.role === "assistant") {
          appendBubble(m.role, m.content || "");
        }
        if (m.tool_calls) {
          appendToolResult({ tool_results: m.tool_calls });
        }
      });
    }
    const panel = $("#chat-history-panel");
    if (panel) panel.hidden = true;
    toast(t("chatOpened"), "ok");
  }

  function initChat() {
    const thread = $("#chat-thread");
    if (!thread.dataset.ready) {
      thread.dataset.ready = "1";
      thread.innerHTML = "";
      appendBubble("assistant", t("chatWelcome"));
    }
    ensureChatSession().catch((err) => toast(err.message, "error"));
    const newBtn = $("#btn-new-chat");
    const histBtn = $("#btn-chat-history");
    if (newBtn && !newBtn.dataset.wired) {
      newBtn.dataset.wired = "1";
      newBtn.addEventListener("click", () => {
        startNewChat().catch((err) => toast(err.message, "error"));
      });
    }
    if (histBtn && !histBtn.dataset.wired) {
      histBtn.dataset.wired = "1";
      histBtn.addEventListener("click", () => {
        const panel = $("#chat-history-panel");
        if (panel && !panel.hidden) {
          panel.hidden = true;
          return;
        }
        loadChatHistoryList().catch((err) => toast(err.message, "error"));
      });
    }
  }

  function appendBubble(role, text) {
    const thread = $("#chat-thread");
    const div = document.createElement("div");
    div.className = `bubble ${role}`;
    div.textContent = text;
    thread.appendChild(div);
    thread.scrollTop = thread.scrollHeight;
    return div;
  }

  function prefersReducedMotion() {
    try {
      return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    } catch (_) {
      return false;
    }
  }

  /**
   * Stream assistant text character-by-character (token-like reveal).
   * History / welcome should use appendBubble (instant).
   */
  function streamAssistantBubble(text) {
    const full = String(text ?? "");
    const thread = $("#chat-thread");
    if (!thread) return Promise.resolve(null);
    if (!full || prefersReducedMotion()) {
      return Promise.resolve(appendBubble("assistant", full));
    }

    const div = document.createElement("div");
    div.className = "bubble assistant streaming";
    const body = document.createElement("span");
    body.className = "stream-text";
    const caret = document.createElement("span");
    caret.className = "stream-caret";
    caret.setAttribute("aria-hidden", "true");
    div.appendChild(body);
    div.appendChild(caret);
    thread.appendChild(div);
    thread.scrollTop = thread.scrollHeight;

    // Adaptive pace: short replies feel natural; long ones finish sooner
    const len = full.length;
    const charsPerTick = len > 900 ? 5 : len > 400 ? 3 : len > 160 ? 2 : 1;
    const delayMs = len > 900 ? 8 : len > 400 ? 12 : 16;

    return new Promise((resolve) => {
      let i = 0;
      const step = () => {
        i = Math.min(len, i + charsPerTick);
        body.textContent = full.slice(0, i);
        thread.scrollTop = thread.scrollHeight;
        if (i >= len) {
          div.classList.remove("streaming");
          caret.remove();
          resolve(div);
          return;
        }
        window.setTimeout(step, delayMs);
      };
      step();
    });
  }

  function appendToolResult(payload) {
    const thread = $("#chat-thread");
    let results = payload.tool_results;
    if (!results && payload.tools) {
      const tools = payload.tools;
      if (Array.isArray(tools)) results = tools;
      else if (Array.isArray(tools.tool_calls)) results = tools.tool_calls;
      else results = [tools];
    }
    const list = Array.isArray(results) ? results : results ? [results] : [];
    const imgs = [];
    list.forEach((tr) => {
      if (!tr) return;
      const res = tr.result || tr;
      const img = overlaySrc(res) || overlaySrc(tr) || overlaySrc(payload);
      if (img) imgs.push(img);
    });
    const topImg = overlaySrc(payload);
    if (topImg && !imgs.includes(topImg)) imgs.push(topImg);
    if (!imgs.length) return;
    // Replace prior chart images in this thread so replots don't stack conflicting overlays.
    thread.querySelectorAll(".tool-block.clean, img.overlay-img").forEach((el) => el.remove());
    imgs.forEach((img) => {
      const block = document.createElement("div");
      block.className = "tool-block clean";
      block.innerHTML = `<img class="overlay-img" src="${escapeHtml(img)}" alt="${escapeHtml(t("chartOverlayAlt"))}" />`;
      thread.appendChild(block);
    });
    thread.scrollTop = thread.scrollHeight;
  }

  /**
   * POST /api/chat/stream — parse SSE token/result events.
   * Throws if the endpoint is unavailable so caller can fall back to /api/chat.
   */
  async function chatViaStream(body, { signal, onToken } = {}) {
    let res;
    try {
      res = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: {
          Accept: "text/event-stream",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
        signal,
      });
    } catch (err) {
      if (err && err.name === "AbortError") throw err;
      const e = new Error(t("serverUnreachable"));
      e.cause = err;
      e.streamUnavailable = true;
      throw e;
    }
    if (!res.ok || !res.body) {
      const e = new Error(`stream_unavailable (${res.status})`);
      e.status = res.status;
      e.streamUnavailable = true;
      throw e;
    }
    const ctype = (res.headers.get("content-type") || "").toLowerCase();
    if (ctype.includes("application/json") && !ctype.includes("text/event-stream")) {
      const e = new Error("stream_unavailable");
      e.streamUnavailable = true;
      throw e;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let resultPayload = null;

    const dispatchBlock = (block) => {
      let ev = "message";
      const dataLines = [];
      block.split(/\r?\n/).forEach((line) => {
        if (line.startsWith("event:")) ev = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
      });
      if (!dataLines.length && ev === "message") return;
      const raw = dataLines.join("\n");
      let data = raw;
      try {
        data = JSON.parse(raw);
      } catch {
        /* keep raw string */
      }
      if (ev === "token") {
        const text = typeof data === "string" ? data : data && data.text;
        if (text && onToken) onToken(text);
      } else if (ev === "result") {
        resultPayload = data;
      } else if (ev === "error") {
        const msg =
          (data && (data.error || data.detail || data.message)) || "stream error";
        throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buffer.search(/\r?\n\r?\n/)) !== -1) {
        const match = buffer.match(/\r?\n\r?\n/);
        const block = buffer.slice(0, sep);
        buffer = buffer.slice(sep + (match ? match[0].length : 2));
        if (block.trim()) dispatchBlock(block);
      }
    }
    if (buffer.trim()) dispatchBlock(buffer);

    if (!resultPayload) {
      const e = new Error("stream_incomplete");
      e.streamUnavailable = true;
      throw e;
    }
    return resultPayload;
  }

  async function chatRequest(body, { signal, onToken } = {}) {
    try {
      return await chatViaStream(body, { signal, onToken });
    } catch (err) {
      if (err && err.name === "AbortError") throw err;
      if (!err.streamUnavailable) throw err;
      const data = await api("/chat", { method: "POST", body, signal });
      const reply = data.reply || data.message || data.response || data.answer || "";
      if (reply && onToken) onToken(reply);
      return data;
    }
  }

  function createStreamingAssistantBubble() {
    const thread = $("#chat-thread");
    const div = document.createElement("div");
    div.className = "bubble assistant streaming";
    const body = document.createElement("span");
    body.className = "stream-text";
    const caret = document.createElement("span");
    caret.className = "stream-caret";
    caret.setAttribute("aria-hidden", "true");
    div.appendChild(body);
    div.appendChild(caret);
    thread.appendChild(div);
    thread.scrollTop = thread.scrollHeight;
    const reducedMotion = prefersReducedMotion();
    let queue = "";
    let timer = null;
    let finishing = false;

    const cleanup = (finalText) => {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      if (finalText != null && finalText !== "") body.textContent = finalText;
      div.classList.remove("streaming");
      caret.remove();
      if (!body.textContent) {
        div.remove();
        return null;
      }
      return div;
    };

    const schedule = () => {
      if (timer) return;
      timer = window.setTimeout(drain, 14);
    };

    const drain = () => {
      timer = null;
      if (!queue.length) {
        if (finishing) cleanup();
        return;
      }
      const qLen = queue.length;
      const charsPerTick = qLen > 40 ? 3 : qLen > 16 ? 2 : 1;
      body.textContent += queue.slice(0, charsPerTick);
      queue = queue.slice(charsPerTick);
      thread.scrollTop = thread.scrollHeight;
      if (queue.length) {
        schedule();
      } else if (finishing) {
        cleanup();
      }
    };

    return {
      el: div,
      append(text) {
        if (!text) return;
        const chunk = String(text);
        if (reducedMotion) {
          body.textContent += chunk;
          thread.scrollTop = thread.scrollHeight;
          return;
        }
        queue += chunk;
        schedule();
      },
      finish(finalText) {
        if (finalText != null && finalText !== "") return cleanup(finalText);
        if (reducedMotion) return cleanup();
        if (!queue.length) return cleanup();
        finishing = true;
        if (!timer) {
          schedule();
        }
        return div;
      },
    };
  }

  function wireChat() {
    const form = $("#chat-form");
    const input = $("#chat-input");
    const attachBtn = $("#chat-attach");
    const cancelBtn = $("#chat-cancel");
    let chatAbort = null;

    function setChatBusy(busy) {
      const btn = $("#chat-send");
      setLoading(btn, busy);
      if (cancelBtn) cancelBtn.hidden = !busy;
      if (attachBtn) attachBtn.disabled = busy;
      if (input) input.disabled = busy;
    }

    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 120) + "px";
    });
    if (cancelBtn && !cancelBtn.dataset.wired) {
      cancelBtn.dataset.wired = "1";
      cancelBtn.addEventListener("click", () => {
        if (chatAbort) chatAbort.abort();
      });
    }
    if (attachBtn && !attachBtn.dataset.wired) {
      attachBtn.dataset.wired = "1";
      attachBtn.addEventListener("click", () => {
        toast(t("attachComingSoon"), "info");
      });
    }
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      appendBubble("user", message);
      input.value = "";
      input.style.height = "auto";
      if (chatAbort) chatAbort.abort();
      chatAbort = new AbortController();
      const { signal } = chatAbort;
      setChatBusy(true);
      const scene = appendBubble("tool-scene", t("thinking"));
      let streamBubble = null;
      const applyChatResult = async (data, streamed) => {
        if (scene.isConnected) scene.remove();
        if (data.session_id) {
          state.chatSessionId = data.session_id;
          localStorage.setItem(STORAGE_SESSION, data.session_id);
        }
        const reply =
          data.reply || data.message || data.response || data.answer || t("done");
        if (streamBubble) {
          streamBubble.finish(streamed ? undefined : reply);
          streamBubble = null;
        } else if (!streamed) {
          await streamAssistantBubble(reply);
        }
        const hasOverlay =
          overlaySrc(data) ||
          (Array.isArray(data.tool_results) &&
            data.tool_results.some((tr) => overlaySrc(tr) || overlaySrc(tr && tr.result)));
        if (hasOverlay || data.tool_results) {
          appendToolResult(data);
        }
      };
      try {
        await ensureChatSession();
        const body = {
          message,
          session_id: state.chatSessionId || undefined,
          child_id:
            (state.activeChild && (state.activeChild.child_id || state.activeChild.id)) || undefined,
          ui_lang: state.lang,
        };
        let gotToken = false;
        const data = await chatRequest(body, {
          signal,
          onToken: (text) => {
            if (!gotToken) {
              if (scene.isConnected) scene.remove();
              streamBubble = createStreamingAssistantBubble();
              gotToken = true;
            }
            if (streamBubble) streamBubble.append(text);
          },
        });
        await applyChatResult(data, gotToken);
      } catch (err) {
        if (scene.isConnected) scene.remove();
        if (streamBubble) {
          streamBubble.finish();
          streamBubble = null;
        }
        if (err && err.name === "AbortError") {
          appendBubble("system", t("chatCancelled"));
          return;
        }
        if (String(err.message || "").includes("session_not_found") || err.status === 404) {
          state.chatSessionId = null;
          localStorage.removeItem(STORAGE_SESSION);
          try {
            await ensureChatSession();
            let gotToken = false;
            const data = await chatRequest(
              {
                message,
                session_id: state.chatSessionId,
                child_id:
                  (state.activeChild && (state.activeChild.child_id || state.activeChild.id)) ||
                  undefined,
                ui_lang: state.lang,
              },
              {
                signal,
                onToken: (text) => {
                  if (!gotToken) {
                    streamBubble = createStreamingAssistantBubble();
                    gotToken = true;
                  }
                  if (streamBubble) streamBubble.append(text);
                },
              }
            );
            await applyChatResult(data, gotToken);
            return;
          } catch (err2) {
            if (err2 && err2.name === "AbortError") {
              appendBubble("system", t("chatCancelled"));
              return;
            }
            appendBubble("system", err2.message);
            toast(err2.message, "error");
            return;
          }
        }
        appendBubble("system", err.message);
        toast(err.message, "error");
      } finally {
        setChatBusy(false);
        input.focus();
      }
    });
  }

  /* —— Growth —— */
  function syncGrowthSexFromChild() {
    const sel = $("#growth-child");
    const id = sel && sel.value;
    if (!id) return;
    const child = state.children.find((c) => (c.child_id || c.id) === id);
    if (child && child.sex) {
      const radio = $(`#growth-form input[name="sex"][value="${child.sex}"]`);
      if (radio) radio.checked = true;
    }
  }

  async function fetchGrowthCurves(params) {
    const q = new URLSearchParams();
    if (params.sex) q.set("sex", params.sex);
    if (params.measure) q.set("measure", params.measure);
    if (params.chart_standard) q.set("chart_standard", params.chart_standard);
    if (params.gestational_age_weeks != null && params.gestational_age_weeks !== "") {
      q.set("gestational_age_weeks", String(params.gestational_age_weeks));
    }
    if (params.age_max != null) q.set("age_max", String(params.age_max));
    try {
      const data = await api(`/growth/curves?${q.toString()}`);
      if (data && data.ok !== false && data.ages && data.curves) return data;
    } catch (_) {
      /* endpoint missing or failed — PNG overlay fallback */
    }
    return null;
  }

  function renderGrowthSvg(curvesData, childPoint) {
    const ages = curvesData.ages || [];
    const curves = curvesData.curves || {};
    const pctKeys = (curvesData.percentiles || Object.keys(curves))
      .map(String)
      .filter((p) => Array.isArray(curves[p]) && curves[p].length);
    if (!ages.length || !pctKeys.length) return "";

    const pad = { top: 28, right: 16, bottom: 36, left: 44 };
    const W = 640;
    const H = 360;
    const innerW = W - pad.left - pad.right;
    const innerH = H - pad.top - pad.bottom;

    let yMin = Infinity;
    let yMax = -Infinity;
    pctKeys.forEach((p) => {
      curves[p].forEach((v) => {
        if (Number.isFinite(v)) {
          yMin = Math.min(yMin, v);
          yMax = Math.max(yMax, v);
        }
      });
    });
    if (childPoint && Number.isFinite(childPoint.y)) {
      yMin = Math.min(yMin, childPoint.y);
      yMax = Math.max(yMax, childPoint.y);
    }
    if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMax <= yMin) {
      yMin = 0;
      yMax = 1;
    }
    const yPad = (yMax - yMin) * 0.08 || 0.1;
    yMin -= yPad;
    yMax += yPad;

    const x0 = ages[0];
    const x1 = ages[ages.length - 1];
    const xSpan = x1 - x0 || 1;
    const sx = (x) => pad.left + ((x - x0) / xSpan) * innerW;
    const sy = (y) => pad.top + ((yMax - y) / (yMax - yMin)) * innerH;

    const polylines = pctKeys
      .map((p) => {
        const pts = ages
          .map((age, i) => {
            const y = curves[p][i];
            return Number.isFinite(y) ? `${sx(age).toFixed(1)},${sy(y).toFixed(1)}` : null;
          })
          .filter(Boolean)
          .join(" ");
        return `<polyline class="curve-p${escapeHtml(p)}" points="${pts}" />`;
      })
      .join("");

    let childMark = "";
    if (childPoint && Number.isFinite(childPoint.x) && Number.isFinite(childPoint.y)) {
      childMark = `<circle class="child-point" cx="${sx(childPoint.x).toFixed(1)}" cy="${sy(
        childPoint.y
      ).toFixed(1)}" r="6" />`;
    }

    const xLabel = curvesData.age_unit === "weeks" ? t("ageWeeks") : t("ageMonths");
    const yLabel = curvesData.units || t("value");
    const title = escapeHtml(t("percentileChart"));

    return `<div class="growth-svg-wrap" role="img" aria-label="${title}">
      <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <text x="${pad.left}" y="18" fill="#3a5560" font-size="13" font-weight="700">${title}</text>
        <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${H - pad.bottom}" stroke="#c5d5dc" />
        <line x1="${pad.left}" y1="${H - pad.bottom}" x2="${W - pad.right}" y2="${H - pad.bottom}" stroke="#c5d5dc" />
        ${polylines}
        ${childMark}
        <text x="${W / 2}" y="${H - 8}" text-anchor="middle" fill="#5a7380" font-size="11">${escapeHtml(
          xLabel
        )}</text>
        <text x="12" y="${H / 2}" text-anchor="middle" fill="#5a7380" font-size="11"
          transform="rotate(-90 12 ${H / 2})">${escapeHtml(yLabel)}</text>
      </svg>
    </div>`;
  }

  function wireGrowth() {
    $("#growth-child").addEventListener("change", (e) => {
      const id = e.target.value;
      if (id) {
        const child = state.children.find((c) => (c.child_id || c.id) === id);
        if (child) saveActiveChild(child);
      }
      syncGrowthSexFromChild();
    });

    $("#growth-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const form = e.target;
      const btn = $("#growth-submit");
      const fd = new FormData(form);
      const body = {
        child_id: fd.get("child_id") || undefined,
        sex: fd.get("sex"),
        measure: fd.get("measure"),
        value: Number(fd.get("value")),
      };
      const weeksRaw = fd.get("weeks");
      const monthsRaw = fd.get("age_months");
      if (weeksRaw !== "" && weeksRaw != null) body.weeks = Number(weeksRaw);
      if (monthsRaw !== "" && monthsRaw != null) body.age_months = Number(monthsRaw);
      if (body.child_id) {
        const child = state.children.find((c) => (c.child_id || c.id) === body.child_id);
        if (child && child.gestational_age_weeks != null) {
          body.gestational_age_weeks = Number(child.gestational_age_weeks);
        }
      }
      if (!body.child_id) delete body.child_id;
      if (body.weeks == null && body.age_months == null) {
        toast(t("ageWeeksHint"), "error");
        return;
      }
      setLoading(btn, true);
      const out = $("#growth-result");
      out.hidden = true;
      try {
        const data = await api("/growth", { method: "POST", body });
        const centile = data.centile != null ? Number(data.centile).toFixed(1) : "—";
        const z = data.z_score != null ? Number(data.z_score).toFixed(2) : "—";
        const track = data.track_status || data.status || "";
        const badgeClass =
          /below_3|above_97|investigate|alert|high/i.test(track)
            ? "alert"
            : /outer|monitor|warn/i.test(track)
              ? "warn"
              : "";
        const img = overlaySrc(data);

        let svgHtml = "";
        const curves = await fetchGrowthCurves({
          sex: data.sex || body.sex,
          measure: data.measure || body.measure,
          chart_standard: data.chart_standard,
          gestational_age_weeks:
            data.gestational_age_weeks != null
              ? data.gestational_age_weeks
              : body.gestational_age_weeks,
          age_max:
            data.chart_standard === "who_term" || data.age_months != null
              ? Math.max(24, Number(data.age_months || body.age_months || 12) + 2)
              : Math.max(64, Number(data.weeks || body.weeks || 40) + 4),
        });
        if (curves) {
          const childX =
            curves.age_unit === "months"
              ? Number(data.age_months != null ? data.age_months : body.age_months)
              : Number(data.weeks != null ? data.weeks : body.weeks);
          const childY = Number(data.value != null ? data.value : body.value);
          svgHtml = renderGrowthSvg(curves, { x: childX, y: childY });
        }

        out.innerHTML = `
          <h3>${escapeHtml(t("growthResult"))}</h3>
          ${track ? `<span class="badge ${badgeClass}">${escapeHtml(track.replace(/_/g, " "))}</span>` : ""}
          <div class="stat-row">
            <div class="stat"><span class="val">${escapeHtml(centile)}</span><span class="lbl">${escapeHtml(
              t("centile")
            )}</span></div>
            <div class="stat"><span class="val">${escapeHtml(z)}</span><span class="lbl">${escapeHtml(
              t("zScore")
            )}</span></div>
            <div class="stat"><span class="val">${escapeHtml(
              String(data.value ?? body.value)
            )}</span><span class="lbl">${escapeHtml(
              data.units?.[data.measure] || data.units || data.measure || t("value")
            )}</span></div>
          </div>
          ${data.summary ? `<p>${escapeHtml(data.summary)}</p>` : ""}
          ${
            svgHtml ||
            (img
              ? `<img class="overlay-img" src="${escapeHtml(img)}" alt="${escapeHtml(
                  t("growthChartAlt")
                )}" />`
              : "")
          }
          ${!svgHtml && !img ? `<p class="muted">${escapeHtml(t("curveLoadFailed"))}</p>` : ""}
        `;
        out.hidden = false;
        out.scrollIntoView({ behavior: "smooth", block: "nearest" });
      } catch (err) {
        toast(err.message, "error");
      } finally {
        setLoading(btn, false);
      }
    });
  }

  /* —— Screening —— */
  function showScreeningPicker() {
    state.quiz = null;
    $("#screening-picker").hidden = false;
    $("#screening-quiz").hidden = true;
    $("#screening-report").hidden = true;
    $("#screening-title").textContent = t("screeningTitle");
    $("#screening-sub").textContent = t("screeningSub");
    $("#screening-back").href = "#/";
    $("#screening-back").textContent = t("home");
  }

  function buildAsqAgeChips() {
    const grid = $("#asq-ages");
    grid.innerHTML = ASQ_AGES.map(
      (m) => `<button type="button" class="chip" data-age="${m}">${m}m</button>`
    ).join("");
    grid.addEventListener("click", (e) => {
      const btn = e.target.closest(".chip");
      if (!btn) return;
      startAsq(Number(btn.dataset.age));
    });
  }

  function normalizeQuestions(payload, kind) {
    if (kind === "asq") {
      const domains = payload.domains || payload;
      const items = [];
      (Array.isArray(domains) ? domains : []).forEach((dom) => {
        if (dom.id === "overall") return;
        (dom.questions || []).forEach((q) => {
          items.push({
            domain: dom.id || dom.title_en || "domain",
            domainTitle: dom.title_en || dom.id || "",
            id: q.id,
            text: (state.lang === "fa" && (q.text_fa || q.text)) || q.text_en || q.text || q.question,
            options: [
              { value: "yes", label: t("yes"), cls: "yes" },
              { value: "sometimes", label: t("sometimes"), cls: "sometimes" },
              { value: "not_yet", label: t("notYet"), cls: "not_yet" },
            ],
          });
        });
      });
      return items;
    }
    const qs = payload.questions || payload;
    return (Array.isArray(qs) ? qs : []).map((q) => ({
      domain: "M-CHAT-R",
      domainTitle: "M-CHAT-R",
      id: q.id,
      text: (state.lang === "fa" && (q.text_fa || q.text)) || q.text_en || q.text || q.question,
      options: [
        { value: "yes", label: t("yes"), cls: "yes" },
        { value: "no", label: t("no"), cls: "no" },
      ],
    }));
  }

  async function startAsq(age) {
    try {
      toast(t("loadingAsq", { age }));
      const data = await api(`/asq/${age}/questions`);
      const items = normalizeQuestions(data, "asq");
      if (!items.length) throw new Error(t("noAsqQuestions"));
      state.quiz = {
        kind: "asq",
        age,
        items,
        index: 0,
        answers: {},
      };
      $("#screening-title").textContent = t("asqMonths", { age });
      $("#screening-sub").textContent = t("asqAnswerHint");
      beginQuiz();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  async function startMchat() {
    try {
      toast(t("loadingMchat"));
      const data = await api("/mchat/questions");
      const items = normalizeQuestions(data, "mchat");
      if (!items.length) throw new Error(t("noMchatQuestions"));
      state.quiz = {
        kind: "mchat",
        items,
        index: 0,
        answers: {},
      };
      $("#screening-title").textContent = t("mchatTitle");
      $("#screening-sub").textContent = t("mchatAnswerHint");
      beginQuiz();
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function beginQuiz() {
    $("#screening-picker").hidden = true;
    $("#screening-report").hidden = true;
    $("#screening-quiz").hidden = false;
    const back = $("#screening-back");
    back.href = "#/screening";
    back.textContent = t("questionnaires");
    renderQuizStep();
  }

  function renderQuizStep() {
    const q = state.quiz;
    if (!q) return;
    const item = q.items[q.index];
    const total = q.items.length;
    const pct = (q.index / total) * 100;
    $("#quiz-progress").style.width = `${pct}%`;
    $("#quiz-meta").textContent = t("questionOf", { n: q.index + 1, total });
    $("#quiz-domain").textContent = item.domainTitle || item.domain;
    $("#quiz-question").textContent = item.text;
    const row = $("#quiz-answers");
    const key = answerKey(item);
    const current = q.answers[key];
    row.innerHTML = item.options
      .map(
        (o) =>
          `<button type="button" class="ans-btn ${o.cls}${current === o.value ? " selected" : ""}" data-value="${o.value}">${escapeHtml(o.label)}</button>`
      )
      .join("");
    $$(".ans-btn", row).forEach((btn) => {
      btn.addEventListener("click", () => {
        q.answers[key] = btn.dataset.value;
        $$(".ans-btn", row).forEach((b) => b.classList.toggle("selected", b === btn));
        $("#quiz-next").disabled = false;
        setTimeout(() => {
          if (q.index < q.items.length - 1 && q.answers[key]) {
            q.index += 1;
            renderQuizStep();
          } else if (q.index === q.items.length - 1) {
            $("#quiz-next").textContent = t("seeResults");
          }
        }, 220);
      });
    });
    $("#quiz-prev").disabled = q.index === 0;
    $("#quiz-next").disabled = !current;
    $("#quiz-next").textContent = q.index === q.items.length - 1 ? t("seeResults") : t("next");
  }

  function answerKey(item) {
    return `${item.domain}::${item.id}`;
  }

  function wireQuizNav() {
    $("#screening-back").addEventListener("click", (e) => {
      if (state.quiz && !$("#screening-quiz").hidden) {
        e.preventDefault();
        showScreeningPicker();
      }
    });
    $("#quiz-prev").addEventListener("click", () => {
      if (!state.quiz || state.quiz.index <= 0) return;
      state.quiz.index -= 1;
      renderQuizStep();
    });
    $("#quiz-next").addEventListener("click", async () => {
      const q = state.quiz;
      if (!q) return;
      const item = q.items[q.index];
      if (!q.answers[answerKey(item)]) return;
      if (q.index < q.items.length - 1) {
        q.index += 1;
        renderQuizStep();
        return;
      }
      await submitQuiz();
    });
    $("#start-mchat").addEventListener("click", startMchat);
    $("#screening-child").addEventListener("change", (e) => {
      const id = e.target.value;
      if (!id) return;
      const child = state.children.find((c) => (c.child_id || c.id) === id);
      if (child) saveActiveChild(child);
    });
  }

  async function submitQuiz() {
    const q = state.quiz;
    const btn = $("#quiz-next");
    setLoading(btn, true);
    try {
      const childId =
        $("#screening-child").value ||
        (state.activeChild && (state.activeChild.child_id || state.activeChild.id)) ||
        undefined;

      let data;
      if (q.kind === "asq") {
        const domain_answers = {};
        q.items.forEach((item) => {
          const ans = q.answers[answerKey(item)];
          if (!ans) return;
          if (!domain_answers[item.domain]) domain_answers[item.domain] = [];
          domain_answers[item.domain].push(ans);
        });
        data = await api("/asq/score", {
          method: "POST",
          body: {
            child_id: childId,
            age_months: q.age,
            domain_answers,
          },
        });
        renderAsqReport(data);
      } else {
        const answers = {};
        q.items.forEach((item) => {
          answers[item.id] = q.answers[answerKey(item)];
        });
        data = await api("/mchat/score", {
          method: "POST",
          body: { child_id: childId, answers },
        });
        renderMchatReport(data);
      }
      $("#screening-quiz").hidden = true;
      $("#quiz-progress").style.width = "100%";
    } catch (err) {
      toast(err.message, "error");
    } finally {
      setLoading(btn, false);
    }
  }

  function renderAsqReport(data) {
    const report = $("#screening-report");
    const result = data.result || data;
    const domains = result.domains || data.domains || {};
    const needs = result.needs_referral ?? data.needs_referral;
    const summary = data.parent_report || result.summary || data.summary || "";
    const cutoffSource = result.cutoff_source || data.cutoff_source || "";
    const badge = needs ? "alert" : "";
    const rows = Object.entries(domains)
      .map(([id, d]) => {
        const below = d.below_cutoff;
        return `<li><span>${escapeHtml(id.replace(/_/g, " "))}</span>
          <span class="badge ${below ? "alert" : ""}">${escapeHtml(String(d.total ?? "—"))}${
          d.cutoff != null ? ` / cut ${d.cutoff}` : ""
        }</span></li>`;
      })
      .join("");
    const cutoffNote =
      cutoffSource && cutoffSource !== "official_asq3"
        ? `<p class="muted"><small>${escapeHtml(
            t("cutoffSourceNote", { source: cutoffSource })
          )}</small></p>`
        : "";
    report.innerHTML = `
      <h3>${escapeHtml(t("asqReport"))}</h3>
      <span class="badge ${badge}">${escapeHtml(
        needs ? t("referralSuggested") : t("noDomainBelow")
      )}</span>
      <p>${escapeHtml(summary)}</p>
      ${cutoffNote}
      <ul class="domain-list">${rows}</ul>
      <button type="button" class="btn btn-secondary btn-block" id="screening-again" style="margin-top:1rem">${escapeHtml(
        t("anotherQuestionnaire")
      )}</button>
    `;
    report.hidden = false;
    $("#screening-again").addEventListener("click", showScreeningPicker);
  }

  function renderMchatReport(data) {
    const report = $("#screening-report");
    const result = data.result || data;
    const risk = String(result.risk || data.risk || "").toLowerCase();
    const badge = risk === "high" ? "alert" : risk === "medium" ? "warn" : "";
    const summary = data.parent_report || result.summary || data.summary || "";
    const note = result.note || data.note || "";
    report.innerHTML = `
      <h3>${escapeHtml(t("mchatReport"))}</h3>
      <span class="badge ${badge}">${escapeHtml(t("riskLabel", { risk: risk || "—" }))}</span>
      <div class="stat-row">
        <div class="stat"><span class="val">${escapeHtml(
          String(result.total_failed ?? data.total_failed ?? "—")
        )}</span><span class="lbl">${escapeHtml(t("failedItems"))}</span></div>
      </div>
      <p>${escapeHtml(summary)}</p>
      ${note ? `<p class="muted">${escapeHtml(note)}</p>` : ""}
      <button type="button" class="btn btn-secondary btn-block" id="screening-again" style="margin-top:1rem">${escapeHtml(
        t("anotherQuestionnaire")
      )}</button>
    `;
    report.hidden = false;
    $("#screening-again").addEventListener("click", showScreeningPicker);
  }

  /* —— Boot —— */
  async function boot() {
    state.activeChild = normalizeChild(loadJson(STORAGE_CHILD, null));
    applyI18n();
    const langBtn = $("#lang-toggle");
    if (langBtn) {
      langBtn.addEventListener("click", () => setLang(state.lang === "fa" ? "en" : "fa"));
    }
    renderChildChip();
    buildAsqAgeChips();
    wireChildForm();
    wireChat();
    wireGrowth();
    wireQuizNav();
    window.addEventListener("hashchange", navigate);
    if (!location.hash) location.hash = "#/";
    navigate();
    checkHealth();
    try {
      const data = await api("/children");
      state.children = Array.isArray(data) ? data : data.children || [];
      fillChildSelects();
    } catch {
      /* offline ok */
    }
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
