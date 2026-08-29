/**
 * Nestling — parental SPA
 * Runtime values (API base, endpoint paths, timings, domain limits) come from
 * config.js; every user-facing string comes from i18n.js. Keep literals out of
 * this module so the app runs unchanged on any host, port or proxy prefix.
 */
(function () {
  "use strict";

  const CFG = window.NESTLING_CONFIG;
  if (!CFG) {
    document.addEventListener("DOMContentLoaded", () => {
      const main = document.getElementById("main");
      if (main) main.textContent = "Nestling could not load its configuration (config.js).";
    });
    return;
  }

  const API_BASE = CFG.api.base;
  const EP = CFG.api.paths;
  const TIME = CFG.timing;
  const LIMITS = CFG.limits;
  const REVEAL = CFG.reveal;
  const CHART = CFG.chart;
  const STORAGE = CFG.storageKeys;
  const ASQ_AGES = LIMITS.asqAges;

  const API_ORIGIN = (() => {
    try {
      return new URL(API_BASE, window.location.href).origin;
    } catch (_) {
      return window.location.origin;
    }
  })();

  /** localStorage throws in private mode / blocked-cookie contexts. */
  const store = {
    get(key) {
      try {
        return localStorage.getItem(key);
      } catch (_) {
        return null;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem(key, value);
      } catch (_) {
        /* storage unavailable — session stays in memory */
      }
    },
    remove(key) {
      try {
        localStorage.removeItem(key);
      } catch (_) {
        /* ignore */
      }
    },
  };

  const state = {
    children: [],
    activeChild: null,
    chatSessionId: store.get(STORAGE.chatSession) || null,
    quiz: null,
    lang: normalizeLang(store.get(STORAGE.lang)),
    screeningAgeMonths: null,
    screeningHistoryOpen: false,
    chatHistoryOpen: false,
    lastDossier: null,
    lastGrowth: null,
    lastReport: null,
    chatAbort: null,
    childrenToken: 0,
    dossierToken: 0,
    quizAdvanceTimer: null,
    ageInputTimer: null,
  };

  /** Active text-reveal animations, so clearing the thread can stop them. */
  const reveals = new Set();

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  function normalizeLang(lang) {
    return lang === "fa" ? "fa" : "en";
  }

  function i18nPack() {
    const packs = window.NESTLING_I18N || {};
    return packs[state.lang] || packs.en || {};
  }

  function t(key, vars) {
    const packs = window.NESTLING_I18N || {};
    const pack = packs[state.lang] || {};
    let s = pack[key] != null ? pack[key] : (packs.en && packs.en[key]) || key;
    if (vars && typeof s === "string") {
      Object.keys(vars).forEach((k) => {
        s = s.replace(new RegExp(`\\{${k}\\}`, "g"), String(vars[k]));
      });
    }
    return s;
  }

  /** Locale-aware digits (Persian numerals in fa) for display only. */
  function fmtNum(value, digits) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    const max = digits != null ? digits : Number.isInteger(n) ? 0 : 1;
    try {
      return new Intl.NumberFormat(i18nPack().numberLocale || undefined, {
        minimumFractionDigits: digits != null ? digits : 0,
        maximumFractionDigits: max,
      }).format(n);
    } catch (_) {
      return String(n);
    }
  }

  /**
   * Set translated text and remember the key (plus interpolation vars) so a
   * later language switch re-renders content that JS wrote.
   */
  function setI18nText(el, key, vars) {
    if (!el) return;
    el.setAttribute("data-i18n", key);
    if (vars) el.dataset.i18nVars = JSON.stringify(vars);
    else delete el.dataset.i18nVars;
    el.textContent = t(key, vars);
  }

  function applyI18n() {
    const pack = i18nPack();
    document.documentElement.lang = pack.lang || state.lang;
    document.documentElement.dir = pack.dir || "ltr";
    document.title = pack.brand || t("brand");
    $$("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (!key) return;
      let vars = null;
      if (el.dataset.i18nVars) {
        try {
          vars = JSON.parse(el.dataset.i18nVars);
        } catch (_) {
          vars = null;
        }
      }
      if (vars) el.textContent = t(key, vars);
      else if (pack[key] != null) el.textContent = pack[key];
    });
    $$("[data-i18n-aria-label]").forEach((el) => {
      const key = el.getAttribute("data-i18n-aria-label");
      if (key && pack[key] != null) el.setAttribute("aria-label", pack[key]);
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
    if (toggle && pack.langToggle) toggle.textContent = pack.langToggle;
  }

  /** Re-render everything JS produced, so switching language never leaves mixed text. */
  function setLang(lang) {
    state.lang = normalizeLang(lang);
    store.set(STORAGE.lang, state.lang);
    applyI18n();
    applyFieldConstraints();
    renderChildChip();

    const thread = $("#chat-thread");
    if (thread && thread.dataset.ready) {
      const first = thread.querySelector(".bubble.assistant");
      if (first && thread.querySelectorAll(".bubble.user").length === 0) {
        first.textContent = t("chatWelcome");
      }
    }
    if (state.chatHistoryOpen) loadChatHistoryList();
    if (currentPath() === "/child") loadChildren();
    else if (state.activeChild) loadChildDossier(activeChildId());
    if (state.lastGrowth) renderGrowthResult(state.lastGrowth);
    if (state.lastReport) renderScreeningReport(state.lastReport);
    if (state.quiz) relabelQuizForLang();
    else if (currentPath() === "/screening") refreshScreeningPicker();
  }

  /* —— Utils —— */
  function loadJson(key, fallback) {
    try {
      const raw = store.get(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (_) {
      return fallback;
    }
  }

  function ageMonthsFromDob(dobStr) {
    if (!dobStr) return null;
    const dob = new Date(String(dobStr).slice(0, 10) + "T00:00:00");
    if (Number.isNaN(dob.getTime())) return null;
    const now = new Date();
    if (dob > now) return 0;
    let months =
      (now.getFullYear() - dob.getFullYear()) * 12 + (now.getMonth() - dob.getMonth());
    const dayFrac = (now.getDate() - dob.getDate()) / LIMITS.daysPerMonth;
    months += dayFrac;
    return Math.max(0, Math.round(months * 10) / 10);
  }

  function correctedAgeMonths(chronoMonths, gaWeeks) {
    if (chronoMonths == null || gaWeeks == null) return chronoMonths;
    const ga = Number(gaWeeks);
    if (!(ga < LIMITS.pretermWeeks)) return chronoMonths;
    const earlyWeeks = Math.max(0, LIMITS.fullTermWeeks - ga);
    return Math.max(
      0,
      Math.round((Number(chronoMonths) - earlyWeeks / LIMITS.weeksPerMonth) * 10) / 10
    );
  }

  function isPreterm(gaWeeks) {
    return gaWeeks != null && Number(gaWeeks) < LIMITS.pretermWeeks;
  }

  function maturityLabel(gaWeeks) {
    if (gaWeeks == null || Number.isNaN(Number(gaWeeks))) return "";
    return isPreterm(gaWeeks) ? t("preterm") : t("term");
  }

  function formatAgeLabel(months) {
    if (months == null || Number.isNaN(Number(months))) return "—";
    return t("ageMonthsValue", { age: fmtNum(months) });
  }

  const MEASURE_KEYS = { weight: "mWeight", length: "mLength", head_circumference: "mHc" };

  function measureLabel(measure) {
    const key = MEASURE_KEYS[String(measure || "")];
    return key ? t(key) : String(measure || "");
  }

  /** "gross_motor" -> domainGrossMotor, falling back to a readable id. */
  function keyFromId(prefix, id) {
    const parts = String(id || "")
      .split(/[^A-Za-z0-9]+/)
      .filter(Boolean)
      .map((p) => p.charAt(0).toUpperCase() + p.slice(1));
    return prefix + parts.join("");
  }

  function domainLabel(id) {
    const key = keyFromId("domain", id);
    const pack = i18nPack();
    if (pack[key] != null) return pack[key];
    return String(id || "").replace(/_/g, " ");
  }

  function trackLabel(track) {
    const raw = String(track || "");
    if (!raw) return "";
    const key = keyFromId("track", raw);
    const pack = i18nPack();
    if (pack[key] != null) return pack[key];
    return raw.replace(/_/g, " ");
  }

  function buildAsqWindows() {
    const ages = ASQ_AGES;
    const map = {};
    ages.forEach((age, i) => {
      const prev = i === 0 ? null : ages[i - 1];
      const next = i === ages.length - 1 ? null : ages[i + 1];
      const lo = prev == null ? Math.max(0, age - (next - age) / 2) : (prev + age) / 2;
      const hi = next == null ? age + (age - prev) / 2 : (age + next) / 2;
      map[age] = { lo, hi };
    });
    return map;
  }

  const ASQ_WINDOWS = buildAsqWindows();

  function relevantAsqAges(ageMonths) {
    if (ageMonths == null || Number.isNaN(Number(ageMonths))) return { current: [], upcoming: [] };
    const age = Number(ageMonths);
    const windows = ASQ_WINDOWS;
    const current = ASQ_AGES.filter((a) => age >= windows[a].lo && age < windows[a].hi);
    if (current.length) {
      const lastCurrent = current[current.length - 1];
      const nextIdx = ASQ_AGES.indexOf(lastCurrent) + 1;
      const upcoming = nextIdx < ASQ_AGES.length ? [ASQ_AGES[nextIdx]] : [];
      return { current, upcoming };
    }
    const upcoming = ASQ_AGES.filter((a) => a > age).slice(0, 1);
    const recent = [...ASQ_AGES]
      .reverse()
      .find((a) => a <= age && age - a <= LIMITS.asqRecencyMonths);
    return { current: recent != null ? [recent] : [], upcoming };
  }

  function resolveScreeningAge() {
    const input = $("#screening-age-months");
    const typed = input && input.value !== "" ? Number(input.value) : null;
    if (typed != null && !Number.isNaN(typed) && typed >= 0) {
      state.screeningAgeMonths = typed;
      return { chrono: typed, screening: typed, source: "input" };
    }
    const child = state.activeChild;
    if (child && child.date_of_birth) {
      const chrono = ageMonthsFromDob(child.date_of_birth);
      const screening = correctedAgeMonths(chrono, child.gestational_age_weeks);
      state.screeningAgeMonths = screening;
      return { chrono, screening, source: "dob", ga: child.gestational_age_weeks };
    }
    if (state.screeningAgeMonths != null) {
      return { chrono: state.screeningAgeMonths, screening: state.screeningAgeMonths, source: "cached" };
    }
    return { chrono: null, screening: null, source: null };
  }

  function saveActiveChild(child) {
    const normalized = normalizeChild(child);
    state.activeChild = normalized;
    if (normalized) store.set(STORAGE.activeChild, JSON.stringify(normalized));
    else store.remove(STORAGE.activeChild);
    renderChildChip();
  }

  function childIdOf(child) {
    if (!child) return null;
    return child.child_id || child.id || null;
  }

  function activeChildId() {
    return childIdOf(state.activeChild);
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
    if (!chip) return;
    if (state.activeChild && state.activeChild.name) {
      chip.hidden = false;
      const mat = maturityLabel(state.activeChild.gestational_age_weeks);
      const age =
        state.activeChild.date_of_birth != null
          ? ageMonthsFromDob(state.activeChild.date_of_birth)
          : null;
      const ageBit = age != null ? formatAgeLabel(age) : "";
      chip.textContent = [state.activeChild.name, mat, ageBit].filter(Boolean).join(" · ");
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
      state.lastDossier = null;
      return;
    }
    const token = ++state.dossierToken;
    panel.hidden = false;
    panel.setAttribute("aria-busy", "true");
    if (!body.childElementCount) renderMessage(body, t("loading"));
    try {
      const data = await api(EP.childDossier(childId));
      // A newer selection won already — drop this response.
      if (token !== state.dossierToken) return;
      state.lastDossier = data;
      const p = data.profile || {};
      const growth = data.growth || [];
      const screens = data.screenings || [];
      const overlays = data.overlays || [];
      const chrono = ageMonthsFromDob(p.date_of_birth);
      const corr = correctedAgeMonths(chrono, p.gestational_age_weeks);
      const maturity =
        data.maturity === "preterm"
          ? t("preterm")
          : data.maturity === "term"
            ? t("term")
            : "";
      const sexLabel =
        p.sex === "female" ? t("girl") : p.sex === "male" ? t("boy") : p.sex || "";

      const latestByMeasure = {};
      growth.forEach((g) => {
        latestByMeasure[g.measure] = g;
      });
      const growthBits = Object.keys(MEASURE_KEYS)
        .map((m) => latestByMeasure[m])
        .filter(Boolean)
        .slice(0, LIMITS.dossierGrowthRows)
        .map((g) => {
          const cent =
            g.centile != null ? ` · ${t("centileShort", { n: fmtNum(g.centile, 0) })}` : "";
          return `<li><strong>${escapeHtml(measureLabel(g.measure))}</strong> ${escapeHtml(
            fmtNum(g.value)
          )}${escapeHtml(cent)}</li>`;
        })
        .join("");

      const lastScreen = screens.length ? screens[screens.length - 1] : null;
      const screenSummary = lastScreen
        ? `<strong>${escapeHtml(lastScreen.instrument || t("screeningFallback"))}</strong> — ${escapeHtml(
            (lastScreen.result && lastScreen.result.summary) || t("done")
          )}`
        : escapeHtml(t("noScreensYet"));

      const charts = overlays
        .slice(0, LIMITS.dossierChartsMax)
        .map((o) => ({
          url: safeOverlayUrl(o.url) || (o.filename ? apiUrl(EP.overlay(o.filename)) : null),
          label: o.measure ? measureLabel(o.measure) : t("chartOverlayAlt"),
        }))
        .filter((o) => o.url);
      const chartsHtml = charts.length
        ? `<div class="dossier-charts compact">${charts
            .map(
              (o) =>
                `<a href="${escapeHtml(o.url)}" target="_blank" rel="noopener" title="${escapeHtml(
                  o.label
                )}"><img class="overlay-img" data-authed-src="${escapeHtml(o.url)}" alt="${escapeHtml(
                  o.label
                )}" loading="lazy" /></a>`
            )
            .join("")}</div>`
        : "";

      body.innerHTML = `
        <div class="summary-hero">
          <div class="summary-hero-text">
            <h3 class="summary-name">${escapeHtml(p.name || "")}</h3>
            <p class="summary-meta">${escapeHtml(
              [
                sexLabel,
                maturity,
                p.gestational_age_weeks != null
                  ? `${t("gaLabel")} ${t("weeksSuffix", {
                      n: fmtNum(p.gestational_age_weeks),
                    })}`
                  : "",
              ]
                .filter(Boolean)
                .join(" · ")
            )}</p>
          </div>
          <div class="summary-age-pill">
            <span class="lbl">${escapeHtml(t("ageLabel"))}</span>
            <span class="val">${escapeHtml(chrono != null ? formatAgeLabel(chrono) : "—")}</span>
            ${
              corr != null &&
              chrono != null &&
              Math.abs(corr - chrono) >= LIMITS.correctedAgeNoticeMonths
                ? `<span class="corr">${escapeHtml(
                    t("correctedAgeNote", { age: fmtNum(corr) })
                  )}</span>`
                : ""
            }
          </div>
        </div>
        <div class="summary-grid">
          <div class="summary-card">
            <h4>${escapeHtml(t("latestGrowth"))}</h4>
            ${
              growthBits
                ? `<ul class="dossier-list tight">${growthBits}</ul>`
                : `<p class="muted">${escapeHtml(t("noGrowthYet"))}</p>`
            }
          </div>
          <div class="summary-card">
            <h4>${escapeHtml(t("screeningTitle"))}</h4>
            <p class="muted tight">${escapeHtml(
              t("screeningCount", { n: fmtNum(screens.length, 0) })
            )}</p>
            <p class="summary-last">${screenSummary}</p>
          </div>
          ${
            chartsHtml
              ? `<div class="summary-card summary-card-wide"><h4>${escapeHtml(
                  t("chartsSaved")
                )}</h4>${chartsHtml}</div>`
              : ""
          }
        </div>
        <div class="summary-actions">
          <a class="btn btn-secondary btn-sm" href="#/growth">${escapeHtml(t("openGrowth"))}</a>
          <a class="btn btn-primary btn-sm" href="#/screening">${escapeHtml(t("openScreening"))}</a>
        </div>
      `;
      hydrateAuthedImages(panel);
      panel.hidden = false;
    } catch (err) {
      if (token !== state.dossierToken || isAbortError(err)) return;
      state.lastDossier = null;
      renderMessage(body, `${t("dossierLoadFailed")} ${err.message}`, {
        retry: () => loadChildDossier(childId),
      });
      panel.hidden = false;
    } finally {
      if (token === state.dossierToken) panel.removeAttribute("aria-busy");
    }
  }

  function escapeHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /**
   * Text-only status/error renderer with an optional retry button, so a failed
   * fetch never leaves a spinner or an empty panel behind.
   */
  function renderMessage(container, message, { retry, className = "muted" } = {}) {
    if (!container) return;
    container.textContent = "";
    const p = document.createElement("p");
    p.className = className;
    p.textContent = message;
    container.appendChild(p);
    if (typeof retry === "function") {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-ghost btn-sm retry-btn";
      btn.textContent = t("retry");
      btn.addEventListener("click", retry);
      container.appendChild(btn);
    }
  }

  let toastHideTimer = null;
  let toastFadeTimer = null;

  function toast(message, type = "info") {
    const el = $("#toast");
    if (!el || !message) return;
    window.clearTimeout(toastHideTimer);
    window.clearTimeout(toastFadeTimer);
    el.textContent = message;
    el.classList.toggle("error", type === "error");
    el.hidden = false;
    el.classList.add("show");
    toastHideTimer = window.setTimeout(() => {
      el.classList.remove("show");
      toastFadeTimer = window.setTimeout(() => {
        el.hidden = true;
      }, TIME.toastFadeMs);
    }, TIME.toastVisibleMs);
  }

  function setLoading(btn, loading) {
    if (!btn) return;
    btn.disabled = loading;
    btn.classList.toggle("loading", loading);
    const spin = $(".spinner", btn);
    if (spin) spin.hidden = !loading;
  }

  function apiUrl(path) {
    return `${API_BASE}${path}`;
  }

  function isAbortError(err) {
    return !!err && (err.name === "AbortError" || err.name === "TimeoutError");
  }

  function timeoutError() {
    const e = new Error(t("requestTimeout"));
    e.timeout = true;
    return e;
  }

  /** Pick the friendliest message out of FastAPI's error shapes. */
  function errorMessage(data, status) {
    const d = data && (data.detail != null ? data.detail : data.message || data.error);
    if (typeof d === "string" && d.trim()) return d;
    if (Array.isArray(d)) {
      const first = d.find((x) => x && (x.msg || x.detail));
      if (first) return String(first.msg || first.detail);
    }
    if (d && typeof d === "object" && typeof d.detail === "string") return d.detail;
    return t("requestFailed", { status });
  }

  /**
   * Ask the operator for the API key when the backend demands one. Kept
   * deliberately minimal (window.prompt) so it works before any UI has
   * rendered — a 401 can happen on the very first request.
   */
  /**
   * Show a blocking sign-in overlay and resolve once the user authenticates.
   * Built as an overlay rather than a separate page so a 401 on any request
   * can recover in place without losing what the user was doing.
   */
  function showLoginOverlay(message) {
    return new Promise((resolve) => {
      const existing = document.getElementById("nestling-login-overlay");
      if (existing) existing.remove();

      const overlay = document.createElement("div");
      overlay.id = "nestling-login-overlay";
      overlay.setAttribute("role", "dialog");
      overlay.setAttribute("aria-modal", "true");
      overlay.style.cssText =
        "position:fixed;inset:0;z-index:99999;display:flex;align-items:center;" +
        "justify-content:center;background:rgba(15,23,42,.72);backdrop-filter:blur(4px);" +
        "padding:16px;";

      const card = document.createElement("form");
      card.style.cssText =
        "background:#fff;color:#0f172a;border-radius:16px;padding:24px;width:100%;" +
        "max-width:360px;box-shadow:0 20px 60px rgba(0,0,0,.35);font-family:inherit;" +
        "display:flex;flex-direction:column;gap:12px;";
      const inputCss =
        "padding:12px;border:1px solid #cbd5e1;border-radius:10px;font-size:16px;";
      card.innerHTML =
        '<h2 id="nestling-auth-title" style="margin:0;font-size:1.15rem;">Sign in to Nestling</h2>' +
        '<p id="nestling-login-msg" style="margin:0;font-size:.85rem;color:#64748b;"></p>' +
        '<input id="nestling-login-user" name="username" autocomplete="username" ' +
        'placeholder="Username" style="' + inputCss + '">' +
        '<input id="nestling-login-pass" name="password" type="password" autocomplete="current-password" ' +
        'placeholder="Password" style="' + inputCss + '">' +
        '<button id="nestling-auth-submit" type="submit" style="padding:12px;border:0;border-radius:10px;' +
        'background:#0f766e;color:#fff;font-size:16px;font-weight:600;cursor:pointer;">Sign in</button>' +
        '<button id="nestling-auth-toggle" type="button" style="background:none;border:0;color:#0f766e;' +
        'font-size:.85rem;cursor:pointer;text-decoration:underline;padding:4px;">' +
        "Don't have an account? Create one</button>";

      overlay.appendChild(card);
      document.body.appendChild(overlay);

      const msgEl = card.querySelector("#nestling-login-msg");
      const userEl = card.querySelector("#nestling-login-user");
      const passEl = card.querySelector("#nestling-login-pass");
      const titleEl = card.querySelector("#nestling-auth-title");
      const submitEl = card.querySelector("#nestling-auth-submit");
      const toggleEl = card.querySelector("#nestling-auth-toggle");
      let mode = "login";

      msgEl.textContent = message || "Please sign in to continue.";
      setTimeout(() => userEl.focus(), 50);

      toggleEl.addEventListener("click", () => {
        mode = mode === "login" ? "register" : "login";
        const registering = mode === "register";
        titleEl.textContent = registering ? "Create your account" : "Sign in to Nestling";
        submitEl.textContent = registering ? "Create account" : "Sign in";
        toggleEl.textContent = registering
          ? "Already have an account? Sign in"
          : "Don't have an account? Create one";
        passEl.setAttribute("autocomplete", registering ? "new-password" : "current-password");
        msgEl.style.color = "#64748b";
        msgEl.textContent = registering
          ? "Your children's data is private to your account."
          : "Please sign in to continue.";
      });

      card.addEventListener("submit", async (ev) => {
        ev.preventDefault();
        const registering = mode === "register";
        msgEl.style.color = "#64748b";
        msgEl.textContent = registering ? "Creating account…" : "Signing in…";
        try {
          const res = await fetch(apiUrl(registering ? "/auth/register" : "/auth/login"), {
            method: "POST",
            headers: { "Content-Type": "application/json", Accept: "application/json" },
            body: JSON.stringify({ username: userEl.value, password: passEl.value }),
          });
          const data = await res.json().catch(() => null);
          if (!res.ok) {
            const detail =
              (data && data.detail && data.detail.detail) ||
              (registering ? "Could not create account" : "Sign-in failed");
            msgEl.textContent = detail;
            msgEl.style.color = "#b91c1c";
            passEl.value = "";
            return;
          }
          CFG.api.setToken(data.token, data.username);
          overlay.remove();
          renderAccountUi();
          resolve(true);
        } catch (_) {
          msgEl.textContent = "Could not reach the server.";
          msgEl.style.color = "#b91c1c";
        }
      });
    });
  }

  /**
   * Sign out on this device.
   *
   * Clears the session token *and* the per-account view state. Leaving the
   * active child or chat-session id behind would show the next account this
   * one's data, which is exactly the leak the server-side scoping prevents.
   */
  function logOut() {
    CFG.api.setToken("");
    try {
      window.localStorage.removeItem("nestling_api_key");
    } catch (_) {
      /* storage unavailable */
    }
    store.remove(STORAGE.activeChild);
    store.remove(STORAGE.chatSession);
    state.chatSessionId = null;
    state.activeChildId = null;
    state.lastDossier = null;
    window.location.hash = "#/";
    window.location.reload();
  }

  /** Refresh the account button/menu from the current session token. */
  function renderAccountUi() {
    const btn = document.getElementById("account-btn");
    const initialEl = document.getElementById("account-initial");
    if (!btn) return;
    const username = CFG.api.username || "";
    const signedIn = !!CFG.api.key;
    btn.hidden = !signedIn;
    if (!signedIn) return;
    if (initialEl) {
      initialEl.textContent = (username.trim()[0] || "?").toUpperCase();
    }
  }

  function initAccountUi() {
    const btn = document.getElementById("account-btn");
    if (!btn) return;
    // The avatar opens Account settings directly. An always-mounted dropdown
    // was both a second place for sign-out to live and a rendering hazard:
    // its `display: flex` beat the `hidden` attribute, so it never hid.
    btn.addEventListener("click", () => showAccountSettings());
    renderAccountUi();
  }

  /** Account settings panel: identity, privacy note, and sign-out. */
  function showAccountSettings() {
    const existing = document.getElementById("nestling-account-overlay");
    if (existing) existing.remove();
    const overlay = document.createElement("div");
    overlay.id = "nestling-account-overlay";
    overlay.className = "nestling-modal-overlay";
    const card = document.createElement("div");
    card.className = "nestling-modal-card";
    card.innerHTML =
      `<h2 class="nestling-modal-title">${escapeHtml(t("accountSettingsTitle"))}</h2>` +
      `<p class="nestling-modal-row"><span>${escapeHtml(t("accountUsername"))}</span>` +
      `<strong>${escapeHtml(CFG.api.username || "—")}</strong></p>` +
      `<p class="nestling-modal-note">${escapeHtml(t("accountDataNote"))}</p>` +
      `<p class="nestling-modal-note">${escapeHtml(t("accountSessionNote"))}</p>` +
      `<button type="button" class="btn btn-ghost btn-sm nestling-modal-danger" id="nestling-clear-history">${escapeHtml(
        t("clearHistory")
      )}</button>` +
      `<p class="nestling-modal-note">${escapeHtml(t("clearHistoryNote"))}</p>` +
      `<div class="nestling-modal-actions">` +
      `<button type="button" class="btn btn-ghost btn-sm" id="nestling-account-close">${escapeHtml(
        t("close")
      )}</button>` +
      `<button type="button" class="btn btn-primary btn-sm" id="nestling-account-logout">${escapeHtml(
        t("logOut")
      )}</button>` +
      `</div>`;
    overlay.appendChild(card);
    document.body.appendChild(overlay);
    card.querySelector("#nestling-account-close").addEventListener("click", () => overlay.remove());
    card.querySelector("#nestling-clear-history").addEventListener("click", async (ev) => {
      if (!window.confirm(t("clearHistoryConfirm"))) return;
      const btn = ev.currentTarget;
      btn.disabled = true;
      try {
        const res = await api(EP.sessions, { method: "DELETE" });
        // The open conversation no longer exists server-side; drop the local
        // pointer too or the next turn would post to a deleted session.
        store.remove(STORAGE.chatSession);
        state.chatSessionId = null;
        toast(t("clearHistoryDone", { count: (res && res.deleted) || 0 }));
        overlay.remove();
        if (location.hash.startsWith("#/chat")) location.reload();
      } catch (err) {
        toast(err.message || t("clearHistoryFailed"), "error");
        btn.disabled = false;
      }
    });
    card.querySelector("#nestling-account-logout").addEventListener("click", () => {
      if (window.confirm(t("logOutConfirm"))) logOut();
    });
    overlay.addEventListener("click", (ev) => {
      if (ev.target === overlay) overlay.remove();
    });
  }

  async function promptForApiKey() {
    // Prefer the interactive form when the server has login configured;
    // fall back to pasting a raw API key for headless/private deployments.
    let loginRequired = false;
    try {
      const res = await fetch(apiUrl("/auth/config"), { headers: { Accept: "application/json" } });
      if (res.ok) loginRequired = !!(await res.json()).login_required;
    } catch (_) {
      /* fall through to the key prompt */
    }
    if (loginRequired) {
      await showLoginOverlay("Your session expired or you are not signed in.");
      return CFG.api.key;
    }
    const entered = window.prompt(
      "This Nestling server requires an API key.\n" +
        "Paste the NESTLING_API_KEY value from the server's .env file:",
      CFG.api.key || ""
    );
    if (entered === null) return "";
    return CFG.api.setKey(entered);
  }

  /**
   * Single fetch wrapper: JSON encoding, caller-provided AbortSignal, hard
   * timeout so a hung backend can never leave the UI spinning, and typed errors.
   */
  async function api(path, options = {}) {
    const {
      timeoutMs = TIME.requestTimeoutMs,
      signal: outerSignal,
      _retriedAuth = false,
      ...rest
    } = options;
    const isForm = typeof FormData !== "undefined" && rest.body instanceof FormData;
    const opts = {
      ...rest,
      headers: {
        Accept: "application/json",
        ...(rest.body && !isForm ? { "Content-Type": "application/json" } : {}),
        // No-op unless the deployment sets NESTLING_API_KEY.
        ...(CFG.api.authHeaders ? CFG.api.authHeaders() : {}),
        ...(rest.headers || {}),
      },
    };
    if (opts.body && typeof opts.body === "object" && !isForm) {
      opts.body = JSON.stringify(opts.body);
    }

    const ctrl = new AbortController();
    const relayAbort = () => ctrl.abort();
    if (outerSignal) {
      if (outerSignal.aborted) ctrl.abort();
      else outerSignal.addEventListener("abort", relayAbort, { once: true });
    }
    let timedOut = false;
    const timer = window.setTimeout(() => {
      timedOut = true;
      ctrl.abort();
    }, timeoutMs);
    opts.signal = ctrl.signal;

    try {
      let res;
      try {
        res = await fetch(apiUrl(path), opts);
      } catch (err) {
        if (timedOut) throw timeoutError();
        if (isAbortError(err)) throw err;
        const e = new Error(t("serverUnreachable"));
        e.cause = err;
        e.offline = true;
        throw e;
      }
      let text;
      try {
        text = await res.text();
      } catch (err) {
        if (timedOut) throw timeoutError();
        throw err;
      }
      let data = null;
      if (text) {
        try {
          data = JSON.parse(text);
        } catch (_) {
          data = { raw: text };
        }
      }
      if (!res.ok) {
        // The deployment requires an API key we don't have yet (or ours was
        // rotated). Ask once, store it, and retry transparently so the
        // operator isn't left staring at a dead UI.
        if (res.status === 401 && !_retriedAuth) {
          const entered = await promptForApiKey();
          if (entered) {
            return api(path, { ...options, _retriedAuth: true });
          }
        }
        const e = new Error(errorMessage(data, res.status));
        e.status = res.status;
        e.data = data;
        throw e;
      }
      return data;
    } finally {
      window.clearTimeout(timer);
      if (outerSignal) outerSignal.removeEventListener("abort", relayAbort);
    }
  }

  /**
   * Only same-origin (or configured API origin) image URLs are rendered, so a
   * compromised/odd API payload cannot inject `javascript:` or third-party URLs.
   */
  /**
   * Point an <img> at an API-served image.
   *
   * A plain src= cannot carry the Authorization header, so once the server
   * requires auth every /api/overlays/* request 401s and the chart renders
   * broken. Fetch it with credentials instead and hand the element a blob URL.
   * data: URIs and cross-origin URLs are passed straight through.
   */
  function setAuthedImageSrc(img, url) {
    if (!url) return;
    const isApiPath = /^\/?api\//.test(url) || url.indexOf(CFG.api.base) === 0;
    if (!isApiPath || url.startsWith("data:")) {
      img.src = url;
      return;
    }
    const headers = CFG.api.authHeaders ? CFG.api.authHeaders() : {};
    if (!headers.Authorization && !headers["X-API-Key"]) {
      img.src = url; // unauthenticated deployment — direct src still works
      return;
    }
    fetch(url, { headers })
      .then((res) => (res.ok ? res.blob() : Promise.reject(new Error(String(res.status)))))
      .then((blob) => {
        const objectUrl = URL.createObjectURL(blob);
        img.src = objectUrl;
        // Release the blob once the browser has decoded it.
        img.addEventListener("load", () => URL.revokeObjectURL(objectUrl), { once: true });
      })
      .catch(() => {
        img.alt = t("chartOverlayAlt");
        img.classList.add("overlay-img-failed");
      });
  }

  /** Attach authenticated sources to any [data-authed-src] images under `root`. */
  function hydrateAuthedImages(root) {
    if (!root) return;
    root.querySelectorAll("img[data-authed-src]").forEach((img) => {
      const url = img.getAttribute("data-authed-src");
      img.removeAttribute("data-authed-src");
      setAuthedImageSrc(img, url);
    });
  }

  function safeOverlayUrl(raw) {
    if (!raw) return null;
    const s = String(raw).trim();
    if (!s) return null;
    if (/^data:image\/(png|jpeg|webp|gif);base64,[a-z0-9+/=\s]+$/i.test(s)) return s;
    let url;
    try {
      url = new URL(s, window.location.href);
    } catch (_) {
      return null;
    }
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    if (url.origin !== window.location.origin && url.origin !== API_ORIGIN) return null;
    return url.href;
  }

  function overlaySrc(data) {
    if (!data || typeof data !== "object") return null;
    const direct = data.overlay_url || data.image_url || data.overlay_image;
    if (direct) return safeOverlayUrl(direct);
    // API often returns a bare filename in `overlay` / `overlay_filename`.
    const named =
      data.overlay_filename ||
      (data.overlay && !/[\\/]/.test(String(data.overlay)) ? data.overlay : null);
    if (named) return apiUrl(EP.overlay(String(named)));
    const path = data.overlay_path || data.overlay;
    if (path) {
      const file = String(path).split(/[/\\]/).pop();
      if (file) return apiUrl(EP.overlay(file));
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
    const leavingChat = path !== "/chat";
    const leavingScreening = path !== "/screening";
    $$(".view").forEach((v) => {
      v.hidden = v.id !== id;
    });
    const topbar = $("#topbar");
    if (topbar) topbar.classList.toggle("at-home", path === "/");

    // Never leave work running for a view the parent has left.
    if (leavingChat) abortChat();
    if (leavingScreening) clearQuizAdvanceTimer();

    if (path === "/child") loadChildren();
    if (path === "/chat") initChat();
    if (path === "/growth") {
      fillChildSelects();
      syncGrowthSexFromChild();
    }
    if (path === "/screening") {
      fillChildSelects();
      syncScreeningAgeFromChild();
      if (!state.quiz) showScreeningPicker();
    }
    window.scrollTo({ top: 0, behavior: prefersReducedMotion() ? "auto" : "smooth" });
  }

  /* —— Home health —— */
  async function checkHealth() {
    const el = $("#health-status");
    if (!el) return;
    try {
      await api(EP.health);
      setI18nText(el, "ready");
      el.className = "home-foot ok";
    } catch (_) {
      setI18nText(el, "offline");
      el.className = "home-foot bad";
    }
  }

  /* —— Children —— */
  /** Newest first, active child pinned, capped — the API list is unpaginated. */
  function childrenForSelect() {
    const activeId = activeChildId();
    const sorted = state.children.slice().sort((a, b) => {
      const at = String(a.created_at || "");
      const bt = String(b.created_at || "");
      return bt.localeCompare(at);
    });
    const capped = sorted.slice(0, LIMITS.childSelectMax);
    if (activeId && !capped.some((c) => childIdOf(c) === activeId)) {
      const active = state.children.find((c) => childIdOf(c) === activeId);
      if (active) capped.unshift(active);
    }
    return capped;
  }

  function fillChildSelects() {
    const activeId = activeChildId();
    const options = childrenForSelect();
    [$("#growth-child"), $("#screening-child")].forEach((sel) => {
      if (!sel) return;
      const previous = sel.value;
      // Rebuilt wholesale so repeated loads can't stack duplicate options.
      sel.textContent = "";
      const none = document.createElement("option");
      none.value = "";
      none.textContent = t("none");
      sel.appendChild(none);
      options.forEach((c) => {
        const id = childIdOf(c);
        if (!id) return;
        const opt = document.createElement("option");
        opt.value = id;
        opt.textContent = c.name || id;
        sel.appendChild(opt);
      });
      const wanted =
        previous && options.some((c) => childIdOf(c) === previous)
          ? previous
          : activeId || "";
      sel.value = wanted || "";
    });
  }

  async function selectChild(child) {
    const id = childIdOf(child);
    if (!id) return;
    saveActiveChild(child);
    // A different child means a different conversation context.
    abortChat();
    state.chatSessionId = null;
    store.remove(STORAGE.chatSession);
    const thread = $("#chat-thread");
    if (thread) {
      thread.dataset.ready = "";
      thread.textContent = "";
    }
    $$(".child-chip", $("#children-list")).forEach((b) => {
      const isActive = b.dataset.id === id;
      b.classList.toggle("active", isActive);
      b.setAttribute("aria-pressed", String(isActive));
    });
    fillChildSelects();
    syncScreeningAgeFromChild();
    toast(t("childSelected", { name: child.name || "" }));
    await loadChildDossier(id);
  }

  async function loadChildren() {
    const list = $("#children-list");
    const emptyHint = $("#child-empty-hint");
    if (!list) return;
    const token = ++state.childrenToken;
    list.setAttribute("aria-busy", "true");
    if (!list.querySelector(".child-chip")) renderMessage(list, t("loading"));
    try {
      const data = await api(EP.children);
      if (token !== state.childrenToken) return;
      state.children = Array.isArray(data) ? data : data.children || [];
      fillChildSelects();
      const panel = $("#child-dossier");
      if (!state.children.length) {
        list.textContent = "";
        if (emptyHint) emptyHint.hidden = false;
        if (panel) panel.hidden = true;
        return;
      }
      if (emptyHint) emptyHint.hidden = true;
      const activeId = activeChildId();
      // Active child first, then most recently added.
      const ordered = state.children.slice().sort((a, b) => {
        if (activeId) {
          if (childIdOf(a) === activeId) return -1;
          if (childIdOf(b) === activeId) return 1;
        }
        return String(b.created_at || "").localeCompare(String(a.created_at || ""));
      });
      // Rebuilt from scratch each time: no duplicate rows, no stale listeners.
      list.textContent = "";
      ordered.slice(0, LIMITS.childChipsMax).forEach((c) => {
        const id = childIdOf(c);
        if (!id) return;
        const isActive = id === activeId;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `child-chip${isActive ? " active" : ""}`;
        btn.dataset.id = id;
        btn.setAttribute("aria-pressed", String(isActive));
        const name = document.createElement("span");
        name.className = "child-chip-name";
        name.textContent = c.name || id;
        btn.appendChild(name);
        const age = ageMonthsFromDob(c.date_of_birth);
        const metaText = [
          maturityLabel(c.gestational_age_weeks),
          age != null ? formatAgeLabel(age) : "",
        ]
          .filter(Boolean)
          .join(" · ");
        if (metaText) {
          const meta = document.createElement("span");
          meta.className = "meta";
          meta.textContent = metaText;
          btn.appendChild(meta);
        }
        btn.addEventListener("click", () => {
          selectChild(c);
        });
        list.appendChild(btn);
      });
      if (activeId) await loadChildDossier(activeId);
      else if (panel) panel.hidden = true;
    } catch (err) {
      if (token !== state.childrenToken || isAbortError(err)) return;
      renderMessage(list, `${t("childrenLoadFailed")} ${err.message}`, {
        retry: () => loadChildren(),
      });
    } finally {
      if (token === state.childrenToken) list.removeAttribute("aria-busy");
    }
  }

  function setAddChildOpen(open) {
    const form = $("#child-form");
    const toggle = $("#toggle-add-child");
    if (!form) return;
    form.hidden = !open;
    if (toggle) {
      setI18nText(toggle, open ? "hideAddChild" : "addChild");
      toggle.setAttribute("aria-expanded", String(!!open));
    }
    if (open) {
      const first = form.querySelector('input[name="name"]');
      if (first) first.focus();
    }
  }

  function wireChildForm() {
    const toggle = $("#toggle-add-child");
    const cancel = $("#cancel-add-child");
    if (toggle) {
      toggle.addEventListener("click", () => {
        const form = $("#child-form");
        setAddChildOpen(form && form.hidden);
      });
    }
    if (cancel) {
      cancel.addEventListener("click", () => setAddChildOpen(false));
    }
    const form0 = $("#child-form");
    if (!form0) return;
    form0.addEventListener("submit", async (e) => {
      e.preventDefault();
      const form = e.target;
      const btn = $("#child-submit");
      const fd = new FormData(form);
      const dob = String(fd.get("date_of_birth") || "").trim();
      const body = {
        name: String(fd.get("name") || "").trim(),
        sex: fd.get("sex"),
        gestational_age_weeks: Number(fd.get("gestational_age_weeks")),
      };
      if (dob) body.date_of_birth = dob;
      setLoading(btn, true);
      try {
        const created = await api(EP.children, { method: "POST", body });
        saveActiveChild(created);
        form.reset();
        setAddChildOpen(false);
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
  /** Cancel any in-flight chat request and stop text reveals. */
  function abortChat() {
    if (state.chatAbort) {
      state.chatAbort.abort();
      state.chatAbort = null;
    }
    cancelReveals();
    setChatBusy(false);
  }

  function cancelReveals(finishText = true) {
    [...reveals].forEach((r) => r.stop(finishText));
    reveals.clear();
  }

  async function ensureChatSession() {
    if (state.chatSessionId) return state.chatSessionId;
    const childId = activeChildId() || undefined;
    const data = await api(EP.sessions, {
      method: "POST",
      body: childId ? { child_id: childId } : {},
    });
    state.chatSessionId = data.session_id;
    store.set(STORAGE.chatSession, data.session_id);
    return state.chatSessionId;
  }

  async function startNewChat() {
    abortChat();
    state.chatSessionId = null;
    store.remove(STORAGE.chatSession);
    const thread = $("#chat-thread");
    if (thread) {
      thread.dataset.ready = "";
      thread.textContent = "";
    }
    setChatHistoryOpen(false);
    await ensureChatSession();
    initChat();
    toast(t("newChatStarted"));
  }

  function setChatHistoryOpen(open) {
    state.chatHistoryOpen = !!open;
    const panel = $("#chat-history-panel");
    if (panel) panel.hidden = !open;
    const btn = $("#btn-chat-history");
    if (btn) btn.setAttribute("aria-expanded", String(!!open));
  }

  async function loadChatHistoryList() {
    const list = $("#chat-history-list");
    if (!list) return;
    setChatHistoryOpen(true);
    renderMessage(list, t("loading"));
    const childId = activeChildId() || "";
    const q = new URLSearchParams();
    if (childId) q.set("child_id", childId);
    q.set("limit", String(LIMITS.chatHistoryLimit));
    try {
      const data = await api(`${EP.sessions}?${q.toString()}`);
      const sessions = data.sessions || [];
      if (!sessions.length) {
        renderMessage(list, t("noChatHistory"));
        return;
      }
      list.textContent = "";
      sessions.forEach((s) => {
        if (!s || !s.session_id) return;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "history-item";
        const title = document.createElement("strong");
        title.textContent =
          String(s.title || s.preview || s.session_id).slice(
            0,
            LIMITS.chatHistoryTitleMaxChars
          ) || t("chatFallbackTitle");
        const meta = document.createElement("span");
        meta.className = "muted";
        meta.textContent = `${fmtNum(s.message_count || 0, 0)} · ${formatTimestamp(s.updated_at)}`;
        btn.appendChild(title);
        btn.appendChild(meta);
        btn.addEventListener("click", () => {
          openChatSession(s.session_id).catch((err) => {
            toast(`${t("chatOpenFailed")} ${err.message}`, "error");
          });
        });
        list.appendChild(btn);
      });
    } catch (err) {
      if (isAbortError(err)) return;
      renderMessage(list, `${t("chatHistoryLoadFailed")} ${err.message}`, {
        retry: () => loadChatHistoryList(),
      });
    }
  }

  function formatTimestamp(raw) {
    return String(raw || "")
      .slice(0, LIMITS.timestampChars)
      .replace("T", " ");
  }

  async function openChatSession(sessionId) {
    if (!sessionId) return;
    const data = await api(EP.session(sessionId));
    abortChat();
    state.chatSessionId = sessionId;
    store.set(STORAGE.chatSession, sessionId);
    const thread = $("#chat-thread");
    if (!thread) return;
    thread.dataset.ready = "1";
    thread.textContent = "";
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
    setChatHistoryOpen(false);
    toast(t("chatOpened"));
  }

  function initChat() {
    const thread = $("#chat-thread");
    if (!thread) return;
    if (!thread.dataset.ready) {
      thread.dataset.ready = "1";
      thread.textContent = "";
      appendBubble("assistant", t("chatWelcome"));
    }
    ensureChatSession().catch((err) => {
      if (!isAbortError(err)) toast(err.message, "error");
    });
    const newBtn = $("#btn-new-chat");
    const histBtn = $("#btn-chat-history");
    if (newBtn && !newBtn.dataset.wired) {
      newBtn.dataset.wired = "1";
      newBtn.addEventListener("click", () => {
        startNewChat().catch((err) => {
          if (!isAbortError(err)) toast(err.message, "error");
        });
      });
    }
    if (histBtn && !histBtn.dataset.wired) {
      histBtn.dataset.wired = "1";
      histBtn.setAttribute("aria-expanded", "false");
      histBtn.addEventListener("click", () => {
        if (state.chatHistoryOpen) {
          setChatHistoryOpen(false);
          return;
        }
        loadChatHistoryList();
      });
    }
  }

  function appendBubble(role, text) {
    const thread = $("#chat-thread");
    if (!thread) return null;
    const div = document.createElement("div");
    div.className = `bubble ${role}`;
    div.textContent = text;
    thread.appendChild(div);
    thread.scrollTop = thread.scrollHeight;
    return div;
  }

  /**
   * Shell for a bubble whose text arrives progressively. Hidden from assistive
   * tech while streaming so screen readers hear the finished reply once.
   */
  function createBubbleShell(thread) {
    const div = document.createElement("div");
    div.className = "bubble assistant streaming";
    div.setAttribute("aria-hidden", "true");
    const body = document.createElement("span");
    body.className = "stream-text";
    const caret = document.createElement("span");
    caret.className = "stream-caret";
    caret.setAttribute("aria-hidden", "true");
    div.appendChild(body);
    div.appendChild(caret);
    thread.appendChild(div);
    thread.scrollTop = thread.scrollHeight;
    return { div, body, caret };
  }

  function revealPace(length) {
    const charsPerTick =
      length > REVEAL.longTextChars
        ? REVEAL.charsPerTickMax
        : length > REVEAL.mediumTextChars
          ? REVEAL.charsPerTickMedium
          : length > REVEAL.shortTextChars
            ? REVEAL.charsPerTickSmall
            : REVEAL.charsPerTickMin;
    const delayMs =
      length > REVEAL.longTextChars
        ? TIME.revealTickFastMs
        : length > REVEAL.mediumTextChars
          ? TIME.revealTickMediumMs
          : TIME.revealTickSlowMs;
    return { charsPerTick, delayMs };
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

    const { div, body, caret } = createBubbleShell(thread);
    // Adaptive pace: short replies feel natural; long ones finish sooner
    const len = full.length;
    const { charsPerTick, delayMs } = revealPace(len);

    return new Promise((resolve) => {
      let i = 0;
      let timer = null;
      let done = false;
      const settle = (showAll) => {
        if (done) return;
        done = true;
        if (timer) {
          window.clearTimeout(timer);
          timer = null;
        }
        if (showAll) body.textContent = full;
        div.classList.remove("streaming");
        div.removeAttribute("aria-hidden");
        caret.remove();
        reveals.delete(handle);
        resolve(div);
      };
      const handle = { stop: (finishText) => settle(!!finishText) };
      reveals.add(handle);
      const step = () => {
        timer = null;
        if (!div.isConnected) {
          settle(false);
          return;
        }
        i = Math.min(len, i + charsPerTick);
        body.textContent = full.slice(0, i);
        thread.scrollTop = thread.scrollHeight;
        if (i >= len) {
          settle(false);
          return;
        }
        timer = window.setTimeout(step, delayMs);
      };
      step();
    });
  }

  function appendToolResult(payload) {
    const thread = $("#chat-thread");
    if (!thread || !payload) return;
    let results = payload.tool_results;
    if (!results && payload.tools) {
      const tools = payload.tools;
      if (Array.isArray(tools)) results = tools;
      else if (Array.isArray(tools.tool_calls)) results = tools.tool_calls;
      else results = [tools];
    }
    const list = Array.isArray(results) ? results : results ? [results] : [];
    // A Set keeps one image per URL: repeated plots can't duplicate overlays.
    const imgs = new Set();
    list.forEach((tr) => {
      if (!tr) return;
      const res = tr.result || tr;
      const img = overlaySrc(res) || overlaySrc(tr) || overlaySrc(payload);
      if (img) imgs.add(img);
    });
    const topImg = overlaySrc(payload);
    if (topImg) imgs.add(topImg);
    if (!imgs.size) return;
    // Replace prior chart images in this thread so replots don't stack conflicting overlays.
    thread.querySelectorAll(".tool-block.clean").forEach((el) => el.remove());
    imgs.forEach((src) => {
      const block = document.createElement("div");
      block.className = "tool-block clean";
      const img = document.createElement("img");
      img.className = "overlay-img";
      img.alt = t("chartOverlayAlt");
      img.loading = "lazy";
      setAuthedImageSrc(img, src);
      block.appendChild(img);
      thread.appendChild(block);
    });
    thread.scrollTop = thread.scrollHeight;
  }

  /**
   * POST /api/chat/stream — parse SSE token/result events.
   * Throws if the endpoint is unavailable so caller can fall back to /api/chat.
   */
  async function chatViaStream(body, { signal, onToken } = {}) {
    // Chain an internal controller so an idle stream can be torn down without
    // being confused with a parent-initiated cancel.
    const ctrl = new AbortController();
    const relayAbort = () => ctrl.abort();
    if (signal) {
      if (signal.aborted) ctrl.abort();
      else signal.addEventListener("abort", relayAbort, { once: true });
    }
    let stalled = false;
    let idleTimer = null;
    const armIdleTimer = () => {
      if (idleTimer) window.clearTimeout(idleTimer);
      idleTimer = window.setTimeout(() => {
        stalled = true;
        ctrl.abort();
      }, TIME.streamIdleTimeoutMs);
    };
    const disarm = () => {
      if (idleTimer) window.clearTimeout(idleTimer);
      idleTimer = null;
      if (signal) signal.removeEventListener("abort", relayAbort);
    };

    let res;
    try {
      armIdleTimer();
      res = await fetch(apiUrl(EP.chatStream), {
        method: "POST",
        headers: {
          Accept: "text/event-stream",
          "Content-Type": "application/json",
          ...(CFG.api.authHeaders ? CFG.api.authHeaders() : {}),
        },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });
    } catch (err) {
      disarm();
      if (stalled) throw timeoutError();
      if (isAbortError(err)) throw err;
      const e = new Error(t("serverUnreachable"));
      e.cause = err;
      e.streamUnavailable = true;
      throw e;
    }
    if (!res.ok || !res.body) {
      disarm();
      const e = new Error(errorMessage(null, res.status));
      e.status = res.status;
      e.streamUnavailable = true;
      throw e;
    }
    const ctype = (res.headers.get("content-type") || "").toLowerCase();
    if (ctype.includes("application/json") && !ctype.includes("text/event-stream")) {
      disarm();
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

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        armIdleTimer();
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
    } catch (err) {
      // Aborted mid-stream: a stall is retryable, a parent cancel is not an error.
      if (stalled) {
        const e = new Error(t("streamStalled"));
        e.timeout = true;
        throw e;
      }
      throw err;
    } finally {
      disarm();
      try {
        reader.cancel();
      } catch (_) {
        /* already closed */
      }
    }

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
      if (isAbortError(err)) throw err;
      if (!err.streamUnavailable) throw err;
      // Streaming endpoint missing or wrong content type — fall back to /chat.
      const data = await api(EP.chat, {
        method: "POST",
        body,
        signal,
        timeoutMs: TIME.chatRequestTimeoutMs,
      });
      const reply = data.reply || data.message || data.response || data.answer || "";
      if (reply && onToken) onToken(reply);
      return data;
    }
  }

  function createStreamingAssistantBubble() {
    const thread = $("#chat-thread");
    if (!thread) return null;
    const { div, body, caret } = createBubbleShell(thread);
    const reducedMotion = prefersReducedMotion();
    let queue = "";
    let timer = null;
    let finishing = false;
    let closed = false;

    const cleanup = (finalText) => {
      if (timer) {
        window.clearTimeout(timer);
        timer = null;
      }
      if (closed) return div.isConnected ? div : null;
      closed = true;
      reveals.delete(handle);
      if (finalText != null && finalText !== "") body.textContent = finalText;
      div.classList.remove("streaming");
      div.removeAttribute("aria-hidden");
      caret.remove();
      if (!body.textContent) {
        div.remove();
        return null;
      }
      return div;
    };

    const handle = {
      stop(finishText) {
        // Flush whatever is queued so a cancelled reveal never truncates text.
        cleanup(finishText && queue ? body.textContent + queue : undefined);
      },
    };
    reveals.add(handle);

    const schedule = () => {
      if (timer) return;
      timer = window.setTimeout(drain, TIME.streamTickMs);
    };

    const drain = () => {
      timer = null;
      if (!div.isConnected) {
        cleanup();
        return;
      }
      if (!queue.length) {
        if (finishing) cleanup();
        return;
      }
      const qLen = queue.length;
      const charsPerTick =
        qLen > REVEAL.queueLargeChars
          ? REVEAL.charsPerTickMedium
          : qLen > REVEAL.queueMediumChars
            ? REVEAL.charsPerTickSmall
            : REVEAL.charsPerTickMin;
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
        if (!text || closed) return;
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
        if (!timer) schedule();
        return div;
      },
    };
  }

  /** Composer autosize ceiling — owned by styles.css (--composer-max-h). */
  function composerMaxHeight() {
    const raw = getComputedStyle(document.documentElement).getPropertyValue(
      "--composer-max-h"
    );
    const px = parseFloat(raw);
    return Number.isFinite(px) && px > 0 ? px : LIMITS.composerMaxHeightPx;
  }

  function setChatBusy(busy) {
    setLoading($("#chat-send"), busy);
    const cancelBtn = $("#chat-cancel");
    const attachBtn = $("#chat-attach");
    const input = $("#chat-input");
    const thread = $("#chat-thread");
    if (cancelBtn) cancelBtn.hidden = !busy;
    if (attachBtn) attachBtn.disabled = busy;
    if (input) input.disabled = busy;
    if (thread) {
      if (busy) thread.setAttribute("aria-busy", "true");
      else thread.removeAttribute("aria-busy");
    }
  }

  function wireChat() {
    const form = $("#chat-form");
    const input = $("#chat-input");
    const attachBtn = $("#chat-attach");
    const cancelBtn = $("#chat-cancel");
    const fileInput = $("#chat-file");
    if (!form || !input) return;

    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, composerMaxHeight()) + "px";
    });
    if (cancelBtn && !cancelBtn.dataset.wired) {
      cancelBtn.dataset.wired = "1";
      cancelBtn.addEventListener("click", () => {
        if (state.chatAbort) state.chatAbort.abort();
      });
    }
    if (attachBtn && fileInput && !attachBtn.dataset.wired) {
      attachBtn.dataset.wired = "1";
      attachBtn.addEventListener("click", () => fileInput.click());
      fileInput.addEventListener("change", () => {
        const file = fileInput.files && fileInput.files[0];
        fileInput.value = "";
        if (file) sendPhoto(file);
      });
    }
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      appendBubble("user", message);
      input.value = "";
      input.style.height = "auto";
      abortChat();
      state.chatAbort = new AbortController();
      const { signal } = state.chatAbort;
      setChatBusy(true);
      const scene = appendBubble("tool-scene", t("thinking"));
      let streamBubble = null;
      const applyChatResult = async (data, streamed) => {
        if (scene.isConnected) scene.remove();
        if (data.session_id) {
          state.chatSessionId = data.session_id;
          store.set(STORAGE.chatSession, data.session_id);
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
      const runTurn = async () => {
        await ensureChatSession();
        let gotToken = false;
        const data = await chatRequest(
          {
            message,
            session_id: state.chatSessionId || undefined,
            child_id: activeChildId() || undefined,
            ui_lang: state.lang,
          },
          {
            signal,
            onToken: (text) => {
              if (!gotToken) {
                if (scene && scene.isConnected) scene.remove();
                streamBubble = createStreamingAssistantBubble();
                gotToken = true;
              }
              if (streamBubble) streamBubble.append(text);
            },
          }
        );
        await applyChatResult(data, gotToken);
      };

      try {
        await runTurn();
      } catch (err) {
        if (scene && scene.isConnected) scene.remove();
        if (streamBubble) {
          streamBubble.finish();
          streamBubble = null;
        }
        if (isAbortError(err)) {
          appendBubble("system", t("chatCancelled"));
          return;
        }
        // Stale session id (e.g. DB reset): drop it and retry once.
        const stale =
          String(err.message || "").includes("session_not_found") || err.status === 404;
        if (stale) {
          state.chatSessionId = null;
          store.remove(STORAGE.chatSession);
          try {
            await runTurn();
            return;
          } catch (err2) {
            if (isAbortError(err2)) {
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
        if (state.chatAbort && state.chatAbort.signal === signal) state.chatAbort = null;
        setChatBusy(false);
        if (!input.disabled) input.focus();
      }
    });
  }

  /**
   * Parent photo -> /chat/vision (multipart). Validates type/size client-side
   * against the same ceiling the API enforces, and always resolves the UI state.
   */
  async function sendPhoto(file) {
    if (!file) return;
    if (file.type && !LIMITS.acceptedImageTypes.includes(file.type)) {
      toast(t("photoUnsupported"), "error");
      return;
    }
    if (file.size > LIMITS.maxUploadBytes) {
      toast(
        t("photoTooLarge", {
          mb: fmtNum(LIMITS.maxUploadBytes / LIMITS.bytesPerMegabyte, 0),
        }),
        "error"
      );
      return;
    }
    const input = $("#chat-input");
    const caption = input ? input.value.trim() : "";
    if (input) {
      input.value = "";
      input.style.height = "auto";
    }

    const thread = $("#chat-thread");
    if (thread && !thread.dataset.ready) thread.dataset.ready = "1";
    const bubble = appendBubble("user", caption);
    if (bubble) {
      const objectUrl = URL.createObjectURL(file);
      const img = document.createElement("img");
      img.className = "bubble-photo";
      img.alt = t("photoAlt");
      img.src = objectUrl;
      img.addEventListener("load", () => URL.revokeObjectURL(objectUrl), { once: true });
      img.addEventListener("error", () => URL.revokeObjectURL(objectUrl), { once: true });
      bubble.appendChild(img);
    }

    abortChat();
    state.chatAbort = new AbortController();
    const { signal } = state.chatAbort;
    setChatBusy(true);
    const scene = appendBubble("tool-scene", t("photoSending"));
    try {
      const fd = new FormData();
      fd.append("image", file, file.name || "photo.png");
      fd.append("message", caption);
      if (state.chatSessionId) fd.append("session_id", state.chatSessionId);
      const childId = activeChildId();
      if (childId) fd.append("child_id", childId);
      fd.append("ui_lang", state.lang);
      const data = await api(EP.chatVision, {
        method: "POST",
        body: fd,
        signal,
        timeoutMs: TIME.visionTimeoutMs,
      });
      if (scene && scene.isConnected) scene.remove();
      if (data.session_id) {
        state.chatSessionId = data.session_id;
        store.set(STORAGE.chatSession, data.session_id);
      }
      await streamAssistantBubble(
        data.reply || data.message || data.response || data.answer || t("done")
      );
      appendToolResult(data);
    } catch (err) {
      if (scene && scene.isConnected) scene.remove();
      if (isAbortError(err)) {
        appendBubble("system", t("chatCancelled"));
        return;
      }
      appendBubble("system", `${t("visionFailed")} — ${err.message}`);
      toast(err.message, "error");
    } finally {
      if (state.chatAbort && state.chatAbort.signal === signal) state.chatAbort = null;
      setChatBusy(false);
    }
  }

  /* —— Growth —— */
  function syncGrowthSexFromChild() {
    const sel = $("#growth-child");
    const id = sel && sel.value;
    if (!id) return;
    const child = state.children.find((c) => childIdOf(c) === id);
    if (child && (child.sex === "male" || child.sex === "female")) {
      const radio = $(`#growth-form input[name="sex"][value="${child.sex}"]`);
      if (radio) radio.checked = true;
    }
  }

  async function fetchGrowthCurves(params, signal) {
    const q = new URLSearchParams();
    if (params.sex) q.set("sex", params.sex);
    if (params.measure) q.set("measure", params.measure);
    if (params.chart_standard) q.set("chart_standard", params.chart_standard);
    if (params.gestational_age_weeks != null && params.gestational_age_weeks !== "") {
      q.set("gestational_age_weeks", String(params.gestational_age_weeks));
    }
    if (params.age_max != null) q.set("age_max", String(params.age_max));
    try {
      const data = await api(`${EP.growthCurves}?${q.toString()}`, { signal });
      if (data && data.ok !== false && data.ages && data.curves) return data;
    } catch (_) {
      /* endpoint missing or failed — PNG overlay fallback keeps the view usable */
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

    const pad = CHART.padding;
    const W = CHART.width;
    const H = CHART.height;
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
    const yPad = (yMax - yMin) * CHART.yPadRatio || CHART.yPadFallback;
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
        // Percentile keys land in a class name — keep them numeric only.
        const safeKey = String(p).replace(/[^0-9]/g, "");
        return `<polyline class="curve-p${safeKey}" points="${pts}" />`;
      })
      .join("");

    let childMark = "";
    if (childPoint && Number.isFinite(childPoint.x) && Number.isFinite(childPoint.y)) {
      childMark = `<circle class="child-point" cx="${sx(childPoint.x).toFixed(1)}" cy="${sy(
        childPoint.y
      ).toFixed(1)}" r="${CHART.pointRadius}" />`;
    }

    const xLabel =
      curvesData.age_unit === "weeks" ? t("ageAxisWeeks") : t("ageAxisMonths");
    const yLabel =
      typeof curvesData.units === "string" && curvesData.units
        ? curvesData.units
        : t("valueUnit");
    const title = escapeHtml(t("percentileChart"));
    const axisY = H - pad.bottom;

    // Colors, weights and font sizes come from styles.css (.growth-svg-wrap …).
    return `<div class="growth-svg-wrap" role="img" aria-label="${title}">
      <svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <text class="gsvg-title" x="${pad.left}" y="${CHART.titleY}">${title}</text>
        <line class="gsvg-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${axisY}" />
        <line class="gsvg-axis" x1="${pad.left}" y1="${axisY}" x2="${W - pad.right}" y2="${axisY}" />
        ${polylines}
        ${childMark}
        <text class="gsvg-axis-label" x="${W / 2}" y="${
          H - CHART.axisLabelOffset
        }" text-anchor="middle">${escapeHtml(xLabel)}</text>
        <text class="gsvg-axis-label" x="${CHART.yAxisLabelX}" y="${
          H / 2
        }" text-anchor="middle" transform="rotate(-90 ${CHART.yAxisLabelX} ${
          H / 2
        })">${escapeHtml(yLabel)}</text>
      </svg>
    </div>`;
  }

  function growthBadgeClass(track) {
    if (/below_3|above_97|investigate|alert|high/i.test(track)) return "alert";
    if (/outer|monitor|warn/i.test(track)) return "warn";
    return "";
  }

  /** Unit label for the measured value, tolerating string or per-measure map. */
  function measureUnitLabel(data, body) {
    const measure = (data && data.measure) || (body && body.measure);
    const units = data && data.units;
    if (units && typeof units === "object") {
      const u = units[measure];
      if (typeof u === "string" && u) return u;
    } else if (typeof units === "string" && units) {
      return units;
    }
    return measureLabel(measure) || t("value");
  }

  /**
   * Renders (or re-renders, e.g. after a language switch) the growth result
   * from the cached payload — one block, replaced wholesale, so repeated
   * calculations can never stack duplicate charts.
   */
  function renderGrowthResult(cached) {
    const out = $("#growth-result");
    if (!out || !cached) return;
    const { data, body, curves } = cached;
    const centile = data.centile != null ? fmtNum(data.centile, 1) : "—";
    const z = data.z_score != null ? fmtNum(data.z_score, 2) : "—";
    const track = data.track_status || data.status || "";
    const badgeClass = growthBadgeClass(track);
    const img = overlaySrc(data);
    let svgHtml = "";
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
      ${
        track
          ? `<span class="badge ${badgeClass}">${escapeHtml(trackLabel(track))}</span>`
          : ""
      }
      <div class="stat-row">
        <div class="stat"><span class="val">${escapeHtml(centile)}</span><span class="lbl">${escapeHtml(
          t("centile")
        )}</span></div>
        <div class="stat"><span class="val">${escapeHtml(z)}</span><span class="lbl">${escapeHtml(
          t("zScore")
        )}</span></div>
        <div class="stat"><span class="val">${escapeHtml(
          fmtNum(data.value != null ? data.value : body.value, 2)
        )}</span><span class="lbl">${escapeHtml(measureUnitLabel(data, body))}</span></div>
      </div>
      ${data.summary ? `<p>${escapeHtml(data.summary)}</p>` : ""}
      ${
        svgHtml ||
        (img
          ? `<img class="overlay-img" data-authed-src="${escapeHtml(img)}" alt="${escapeHtml(
              t("growthChartAlt")
            )}" loading="lazy" />`
          : "")
      }
      ${!svgHtml && !img ? `<p class="muted">${escapeHtml(t("curveLoadFailed"))}</p>` : ""}
    `;
    // src is set after insertion so the request can carry auth headers.
    hydrateAuthedImages(out);
    out.hidden = false;
  }

  function wireGrowth() {
    const childSel = $("#growth-child");
    const growthForm = $("#growth-form");
    if (!childSel || !growthForm) return;
    childSel.addEventListener("change", (e) => {
      const id = e.target.value;
      if (id) {
        const child = state.children.find((c) => childIdOf(c) === id);
        if (child) saveActiveChild(child);
      }
      syncGrowthSexFromChild();
    });

    growthForm.addEventListener("submit", async (e) => {
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
        const child = state.children.find((c) => childIdOf(c) === body.child_id);
        if (child && child.gestational_age_weeks != null) {
          body.gestational_age_weeks = Number(child.gestational_age_weeks);
        }
      }
      if (!body.child_id) delete body.child_id;
      if (body.weeks == null && body.age_months == null) {
        toast(t("enterAgeError"), "error");
        return;
      }
      setLoading(btn, true);
      const out = $("#growth-result");
      if (out) {
        out.hidden = false;
        out.setAttribute("aria-busy", "true");
        renderMessage(out, t("loading"));
      }
      try {
        const data = await api(EP.growth, { method: "POST", body });
        const usesMonths =
          data.chart_standard === "who_term" || data.age_months != null;
        const curves = await fetchGrowthCurves({
          sex: data.sex || body.sex,
          measure: data.measure || body.measure,
          chart_standard: data.chart_standard,
          gestational_age_weeks:
            data.gestational_age_weeks != null
              ? data.gestational_age_weeks
              : body.gestational_age_weeks,
          age_max: usesMonths
            ? Math.max(
                LIMITS.growthMonthsMax,
                Number(
                  data.age_months || body.age_months || LIMITS.growthCurveFallbackMonths
                ) + LIMITS.growthCurveMonthsPadding
              )
            : Math.max(
                LIMITS.growthWeeksMax,
                Number(data.weeks || body.weeks || LIMITS.fullTermWeeks) +
                  LIMITS.growthCurveWeeksPadding
              ),
        });
        state.lastGrowth = { data, body, curves };
        renderGrowthResult(state.lastGrowth);
        if (out) {
          out.scrollIntoView({
            behavior: prefersReducedMotion() ? "auto" : "smooth",
            block: "nearest",
          });
        }
      } catch (err) {
        state.lastGrowth = null;
        if (!isAbortError(err) && out) {
          renderMessage(out, err.message, {
            retry: () => growthForm.requestSubmit(),
          });
        }
        toast(err.message, "error");
      } finally {
        if (out) out.removeAttribute("aria-busy");
        setLoading(btn, false);
      }
    });
  }

  /* —— Screening —— */
  function syncScreeningAgeFromChild() {
    const input = $("#screening-age-months");
    const child = state.activeChild;
    if (!input) return;
    if (child && child.date_of_birth) {
      const chrono = ageMonthsFromDob(child.date_of_birth);
      const screening = correctedAgeMonths(chrono, child.gestational_age_weeks);
      if (screening != null) {
        input.value = String(screening);
        state.screeningAgeMonths = screening;
      }
    }
  }

  function showScreeningPicker() {
    state.quiz = null;
    clearQuizAdvanceTimer();
    const picker = $("#screening-picker");
    const quiz = $("#screening-quiz");
    const report = $("#screening-report");
    if (picker) picker.hidden = false;
    if (quiz) quiz.hidden = true;
    if (report) report.hidden = true;
    state.lastReport = null;
    setI18nText($("#screening-title"), "screeningTitle");
    setI18nText($("#screening-sub"), "screeningSub");
    const back = $("#screening-back");
    if (back) {
      back.href = "#/";
      setI18nText(back, "home");
    }
    refreshScreeningPicker();
  }

  function refreshScreeningPicker() {
    const ageInfo = resolveScreeningAge();
    const badge = $("#screening-age-badge");
    const age = ageInfo.screening;
    if (badge) {
      if (age != null) {
        badge.hidden = false;
        let text = t("ageKnownBadge", { age: fmtNum(age) });
        if (
          ageInfo.source === "dob" &&
          ageInfo.chrono != null &&
          Math.abs(ageInfo.chrono - age) >= LIMITS.correctedAgeNoticeMonths
        ) {
          text += " · " + t("correctedAgeNote", { age: fmtNum(age) });
        }
        badge.textContent = text;
      } else {
        badge.hidden = true;
        badge.textContent = "";
      }
    }
    renderRelevantTests(age);
    if (state.screeningHistoryOpen) renderScreeningHistory();
    const histBtn = $("#btn-prev-results");
    if (histBtn) {
      setI18nText(
        histBtn,
        state.screeningHistoryOpen ? "hidePreviousResults" : "previousResults"
      );
      histBtn.setAttribute("aria-expanded", String(state.screeningHistoryOpen));
    }
  }

  /** One test card; built as DOM nodes so listeners die with the node. */
  function buildTestCard({ kind, badgeKey, title, sub, onClick, mutedBadge }) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `test-card ${kind}`;
    const badge = document.createElement("span");
    badge.className = `test-card-badge${mutedBadge ? " muted-badge" : ""}`;
    badge.textContent = t(badgeKey);
    const titleEl = document.createElement("span");
    titleEl.className = "test-card-title";
    titleEl.textContent = title;
    const subEl = document.createElement("span");
    subEl.className = "test-card-sub";
    subEl.textContent = sub;
    btn.append(badge, titleEl, subEl);
    btn.addEventListener("click", onClick);
    return btn;
  }

  function renderRelevantTests(ageMonths) {
    const grid = $("#asq-ages");
    const mchatSlot = $("#mchat-slot");
    if (!grid) return;

    if (ageMonths == null || Number.isNaN(Number(ageMonths))) {
      renderMessage(grid, t("enterAgeForTests"));
      if (mchatSlot) mchatSlot.textContent = "";
      return;
    }

    const { current, upcoming } = relevantAsqAges(ageMonths);
    const cards = [];
    current.forEach((m) => {
      cards.push(
        buildTestCard({
          kind: "current",
          badgeKey: "testCurrent",
          title: t("startAsqAge", { age: fmtNum(m, 0) }),
          sub: t("asqTitle"),
          onClick: () => startAsq(m),
        })
      );
    });
    upcoming.forEach((m) => {
      if (current.includes(m)) return;
      cards.push(
        buildTestCard({
          kind: "upcoming",
          badgeKey: "testUpcoming",
          mutedBadge: true,
          title: t("startAsqAge", { age: fmtNum(m, 0) }),
          sub: t("asqTitle"),
          onClick: () => startAsq(m),
        })
      );
    });

    // Rebuilt from scratch: repeated age edits can't duplicate cards.
    if (!cards.length) {
      renderMessage(grid, t("noRelevantTests"));
    } else {
      grid.textContent = "";
      cards.forEach((c) => grid.appendChild(c));
    }

    if (mchatSlot) {
      mchatSlot.textContent = "";
      const age = Number(ageMonths);
      const inMchatWindow =
        age >= LIMITS.mchatAgeMinMonths && age <= LIMITS.mchatAgeMaxMonths;
      if (inMchatWindow) {
        mchatSlot.appendChild(
          buildTestCard({
            kind: "current",
            badgeKey: "testCurrent",
            title: t("startMchat"),
            sub: `${t("mchatTitle")} · ${t("mchatWindow", {
              min: fmtNum(LIMITS.mchatAgeMinMonths, 0),
              max: fmtNum(LIMITS.mchatAgeMaxMonths, 0),
            })}`,
            onClick: () => startMchat(),
          })
        );
      }
    }
  }

  async function renderScreeningHistory() {
    const panel = $("#screening-history");
    const body = $("#screening-history-body");
    if (!panel || !body) return;
    panel.hidden = false;
    const sel = $("#screening-child");
    const childId = (sel && sel.value) || activeChildId();
    if (!childId) {
      renderMessage(body, t("selectChildForHistory"));
      return;
    }
    renderMessage(body, t("loading"));
    panel.setAttribute("aria-busy", "true");
    try {
      const data = await api(EP.childDossier(childId));
      const screens = data.screenings || [];
      if (!screens.length) {
        renderMessage(body, t("noHistoryYet"));
        return;
      }
      body.innerHTML = `<ul class="history-list">${screens
        .slice()
        .reverse()
        .map((s) => {
          const when = formatTimestamp(s.recorded_at);
          const summary =
            (s.result && (s.result.summary || s.result.risk || s.result.parent_report)) ||
            "";
          const ageValue =
            s.result && s.result.age_months != null
              ? s.result.age_months
              : s.age_months != null
                ? s.age_months
                : null;
          const ageBit =
            ageValue != null
              ? ` · ${t("monthsSuffix", { n: fmtNum(ageValue) })}`
              : "";
          return `<li>
            <div class="history-row">
              <strong>${escapeHtml(
                s.instrument || t("screeningFallback")
              )}${escapeHtml(ageBit)}</strong>
              <span class="meta">${escapeHtml(when)}</span>
            </div>
            <p>${escapeHtml(String(summary || t("done")))}</p>
          </li>`;
        })
        .join("")}</ul>`;
    } catch (err) {
      if (isAbortError(err)) return;
      renderMessage(body, `${t("historyLoadFailed")} ${err.message}`, {
        retry: () => renderScreeningHistory(),
      });
    } finally {
      panel.removeAttribute("aria-busy");
    }
  }

  function localizedText(obj) {
    if (!obj) return "";
    if (state.lang === "fa" && obj.text_fa) return obj.text_fa;
    return obj.text_en || obj.text || obj.question || obj.text_fa || "";
  }

  function domainTitleOf(dom) {
    if (state.lang === "fa" && dom.title_fa) return dom.title_fa;
    return dom.title_en || domainLabel(dom.id);
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
            domainTitle: domainTitleOf(dom),
            id: q.id,
            text: localizedText(q),
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
    const mchatTitle = t("mchatTitle");
    return (Array.isArray(qs) ? qs : []).map((q) => ({
      domain: "mchat",
      domainTitle: mchatTitle,
      id: q.id,
      text: localizedText(q),
      options: [
        { value: "yes", label: t("yes"), cls: "yes" },
        { value: "no", label: t("no"), cls: "no" },
      ],
    }));
  }

  /**
   * Re-derive question/answer labels in the new language from the cached API
   * payload, keeping answers (keyed by domain::id, which is language-neutral).
   */
  function relabelQuizForLang() {
    const q = state.quiz;
    if (!q || !q.payload) return;
    q.items = normalizeQuestions(q.payload, q.kind);
    if (q.kind === "asq") {
      setI18nText($("#screening-title"), "asqMonths", { age: fmtNum(q.age, 0) });
      setI18nText($("#screening-sub"), "asqAnswerHint");
    } else {
      setI18nText($("#screening-title"), "mchatTitle");
      setI18nText($("#screening-sub"), "mchatAnswerHint");
    }
    const back = $("#screening-back");
    if (back) setI18nText(back, "questionnaires");
    if (q.index >= q.items.length) q.index = Math.max(0, q.items.length - 1);
    renderQuizStep();
  }

  async function startAsq(age) {
    const grid = $("#asq-ages");
    try {
      toast(t("loadingAsq", { age: fmtNum(age, 0) }));
      const data = await api(EP.asqQuestions(age));
      const items = normalizeQuestions(data, "asq");
      if (!items.length) throw new Error(t("noAsqQuestions"));
      state.quiz = {
        kind: "asq",
        age,
        items,
        payload: data,
        index: 0,
        answers: {},
      };
      setI18nText($("#screening-title"), "asqMonths", { age: fmtNum(age, 0) });
      setI18nText($("#screening-sub"), "asqAnswerHint");
      beginQuiz();
    } catch (err) {
      if (isAbortError(err)) return;
      toast(err.message, "error");
      if (grid) {
        renderMessage(grid, err.message, { retry: () => startAsq(age) });
      }
    }
  }

  async function startMchat() {
    const slot = $("#mchat-slot");
    try {
      toast(t("loadingMchat"));
      const data = await api(EP.mchatQuestions);
      const items = normalizeQuestions(data, "mchat");
      if (!items.length) throw new Error(t("noMchatQuestions"));
      state.quiz = {
        kind: "mchat",
        items,
        payload: data,
        index: 0,
        answers: {},
      };
      setI18nText($("#screening-title"), "mchatTitle");
      setI18nText($("#screening-sub"), "mchatAnswerHint");
      beginQuiz();
    } catch (err) {
      if (isAbortError(err)) return;
      toast(err.message, "error");
      if (slot) {
        renderMessage(slot, err.message, { retry: () => startMchat() });
      }
    }
  }

  function beginQuiz() {
    clearQuizAdvanceTimer();
    const picker = $("#screening-picker");
    const report = $("#screening-report");
    const quiz = $("#screening-quiz");
    if (picker) picker.hidden = true;
    if (report) report.hidden = true;
    if (quiz) quiz.hidden = false;
    const back = $("#screening-back");
    if (back) {
      back.href = "#/screening";
      setI18nText(back, "questionnaires");
    }
    renderQuizStep();
  }

  function clearQuizAdvanceTimer() {
    if (state.quizAdvanceTimer) {
      window.clearTimeout(state.quizAdvanceTimer);
      state.quizAdvanceTimer = null;
    }
  }

  function renderQuizStep() {
    const q = state.quiz;
    if (!q) return;
    // A pending auto-advance from the previous answer must never fire into this step.
    clearQuizAdvanceTimer();
    const item = q.items[q.index];
    if (!item) return;
    const total = q.items.length;
    const pct = total ? (q.index / total) * 100 : 0;
    const fill = $("#quiz-progress");
    const bar = $("#quiz-progress-bar");
    if (fill) fill.style.width = `${pct}%`;
    if (bar) bar.setAttribute("aria-valuenow", String(Math.round(pct)));
    const meta = $("#quiz-meta");
    if (meta) {
      meta.textContent = t("questionOf", {
        n: fmtNum(q.index + 1, 0),
        total: fmtNum(total, 0),
      });
    }
    const domainEl = $("#quiz-domain");
    if (domainEl) domainEl.textContent = item.domainTitle || domainLabel(item.domain);
    const questionEl = $("#quiz-question");
    if (questionEl) questionEl.textContent = item.text;
    const row = $("#quiz-answers");
    const nextBtn = $("#quiz-next");
    const prevBtn = $("#quiz-prev");
    const key = answerKey(item);
    const current = q.answers[key];
    if (row) {
      // Fresh buttons each step, so click handlers can't accumulate.
      row.textContent = "";
      item.options.forEach((o) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `ans-btn ${o.cls}${current === o.value ? " selected" : ""}`;
        btn.dataset.value = o.value;
        btn.textContent = o.label;
        btn.setAttribute("aria-pressed", String(current === o.value));
        btn.addEventListener("click", () => {
          q.answers[key] = o.value;
          $$(".ans-btn", row).forEach((b) => {
            const selected = b === btn;
            b.classList.toggle("selected", selected);
            b.setAttribute("aria-pressed", String(selected));
          });
          if (nextBtn) nextBtn.disabled = false;
          clearQuizAdvanceTimer();
          state.quizAdvanceTimer = window.setTimeout(() => {
            state.quizAdvanceTimer = null;
            if (state.quiz !== q) return;
            if (q.index < q.items.length - 1) {
              q.index += 1;
              renderQuizStep();
            } else if (nextBtn) {
              setI18nText(nextBtn, "seeResults");
            }
          }, TIME.quizAdvanceMs);
        });
        row.appendChild(btn);
      });
    }
    if (prevBtn) prevBtn.disabled = q.index === 0;
    if (nextBtn) {
      nextBtn.disabled = !current;
      setI18nText(nextBtn, q.index === q.items.length - 1 ? "seeResults" : "next");
    }
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
    const ageInput = $("#screening-age-months");
    if (ageInput) {
      // Debounced: typing "12" shouldn't rebuild the card list twice per keystroke.
      ageInput.addEventListener("input", () => {
        const v = ageInput.value === "" ? null : Number(ageInput.value);
        state.screeningAgeMonths = v != null && !Number.isNaN(v) ? v : null;
        if (state.ageInputTimer) window.clearTimeout(state.ageInputTimer);
        state.ageInputTimer = window.setTimeout(() => {
          state.ageInputTimer = null;
          refreshScreeningPicker();
        }, TIME.ageInputDebounceMs);
      });
    }
    const histBtn = $("#btn-prev-results");
    if (histBtn) {
      histBtn.addEventListener("click", () => {
        state.screeningHistoryOpen = !state.screeningHistoryOpen;
        const panel = $("#screening-history");
        setI18nText(
          histBtn,
          state.screeningHistoryOpen ? "hidePreviousResults" : "previousResults"
        );
        histBtn.setAttribute("aria-expanded", String(state.screeningHistoryOpen));
        if (!state.screeningHistoryOpen) {
          if (panel) panel.hidden = true;
          return;
        }
        renderScreeningHistory();
      });
    }
    const childSel = $("#screening-child");
    if (childSel) {
      childSel.addEventListener("change", (e) => {
        const id = e.target.value;
        if (id) {
          const child = state.children.find((c) => childIdOf(c) === id);
          if (child) {
            saveActiveChild(child);
            syncScreeningAgeFromChild();
          }
        }
        refreshScreeningPicker();
        if (state.screeningHistoryOpen) renderScreeningHistory();
      });
    }
  }

  async function submitQuiz() {
    const q = state.quiz;
    if (!q) return;
    const btn = $("#quiz-next");
    setLoading(btn, true);
    try {
      const sel = $("#screening-child");
      const childId = (sel && sel.value) || activeChildId() || undefined;

      let data;
      if (q.kind === "asq") {
        const domain_answers = {};
        q.items.forEach((item) => {
          const ans = q.answers[answerKey(item)];
          if (!ans) return;
          if (!domain_answers[item.domain]) domain_answers[item.domain] = [];
          domain_answers[item.domain].push(ans);
        });
        data = await api(EP.asqScore, {
          method: "POST",
          body: {
            child_id: childId,
            age_months: q.age,
            domain_answers,
          },
        });
        state.lastReport = { kind: "asq", data };
      } else {
        const answers = {};
        q.items.forEach((item) => {
          answers[item.id] = q.answers[answerKey(item)];
        });
        data = await api(EP.mchatScore, {
          method: "POST",
          body: { child_id: childId, answers },
        });
        state.lastReport = { kind: "mchat", data };
      }
      renderScreeningReport(state.lastReport);
      const quiz = $("#screening-quiz");
      if (quiz) quiz.hidden = true;
      const fill = $("#quiz-progress");
      const bar = $("#quiz-progress-bar");
      if (fill) fill.style.width = "100%";
      if (bar) bar.setAttribute("aria-valuenow", "100");
      // Quiz is finished: returning to Tests should show the picker, not a stale quiz.
      state.quiz = null;
      clearQuizAdvanceTimer();
    } catch (err) {
      if (!isAbortError(err)) toast(err.message, "error");
    } finally {
      setLoading(btn, false);
    }
  }

  function renderScreeningReport(cached) {
    if (!cached) return;
    if (cached.kind === "asq") renderAsqReport(cached.data);
    else renderMchatReport(cached.data);
  }

  function wireScreeningAgain(report) {
    const again = $("#screening-again", report);
    if (again) again.addEventListener("click", showScreeningPicker);
  }

  function renderAsqReport(data) {
    const report = $("#screening-report");
    if (!report) return;
    const result = data.result || data;
    const domains = result.domains || data.domains || {};
    const needs = result.needs_referral != null ? result.needs_referral : data.needs_referral;
    const summary = data.parent_report || result.summary || data.summary || "";
    const cutoffSource = result.cutoff_source || data.cutoff_source || "";
    const badge = needs ? "alert" : "";
    const rows = Object.entries(domains)
      .map(([id, d]) => {
        const below = d && d.below_cutoff;
        const total = d && d.total != null ? fmtNum(d.total, 0) : "—";
        const cut =
          d && d.cutoff != null ? ` / ${t("cutoffShort", { n: fmtNum(d.cutoff, 0) })}` : "";
        return `<li><span>${escapeHtml(domainLabel(id))}</span>
          <span class="badge ${below ? "alert" : ""}">${escapeHtml(total)}${escapeHtml(
            cut
          )}</span></li>`;
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
      <p>${escapeHtml(String(summary))}</p>
      ${cutoffNote}
      <ul class="domain-list">${rows}</ul>
      <button type="button" class="btn btn-secondary btn-block report-again" id="screening-again">${escapeHtml(
        t("anotherQuestionnaire")
      )}</button>
    `;
    report.hidden = false;
    wireScreeningAgain(report);
  }

  function renderMchatReport(data) {
    const report = $("#screening-report");
    if (!report) return;
    const result = data.result || data;
    const risk = String(result.risk || data.risk || "").toLowerCase();
    const badge = risk === "high" ? "alert" : risk === "medium" ? "warn" : "";
    const summary = data.parent_report || result.summary || data.summary || "";
    const note = result.note || data.note || "";
    const failed =
      result.total_failed != null
        ? result.total_failed
        : data.total_failed != null
          ? data.total_failed
          : null;
    report.innerHTML = `
      <h3>${escapeHtml(t("mchatReport"))}</h3>
      <span class="badge ${badge}">${escapeHtml(t("riskLabel", { risk: risk || "—" }))}</span>
      <div class="stat-row">
        <div class="stat"><span class="val">${escapeHtml(
          failed != null ? fmtNum(failed, 0) : "—"
        )}</span><span class="lbl">${escapeHtml(t("failedItems"))}</span></div>
      </div>
      <p>${escapeHtml(String(summary))}</p>
      ${note ? `<p class="muted">${escapeHtml(String(note))}</p>` : ""}
      <button type="button" class="btn btn-secondary btn-block report-again" id="screening-again">${escapeHtml(
        t("anotherQuestionnaire")
      )}</button>
    `;
    report.hidden = false;
    wireScreeningAgain(report);
  }

  /* —— Field constraints (single source: config.js) —— */
  const FIELD_CONSTRAINTS = {
    childName: { maxLength: LIMITS.childNameMaxChars },
    chatMessage: { maxLength: LIMITS.chatMessageMaxChars },
    gaWeeks: {
      min: LIMITS.gaWeeksMin,
      max: LIMITS.gaWeeksMax,
      step: LIMITS.gaStepWeeks,
    },
    growthWeeks: {
      min: LIMITS.growthWeeksMin,
      max: LIMITS.growthWeeksMax,
      step: LIMITS.ageStepMonths,
    },
    growthMonths: {
      min: LIMITS.growthMonthsMin,
      max: LIMITS.growthMonthsMax,
      step: LIMITS.ageStepMonths,
    },
    growthValue: { min: LIMITS.growthValueMin, step: LIMITS.growthValueStep },
    screeningAge: {
      min: LIMITS.screeningAgeMonthsMin,
      max: LIMITS.screeningAgeMonthsMax,
      step: LIMITS.ageStepMonths,
    },
  };

  function applyFieldConstraints() {
    $$("[data-constraint]").forEach((el) => {
      const spec = FIELD_CONSTRAINTS[el.getAttribute("data-constraint")];
      if (!spec) return;
      if (spec.min != null) el.min = String(spec.min);
      if (spec.max != null) el.max = String(spec.max);
      if (spec.step != null) el.step = String(spec.step);
      if (spec.maxLength != null) el.maxLength = spec.maxLength;
    });
    const file = $("#chat-file");
    if (file) file.accept = LIMITS.acceptedImageTypes.join(",");
  }

  /* —— Boot —— */
  async function boot() {
    state.activeChild = normalizeChild(loadJson(STORAGE.activeChild, null));
    applyI18n();
    applyFieldConstraints();
    const langBtn = $("#lang-toggle");
    if (langBtn) {
      langBtn.addEventListener("click", () => setLang(state.lang === "fa" ? "en" : "fa"));
    }
    renderChildChip();
    initAccountUi();
    wireChildForm();
    wireChat();
    wireGrowth();
    wireQuizNav();
    window.addEventListener("hashchange", navigate);
    // Leaving the page shouldn't leave a stream half-read.
    window.addEventListener("pagehide", () => abortChat());
    if (!location.hash) location.hash = "#/";
    navigate();
    checkHealth();
    try {
      const data = await api(EP.children);
      state.children = Array.isArray(data) ? data : data.children || [];
      fillChildSelects();
    } catch (_) {
      /* offline is fine — views show their own error/empty states */
    }
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
