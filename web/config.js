/**
 * Nestling frontend configuration — the single source for API location,
 * endpoint paths, timings and domain limits. Nothing host- or port-specific
 * is hardcoded: the API base is derived from the document location, so the
 * same bundle works on localhost, in Docker, or behind a reverse proxy that
 * mounts the app under a sub-path.
 *
 * Override the base (e.g. API on another origin) without touching JS:
 *   <meta name="nestling-api-base" content="https://api.example.org/api" />
 */
(function () {
  "use strict";

  function resolveApiBase() {
    const meta = document.querySelector('meta[name="nestling-api-base"]');
    const configured = meta && meta.getAttribute("content");
    const candidate = configured && configured.trim() ? configured.trim() : "api/";
    try {
      return new URL(candidate, document.baseURI).href.replace(/\/+$/, "");
    } catch (_) {
      return "/api";
    }
  }

  /**
   * API key for deployments that set NESTLING_API_KEY. Read from (in order):
   *   1. <meta name="nestling-api-key" content="..."> — server-rendered
   *   2. localStorage "nestling_api_key" — entered once by the operator
   * Returns "" when unset, in which case no auth header is sent and an
   * unauthenticated backend behaves exactly as before.
   */
  function resolveApiKey() {
    const meta = document.querySelector('meta[name="nestling-api-key"]');
    const fromMeta = meta && meta.getAttribute("content");
    if (fromMeta && fromMeta.trim()) return fromMeta.trim();
    try {
      return (
        window.localStorage.getItem("nestling_session_token") ||
        window.localStorage.getItem("nestling_api_key") ||
        ""
      ).trim();
    } catch (_) {
      // Private mode / blocked storage — treat as no key rather than throwing.
      return "";
    }
  }

  const ASQ_AGES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 27, 30, 33, 36, 42, 48, 54, 60];

  window.NESTLING_CONFIG = {
    api: {
      base: resolveApiBase(),
      /** Empty string when the backend runs unauthenticated. */
      key: resolveApiKey(),
      /** Persist a key entered by the operator (used on a 401). */
      setKey(value) {
        const v = (value || "").trim();
        try {
          if (v) window.localStorage.setItem("nestling_api_key", v);
          else window.localStorage.removeItem("nestling_api_key");
        } catch (_) {
          /* storage unavailable — key stays in memory only */
        }
        this.key = v;
        return v;
      },
      /** Display name of the signed-in account, "" when unauthenticated. */
      username: (function () {
        try {
          return window.localStorage.getItem("nestling_username") || "";
        } catch (_) {
          return "";
        }
      })(),
      /** Store a session token (and optionally the username) from /auth/login. */
      setToken(token, username) {
        const v = (token || "").trim();
        try {
          if (v) window.localStorage.setItem("nestling_session_token", v);
          else window.localStorage.removeItem("nestling_session_token");
          if (username) window.localStorage.setItem("nestling_username", username);
          else if (!v) window.localStorage.removeItem("nestling_username");
        } catch (_) {
          /* storage unavailable — token stays in memory only */
        }
        this.key = v;
        if (username) this.username = username;
        else if (!v) this.username = "";
        return v;
      },
      /**
       * Auth headers for every request; {} when unauthenticated.
       * Sent as both a bearer token and X-API-Key so the same value works
       * whether it is a login session token or a shared API key.
       */
      authHeaders() {
        if (!this.key) return {};
        return { Authorization: `Bearer ${this.key}`, "X-API-Key": this.key };
      },
      /** Every API path lives here — no '/api/...' literals elsewhere. */
      paths: {
        health: "/health",
        ready: "/ready",
        children: "/children",
        childDossier: (id) => `/children/${encodeURIComponent(id)}/dossier`,
        sessions: "/sessions",
        session: (id) => `/sessions/${encodeURIComponent(id)}`,
        chat: "/chat",
        chatStream: "/chat/stream",
        chatVision: "/chat/vision",
        growth: "/growth",
        growthCurves: "/growth/curves",
        asqQuestions: (age) => `/asq/${encodeURIComponent(age)}/questions`,
        asqScore: "/asq/score",
        mchatQuestions: "/mchat/questions",
        mchatScore: "/mchat/score",
        overlay: (filename) => `/overlays/${encodeURIComponent(filename)}`,
      },
    },

    storageKeys: {
      activeChild: "nestling_active_child",
      chatSession: "nestling_chat_session",
      lang: "nestling_lang",
    },

    timing: {
      requestTimeoutMs: 20000,
      chatRequestTimeoutMs: 180000,
      streamIdleTimeoutMs: 60000,
      visionTimeoutMs: 180000,
      toastVisibleMs: 3200,
      toastFadeMs: 350,
      streamTickMs: 14,
      revealTickFastMs: 8,
      revealTickMediumMs: 12,
      revealTickSlowMs: 16,
      quizAdvanceMs: 220,
      ageInputDebounceMs: 250,
    },

    limits: {
      /** Weeks of gestation accepted by the add-child form. */
      gaWeeksMin: 22,
      gaWeeksMax: 45,
      gaStepWeeks: 0.5,
      /** Clinical boundaries: < 37w is preterm, 40w is full term reference. */
      pretermWeeks: 37,
      fullTermWeeks: 40,
      weeksPerMonth: 4.345,
      daysPerMonth: 30.4375,
      growthWeeksMin: 0,
      growthWeeksMax: 64,
      growthMonthsMin: 0,
      growthMonthsMax: 24,
      growthValueMin: 0.1,
      growthValueStep: 0.01,
      ageStepMonths: 0.5,
      screeningAgeMonthsMin: 0,
      screeningAgeMonthsMax: 72,
      childNameMaxChars: 80,
      chatMessageMaxChars: 2000,
      chatHistoryLimit: 30,
      chatHistoryTitleMaxChars: 80,
      childChipsMax: 12,
      /** The children API is unpaginated; keep the pickers usable regardless. */
      childSelectMax: 50,
      dossierGrowthRows: 3,
      dossierChartsMax: 3,
      timestampChars: 16,
      composerMaxHeightPx: 120,
      /** Mirrors the API's NESTLING_MAX_UPLOAD_BYTES ceiling. */
      maxUploadBytes: 8000000,
      bytesPerMegabyte: 1000000,
      acceptedImageTypes: ["image/png", "image/jpeg", "image/webp", "image/gif"],
      /** ASQ intervals + M-CHAT-R validated window (months). */
      asqAges: ASQ_AGES,
      asqRecencyMonths: 1.25,
      mchatAgeMinMonths: 16,
      mchatAgeMaxMonths: 30,
      /** Only flag corrected age when it differs from chronological by this much. */
      correctedAgeNoticeMonths: 0.3,
      growthCurveMonthsPadding: 2,
      growthCurveWeeksPadding: 4,
      growthCurveFallbackMonths: 12,
    },

    /** Streamed-text reveal pacing thresholds (characters). */
    reveal: {
      longTextChars: 900,
      mediumTextChars: 400,
      shortTextChars: 160,
      queueLargeChars: 40,
      queueMediumChars: 16,
      charsPerTickMax: 5,
      charsPerTickMedium: 3,
      charsPerTickSmall: 2,
      charsPerTickMin: 1,
    },

    /** Percentile chart geometry (colors live in styles.css tokens). */
    chart: {
      width: 640,
      height: 360,
      padding: { top: 28, right: 16, bottom: 36, left: 44 },
      yPadRatio: 0.08,
      yPadFallback: 0.1,
      pointRadius: 6,
      titleY: 18,
      axisLabelOffset: 8,
      yAxisLabelX: 12,
    },
  };
})();
