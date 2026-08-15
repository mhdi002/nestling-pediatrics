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

  const ASQ_AGES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 27, 30, 33, 36, 42, 48, 54, 60];

  window.NESTLING_CONFIG = {
    api: {
      base: resolveApiBase(),
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
