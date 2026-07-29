# English Multi-Turn Story Verification

**Date:** 2026-07-28  
**API:** `http://127.0.0.1:8015`  
**Config:** `NESTLING_LOAD_MODELS=0`  
**Child:** term male, GA 39 weeks (`VerifyEN`)

## Setup

| Step | Result | Notes |
|------|--------|-------|
| `POST /api/children` (GA 39, male) | PASS | Term child created |
| `POST /api/sessions` (bound to child) | PASS | Single session for all turns |
| `GET /api/sessions?limit=10` | PASS | Returns 10 sessions |

## Chat turns (one session)

| # | User message | Result | Intents | Tools | Key checks |
|---|--------------|--------|---------|-------|------------|
| 1 | hi | **PASS** | `help` | — | Nestling greeting, capability list |
| 2 | show my child profile | **PASS** | `history` | `get_child_summary` | Profile summary; no chart overlay; no RAG dump |
| 3 | boy weight 40 weeks 3.2 kg | **PASS** | `growth` | `overlay_growth_on_chart` | WHO term chart; age ≈0.23 mo (~0.2 m); centile ≈20.2 (>10); overlay PNG generated |
| 4 | show my child chart | **PASS** | `growth` | `overlay_growth_on_chart` | Re-plots overlay from saved measurement; **no** `history` intent |
| 5 | when will my son talk? | **PASS** | `medical`, `screening` | — | Medical RAG on speech; **no** `overlay_growth_on_chart` |
| 6 | so its okey now | **PASS** | `reassure` | — | Concrete reassurance (not generic “I hear you”) |
| 7 | tell me about iron | **PASS** | `medical` | — | Iron guidance from RAG; no chart tools |
| 8 | what was my child's last growth result? | **PASS** | `history` | `get_child_summary` | Recalls **3.2 kg**; growth points = **1** |

## Summary

**Overall: PASS (8/8 turns + sessions API)**

Raw transcript: `docs/_verify_en_story_raw.json`  
Re-run: `python docs/_verify_en_story.py`

## Bug found and fixed

**Duplicate growth records on chart replay**

- **Symptom:** After “boy weight 40 weeks 3.2 kg” then “show my child chart”, the dossier showed **2** growth points (same 3.2 kg stored twice).
- **Cause:** `ParentAssistant.chat()` persisted growth to SQLite on every successful `overlay_growth_on_chart` call, including re-plots that only hydrate slots from history.
- **Fix:** Only call `db.add_growth()` when the **current turn** includes a new measurement (`value` in turn-local slot extraction). Chart-only requests re-plot without duplicating DB rows.
- **Files:** `assistant/agent/orchestrator.py`, `tests/test_intent_routing.py` (assertion that replay keeps 1 growth point)

## Notes

- Speech turn carries both `medical` and `screening` intents (developmental concern routing); no growth tools fire — acceptable.
- Overlay filename pattern: `overlay_{child_id}_weight_0.2m.png`
