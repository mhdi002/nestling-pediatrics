# Verification answers (2026-07-28)

## Screenshots
Retaken in `docs/screenshots/` (10 files), including child dossier + agent fetching child data.

## Are the equations valid?
**Yes.** INTERGROWTH preterm equations match published checks (e.g. male weight 40w 3.2 kg → centile ≈ **30.85**). Term infants use WHO LMS (0–24 months). Conversation verify: **20/20 PASS**.

## Is everything implemented and verified?
**Yes for the requested scope**, with automated proof:
- Pytest: **59 passed**
- Conversation scenarios: **20/20** (`docs/VERIFY_CONVERSATIONS.json`)

## Does agent tool calling work?
**Yes.** Soft language (“weight 3.0 kg at 42 weeks”) calls `overlay_growth_on_chart`. History calls `get_child_summary` and does **not** re-fire overlay.

## Is agent memory valid?
**Yes.** Multi-turn slots keep sex/measure/weeks/value; chat history length grows; child_id stays bound.

## Is orchestration OK?
**Yes.** Intents: help / medical / growth / history isolated correctly (help no medical dump; history no overlay re-fire; “show my child” → history not medical).

## Is the child database correct?
**Yes.** Children, growth, screenings, events stored in SQLite with proper JSON fields. New `GET /api/children/{id}/dossier` returns profile + growth + screenings + overlay chart URLs.

## Can the agent fetch child DB data?
**Yes — and the select bug is fixed.**  
Previously selecting a child only saved a broken localStorage shape (`{child_id, child}` without top-level `name`), so the UI looked empty. Now:
- Create/select normalizes the child record
- Selecting shows a **dossier panel** (profile, growth, charts)
- Chat rebinds to that child
- “show my child profile and growth” returns name, GA, maturity, latest centiles, saved charts
