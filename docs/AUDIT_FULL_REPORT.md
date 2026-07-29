# Nestling — Full Accuracy & UI Audit

**Date:** 2026-07-28  
**Verdict: PASS** — equations, scoring, agent routing, translation, API, and UI workflow verified.

## Honest answer to “did you save UI screenshots before?”

Earlier proofs had **agent transcripts + chart overlays**, but **not** a full Nestling SPA walkthrough.  
That gap is closed now: **8 real browser screenshots** under `docs/screenshots/`.

## Critical bug found & fixed

| Bug | Impact | Fix |
|-----|--------|-----|
| `POST /api/chat` required `session_id` | UI first message sent without session → **HTTP 422** → chat never replied | `session_id` optional; API **auto-creates** session; UI calls `/api/sessions` on chat open |
| Tool results shape `{tool_calls:[…]}` | Overlay image often not rendered in chat | `appendToolResult` unwraps `tools.tool_calls` |

Regression test: `test_chat_auto_creates_session` — included in **50 passed**.

## Automated results

| Suite | Result |
|-------|--------|
| Pytest | **50 passed**, 1 Starlette/httpx warning |
| Accuracy audit | **19/19 PASS** → `docs/AUDIT_ACCURACY.json` |

### Accuracy highlights

- INTERGROWTH published: male 27w 0.99kg ≈ **96.89**; female 27w 0.91kg ≈ **97.12**; female length 64w 64.68 **z≈0**
- Agent target: male weight **40w / 3.2 kg → centile ≈30.8, z ≈ −0.50**
- ASQ: 6×yes=60; 6×not_yet=0 + referral; M-CHAT low/medium/high + reverse items 2,5,12
- Help intent: no medical dump; iron: clean `info_*` RAG; history: no re-fire overlay
- FA→EN pending count: **0**

## UI workflow screenshots (Playwright, live server)

| # | File | What it proves |
|---|------|----------------|
| 01 | `docs/screenshots/01_home.png` | Home hero + CTAs |
| 02 | `docs/screenshots/02_child.png` | Child “DemoShot” created |
| 03 | `docs/screenshots/03_chat_help.png` | Help reply (capabilities, no ASQ dump) |
| 04 | `docs/screenshots/04_chat_iron.png` | Medical RAG on iron |
| 05 | `docs/screenshots/05_chat_growth.png` | Multi-turn boy → overlay **centile≈30.8** + chart |
| 06 | `docs/screenshots/06_growth_form.png` | Growth form same numbers + overlay |
| 07 | `docs/screenshots/07_screening.png` | ASQ ages + M-CHAT picker |
| 08 | `docs/screenshots/08_home_mobile.png` | Mobile 390×844 home |

Capture script: `docs/_capture_ui_screenshots.py`  
Accuracy script: `docs/_accuracy_audit.py`

## How to re-verify

```powershell
cd C:\Users\mhf\Desktop\pedriatrics
$env:NESTLING_LOAD_MODELS='0'; $env:MPLBACKEND='Agg'
python -m pytest tests -q
python docs\_accuracy_audit.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8002
# then: python docs\_capture_ui_screenshots.py
```

Open UI: http://127.0.0.1:8002
