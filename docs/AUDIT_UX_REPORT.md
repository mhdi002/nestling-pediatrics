# Nestling UX + accuracy pass (2026-07-28)

## Verdict: PASS

### Fixes shipped
1. **Child page toast** — toast moved to **top** (no longer covers Save).
2. **Chat tool JSON** — removed; brief “Using clinical tools…” scene then clean reply + chart image.
3. **App icon** — detailed SVG nest (eggs, bird, sun, woven bowl) in header + hero.
4. **Persian + RTL** — language toggle; all chrome strings via `web/i18n.js`; FA↔EN chat translator (`assistant/runtime_translate.py`).
5. **Simple parent language** — soft phrases auto-call overlay tools (no need to say “overlay”).
6. **نارس / طبیعی routing** — GA &lt; 37 → INTERGROWTH preterm; GA ≥ 37 → WHO term equations (`who_term_equations.py`).
7. **Chat composer overlap** — thread padding + sticky composer z-index.

### Verification
| Check | Result |
|-------|--------|
| Pytest | **58 passed** |
| Accuracy audit | **19/19** |
| ASQ ages 4–60m load+score | **18/18** |
| Overlay in chat screenshot | **1 image** rendered |
| Tool `<pre>` JSON | **0** |

### Screenshots (`docs/screenshots/`)
- `01_home.png`, `01b_home_fa_rtl.png` (Persian RTL)
- `02_child.png`, `03_chat_help.png`
- `05_chat_growth.png` (simple language + chart, no JSON)
- `05b_chat_fa_term.png` (Persian soft phrase)
- `06_growth_form.png`, `07_screening.png`, `07b_asq_quiz.png`, `08_home_mobile.png`

### How parents talk now
- EN: “my boy weighs 3.2 kg at 40 weeks”
- FA: “پسرم ۲ ماهشه وزنش ۵.۶ کیلو”
- Agent picks chart from child’s birth GA automatically.
