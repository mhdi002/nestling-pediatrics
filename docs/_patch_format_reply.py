# -*- coding: utf-8 -*-
from pathlib import Path

path = Path("assistant/agent/orchestrator.py")
text = path.read_text(encoding="utf-8")
start = text.index("    def _format_reply(")
end = text.index("    def handle(self, query: str, child_id: str | None = None) -> dict:")

new = '''    def _format_reply(
        self, out: dict, intents: set[str] | None = None, reply_lang: str = "en"
    ) -> str:
        from assistant.runtime_translate import translate_en_to_fa
        from assistant.parent_voice import (
            child_summary_chat,
            growth_plot_chat,
            medical_chat_answer,
            open_chat_turn,
        )

        intents = intents or set(out.get("intents") or [])
        parts: list[str] = []
        fa = reply_lang == "fa"
        measure_q = bool(out.get("explain_measure"))
        slots = out.get("slots") or {}
        has_growth = bool(
            slots.get("last_centile") is not None
            or slots.get("value") is not None
            or out.get("growth_analysis")
        )

        if "reassure" in intents:
            parts.append(
                "بله — با توجه به حرف‌تان، فعلاً جای نگرانی فوری به نظر نمی‌رسد. "
                "کنار شما هستم؛ اگر چیزی عوض شد بگویید."
                if fa
                else "Yes — from what you've shared, that usually sounds okay for now. "
                "I'm here with you; if something changes, just tell me."
            )
        elif "growth_analysis" in intents:
            snap = out.get("growth_analysis") or {}
            if snap.get("missing"):
                parts.append(
                    "هنوز نتیجه رشدی برای با هم دیدن نداریم. وزن/قد و سن را بفرستید "
                    "یا بگویید نمودار را نشان بدهم."
                    if fa
                    else "We don't have a growth result to look at together yet. "
                    "Send weight/length and age, or ask me to show the chart."
                )
            else:
                measure = snap.get("measure") or "growth"
                value = snap.get("value")
                centile = snap.get("centile")
                age_m = snap.get("age_months")
                lead = "بگذارید با هم نگاه کنیم: " if fa else "Let's look at it together: "
                bits = []
                if value is not None:
                    bits.append(f"{measure} {value}")
                if age_m is not None:
                    try:
                        bits.append(
                            f"حدود {float(age_m):.1f} ماهگی"
                            if fa
                            else f"around {float(age_m):.1f} months"
                        )
                    except (TypeError, ValueError):
                        pass
                if centile is not None:
                    try:
                        bits.append(
                            f"حدود صدک {float(centile):.0f}"
                            if fa
                            else f"about the {float(centile):.0f}th centile"
                        )
                    except (TypeError, ValueError):
                        pass
                if bits:
                    lead += ", ".join(bits) + ". "
                parts.append(
                    lead
                    + interpret_track_status(
                        snap.get("track_status"), snap.get("centile"), fa=fa
                    )
                )
                parts.append("سوال دیگری هم دارید؟" if fa else "Anything else on your mind about this?")
        elif "slot_update" in intents and "medical" not in intents and "growth" not in intents:
            remembered = ", ".join(
                f"{k}={v}"
                for k, v in slots.items()
                if k
                not in {
                    "child_id",
                    "gestational_age_weeks",
                    "chart_standard",
                    "want_overlay",
                    "last_centile",
                    "last_z_score",
                    "last_track_status",
                    "last_measure",
                    "last_value",
                    "last_age_months",
                    "last_chart_standard",
                }
            )
            if fa:
                parts.append(
                    "باشه، یادداشت کردم"
                    + (f" ({remembered})" if remembered else "")
                    + ". هر سوالی دارید بپرسید."
                )
            else:
                parts.append(
                    "Got it — I've noted that"
                    + (f" ({remembered})" if remembered else "")
                    + ". Ask me anything next."
                )
        elif "help" in intents:
            parts.append(
                "سلام! من نستلینگ هستم — مثل یک دستیار والدین کنار شما. "
                "می‌توانیم درباره رشد، تغذیه، خواب یا نگرانی‌های تکاملی حرف بزنیم "
                "و نمودار را با هم بکشیم. ساده بگویید چه چیزی ذهنتان را مشغول کرده."
                if fa
                else "Hi — I'm Nestling, here to chat with you about your little one. "
                "We can talk through growth, feeding, sleep, or developmental worries, "
                "and plot charts together. Just say what's on your mind."
            )
        elif (
            "chat" in intents
            and not out.get("tools", {}).get("tool_calls")
            and not out.get("medical_rag")
        ):
            parts.append(open_chat_turn(fa=fa, has_growth=has_growth))

        if "growth" in intents and not (out.get("tools") or {}).get("tool_calls"):
            if measure_q or "measure" in (out.get("missing_slots") or []):
                parts.append(
                    "برای نمودار بگویید وزن، قد یا دور سر را دارید — مثلاً: پسر، وزن، ۴۰ هفته، ۳٫۲ کیلو."
                    if fa
                    else "For the chart, tell me weight, length, or head — e.g. boy, weight, 40 weeks, 3.2 kg."
                )

        if (
            out.get("missing_slots")
            and "growth" in intents
            and not (out.get("tools") or {}).get("tool_calls")
        ):
            missing = out["missing_slots"]
            ask = [m for m in missing if not (measure_q and m == "measure")]
            if ask:
                hints = {
                    "sex": "boy or girl" if not fa else "پسر یا دختر",
                    "measure": "weight, length, or head" if not fa else "وزن، قد یا دور سر",
                    "age (weeks or months)": "age" if not fa else "سن",
                    "value": "the number" if not fa else "عدد اندازه",
                }
                pretty = [hints.get(m, m) for m in ask]
                parts.append(
                    ("برای ادامه هنوز لازم دارم: " + "؛ ".join(pretty) + ".")
                    if fa
                    else ("To continue I still need: " + "; ".join(pretty) + ".")
                )

        for tc in out.get("tools", {}).get("tool_calls", []):
            res = tc.get("result") or {}
            name = tc.get("name")
            if name in {"overlay_growth_on_chart", "growth_percentile"} and res.get("ok"):
                parts.append(growth_plot_chat(res, fa=fa))
            elif name == "get_child_summary" and res.get("ok"):
                summary = res.get("summary") or ""
                parts.append(child_summary_chat(summary, fa=fa))
            elif res.get("ok") is False and res.get("detail"):
                parts.append(
                    ("متأسفم، مشکلی پیش آمد: " if fa else "Sorry, something went wrong: ")
                    + str(res["detail"])
                )
            elif name not in {
                "overlay_growth_on_chart",
                "growth_percentile",
                "get_child_summary",
            }:
                if fa and res.get("summary_fa"):
                    parts.append(res["summary_fa"])
                elif res.get("summary"):
                    parts.append(res["summary"])

        if out.get("medical_rag"):
            ans = out["medical_rag"].get("answer", "")
            if fa and ans:
                ans = translate_en_to_fa(ans)
            parts.append(medical_chat_answer(ans, fa=fa))

        if "screening" in intents and "medical" in intents:
            parts.append(
                "اگر دوست دارید از بخش غربالگری، ASQ مناسب سن را هم می‌توانید شروع کنید — جایگزین پزشک نیست."
                if fa
                else "If you'd like, you can also try the age-matched ASQ in screening — it doesn't replace your clinician."
            )

        seen: set[str] = set()
        uniq: list[str] = []
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                uniq.append(p)
        if not uniq:
            uniq.append(open_chat_turn(fa=fa, has_growth=has_growth))
        return "\\n\\n".join(uniq)

'''

# Fix accidental double-escaped newlines in return
new = new.replace('return "\\\\n\\\\n".join(uniq)', 'return "\\n\\n".join(uniq)')

path.write_text(text[:start] + new + "\n" + text[end:], encoding="utf-8")
print("ok", start, end)
