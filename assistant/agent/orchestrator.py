#!/usr/bin/env python3
"""
Parent assistant orchestrator.

Every chat turn runs on the TypeScript RLM orchestrator (assistant/rlm/): the
model reads the care library under knowledge/ and calls the clinical tools from
a sandboxed REPL. This class owns sessions, the child record, growth/screening
helpers for the API, and the photo (vision) path.

Deterministic clinical tools compute all growth/screening numbers (no LLM math).
Full chat session memory persists multi-turn slots + messages.
"""

from __future__ import annotations

import logging
import re

from assistant.config import EN_DIR
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB
from assistant.refdata import weeks_per_month
from assistant.runtime_translate import translate_for_models
from assistant.settings import get_settings
from assistant.tools.clinical import dispatch_tool
from assistant.tools.who_term_equations import classify_maturity
from assistant.agent.slots import extract_growth_slots

log = logging.getLogger(__name__)


class ParentAssistant:
    def __init__(
        self,
        db: ChildMemoryDB | None = None,
        chat_memory: ChatMemory | None = None,
        use_xlam: bool | None = None,
        use_llm: bool | None = None,
        *,
        use_pleias: bool | None = None,
    ):
        self.db = db or ChildMemoryDB()
        self.chat_memory = chat_memory or ChatMemory()
        # The LLM drives every turn through the RLM orchestrator; this flag only
        # says whether one is configured (vision and memory distillation use it).
        load = bool(get_settings().nestling_load_models)
        if use_llm is None:
            if use_pleias is not None:
                use_llm = bool(use_pleias)
            else:
                from assistant.llm.qwen_client import llm_enabled

                use_llm = llm_enabled()
        self.use_llm = bool(use_llm)
        self.use_pleias = self.use_llm  # backward compat for callers/tests
        _ = load, use_xlam
        # Procedural, semantic and episodic memory. Built lazily on first use
        # so a construction failure cannot stop the app from starting; memory
        # degrading is survivable, not booting is not.
        self._memory = None
        self._rlm_host = None

    @property
    def memory(self):
        from assistant.memory.system import MemorySystem

        if self._memory is None:
            try:
                self._memory = MemorySystem()
            except Exception as exc:
                log.warning("Memory system unavailable: %s", exc)
                self._memory = False
        return self._memory or None


    @property
    def rlm(self):
        if self._rlm_host is None:
            from assistant.rlm.host import get_rlm_host

            self._rlm_host = get_rlm_host()
        return self._rlm_host


    def analyze_parent_photo(
        self,
        image_bytes: bytes,
        *,
        mime: str = "image/png",
        prompt: str = "",
        ui_lang: str | None = None,
    ) -> dict:
        """Vision path for parent-sent photos. Never diagnoses."""
        from assistant.llm.qwen_client import get_qwen, llm_enabled
        from assistant.parent_voice import medical_chat_answer
        from assistant.runtime_translate import ensure_pure_lang, translate_for_models

        caption = (prompt or "").strip()
        neutral = "What can you tell me about this photo?"
        reply_lang, en_prompt = translate_for_models(
            caption or neutral,
            ui_lang=ui_lang,
        )
        _ = en_prompt
        vision_text = ""
        mode = "no_vision"
        model = None
        if llm_enabled():
            try:
                client = get_qwen()
                if client.vision_ready:
                    # Ask for the description first, always. Without that the
                    # model was told only that the parent "sent a photo without
                    # text", and it answered by listing what it could not know
                    # -- "there is no text attached, I cannot see the child's
                    # age" -- while looking straight at the picture. A parent
                    # who sends a photo and no words wants to be told what is
                    # in it, not what is missing.
                    user_q = (
                        "Start by describing what you can actually see in this "
                        "photo in one or two plain sentences: where on the body "
                        "it is if you can tell, and the colour, size, pattern "
                        "and texture of anything on the skin. Then give likely "
                        "benign possibilities and the red flags that mean urgent "
                        "care. Educational guidance only, never a diagnosis. "
                        "Do not ask for details the parent has not given yet and "
                        "do not say what you cannot see -- describe what you can."
                    )
                    if caption:
                        user_q = f"The parent asks: {caption}\n\n{user_q}"
                    grounded_prompt = f"{user_q}\n\nKeep the response concise and parent-friendly."
                    vision_text = client.analyze_image(
                        image_bytes=image_bytes,
                        prompt=grounded_prompt,
                        mime=mime,
                        max_tokens=get_settings().llm_max_tokens_vision,
                    )
                    mode = "vision"
                    model = get_settings().nestling_vision_model
            except Exception as exc:
                log.warning("Vision analysis failed: %s", exc)
                vision_text = ""
                mode = f"vision_failed:{exc}"
        if not vision_text:
            # Without a vision model nothing has looked at the photo, so say so and
            # ask for words; the chat answers a description from the care library.
            vision_text = (
                "I could not look at the photo just now. Please describe what you see "
                "(where it is, its colour, size and how long it has been there) in the chat, "
                "and I will look it up in the care library."
            )
            if caption:
                vision_text = f"{vision_text} You wrote: {caption}"
            mode = "no_vision"
        spoken = medical_chat_answer(vision_text, fa=(reply_lang == "fa"))
        if reply_lang == "fa":
            spoken = ensure_pure_lang(spoken, "fa")
        else:
            spoken = ensure_pure_lang(spoken, "en")
        return {
            "ok": True,
            "reply": spoken,
            "reply_lang": reply_lang,
            "mode": mode,
            "model": model,
            "disclaimer": "Educational only — not a diagnosis. Seek urgent care for breathing trouble, lethargy, or rapidly spreading infection signs.",
        }


    def record_growth_and_overlay(
        self,
        child_id: str,
        sex: str,
        measure: str,
        weeks: float | None,
        value: float,
        age_months: float | None = None,
    ) -> dict:
        child = self.db.get_child(child_id) or {}
        ga = child.get("gestational_age_weeks")
        overlay = dispatch_tool(
            "overlay_growth_on_chart",
            {
                "sex": sex,
                "measure": measure,
                "weeks": weeks,
                "value": value,
                "age_months": age_months,
                "gestational_age_weeks": ga,
                "child_id": child_id,
            },
            db=self.db,
        )
        if overlay.get("ok"):
            store_weeks = overlay.get("weeks")
            if store_weeks is None and overlay.get("age_months") is not None:
                store_weeks = float(overlay["age_months"]) * weeks_per_month()
            self.db.add_growth(
                child_id,
                weeks=store_weeks if store_weeks is not None else (weeks or 0),
                measure=overlay["measure"],
                value=value,
                z_score=overlay.get("z_score"),
                centile=overlay.get("centile"),
                track_status=overlay.get("track_status"),
                age_months=(
                    float(overlay["age_months"])
                    if overlay.get("age_months") is not None
                    else age_months
                ),
            )
        return overlay

    def run_asq_session(self, child_id: str, age_months: int, domain_answers: dict) -> dict:
        result = dispatch_tool("score_asq_questionnaire", {"domain_answers": domain_answers}, db=self.db)
        if result.get("ok"):
            self.db.add_screening(child_id, "ASQ", domain_answers, result, age_months=age_months)
        asq_path = EN_DIR / "asq" / f"{age_months}m.json"
        return {
            "instrument": "ASQ",
            "age_months": age_months,
            "result": result,
            "questionnaire_available": asq_path.exists(),
            "parent_report": self._asq_parent_report(result) if result.get("ok") else result.get("detail"),
        }

    def run_mchat_session(self, child_id: str, answers: dict) -> dict:
        result = dispatch_tool("score_mchat", {"answers": answers}, db=self.db)
        if result.get("ok"):
            self.db.add_screening(child_id, "M-CHAT-R", answers, result)
        return {
            "instrument": "M-CHAT-R",
            "result": result,
            "parent_report": result.get("summary") or result.get("detail"),
        }

    def _asq_parent_report(self, result: dict) -> str:
        lines = ["ASQ results:"]
        for dom, res in result.get("domains", {}).items():
            flag = "BELOW cutoff — discuss with clinician" if res["below_cutoff"] else "above cutoff"
            lines.append(f"- {dom}: {res['total']}/{res['max']} ({flag})")
        if result.get("needs_referral"):
            lines.append(
                "One or more domains are below the cutoff. Please follow up with your pediatric clinician."
            )
        else:
            lines.append("No domain scored below the cutoff in this session.")
        return "\n".join(lines)

    def start_session(self, child_id: str | None = None) -> str:
        return self.chat_memory.create_session(child_id=child_id)

    def _chat_rlm(
        self,
        *,
        session_id: str,
        user_message: str,
        en_message: str,
        reply_lang: str,
        ui_lang: str | None,
        child_id: str | None,
        owner_user_id: str | None,
        ambiguous_children: list[dict],
    ) -> dict:
        """RLM chat path: knowledge via REPL, clinical tools via host, LTM embeddings only."""
        from assistant.runtime_translate import ensure_pure_lang, translate_en_to_fa

        new_slots = extract_growth_slots(en_message)
        for k, v in extract_growth_slots(user_message).items():
            new_slots.setdefault(k, v)
        if child_id:
            new_slots["child_id"] = child_id
            child = self.db.get_child(child_id) or {}
            if child.get("gestational_age_weeks") is not None:
                new_slots["gestational_age_weeks"] = child["gestational_age_weeks"]
                maturity = classify_maturity(child["gestational_age_weeks"])
                if maturity == "term":
                    new_slots.setdefault("chart_standard", "who_term")
                elif maturity == "preterm":
                    new_slots.setdefault("chart_standard", "intergrowth_preterm")
            if child.get("sex") and "sex" not in new_slots:
                new_slots["sex"] = child["sex"]
        slots = self.chat_memory.merge_slots(session_id, new_slots)

        child_context = ""
        if child_id:
            try:
                # The orchestrator has the conversation and its own long-term
                # memory; it is given the record, not a list of old questions.
                child_context = (
                    self.db.child_context_text(
                        child_id, owner_user_id=owner_user_id, include_events=False
                    )
                    or ""
                )
            except Exception as exc:
                log.warning("child_context_text failed: %s", exc)

        if ambiguous_children:
            # Two children can share a name, so each is shown with what tells
            # them apart in the record: date of birth, else sex and gestation.
            def _label(c: dict) -> str:
                name = str(c.get("name") or c.get("child_id"))
                if c.get("date_of_birth"):
                    born = "born " if reply_lang != "fa" else "متولد "
                    return f"{name} ({born}{c['date_of_birth']})"
                bits = [str(c.get("sex") or "")]
                if c.get("gestational_age_weeks") is not None:
                    bits.append(f"{float(c['gestational_age_weeks']):g} wk")
                extra = ", ".join(b for b in bits if b)
                return f"{name} ({extra})" if extra else name

            names = "; ".join(_label(c) for c in ambiguous_children)
            reply = (
                f"Which child did you mean? I found more than one match: {names}."
                if reply_lang != "fa"
                else f"کدام کودک را می‌گویید؟ چند نام مشابه پیدا کردم: {names}."
            )
            self.chat_memory.add_message(session_id, "assistant", reply)
            return {
                "session_id": session_id,
                "child_id": child_id,
                "slots": slots,
                "intents": ["chat"],
                "reply": reply,
                "reply_lang": reply_lang,
                "ui_lang": ui_lang,
                "ambiguous_children": ambiguous_children,
                "tools": {"tool_calls": []},
                "tool_results": [],
                "models": {
                    "tool_calling": "rlm_repl",
                    "rlm": True,
                },
            }

        stream_buf: list[str] = []
        turn_events: list[dict] = []

        def _emit(ev: dict) -> None:
            turn_events.append(ev)
            if ev.get("type") == "text_delta" and ev.get("text"):
                stream_buf.append(str(ev["text"]))

        rlm_out = self.rlm.run_chat_turn(
            session_id=session_id,
            user_message=user_message,
            en_message=en_message,
            child_id=child_id,
            owner_user_id=owner_user_id,
            child_context=child_context,
            reply_lang=reply_lang or "en",
            db=self.db,
            emit=_emit,
        )
        reply = (rlm_out.get("reply") or "").strip()
        if not reply:
            reply = (
                "I could not produce a reply just now. Please try again."
                if reply_lang != "fa"
                else "الان نتوانستم پاسخ بدهم. لطفاً دوباره تلاش کنید."
            )
        if reply_lang == "fa":
            # Model may answer in English when the agent prompt used the EN translation.
            if re.search(r"[A-Za-z]{4,}", reply) and not re.search(r"[\u0600-\u06FF]", reply):
                try:
                    reply = translate_en_to_fa(reply)
                except Exception:
                    pass
            reply = ensure_pure_lang(reply, "fa")
        else:
            reply = ensure_pure_lang(reply, "en")

        # Persist growth overlays that clinical tools produced.
        for tr in rlm_out.get("tool_results") or []:
            res = tr if isinstance(tr, dict) else {}
            if res.get("name") == "overlay_growth_on_chart" and res.get("ok") and child_id:
                try:
                    self.db.add_growth(
                        child_id,
                        weeks=res.get("weeks") or 0,
                        measure=res.get("measure") or "weight",
                        value=float(res.get("value") or 0),
                        z_score=res.get("z_score"),
                        centile=res.get("centile") or res.get("percentile"),
                        track_status=res.get("track"),
                    )
                except Exception as exc:
                    log.warning("Could not persist growth from RLM tool: %s", exc)

        public_tools = []
        for tr in rlm_out.get("tool_results") or []:
            if not isinstance(tr, dict):
                continue
            public_tools.append(
                {
                    "name": tr.get("name"),
                    "summary": tr.get("summary") or tr.get("detail"),
                    "overlay_filename": tr.get("overlay_filename") or tr.get("overlay"),
                    "centile": tr.get("centile") or tr.get("percentile"),
                    "z_score": tr.get("z_score"),
                    "track_status": tr.get("track"),
                    "chart_standard": tr.get("chart_standard"),
                    "ok": tr.get("ok"),
                }
            )

        self.chat_memory.add_message(
            session_id,
            "assistant",
            reply,
            tool_calls=public_tools,
            meta={"slots": slots, "rlm": True, "steps": rlm_out.get("steps")},
        )
        self._remember_for_child(
            child_id,
            owner_user_id=owner_user_id,
            user_message=en_message or user_message,
            intents={"medical", "growth"} if public_tools else {"medical"},
            slots=slots,
            session_id=session_id,
        )

        out = {
            "session_id": session_id,
            "child_id": child_id,
            "slots": slots,
            "intents": ["rlm"],
            "reply": reply,
            "reply_lang": reply_lang,
            "ui_lang": ui_lang,
            "tools": rlm_out.get("tools") or {"tool_calls": []},
            "tool_results": public_tools,
            "missing_slots": [],
            "ambiguous_children": ambiguous_children,
            "memory": rlm_out.get("memory") or {},
            "history": self.chat_memory.get_history(session_id),
            "rlm": rlm_out.get("rlm") or {},
            # The plan a multi-part request was carried out as, for the UI to show.
            "task_board": rlm_out.get("task_board"),
            "stream_tokens": stream_buf,
            "models": {
                "tool_calling": "rlm_repl",
                "llm": get_settings().nestling_llm_model,
                "rlm": True,
            },
        }
        return out

    def _children_named_in(
        self, owner_user_id: str | None, *messages: str
    ) -> list[dict]:
        """
        Children whose stored name appears in any of `messages`.

        The roster is read from the DB every turn rather than matched against a
        list in code, so adding a child needs no rule change. Matching both the
        raw and the translated message matters because transliteration mangles
        Persian names ("مونیکا" does not reliably come back as "Monica").
        """
        min_chars = get_settings().nestling_child_name_min_chars
        haystacks = [(m or "").lower() for m in messages if m]
        hits: dict[str, dict] = {}
        for row in self.db.list_children(owner_user_id=owner_user_id):
            name = str(row.get("name") or "").strip()
            if len(name) < min_chars:
                continue
            if not any(name.lower() in hay for hay in haystacks):
                continue
            for match in self.db.find_children_by_name(name, owner_user_id=owner_user_id):
                hits[match["child_id"]] = match
        return list(hits.values())

    def chat(
        self,
        session_id: str,
        user_message: str,
        child_id: str | None = None,
        ui_lang: str | None = None,
        owner_user_id: str | None = None,
    ) -> dict:
        """
        Full multi-turn chat with persistent memory.
        Persian messages are translated to English for the agent; replies return in the parent language.
        """
        session = self.chat_memory.get_session(session_id)
        if not session:
            session_id = self.chat_memory.create_session(child_id=child_id)
            session = self.chat_memory.get_session(session_id)

        if child_id:
            self.chat_memory.set_child(session_id, child_id)
        elif session.get("child_id"):
            child_id = session["child_id"]

        reply_lang, en_message = translate_for_models(user_message, ui_lang=ui_lang)
        self.chat_memory.add_message(session_id, "user", user_message)

        # A parent can name the child instead of picking one in the UI
        # ("how is Monica doing?"). Scope to the session owner so a name can
        # never reach another account's record.
        owner_user_id = owner_user_id or (session.get("owner_user_id") or None)
        ambiguous_children: list[dict] = []
        if not child_id:
            named = self._children_named_in(owner_user_id, user_message, en_message)
            if len(named) == 1:
                child_id = named[0]["child_id"]
                self.chat_memory.set_child(session_id, child_id)
            elif len(named) > 1:
                # Never guess between siblings — the reply asks which one.
                ambiguous_children = named
            else:
                # No child picked in the UI and none named. An account with a
                # single child means that child: without this, what the parent
                # said ("she has had diarrhoea since yesterday") was saved to no
                # child and missing from the next conversation about her. With
                # several children nothing is assumed.
                roster = self.db.list_children(owner_user_id=owner_user_id)
                if len(roster) == 1:
                    child_id = roster[0]["child_id"]
                    self.chat_memory.set_child(session_id, child_id)

        # Every turn runs on the RLM orchestrator; the retrieval (RAG) path is gone.
        return self._chat_rlm(
            session_id=session_id,
            user_message=user_message,
            en_message=en_message,
            reply_lang=reply_lang,
            ui_lang=ui_lang,
            child_id=child_id,
            owner_user_id=owner_user_id,
            ambiguous_children=ambiguous_children,
        )


    def _remember_for_child(
        self,
        child_id: str | None,
        *,
        owner_user_id: str | None,
        user_message: str,
        intents: set[str],
        slots: dict,
        session_id: str,
    ) -> None:
        """
        Write one salient turn to the child's durable timeline.

        Only clinically meaningful intents are kept: everything the parent says
        would otherwise accumulate as noise and crowd out the facts a later
        session actually needs. The note records the parent's own words — the
        assistant reply is largely derived from them plus retrieved guidance,
        so storing the concern is what makes the follow-up possible.
        """
        settings = get_settings()
        if not child_id or not settings.nestling_child_memory_enabled:
            return
        if not (intents & settings.child_memory_intents):
            return
        text = (user_message or "").strip()
        if not text:
            return
        topic = str(slots.get("last_topic") or "").strip()
        summary = f"{topic}: {text}" if topic and topic not in {"chat", ""} else text
        self.db.remember_note(
            child_id,
            summary,
            payload={
                "intents": sorted(intents),
                "session_id": session_id,
                "topic": topic or None,
            },
            owner_user_id=owner_user_id,
        )


    def handle(self, query: str, child_id: str | None = None) -> dict:
        sid = self.start_session(child_id=child_id)
        return self.chat(sid, query, child_id=child_id)

    def close(self) -> None:
        """Release SQLite handles (needed on Windows before deleting temp DBs)."""
        for name, closer in (("child_db", self.db.close), ("chat_memory", self.chat_memory.close)):
            try:
                closer()
            except Exception as exc:
                log.warning("Error closing %s: %s", name, exc)
