"""RLM turn engine — REPL loop with compaction, continuation guards, and tools."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from assistant.rlm.answer import (
    parent_facing,
    script_hint,
    speak_parent_answer,
    this_turn_evidence,
)
from assistant.rlm.budget import ModelLimits, compute_budget, model_limits, prompt_limits
from assistant.rlm.cells import (
    CellFenceFilter,
    extract_cells,
    fallback_cells,
    fence_labels,
    looks_executable,
    same_error_hint,
    strip_cells,
    strip_thinking,
    syntax_hint,
    timeout_hint,
    unclosed_cell_at,
)
from assistant.rlm.compact import summarize_in_chunks
from assistant.rlm.guard import ReplyGuard
from assistant.rlm.kernel import KernelFactory, PythonKernel
from assistant.rlm.llm import ContextOverflowError, complete, stream_chat
from assistant.rlm.prompts import get_prompt, pediatric_system_overlay, render
from assistant.rlm.tokens import chars_for_tokens, clip_chars, estimate_tokens
from assistant.rlm.tools import ToolContext, ToolExecutor, render_tool_surface
from assistant.rlm.transcript import (
    HostBlocks,
    TranscriptEntry,
    build_messages,
    estimate,
    to_units,
    truncate_old_outputs,
)
from assistant.settings import get_settings

log = logging.getLogger(__name__)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _parent_facing(text: str) -> str:
    return parent_facing(text)


def _needs_clinical_tool(user_text: str, reply: str, tool_calls: list) -> bool:
    """True when the turn talks about percentiles without having called a growth tool."""
    used = any(
        "growth_percentile" in str(t.get("name") or "")
        or "overlay_growth" in str(t.get("name") or "")
        for t in tool_calls
    )
    if used:
        return False
    blob = f"{user_text}\n{reply}".lower()
    asked = bool(
        re.search(
            r"percentile|centile|z-?score|growth_percentile|who term|intergrowth",
            blob,
        )
    )
    answered_with_number = bool(
        re.search(r"\b\d{1,2}(st|nd|rd|th)?\s*percentile\b|\bcentile\b|\bz-?score\b", reply.lower())
    )
    return asked and (answered_with_number or "percentile" in user_text.lower() or "centile" in user_text.lower())


def _needs_workspace_read(user_text: str, tool_calls: list) -> bool:
    used = any(
        str(t.get("name") or "").startswith(("fs.", "search."))
        for t in tool_calls
    )
    if used:
        return False
    return bool(
        re.search(
            r"search\.grep|fs\.read|fs\.glob|INDEX\.md|guidelines|knowledge|"
            r"\bgrep\b|\blook(?:\s+up|\s+in)\b|\bsearch\b",
            user_text or "",
            re.I,
        )
    )


def _needs_language_match(user_text: str, reply: str, want: str | None = None) -> bool:
    lang = (want or "").lower()
    if lang not in {"fa", "fa-ir", "persian", "farsi"} and script_hint(user_text) != "fa":
        return False
    return script_hint(reply) != "fa" and bool((reply or "").strip())


def _repl_has_hits(text: str) -> bool:
    t = (text or "").strip()
    if not t or t in {"[]", "(no output)", "None"}:
        return False
    if re.search(r"SyntaxError|NameError|ToolError|Traceback|\[error\]|\[not run\]", t):
        return False
    if re.search(r"\[search\.grep\].*\d+ hits", t):
        return True
    if re.search(r"\S+\.(?:md|txt|json|csv):\d+:", t):
        return True
    if re.search(r"\b(centile|percentile|z_score)\b", t, re.I) and "{" in t:
        return True
    if t.startswith("[") and "hits" in t.lower():
        return True
    if "similar files:" in t.lower():
        return True
    return len(t) > 120 and not t.startswith("I ")


def _repl_is_empty_search(text: str) -> bool:
    t = (text or "").strip()
    if t in {"[]", "(no output)", "None", ""}:
        return True
    lines = [ln for ln in t.splitlines() if ln.strip() and not ln.strip().startswith("[search.grep]")]
    return bool(lines) and all(ln.strip() == "[]" for ln in lines)


@dataclass
class RlmTurnResult:
    final_text: str
    work_log: str = ""
    steps: int = 0
    usage: dict = field(default_factory=dict)
    tool_calls: list = field(default_factory=list)
    compacted: bool = False
    compaction_reason: str | None = None
    stopped_at_limit: bool = False
    events: list = field(default_factory=list)


@dataclass
class _Session:
    chat_id: str
    entries: list[TranscriptEntry]
    kernel: PythonKernel | None = None
    kernel_root: str | None = None
    last_turn_idx: int = -1
    token_scale: float = 1.0
    failed_cell: dict | None = None
    notices: list[str] = field(default_factory=list)
    binding: dict | None = None
    turn_evidence: str = ""


class RlmEngine:
    def __init__(
        self,
        *,
        tools: ToolExecutor | None = None,
        kernels: KernelFactory | None = None,
    ) -> None:
        self.tools = tools or ToolExecutor()
        self.kernels = kernels or KernelFactory()
        self.sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def drop_chat(self, chat_id: str) -> None:
        with self._lock:
            s = self.sessions.pop(chat_id, None)
        if s and s.kernel:
            try:
                s.kernel.shutdown()
            except Exception:
                pass

    def shutdown_all(self) -> None:
        with self._lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()
        for s in sessions:
            if s.kernel:
                try:
                    s.kernel.shutdown()
                except Exception:
                    pass

    def run_turn(
        self,
        *,
        chat_id: str,
        turn_idx: int,
        user_text: str,
        workspace_root: Path,
        blocks: HostBlocks,
        child_id: str | None = None,
        db: Any = None,
        emit: Callable[[dict], None] | None = None,
        abort: Callable[[], bool] | None = None,
        prior_entries: list[TranscriptEntry] | None = None,
    ) -> RlmTurnResult:
        emit = emit or (lambda _e: None)
        abort = abort or (lambda: False)
        s = get_settings()
        limits = model_limits()

        with self._lock:
            session = self.sessions.get(chat_id)
            if session is None:
                session = _Session(chat_id=chat_id, entries=list(prior_entries or []))
                self.sessions[chat_id] = session
            elif prior_entries is not None and session.last_turn_idx < 0 and not session.entries:
                session.entries = list(prior_entries)

        session.entries.append({"kind": "user", "text": user_text, "turn_idx": turn_idx})
        tool_ctx = ToolContext(
            workspace_root=Path(workspace_root).resolve(),
            chat_id=chat_id,
            child_id=child_id,
            db=db,
            notice=lambda t: session.notices.append(t),
        )

        def on_host(rtype: str, payload: dict) -> Any:
            if rtype != "tool":
                raise RuntimeError(f"unknown host request '{rtype}'")
            tool_id = str(payload.get("tool_id") or "")
            args = payload.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            r = self.tools.execute(tool_id, args, tool_ctx)
            if not r.get("ok"):
                raise RuntimeError(r.get("error") or "tool failed")
            return r.get("result")

        result = self._loop(
            session=session,
            turn_idx=turn_idx,
            user_text=user_text,
            blocks=blocks,
            limits=limits,
            tool_ctx=tool_ctx,
            on_host=on_host,
            emit=emit,
            abort=abort,
        )
        session.last_turn_idx = turn_idx
        result.tool_calls = list(tool_ctx.tool_calls)
        result.final_text = _parent_facing(result.final_text)
        return result

    def _finish_turn(
        self,
        *,
        session: _Session,
        turn_idx: int,
        user_text: str,
        blocks: HostBlocks,
        draft: str,
        work_log: list[str],
        steps: int,
        usage: dict,
        compacted: bool,
        compaction_reason: str | None,
        events: list,
        stopped: bool = False,
        tool_ctx: ToolContext | None = None,
    ) -> RlmTurnResult:
        parts = [session.turn_evidence or this_turn_evidence(session.entries, turn_idx)]
        if tool_ctx:
            for tc in tool_ctx.tool_calls:
                if not tc.get("ok"):
                    continue
                res = tc.get("result")
                if res is None:
                    continue
                if isinstance(res, (dict, list)):
                    parts.append(clip_chars(json.dumps(res, default=str), 900))
                else:
                    parts.append(clip_chars(str(res), 900))
        evidence = clip_chars("\n".join(p for p in parts if p), 4500)
        spoken = speak_parent_answer(
            question=user_text,
            lang=str(blocks.get("lang") or ""),
            evidence=evidence,
            memory=blocks.get("memory") or "",
            child=blocks.get("child") or "",
            draft=draft or "",
        )
        return RlmTurnResult(
            final_text=spoken,
            work_log="\n".join(work_log),
            steps=steps,
            usage=usage,
            compacted=compacted,
            compaction_reason=compaction_reason,
            stopped_at_limit=stopped,
            events=events,
        )

    def _system_prompt(self, compact: bool = False) -> str:
        tools = self.tools.list()
        labels = fence_labels()
        body = render(
            get_prompt("rlm/system.md"),
            tool_surface=render_tool_surface(tools, compact=compact),
            max_depth=str(get_settings().rlm_max_subagent_depth),
            can_spawn="no",
            cell_fence_or=" or ".join(f"`{x}`" for x in labels),
            cell_fence_example=labels[0],
        )
        return pediatric_system_overlay().strip() + "\n\n" + body

    def _ensure_kernel(self, session: _Session, tool_ctx: ToolContext, on_host) -> PythonKernel:
        root = str(tool_ctx.workspace_root)
        if session.kernel and session.kernel.is_alive() and session.kernel_root == root:
            # Kernel persists across turns; always re-bind the current turn's
            # tool context or calls land on a stale ToolContext and vanish.
            session.kernel.set_host_handler(on_host)
            return session.kernel
        if session.kernel:
            try:
                session.kernel.shutdown()
            except Exception:
                pass
        s = get_settings()
        session.kernel_root = root
        session.kernel = self.kernels.create(
            cwd=tool_ctx.workspace_root,
            tools=self.tools.list(),
            on_host_request=on_host,
            output_limit_chars=int(s.rlm_cell_output_limit_chars),
        )
        return session.kernel

    def _measure(self, session: _Session, messages: list[dict]) -> int:
        return int(estimate(messages) * (session.token_scale or 1))

    def _loop(
        self,
        *,
        session: _Session,
        turn_idx: int,
        user_text: str,
        blocks: HostBlocks,
        limits: ModelLimits,
        tool_ctx: ToolContext,
        on_host,
        emit,
        abort,
    ) -> RlmTurnResult:
        s = get_settings()
        full_sys = self._system_prompt(False)
        compact_sys: str | None = None
        resume = get_prompt("rlm/compact_resume.md")
        usage = {"inputTokens": 0, "outputTokens": 0}
        work_log: list[str] = []
        events: list[dict] = []
        last_text = ""
        compacted = False
        compaction_reason = None
        cuts = 0
        calm_next = None
        ran: dict[str, dict] = {}
        overflows = 0
        max_steps = int(s.rlm_max_steps)
        tool_nudges = 0
        lang_nudges = 0
        empty_searches = 0
        found_evidence = False
        session.turn_evidence = ""

        def _emit(ev: dict) -> None:
            events.append(ev)
            emit(ev)

        def finish(draft: str, steps: int, *, stopped: bool = False) -> RlmTurnResult:
            return self._finish_turn(
                session=session,
                turn_idx=turn_idx,
                user_text=user_text,
                blocks=blocks,
                draft=draft,
                work_log=work_log,
                steps=steps,
                usage=usage,
                compacted=compacted,
                compaction_reason=compaction_reason,
                events=events,
                stopped=stopped,
                tool_ctx=tool_ctx,
            )

        pre_need = _needs_clinical_tool(user_text, "", []) or _needs_workspace_read(
            user_text, []
        )
        if pre_need:
            pre_cells = fallback_cells(user_text, "")
            if pre_cells:
                tool_nudges = 1
                kernel = self._ensure_kernel(session, tool_ctx, on_host)
                pre_outs: list[str] = []
                for code in pre_cells:
                    cell_id = _new_id("cell")
                    _emit({"type": "code_cell", "cellId": cell_id, "code": code})
                    started = time.time()
                    result = kernel.execute(
                        code,
                        lambda o, cid=cell_id: _emit(
                            {
                                "type": "cell_output",
                                "cellId": cid,
                                "stream": o.get("stream"),
                                "text": o.get("text"),
                            }
                        ),
                        int(s.rlm_cell_timeout_sec) * 1000,
                    )
                    _emit(
                        {
                            "type": "cell_done",
                            "cellId": cell_id,
                            "ok": result.ok,
                            "error": result.error,
                            "durationMs": int((time.time() - started) * 1000),
                        }
                    )
                    chunk = clip_chars(
                        "\n".join(
                            p
                            for p in (
                                result.stdout,
                                f"[result] {result.result}" if result.result else "",
                                f"[error]\n{result.error}" if result.error else "",
                            )
                            if p
                        ).strip()
                        or "(no output)",
                        min(
                            int(s.rlm_cell_output_limit_chars),
                            chars_for_tokens(
                                limits.context_length * float(s.rlm_cell_output_max_share)
                            ),
                        ),
                    )
                    pre_outs.append(chunk)
                    ran[code.strip()] = {"times": 1, "out": chunk}
                    work_log.append(
                        f"prefetch: {clip_chars(code.splitlines()[0] if code else '', 120)}"
                    )
                joined = "\n---\n".join(pre_outs)
                session.turn_evidence = clip_chars(
                    (session.turn_evidence + "\n" + joined).strip(), 4000
                )
                session.entries.append({"kind": "repl", "text": joined, "turn_idx": turn_idx})
                if _repl_has_hits(joined) or tool_ctx.tool_calls:
                    found_evidence = True
                session.entries.append(
                    {
                        "kind": "note",
                        "text": (
                            "The REPL output above is your source. Summarize it for the parent "
                            "now in their language. Do not claim there were no results. "
                            "No code fence."
                        ),
                        "turn_idx": turn_idx,
                    }
                )

        for step in range(max_steps):
            if abort():
                break
            sys = full_sys
            if estimate_tokens(full_sys) > limits.context_length * float(s.rlm_tool_docs_max_share):
                if compact_sys is None:
                    compact_sys = self._system_prompt(True)
                sys = compact_sys
            messages = build_messages(sys, resume, blocks, session.entries)
            messages, did_compact, reason = self._fit_budget(
                session, limits, sys, resume, blocks, messages, _emit
            )
            if did_compact:
                compacted = True
                compaction_reason = reason
            input_tokens = self._measure(session, messages)
            max_tokens = max(
                64,
                min(limits.max_output_tokens, limits.context_length - input_tokens - 128),
            )
            trigger, input_limit = prompt_limits(limits)
            _emit(
                {
                    "type": "context_usage",
                    "usedTokens": input_tokens,
                    "triggerTokens": trigger,
                    "inputLimit": input_limit,
                    "contextLength": limits.context_length,
                }
            )

            raw = ""
            filt = CellFenceFilter()
            thinking_max = chars_for_tokens(
                max(
                    256,
                    min(
                        int(s.rlm_thinking_max_tokens),
                        int(max_tokens * float(s.rlm_thinking_max_share)),
                    ),
                )
            )
            guard = ReplyGuard(
                thinking_max_chars=thinking_max,
                thinking_max_ms=int(s.rlm_thinking_max_sec) * 1000,
                repeat_limit=int(s.rlm_repeat_limit),
            )
            cut_by = None
            enable_thinking = False if calm_next else None
            temperature = 0.7 if calm_next == "repeating" else float(s.rlm_temperature)
            calm_next = None
            empty_retries = 0
            max_empty = int(s.rlm_empty_stream_retries)

            while True:
                raw = ""
                filt = CellFenceFilter()
                try:
                    for chunk in stream_chat(
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        enable_thinking=enable_thinking,
                        signal_aborted=abort,
                    ):
                        if abort():
                            break
                        if chunk["type"] == "thinking":
                            cut_by = guard.push("thinking", chunk["text"])
                            _emit({"type": "thinking_delta", "text": chunk["text"]})
                            if cut_by:
                                break
                        elif chunk["type"] == "text":
                            raw += chunk["text"]
                            visible = filt.push(chunk["text"])
                            if visible:
                                _emit({"type": "text_delta", "text": visible})
                            cut_by = guard.push("text", chunk["text"])
                            if cut_by:
                                break
                        elif chunk["type"] == "usage":
                            usage["inputTokens"] += chunk.get("inputTokens") or 0
                            usage["outputTokens"] += chunk.get("outputTokens") or 0
                    rest = filt.flush()
                    if rest and not cut_by:
                        _emit({"type": "text_delta", "text": rest})
                    overflows = 0
                except ContextOverflowError as exc:
                    overflows += 1
                    log.warning("context overflow; compacting: %s", exc)
                    for e in session.entries:
                        if e.get("kind") == "repl" and len(e.get("text") or "") > 400:
                            e["text"] = clip_chars(e.get("text") or "", 400)
                    self._compact_transcript(session, limits, True, _emit)
                    compacted = True
                    compaction_reason = "in_turn"
                    if overflows > 3:
                        fallback = (last_text or "").strip() or (
                            "I had to shorten this chat to fit the model window. "
                            "Please ask that last question again in a shorter form."
                        )
                        compacted = True
                        compaction_reason = "in_turn"
                        return finish(fallback, step + 1)
                    break  # retry outer step
                except Exception:
                    raise

                if cut_by:
                    break
                if (raw or "").strip() or empty_retries >= max_empty:
                    break
                empty_retries += 1
                log.warning("empty chat stream; retry %s/%s", empty_retries, max_empty)
                _emit({"type": "status", "text": f"Empty model reply; retrying ({empty_retries})"})
                time.sleep(float(s.rlm_empty_stream_backoff_sec))

            if overflows and not raw and not cut_by:
                continue

            if cut_by:
                cuts += 1
                if cuts > int(s.rlm_max_cuts_in_a_row):
                    advice = (
                        "The model kept looping, so this turn was stopped. "
                        "Try a shorter, more specific request."
                    )
                    final = f"{last_text}\n\n({advice})" if last_text else advice
                    return finish(final, step + 1)
                note = render(
                    get_prompt(
                        "rlm/cut_repeating.md"
                        if cut_by == "repeating"
                        else "rlm/cut_thinking.md"
                    ),
                    tokens=str(round(guard.thinking_chars / 4)),
                ).strip()
                session.entries.append({"kind": "note", "text": note, "turn_idx": turn_idx})
                calm_next = cut_by
                continue
            cuts = 0

            text = strip_thinking(raw)
            cut_at = unclosed_cell_at(text)
            if cut_at >= 0:
                cut = text[cut_at:]
                text = f"{text[:cut_at]}{clip_chars(cut, 600)}\n[cell cut off at the output limit after {len(cut)} chars]"
            session.entries.append({"kind": "assistant", "text": text, "turn_idx": turn_idx})
            cells = [c for c in extract_cells(text) if looks_executable(c)]
            prose = strip_cells(text[:cut_at] if cut_at >= 0 else text)
            needs_clinical = _needs_clinical_tool(
                user_text, prose or last_text, tool_ctx.tool_calls
            )
            needs_ws = _needs_workspace_read(user_text, tool_ctx.tool_calls) and not found_evidence
            needs_tool = needs_clinical or needs_ws
            if found_evidence:
                cells = [
                    c
                    for c in cells
                    if "search.grep" not in c and "fs.glob" not in c
                ]
            if needs_tool and tool_nudges < 1:
                recovered = fallback_cells(user_text, text)
                if recovered:
                    cells = recovered
                    tool_nudges += 1
                    _emit({"type": "status", "text": "Recovered tool call as a REPL cell"})
            elif found_evidence and not needs_clinical:
                cells = []
            if prose:
                last_text = prose
                # Emit remaining prose as result-facing text if stream filtered cells only
            if not cells:
                if cut_at >= 0:
                    session.entries.append(
                        {
                            "kind": "note",
                            "text": "Your last reply hit the output limit before the repl cell closed.",
                            "turn_idx": turn_idx,
                        }
                    )
                    continue
                final_prose = prose or last_text
                if found_evidence and (prose or last_text):
                    return finish(prose or last_text, step + 1)
                if needs_tool and tool_nudges < 1 and step + 1 < max_steps:
                    tool_nudges += 1
                    example = (fence_labels() or ["repl"])[0]
                    sample = fallback_cells(user_text, text)
                    shown = sample[0] if sample else f'print(search.grep(pattern="...", glob="**/*.md", ignore_case=True))'
                    session.entries.append(
                        {
                            "kind": "note",
                            "text": (
                                "Your last message had no executable fence, so no tool ran. "
                                f"Reply with ONLY one `{example}` cell that calls the tool, "
                                "then stop. Prose is not executed.\n\n"
                                f"```{example}\n{shown}\n```"
                            ),
                            "turn_idx": turn_idx,
                        }
                    )
                    _emit({"type": "status", "text": "Requiring a fenced REPL tool call before final answer"})
                    continue
                if (
                    _needs_language_match(user_text, final_prose, blocks.get("lang"))
                    and lang_nudges < 1
                    and step + 1 < max_steps
                ):
                    lang_nudges += 1
                    session.entries.append(
                        {
                            "kind": "note",
                            "text": (
                                "The parent wrote in Persian. Rewrite your final answer entirely "
                                "in Persian, with no English sentences and no code fence."
                            ),
                            "turn_idx": turn_idx,
                        }
                    )
                    continue
                if not (final_prose or "").strip():
                    # Hard guard against empty final replies
                    last_text = (
                        "I could not produce a reply just now. Please try again, "
                        "or rephrase the question."
                    )
                    _emit({"type": "text_delta", "text": last_text})
                    final_prose = last_text
                return finish(final_prose, step + 1)

            kernel = self._ensure_kernel(session, tool_ctx, on_host)
            outputs: list[str] = []
            for code in cells:
                if abort():
                    break
                cell_id = _new_id("cell")
                _emit({"type": "code_cell", "cellId": cell_id, "code": code})
                key = code.strip()
                seen = ran.get(key)
                if seen and seen["times"] >= int(s.rlm_max_identical_cells):
                    msg = (
                        f"[not run] This exact cell already ran {seen['times']} times "
                        "with the same output; take a different step."
                    )
                    _emit({"type": "cell_done", "cellId": cell_id, "ok": False, "error": msg})
                    outputs.append(msg)
                    break
                started = time.time()
                result = kernel.execute(
                    code,
                    lambda o, cid=cell_id: _emit(
                        {"type": "cell_output", "cellId": cid, "stream": o.get("stream"), "text": o.get("text")}
                    ),
                    int(s.rlm_cell_timeout_sec) * 1000,
                )
                notices = session.notices
                session.notices = []
                notice_text = f"[IDE]\n" + "\n".join(notices) if notices else ""
                _emit(
                    {
                        "type": "cell_done",
                        "cellId": cell_id,
                        "ok": result.ok,
                        "error": result.error,
                        "durationMs": int((time.time() - started) * 1000),
                    }
                )
                repeated = ""
                if not result.ok and result.error and session.failed_cell:
                    repeated = same_error_hint(session.failed_cell.get("error") or "", result.error)
                if result.ok:
                    session.failed_cell = None
                elif result.error and not result.timed_out and not result.interrupted:
                    session.failed_cell = {"code": code, "error": result.error}
                parts = [
                    result.stdout,
                    f"[stderr]\n{result.stderr}" if result.stderr else "",
                    f"[result] {result.result}" if result.result else "",
                    f"[error]\n{result.error}" if result.error else "",
                    syntax_hint(code, result.error or "") if result.error else "",
                    repeated,
                    notice_text,
                    timeout_hint(int(s.rlm_cell_timeout_sec)) if result.timed_out else "",
                ]
                out = clip_chars(
                    "\n".join(p for p in parts if p).strip() or "(no output)",
                    min(
                        int(s.rlm_cell_output_limit_chars),
                        chars_for_tokens(limits.context_length * float(s.rlm_cell_output_max_share)),
                    ),
                )
                times = (seen["times"] + 1) if seen and seen.get("out") == out else 1
                ran[key] = {"times": times, "out": out}
                if times > 1:
                    out += f"\n[hint] You already ran this exact cell {times - 1} time(s)."
                outputs.append(out)
                work_log.append(
                    f"cell: {clip_chars('|'.join(code.splitlines()[:3]), 160)} -> {clip_chars(out.replace(chr(10), ' '), 200)}"
                )
                if not result.ok and (result.interrupted or result.timed_out):
                    break
            joined = "\n---\n".join(outputs)
            session.turn_evidence = clip_chars(
                (session.turn_evidence + "\n" + joined).strip(), 4000
            )
            session.entries.append(
                {"kind": "repl", "text": joined, "turn_idx": turn_idx}
            )
            if _repl_has_hits(joined):
                found_evidence = True
                empty_searches = 0
                session.entries.append(
                    {
                        "kind": "note",
                        "text": (
                            "The REPL output above is your source. Summarize it for the parent "
                            "now in their language. Do not say there were no results. "
                            "Do not search again this turn. No code fence."
                        ),
                        "turn_idx": turn_idx,
                    }
                )
            elif _repl_is_empty_search(joined):
                empty_searches += 1
                if empty_searches >= int(s.rlm_empty_search_abort):
                    session.entries.append(
                        {
                            "kind": "note",
                            "text": (
                                "Repeated searches returned nothing new. Answer from INDEX.md "
                                "or the hits you already have. Next message: plain text only."
                            ),
                            "turn_idx": turn_idx,
                        }
                    )
                    found_evidence = True
            skipped_only = bool(outputs) and all("[not run]" in o for o in outputs)
            if skipped_only and (last_text or "").strip():
                return finish(last_text, step + 1)
            left = max_steps - step - 1
            if left > 0 and left == int(s.rlm_step_warn_remaining):
                session.entries.append(
                    {
                        "kind": "note",
                        "text": f"{left} steps are left in this turn. Summarize progress and finish or leave a clear next step.",
                        "turn_idx": turn_idx,
                    }
                )

        return finish(last_text, max_steps, stopped=True)

    def _fit_budget(
        self,
        session: _Session,
        limits: ModelLimits,
        sys: str,
        resume: str,
        blocks: HostBlocks,
        messages: list[dict],
        emit,
    ) -> tuple[list[dict], bool, str | None]:
        s = get_settings()
        trigger, input_limit = prompt_limits(limits)
        tokens = self._measure(session, messages)
        user_turns = sum(1 for e in session.entries if e.get("kind") == "user")
        min_turns = int(s.rlm_compact_min_user_turns)
        if tokens < trigger and tokens < input_limit:
            return messages, False, None
        if tokens < input_limit and user_turns < min_turns:
            return messages, False, None
        changed = False
        reason = None
        truncated, did = truncate_old_outputs(
            session.entries,
            int(s.rlm_old_output_keep_chars),
            2,
        )
        if did:
            session.entries = truncated
            messages = build_messages(sys, resume, blocks, session.entries)
            tokens = self._measure(session, messages)
            changed = True
            reason = "truncate"
            if tokens < trigger and tokens < input_limit:
                return messages, changed, reason
        self._compact_transcript(session, limits, False, emit)
        messages = build_messages(sys, resume, blocks, session.entries)
        return messages, True, "rlm"

    def _compact_transcript(
        self,
        session: _Session,
        limits: ModelLimits,
        aggressive: bool,
        emit,
    ) -> None:
        s = get_settings()
        units = to_units(session.entries)
        if len(units) < 2:
            return
        b = compute_budget(limits)
        keep_budget = b.keep // 2 if aggressive else b.keep
        used = 0
        cut = len(units)
        for i in range(len(units) - 1, -1, -1):
            cost = int(
                sum(estimate_tokens(e.get("text") or "") for e in units[i])
                * (session.token_scale or 1)
            )
            if i < len(units) - 1 and used + cost > keep_budget:
                break
            used += cost
            cut = i
        if cut == 0:
            cut = 1
        fold = [e for u in units[:cut] for e in u]
        keep = [e for u in units[cut:] for e in u]
        request = next((e for e in reversed(session.entries) if e.get("kind") == "user"), None)
        if request and request in fold:
            max_chars = max(2000, b.newest_limit * 4)
            text = request.get("text") or ""
            keep = [
                {
                    **request,
                    "text": clip_chars(text, max_chars) if len(text) > max_chars else text,
                },
                *keep,
            ]
        previous = next(
            (e.get("text") for e in reversed(session.entries) if e.get("kind") == "checkpoint"),
            None,
        )
        body = []
        for e in fold:
            kind = e.get("kind")
            text = e.get("text") or ""
            if kind == "assistant":
                body.append({"role": "assistant", "content": text})
            elif kind == "repl":
                body.append({"role": "user", "content": f"[REPL output]\n{clip_chars(text, 4000)}"})
            elif kind == "note":
                body.append({"role": "user", "content": f"[host note] {text}"})
            else:
                body.append({"role": "user", "content": text})
        summary = summarize_in_chunks(
            complete_fn=complete,
            previous=previous,
            body=body,
            max_tokens=max(512, int(b.output_reserve * float(s.rlm_compact_summary_max_share))),
            context_length=limits.context_length,
        )
        session.entries = [{"kind": "checkpoint", "text": summary}, *keep]
        emit(
            {
                "type": "compaction",
                "checkpoint": {
                    "summary": summary,
                    "foldedTurns": len(fold),
                    "keptTurns": len(keep),
                    "compactedBy": "in_turn" if aggressive else "rlm",
                },
            }
        )
