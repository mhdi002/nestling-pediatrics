"""Host tools exposed to the REPL: confined fs/search + clinical dispatch."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from assistant.settings import get_settings
from assistant.tools.clinical import TOOL_SPECS, dispatch_tool

log = logging.getLogger(__name__)


@dataclass
class ToolDescriptor:
    id: str
    group: str
    description: str
    parameters: dict
    returns: str = "Any"
    side_effects: bool = False


@dataclass
class ToolContext:
    workspace_root: Path
    chat_id: str
    child_id: str | None = None
    db: Any = None
    notice: Callable[[str], None] | None = None
    tool_calls: list[dict] = field(default_factory=list)


def resolve_inside(root: Path, rel: str) -> Path:
    root = root.resolve()
    text = (rel or ".").replace("\\", "/").lstrip("./")
    # Workspace root already *is* knowledge/; models often prefix paths with it.
    root_name = root.name.lower()
    if text == root_name or text.startswith(root_name + "/"):
        text = text[len(root_name) :].lstrip("/") or "."
    if text.startswith("knowledge/"):
        text = text[len("knowledge/") :] or "."
    candidate = (root / text).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"path escapes workspace: {rel}") from exc
    return candidate


def _is_probably_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:8000]
    except OSError:
        return True
    if b"\x00" in chunk:
        return True
    return False


def _relax_glob(pattern: str | None) -> str | None:
    if not pattern:
        return None
    g = str(pattern).replace("\\", "/").lstrip("./")
    if g in {".", "**", "**/*"}:
        return None
    if "*" in g and not g.startswith("**/") and "/" not in g:
        return "**/" + g
    return g


def _similar_files(root: Path, rel: str, *, limit: int = 8) -> list[str]:
    name = Path(str(rel or "")).name.lower()
    stem = Path(str(rel or "")).stem.lower().replace("_", "").replace("-", "")
    if not stem and not name:
        return []
    hits: list[str] = []
    exact: list[str] = []
    try:
        files = root.rglob("*")
    except OSError:
        return []
    for f in files:
        if not f.is_file() or f.name.startswith("."):
            continue
        rel_p = f.relative_to(root).as_posix()
        fn = f.name.lower()
        st = f.stem.lower().replace("_", "").replace("-", "")
        if fn == name or st == stem:
            exact.append(rel_p)
        elif stem and (stem in st or st in stem) and len(stem) >= 4:
            hits.append(rel_p)
        if len(exact) + len(hits) >= limit:
            break
    return (exact + hits)[:limit]


def _search_tokens(pattern: str) -> list[str]:
    parts = [t for t in re.split(r"[^\w\u0600-\u06FF]+", pattern or "") if t]
    return [t for t in parts if len(t) >= 3 and not t.isdigit()]


def _glob_to_re(pattern: str) -> re.Pattern[str]:
    # Escape then restore * and **
    p = pattern.replace("\\", "/")
    parts: list[str] = []
    i = 0
    while i < len(p):
        if p.startswith("**/", i):
            parts.append("(.*/)?")
            i += 3
        elif p[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif p[i] == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(p[i]))
            i += 1
    return re.compile("^" + "".join(parts) + "$")


def tool_doc(t: ToolDescriptor) -> str:
    req = t.parameters.get("required") or []
    props = t.parameters.get("properties") or {}
    params = []
    for name, spec in props.items():
        mark = "" if name in req else "?"
        params.append(f"{name}{mark}: {(spec or {}).get('type', 'Any')}")
    sig = ", ".join(params)
    return f"{t.id}({sig}) -> {t.returns}\n{t.description}"


def build_tool_descriptors() -> list[ToolDescriptor]:
    s = get_settings()
    tools = [
        ToolDescriptor(
            id="fs.read_file",
            group="fs",
            description="Read a text file from the knowledge workspace. Use start_line/end_line to page large files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to knowledge root"},
                    "start_line": {"type": "integer", "description": "First line (1-based)", "default": 1},
                    "end_line": {"type": "integer", "description": "Last line inclusive"},
                },
                "required": ["path"],
            },
            returns="str",
        ),
        ToolDescriptor(
            id="fs.list_dir",
            group="fs",
            description="List a directory under the knowledge workspace. Directories end with '/'.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory relative to knowledge root", "default": "."},
                },
            },
            returns="list[str]",
        ),
        ToolDescriptor(
            id="fs.glob",
            group="fs",
            description="Find files by glob pattern under knowledge/, e.g. 'guidelines/**/*.md'.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "max_results": {"type": "integer", "default": int(s.rlm_fs_glob_max_results)},
                },
                "required": ["pattern"],
            },
            returns="list[str]",
        ),
        ToolDescriptor(
            id="search.grep",
            group="search",
            description="Search file contents with a regular expression. Returns 'path:line: text' matches.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "glob": {"type": "string"},
                    "ignore_case": {"type": "boolean", "default": False},
                    "max_results": {"type": "integer", "default": int(s.rlm_search_max_results)},
                },
                "required": ["pattern"],
            },
            returns="list[str]",
        ),
        ToolDescriptor(
            id="fs.read_pdf_text",
            group="fs",
            description="Extract text from a PDF under knowledge/ (paged). Prefer search.grep on .md extracts when available.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_pages": {"type": "integer", "default": int(s.rlm_pdf_max_pages)},
                    "start_page": {"type": "integer", "default": 1},
                },
                "required": ["path"],
            },
            returns="str",
        ),
    ]
    for spec in TOOL_SPECS:
        name = spec["name"]
        desc = str(spec.get("description") or name)
        params = dict(spec.get("parameters") or {})
        # Clarify WHO vs INTERGROWTH routing for growth tools — the live
        # dispatch supports both via chart_standard / gestational_age_weeks.
        if name in {"growth_percentile", "overlay_growth_on_chart"}:
            desc = (
                "Compute growth percentiles/z-scores with deterministic WHO term "
                "or INTERGROWTH preterm equations (never invent numbers). "
                "Pass gestational_age_weeks and/or chart_standard="
                "'who_term'|'intergrowth_preterm'. For term babies use who_term "
                "with postnatal weeks or age_months; for preterm use "
                "intergrowth_preterm with postmenstrual weeks."
            )
            props = dict(params.get("properties") or {})
            props.setdefault(
                "gestational_age_weeks",
                {"type": "number", "description": "Gestational age at birth in weeks"},
            )
            props.setdefault(
                "chart_standard",
                {
                    "type": "string",
                    "enum": ["who_term", "intergrowth_preterm"],
                    "description": "Which chart equations to use",
                },
            )
            props.setdefault(
                "age_months",
                {"type": "number", "description": "Chronological age in months (WHO term)"},
            )
            params = {**params, "properties": props}
        tools.append(
            ToolDescriptor(
                id=f"clinical.{name}",
                group="clinical",
                description=desc,
                parameters=params,
                returns="dict",
                side_effects=name
                in {
                    "overlay_growth_on_chart",
                    "score_asq_questionnaire",
                    "score_mchat",
                },
            )
        )
    return tools


def render_tool_surface(tools: list[ToolDescriptor], compact: bool = False) -> str:
    lines: list[str] = []
    for t in tools:
        if compact:
            lines.append(f"- {t.id}: {t.description[:120]}")
        else:
            lines.append(tool_doc(t))
            lines.append("")
    return "\n".join(lines).strip()


class ToolExecutor:
    def __init__(self) -> None:
        self.descriptors = {t.id: t for t in build_tool_descriptors()}

    def list(self) -> list[ToolDescriptor]:
        return list(self.descriptors.values())

    def execute(self, tool_id: str, args: dict, ctx: ToolContext) -> dict:
        s = get_settings()
        allow = {
            x.strip()
            for x in (s.rlm_tool_allowlist or "").split(",")
            if x.strip()
        }
        group = tool_id.split(".", 1)[0] if "." in tool_id else tool_id
        if allow and group not in allow and tool_id not in allow:
            return {"ok": False, "error": f"tool group '{group}' is not allowed"}
        try:
            result = self._run(tool_id, args or {}, ctx)
            ctx.tool_calls.append({"name": tool_id, "arguments": args, "ok": True, "result": result})
            return {"ok": True, "result": result}
        except Exception as exc:
            log.warning("tool %s failed: %s", tool_id, exc)
            ctx.tool_calls.append({"name": tool_id, "arguments": args, "ok": False, "error": str(exc)})
            return {"ok": False, "error": str(exc)}

    def _run(self, tool_id: str, a: dict, ctx: ToolContext) -> Any:
        s = get_settings()
        if tool_id == "fs.read_file":
            asked = str(a.get("path") or "")
            path = resolve_inside(ctx.workspace_root, asked)
            if not path.is_file():
                alts = _similar_files(ctx.workspace_root, asked)
                if len(alts) == 1:
                    path = resolve_inside(ctx.workspace_root, alts[0])
                elif alts:
                    return (
                        f"[not found: {asked}; similar files: {', '.join(alts)}. "
                        "Read one of these, or search.grep the topic.]"
                    )
                else:
                    return (
                        f"[not found: {asked}; no similar filename. "
                        "Use search.grep on INDEX.md or the topic.]"
                    )
            if _is_probably_binary(path) and path.suffix.lower() not in {".md", ".txt", ".json", ".yaml", ".yml", ".csv"}:
                raise ValueError("binary file; not readable as text (try fs.read_pdf_text for PDFs)")
            lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
            start = max(1, int(a.get("start_line") or 1))
            end = min(len(lines), int(a["end_line"]) if a.get("end_line") else len(lines))
            text = "\n".join(lines[start - 1 : end])
            max_chars = int(s.rlm_fs_read_max_chars)
            if len(text) > max_chars:
                shown = text[:max_chars].count("\n") + 1
                text = (
                    f"{text[:max_chars]}\n[truncated: file has {len(lines)} lines; "
                    f"showed lines {start}-{start + shown - 1}. Read further with start_line={start + shown}]"
                )
            elif start > 1 or end < len(lines):
                text += f"\n[lines {start}-{end} of {len(lines)}]"
            return text

        if tool_id == "fs.list_dir":
            path = resolve_inside(ctx.workspace_root, str(a.get("path") or "."))
            if not path.is_dir():
                raise NotADirectoryError(str(a.get("path")))
            entries = []
            for e in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if e.name.startswith("."):
                    continue
                entries.append(e.name + ("/" if e.is_dir() else ""))
            return entries

        if tool_id == "fs.glob":
            pattern = str(a.get("pattern") or "")
            max_results = int(a.get("max_results") or s.rlm_fs_glob_max_results)
            cre = _glob_to_re(pattern.lstrip("./"))
            out: list[str] = []
            for f in ctx.workspace_root.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(ctx.workspace_root).as_posix()
                if cre.match(rel):
                    out.append(rel)
                    if len(out) >= max_results:
                        break
            return out

        if tool_id == "search.grep":
            max_results = int(a.get("max_results") or s.rlm_search_max_results)
            max_bytes = int(s.rlm_grep_file_max_bytes)
            base = resolve_inside(ctx.workspace_root, str(a.get("path") or "."))
            orig_pat = str(a.get("pattern") or "")
            glob = _relax_glob(str(a["glob"]) if a.get("glob") else None)

            def scan(pat: str, ignore_case: bool, glob_pat: str | None) -> list[str]:
                flags = re.I if ignore_case else 0
                try:
                    cre = re.compile(pat, flags)
                except re.error as exc:
                    raise ValueError(f"invalid regex: {exc}") from exc
                glob_re = _glob_to_re(glob_pat) if glob_pat else None
                hits: list[str] = []
                files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
                tok = [t.lower() for t in _search_tokens(orig_pat)]
                files.sort(
                    key=lambda f: (
                        0 if any(t in f.name.lower() for t in tok) else 1,
                        f.suffix.lower() != ".md",
                        f.as_posix(),
                    )
                )
                for f in files:
                    rel = f.relative_to(ctx.workspace_root).as_posix()
                    if glob_re and not glob_re.match(rel):
                        continue
                    try:
                        if f.stat().st_size > max_bytes:
                            continue
                    except OSError:
                        continue
                    if _is_probably_binary(f):
                        continue
                    try:
                        lines = f.read_text(encoding="utf-8", errors="replace").split("\n")
                    except OSError:
                        continue
                    for n, line in enumerate(lines, 1):
                        if cre.search(line):
                            hits.append(f"{rel}:{n}: {line.strip()[:300]}")
                            if len(hits) >= max_results:
                                return hits
                return hits

            ignore = bool(a.get("ignore_case"))
            out = scan(orig_pat, ignore, glob)
            note = ""
            if not out and not ignore:
                out = scan(orig_pat, True, glob)
                if out:
                    note = f"[search.grep] 0 hits for {orig_pat!r}; retried ignore_case=True ({len(out)} hits)."
            if not out:
                tokens = _search_tokens(orig_pat)
                if len(tokens) >= 2:
                    alt = "|".join(re.escape(t) for t in tokens)
                    out = scan(alt, True, glob)
                    if out:
                        note = (
                            f"[search.grep] phrase {orig_pat!r} missed; "
                            f"broadened to /{alt}/i ({len(out)} hits)."
                        )
                elif len(tokens) == 1:
                    out = scan(re.escape(tokens[0]), True, glob)
                    if out:
                        note = f"[search.grep] broadened to /{tokens[0]}/i ({len(out)} hits)."
            if not out and glob:
                tokens = _search_tokens(orig_pat) or [orig_pat]
                alt = orig_pat if len(tokens) < 2 else "|".join(re.escape(t) for t in tokens)
                out = scan(alt, True, None)
                if out:
                    note = f"[search.grep] dropped glob {glob!r}; {len(out)} hits."
            if note:
                return [note, *out]
            return out

        if tool_id == "fs.read_pdf_text":
            path = resolve_inside(ctx.workspace_root, str(a.get("path") or ""))
            if not path.is_file():
                raise FileNotFoundError(str(a.get("path")))
            start = max(1, int(a.get("start_page") or 1))
            max_pages = int(a.get("max_pages") or s.rlm_pdf_max_pages)
            try:
                import fitz  # pymupdf
            except ImportError as exc:
                raise RuntimeError("pymupdf is required to read PDFs") from exc
            doc = fitz.open(path)
            try:
                end = min(len(doc), start - 1 + max_pages)
                parts = []
                for i in range(start - 1, end):
                    parts.append(f"--- page {i + 1} ---\n{doc[i].get_text()}")
                return "\n".join(parts) or "(no extractable text)"
            finally:
                doc.close()

        if tool_id.startswith("clinical."):
            name = tool_id.split(".", 1)[1]
            args = dict(a)
            if ctx.child_id and "child_id" not in args and name in {
                "overlay_growth_on_chart",
                "get_child_summary",
            }:
                args["child_id"] = ctx.child_id
            result = dispatch_tool(name, args, db=ctx.db)
            # Keep JSON-serializable for the kernel.
            return json.loads(json.dumps(result, default=str))

        raise ValueError(f"unknown tool: {tool_id}")
