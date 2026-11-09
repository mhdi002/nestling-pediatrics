"""Parse and filter executable fenced cells from model replies."""

from __future__ import annotations

import re

from assistant.settings import get_settings

_ANY_FENCE_RE = re.compile(
    r"```([A-Za-z0-9_+-]*)[^\n]*\n([\s\S]*?)(?:^|\n)```[ \t]*(?=\n|$)",
    re.MULTILINE,
)
_KWARG_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*("
    r"'[^']*'|\"[^\"]*\"|-?\d+\.?\d*|True|False|None"
    r")"
)
_PATH_RE = re.compile(
    r"\b([A-Za-z0-9_./-]+\.(?:md|txt|json|csv|ya?ml|pdf))\b"
)
_TOPIC_RE = re.compile(
    r"(?:look up|look in|grep(?:\s+for)?|search(?:\s+\w+)?\s+for|say about|"
    r"about|guidance for(?: a)?)\s+([^.\n]{3,80}?)(?:\s+(?:in|for|with|and|using)\b|$)",
    re.IGNORECASE,
)


def fence_labels() -> list[str]:
    raw = (get_settings().rlm_cell_fence_labels or "repl").strip()
    labels = [x.strip().lower() for x in raw.split(",") if x.strip()]
    return labels or ["repl"]


def call_prefixes() -> list[str]:
    s = get_settings()
    raw = (s.rlm_call_prefixes or s.rlm_tool_allowlist or "fs,search,clinical").strip()
    prefixes = [x.strip() for x in raw.split(",") if x.strip()]
    return prefixes or ["fs", "search", "clinical"]


def _cell_re() -> re.Pattern[str]:
    labels = "|".join(re.escape(x) for x in fence_labels())
    return re.compile(
        rf"```(?:{labels})[^\n]*\n([\s\S]*?)(?:^|\n)```[ \t]*(?=\n|$)",
        re.MULTILINE | re.IGNORECASE,
    )


def looks_executable(code: str) -> bool:
    body = code or ""
    prefixes = "|".join(re.escape(p) for p in call_prefixes())
    if re.search(rf"\b(?:{prefixes})\.\w+\s*\(", body):
        return True
    if re.search(r"\b(print\s*\(|import json\b)", body):
        return True
    return False


def extract_cells(text: str) -> list[str]:
    cells: list[str] = []
    seen: set[str] = set()
    labels = set(fence_labels())
    for m in _ANY_FENCE_RE.finditer(text or ""):
        label = (m.group(1) or "").strip().lower()
        code = (m.group(2) or "").rstrip()
        if not code.strip():
            continue
        if label in labels or (not label and looks_executable(code)):
            key = code.strip()
            if key not in seen:
                seen.add(key)
                cells.append(code)
    if cells:
        return cells
    return extract_naked_calls(text or "")


def _balanced_call(src: str, start: int) -> str | None:
    i = src.find("(", start)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(src)):
        ch = src[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return src[start:j + 1]
        elif ch in "\n" and depth == 0:
            break
    return None


def extract_naked_calls(text: str) -> list[str]:
    """Recover `clinical.foo(...)` / `search.grep(...)` written outside a fence."""
    src = text or ""
    cells: list[str] = []
    seen: set[str] = set()
    prefixes = "|".join(re.escape(p) for p in call_prefixes())
    for m in re.finditer(rf"\b(?:{prefixes})\.\w+", src):
        call = _balanced_call(src, m.start())
        if not call or "(" not in call:
            continue
        wrapped = f"import json\nprint(json.dumps({call}, default=str))"
        if wrapped not in seen:
            seen.add(wrapped)
            cells.append(wrapped)
    return cells[:4]


def _wrap_call(call: str) -> str:
    return f"import json\nprint(json.dumps({call}, default=str))"


def fallback_cells(user_text: str, assistant_text: str = "") -> list[str]:
    """Build a REPL cell from a tool the parent asked for when the model did not fence one."""
    naked = extract_naked_calls(f"{assistant_text}\n{user_text}")
    if naked:
        return naked[:2]
    blob = f"{assistant_text}\n{user_text}"
    prefixes = "|".join(re.escape(p) for p in call_prefixes())
    named = re.search(rf"\b((?:{prefixes})\.\w+)", blob)
    if named:
        kwargs = _KWARG_RE.findall(blob)
        if kwargs:
            args = ", ".join(f"{k}={v}" for k, v in kwargs)
            return [_wrap_call(f"{named.group(1)}({args})")]
    path_m = _PATH_RE.search(user_text or "")
    if path_m and re.search(r"fs\.read|INDEX\.md|read\s+\S+\.md", user_text or "", re.I):
        path = path_m.group(1).lstrip("./")
        if path.startswith("knowledge/"):
            path = path[len("knowledge/") :]
        return [f'print(fs.read_file(path="{path}"))']
    if re.search(
        r"search\.grep|fs\.read|knowledge|guidelines|INDEX\.md|\bgrep\b|\blook(?:\s+up|\s+in)\b|\bsearch\b",
        user_text or "",
        re.I,
    ):
        topic = _grep_topic(user_text or "")
        if topic:
            return [
                f'print(search.grep(pattern={topic!r}, glob="**/*.md", ignore_case=True, max_results=25))'
            ]
    return []


def _grep_topic(user_text: str) -> str:
    quoted = re.findall(r"['\"]([^'\"]{2,40})['\"]", user_text)
    skip = {"repl", "python", "search.grep", "fs.read_file"}
    for q in quoted:
        if q.lower() not in skip and not q.startswith("`"):
            return q
    m = _TOPIC_RE.search(user_text)
    if m:
        topic = re.sub(r"\s+", " ", m.group(1)).strip(" ,:;/-")
        topic = re.split(r"\s+(?:in|for|with|using|from)\b", topic, maxsplit=1)[0].strip()
        if 2 <= len(topic) <= 60:
            return topic
    words = re.findall(r"[A-Za-z\u0600-\u06FF]{4,}", user_text)
    filler = {
        w.strip().lower()
        for w in (get_settings().rlm_grep_topic_stopwords or "").split(",")
        if w.strip()
    }
    picked = [w for w in words if w.lower() not in filler]
    return " ".join(picked[:4]) if picked else (words[0] if words else "care")


def strip_cells(text: str) -> str:
    out = _cell_re().sub("", text or "")
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def unclosed_cell_at(text: str) -> int:
    rest = _cell_re().sub(lambda m: " " * len(m.group(0)), text or "")
    earliest = -1
    for label in fence_labels():
        token = "```" + label
        at = rest.lower().find(token)
        if at >= 0 and (earliest < 0 or at < earliest):
            earliest = at
    return earliest


def strip_thinking(text: str) -> str:
    t = text or ""
    t = re.sub(r"<think>[\s\S]*?</think>", "", t)
    t = re.sub(r"^[\s\S]*?</think>", "", t)
    t = re.sub(r"<think>[\s\S]*$", "", t)
    return t.strip()


def timeout_hint(limit_sec: int) -> str:
    return (
        f"[cell timed out after {limit_sec}s] repl cells are for quick steps; "
        "this work needs longer than the cell limit. Split into smaller reads/searches."
    )


def syntax_hint(code: str, error: str) -> str:
    if "SyntaxError" not in (error or ""):
        return ""
    if re.search(
        r"^\s*(const|let|var|function|import .* from|export )\b|console\.log\(|require\(|=>",
        code or "",
        re.M,
    ):
        return "[hint] repl blocks run Python only."
    return ""


def same_error_hint(prev_error: str, error: str) -> str:
    def last_line(s: str) -> str:
        lines = [l for l in (s or "").strip().split("\n") if l.strip()]
        return lines[-1].strip() if lines else ""

    a = last_line(prev_error)
    if not a or a != last_line(error):
        return ""
    return f"[hint] Same error as the previous attempt ({a[:200]}); change approach."


class CellFenceFilter:
    """Streaming filter: pass prose, swallow configured executable fences."""

    def __init__(self) -> None:
        self.buf = ""
        self.in_cell = False
        self._opens = ["```" + lab for lab in fence_labels()]
        self._close = re.compile(r"(?:^|\n)```[ \t]*(?=\n|$)")

    def _find_open(self, s: str) -> int:
        best = -1
        low = s.lower()
        for token in self._opens:
            at = low.find(token.lower())
            if at >= 0 and (best < 0 or at < best):
                best = at
        return best

    def _open_len_at(self, s: str, i: int) -> int:
        low = s.lower()
        for token in self._opens:
            if low.startswith(token.lower(), i):
                return len(token)
        return 0

    def push(self, chunk: str) -> str:
        self.buf += chunk
        out = ""
        while True:
            if not self.in_cell:
                i = self._find_open(self.buf)
                if i == -1:
                    hold = 0
                    for token in self._opens:
                        hold = max(hold, self._partial_suffix(self.buf, token))
                    out += self.buf[: len(self.buf) - hold]
                    self.buf = self.buf[len(self.buf) - hold :]
                    return out
                out += self.buf[:i]
                n = self._open_len_at(self.buf, i)
                self.buf = self.buf[i + n :]
                self.in_cell = True
            else:
                nl = self.buf.find("\n")
                if nl == -1:
                    return out
                m = self._close.search(self.buf[nl:])
                if not m:
                    return out
                end = nl + m.start() + len(m.group(0))
                if end >= len(self.buf):
                    return out
                self.buf = self.buf[end:]
                self.in_cell = False

    def draft(self) -> str | None:
        if not self.in_cell:
            return None
        nl = self.buf.find("\n")
        return "" if nl == -1 else self.buf[nl + 1 :]

    def flush(self) -> str:
        rest = "" if self.in_cell else self.buf
        self.buf = ""
        self.in_cell = False
        return rest

    @staticmethod
    def _partial_suffix(s: str, token: str) -> int:
        sl = s.lower()
        tl = token.lower()
        for n in range(min(len(token) - 1, len(s)), 0, -1):
            if tl.startswith(sl[-n:]):
                return n
        return 0
