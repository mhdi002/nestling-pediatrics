"""IPython REPL kernel host — JSONL protocol matching vendor/rlm-ide/python/rlm_worker."""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from assistant.rlm.tools import ToolDescriptor, tool_doc
from assistant.settings import get_settings

log = logging.getLogger(__name__)


@dataclass
class CellResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    result: str = ""
    error: str | None = None
    timed_out: bool = False
    interrupted: bool = False


HostRequestHandler = Callable[[str, dict], Any]


def worker_script() -> Path:
    s = get_settings()
    override = getattr(s, "rlm_worker_script", None)
    if override:
        return Path(override).resolve()
    root = Path(s.nestling_root).resolve()
    canonical = root / "assistant" / "rlm" / "worker" / "worker.py"
    if canonical.is_file():
        return canonical
    raise FileNotFoundError(f"rlm_worker worker.py not found: {canonical}")


class PythonKernel:
    def __init__(
        self,
        *,
        cwd: Path,
        tools: list[ToolDescriptor],
        on_host_request: HostRequestHandler,
        output_limit_chars: int,
    ) -> None:
        self.cwd = Path(cwd).resolve()
        self.tools = tools
        self.on_host_request = on_host_request
        self.output_limit_chars = output_limit_chars
        self.proc: subprocess.Popen[str] | None = None
        self.alive = False
        self._pending: dict[str, dict] = {}
        self._waiters: dict[str, threading.Event] = {}
        self._waiter_errors: dict[str, str] = {}
        self._cell_seq = 0
        self._lock = threading.Lock()
        self._stderr_tail = ""
        self.id = f"k_{int(time.time() * 1000) % 10_000_000}"

    def set_host_handler(self, on_host_request: HostRequestHandler) -> None:
        """Re-bind host tool routing for the active turn (kernel outlives turns)."""
        self.on_host_request = on_host_request

    def is_alive(self) -> bool:
        return self.alive and self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        s = get_settings()
        script = worker_script()
        py = s.rlm_python or shutil.which("python3") or "python3"
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "RLM_MEMORY_MB": str(int(s.rlm_kernel_memory_mb)),
        }
        base_cmd = [py, "-u", str(script)]
        # Prefer bubblewrap when available and requested; fall back to unsandboxed
        # when bwrap is blocked (common in restricted kernels). File tools stay
        # confined to the workspace by resolve_inside regardless.
        candidates: list[list[str]] = []
        if s.rlm_sandbox_prefer_bwrap and shutil.which("bwrap"):
            bwrap = [
                "bwrap",
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--bind",
                str(self.cwd),
                str(self.cwd),
                "--die-with-parent",
                "--chdir",
                str(self.cwd),
            ]
            if s.rlm_sandbox_no_network:
                bwrap.append("--unshare-net")
            candidates.append(bwrap + base_cmd)
        candidates.append(base_cmd)

        last_err: Exception | None = None
        for cmd in candidates:
            self._stderr_tail = ""
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(self.cwd),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            threading.Thread(target=self._read_stdout, daemon=True).start()
            threading.Thread(target=self._read_stderr, daemon=True).start()
            try:
                self._wait_for("ready", int(s.rlm_kernel_ready_timeout_sec))
                last_err = None
                if cmd is not candidates[-1]:
                    log.info("Python kernel started under sandbox: %s", cmd[0])
                break
            except Exception as exc:
                last_err = exc
                log.warning("kernel start failed with %s (%s); trying next mode", cmd[0], exc)
                try:
                    if self.proc and self.proc.poll() is None:
                        self.proc.kill()
                except Exception:
                    pass
                self.proc = None
                self._waiters.clear()
                continue
        if last_err is not None:
            raise last_err
        self.alive = True
        self._send(
            {
                "op": "bootstrap",
                "output_limit": self.output_limit_chars,
                "rewrite_guard": {
                    "min_lines": int(s.rlm_rewrite_min_lines),
                    "max_changed": int(s.rlm_rewrite_max_changed),
                },
                "tools": [
                    {
                        "id": t.id,
                        "params": sorted(
                            (t.parameters.get("properties") or {}).keys(),
                            key=lambda k: 0 if k in (t.parameters.get("required") or []) else 1,
                        ),
                        "doc": tool_doc(t),
                    }
                    for t in self.tools
                ],
            }
        )
        self._wait_for("bootstrapped", int(s.rlm_kernel_ready_timeout_sec))

    def _wait_for(self, op: str, timeout_sec: float) -> None:
        ev = threading.Event()
        self._waiters[op] = ev
        if not ev.wait(timeout_sec):
            self._waiters.pop(op, None)
            raise TimeoutError(f"timeout waiting for {op}: {self._stderr_tail}")
        err = self._waiter_errors.pop(op, None)
        if err:
            raise RuntimeError(err)

    def _send(self, obj: dict) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("kernel not started")
        with self._lock:
            self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._on_msg(msg)

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for chunk in self.proc.stderr:
            self._stderr_tail = (self._stderr_tail + chunk)[-4000:]

    def _on_msg(self, msg: dict) -> None:
        op = str(msg.get("op") or "")
        w = self._waiters.pop(op, None)
        if w:
            w.set()
            return
        if op == "stream":
            pend = self._pending.get(str(msg.get("id")))
            if pend and pend.get("on_output"):
                pend["on_output"](
                    {
                        "stream": "stderr" if msg.get("stream") == "stderr" else "stdout",
                        "text": str(msg.get("text") or ""),
                    }
                )
        elif op == "exec_result":
            mid = str(msg.get("id"))
            pend = self._pending.pop(mid, None)
            if not pend:
                return
            if msg.get("result") and pend.get("on_output"):
                pend["on_output"]({"stream": "result", "text": str(msg.get("result"))})
            pend["event"].set()
            pend["result"] = CellResult(
                ok=bool(msg.get("ok")),
                stdout=str(msg.get("stdout") or ""),
                stderr=str(msg.get("stderr") or ""),
                result=str(msg.get("result") or ""),
                error=msg.get("error"),
                timed_out=False,
                interrupted=bool(msg.get("interrupted")),
            )
        elif op == "host_request":
            rid = str(msg.get("id"))
            rtype = str(msg.get("type") or "")
            payload = msg.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {}

            def reply() -> None:
                try:
                    result = self.on_host_request(rtype, payload)
                    self._send({"op": "host_response", "id": rid, "ok": True, "result": result})
                except Exception as exc:
                    self._send(
                        {
                            "op": "host_response",
                            "id": rid,
                            "ok": False,
                            "error": str(exc),
                        }
                    )

            threading.Thread(target=reply, daemon=True).start()

    def execute(
        self,
        code: str,
        on_output: Callable[[dict], None] | None,
        timeout_ms: int,
    ) -> CellResult:
        if not self.is_alive():
            self.start()
        self._cell_seq += 1
        cid = f"c{self._cell_seq}"
        ev = threading.Event()
        holder: dict[str, Any] = {"event": ev, "on_output": on_output, "result": None}
        self._pending[cid] = holder
        self._send({"op": "exec", "id": cid, "code": code})
        if not ev.wait(timeout_ms / 1000.0):
            self.interrupt()
            self._pending.pop(cid, None)
            return CellResult(
                ok=False,
                error=f"cell timed out after {round(timeout_ms / 1000)}s",
                timed_out=True,
                interrupted=True,
            )
        return holder.get("result") or CellResult(ok=False, error="no result")

    def interrupt(self) -> None:
        try:
            self._send({"op": "interrupt"})
        except Exception:
            if self.proc and self.proc.poll() is None:
                self.proc.send_signal(signal.SIGINT)

    def shutdown(self) -> None:
        try:
            self._send({"op": "shutdown"})
        except Exception:
            pass
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
        self.alive = False
        self.proc = None


class KernelFactory:
    def create(
        self,
        *,
        cwd: Path,
        tools: list[ToolDescriptor],
        on_host_request: HostRequestHandler,
        output_limit_chars: int,
    ) -> PythonKernel:
        k = PythonKernel(
            cwd=cwd,
            tools=tools,
            on_host_request=on_host_request,
            output_limit_chars=output_limit_chars,
        )
        k.start()
        return k
