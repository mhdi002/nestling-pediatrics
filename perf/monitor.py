"""Sample CPU/RAM of the app server and of Locust itself during a run.

Separating the two is the whole point: if Locust is pegged and the server is
idle, the measured ceiling is the test machine, not the app.
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from perf import config as cfg


def find_processes(match: str) -> list[psutil.Process]:
    """Processes matching ``match``.

    ``pid:1234`` targets one process tree exactly — needed on a dev box where a
    second, unrelated uvicorn may be running and would pollute the numbers.
    Anything else is a case-insensitive command-line substring.
    """
    if match.lower().startswith("pid:"):
        try:
            return [psutil.Process(int(match.split(":", 1)[1]))]
        except (ValueError, psutil.NoSuchProcess):
            return []
    needle = match.lower()
    found: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or []).lower()
            if needle in cmd:
                found.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def _tree(procs: list[psutil.Process]) -> list[psutil.Process]:
    out: dict[int, psutil.Process] = {}
    for p in procs:
        try:
            out[p.pid] = p
            for child in p.children(recursive=True):
                out[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return list(out.values())


@dataclass
class Sample:
    t: float
    cpu_percent: float
    rss_mb: float
    threads: int
    num_procs: int


@dataclass
class GroupStats:
    label: str
    samples: list[Sample] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.samples:
            return {"label": self.label, "samples": 0}
        cpus = sorted(s.cpu_percent for s in self.samples)
        rss = [s.rss_mb for s in self.samples]
        return {
            "label": self.label,
            "samples": len(self.samples),
            "cpu_percent_avg": round(sum(cpus) / len(cpus), 1),
            "cpu_percent_p95": round(cpus[int(len(cpus) * 0.95) - 1], 1),
            "cpu_percent_max": round(cpus[-1], 1),
            "rss_mb_start": round(rss[0], 1),
            "rss_mb_end": round(rss[-1], 1),
            "rss_mb_max": round(max(rss), 1),
            "rss_growth_mb": round(rss[-1] - rss[0], 1),
            "threads_max": max(s.threads for s in self.samples),
            "procs_seen": max(s.num_procs for s in self.samples),
        }


class ResourceMonitor:
    """Background sampler. ``cpu_percent`` is normalised to whole-machine percent."""

    def __init__(self, groups: dict[str, str], interval: float | None = None):
        self.interval = interval or cfg.MONITOR_INTERVAL
        self.groups = {label: GroupStats(label) for label in groups}
        self._matches = groups
        self._procs: dict[str, list[psutil.Process]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu_count = psutil.cpu_count(logical=True) or 1

    def _refresh(self) -> None:
        for label, match in self._matches.items():
            procs = _tree(find_processes(match))
            for p in procs:
                try:
                    p.cpu_percent(None)  # prime the per-process delta
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self._procs[label] = procs

    def _sample_once(self) -> None:
        now = time.time()
        for label, procs in self._procs.items():
            cpu = 0.0
            rss = 0.0
            threads = 0
            alive = 0
            for p in procs:
                try:
                    cpu += p.cpu_percent(None)
                    mem = p.memory_info()
                    rss += mem.rss / (1024 * 1024)
                    threads += p.num_threads()
                    alive += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if alive:
                self.groups[label].samples.append(
                    Sample(
                        t=now,
                        cpu_percent=round(cpu / self.cpu_count, 2),
                        rss_mb=round(rss, 2),
                        threads=threads,
                        num_procs=alive,
                    )
                )

    def _loop(self) -> None:
        self._refresh()
        time.sleep(self.interval)
        ticks = 0
        while not self._stop.is_set():
            self._sample_once()
            ticks += 1
            # Re-discover periodically so worker respawns / new replicas are picked up.
            if ticks % 15 == 0:
                self._refresh()
            self._stop.wait(self.interval)

    def start(self) -> "ResourceMonitor":
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval * 3)
        return {label: g.summary() for label, g in self.groups.items()}

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["group", "timestamp", "cpu_percent", "rss_mb", "threads", "procs"])
            for label, group in self.groups.items():
                for s in group.samples:
                    writer.writerow(
                        [label, round(s.t, 2), s.cpu_percent, s.rss_mb, s.threads, s.num_procs]
                    )

    def __enter__(self) -> "ResourceMonitor":
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False
