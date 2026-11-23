"""并行调度、预算控制、阶段化快照与续跑。

可观测点按“窗口训练长度”计费；调度器维护全局剩余预算并按序列 ID 稳定顺序
分配单序列配额，超配额即中止该序列并保留待办游标，记 E_SERIES_BUDGET_EXHAUSTED。
线程池并发数严格不超过配置；所有产物先写临时目录再整体替换。
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .cache import CacheStore, Ledger
from .config import Config
from .engine import process_window
from .errors import WorkbenchError, E_SERIES_BUDGET_EXHAUSTED, E_GLOBAL_BUDGET_EXHAUSTED
from .inject import Injection, apply_row_injections, window_injection_error
from .sample import read_jsonl
from .util import stable_sort_key
from .windows import build_windows


@dataclass
class RunResult:
    config_hash: str
    exchange_version: str
    results: list = field(default_factory=list)        # 已排序 WindowResult
    failures: list[dict] = field(default_factory=list)
    ledger: Ledger = field(default_factory=Ledger)
    cursors: dict[str, int] = field(default_factory=dict)
    global_budget_left: int = 0
    injected: list[str] = field(default_factory=list)
    resumed: bool = False


class Scheduler:
    def __init__(self, cfg: Config, cache: CacheStore, ledger: Ledger,
                 injections: list[Injection]):
        self.cfg = cfg
        self.cache = cache
        self.ledger = ledger
        self.injections = injections
        self.global_left = cfg.execution.observation_budget
        self.series_left = {s: cfg.execution.series_quota for s in self._injected_series()}
        self._lock = threading.Lock()
        self.failures: list[dict] = []
        self.cursors: dict[str, int] = {}
        self.completed_windows = 0

    def _injected_series(self) -> list[str]:
        return sorted({inj.series_id for inj in self.injections})

    def _charge(self, series_id: str, cost: int) -> str | None:
        """线程安全扣费，返回触发的业务码（None 表示通过）。"""
        with self._lock:
            if self.series_left.get(series_id, 0) < cost:
                return E_SERIES_BUDGET_EXHAUSTED
            if self.global_left < cost:
                return E_GLOBAL_BUDGET_EXHAUSTED
            self.series_left[series_id] -= cost
            self.global_left -= cost
            return None

    def run(self, grouped_rows: dict[str, list[dict]]) -> RunResult:
        cfg = self.cfg
        all_results: list = []
        series_ids = sorted(grouped_rows.keys())
        # 确保注入但无数据的序列也在配额表中
        for sid in series_ids:
            self.series_left.setdefault(sid, cfg.execution.series_quota)

        tasks: list[tuple[str, int, object, int]] = []
        for sid in series_ids:
            rows = [dict(r) for r in grouped_rows[sid]]
            hit_injections = apply_row_injections(rows, self.injections)
            try:
                windows = build_windows(rows, cfg)
            except WorkbenchError as exc:
                if hit_injections:
                    exc.injection_id = hit_injections[0].injection_id
                self._fail(exc, sid)
                self.cursors[sid] = 0
                continue
            cost = cfg.window_length
            for win in windows:
                tasks.append((sid, win.window_index, win, cost))

        workers = min(cfg.execution.max_workers, max(1, len(tasks)))
        ordered: list = []
        if tasks:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._run_task, t) for t in tasks]
                for fut in futures:
                    item = fut.result()
                    if item is not None:
                        ordered.append(item)

        # 无论线程如何调度，最终严格按稳定排序键串行归并
        ordered.sort(key=lambda item: item[0].sort_key)
        for win_result, _ensemble in ordered:
            all_results.append(win_result)

        for sid in series_ids:
            done = max((w.window_index + 1 for w in all_results if w.series_id == sid),
                       default=self.cursors.get(sid, 0))
            self.cursors.setdefault(sid, done)

        return RunResult(
            config_hash=cfg.config_hash(),
            exchange_version=cfg.exchange_version,
            results=all_results,
            failures=self._sorted_failures(),
            ledger=self.ledger,
            cursors=dict(sorted(self.cursors.items())),
            global_budget_left=max(self.global_left, 0),
            injected=sorted({inj.injection_id for inj in self.injections}),
        )

    def _run_task(self, task):
        sid, window_index, win, cost = task
        # 窗口级注入（窗口超长 / 量化档位越界）
        inj_err = window_injection_error(
            sid, window_index, self.injections,
            available_len=len(win.timestamps), required_len=self.cfg.window_length)
        if inj_err is not None:
            self._fail(inj_err, sid, window_index)
            self._mark_cursor(sid, window_index)
            return None

        code = self._charge(sid, cost)
        if code is not None:
            exc = WorkbenchError(
                code,
                f"序列 {sid} 窗口 {window_index} 观测点预算耗尽",
                context={"series_id": sid, "window_index": window_index,
                         "cost": cost, "global_left": self.global_left,
                         "series_left": self.series_left.get(sid, 0)})
            self._fail(exc, sid, window_index)
            self._mark_cursor(sid, window_index)
            return None

        try:
            win_result, ensemble = process_window(win, self.cfg, self.cache, self.ledger)
            with self._lock:
                self.completed_windows += 1
                if self.completed_windows % self.cfg.execution.snapshot_every_windows == 0:
                    self.cache.flush(self.cfg.execution.temp_dir)
            return win_result, ensemble
        except WorkbenchError as exc:
            self._fail(exc, sid, window_index)
            self._mark_cursor(sid, window_index)
            return None

    def _mark_cursor(self, sid: str, window_index: int) -> None:
        with self._lock:
            self.cursors[sid] = max(self.cursors.get(sid, 0), window_index)

    def _fail(self, exc: WorkbenchError, sid: str, window_index: int | None = None) -> None:
        rec = exc.to_record()
        rec["series_id"] = sid
        if window_index is not None:
            rec["window_index"] = window_index
        rec["sort_key"] = stable_sort_key(sid, window_index if window_index is not None else -1,
                                          rec["code"])
        with self._lock:
            self.failures.append(rec)

    def _sorted_failures(self) -> list[dict]:
        return sorted(self.failures, key=lambda r: (r["series_id"],
                                                    r.get("window_index", -1),
                                                    r["code"],
                                                    r.get("injection_id", "")))


def write_snapshot(manifest_path: Path, run: RunResult) -> None:
    """阶段产物先写临时目录，再整体替换提交清单。"""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "global_version": 1,
        "config_hash": run.config_hash,
        "exchange_version_fingerprint": run.exchange_version,
        "completed_window_cursors": run.cursors,
        "failure_count": len(run.failures),
        "injection_ids": run.injected,
    }
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, manifest_path)


def can_resume(manifest_path: Path, cfg: Config) -> bool:
    """启动时以配置哈希与交换版本判定是否续跑。"""
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (manifest.get("config_hash") == cfg.config_hash()
            and manifest.get("exchange_version_fingerprint") == cfg.exchange_version)


def run_pipeline(cfg: Config, sample_path: str | Path, work_dir: str | Path,
                 injections: list[Injection] | None = None,
                 fresh: bool = False) -> RunResult:
    work_dir = Path(work_dir)
    cache_path = work_dir / "cache" / "content_cache.json"
    manifest_path = work_dir / "snapshot" / "manifest.json"
    if fresh and cache_path.exists():
        cache_path.unlink()
    if fresh and manifest_path.exists():
        manifest_path.unlink()

    resumed = (not fresh) and can_resume(manifest_path, cfg)
    cache = CacheStore(cache_path)
    ledger = Ledger()
    grouped = read_jsonl(sample_path)
    scheduler = Scheduler(cfg, cache, ledger, injections or [])
    run = scheduler.run(grouped)
    run.resumed = resumed
    cache.flush(Path(work_dir) / cfg.execution.temp_dir)
    write_snapshot(manifest_path, run)
    return run
