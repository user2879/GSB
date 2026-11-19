"""Candidate previews, interaction budgets and atomic snapshot swapping."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .constants import DEFAULT_PREVIEW_BUDGET_MS, PREVIEW_BUDGET_MS
from .errors import BusinessError, ErrorCode
from .io import atomic_write_json, sha256_json
from .model import DefectEntry, Snapshot, Thresholds
from .pipeline import apply_incremental, check_consistency, rebuild_full_for_consistency
from .persistence import StateStore, utc_now


@dataclass(frozen=True)
class PreviewResult:
    base_content_hash: str
    base_version_label: str
    candidate: Snapshot
    elapsed_ms: float
    budget_ms: int
    over_budget: bool
    frame_timings_ms: dict[int, float]
    changed_keys: tuple[str, ...]
    budget_error: dict[str, Any] | None

    def raise_if_over_budget(self) -> None:
        if self.over_budget:
            raise BusinessError(
                ErrorCode.E_BUDGET_EXCEEDED,
                "candidate preview exceeded the declared interaction budget",
                {
                    "elapsed_ms": self.elapsed_ms,
                    "budget_ms": self.budget_ms,
                    "base_content_hash": self.base_content_hash,
                    "base_version_label": self.base_version_label,
                },
            )


class PerformanceRecorder:
    """Persists per-frame timings and cumulative over-budget counts."""

    def __init__(self, workspace: Path, namespace: str = "default") -> None:
        self.workspace = Path(workspace).resolve()
        self.path = self.workspace / "runtime" / "state" / namespace / "performance.json"
        self.events_dir = self.workspace / "runtime" / "logs" / "interaction"
        self._lock = threading.Lock()
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "schema": "mocap-performance/v1",
                "preview_count": 0,
                "over_budget_count": 0,
                "max_elapsed_ms": 0.0,
                "last": None,
            }
        from .io import read_json

        return read_json(self.path)

    def record(self, result: PreviewResult, operation: str) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
            state["preview_count"] = int(state.get("preview_count", 0)) + 1
            if result.over_budget:
                state["over_budget_count"] = int(state.get("over_budget_count", 0)) + 1
            state["max_elapsed_ms"] = max(float(state.get("max_elapsed_ms", 0.0)), result.elapsed_ms)
            payload = {
                "operation": operation,
                "base_content_hash": result.base_content_hash,
                "base_version_label": result.base_version_label,
                "candidate_content_hash": result.candidate.content_hash,
                "elapsed_ms": round(result.elapsed_ms, 6),
                "budget_ms": result.budget_ms,
                "over_budget": result.over_budget,
                "over_budget_count": state["over_budget_count"],
                "frame_timings_ms": {
                    str(frame): round(value, 6)
                    for frame, value in sorted(result.frame_timings_ms.items())
                },
                "timestamp_utc": utc_now(),
            }
            state["last"] = payload
            atomic_write_json(self.path, state)
            self.events_dir.mkdir(parents=True, exist_ok=True)
            event_name = sha256_json(payload)[:20]
            atomic_write_json(self.events_dir / f"{event_name}.json", payload)
            self._state = state
            return state

    @property
    def state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)


class PreviewEngine:
    def __init__(self, recorder: PerformanceRecorder | None = None) -> None:
        self.recorder = recorder

    def generate(
        self,
        base: Snapshot,
        truth_entries: tuple[DefectEntry, ...],
        thresholds: Thresholds | None = None,
        keyframes: frozenset[int] | set[int] | None = None,
        suppressed_auto: frozenset[int] | set[int] | None = None,
        operation: str = "preview",
    ) -> PreviewResult:
        budget_ms = PREVIEW_BUDGET_MS.get(base.frame_count, DEFAULT_PREVIEW_BUDGET_MS)
        timings: dict[int, float] = {}
        started = time.perf_counter()
        candidate = apply_incremental(
            base,
            truth_entries,
            thresholds=thresholds,
            keyframes=keyframes,
            sequence=base.version.sequence + 1,
            committed=False,
            suppressed_auto=suppressed_auto,
            frame_timings_ms=timings,
        )
        full = rebuild_full_for_consistency(candidate, truth_entries)
        candidate = replace(candidate, consistency=check_consistency(candidate, full))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        over_budget = elapsed_ms > budget_ms
        error = None
        if over_budget:
            error = BusinessError(
                ErrorCode.E_BUDGET_EXCEEDED,
                "candidate preview exceeded the declared interaction budget",
                {"elapsed_ms": elapsed_ms, "budget_ms": budget_ms},
            ).to_dict()
        result = PreviewResult(
            base_content_hash=base.content_hash,
            base_version_label=base.version.label,
            candidate=candidate,
            elapsed_ms=elapsed_ms,
            budget_ms=budget_ms,
            over_budget=over_budget,
            frame_timings_ms=timings,
            changed_keys=candidate.closure.changed_keys if candidate.closure else (),
            budget_error=error,
        )
        if self.recorder:
            self.recorder.record(result, operation)
        return result


class SnapshotManager:
    """Single-pointer committed view plus one background candidate preview."""

    def __init__(
        self,
        initial: Snapshot,
        engine: PreviewEngine | None = None,
    ) -> None:
        self._current = initial
        self._preview: PreviewResult | None = None
        self._pending: tuple[str, str, Future[PreviewResult]] | None = None
        self._last_error: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mocap-preview")
        self.engine = engine or PreviewEngine()
        self.render_count = 0

    def current(self) -> Snapshot:
        with self._lock:
            self.render_count += 1
            return self._current

    @property
    def preview(self) -> PreviewResult | None:
        with self._lock:
            return self._preview

    @property
    def last_error(self) -> dict[str, Any] | None:
        with self._lock:
            return self._last_error

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending is not None and not self._pending[2].done()

    def submit_preview(
        self,
        truth_entries: tuple[DefectEntry, ...],
        thresholds: Thresholds | None = None,
        keyframes: frozenset[int] | set[int] | None = None,
        suppressed_auto: frozenset[int] | set[int] | None = None,
        operation: str = "preview",
    ) -> None:
        with self._lock:
            base = self._current
            if self._pending is not None:
                self._pending[2].cancel()
            future = self._executor.submit(
                self.engine.generate,
                base,
                truth_entries,
                thresholds,
                keyframes,
                suppressed_auto,
                operation,
            )
            self._pending = (base.content_hash, base.version.label, future)
            self._last_error = None

    def poll(self) -> PreviewResult | None:
        with self._lock:
            if self._pending is None or not self._pending[2].done():
                return None
            base_hash, base_label, future = self._pending
            self._pending = None
            try:
                result = future.result()
            except Exception as exc:
                if isinstance(exc, BusinessError):
                    self._last_error = exc.to_dict()
                else:
                    self._last_error = {
                        "ok": False,
                        "code": "E_BUDGET_EXCEEDED" if isinstance(exc, TimeoutError) else "E_SNAPSHOT_STALE",
                        "message": str(exc),
                        "context": {},
                    }
                return None
            current_stale = (
                self._current.content_hash != base_hash
                or self._current.version.label != base_label
            )
            result_stale = (
                result.base_content_hash != base_hash
                or result.base_version_label != base_label
            )
            if current_stale or result_stale:
                self._last_error = BusinessError(
                    ErrorCode.E_SNAPSHOT_STALE,
                    "background candidate was based on a stale committed snapshot",
                    {
                        "expected_content_hash": base_hash,
                        "expected_version_label": base_label,
                        "result_content_hash": result.base_content_hash,
                        "result_version_label": result.base_version_label,
                        "current_content_hash": self._current.content_hash,
                        "current_version_label": self._current.version.label,
                    },
                ).to_dict()
                return None
            self._preview = result
            return result

    def commit_preview(self, store: StateStore, truth_entries: tuple[DefectEntry, ...]) -> Snapshot:
        with self._lock:
            preview = self._preview
            if preview is None:
                raise BusinessError(
                    ErrorCode.E_SNAPSHOT_STALE,
                    "there is no completed candidate preview to commit",
                    {},
                )
            if (
                preview.base_content_hash != self._current.content_hash
                or preview.base_version_label != self._current.version.label
            ):
                raise BusinessError(
                    ErrorCode.E_SNAPSHOT_STALE,
                    "candidate preview no longer matches the committed base",
                    {
                        "base_content_hash": preview.base_content_hash,
                        "base_version_label": preview.base_version_label,
                        "current_content_hash": self._current.content_hash,
                        "current_version_label": self._current.version.label,
                    },
                )
            committed = replace(preview.candidate, committed=True)
            if committed.consistency is not None and not committed.consistency.consistent:
                raise BusinessError(
                    ErrorCode.E_CLOSURE_MISMATCH,
                    "candidate consistency check failed; commit refused",
                    committed.consistency.first_mismatch or {},
                )
            # Persist first; only a successful ledger commit swaps the in-memory
            # single pointer. Readers therefore never observe a mixed snapshot.
            store.commit(committed, truth_entries)
            self._current = committed
            self._preview = None
            return committed

    def swap_committed(self, snapshot: Snapshot) -> None:
        with self._lock:
            self._current = snapshot
            self._preview = None

    def shutdown(self) -> None:
        with self._lock:
            if self._pending is not None:
                self._pending[2].cancel()
            self._executor.shutdown(wait=False, cancel_futures=True)
