from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .canonical import content_hash, stage_key
from .checkpoint import (
    ENGINE_STATE_VERSION,
    CheckpointStore,
    LoadedCheckpoint,
    checkpoint_compat,
)
from .config import MAX_WINDOWS, WindowConfig, load_config, validate_max_windows
from .errors import CheckpointError, ConfigError
from .events import Event, EventLoadResult, ValidationFailure, load_events
from .features import (
    compute_visual,
    evidence_vectors,
    fusion_stage_key,
    fuse_payload,
    projection_matrix,
    select_frames,
    visual_stage_key,
)
from .plan import InjectionPlan, load_plan


REPORT_SCHEMA = "mmwin-report-v1"
LATE_BOUNDARY_RULE = (
    "late iff max_seen_timestamp_ms - event_timestamp_ms > allowed_lateness_ms; "
    "equality is on-time; close iff window_end_ms <= watermark_ms"
)


def _failure_sort_key(failure: dict[str, Any]) -> tuple[Any, ...]:
    return (
        failure.get("window_start_ms") if failure.get("window_start_ms") is not None else -1,
        failure.get("window_end_ms") if failure.get("window_end_ms") is not None else -1,
        failure.get("event_id") or "",
        failure.get("arrival_seq") if failure.get("arrival_seq") is not None else -1,
        failure.get("code") or "",
        failure.get("message") or "",
        failure.get("detail") or "",
    )


def canonical_business_projection(report: dict[str, Any]) -> dict[str, Any]:
    """Fields that must be byte-for-byte equivalent between incremental and full runs."""
    return {
        "schema": report.get("schema"),
        "meta": report.get("meta"),
        "watermark": report.get("watermark"),
        "delivery_order": report.get("delivery_order"),
        "windows": report.get("windows"),
        "resettlements": report.get("resettlements"),
        "empty_windows": report.get("empty_windows"),
        "failures": report.get("failures"),
        "pending": report.get("pending"),
        "invalidation_manifest": report.get("invalidation_manifest"),
    }


@dataclass
class Snapshot:
    snapshot_id: str
    reason: str
    event_ids: list[str]
    created_at_arrival_seq: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "reason": self.reason,
            "event_ids": list(self.event_ids),
            "created_at_arrival_seq": self.created_at_arrival_seq,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Snapshot":
        return cls(
            snapshot_id=value["snapshot_id"],
            reason=value["reason"],
            event_ids=list(value["event_ids"]),
            created_at_arrival_seq=value.get("created_at_arrival_seq"),
        )


@dataclass
class WindowState:
    window_id: str
    start_ms: int
    end_ms: int
    member_ids: list[str] = field(default_factory=list)
    closed: bool = False
    closed_at_arrival_seq: int | None = None
    snapshots: list[Snapshot] = field(default_factory=list)
    latest_snapshot_id: str | None = None

    @property
    def latest_snapshot(self) -> Snapshot | None:
        if self.latest_snapshot_id is None:
            return None
        for snapshot in self.snapshots:
            if snapshot.snapshot_id == self.latest_snapshot_id:
                return snapshot
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "member_ids": list(self.member_ids),
            "closed": self.closed,
            "closed_at_arrival_seq": self.closed_at_arrival_seq,
            "snapshots": [snapshot.as_dict() for snapshot in self.snapshots],
            "latest_snapshot_id": self.latest_snapshot_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WindowState":
        return cls(
            window_id=value["window_id"],
            start_ms=int(value["start_ms"]),
            end_ms=int(value["end_ms"]),
            member_ids=list(value.get("member_ids", [])),
            closed=bool(value.get("closed", False)),
            closed_at_arrival_seq=value.get("closed_at_arrival_seq"),
            snapshots=[Snapshot.from_dict(item) for item in value.get("snapshots", [])],
            latest_snapshot_id=value.get("latest_snapshot_id"),
        )


class AggregationEngine:
    def __init__(
        self,
        sample_dir: str | Path,
        config: WindowConfig,
        event_result: EventLoadResult,
        plan: InjectionPlan,
        checkpoint_dir: str | Path,
        *,
        batch_event_budget: int | None = None,
        reset: bool = False,
    ):
        self.sample_dir = Path(sample_dir).resolve()
        self.config = config
        self.event_result = event_result
        self.plan = plan
        if batch_event_budget is not None:
            if batch_event_budget <= 0:
                raise ValueError("batch_event_budget must be positive")
            self.batch_event_budget = int(batch_event_budget)
        else:
            self.batch_event_budget = config.batch_event_budget
        self.checkpoint = CheckpointStore(checkpoint_dir)
        self.reset_requested = reset
        self.diagnostics: list[dict[str, Any]] = []
        self.checkpoint_loaded: LoadedCheckpoint | None = None
        self.resume_used = False
        self.checkpoint_version = 0
        self.cache_entries: dict[str, Any] = {}
        self.previous_manifest: dict[str, Any] | None = None
        self.cache_stats: dict[str, int] = {
            "membership_hits": 0,
            "membership_misses": 0,
            "visual_hits": 0,
            "visual_misses": 0,
            "visual_computations": 0,
            "fusion_hits": 0,
            "fusion_misses": 0,
            "fusion_computations": 0,
            "resettle_hits": 0,
            "resettle_misses": 0,
        }
        self.scheduler_trace: list[dict[str, Any]] = []
        self.snapshot_visual: dict[str, dict[str, Any]] = {}
        self.snapshot_fusion: dict[str, dict[str, Any]] = {}
        self.snapshot_meta: dict[str, dict[str, Any]] = {}
        self.window_stage_status: dict[str, dict[str, str]] = {}
        self.window_stage_keys: dict[str, dict[str, Any]] = {}
        self.window_resettle_status: dict[str, str] = {}
        self.resettlement_records: list[dict[str, Any]] = []
        self.matrix = projection_matrix(config)

        self.windows: dict[str, WindowState] = {}
        self.event_windows: dict[str, list[str]] = {}
        self.window_order: list[str] = []
        self._build_windows()

        self.max_seen_ts: int | None = None
        self.watermark_ms: int | None = None
        self.delivered_ids: list[str] = []
        self._delivered_set: set[str] = set()
        self.processed_count = 0
        self.batch_index = 0
        self.stream_closed = False
        self.delivery_trace: list[dict[str, Any]] = []
        self.batch_trace: list[dict[str, Any]] = []
        self.replay_failures: list[dict[str, Any]] = []
        self.resolve_records: list[dict[str, Any]] = []
        self.snapshot_counter = 0
        self.closed_windows_since_commit = 0
        self.last_materialization: dict[str, Any] | None = None

        self._load_checkpoint()

    # ------------------------------------------------------------------
    # Window construction and checkpoint loading
    # ------------------------------------------------------------------
    def _build_windows(self) -> None:
        starts: set[int] = set()
        event_windows: dict[str, list[str]] = {}
        for event in self.event_result.events.values():
            lowest = max(0, (event.timestamp_ms - self.config.window_length_ms) // self.config.window_step_ms + 1)
            highest = event.timestamp_ms // self.config.window_step_ms
            ids: list[str] = []
            for index in range(lowest, highest + 1):
                start = index * self.config.window_step_ms
                end = start + self.config.window_length_ms
                if start <= event.timestamp_ms < end:
                    starts.add(start)
                    ids.append(self._window_id(start, end))
                    if len(starts) > MAX_WINDOWS:
                        raise ConfigError(
                            f"window count exceeds safety limit {MAX_WINDOWS}"
                        )
            event_windows[event.event_id] = ids
        validate_max_windows(len(starts))
        self.event_windows = event_windows
        self.window_order = []
        for start in sorted(starts):
            end = start + self.config.window_length_ms
            window_id = self._window_id(start, end)
            self.windows[window_id] = WindowState(window_id, start, end)
            self.window_order.append(window_id)

    @staticmethod
    def _window_id(start: int, end: int) -> str:
        return f"w-{start:010d}-{end:010d}"

    def _load_checkpoint(self) -> None:
        if self.reset_requested:
            self.checkpoint.reset()
            self.checkpoint_version = 0
            return
        loaded: LoadedCheckpoint | None = None
        try:
            loaded = self.checkpoint.load()
        except CheckpointError as exc:
            self.diagnostics.append(
                {
                    "code": "E_CHECKPOINT_VERSION_MISMATCH",
                    "message": str(exc),
                    "action": "full_replay",
                }
            )
            return
        if loaded is None:
            return
        self.checkpoint_loaded = loaded
        self.previous_manifest = loaded.manifest
        self.checkpoint_version = int(loaded.manifest.get("global_version", 0))
        cache_payload = loaded.payloads.get("cache")
        if isinstance(cache_payload, dict) and isinstance(cache_payload.get("entries"), dict):
            self.cache_entries = dict(cache_payload["entries"])
        compatible, code = checkpoint_compat(
            loaded.manifest,
            config_hash=self.config.config_hash,
            engine_version=ENGINE_STATE_VERSION,
        )
        if not compatible:
            self.diagnostics.append(
                {
                    "code": code or "E_CHECKPOINT_VERSION_MISMATCH",
                    "message": "checkpoint engine version or configuration hash does not match",
                    "action": "full_replay",
                }
            )
            return
        if loaded.manifest.get("plan_hash") != self.plan.plan_hash:
            self.diagnostics.append(
                {
                    "code": "E_CHECKPOINT_PLAN_CHANGED",
                    "message": "injection plan changed; replaying from arrival zero with content cache",
                    "action": "full_replay",
                }
            )
            return
        if loaded.manifest.get("source_hash") != self.event_result.source_hash:
            self.diagnostics.append(
                {
                    "code": "E_CHECKPOINT_SOURCE_CHANGED",
                    "message": "event source changed; replaying from arrival zero with content cache",
                    "action": "full_replay",
                }
            )
            return
        runtime = loaded.payloads.get("runtime")
        if not isinstance(runtime, dict) or not isinstance(runtime.get("state"), dict):
            self.diagnostics.append(
                {
                    "code": "E_CHECKPOINT_VERSION_MISMATCH",
                    "message": "checkpoint runtime state is missing or malformed",
                    "action": "full_replay",
                }
            )
            return
        state = runtime["state"]
        if state.get("engine_state_version") != ENGINE_STATE_VERSION:
            self.diagnostics.append(
                {
                    "code": "E_CHECKPOINT_VERSION_MISMATCH",
                    "message": "checkpoint state version mismatch",
                    "action": "full_replay",
                }
            )
            return
        try:
            self._restore_state(state, runtime)
        except (KeyError, TypeError, ValueError) as exc:
            self.diagnostics.append(
                {
                    "code": "E_CHECKPOINT_VERSION_MISMATCH",
                    "message": f"checkpoint state cannot be restored: {exc}",
                    "action": "full_replay",
                }
            )
            self._reset_replay_state()
            return
        self.resume_used = True

    def _reset_replay_state(self) -> None:
        for window in self.windows.values():
            window.member_ids.clear()
            window.closed = False
            window.closed_at_arrival_seq = None
            window.snapshots.clear()
            window.latest_snapshot_id = None
        self.max_seen_ts = None
        self.watermark_ms = None
        self.delivered_ids.clear()
        self._delivered_set.clear()
        self.processed_count = 0
        self.batch_index = 0
        self.stream_closed = False
        self.delivery_trace.clear()
        self.batch_trace.clear()
        self.replay_failures.clear()
        self.resolve_records.clear()
        self.snapshot_counter = 0
        self.closed_windows_since_commit = 0

    def _restore_state(self, state: dict[str, Any], runtime: dict[str, Any]) -> None:
        stored_windows = state.get("windows")
        if not isinstance(stored_windows, list):
            raise ValueError("windows missing")
        restored: dict[str, WindowState] = {}
        for item in stored_windows:
            window = WindowState.from_dict(item)
            if window.window_id not in self.windows or (
                window.start_ms,
                window.end_ms,
            ) != (self.windows[window.window_id].start_ms, self.windows[window.window_id].end_ms):
                raise ValueError("window definitions changed")
            restored[window.window_id] = window
        if set(restored) != set(self.windows):
            raise ValueError("window set mismatch")
        self.windows = restored
        self.max_seen_ts = state.get("max_seen_ts")
        self.watermark_ms = state.get("watermark_ms")
        self.delivered_ids = list(state.get("delivered_ids", []))
        self._delivered_set = set(self.delivered_ids)
        self.processed_count = int(state.get("processed_count", 0))
        if self.processed_count < 0 or self.processed_count > len(self.plan.arrivals):
            raise ValueError("processed arrival count out of range")
        self.batch_index = int(state.get("batch_index", 0))
        self.stream_closed = bool(state.get("stream_closed", False))
        self.delivery_trace = list(state.get("delivery_trace", []))
        self.batch_trace = list(state.get("batch_trace", []))
        self.replay_failures = list(runtime.get("replay_failures", []))
        self.resolve_records = list(state.get("resolve_records", []))
        self.snapshot_counter = int(state.get("snapshot_counter", 0))

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------
    def run(self, *, max_batches: int | None = None) -> dict[str, Any]:
        if max_batches is not None and max_batches < 0:
            raise ValueError("max_batches must be >= 0")
        remaining_batches = max_batches
        while self.processed_count < len(self.plan.arrivals):
            if remaining_batches is not None and remaining_batches <= 0:
                break
            batch = list(
                self.plan.arrivals[
                    self.processed_count : self.processed_count + self.batch_event_budget
                ]
            )
            if not batch:
                break
            batch_delivery: list[dict[str, Any]] = []
            for arrival in batch:
                trace = self._process_arrival(arrival)
                batch_delivery.append(trace)
                self.processed_count += 1
            self.batch_index += 1
            self.batch_trace.append(
                {
                    "batch_index": self.batch_index - 1,
                    "arrival_seqs": [arrival.arrival_seq for arrival in batch],
                    "event_ids": [arrival.event_id for arrival in batch],
                    "processed_through_count": self.processed_count,
                }
            )
            if self.closed_windows_since_commit >= self.config.checkpoint_interval_windows:
                self._materialize(track=False)
                self._commit()
            if remaining_batches is not None:
                remaining_batches -= 1

        all_processed = self.processed_count >= len(self.plan.arrivals)
        if all_processed and not self.stream_closed:
            self._close_end_of_stream()
        materialization = self._materialize(track=True)
        pending = list(self.plan.arrivals[self.processed_count :])
        failures = self._collect_failures(pending)
        report = self._build_report(
            failures=failures,
            pending=pending,
            materialization=materialization,
            all_processed=all_processed,
        )
        manifest = self._commit(report=report)
        report["checkpoint"] = self._checkpoint_report(manifest)
        return report

    def _process_arrival(self, arrival: Any) -> dict[str, Any]:
        event = self.event_result.events.get(arrival.event_id)
        max_seen_before = self.max_seen_ts
        trace: dict[str, Any] = {
            "arrival_seq": arrival.arrival_seq,
            "declared_index": arrival.declared_index,
            "event_id": arrival.event_id,
            "delay_ms": arrival.delay_ms,
            "event_timestamp_ms": None if event is None else event.timestamp_ms,
            "status": "accepted",
            "watermark_after_ms": self.watermark_ms,
        }
        if event is None:
            in_source = arrival.event_id in self.event_result.source_ids
            trace["status"] = "invalid"
            if not in_source:
                self.replay_failures.append(
                    ValidationFailure(
                        code="E_EVENT_INVALID",
                        event_id=arrival.event_id,
                        message="injection plan references an event absent from all sources",
                        arrival_seq=arrival.arrival_seq,
                    ).as_dict()
                )
            self.delivery_trace.append(trace)
            return trace
        if event.event_id in self._delivered_set:
            trace["status"] = "duplicate"
            self.replay_failures.append(
                ValidationFailure(
                    code="E_EVENT_INVALID",
                    event_id=event.event_id,
                    message="duplicate event delivery rejected",
                    arrival_seq=arrival.arrival_seq,
                    window_start_ms=min(
                        (self.windows[item].start_ms for item in self.event_windows[event.event_id]),
                        default=None,
                    ),
                    window_end_ms=max(
                        (self.windows[item].end_ms for item in self.event_windows[event.event_id]),
                        default=None,
                    ),
                ).as_dict()
            )
            self.delivery_trace.append(trace)
            return trace

        self._delivered_set.add(event.event_id)
        self.delivered_ids.append(event.event_id)
        lateness = 0 if max_seen_before is None else max_seen_before - event.timestamp_ms
        for window_id in self.event_windows[event.event_id]:
            window = self.windows[window_id]
            if window.closed:
                before = window.latest_snapshot_id
                if self.config.late_policy == "drop":
                    self.resolve_records.append(
                        {
                            "window_id": window_id,
                            "window_start_ms": window.start_ms,
                            "window_end_ms": window.end_ms,
                            "late_event_id": event.event_id,
                            "arrival_seq": arrival.arrival_seq,
                            "lateness_ms": lateness,
                            "policy": "drop",
                            "applied": False,
                            "before_snapshot_id": before,
                            "after_snapshot_id": before,
                        }
                    )
                else:
                    window.member_ids.append(event.event_id)
                    snapshot = self._new_snapshot(
                        window, "resettle", arrival.arrival_seq
                    )
                    self.resolve_records.append(
                        {
                            "window_id": window_id,
                            "window_start_ms": window.start_ms,
                            "window_end_ms": window.end_ms,
                            "late_event_id": event.event_id,
                            "arrival_seq": arrival.arrival_seq,
                            "lateness_ms": lateness,
                            "policy": "resettle",
                            "applied": True,
                            "before_snapshot_id": before,
                            "after_snapshot_id": snapshot.snapshot_id,
                        }
                    )
            else:
                window.member_ids.append(event.event_id)

        self.max_seen_ts = (
            event.timestamp_ms
            if self.max_seen_ts is None
            else max(self.max_seen_ts, event.timestamp_ms)
        )
        self.watermark_ms = self.max_seen_ts - self.config.allowed_lateness_ms
        for window_id in self.window_order:
            window = self.windows[window_id]
            if not window.closed and window.end_ms <= self.watermark_ms:
                self._close_window(window, "watermark", arrival.arrival_seq)
        trace["watermark_after_ms"] = self.watermark_ms
        self.delivery_trace.append(trace)
        return trace

    def _new_snapshot(
        self, window: WindowState, reason: str, arrival_seq: int | None
    ) -> Snapshot:
        self.snapshot_counter += 1
        event_ids = sorted(
            window.member_ids,
            key=lambda event_id: (
                self.event_result.events[event_id].timestamp_ms,
                event_id,
            ),
        )
        snapshot = Snapshot(
            snapshot_id=f"s{self.snapshot_counter:08d}",
            reason=reason,
            event_ids=event_ids,
            created_at_arrival_seq=arrival_seq,
        )
        window.snapshots.append(snapshot)
        window.latest_snapshot_id = snapshot.snapshot_id
        return snapshot

    def _close_window(self, window: WindowState, reason: str, arrival_seq: int | None) -> None:
        if window.closed:
            return
        window.closed = True
        window.closed_at_arrival_seq = arrival_seq
        self._new_snapshot(window, reason, arrival_seq)
        self.closed_windows_since_commit += 1

    def _close_end_of_stream(self) -> None:
        for window_id in self.window_order:
            window = self.windows[window_id]
            if not window.closed:
                self._close_window(window, "end_of_stream", None)
        self.stream_closed = True

    # ------------------------------------------------------------------
    # Stage materialization and cache
    # ------------------------------------------------------------------
    def _cache_get(self, key: str) -> Any | None:
        return self.cache_entries.get(key)

    def _cache_put(self, key: str, value: Any) -> None:
        self.cache_entries[key] = value

    def _snapshot_events(self, snapshot: Snapshot) -> list[Event]:
        return [self.event_result.events[event_id] for event_id in snapshot.event_ids]

    def _refs(self, events: Iterable[Event]) -> list[dict[str, Any]]:
        return [event.reference() for event in sorted(events, key=lambda item: (item.timestamp_ms, item.event_id))]

    @staticmethod
    def _stable_task_key(start: int, end: int, phase: str, snapshot_id: str) -> list[Any]:
        return [start, end, phase, snapshot_id]

    def _materialize(self, *, track: bool) -> dict[str, Any]:
        if track:
            self.scheduler_trace = []
        prepared: list[dict[str, Any]] = []
        for window_id in self.window_order:
            window = self.windows[window_id]
            for snapshot in window.snapshots:
                events = self._snapshot_events(snapshot)
                refs = self._refs(events)
                membership_key = stage_key(
                    "membership",
                    self.config.stage_config_hash("membership"),
                    window.start_ms,
                    window.end_ms,
                    refs,
                )
                cached_membership = self._cache_get(membership_key)
                membership_status = "reused" if cached_membership is not None else "computed"
                if cached_membership is None:
                    self._cache_put(membership_key, {"event_refs": refs})
                    self.cache_stats["membership_misses"] += 1
                else:
                    self.cache_stats["membership_hits"] += 1
                overflow = len(events) > self.config.max_events_per_window
                prepared.append(
                    {
                        "window": window,
                        "snapshot": snapshot,
                        "events": events,
                        "refs": refs,
                        "membership_key": membership_key,
                        "membership_status": membership_status,
                        "overflow": overflow,
                    }
                )

        visual_tasks: list[dict[str, Any]] = []
        for item in prepared:
            if item["overflow"]:
                item["visual_key"] = None
                item["visual_status"] = "skipped"
                item["fusion_key"] = None
                item["fusion_status"] = "skipped"
                continue
            frame_events = [event for event in item["events"] if event.kind == "image"]
            selected = select_frames(
                self.config,
                item["window"].start_ms,
                item["window"].end_ms,
                frame_events,
            )
            selected_ids = [event.event_id for event in selected]
            visual_key = visual_stage_key(
                self.config,
                item["window"].start_ms,
                item["window"].end_ms,
                frame_events,
                selected_ids,
            )
            item["frame_events"] = frame_events
            item["selected"] = selected
            item["visual_key"] = visual_key
            cached = self._cache_get(visual_key)
            if cached is not None:
                self.snapshot_visual[item["snapshot"].snapshot_id] = cached
                item["visual_status"] = "reused"
                self.cache_stats["visual_hits"] += 1
            else:
                item["visual_status"] = "computed"
                visual_tasks.append(item)
            if track:
                self.scheduler_trace.append(
                    {
                        "phase": "visual",
                        "stable_sort_key": self._stable_task_key(
                            item["window"].start_ms,
                            item["window"].end_ms,
                            "visual",
                            item["snapshot"].snapshot_id,
                        ),
                        "window_id": item["window"].window_id,
                        "snapshot_id": item["snapshot"].snapshot_id,
                        "cache": "hit" if cached is not None else "miss",
                    }
                )

        visual_tasks.sort(key=lambda item: (item["window"].start_ms, item["window"].end_ms, item["snapshot"].snapshot_id))
        if visual_tasks:
            with ThreadPoolExecutor(max_workers=self.config.thread_pool_size) as executor:
                futures = []
                for item in visual_tasks:
                    futures.append(
                        executor.submit(
                            compute_visual,
                            self.config,
                            self.sample_dir,
                            item["window"].start_ms,
                            item["window"].end_ms,
                            item["frame_events"],
                            self.matrix,
                        )
                    )
                for item, future in zip(visual_tasks, futures):
                    artifact = future.result()
                    self.snapshot_visual[item["snapshot"].snapshot_id] = artifact
                    self._cache_put(item["visual_key"], artifact)
                    self.cache_stats["visual_misses"] += 1
                    self.cache_stats["visual_computations"] += 1

        fusion_tasks: list[dict[str, Any]] = []
        for item in prepared:
            if item["overflow"]:
                continue
            visual = self.snapshot_visual[item["snapshot"].snapshot_id]
            fusion_key = fusion_stage_key(
                self.config,
                item["refs"],
                item["visual_key"],
            )
            item["fusion_key"] = fusion_key
            cached = self._cache_get(fusion_key)
            if cached is not None:
                self.snapshot_fusion[item["snapshot"].snapshot_id] = cached
                item["fusion_status"] = "reused"
                self.cache_stats["fusion_hits"] += 1
            else:
                item["fusion_status"] = "computed"
                text_vector, sensor_vector, evidence_meta = evidence_vectors(
                    self.config, item["events"]
                )
                item["text_vector"] = text_vector
                item["sensor_vector"] = sensor_vector
                item["evidence_meta"] = evidence_meta
                fusion_tasks.append(item)
            if track:
                self.scheduler_trace.append(
                    {
                        "phase": "fusion",
                        "stable_sort_key": self._stable_task_key(
                            item["window"].start_ms,
                            item["window"].end_ms,
                            "fusion",
                            item["snapshot"].snapshot_id,
                        ),
                        "window_id": item["window"].window_id,
                        "snapshot_id": item["snapshot"].snapshot_id,
                        "cache": "hit" if cached is not None else "miss",
                    }
                )

        fusion_tasks.sort(key=lambda item: (item["window"].start_ms, item["window"].end_ms, item["snapshot"].snapshot_id))
        if fusion_tasks:
            with ThreadPoolExecutor(max_workers=self.config.thread_pool_size) as executor:
                futures = []
                for item in fusion_tasks:
                    visual = self.snapshot_visual[item["snapshot"].snapshot_id]
                    futures.append(
                        executor.submit(
                            fuse_payload,
                            self.config,
                            item["text_vector"],
                            item["sensor_vector"],
                            visual["vector_q"],
                        )
                    )
                for item, future in zip(fusion_tasks, futures):
                    payload = future.result()
                    self.snapshot_fusion[item["snapshot"].snapshot_id] = payload
                    self._cache_put(item["fusion_key"], payload)
                    self.cache_stats["fusion_misses"] += 1
                    self.cache_stats["fusion_computations"] += 1

        self.snapshot_meta = {
            item["snapshot"].snapshot_id: {
                "membership_key": item["membership_key"],
                "membership_status": item["membership_status"],
                "visual_key": item["visual_key"],
                "visual_status": item["visual_status"],
                "fusion_key": item["fusion_key"],
                "fusion_status": item["fusion_status"],
                "overflow": item["overflow"],
            }
            for item in prepared
        }
        self._materialize_resettlements(prepared, track=track)
        if track:
            self.scheduler_trace.sort(
                key=lambda item: (
                    item["stable_sort_key"][0],
                    item["stable_sort_key"][1],
                    item["stable_sort_key"][2],
                    item["stable_sort_key"][3],
                )
            )
        return {"prepared": prepared, "scheduler_trace": self.scheduler_trace}

    def _summary_for_snapshot(self, snapshot_id: str | None) -> dict[str, Any] | None:
        if snapshot_id is None:
            return None
        meta = self.snapshot_meta.get(snapshot_id)
        if meta is None:
            return None
        if meta["overflow"]:
            return {
                "status": "overflow",
                "confidence_q": None,
                "fusion_key": None,
            }
        payload = self.snapshot_fusion[snapshot_id]
        return {
            "status": "ok",
            "confidence_q": payload["confidence_q"],
            "logits_q": payload["logits_q"],
            "contributions_q": payload["contributions_q"],
            "fusion_key": meta["fusion_key"],
        }

    def _materialize_resettlements(self, prepared: list[dict[str, Any]], *, track: bool) -> None:
        by_window: dict[str, list[dict[str, Any]]] = {}
        for record in self.resolve_records:
            by_window.setdefault(record["window_id"], []).append(record)
        records: list[dict[str, Any]] = []
        for window_id in self.window_order:
            window = self.windows[window_id]
            snapshot_indices = {
                snapshot.snapshot_id: index for index, snapshot in enumerate(window.snapshots)
            }
            snapshot_chain = []
            for snapshot in window.snapshots:
                meta = self.snapshot_meta.get(snapshot.snapshot_id, {})
                snapshot_chain.append(
                    {
                        "reason": snapshot.reason,
                        "membership_key": meta.get("membership_key"),
                        "visual_key": meta.get("visual_key"),
                        "fusion_key": meta.get("fusion_key"),
                    }
                )
            window_resolve = sorted(
                by_window.get(window_id, []),
                key=lambda item: (item["arrival_seq"], item["late_event_id"]),
            )
            resolve_chain = []
            for item in window_resolve:
                resolve_chain.append(
                    {
                        "late_event_id": item["late_event_id"],
                        "arrival_seq": item["arrival_seq"],
                        "lateness_ms": item["lateness_ms"],
                        "policy": item["policy"],
                        "applied": item["applied"],
                        "before_index": snapshot_indices.get(item["before_snapshot_id"]),
                        "after_index": snapshot_indices.get(item["after_snapshot_id"]),
                    }
                )
            resettle_key = stage_key(
                "resettle",
                self.config.stage_config_hash("resettle"),
                window.window_id,
                snapshot_chain,
                resolve_chain,
            )
            cached = self._cache_get(resettle_key)
            if cached is not None:
                status = "reused"
                self.cache_stats["resettle_hits"] += 1
                semantic_records = cached
            else:
                status = "computed"
                self.cache_stats["resettle_misses"] += 1
                semantic_records = []
                for item, semantic in zip(window_resolve, resolve_chain):
                    before_index = semantic["before_index"]
                    after_index = semantic["after_index"]
                    before_id = (
                        window.snapshots[before_index].snapshot_id
                        if before_index is not None
                        else None
                    )
                    after_id = (
                        window.snapshots[after_index].snapshot_id
                        if after_index is not None
                        else None
                    )
                    semantic_records.append(
                        {
                            "late_event_id": item["late_event_id"],
                            "arrival_seq": item["arrival_seq"],
                            "lateness_ms": item["lateness_ms"],
                            "policy": item["policy"],
                            "applied": item["applied"],
                            "before_index": before_index,
                            "after_index": after_index,
                            "old_conclusion": self._summary_for_snapshot(before_id),
                            "new_conclusion": self._summary_for_snapshot(after_id),
                        }
                    )
                self._cache_put(resettle_key, semantic_records)
            window_records = []
            for semantic in semantic_records:
                before_index = semantic["before_index"]
                after_index = semantic["after_index"]
                window_records.append(
                    {
                        "window_id": window_id,
                        "window_start_ms": window.start_ms,
                        "window_end_ms": window.end_ms,
                        "late_event_id": semantic["late_event_id"],
                        "arrival_seq": semantic["arrival_seq"],
                        "lateness_ms": semantic["lateness_ms"],
                        "policy": semantic["policy"],
                        "applied": semantic["applied"],
                        "old_snapshot_id": (
                            window.snapshots[before_index].snapshot_id
                            if before_index is not None
                            else None
                        ),
                        "new_snapshot_id": (
                            window.snapshots[after_index].snapshot_id
                            if after_index is not None
                            else None
                        ),
                        "old_conclusion": semantic["old_conclusion"],
                        "new_conclusion": semantic["new_conclusion"],
                    }
                )
            self.window_resettle_status[window_id] = status
            self.window_stage_keys[window_id] = {
                "resettle": resettle_key,
                "snapshot_chain": snapshot_chain,
            }
            records.extend(window_records)
        records.sort(
            key=lambda item: (
                item["window_start_ms"],
                item["window_end_ms"],
                item["arrival_seq"],
                item["late_event_id"],
            )
        )
        self.resettlement_records = records

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _window_result(self, window: WindowState) -> dict[str, Any]:
        snapshot = window.latest_snapshot
        if snapshot is None:
            return {
                "window_id": window.window_id,
                "window_start_ms": window.start_ms,
                "window_end_ms": window.end_ms,
                "status": "open",
                "closed": False,
                "event_count": len(window.member_ids),
                "event_refs": self._refs(
                    self.event_result.events[event_id] for event_id in window.member_ids
                ),
                "empty_visual": None,
                "visual": None,
                "confidence_q": None,
                "logits_q": None,
                "contributions_q": None,
                "stage_keys": {},
            }
        events = self._snapshot_events(snapshot)
        refs = self._refs(events)
        meta = self.snapshot_meta[snapshot.snapshot_id]
        stage_keys = {
            "membership": meta["membership_key"],
            "visual": meta["visual_key"],
            "fusion": meta["fusion_key"],
            "resettle": self.window_stage_keys.get(window.window_id, {}).get("resettle"),
        }
        if meta["overflow"]:
            return {
                "window_id": window.window_id,
                "window_start_ms": window.start_ms,
                "window_end_ms": window.end_ms,
                "status": "overflow",
                "closed": True,
                "close_reason": snapshot.reason,
                "closed_at_arrival_seq": snapshot.created_at_arrival_seq,
                "event_count": len(events),
                "event_refs": refs,
                "empty_visual": None,
                "visual": None,
                "confidence_q": None,
                "logits_q": None,
                "contributions_q": None,
                "stage_keys": stage_keys,
            }
        visual = self.snapshot_visual[snapshot.snapshot_id]
        fusion = self.snapshot_fusion[snapshot.snapshot_id]
        return {
            "window_id": window.window_id,
            "window_start_ms": window.start_ms,
            "window_end_ms": window.end_ms,
            "status": "ok",
            "closed": True,
            "close_reason": snapshot.reason,
            "closed_at_arrival_seq": snapshot.created_at_arrival_seq,
            "event_count": len(events),
            "event_refs": refs,
            "empty_visual": visual["empty_visual"],
            "visual": {
                "empty_visual": visual["empty_visual"],
                "selected_frame_ids": visual["selected_frame_ids"],
                "selected_frame_indices": visual["selected_frame_indices"],
                "vector_q": visual["vector_q"],
            },
            "confidence_q": fusion["confidence_q"],
            "logits_q": fusion["logits_q"],
            "temperature_q": fusion["temperature_q"],
            "contributions_q": fusion["contributions_q"],
            "text_vector_q": fusion["text_vector_q"],
            "sensor_vector_q": fusion["sensor_vector_q"],
            "stage_keys": stage_keys,
        }

    def _collect_failures(self, pending: list[Any]) -> list[dict[str, Any]]:
        failures = [failure.as_dict() for failure in self.event_result.failures]
        failures.extend(self.replay_failures)
        for window_id in self.window_order:
            window = self.windows[window_id]
            snapshot = window.latest_snapshot
            if snapshot is None:
                continue
            if len(snapshot.event_ids) > self.config.max_events_per_window:
                failures.append(
                    ValidationFailure(
                        code="E_WINDOW_OVERFLOW",
                        event_id=None,
                        message=(
                            f"window {window.window_id} has {len(snapshot.event_ids)} events, "
                            f"limit is {self.config.max_events_per_window}"
                        ),
                        window_start_ms=window.start_ms,
                        window_end_ms=window.end_ms,
                        detail=window.window_id,
                    ).as_dict()
                )
        for arrival in pending:
            event = self.event_result.events.get(arrival.event_id)
            windows = self.event_windows.get(arrival.event_id, [])
            failures.append(
                ValidationFailure(
                    code="E_EVENT_BUDGET_PENDING",
                    event_id=arrival.event_id,
                    message="event retained in pending queue because the batch budget was exhausted",
                    arrival_seq=arrival.arrival_seq,
                    window_start_ms=min(
                        (self.windows[item].start_ms for item in windows), default=None
                    ),
                    window_end_ms=max(
                        (self.windows[item].end_ms for item in windows), default=None
                    ),
                    detail="expected on next batch; pending entries are processed first",
                ).as_dict()
            )
        failures.sort(key=_failure_sort_key)
        return failures

    def _build_report(
        self,
        *,
        failures: list[dict[str, Any]],
        pending: list[Any],
        materialization: dict[str, Any],
        all_processed: bool,
    ) -> dict[str, Any]:
        windows = [self._window_result(self.windows[window_id]) for window_id in self.window_order]
        empty_windows = [
            window["window_id"]
            for window in windows
            if window.get("status") == "ok" and window.get("empty_visual") is True
        ]
        stage_manifest = []
        for window in windows:
            keys = window.get("stage_keys", {})
            snapshot_id = self.windows[window["window_id"]].latest_snapshot_id
            meta = self.snapshot_meta.get(snapshot_id or "", {})
            stage_manifest.append(
                {
                    "window_id": window["window_id"],
                    "event_refs": [
                        {
                            "event_id": ref["event_id"],
                            "fingerprint": ref["fingerprint"],
                        }
                        for ref in window.get("event_refs", [])
                    ],
                    "stage_keys": keys,
                    "cache_status": {
                        "membership": meta.get("membership_status"),
                        "visual": meta.get("visual_status"),
                        "fusion": meta.get("fusion_status"),
                        "resettle": self.window_resettle_status.get(window["window_id"]),
                    },
                }
            )
        invalidation_windows = [
            {
                "window_id": entry["window_id"],
                "event_refs": entry["event_refs"],
                "stage_keys": entry["stage_keys"],
            }
            for entry in stage_manifest
        ]
        invalidation_manifest = {
            "windows": invalidation_windows,
            "fingerprint": content_hash(invalidation_windows),
        }
        meta = {
            "report_schema": REPORT_SCHEMA,
            "engine_state_version": ENGINE_STATE_VERSION,
            "config_hash": self.config.config_hash,
            "semantic_config_hash": self.config.semantic_hash,
            "plan_hash": self.plan.plan_hash,
            "source_hash": self.event_result.source_hash,
            "event_set_hash": content_hash(
                [event.canonical_payload for event in sorted(
                    self.event_result.events.values(),
                    key=lambda item: (item.timestamp_ms, item.event_id),
                )]
            ),
            "quantization_level": self.config.quantization_level,
            "fractional_bits": self.config.fractional_bits,
            "quant_scale": self.config.quant_scale,
            "late_boundary_rule": LATE_BOUNDARY_RULE,
            "late_policy": self.config.late_policy,
            "window_length_ms": self.config.window_length_ms,
            "window_step_ms": self.config.window_step_ms,
            "allowed_lateness_ms": self.config.allowed_lateness_ms,
            "event_count": len(self.event_result.events),
            "arrival_count": len(self.plan.arrivals),
            "processed_arrival_count": self.processed_count,
            "thread_pool_size": self.config.thread_pool_size,
            "max_events_per_window": self.config.max_events_per_window,
        }
        report = {
            "schema": REPORT_SCHEMA,
            "meta": meta,
            "watermark": {
                "rule": self.config.watermark_rule,
                "final_watermark_ms": self.watermark_ms,
                "max_seen_timestamp_ms": self.max_seen_ts,
                "allowed_lateness_ms": self.config.allowed_lateness_ms,
                "boundary_rule": LATE_BOUNDARY_RULE,
            },
            "delivery_order": self.delivery_trace,
            "budget": {
                "batch_event_budget": self.batch_event_budget,
                "batch_count": len(self.batch_trace),
                "batches": self.batch_trace,
                "scheduler_task_order": materialization["scheduler_trace"],
            },
            "windows": windows,
            "resettlements": self.resettlement_records,
            "empty_windows": empty_windows,
            "failures": failures,
            "pending": [
                {
                    "arrival_seq": arrival.arrival_seq,
                    "event_id": arrival.event_id,
                    "delay_ms": arrival.delay_ms,
                }
                for arrival in pending
            ],
            "diagnostics": self.diagnostics,
            "cache": {
                "stats": dict(self.cache_stats),
                "window_manifest": stage_manifest,
            },
            "invalidation_manifest": invalidation_manifest,
            "checkpoint": self._checkpoint_report(None),
        }
        self.last_materialization = materialization
        self._last_report = report
        return report

    # ------------------------------------------------------------------
    # Checkpoint commit
    # ------------------------------------------------------------------
    def _runtime_state(self) -> dict[str, Any]:
        return {
            "engine_state_version": ENGINE_STATE_VERSION,
            "max_seen_ts": self.max_seen_ts,
            "watermark_ms": self.watermark_ms,
            "delivered_ids": list(self.delivered_ids),
            "processed_count": self.processed_count,
            "batch_index": self.batch_index,
            "stream_closed": self.stream_closed,
            "delivery_trace": self.delivery_trace,
            "batch_trace": self.batch_trace,
            "resolve_records": self.resolve_records,
            "snapshot_counter": self.snapshot_counter,
            "windows": [self.windows[window_id].as_dict() for window_id in self.window_order],
        }

    def _checkpoint_report(self, manifest: dict[str, Any] | None) -> dict[str, Any]:
        last_seq = None
        if self.processed_count > 0:
            last_seq = self.plan.arrivals[self.processed_count - 1].arrival_seq
        return {
            "used": self.previous_manifest is not None,
            "resumed": self.resume_used,
            "global_version": self.checkpoint_version,
            "config_hash": self.config.config_hash,
            "plan_hash": self.plan.plan_hash,
            "source_hash": self.event_result.source_hash,
            "processed_arrival_count": self.processed_count,
            "processed_arrival_seq": last_seq,
            "pending_arrival_seqs": [
                arrival.arrival_seq for arrival in self.plan.arrivals[self.processed_count :]
            ],
            "watermark_ms": self.watermark_ms,
            "stage_versions": (
                {
                    stage: entry.get("version")
                    for stage, entry in manifest.get("stages", {}).items()
                }
                if manifest is not None
                else {}
            ),
        }

    def _commit(self, *, report: dict[str, Any] | None = None) -> dict[str, Any]:
        if report is None:
            report = getattr(self, "_last_report", None)
        if report is None:
            report = self._build_report(
                failures=self._collect_failures(list(self.plan.arrivals[self.processed_count :])),
                pending=list(self.plan.arrivals[self.processed_count :]),
                materialization={"scheduler_trace": []},
                all_processed=self.processed_count >= len(self.plan.arrivals),
            )
        visual_payload = {
            snapshot_id: {
                "visual": self.snapshot_visual[snapshot_id],
                "meta": self.snapshot_meta.get(snapshot_id),
            }
            for snapshot_id in sorted(self.snapshot_visual)
        }
        fusion_payload = {
            snapshot_id: payload for snapshot_id, payload in sorted(self.snapshot_fusion.items())
        }
        payloads = {
            "replay": {
                "delivered_event_ids": list(self.delivered_ids),
                "delivery_order": self.delivery_trace,
                "max_seen_timestamp_ms": self.max_seen_ts,
                "watermark_ms": self.watermark_ms,
                "processed_arrival_count": self.processed_count,
            },
            "membership": {
                "event_window_map": self.event_windows,
                "windows": [self.windows[window_id].as_dict() for window_id in self.window_order],
            },
            "visual": {"snapshots": visual_payload},
            "fusion": {"snapshots": fusion_payload},
            "resettle": {"records": self.resettlement_records},
            "pending": {
                "arrivals": [
                    arrival.as_dict()
                    for arrival in self.plan.arrivals[self.processed_count :]
                ]
            },
            "runtime": {
                "state": self._runtime_state(),
                "replay_failures": self.replay_failures,
                "resettlement_records": self.resettlement_records,
            },
            "cache": {"entries": dict(sorted(self.cache_entries.items()))},
        }
        self.checkpoint_version += 1
        last_seq = None
        if self.processed_count > 0:
            last_seq = self.plan.arrivals[self.processed_count - 1].arrival_seq
        manifest = self.checkpoint.commit(
            global_version=self.checkpoint_version,
            config_hash=self.config.config_hash,
            plan_hash=self.plan.plan_hash,
            source_hash=self.event_result.source_hash,
            processed_arrival_seq=last_seq,
            processed_arrival_count=self.processed_count,
            pending_arrival_seqs=[
                arrival.arrival_seq for arrival in self.plan.arrivals[self.processed_count :]
            ],
            watermark_ms=self.watermark_ms,
            payloads=payloads,
        )
        self.closed_windows_since_commit = 0
        return manifest


def run_engine(
    sample_dir: str | Path,
    config_path: str | Path,
    plan_path: str | Path,
    checkpoint_dir: str | Path,
    *,
    batch_event_budget: int | None = None,
    max_batches: int | None = None,
    reset: bool = False,
    frames_jsonl: str | Path | None = None,
    text_jsonl: str | Path | None = None,
    sensor_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    events = load_events(
        sample_dir,
        config,
        frames_jsonl=frames_jsonl,
        text_jsonl=text_jsonl,
        sensor_jsonl=sensor_jsonl,
    )
    plan = load_plan(plan_path)
    engine = AggregationEngine(
        sample_dir,
        config,
        events,
        plan,
        checkpoint_dir,
        batch_event_budget=batch_event_budget,
        reset=reset,
    )
    return engine.run(max_batches=max_batches)
