"""Read-only presentation model shared by the GUI and automated tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .engine import Engine
from .model import SessionSpec, digest_value, load_session
from .replay import ReplayResult, replay_from_snapshot
from .snapshot import RestoreResult, SnapshotInfo, SnapshotStore


class WorkbenchViewModel:
    def __init__(
        self,
        spec: SessionSpec,
        *,
        snapshot_db: str | Path = "snapshots.sqlite3",
        auto_run: bool = True,
    ) -> None:
        self.spec = spec
        self.engine = Engine(spec)
        if auto_run:
            self.engine.run()
        self.store = SnapshotStore(snapshot_db)
        self.last_replay: ReplayResult | None = None
        self.last_restore: RestoreResult | None = None
        self.status_message = "ready"
        self.status_code = "OK"

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        snapshot_db: str | Path = "snapshots.sqlite3",
        auto_run: bool = True,
    ) -> "WorkbenchViewModel":
        return cls(load_session(path), snapshot_db=snapshot_db, auto_run=auto_run)

    def close(self) -> None:
        self.store.close()

    def _set_ok(self, message: str) -> None:
        self.status_message = message
        self.status_code = "OK"

    def run(self) -> Engine:
        self.engine.run()
        self._set_ok("session advanced to quiescence")
        return self.engine

    def step(self) -> bool:
        advanced = self.engine.step()
        self._set_ok("event stepped" if advanced else "no events remain")
        return advanced

    def save_snapshot(self, label: str) -> SnapshotInfo:
        info = self.store.save(self.engine, label)
        self._set_ok(f"snapshot {info.snapshot_id} saved")
        return info

    def restore_snapshot(self, snapshot_id: int) -> RestoreResult:
        result = self.store.restore(snapshot_id, self.spec, self.engine)
        self.spec = result.spec
        self.engine = result.engine
        self.last_restore = result
        if result.whole_replacement:
            self._set_ok(
                f"snapshot {snapshot_id} replaced the whole session "
                f"({len(result.diff['added'])} added, {len(result.diff['removed'])} removed, "
                f"{len(result.diff['changed'])} changed)"
            )
        else:
            self._set_ok(f"snapshot {snapshot_id} restored")
        return result

    def replay_snapshot(self, snapshot_id: int) -> ReplayResult:
        reference = Engine(self.spec).run()
        result = replay_from_snapshot(
            self.store,
            snapshot_id,
            reference_spec=self.spec,
            reference_engine=reference,
        )
        self.last_replay = result
        self._set_ok(f"snapshot {snapshot_id} replay matched {result.frames_compared} frames")
        return result

    def snapshot_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "id": info.snapshot_id,
                "label": info.label,
                "version": info.declared_version,
                "next_event": info.next_event_seq,
                "digest": info.state_digest[:12],
                "read_only": "yes" if info.read_only else "no",
            }
            for info in self.store.list()
        ]

    def frame_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "frame": frame.frame_seq,
                "event": "" if frame.event_seq is None else frame.event_seq,
                "type": frame.frame_type,
                "stream": frame.stream_id,
                "bytes": frame.data_bytes,
                "conn_before": frame.connection_window_before,
                "conn_after": frame.connection_window_after,
                "stream_before": frame.stream_window_before,
                "stream_after": frame.stream_window_after,
                "strategy": frame.strategy,
                "business": frame.business_code,
            }
            for frame in sorted(self.engine.frames, key=lambda item: item.frame_seq)
        ]

    def stream_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "stream": stream.stream_id,
                "priority": stream.priority,
                "state": stream.state,
                "body_remaining": stream.body_remaining,
                "stream_window": stream.window,
                "conn_window": self.engine.connection_window,
                "actual_bytes": stream.granted_bytes,
                "expected_bytes": stream.expected_bytes,
                "gap_rounds": stream.max_grant_gap_rounds,
            }
            for stream in (self.engine.streams[stream_id] for stream_id in sorted(self.engine.streams))
        ]

    def transition_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "stream": item.stream_id,
                "event": "" if item.event_seq is None else item.event_seq,
                "frame": "" if item.frame_seq is None else item.frame_seq,
                "from": item.from_state,
                "to": item.to_state,
                "reason": item.reason,
            }
            for index, item in enumerate(self.engine.transitions, start=1)
        ]

    def fairness_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "stream": item["stream_id"],
                "priority": item["priority"],
                "expected": item["expected_bytes"],
                "actual": item["actual_bytes"],
                "deviation": item["deviation_bytes"],
                "ratio": f"{item['deviation_ratio']:.6f}",
                "state": item["state"],
            }
            for item in self.engine.fairness_report()
        ]

    def warning_rows(self) -> list[dict[str, Any]]:
        warnings = sorted(
            self.engine.starvation_warnings(),
            key=lambda item: (
                item.get("start_round", 0),
                item["stream_id"],
                item["type"],
                item.get("source_event_seq") if item.get("source_event_seq") is not None else -1,
            ),
        )
        return [
            {
                "type": item["type"],
                "stream": item["stream_id"],
                "priority": item["priority"],
                "gap_rounds": item["gap_rounds"],
                "start_event": item.get("start_event_seq"),
                "source_event": item.get("source_event_seq"),
                "beneficiary": item.get("beneficiary_stream_id"),
                "deviation_ratio": item["deviation_ratio"],
            }
            for item in warnings
        ]

    def strategy_text(self) -> str:
        selection = self.engine.selection
        return (
            f"{selection.strategy} / {selection.mode} / {selection.reason_code} | "
            f"tie={str(selection.tie_affected).lower()} "
            f"diff={selection.frame_difference} threshold={selection.tie_threshold_frames} | "
            f"streams={selection.params.get('stream_count', 0)} "
            f"pending={selection.params.get('pending_frames', 0)} "
            f"avg_window={selection.params.get('average_stream_window', 0.0)}"
        )

    def status_rows(self) -> dict[str, Any]:
        return {
            "session": self.spec.session_id,
            "declared_version": self.spec.declared_version,
            "content_digest": self.spec.content_digest,
            "connection_window": self.engine.connection_window,
            "current_strategy": self.engine.current_strategy,
            "selection": self.engine.selection.to_dict(),
            "frames": len(self.engine.frames),
            "transitions": len(self.engine.transitions),
            "warnings": len(self.engine.warnings),
            "status_code": self.status_code,
            "status_message": self.status_message,
            "next_event_seq": self.engine.next_event_seq,
            "engine_digest": self.engine.run_digest(),
        }

    def snapshot_payload(self) -> dict[str, Any]:
        return {
            "frames": self.frame_rows(),
            "streams": self.stream_rows(),
            "transitions": self.transition_rows(),
            "fairness": self.fairness_rows(),
            "warnings": self.warning_rows(),
            "snapshots": self.snapshot_rows(),
            "strategy": self.strategy_text(),
            "status": self.status_rows(),
        }

    def render_signature(self) -> str:
        return digest_value(self.snapshot_payload())
