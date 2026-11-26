"""Atomic staged snapshots and post-termination recovery.

Stage outputs, todo queue, frame/tile cursors and consumed budget are first
written into a temporary directory. Commit only replaces the live snapshot
after the global version and configuration hash still match, so a concurrent
edit cannot be silently overwritten. Recovery after a forced termination uses
only the last committed snapshot.
"""

from __future__ import annotations

import json
import os
import signal
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

from .codes import BizCode
from .fingerprint import fingerprint
from .pipeline import PipelineState


@dataclass
class Snapshot:
    global_version: int
    config_hash: str
    state_fp: str
    scheduler_state: dict
    completed_frames: list[int]
    completed_tiles: dict
    consumed_budget: int
    pending_items: list
    failures: list
    cache_root: str

    def as_dict(self) -> dict:
        return asdict(self)


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.live_path = self.root / "snapshot.json"
        self.diagnostic_path = self.root / "diagnostics.json"

    def write_staged(self, snapshot: Snapshot,
                     expected_version: int,
                     expected_config_hash: str) -> tuple[bool, str]:
        if (snapshot.global_version != expected_version
                or snapshot.config_hash != expected_config_hash):
            return False, BizCode.E_SNAPSHOT_VERSION.value
        tmp_dir = Path(tempfile.mkdtemp(prefix="fx-snap-", dir=self.root))
        tmp_path = tmp_dir / "snapshot.json"
        tmp_path.write_text(
            json.dumps(snapshot.as_dict(), sort_keys=True, indent=1),
            encoding="utf-8")
        os.replace(tmp_path, self.live_path)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass
        return True, BizCode.OK.value

    def load(self) -> Snapshot | None:
        if not self.live_path.exists():
            return None
        data = json.loads(self.live_path.read_text(encoding="utf-8"))
        return Snapshot(**data)

    def recover(self, state: PipelineState) -> tuple[Snapshot | None, list[str]]:
        notes: list[str] = []
        snapshot = self.load()
        if snapshot is None:
            notes.append("no committed snapshot; cold start required")
            return None, notes
        current_hash = fingerprint(
            state.cfg.config_fingerprint_material(), "config")
        if snapshot.config_hash != current_hash:
            notes.append(
                f"{BizCode.E_SNAPSHOT_VERSION.value}: config changed since "
                "last committed snapshot; partial artifacts retained but "
                "cursor state is not trusted")
            return snapshot, notes
        if snapshot.state_fp != state.state_fp():
            notes.append("state fingerprint differs; resuming from committed "
                         "cache and recomputing changed entries")
        else:
            notes.append("state fingerprint matches; resuming cursors")
        return snapshot, notes

    def write_diagnostics(self, payload: dict) -> None:
        tmp = self.diagnostic_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2),
                       encoding="utf-8")
        os.replace(tmp, self.diagnostic_path)


class TerminationHandler:
    """Best-effort flush on handleable signals; never promises an exit report."""

    def __init__(self, store: SnapshotStore, flusher) -> None:
        self.store = store
        self.flusher = flusher
        self.signals: list[int] = []
        self._previous: dict[int, object] = {}

    def install(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle)
            except (ValueError, OSError):  # pragma: no cover - non-main thread
                continue

    def restore(self) -> None:
        for sig, previous in self._previous.items():
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):  # pragma: no cover
                continue

    def _handle(self, signum, frame) -> None:  # pragma: no cover - signal path
        self.signals.append(int(signum))
        try:
            snapshot = self.flusher()
            self.store.write_staged(
                snapshot, snapshot.global_version, snapshot.config_hash)
            self.store.write_diagnostics({
                "signal": int(signum),
                "flushed": True,
                "global_version": snapshot.global_version,
                "completed_frames": snapshot.completed_frames,
                "consumed_budget": snapshot.consumed_budget,
                "pending_items": snapshot.pending_items,
            })
        except Exception as exc:  # diagnostics must not mask the signal
            try:
                self.store.write_diagnostics(
                    {"signal": int(signum), "flushed": False, "error": str(exc)})
            except Exception:
                pass
        finally:
            previous = self._previous.get(signum)
            if callable(previous):
                previous(signum, frame)
            elif previous == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
