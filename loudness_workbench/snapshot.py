"""Immutable measurement snapshots with monotonic version control.

The UI never reads live measurement state directly: the engine freezes a
:class:`Snapshot`, hands it to the :class:`SnapshotStore`, and widgets only
consume accepted snapshots. Any attempt to install an older version is
rejected with ``E_SNAPSHOT_STALE``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .errors import E_SNAPSHOT_STALE, WorkbenchError


@dataclass(frozen=True)
class Snapshot:
    version: int
    mode: str  # "realtime" | "offline"
    start_sample: int
    end_sample: int
    sample_rate: int
    block_ms: int
    payload: dict[str, Any]

    @property
    def coverage_s(self) -> tuple[float, float]:
        return self.start_sample / self.sample_rate, self.end_sample / self.sample_rate

    def derive(self, **changes) -> "Snapshot":
        return replace(self, **changes)


class SnapshotStore:
    """Append-only store that refuses stale version roll-back."""

    def __init__(self) -> None:
        self._snapshots: list[Snapshot] = []

    @property
    def current(self) -> Snapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    @property
    def latest_version(self) -> int:
        return self._snapshots[-1].version if self._snapshots else -1

    def commit(self, snapshot: Snapshot) -> Snapshot:
        current_version = self.latest_version
        if snapshot.version <= current_version:
            raise WorkbenchError(
                E_SNAPSHOT_STALE,
                f"rejecting snapshot v{snapshot.version}, current is v{current_version}",
            )
        self._snapshots.append(snapshot)
        return snapshot

    def history(self) -> tuple[Snapshot, ...]:
        return tuple(self._snapshots)

    def next_version(self) -> int:
        return self.latest_version + 1
