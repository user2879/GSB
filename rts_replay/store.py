"""On-disk persistence for snapshots, manifests and reports.

Snapshots are written to a temporary file first and then atomically renamed, so
an externally killed process never leaves a half-written snapshot behind: the
previous snapshot stays valid and recovery restarts from the newest complete
one. Snapshot files embed the format version and a content state hash; loading
an older/incompatible version or tampered content raises ``E_SNAPSHOT_STALE``.
"""
from __future__ import annotations

import glob
import json
import os

from .errors import ErrorCode, RtsError
from .hashing import canonical_dumps
from .state import Snapshot, snapshot_from_dict


def snapshot_path(snapshot_dir: str, tick: int) -> str:
    return os.path.join(snapshot_dir, f"snapshot_{tick:08d}.json")


def save_snapshot(snapshot_dir: str, snapshot: Snapshot) -> str:
    os.makedirs(snapshot_dir, exist_ok=True)
    final_path = snapshot_path(snapshot_dir, snapshot.tick)
    tmp_path = final_path + ".tmp"
    with open(tmp_path, "wb") as handle:
        handle.write(canonical_dumps(snapshot.to_dict()))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, final_path)
    return final_path


def load_snapshot(path: str) -> Snapshot:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.loads(handle.read())
    except (OSError, ValueError) as exc:
        raise RtsError(
            ErrorCode.E_SNAPSHOT_STALE,
            f"cannot read snapshot {path}: {exc}",
            context={"path": path},
        ) from exc
    return snapshot_from_dict(raw)


def list_snapshot_files(snapshot_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(snapshot_dir, "snapshot_*.json")))


def newest_snapshot(snapshot_dir: str) -> Snapshot | None:
    files = list_snapshot_files(snapshot_dir)
    if not files:
        return None
    return load_snapshot(files[-1])


def nearest_snapshot_at_or_before(snapshot_dir: str, tick: int) -> Snapshot | None:
    """Find the newest persisted snapshot whose tick is <= *tick*."""
    candidate: Snapshot | None = None
    for path in list_snapshot_files(snapshot_dir):
        snapshot = load_snapshot(path)
        if snapshot.tick <= tick and (candidate is None or snapshot.tick > candidate.tick):
            candidate = snapshot
    return candidate


def write_json(path: str, payload: dict | list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as handle:
        handle.write(canonical_dumps(payload))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(tmp_path, path)
