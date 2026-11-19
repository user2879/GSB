"""Durable step snapshots, versioning and tamper detection.

Snapshot directory layout::

    <snapshot_dir>/
        index.json                 # pointers to the current and archived runs
        current/
            snapshot_NNNN.json     # one per successfully applied constraint
            latest.json            # copy of the newest snapshot
            interrupted.json       # present only while a run is suspended
        archive/<version_id>/...   # snapshots belonging to older versions

Every snapshot carries a SHA-256 digest over its canonical content.  A missing
or mismatched digest invalidates the snapshot (``E_DIGEST``).  Writes are
atomic (temp file + ``os.replace``), so a hard kill can only leave either the
previous or the next complete snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Any

from .errors import E_DIGEST, E_IO, E_VERSION, SolverError

ENGINE_VERSION = "level-solver-1.0"
INTERRUPT_EXIT_CODE = 130


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_version_id(spec: Any) -> str:
    """Deterministic version identifier for a normalised specification."""
    payload = {
        "engine": ENGINE_VERSION,
        "width": spec.width,
        "height": spec.height,
        "blocked": sorted(spec.blocked),
        "seed": spec.seed,
        "concurrency": spec.concurrency,
        "memory_budget": spec.memory_budget,
        "max_rollback": spec.max_rollback,
        "alternative_limit": spec.alternative_limit,
        "constraints": [
            {"id": c.cid, "type": c.type, "order": c.order, "params": c.params}
            for c in spec.constraints
        ],
    }
    return sha256_text(canonical_json(payload))[:16]


@dataclass
class FrameRecord:
    cid: str
    ctype: str
    attempts: int
    label: str


@dataclass
class Snapshot:
    version_id: str
    sequence: int
    next_index: int
    grid: list[list[int]]
    rooms: list[dict[str, int]]
    frames: list[FrameRecord]
    retries: list[int]
    constraint_counts: dict[str, int]
    conflicts: list[dict[str, Any]]
    rng_state: int
    seed: int

    def to_content(self) -> dict[str, Any]:
        return {
            "engine_version": ENGINE_VERSION,
            "version_id": self.version_id,
            "sequence": self.sequence,
            "next_index": self.next_index,
            "grid": self.grid,
            "rooms": self.rooms,
            "frames": [asdict(frame) for frame in self.frames],
            "retries": list(self.retries),
            "constraint_counts": self.constraint_counts,
            "conflicts": self.conflicts,
            "rng_state": self.rng_state,
            "seed": self.seed,
        }

    @classmethod
    def from_content(cls, content: dict[str, Any]) -> "Snapshot":
        return cls(
            version_id=content["version_id"],
            sequence=content["sequence"],
            next_index=content["next_index"],
            grid=content["grid"],
            rooms=content["rooms"],
            frames=[FrameRecord(**frame) for frame in content["frames"]],
            retries=list(content.get("retries", [])),
            constraint_counts=dict(content["constraint_counts"]),
            conflicts=list(content["conflicts"]),
            rng_state=content["rng_state"],
            seed=content["seed"],
        )


def _atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = os.path.join(directory, f".{os.path.basename(path)}.tmp-{os.getpid()}")
    try:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise SolverError(E_IO, f"failed writing {path}: {exc}") from exc


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SolverError(E_DIGEST, f"snapshot '{os.path.basename(path)}' is unreadable: {exc}")


def snapshot_digest(content: dict[str, Any]) -> str:
    return sha256_text(canonical_json(content))


def parse_snapshot_file(path: str) -> Snapshot:
    document = _read_json(path)
    if not isinstance(document, dict) or "digest" not in document or "content" not in document:
        raise SolverError(E_DIGEST, f"snapshot '{os.path.basename(path)}' is missing its digest")
    content = document["content"]
    recorded = document["digest"]
    actual = snapshot_digest(content)
    if not isinstance(recorded, str) or recorded != actual:
        raise SolverError(
            E_DIGEST,
            f"snapshot '{os.path.basename(path)}' digest mismatch",
            {"recorded": recorded, "actual": actual},
        )
    try:
        return Snapshot.from_content(content)
    except (KeyError, TypeError) as exc:
        raise SolverError(E_DIGEST, f"snapshot '{os.path.basename(path)}' is malformed: {exc}")


class SnapshotStore:
    """File-system backed snapshot store with version-aware replacement."""

    def __init__(self, base_dir: str, version_id: str) -> None:
        self.base_dir = os.path.abspath(base_dir)
        self.version_id = version_id
        self.current_dir = os.path.join(base_dir, "current")
        self.archive_dir = os.path.join(base_dir, "archive")
        self.index_path = os.path.join(base_dir, "index.json")
        self.latest_path = os.path.join(self.current_dir, "latest.json")
        self.interrupt_path = os.path.join(self.current_dir, "interrupted.json")

    # ------------------------------------------------------------------ setup
    def initialize(self) -> None:
        os.makedirs(self.archive_dir, exist_ok=True)
        index = {"current_version": self.version_id, "archives": []}
        version_matches = False
        existing: dict[str, Any] = {}
        if os.path.exists(self.index_path):
            existing = _read_json(self.index_path)
            index["archives"] = existing.get("archives", [])
            version_matches = existing.get("current_version") == self.version_id

        has_legacy_snapshots = bool(self.list_snapshot_paths())
        if not version_matches and not os.path.exists(self.index_path) and has_legacy_snapshots:
            # Snapshots from an earlier engine run predate the index file.
            # Trust their embedded version id instead of archrowing blindly.
            snapshot, _ = self.load_latest_valid()
            if snapshot is not None and snapshot.version_id == self.version_id:
                version_matches = True
        if not version_matches and (os.path.isdir(self.current_dir) and has_legacy_snapshots):
            old_version = existing.get("current_version", "unknown")
            archive_target = os.path.join(self.archive_dir, old_version)
            counter = 1
            while os.path.exists(archive_target):
                archive_target = os.path.join(self.archive_dir, f"{old_version}-{counter}")
                counter += 1
            shutil.move(self.current_dir, archive_target)
            index["archives"] = index["archives"] + [old_version]

        os.makedirs(self.current_dir, exist_ok=True)
        if not os.path.exists(self.index_path) or not version_matches:
            _atomic_write_text(self.index_path, canonical_json(index))

    def _snapshot_path(self, sequence: int) -> str:
        return os.path.join(self.current_dir, f"snapshot_{sequence:04d}.json")

    # ------------------------------------------------------------------ writes
    def save_snapshot(self, snapshot: Snapshot) -> str:
        if snapshot.version_id != self.version_id:
            raise SolverError(E_VERSION, "refusing to persist a foreign-version snapshot")
        content = snapshot.to_content()
        document = {"digest": snapshot_digest(content), "content": content}
        path = self._snapshot_path(snapshot.sequence)
        _atomic_write_text(path, canonical_json(document))
        _atomic_write_text(self.latest_path, canonical_json(document))
        return path

    def write_interrupted(self, snapshot: Snapshot, reason: str, exit_code: int) -> None:
        payload = {
            "version_id": self.version_id,
            "sequence": snapshot.sequence,
            "next_index": snapshot.next_index,
            "reason": reason,
            "exit_code": exit_code,
        }
        _atomic_write_text(self.interrupt_path, canonical_json(payload))

    def clear_interrupted(self) -> None:
        if os.path.exists(self.interrupt_path):
            os.unlink(self.interrupt_path)

    # ------------------------------------------------------------------ reads
    def list_snapshot_paths(self) -> list[str]:
        if not os.path.isdir(self.current_dir):
            return []
        names = sorted(
            name
            for name in os.listdir(self.current_dir)
            if name.startswith("snapshot_") and name.endswith(".json")
        )
        return [os.path.join(self.current_dir, name) for name in names]

    def load_latest_valid(self) -> tuple[Snapshot | None, SolverError | None]:
        """Return the newest intact snapshot, or a digest error for reporting."""
        paths = self.list_snapshot_paths()
        digest_error: SolverError | None = None
        valid: Snapshot | None = None
        for path in paths:
            try:
                snapshot = parse_snapshot_file(path)
            except SolverError as exc:
                if exc.code == E_DIGEST:
                    digest_error = exc
                    continue
                raise
            if snapshot.version_id != self.version_id:
                continue
            valid = snapshot
        if valid is not None:
            return valid, digest_error
        if digest_error is not None:
            return None, digest_error
        return None, None

    def read_interrupted(self) -> dict[str, Any] | None:
        if not os.path.exists(self.interrupt_path):
            return None
        return _read_json(self.interrupt_path)

    def verify_all(self) -> list[dict[str, Any]]:
        """Digest every snapshot; used by the CLI verify command."""
        reports = []
        for path in self.list_snapshot_paths():
            try:
                snapshot = parse_snapshot_file(path)
                reports.append(
                    {
                        "file": os.path.basename(path),
                        "ok": True,
                        "version_id": snapshot.version_id,
                        "sequence": snapshot.sequence,
                    }
                )
            except SolverError as exc:
                reports.append(
                    {
                        "file": os.path.basename(path),
                        "ok": False,
                        "code": exc.code,
                        "message": exc.message,
                    }
                )
        return reports
