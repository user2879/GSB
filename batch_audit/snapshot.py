"""Atomic snapshot file used to detect stale generations on reopen."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import SNAPSHOT_FORMAT_VERSION
from .timeutil import utc_now


@dataclass(frozen=True)
class Snapshot:
    generation: int
    fingerprint: Mapping[str, Any]
    status: str
    completed_slots: tuple[int, ...] = field(default_factory=tuple)
    empty_slots: tuple[int, ...] = field(default_factory=tuple)
    unknown_slots: tuple[int, ...] = field(default_factory=tuple)
    written_at: str = field(default_factory=utc_now)
    format_version: int = SNAPSHOT_FORMAT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_format_version": int(self.format_version),
            "version": int(self.generation),
            "generation": int(self.generation),
            "fingerprint": dict(self.fingerprint),
            "status": self.status,
            "completed_slots": list(self.completed_slots),
            "empty_slots": list(self.empty_slots),
            "unknown_slots": list(self.unknown_slots),
            "written_at": self.written_at,
        }


def read_snapshot(path: Path) -> Snapshot | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read snapshot {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("snapshot root must be an object")
    if int(value.get("snapshot_format_version", -1)) != SNAPSHOT_FORMAT_VERSION:
        raise ValueError("unsupported snapshot_format_version")
    generation = value.get("generation", value.get("version"))
    if "generation" in value and "version" in value and value["generation"] != value["version"]:
        raise ValueError("snapshot generation and version disagree")
    if type(generation) is not int or generation <= 0:
        raise ValueError("snapshot generation must be a positive integer")
    fingerprint = value.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("snapshot fingerprint must be an object")
    status = value.get("status")
    if not isinstance(status, str):
        raise ValueError("snapshot status must be a string")
    return Snapshot(
        generation=generation,
        fingerprint=fingerprint,
        status=status,
        completed_slots=_int_tuple(value.get("completed_slots", [])),
        empty_slots=_int_tuple(value.get("empty_slots", [])),
        unknown_slots=_int_tuple(value.get("unknown_slots", [])),
        written_at=str(value.get("written_at", utc_now())),
    )


def write_snapshot(path: Path, snapshot: Snapshot) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        snapshot.as_dict(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def snapshot_from_slots(
    *,
    generation: int,
    fingerprint: Mapping[str, Any],
    status: str,
    slots: Sequence[Mapping[str, Any]],
) -> Snapshot:
    committed = tuple(sorted(int(row["batch_slot"]) for row in slots if row["status"] == "committed"))
    empty = tuple(sorted(int(row["batch_slot"]) for row in slots if row["status"] == "empty"))
    unknown = tuple(
        sorted(
            int(row["batch_slot"])
            for row in slots
            if row["status"] in {"pending", "running", "unknown"}
        )
    )
    return Snapshot(
        generation=int(generation),
        fingerprint=dict(fingerprint),
        status=status,
        completed_slots=tuple(sorted(committed + empty)),
        empty_slots=empty,
        unknown_slots=unknown,
    )


def _int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError("snapshot slot lists must contain integers")
    return tuple(int(item) for item in value)
