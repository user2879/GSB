"""Versioned JSON session snapshots with wholesale replacement on mismatch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from .hashing import canonical_hash


SNAPSHOT_FORMAT_VERSION = 1


def snapshot_content_digest(snapshot: dict[str, Any]) -> str:
    material = {key: value for key, value in snapshot.items() if key not in {"content_digest", "snapshot_digest"}}
    return canonical_hash(material)


def write_snapshot(path: str | Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(snapshot)
    payload["format_version"] = SNAPSHOT_FORMAT_VERSION
    payload["content_digest"] = snapshot_content_digest(payload)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return payload


def read_snapshot(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def open_or_replace_snapshot(
    path: str | Path,
    *,
    session_id: str,
    snapshot_version: int,
    spec_digest: str,
    factory: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open a snapshot or replace it wholesale.

    A missing file is created from ``factory``.  Any format/version/spec/content
    mismatch discards the entire parsed object and writes a fresh object; no
    field-level merge is attempted.
    """

    path = Path(path)
    existing = read_snapshot(path)
    info: dict[str, Any] = {
        "existed": existing is not None,
        "replaced": False,
        "path": str(path),
        "session_id": session_id,
        "snapshot_version": snapshot_version,
        "spec_digest": spec_digest,
    }
    mismatch_reason: str | None = None
    if existing is None:
        mismatch_reason = "missing"
    elif not isinstance(existing, dict):
        mismatch_reason = "not_object"
    elif existing.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        mismatch_reason = "format_version"
    elif existing.get("snapshot_version") != snapshot_version:
        mismatch_reason = "snapshot_version"
    elif existing.get("spec_digest") != spec_digest:
        mismatch_reason = "spec_digest"
    elif existing.get("content_digest") != snapshot_content_digest(existing):
        mismatch_reason = "content_digest"

    if mismatch_reason in {None, "missing"}:
        if existing is not None:
            assert mismatch_reason is None
            info["replaced"] = False
            info["reason"] = None
            return existing, info
    fresh = factory()
    fresh.setdefault("session_id", session_id)
    fresh["snapshot_version"] = snapshot_version
    fresh["spec_digest"] = spec_digest
    written = write_snapshot(path, fresh)
    info["replaced"] = existing is not None and mismatch_reason != "missing"
    info["created"] = mismatch_reason == "missing"
    info["reason"] = mismatch_reason if mismatch_reason != "missing" else None
    return written, info
