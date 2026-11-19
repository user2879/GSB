"""Session snapshots: atomic persistence, versioning and content digests."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any

from .config import canonical_dumps
from .errors import SnapshotError

SNAPSHOT_FORMAT_VERSION = 1


def content_digest(payload: dict[str, Any]) -> str:
    """SHA-256 over the canonical serialization of all content fields."""
    material = {key: value for key, value in payload.items()
                if key not in ("format_version", "content_digest")}
    return hashlib.sha256(
        canonical_dumps(material).encode("utf-8")).hexdigest()


def write_snapshot(path: str, payload: dict[str, Any]) -> None:
    """Atomic snapshot write: temp file + fsync + rename + directory fsync."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    body = dict(payload)
    body["format_version"] = SNAPSHOT_FORMAT_VERSION
    body["content_digest"] = content_digest(body)
    fd, tmp_path = tempfile.mkstemp(prefix=".snap-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(body, handle, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def read_snapshot(path: str, *, expect_format: int = SNAPSHOT_FORMAT_VERSION,
                  tampered: bool = False) -> dict[str, Any]:
    """Load and verify a snapshot.

    ``tampered=True`` simulates on-disk corruption used by the E_SNAPSHOT
    injection harness without touching implementation source code: the digest
    check is performed against the file content and must fail.
    """
    if not os.path.exists(path):
        raise SnapshotError(f"snapshot file not found: {path}", path=path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SnapshotError("snapshot is not valid JSON",
                            path=path, detail=exc.msg)
    if not isinstance(payload, dict):
        raise SnapshotError("snapshot root must be an object", path=path)
    fmt = payload.get("format_version")
    if not isinstance(fmt, int) or fmt != expect_format:
        raise SnapshotError(
            f"unsupported snapshot format version {fmt!r}; expected "
            f"{expect_format}", path=path, got=fmt)
    stored = payload.get("content_digest")
    actual = content_digest(payload)
    if not isinstance(stored, str) or stored != actual:
        raise SnapshotError(
            "snapshot content digest mismatch; full replacement required",
            path=path, stored_digest=stored, recomputed_digest=actual,
            tampered=tampered)
    return payload
