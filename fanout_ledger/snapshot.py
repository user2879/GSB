"""Session snapshots with content digests and atomic whole-file replacement.

A snapshot captures the full resumable state: scene version, payload digest,
mock-service failure configuration, per-node attempt usage, global/per-node
budget consumption, the scheduling trajectory (ordered attempt records), the
aggregation result and the SLO judgment.

The digest covers the canonical JSON of every field except ``digest`` itself,
so tampering with any byte is detected and reported as ``E_SNAPSHOT``.
Writes go through a temporary file + ``os.replace`` so readers only ever see
a fully committed snapshot (no half-written state).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SnapshotError
from .ledger import canonical_json

SNAPSHOT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class Snapshot:
    body: dict[str, Any]
    digest: str
    path: Path | None = None


def cleanup_staging_files(snapshot_dir: str | Path) -> list[str]:
    """Remove atomic-write staging files left by a force-killed process.

    Because snapshots become visible only via ``os.replace``, a ``.snap-*.tmp``
    file can never be a valid committed snapshot; it is safe to delete during
    startup recovery.
    """
    snapshot_dir = Path(snapshot_dir)
    removed: list[str] = []
    if not snapshot_dir.exists():
        return removed
    for staging in snapshot_dir.glob(".snap-*.tmp"):
        staging.unlink()
        removed.append(str(staging))
    return removed


def compute_digest(body: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def build_snapshot(session_id: str, scene_version: str, payload_digest: str,
                   mock_config: list[dict[str, Any]], node_attempts_used: dict[str, int],
                   budget_global_ms: int, budget_per_node_ms: dict[str, int],
                   trajectory: list[dict[str, Any]], metrics: dict[str, Any],
                   judgment: dict[str, Any], status: str) -> Snapshot:
    # Sorted node ids for stable serialized layout; trajectory keeps run order.
    body = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "session_id": session_id,
        "scene_version": scene_version,
        "payload_digest": payload_digest,
        "mock_config": sorted(mock_config, key=lambda item: item["node_id"]),
        "node_attempts_used": dict(sorted(node_attempts_used.items())),
        "budget_global_ms": budget_global_ms,
        "budget_per_node_ms": dict(sorted(budget_per_node_ms.items())),
        "trajectory": trajectory,
        "metrics": metrics,
        "judgment": judgment,
        "status": status,
        "captured_at_ms": int(time.monotonic_ns() // 1_000_000),
    }
    return Snapshot(body=body, digest=compute_digest(body))


def write_snapshot(snapshot: Snapshot, snapshot_dir: str | Path,
                   name: str | None = None) -> Path:
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    cleanup_staging_files(snapshot_dir)
    name = name or f"snapshot-{snapshot.body['session_id']}-{snapshot.body['captured_at_ms']}.json"
    target = snapshot_dir / name
    envelope = {"digest": snapshot.digest, "body": snapshot.body}
    fd, tmp_name = tempfile.mkstemp(prefix=".snap-", suffix=".tmp", dir=str(snapshot_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, sort_keys=True, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)  # atomic whole-file replacement
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return target


def load_snapshot(path: str | Path) -> Snapshot:
    path = Path(path)
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SnapshotError(f"snapshot not found: {path}", {"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise SnapshotError(
            "snapshot file is not valid JSON",
            {"path": str(path), "json_error": exc.msg, "line": exc.lineno},
        ) from exc
    if not isinstance(envelope, dict) or "body" not in envelope or "digest" not in envelope:
        raise SnapshotError("snapshot envelope missing body/digest", {"path": str(path)})
    body = envelope["body"]
    recorded = envelope["digest"]
    actual = compute_digest(body)
    if recorded != actual:
        raise SnapshotError(
            "snapshot content digest mismatch",
            {
                "path": str(path),
                "recorded_digest": recorded,
                "actual_digest": actual,
            },
        )
    return Snapshot(body=body, digest=actual, path=path)


def list_snapshots(snapshot_dir: str | Path) -> list[Path]:
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        return []
    return sorted(
        (p for p in snapshot_dir.glob("snapshot-*.json") if p.is_file()),
        key=lambda p: p.name,
    )


def verify_compatibility(snapshot: Snapshot, scene_version: str, payload_digest: str,
                         mock_config: list[dict[str, Any]]) -> None:
    """Reject non-reproducible sessions before replaying (E_SNAPSHOT context)."""
    problems: list[dict[str, str]] = []
    body = snapshot.body
    if body.get("scene_version") != scene_version:
        problems.append({
            "item": "scene_version",
            "snapshot": str(body.get("scene_version")),
            "current": scene_version,
        })
    if body.get("payload_digest") != payload_digest:
        problems.append({
            "item": "payload_digest",
            "snapshot": str(body.get("payload_digest")),
            "current": payload_digest,
        })
    recorded_mock = {item["node_id"]: item for item in body.get("mock_config", [])}
    current_mock = {item["node_id"]: item for item in mock_config}
    for node_id in sorted(set(recorded_mock) | set(current_mock)):
        before = recorded_mock.get(node_id)
        now = current_mock.get(node_id)
        if before != now:
            problems.append({
                "item": f"mock_config.{node_id}",
                "snapshot": json.dumps(before, sort_keys=True),
                "current": json.dumps(now, sort_keys=True),
            })
    if problems:
        raise SnapshotError(
            "snapshot is not reproducible against the current scene/mock configuration",
            {"problems": problems},
        )
