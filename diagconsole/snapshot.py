"""Session snapshots: chain version, root pointer, node digests, progress cursor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .errors import DiagError, canonical_json, raise_snapshot

SNAPSHOT_VERSION = 1


def chain_fingerprint(revisions) -> dict:
    """Compact fingerprint of the revision chain declarations."""
    payload = [
        {"version": r.version, "parent": r.parent, "values": r.values,
         "strategy": r.strategy, "derives": list(r.derives)}
        for r in revisions
    ]
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return {"digest": digest, "head": payload[-1]["version"],
            "count": len(payload)}


def build_snapshot(spec, root_version: int, nodes: list[dict],
                   progress_cursor: dict) -> dict:
    fingerprint = chain_fingerprint(spec.revisions)
    node_records = []
    for node in sorted(nodes, key=lambda n: (n["version"], n["task_id"],
                                             n["cluster_id"])):
        node_records.append({
            "task_id": node["task_id"],
            "cluster_id": node["cluster_id"],
            "version": node["version"],
            "parent": node["parent"],
            "aggregate": node["aggregate"] if "aggregate" in node
            else json.loads(node["aggregate_json"]),
            "content_digest": node["content_digest"],
            "status": node.get("status", "committed"),
        })
    snapshot = {
        "snapshot_version": SNAPSHOT_VERSION,
        "chain": fingerprint,
        "root_version": root_version,
        "progress_cursor": progress_cursor,
        "nodes": node_records,
    }
    snapshot["snapshot_digest"] = hashlib.sha256(
        canonical_json({key: snapshot[key] for key in
                        ("snapshot_version", "chain", "root_version",
                         "progress_cursor", "nodes")}).encode("utf-8")).hexdigest()
    return snapshot


def save_snapshot(path: str | Path, snapshot: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, sort_keys=True, separators=(",", ":"),
                  ensure_ascii=False)
    tmp.replace(path)


def load_snapshot(path: str | Path, spec=None) -> dict:
    """Load and verify a snapshot.

    Any mismatch (format, snapshot digest, chain fingerprint) raises
    ``E_SNAPSHOT`` so callers fully replace state instead of merging.
    """
    path = Path(path)
    if not path.exists():
        raise DiagError("E_SNAPSHOT", f"snapshot file not found: {path}",
                        {"path": str(path)})
    try:
        with open(path, "r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
    except json.JSONDecodeError as exc:
        raise_snapshot("snapshot is not valid JSON",
                       {"path": str(path), "detail": str(exc)})
    stored = snapshot.get("snapshot_digest")
    body = {key: snapshot.get(key) for key in
            ("snapshot_version", "chain", "root_version",
             "progress_cursor", "nodes")}
    actual = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    if stored != actual:
        raise_snapshot(
            "snapshot content digest does not match; full replacement required",
            {"path": str(path), "expected": stored, "actual": actual})
    if spec is not None:
        current = chain_fingerprint(spec.revisions)
        if current != snapshot.get("chain"):
            raise_snapshot(
                "revision chain version differs from the snapshot; "
                "full replacement required",
                {"path": str(path),
                 "snapshot_chain": snapshot.get("chain"),
                 "current_chain": current})
    return snapshot
