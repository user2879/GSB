from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .latency import STATS_SCHEMA_VERSION
from .ledger import TreeSnapshot
from .util import atomic_write_json, read_json


SESSION_SCHEMA = "cluster-diag-session-v1"


@dataclass(frozen=True)
class SessionOpenResult:
    active: dict[str, Any]
    replaced: bool
    reasons: tuple[str, ...]
    prior: dict[str, Any] | None
    compared_root_revision: int | None
    compared_node_count: int


def build_session_document(
    snapshot: TreeSnapshot,
    latency_stats: dict[str, Any],
    *,
    saved_at_ns: int | None = None,
) -> dict[str, Any]:
    return {
        "schema": SESSION_SCHEMA,
        "root_revision": snapshot.root_revision,
        "root_snapshot_digest": snapshot.snapshot_digest,
        "node_digests": snapshot.node_digest_map(),
        "latency_schema_version": STATS_SCHEMA_VERSION,
        "latency_stats": latency_stats,
        "saved_at_ns": time.time_ns() if saved_at_ns is None else int(saved_at_ns),
        "recovery_policy": "replace-whole-session-on-root-or-node-digest-mismatch",
    }


def save_session(path: str | Path, document: dict[str, Any]) -> None:
    atomic_write_json(path, document)


def _compare_root_and_nodes(document: Any, snapshot: TreeSnapshot) -> list[str]:
    reasons: list[str] = []
    if not isinstance(document, dict):
        return ["snapshot_is_not_an_object"]
    if document.get("schema") != SESSION_SCHEMA:
        reasons.append("snapshot_schema_mismatch")
    if document.get("root_revision") != snapshot.root_revision:
        reasons.append("root_revision_mismatch")
    expected_nodes = snapshot.node_digest_map()
    observed_nodes = document.get("node_digests")
    if not isinstance(observed_nodes, dict):
        reasons.append("node_digests_missing_or_invalid")
    else:
        if observed_nodes != expected_nodes:
            missing = sorted(set(expected_nodes) - set(observed_nodes))
            extra = sorted(set(observed_nodes) - set(expected_nodes))
            changed = sorted(
                key
                for key in set(expected_nodes) & set(observed_nodes)
                if expected_nodes[key] != observed_nodes[key]
            )
            if missing or extra or changed:
                reasons.append("node_digest_set_mismatch")
            # Retain detailed evidence in a stable order for replay/reporting.
            document["_comparison_detail"] = {
                "missing": missing,
                "extra": extra,
                "changed": changed,
            }
    if document.get("latency_schema_version") != STATS_SCHEMA_VERSION:
        reasons.append("latency_schema_version_mismatch")
    return reasons


def open_session(
    path: str | Path,
    *,
    snapshot: TreeSnapshot,
    current_latency_stats: dict[str, Any],
) -> SessionOpenResult:
    snapshot_path = Path(path)
    fresh = build_session_document(snapshot, current_latency_stats)
    if not snapshot_path.exists():
        return SessionOpenResult(
            active=fresh,
            replaced=False,
            reasons=("snapshot_absent_initialized_from_ledger",),
            prior=None,
            compared_root_revision=snapshot.root_revision,
            compared_node_count=snapshot.node_count,
        )
    try:
        prior = read_json(snapshot_path)
    except Exception as exc:
        return SessionOpenResult(
            active=fresh,
            replaced=True,
            reasons=(f"snapshot_unreadable:{type(exc).__name__}",),
            prior=None,
            compared_root_revision=snapshot.root_revision,
            compared_node_count=snapshot.node_count,
        )

    reasons = _compare_root_and_nodes(prior, snapshot)
    if reasons:
        return SessionOpenResult(
            active=fresh,
            replaced=True,
            reasons=tuple(reasons),
            prior=prior,
            compared_root_revision=snapshot.root_revision,
            compared_node_count=snapshot.node_count,
        )
    return SessionOpenResult(
        active=prior,
        replaced=False,
        reasons=("snapshot_matches_ledger",),
        prior=prior,
        compared_root_revision=snapshot.root_revision,
        compared_node_count=snapshot.node_count,
    )
