"""Routing-session snapshots with version- and digest-checked recovery."""

from __future__ import annotations

import json
import os

from .canon import canonical_text, digest
from .errors import BizError, E_SNAPSHOT

SNAPSHOT_FIELDS = (
    "rule_version",
    "ring_version",
    "dep_graph_version",
    "budget_version",
    "leases",
    "cache_digest",
    "ledger_cursor",
)


def build_snapshot(versions, leases, cache_digest, ledger_cursor) -> dict:
    payload = {
        "rule_version": versions.rule_version,
        "ring_version": versions.ring_version,
        "dep_graph_version": versions.dep_graph_version,
        "budget_version": versions.budget_version,
        "leases": leases,
        "cache_digest": cache_digest,
        "ledger_cursor": ledger_cursor,
    }
    return {"payload": payload, "content_digest": digest(payload)}


def save_snapshot(path: str, snapshot: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(canonical_text(snapshot))
    os.replace(tmp, path)


def load_snapshot(path: str, current_versions) -> dict:
    """Load and verify a snapshot.

    Raises E_SNAPSHOT when the content digest does not match or any persisted
    version disagrees with the current one; the caller must then discard the
    snapshot wholesale and rebuild (never patch row by row).
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            snap = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise BizError(E_SNAPSHOT, f"snapshot unreadable: {exc}", field="$file")
    payload = snap.get("payload")
    if not isinstance(payload, dict) or digest(payload) != snap.get("content_digest"):
        raise BizError(
            E_SNAPSHOT, "snapshot content digest mismatch", field="content_digest"
        )
    current = current_versions.as_dict()
    for field in ("rule_version", "ring_version", "dep_graph_version",
                  "budget_version"):
        if payload.get(field) != current[field]:
            raise BizError(
                E_SNAPSHOT,
                f"snapshot {field}={payload.get(field)} != current "
                f"{current[field]}; wholesale replacement required",
                field=field,
            )
    return payload
