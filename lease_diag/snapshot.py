"""Session snapshots: versioning, content digest and whole-state replacement.

A snapshot captures the pool-set version, rollback-target version, scaling
convention version, per-pool watermarks, the full lease set and the ledger
cursor.  The digest is a SHA-256 over the canonical JSON of the payload
excluding the digest itself.  On reopen, any version or digest mismatch means
the on-disk lease/watermark state is distrusted and replaced wholesale from
the ledger (never incrementally merged).
"""
from __future__ import annotations

import hashlib
import json

from .errors import E_SNAPSHOT, LeaseError

SCALING_VERSION = "scaling/fp-v1"


def canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def digest_of(payload: dict) -> str:
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def build_snapshot(*, pool_set_version: str, rollback_version: str,
                   scale: int, watermarks: list[dict], leases: dict,
                   cursor: dict) -> dict:
    payload = {
        "pool_set_version": pool_set_version,
        "rollback_version": rollback_version,
        "scaling_version": SCALING_VERSION,
        "scale": int(scale),
        "watermarks": sorted(watermarks, key=lambda item: item["scope"]),
        "leases": {lease_id: leases[lease_id]
                   for lease_id in sorted(leases)},
        "cursor": cursor,
    }
    return {"payload": payload, "digest": digest_of(payload)}


def expected_versions(*, pool_set_version: str, rollback_version: str,
                      scale: int) -> dict:
    return {
        "pool_set_version": pool_set_version,
        "rollback_version": rollback_version,
        "scaling_version": SCALING_VERSION,
        "scale": int(scale),
    }


def verify_snapshot(snapshot: dict, expected: dict) -> None:
    """Raise E_SNAPSHOT on any digest or version mismatch."""
    payload = snapshot["payload"]
    mismatches: list[str] = []
    actual_digest = digest_of(payload)
    if actual_digest != snapshot["digest"]:
        mismatches.append(
            f"digest mismatch (stored={snapshot['digest'][:12]}..., "
            f"recomputed={actual_digest[:12]}...)")
    for key, value in expected.items():
        if payload.get(key) != value:
            mismatches.append(
                f"version mismatch on {key}: snapshot={payload.get(key)!r} "
                f"current={value!r}")
    if mismatches:
        raise LeaseError(E_SNAPSHOT,
                         "snapshot does not match current session state; "
                         "performing whole-state replacement from ledger",
                         mismatches)


def whole_replace_leases(engine, ledger) -> dict:
    """Rebuild engine state entirely from durable ledger records."""
    engine.leases.clear()
    engine.active_index.clear()
    engine.key_location.clear()
    engine.events.clear()
    engine.decisions.clear()
    engine._seq = 0
    engine._lease_serial.clear()
    engine.rebuild_from_ledger(ledger.events())
    # Restore per (pool, key) lease serials so resumed grants keep the same
    # deterministic lease ids they would have in a single uninterrupted run.
    for lease in engine.leases.values():
        serial = int(lease.lease_id.rsplit("-", 1)[1])
        token = (lease.pool_id, lease.key)
        engine._lease_serial[token] = max(
            engine._lease_serial.get(token, 0), serial)
    return {
        "recovered_leases": len(engine.leases),
        "active_leases": len(engine.active_index),
        "last_event_seq": engine._seq,
        "source": "ledger-whole-replacement",
    }
