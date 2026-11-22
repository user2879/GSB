"""Session snapshots with version-aware wholesale replacement.

A snapshot persists the rule version, concurrency parameter version, breaker
states, per-window receipt digests and ledger cursor, all protected by a
content digest.  Reopening with a different version or digest replaces the
whole snapshot (never an incremental merge) and returns a stable diff listing.
A tampered ledger content summary raises ``E_SNAPSHOT``.
"""

import json
import os

from .canonical import canonical_dumps, content_digest
from .errors import SnapshotError

SNAPSHOT_FORMAT = 1


def build_snapshot(rule_version, params_version, breaker_snapshots,
                   receipt_digests, cursor_sequence, head_digest):
    body = {
        "snapshot_format": SNAPSHOT_FORMAT,
        "rule_version": rule_version,
        "params_version": params_version,
        "breaker_snapshots": sorted(
            breaker_snapshots,
            key=lambda row: (row["service_id"], row["instance_id"]),
        ),
        "receipt_digests": dict(sorted(receipt_digests.items())),
        "cursor_sequence": cursor_sequence,
        "ledger_head_digest": head_digest,
    }
    body["content_digest"] = content_digest({k: v for k, v in body.items()})
    return body


def save_snapshot(path, snapshot):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True, indent=2))


def load_snapshot(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
    except FileNotFoundError as exc:
        raise SnapshotError("snapshot file not found", path=path) from exc
    except json.JSONDecodeError as exc:
        raise SnapshotError("snapshot is not valid JSON", path=path,
                            line=exc.lineno) from exc
    stored_digest = snapshot.pop("content_digest", None)
    actual = content_digest(snapshot)
    if stored_digest != actual:
        raise SnapshotError(
            "snapshot content digest mismatch; wholesale replacement required",
            stored=stored_digest, expected=actual,
        )
    snapshot["content_digest"] = stored_digest
    return snapshot


def diff_snapshots(old, new):
    """Stable, order-independent difference between two snapshots."""
    diff = {
        "rule_version_changed": old.get("rule_version") != new.get("rule_version"),
        "params_version_changed": old.get("params_version") != new.get("params_version"),
        "old_rule_version": old.get("rule_version"),
        "new_rule_version": new.get("rule_version"),
        "receipts_removed": sorted(
            set(old.get("receipt_digests", {})) - set(new.get("receipt_digests", {}))
        ),
        "receipts_added": sorted(
            set(new.get("receipt_digests", {})) - set(old.get("receipt_digests", {}))
        ),
        "receipts_changed": sorted(
            window_id for window_id in (
                set(old.get("receipt_digests", {})) & set(new.get("receipt_digests", {}))
            )
            if old["receipt_digests"][window_id] != new["receipt_digests"][window_id]
        ),
        "cursor_changed": old.get("cursor_sequence") != new.get("cursor_sequence"),
    }
    return diff


def restore_or_replace(path, new_snapshot):
    """Open a prior snapshot; on any mismatch replace it wholesale."""
    if not os.path.exists(path):
        save_snapshot(path, new_snapshot)
        return None, {"replaced": False, "reason": "no-prior-snapshot"}
    old = load_snapshot(path)
    if (old.get("rule_version") != new_snapshot.get("rule_version")
            or old.get("params_version") != new_snapshot.get("params_version")
            or old.get("content_digest") != new_snapshot.get("content_digest")):
        diff = diff_snapshots(old, new_snapshot)
        save_snapshot(path, new_snapshot)
        return old, {"replaced": True, "diff": diff}
    return old, {"replaced": False, "reason": "identical"}


def assert_ledger_matches_snapshot(snapshot, ledger):
    """E_SNAPSHOT when the live ledger head digest disagrees with the snapshot."""
    expected = snapshot.get("ledger_head_digest")
    actual = ledger.head_digest()
    if expected != actual:
        raise SnapshotError(
            "ledger content digest does not match snapshot",
            expected=expected, actual=actual,
        )
    return True


def snapshot_canonical_bytes(snapshot):
    return canonical_dumps({k: v for k, v in snapshot.items() if k != "content_digest"})
