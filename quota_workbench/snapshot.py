"""Session snapshots: version identification, whole-replacement recovery.

A snapshot persists the tenant-set / instance-set / probe-policy /
arbitration-rule versions, the final quota state, the completed-request
cursor and a content digest. When a snapshot is re-opened, its versions
and digest are compared against the current declaration; any mismatch
triggers a whole replacement of tenant, instance and ledger state
(never an incremental merge) and the differing items are listed in a
stable order.
"""

from __future__ import annotations

import hashlib
import json
import os

from . import FORMAT_VERSION
from .errors import SnapshotError
from .slog import StructuredLogger, NULL_LOGGER
from .spec import Spec

SNAPSHOT_FIELDS = (
    "format_version",
    "tenant_set_version",
    "instance_set_version",
    "probe_policy_version",
    "rule_version",
    "quota_final",
    "cursor",
    "spec_digest",
    "content_digest",
)


def _digest_blob(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_snapshot(spec: Spec, quota_final: dict, cursor: int, content_digest: str) -> dict:
    body = {
        "format_version": FORMAT_VERSION,
        "tenant_set_version": spec.tenant_set_version,
        "instance_set_version": spec.instance_set_version,
        "probe_policy_version": spec.probe_policy.policy_version,
        "rule_version": spec.rule_version,
        "quota_final": quota_final,
        "cursor": cursor,
        "spec_digest": spec.digest(),
        "content_digest": content_digest,
    }
    body["snapshot_digest"] = _digest_blob(body)
    return body


def write_snapshot(snapshot: dict, directory: str, name: str = "latest.json") -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(snapshot, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def load_snapshot(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def verify_snapshot(path: str, spec: Spec, ledger_content_digest,
                    logger: StructuredLogger = NULL_LOGGER) -> dict:
    """Compare a snapshot against the current declaration and ledger.
    Returns a report. Raises SnapshotError (E_SNAPSHOT) listing the
    stable diff items when anything mismatches."""
    snap = load_snapshot(path)
    diffs = []

    stored_digest = snap.get("snapshot_digest")
    recomputed = _digest_blob({k: snap.get(k) for k in SNAPSHOT_FIELDS})
    if stored_digest != recomputed:
        diffs.append("snapshot_digest")

    expected = {
        "format_version": FORMAT_VERSION,
        "tenant_set_version": spec.tenant_set_version,
        "instance_set_version": spec.instance_set_version,
        "probe_policy_version": spec.probe_policy.policy_version,
        "rule_version": spec.rule_version,
        "spec_digest": spec.digest(),
    }
    for key in sorted(expected):
        if snap.get(key) != expected[key]:
            diffs.append(key)
    if ledger_content_digest is not None and snap.get("content_digest") != ledger_content_digest:
        diffs.append("content_digest")

    diffs = sorted(set(diffs))
    report = {"snapshot": path, "ok": not diffs, "diffs": diffs}
    if diffs:
        logger.log("snapshot_mismatch", code="E_SNAPSHOT", diffs=diffs, snapshot=path)
        raise SnapshotError(
            "snapshot does not match current declaration; whole replacement required",
            field="snapshot",
            details=report,
        )
    logger.log("snapshot_ok", snapshot=path)
    return report


def replace_all(db_path: str, snapshot_dir: str, logger: StructuredLogger = NULL_LOGGER) -> dict:
    """Whole-replacement recovery: drop the ledger and snapshots so the
    next run rebuilds state from the current declaration. Never merges."""
    removed = []
    if os.path.exists(db_path):
        os.remove(db_path)
        removed.append(db_path)
    if os.path.isdir(snapshot_dir):
        for name in sorted(os.listdir(snapshot_dir)):
            if name.endswith(".json"):
                os.remove(os.path.join(snapshot_dir, name))
                removed.append(os.path.join(snapshot_dir, name))
    logger.log("snapshot_replace_all", removed=removed)
    return {"removed": removed}
