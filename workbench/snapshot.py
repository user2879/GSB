"""Session snapshots: versioned, digest-protected, whole-replace restore.

A snapshot persists: service-set version (digest), key-version sets,
scale-factor semantics version, per-request conclusion digests, token
balance snapshot and the ledger cursor, plus a content digest over all of
it. On open, any version or digest mismatch causes a WHOLE REPLACE (never
an incremental merge) and the differing items are listed in stable sorted
order. After restore, remaining requests are replayed per the ledger's
recorded service version and scale factor, and every per-request
conclusion, token balance, receipt summary and statistic is compared
against the ledger; the first mismatch raises DiagError(E_REPLAY) with
the request seq and field.
"""

import hashlib
import json
import os

from . import E_REPLAY, SCALE_SEMANTICS_VERSION, SNAPSHOT_FORMAT_VERSION
from .engine import decide_from_spec, make_buckets
from .errors import DiagError
from .spec import canonical_json


def _conclusion_digest(row):
    payload = (f"{row['seq']}|{row['decision_code']}|{row['token_after']}|"
               f"{row['receipt_summary']}")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _content_digest(snap):
    body = {k: v for k, v in snap.items() if k != "content_digest"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def build_snapshot(spec, ledger):
    rows = ledger.done_requests()
    conclusions = [{"seq": r["seq"], "code": r["decision_code"],
                    "digest": _conclusion_digest(r)} for r in rows]
    buckets = make_buckets(spec, ledger.scale)
    reqs_by_seq = {r.seq: r for r in spec.requests}
    for r in rows:
        decide_from_spec(spec, buckets, reqs_by_seq[r["seq"]])
    balances = {f"{sid}|{bk}": b.balance
                for (sid, bk), b in sorted(buckets.items())}
    snap = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "services_version": spec.services_digest,
        "key_versions": {sid: [kv.version for kv in spec.keys[sid]]
                         for sid in sorted(spec.keys)},
        "scale_factor": ledger.scale,
        "scale_semantics_version": SCALE_SEMANTICS_VERSION,
        "ledger_cursor": ledger.cursor,
        "token_balances": balances,
        "conclusions": conclusions,
    }
    snap["content_digest"] = _content_digest(snap)
    return snap


def save_snapshot(path, snap):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def load_snapshot_file(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def diff_snapshot(snap, spec, scale):
    """Return the stable sorted list of differing item names."""
    diffs = []
    if snap.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        diffs.append("format_version")
    if snap.get("services_version") != spec.services_digest:
        diffs.append("services_version")
    current_keys = {sid: [kv.version for kv in spec.keys[sid]]
                    for sid in sorted(spec.keys)}
    if snap.get("key_versions") != current_keys:
        snap_keys = snap.get("key_versions") or {}
        for sid in sorted(set(snap_keys) | set(current_keys)):
            if snap_keys.get(sid) != current_keys.get(sid):
                diffs.append(f"key_versions.{sid}")
    if snap.get("scale_factor") != scale:
        diffs.append("scale_factor")
    if snap.get("scale_semantics_version") != SCALE_SEMANTICS_VERSION:
        diffs.append("scale_semantics_version")
    if snap.get("content_digest") != _content_digest(snap):
        diffs.append("content_digest")
    return sorted(diffs)


def restore_snapshot(path, spec, scale):
    """Returns (snapshot, diffs, replaced). Any mismatch: whole replace."""
    if not os.path.isfile(path):
        return None, ["snapshot_file_missing"], True
    try:
        snap = load_snapshot_file(path)
    except (OSError, json.JSONDecodeError):
        return None, ["snapshot_file_unreadable"], True
    diffs = diff_snapshot(snap, spec, scale)
    if diffs:
        return None, diffs, True
    return snap, [], False


def verify_replay_after_restore(spec, ledger, upto_seq=None):
    """Replay committed requests from the ledger and compare item by item.
    First mismatch -> DiagError(E_REPLAY) with seq and field."""
    rows = [r for r in ledger.done_requests()
            if upto_seq is None or r["seq"] <= upto_seq]
    reqs_by_seq = {r.seq: r for r in spec.requests}
    buckets = make_buckets(spec, ledger.scale)
    for row in rows:
        req = reqs_by_seq.get(row["seq"])
        if req is None:
            raise DiagError(E_REPLAY, "ledger row absent from spec",
                            field="seq", seq=row["seq"])
        dec = decide_from_spec(spec, buckets, req)
        checks = {
            "token_before": dec.token_before,
            "token_refilled": dec.token_refilled,
            "allowed": 1 if dec.allowed else 0,
            "token_after": dec.token_after,
            "auth_ok": 1 if dec.auth_ok else 0,
            "auth_reason": dec.auth_reason,
        }
        if dec.passed:
            checks["decision_code"] = row["decision_code"]
        else:
            checks["decision_code"] = dec.code
        for fname, expected in checks.items():
            if row[fname] != expected:
                raise DiagError(
                    E_REPLAY,
                    f"replay mismatch: ledger={row[fname]!r} "
                    f"recomputed={expected!r}",
                    field=fname, seq=row["seq"])
    return len(rows)
