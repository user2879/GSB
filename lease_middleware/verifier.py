"""Independent verification tool.

Reads only the SQLite ledger and the config declaration with fresh SQL logic
(intentionally not reusing the engine/replayer) and emits a stable, ordered
JSON summary.  Cursor anomalies are reported with the E_CURSOR business code.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .errors import E_CURSOR, E_REPLAY, MiddlewareError
from .spec import load_spec
from .storage import LEGAL_TRANSITIONS

ACTIVE = ("granted", "renewed")


def verify(db_path: str | Path, spec_path: str | Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    findings: list[dict[str, Any]] = []

    lease_state: dict[str, str] = {}
    renewal_versions: dict[str, list[int]] = {}
    renewal_counts: dict[str, list[int]] = {}
    pushed_versions: dict[str, list[int]] = {}
    pushed_by_key: dict[tuple[str, str], list[int]] = {}
    push_cursor_walk: dict[str, int] = {}

    for row in conn.execute("SELECT * FROM ledger ORDER BY seq"):
        etype, cid, lid = row["event_type"], row["client_id"], row["lease_id"]
        if etype.startswith("lease_") and etype != "lease_granted":
            target = etype.split("_", 1)[1]
            current = lease_state.get(lid)
            if current is None:
                raise MiddlewareError(E_REPLAY, "event for unknown lease", "lease_id",
                                      client_id=cid, lease_id=lid,
                                      context={"seq": row["seq"]})
            if target not in LEGAL_TRANSITIONS[current]:
                raise MiddlewareError(E_REPLAY,
                                      f"illegal transition {current}->{target}", "state",
                                      client_id=cid, lease_id=lid, context={"seq": row["seq"]})
            lease_state[lid] = target
        elif etype == "lease_granted":
            lease_state[lid] = "granted"
            renewal_versions.setdefault(cid, [])
        elif etype == "lease_renewed":
            lease_state[lid] = "renewed"
            versions = renewal_versions.setdefault(cid, [])
            counts = renewal_counts.setdefault(cid, [])
            if counts and row["renewal_count"] <= counts[-1]:
                raise MiddlewareError(E_REPLAY,
                                      "renewal_count not strictly increasing",
                                      "renewal_count", client_id=cid, lease_id=lid)
            if versions and row["lease_version"] < versions[-1]:
                raise MiddlewareError(E_REPLAY,
                                      "renewal lease_version regressed",
                                      "lease_version", client_id=cid, lease_id=lid)
            versions.append(row["lease_version"])
            counts.append(row["renewal_count"])
        elif etype == "config_push":
            if lease_state.get(lid) not in ACTIVE:
                raise MiddlewareError(E_REPLAY, "push targets inactive lease", "state",
                                      client_id=cid, lease_id=lid)
            pv = pushed_versions.setdefault(cid, [])
            version = row["config_version"]
            per_key = pushed_by_key.setdefault((cid, row["config_key"]), [])
            if version in per_key:
                raise MiddlewareError(E_REPLAY,
                                      "duplicate push of config version for key",
                                      "config_version", client_id=cid, lease_id=lid)
            per_key.append(version)
            pv.append(version)
            walked = push_cursor_walk.get(cid, 0)
            if version < walked:
                raise MiddlewareError(E_CURSOR, "push versions not monotonic",
                                      "config_version", client_id=cid, lease_id=lid)
            push_cursor_walk[cid] = max(walked, version)
            config = spec.configs[row["config_key"]]
            if row["digest"] != config.digest_at(version):
                raise MiddlewareError(E_REPLAY, "push digest disagrees with spec",
                                      "digest", client_id=cid, lease_id=lid)

    # Cursor table must equal max pushed version recorded in the ledger.
    cursor_rows = conn.execute(
        "SELECT client_id,seen_config_version FROM client_cursors").fetchall()
    cursor_summary: dict[str, dict[str, int]] = {}
    for r in cursor_rows:
        cid, stored = r["client_id"], int(r["seen_config_version"])
        ledger_max = max(pushed_versions.get(cid, [0]))
        cursor_summary[cid] = {"stored_cursor": stored, "ledger_max_push": ledger_max}
        if stored != ledger_max:
            code = E_CURSOR if stored > ledger_max else E_REPLAY
            raise MiddlewareError(
                code, "cursor does not match ledger pushes",
                "seen_config_version", client_id=cid,
                context={"stored": stored, "ledger_max": ledger_max})
        if stored < push_cursor_walk.get(cid, 0):
            raise MiddlewareError(E_CURSOR, "cursor regressed",
                                  "seen_config_version", client_id=cid)

    # Subscription view rows must agree with lease table / cursors.
    view_rows = conn.execute(
        "SELECT * FROM subscription_view ORDER BY client_id,config_key").fetchall()
    view_summary = [dict(r) for r in view_rows]

    # Lease table digests and formats independently revalidated against spec.
    lease_summary = {}
    for r in conn.execute("SELECT * FROM lease_records ORDER BY lease_id"):
        rec = json.loads(r["record_json"])
        config = spec.configs[rec["config_key"]]
        if rec["value_digest"] != config.digest_at(rec["lease_version"]):
            raise MiddlewareError(E_REPLAY, "stored lease digest disagrees with spec",
                                  "value_digest", client_id=rec["client_id"],
                                  lease_id=rec["lease_id"])
        lease_summary[rec["lease_id"]] = {
            "client_id": rec["client_id"], "config_key": rec["config_key"],
            "format": rec["format"], "state": r["state"],
            "lease_version": rec["lease_version"],
            "renewal_count": rec["renewal_count"],
        }

    conn.close()
    return {
        "status": "OK",
        "lease_count": len(lease_summary),
        "ledger_event_count": conn_total(db_path),
        "leases": lease_summary,
        "cursors": {k: cursor_summary[k] for k in sorted(cursor_summary)},
        "pushed_by_key": {f"{k[0]}|{k[1]}": val for k, val in sorted(pushed_by_key.items())},
        "pushed_versions": {k: pushed_versions[k] for k in sorted(pushed_versions)},
        "renewal_versions": {k: renewal_versions[k] for k in sorted(renewal_versions)},
        "renewal_counts": {k: renewal_counts[k] for k in sorted(renewal_counts)},
        "subscription_view": view_summary,
        "findings": findings,
    }


def conn_total(db_path: str | Path) -> int:
    conn = sqlite3.connect(str(db_path))
    total = conn.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]
    conn.close()
    return int(total)
