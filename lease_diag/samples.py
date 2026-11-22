"""Deterministic local sample generation (no external services).

The generated scenario deliberately exercises grants, renewals, real-time
expiry, priority preemption at a renewal boundary, rollback convergence and
same-key contention.  Three injection modes corrupt the artefacts so the
E_ROLLBACK / E_GAP / E_SNAPSHOT paths can be demonstrated repeatably.
"""
from __future__ import annotations

import json
from pathlib import Path

POOL_DOC = {
    "pool_set_version": "pool-set/v1",
    "rollback_version": "rollback/v1",
    "pools": [
        {"pool_id": "P_A", "capacity": 2, "lease_ms": 400, "renew_ms": 120,
         "priority": 10, "quota": 2},
        {"pool_id": "P_B", "capacity": 2, "lease_ms": 300, "renew_ms": 90,
         "priority": 5, "quota": 1},
        {"pool_id": "P_C", "capacity": 1, "lease_ms": 500, "renew_ms": 150,
         "priority": 2, "quota": 1},
    ],
    "rollback": [
        {"key": "db.conn", "from_rev": 5, "target_rev": 3},
    ],
}

# (request_id, key, pool, revision, priority, arrive_ms)
REQUEST_ROWS = [
    ("R01", "db.conn",   "P_A", 1,  4,    0),
    ("R02", "cache.ttl", "P_A", 1,  3,   20),
    ("R03", "feat.flag", "P_B", 1,  4,   40),
    ("R04", "db.conn",   "P_A", 2,  4,  100),   # renew, count -> 1
    ("R05", "audit.lvl", "P_B", 1,  2,  160),   # grant (B has room)
    ("R06", "db.conn",   "P_A", 3,  4,  220),   # renew, count -> 2
    ("R07", "throttle",  "P_A", 1,  9,  280),   # preempts cache.ttl (prio 3)
    ("R08", "cache.ttl", "P_A", 1,  3,  300),   # E_LEASE: A still full, no lower victim
    ("R09", "throttle",  "P_A", 2,  9,  340),   # renewal by holder
    ("R10", "db.conn",   "P_A", 5,  4,  400),   # renew; rev 5 -> rollback edge
    ("R11", "audit.lvl", "P_B", 2,  2,  420),   # renew before expiry
    ("R12", "edge.mode", "P_C", 1,  1,  460),   # sole C lease
    ("R13", "edge.mode", "P_C", 2,  1,  520),   # renewal
    ("R14", "feat.flag", "P_B", 2,  4,  560),   # renewal
    ("R15", "cache.ttl", "P_A", 2,  8,  620),   # preempts db.conn(prio4) -> rollback to rev3
    ("R16", "db.conn",   "P_A", 6,  4,  660),   # E_LEASE: key rolled back and A still full
    ("R17", "feat.flag", "P_B", 3,  4,  900),   # after prior B leases expired
    ("R18", "audit.lvl", "P_B", 3,  2,  960),   # grant into recovered room
]


def write_samples(workdir: str | Path,
                  inject: str | None = None) -> dict:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    doc = json.loads(json.dumps(POOL_DOC))
    rows = [list(row) for row in REQUEST_ROWS]

    if inject == "rollback_cycle":
        doc["rollback"].append(
            {"key": "db.conn", "from_rev": 3, "target_rev": 5})
    elif inject == "missing_renewal":
        rows = [row for row in rows if row[0] != "R09"]
        # Keep a full copy for ledger-vs-stream gap detection below.
    # snapshot_digest is injected later against the produced SQLite snapshot.

    pools_path = workdir / "pools.json"
    requests_path = workdir / "requests.ndjson"
    gap_path = workdir / "requests_gap.ndjson"
    with open(pools_path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2, sort_keys=True)
    with open(requests_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_line(row) + "\n")
    # A stream with one renewal removed; used only during recovery to prove
    # that the ledger notices the missing renewal event (E_GAP).
    with open(gap_path, "w", encoding="utf-8") as handle:
        for row in rows:
            if row[0] == "R09":
                continue
            handle.write(_line(row) + "\n")
    return {"pools_path": pools_path, "requests_path": requests_path,
            "gap_path": gap_path, "inject": inject}


def _line(row: list) -> str:
    return json.dumps({
        "request_id": row[0], "key": row[1], "pool_id": row[2],
        "revision": row[3], "priority": row[4],
        "arrive_ms": row[5]}, sort_keys=True)
