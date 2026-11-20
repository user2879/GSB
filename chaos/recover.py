"""Crash recovery: reconcile ledger + snapshots with the current session.

Rules:
  * reopening verifies the ledger content digest (tampering -> E_SNAPSHOT);
  * format version / spec digest / seed / timeline digest mismatches between
    the snapshot (or ledger meta) and the current session trigger a FULL
    REPLACE of injections, transitions, requests, observations and progress
    (never an incremental merge); differences are listed stably (sorted);
  * injections found with committed=0 were in-flight at the crash: they are
    marked `unknown` and scheduled for redo with the original seed;
  * the recovery report states recovered/unknown counts and the redo plan;
    it never fabricates an exit report for the killed process.
"""

import json
import os
import time

from .errors import ChaosError, E_SNAPSHOT
from .ledger import FORMAT_VERSION, Ledger
from .timeline import digest as canonical_digest


class RecoveryReport:
    def __init__(self):
        self.replaced = False
        self.diffs = []
        self.recovered = []
        self.unknown = []
        self.redo_plan = []

    def to_dict(self):
        return {
            "replaced": self.replaced,
            "diffs": self.diffs,
            "recovered_count": len(self.recovered),
            "unknown_count": len(self.unknown),
            "recovered": self.recovered,
            "unknown": self.unknown,
            "redo_plan": self.redo_plan,
        }


def _snapshot_path(snapshot_dir, version):
    return os.path.join(snapshot_dir, "snapshot-v%d.json" % version)


def list_snapshots(snapshot_dir):
    if not os.path.isdir(snapshot_dir):
        return []
    names = [n for n in os.listdir(snapshot_dir)
             if n.startswith("snapshot-v") and n.endswith(".json")]
    return sorted(names)


def write_snapshot(snapshot_dir, meta):
    os.makedirs(snapshot_dir, exist_ok=True)
    path = _snapshot_path(snapshot_dir, FORMAT_VERSION)
    payload = {
        "format_version": FORMAT_VERSION,
        "spec_digest": meta["spec_digest"],
        "seed": meta["seed"],
        "timeline_digest": meta["timeline_digest"],
        "content_digest": meta["content_digest"],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2)
    return path


def _session_meta(spec, timeline_digest_value):
    return {
        "format_version": str(FORMAT_VERSION),
        "spec_digest": canonical_digest(spec.to_dict()),
        "seed": str(spec.seed),
        "timeline_digest": timeline_digest_value,
    }


def _full_replace(ledger, session, report, log):
    with ledger.transaction() as cur:
        for table in ("injections", "transitions", "requests",
                      "observations", "progress"):
            cur.execute("DELETE FROM %s" % table)
        for key, value in session.items():
            ledger.set_meta(key, value, cur)
    report.replaced = True
    log.emit("ledger_full_replace", diffs=report.diffs)


def reconcile(spec, timeline_digest_value, ledger_path, snapshot_dir, log):
    """Open the ledger and reconcile with the session. Returns (Ledger, report)."""
    ledger = Ledger(ledger_path)
    report = RecoveryReport()
    session = _session_meta(spec, timeline_digest_value)
    meta = ledger.meta_dict()

    if not meta:
        with ledger.transaction() as cur:
            for key, value in session.items():
                ledger.set_meta(key, value, cur)
        write_snapshot(snapshot_dir, ledger.meta_dict())
        log.emit("ledger_initialized", ledger=str(ledger_path))
        return ledger, report

    # 1. integrity: content digest must match, otherwise tampering.
    if not ledger.verify_digest():
        ledger.close()
        raise ChaosError(E_SNAPSHOT,
                         "ledger content digest mismatch (tampered or corrupt)",
                         field="meta.content_digest")

    # 2. version / declaration drift -> full replace, never merge.
    checks = ("format_version", "spec_digest", "seed", "timeline_digest")
    diffs = []
    for key in checks:
        if meta.get(key) != session[key]:
            diffs.append({"field": key, "ledger": meta.get(key),
                          "session": session[key]})
    snap_names = list_snapshots(snapshot_dir)
    if snap_names:
        with open(os.path.join(snapshot_dir, snap_names[-1]), encoding="utf-8") as fh:
            snap = json.load(fh)
        for key in checks:
            if str(snap.get(key)) != session[key]:
                diffs.append({"field": "snapshot." + key,
                              "snapshot": snap.get(key),
                              "session": session[key]})
    if diffs:
        report.diffs = sorted(diffs, key=lambda d: d["field"])
        _full_replace(ledger, session, report, log)
        write_snapshot(snapshot_dir, ledger.meta_dict())
        return ledger, report

    # 3. consistent ledger: mark in-flight (uncommitted) injections unknown.
    now_mono = int(time.monotonic() * 1000)
    with ledger.transaction() as cur:
        rows = cur.execute(
            "SELECT seq, state FROM injections WHERE committed=0"
            " ORDER BY seq").fetchall()
        for seq, state in rows:
            cur.execute("UPDATE injections SET state='unknown',"
                        " business_code='E_RECOVER_UNKNOWN' WHERE seq=?", (seq,))
            cur.execute(
                "INSERT INTO transitions(inj_seq,from_state,to_state,reason,mono_ms)"
                " VALUES(?,?,?,?,?)",
                (seq, state, "unknown",
                 "uncommitted at crash; requires redo with original seed", now_mono))
        committed = [r[0] for r in cur.execute(
            "SELECT seq FROM injections WHERE committed=1 ORDER BY seq").fetchall()]
    report.recovered = committed
    report.unknown = [seq for seq, _ in rows]
    report.redo_plan = list(report.unknown)
    if report.unknown:
        log.emit("recovery_unknown", unknown=report.unknown)
    log.emit("recovery_report", **report.to_dict())
    return ledger, report
