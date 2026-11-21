"""Repeatable demo: four scenarios in order.

  1. full-path probing
  2. stratified-path probing + byte-identical verdict/transition check
  3. budget parameter change -> stale marking + wholesale view replacement
  4. hard kill (SIGKILL) -> recovery -> replay verification

Run from the project root:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from lbhc.engine import run_engine          # noqa: E402
from lbhc.ledger import Ledger              # noqa: E402
from lbhc.replay import replay_verify       # noqa: E402
from lbhc.sample import sample_spec         # noqa: E402
from lbhc.spec import load_spec, write_spec  # noqa: E402

OUT = "out"
SPEC = "data/cluster.json"


def banner(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def fresh(name):
    path = os.path.join(OUT, name)
    if os.path.exists(path):
        os.unlink(path)
    return path


def canon(rep):
    states = {r["instance_id"]: (r["state"], r["last_reason"])
              for r in rep["final_view"]}
    trans = [(t["instance_id"], t["from_state"], t["to_state"],
              t["reason_category"]) for t in rep["transitions"]]
    return json.dumps({"states": states, "transitions": trans},
                      sort_keys=True).encode()


def print_view(ledger_path, version):
    db = sqlite3.connect(ledger_path)
    print(f"\nledger {ledger_path} current view (version {version}):")
    for row in db.execute(
            "SELECT instance_id,state,consec_success,consec_failure,"
            "last_reason,last_probe_seq FROM instance_view "
            "WHERE version=? AND stale=0 ORDER BY instance_id",
            (version,)):
        print("  %-4s state=%-9s succ=%d fail=%d reason=%-16s seq=%d" % row)
    print("transitions:")
    for row in db.execute(
            "SELECT probe_seq,instance_id,from_state,to_state,"
            "reason_category FROM transitions WHERE version=? "
            "ORDER BY probe_seq,id", (version,)):
        print("  seq=%-3d %-4s %s -> %s (%s)" % row)
    print("probe outcomes (first 12):")
    for row in db.execute(
            "SELECT seq,pool_id,instance_id,round(wait_ms,1),outcome,"
            "reason FROM probes WHERE version=? ORDER BY seq LIMIT 12",
            (version,)):
        print("  seq=%-3d %-9s %-4s wait=%6.1fms %-7s %s" % row)
    db.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(SPEC):
        write_spec(sample_spec(), SPEC)

    banner("Scenario 1: full-path probing (every instance, every round)")
    r1 = run_engine(SPEC, fresh("demo-full.db"), rounds=4, path="full",
                    report_path=os.path.join(OUT, "demo-full-report.json"),
                    log_name="demo-full")
    print("path:", r1["path"], "|", r1["path_params"]["reason"])
    print("initiated sequence:", r1["initiated_sequence"])
    print("peak inflight: global=%d (budget %d) pools=%s"
          % (r1["budget"]["peak_global_inflight"],
             r1["budget"]["global_budget"],
             r1["budget"]["peak_pool_inflight"]))
    print_view(os.path.join(OUT, "demo-full.db"), r1["spec_version"])

    banner("Scenario 2: stratified path + byte-identical verdict check")
    r2 = run_engine(SPEC, fresh("demo-strat.db"), rounds=4,
                    path="stratified",
                    report_path=os.path.join(OUT, "demo-strat-report.json"),
                    log_name="demo-strat")
    print("path:", r2["path"], "|", r2["path_params"]["reason"])
    print("initiated sequence:", r2["initiated_sequence"])
    identical = canon(r1) == canon(r2)
    print("verdicts+transitions byte-identical across paths:", identical)
    assert identical, "dual-path divergence detected"

    banner("Scenario 3: budget change -> stale guard + wholesale replace")
    evo_db = fresh("demo-evo.db")
    v1 = sample_spec(version=1)
    write_spec(v1, os.path.join(OUT, "demo-cluster-v1.json"))
    ra = run_engine(os.path.join(OUT, "demo-cluster-v1.json"), evo_db,
                    rounds=3, log_name="demo-evo-v1")
    v2 = sample_spec(version=2)
    v2["global"]["concurrency_budget"] = 3
    v2["pools"][0]["concurrency_quota"] = 2
    v2["pools"][0]["interval_ms"] = 300
    v2["pools"][0]["backends"] = [b for b in v2["pools"][0]["backends"]
                                  if b["id"] != "h5"]
    v2["pools"][0]["backends"].append(
        {"id": "h7", "host": "127.0.0.1", "port": 18620, "weight": 55,
         "role": "active", "behavior": "ok"})
    write_spec(v2, os.path.join(OUT, "demo-cluster-v2.json"))
    rb = run_engine(os.path.join(OUT, "demo-cluster-v2.json"), evo_db,
                    rounds=3, log_name="demo-evo-v2")
    print("v1 probes:", ra["replay"]["probes_replayed"],
          "| v2 probes:", rb["replay"]["probes_replayed"])
    print("stale replacement diff (old conclusions vs new):")
    for e in rb["stale_diff"]:
        print(" ", e)
    db = sqlite3.connect(evo_db)
    print("view rows by (version, stale):",
          db.execute("SELECT version,stale,COUNT(*) FROM instance_view "
                     "GROUP BY version,stale ORDER BY version").fetchall())
    db.close()

    banner("Scenario 4: kill -9 -> recovery -> replay verification")
    crash_db = fresh("demo-crash.db")
    crash_report = os.path.join(OUT, "demo-crash-report.json")
    env = dict(os.environ, PYTHONPATH=ROOT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "lbhc.cli", "run", "--spec", SPEC,
         "--ledger", crash_db, "--rounds", "8", "--report", crash_report],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # wait until several probes are committed, then hard-kill the process
    while True:
        time.sleep(0.1)
        try:
            db = sqlite3.connect(crash_db)
            n = db.execute("SELECT COUNT(*) FROM probes").fetchone()[0]
            db.close()
        except sqlite3.Error:
            n = 0
        if n >= 10:
            break
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    print("killed run exit code:", proc.returncode)
    print("exit report written by killed run:", os.path.exists(crash_report),
          "(must be False)")
    db = sqlite3.connect(crash_db)
    committed = db.execute("SELECT COUNT(*) FROM probes").fetchone()[0]
    leftovers = db.execute("SELECT seq,instance_id FROM inflight "
                           "ORDER BY seq").fetchall()
    db.close()
    print("committed probes at kill:", committed,
          "| uncommitted inflight:", leftovers)

    rr = run_engine(SPEC, crash_db, rounds=8, report_path=crash_report,
                    log_name="demo-crash-resume")
    print("resumed run: probes=%d transitions=%d"
          % (rr["replay"]["probes_replayed"],
             rr["replay"]["transitions_verified"]))
    print("exit report written after recovery:",
          os.path.exists(crash_report))
    spec = load_spec(SPEC)
    ledger = Ledger(crash_db)
    result = replay_verify(ledger, spec)
    ledger.close()
    print("replay verification:", result)
    print_view(crash_db, spec.version)

    banner("ALL SCENARIOS PASSED")


if __name__ == "__main__":
    main()
