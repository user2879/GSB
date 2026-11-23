"""Acceptance demo: lease-window convergence, timing anomalies, and
forced-kill recovery with replay verification. Local samples only."""

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def cli(*args, expect=0):
    proc = subprocess.run([sys.executable, "-m", "convd.cli"] + list(args),
                          cwd=ROOT, capture_output=True, text=True,
                          timeout=180)
    if proc.returncode != expect:
        raise SystemExit("cli %s failed: %s%s" % (args, proc.stdout,
                                                  proc.stderr))
    return proc


def cursor_of(db):
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='cursor'").fetchone()
        return json.loads(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def main():
    work = tempfile.mkdtemp(prefix="convd-demo-")
    scenario = os.path.join(ROOT, "samples", "scenario.json")
    executors = os.path.join(ROOT, "samples", "executors.json")

    print("== 1. lease-window convergence + timeout/retry isolation ==")
    db1 = os.path.join(work, "run1.db")
    out = cli("run", "--scenario", scenario, "--executors", executors,
              "--db", db1, "--drop", "exe-beta:3")
    print(json.loads(out.stdout)["aggregate"])
    summary = json.loads(cli("summary", "--db", db1).stdout)
    print("retries:", json.dumps(summary["retries"]))
    print("ledger digest:", summary["ledger_digest"])

    print("== 2. lease expiry (real timing) ==")
    tight = os.path.join(work, "tight.json")
    with open(scenario) as fh:
        data = json.load(fh)
    data["scenario_id"] = "demo-conv-tight"
    data["lease_ms"] = 2
    data["receipt_timeout_ms"] = 300
    with open(tight, "w") as fh:
        json.dump(data, fh)
    ex2 = os.path.join(work, "executors2.json")
    with open(executors) as fh:
        ex = json.load(fh)
    for i, spec in enumerate(ex["executors"]):
        spec["port"] = 39201 + i
    with open(ex2, "w") as fh:
        json.dump(ex, fh)
    db2 = os.path.join(work, "run2.db")
    cli("run", "--scenario", tight, "--executors", ex2, "--db", db2,
        "--base-ms", "30", "--jitter-ms", "0")
    summary2 = json.loads(cli("summary", "--db", db2).stdout)
    reasons = {eid: [n["reason"] for n in s["not_converged"]]
               for eid, s in summary2["summary"].items()}
    print("not-converged reasons:", json.dumps(reasons))

    print("== 3. forced kill -> resume -> replay verification ==")
    db3 = os.path.join(work, "run3.db")
    long_scenario = os.path.join(work, "long.json")
    data["scenario_id"] = "demo-conv-long"
    data["lease_ms"] = 250
    data["receipt_timeout_ms"] = 120
    data["max_rounds"] = 10
    data["pulse_interval_ms"] = 150
    with open(long_scenario, "w") as fh:
        json.dump(data, fh)
    ex3 = os.path.join(work, "executors3.json")
    for i, spec in enumerate(ex["executors"]):
        spec["port"] = 39301 + i
    with open(ex3, "w") as fh:
        json.dump(ex, fh)
    proc = subprocess.Popen(
        [sys.executable, "-m", "convd.cli", "run", "--scenario",
         long_scenario, "--executors", ex3, "--db", db3],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 60
    while cursor_of(db3) < 4 and time.time() < deadline:
        time.sleep(0.1)
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)
    subprocess.run(["pkill", "-f", "convd.executor.*demo-conv-long"],
                   capture_output=True)
    time.sleep(0.3)
    committed = cursor_of(db3)
    print("killed after committed cursor:", committed)
    out = cli("resume", "--scenario", long_scenario, "--executors", ex3,
              "--db", db3)
    print("resumed rounds:", json.loads(out.stdout)["resumed_rounds"])
    out = cli("verify", "--scenario", long_scenario, "--executors", ex3,
              "--db", db3)
    payload = json.loads(out.stdout)
    print("replay:", payload["replay"])
    print("aggregate ok:", payload["aggregate"]["ok"])
    shutil.rmtree(work, ignore_errors=True)
    print("DEMO OK")


if __name__ == "__main__":
    main()
