"""End-to-end acceptance driver; standard library only, Windows/macOS/Linux safe."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
SAMPLES = os.path.join(ROOT, "samples")
ENV = dict(os.environ, PYTHONPATH=SRC)


def cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "swa", *args],
        cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=180,
    )


def require(condition: bool, label: str) -> None:
    print(("PASS" if condition else "FAIL"), "-", label)
    if not condition:
        raise SystemExit(1)


def main() -> None:
    work = tempfile.mkdtemp(prefix="swa-acceptance-")
    spec = os.path.join(SAMPLES, "spec.json")

    def p(name: str, ext: str) -> str:
        return os.path.join(work, f"{name}.{ext}")

    digests = set()
    for limit in (1, 2, 3, 5):
        r = cli("run", "--spec", spec, "--ledger", p(f"g{limit}", "sqlite"),
                "--result", p(f"g{limit}", "json"), "--log", p(f"g{limit}", "log"),
                "--global-limit", str(limit), "--report", p(f"g{limit}-report", "json"))
        require(r.returncode == 0, f"clean run global-limit={limit}")
        digests.add(json.load(open(p(f"g{limit}", "json"), encoding="utf-8"))["result_digest"])
    require(len(digests) == 1, "byte-identical results across scheduling interleavings")

    metrics = json.load(open(p("g3-report", "json"), encoding="utf-8"))["scheduler"]
    require(metrics["max_running_global"] <= 3, "global shard cap never exceeded")
    require(all(v <= 3 for v in metrics["max_running_by_rule"].values()), "per-rule quota never exceeded")
    require(all(v <= 8 for v in metrics["max_memory_items_by_window"].values()), "memory item budget never exceeded")

    r = cli("run", "--spec", spec, "--ledger", p("crash", "sqlite"),
            "--result", p("crash", "json"), "--log", p("crash", "log"),
            "--crash-after-commits", "3", "--report", p("crash-report", "json"))
    require(r.returncode == 137, "forced termination exit code 137")
    r = cli("run", "--spec", spec, "--ledger", p("crash", "sqlite"),
            "--result", p("recovered", "json"), "--log", p("crash2", "log"),
            "--report", p("recovered-report", "json"))
    require(r.returncode == 0, "recovery rerun succeeds")
    conn = sqlite3.connect(p("crash", "sqlite"))
    rows = conn.execute("SELECT count(*) FROM window_results").fetchone()[0]
    distinct = conn.execute(
        "SELECT count(*) FROM (SELECT rule_id, window_start_ms FROM window_results "
        "GROUP BY rule_id, window_start_ms)"
    ).fetchone()[0]
    conn.close()
    require(rows == distinct, "no duplicate window records after recovery")
    require(open(p("recovered", "json"), "rb").read() == open(p("g3", "json"), "rb").read(),
            "recovered result bytes match a clean run")

    r = cli("replay", "--spec", spec, "--ledger", p("g3", "sqlite"),
            "--log", p("replay", "log"), "--report", p("replay", "json"))
    replay = json.load(open(p("replay", "json"), encoding="utf-8"))
    require(r.returncode == 0 and replay["status"] == "ok", "replay verifies committed windows")

    r = cli("run", "--spec", os.path.join(SAMPLES, "spec-v2.json"),
            "--ledger", p("g3", "sqlite"), "--result", p("v2", "json"),
            "--log", p("v2", "log"), "--report", p("v2-report", "json"))
    v2 = json.load(open(p("v2-report", "json"), encoding="utf-8"))
    require(r.returncode == 0 and v2["view_replacement"]["replaced"], "rule evolution replaces the current view")

    # Build a clean v1 ledger, then open it with a same-version parameter drift.
    cli("run", "--spec", spec, "--ledger", p("driftbase", "sqlite"),
        "--result", p("driftbase", "json"), "--log", p("driftbase", "log"))
    r = cli("run", "--spec", os.path.join(SAMPLES, "spec-drift.json"),
            "--ledger", p("driftbase", "sqlite"), "--result", p("drift", "json"),
            "--log", p("drift", "log"), "--report", p("drift-report", "json"))
    drift = json.load(open(p("drift-report", "json"), encoding="utf-8"))
    require(r.returncode == 7 and drift["code"] == "E_REPLAY", "same-version drift reports E_REPLAY")

    r = cli("run", "--spec", os.path.join(SAMPLES, "spec-corrupt.json"),
            "--ledger", p("es", "sqlite"), "--result", p("es", "json"),
            "--log", p("es", "log"), "--inject", "corrupt-spec", "--report", p("es-report", "json"))
    require(r.returncode == 2 and json.load(open(p("es-report", "json"), encoding="utf-8"))["code"] == "E_SPEC", "E_SPEC injection")
    r = cli("run", "--spec", spec, "--ledger", p("el", "sqlite"), "--result", p("el", "json"),
            "--log", p("el", "log"), "--inject", "unsupported-ledger-version", "--report", p("el-report", "json"))
    require(r.returncode == 5 and json.load(open(p("el-report", "json"), encoding="utf-8"))["code"] == "E_LEDGER", "E_LEDGER injection")
    r = cli("run", "--spec", os.path.join(SAMPLES, "spec-tight-budget.json"),
            "--ledger", p("eb", "sqlite"), "--result", p("eb", "json"),
            "--log", p("eb", "log"), "--inject", "budget-exhausted", "--report", p("eb-report", "json"))
    require(r.returncode == 6 and json.load(open(p("eb-report", "json"), encoding="utf-8"))["code"] == "E_BUDGET", "E_BUDGET injection")

    print("\nAll acceptance checks passed. Artifacts:", work)


if __name__ == "__main__":
    main()
