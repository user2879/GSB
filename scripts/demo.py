"""Deterministic end-to-end demonstration for the cfgdist platform.

Runs the whole happy path plus every documented fault-injection business code
against local loopback subprocesses.  Usage:

    python scripts/demo.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(label: str, *args: str, expect: int = 0) -> dict:
    command = [PYTHON, "-m", "cfgdist", *args]
    proc = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=180
    )
    status = "PASS" if proc.returncode == expect else "FAIL"
    print(f"[{status}] {label} (exit={proc.returncode}, expected={expect})")
    if proc.returncode != expect:
        print(proc.stderr.strip()[:800])
    return {"label": label, "exit": proc.returncode, "expected": expect,
            "stdout": proc.stdout, "stderr": proc.stderr}


def main() -> int:
    for folder in ("data", "logs"):
        shutil.rmtree(ROOT / folder, ignore_errors=True)

    common = ["--samples-dir", "samples", "--data-dir", "data",
              "--log-dir", "logs", "--grid-q", "2"]
    results = []
    results.append(run("hard validation", *common, "validate"))
    results.append(run(
        "three-version evolution (real subprocesses + SQLite)",
        *common, "run", "--session-id", "demo",
        "--concurrency", "2", "--max-attempts", "3", "--timeout-ms", "10000",
    ))
    results.append(run("authoritative ledger replay", *common,
                       "replay", "--session-id", "demo"))
    results.append(run("interactive view (headless)", *common,
                       "view", "--snapshot", "data/snapshots/demo-v3.json"))

    # Fault injection: manifest / cycle / grid.
    results.append(run("inject corrupt manifest -> E_MANIFEST",
                       *common, "inject", "corrupt", expect=2))
    results.append(run("inject derivation cycle -> E_CYCLE",
                       *common, "inject", "cycle", expect=3))
    results.append(run("inject illegal grid -> E_GRID",
                       *common, "inject", "bad_grid", "--bad-grid-q", "99", expect=4))

    # Node-level faults on a throwaway session, version 1 only.
    results.append(run("handshake timeout -> E_HANDSHAKE",
                       *common, "--data-dir", "data/fault-hs",
                       "--log-dir", "logs/fault-hs",
                       "run", "--session-id", "fhs", "--versions", "1",
                       "--inject-node", "node-c=handshake_fail", expect=7))
    results.append(run("node startup failure -> E_NODE",
                       *common, "--data-dir", "data/fault-node",
                       "--log-dir", "logs/fault-node",
                       "run", "--session-id", "fnode", "--versions", "1",
                       "--inject-node", "node-b=node_fail", expect=6))

    failures = [r for r in results if r["exit"] != r["expected"]]
    print("\n=== summary ===")
    for r in results:
        print(f"  {r['label']:<48} exit={r['exit']}")
    print(f"\n{len(results) - len(failures)}/{len(results)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
