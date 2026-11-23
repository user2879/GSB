"""One-shot deterministic validation runner.

Runs the clean arbitration demo, all three fault injections, and the
forced-termination recovery path, then prints a compact verification summary.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"


def run(*args: str) -> tuple[int, dict | None]:
    proc = subprocess.run(
        [sys.executable, "-m", "arbworkbench", *args],
        cwd=ROOT, capture_output=True, text=True,
    )
    payload = None
    stream = proc.stderr if proc.returncode else proc.stdout
    try:
        payload = json.loads(stream)
    except json.JSONDecodeError:
        payload = {"raw_tail": stream[-400:]}
    return proc.returncode, payload


def main() -> int:
    print("== clean run ==")
    code, payload = run("--fresh")
    print("exit", code, "unknown_views", payload["unknown_views"])
    for view in payload["views"]:
        print(f"  {view['view_id']} v{view['version']} keys={sorted(view['keys'])}")

    print("\n== fault injections ==")
    for injection, expected in (
        ("spec_corruption", "E_SPEC"),
        ("grid_violation", "E_GRID"),
        ("version_mismatch", "E_VERSION"),
    ):
        code, payload = run("--fresh", "--inject", injection)
        actual = payload.get("code")
        print(f"  {injection}: exit={code} expected={expected} actual={actual} "
              f"location={payload.get('location')}")
        assert code == 2 and actual == expected

    print("\n== forced termination recovery ==")
    code, payload = run("--fresh", "--crash-view", "view-production")
    print("exit", code, "unknown_views", payload["unknown_views"])
    assert code == 0 and payload["unknown_views"] == []

    print("\nALL VALIDATION SCENARIOS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
