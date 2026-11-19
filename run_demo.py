"""End-to-end acceptance demonstration on deterministic local samples.

Runs the full matrix of business scenarios and prints a compact report:
normal solve, conflict, invalid input, interrupt/resume, tamper rejection,
digest fallback, version archiving and hard-kill recovery.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "output" / "demo"
VALID_SPEC = ROOT / "samples" / "sample_valid.json"
CONFLICT_SPEC = ROOT / "samples" / "sample_conflict.json"
INVALID_SPEC = ROOT / "samples" / "sample_invalid.json"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "run_solver.py"), *args],
        capture_output=True, text=True, cwd=ROOT,
    )


def main() -> int:
    shutil.rmtree(DEMO, ignore_errors=True)
    DEMO.mkdir(parents=True, exist_ok=True)
    failures = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        status = "PASS" if condition else "FAIL"
        print(f"[{status}] {name} {detail}")
        if not condition:
            failures.append(name)

    # 1. Normal solve.
    proc = run("solve", "--spec", str(VALID_SPEC),
               "--snapshot-dir", str(DEMO / "valid"),
               "--output", str(DEMO / "valid_result.json"))
    full = json.loads((DEMO / "valid_result.json").read_text())
    check("normal solve exits 0", proc.returncode == 0)
    check("all constraints applied once",
          all(value == 1 for value in full["constraint_counts"].values()),
          str(full["constraint_counts"]))

    # 2. Unsatisfiable specification.
    proc = run("solve", "--spec", str(CONFLICT_SPEC),
               "--snapshot-dir", str(DEMO / "conflict"))
    error = json.loads(proc.stderr)["error"]
    check("conflict returns E_CONFLICT", error["code"] == "E_CONFLICT")
    check("conflict names the failing constraint",
          bool(error["details"].get("constraint")))

    # 3. Invalid input creates nothing.
    proc = run("solve", "--spec", str(INVALID_SPEC),
               "--snapshot-dir", str(DEMO / "invalid"))
    error = json.loads(proc.stderr)["error"]
    check("invalid input returns business error", error["code"].startswith("E_"))
    check("invalid input creates no snapshot directory",
          not (DEMO / "invalid").exists())

    # 4. Determinism: two independent runs are cell-identical.
    other = run("solve", "--spec", str(VALID_SPEC),
                "--snapshot-dir", str(DEMO / "determinism"))
    other_payload = json.loads(other.stdout)
    check("repeated runs produce identical grids",
          other_payload["grid"] == full["grid"])
    check("repeated runs produce identical counts",
          other_payload["constraint_counts"] == full["constraint_counts"])

    # 5. Interrupt then resume.
    proc = run("solve", "--spec", str(VALID_SPEC),
               "--snapshot-dir", str(DEMO / "resume"),
               "--interrupt-after", "2")
    suspended = json.loads(proc.stdout)
    check("interrupt uses exit code 130", proc.returncode == 130)
    check("interrupt reports next index", suspended["next_index"] == 2)
    proc = run("solve", "--spec", str(VALID_SPEC),
               "--snapshot-dir", str(DEMO / "resume"),
               "--output", str(DEMO / "resume_result.json"))
    resumed = json.loads((DEMO / "resume_result.json").read_text())
    check("resume completes with exit 0", proc.returncode == 0)
    check("resume marks itself as resumed", resumed["resumed"] is True)
    check("resumed grid equals uninterrupted grid", resumed["grid"] == full["grid"])

    # 6. Tamper detection.
    snapshots = sorted((DEMO / "resume" / "current").glob("snapshot_*.json"))
    target = snapshots[2]
    document = json.loads(target.read_text())
    document["content"]["grid"][0][0] = 2
    target.write_text(json.dumps(document))
    proc = run("verify", "--spec", str(VALID_SPEC),
               "--snapshot-dir", str(DEMO / "resume"))
    check("tampered snapshot fails verify", proc.returncode == 3)
    check("verify reports E_DIGEST",
          any(s.get("code") == "E_DIGEST" for s in json.loads(proc.stdout)["snapshots"]))

    # 7. Fallback to the newest intact snapshot after tampering the latest one.
    fallback_dir = DEMO / "fallback"
    run("solve", "--spec", str(VALID_SPEC),
        "--snapshot-dir", str(fallback_dir), "--interrupt-after", "3")
    newest = sorted(fallback_dir.glob("current/snapshot_*.json"))[-1]
    document = json.loads(newest.read_text())
    document["digest"] = "0" * 64
    newest.write_text(json.dumps(document))
    proc = run("solve", "--spec", str(VALID_SPEC),
               "--snapshot-dir", str(fallback_dir))
    payload = json.loads(proc.stdout)
    check("fallback resume completes", proc.returncode == 0)
    check("fallback records E_DIGEST warning",
          any(w["code"] == "E_DIGEST" for w in payload["digest_warnings"]))
    check("fallback result matches full run", payload["grid"] == full["grid"])

    # 8. Version change archives the previous run.
    versioned_dir = DEMO / "versioned"
    run("solve", "--spec", str(VALID_SPEC),
        "--snapshot-dir", str(versioned_dir), "--interrupt-after", "2")
    changed = json.loads(VALID_SPEC.read_text())
    changed["seed"] += 1
    changed_spec = DEMO / "changed_spec.json"
    changed_spec.write_text(json.dumps(changed))
    run("solve", "--spec", str(changed_spec),
        "--snapshot-dir", str(versioned_dir))
    check("old version is archived", any((versioned_dir / "archive").iterdir()))

    # 9. Hard-kill simulation: marker lost, snapshots still safe.
    hard_dir = DEMO / "hardkill"
    run("solve", "--spec", str(VALID_SPEC),
        "--snapshot-dir", str(hard_dir), "--interrupt-after", "2")
    (hard_dir / "current" / "interrupted.json").unlink()
    proc = run("solve", "--spec", str(VALID_SPEC),
               "--snapshot-dir", str(hard_dir))
    payload = json.loads(proc.stdout)
    check("hard-kill recovery matches full run", payload["grid"] == full["grid"])

    print()
    if failures:
        print(f"DEMO FAILED: {len(failures)} scenario(s): {failures}")
        return 1
    print("ALL DEMO SCENARIOS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
