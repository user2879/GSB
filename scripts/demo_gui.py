"""Demo: generate the sample, run the workbench, then open the Tkinter
diagnostics UI. Requires a display (run on the Windows desktop)."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    spec = os.path.join(ROOT, "samples", "spec.json")
    requests = os.path.join(ROOT, "samples", "requests.ndjson")
    db = os.path.join(ROOT, "work", "ledger.db")
    snapshots = os.path.join(ROOT, "work", "snapshots")

    subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts", "gen_sample.py")],
                          cwd=ROOT)
    subprocess.check_call([
        sys.executable, "-m", "quota_workbench", "run",
        "--spec", spec, "--requests", requests, "--db", db,
        "--snapshot-dir", snapshots,
    ], cwd=ROOT)
    subprocess.check_call([
        sys.executable, "-m", "quota_workbench", "gui",
        "--db", db, "--snapshot-dir", snapshots,
    ], cwd=ROOT)


if __name__ == "__main__":
    main()
