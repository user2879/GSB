"""Crash recovery demo: kill the process mid-batch, then recover and replay."""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hotgray.gateway import Gateway
from hotgray.samples import build_sample_declaration
from hotgray.spec import validate_declaration

DECL_PATH = os.path.join(ROOT, "samples", "gateway_declaration.json")
DB_PATH = os.path.join(ROOT, "runtime", "demo_recovery.sqlite3")
PHASE1 = os.path.join(ROOT, "scripts", "_phase1_crashable.py")
PHASE2 = os.path.join(ROOT, "scripts", "_phase2_recover.py")


def main():
    os.makedirs(os.path.join(ROOT, "runtime"), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    declaration = build_sample_declaration()
    validate_declaration(declaration)
    with open(DECL_PATH, "w", encoding="utf-8") as handle:
        json.dump(declaration, handle, ensure_ascii=False, indent=2)

    phase1 = subprocess.run([sys.executable, PHASE1, DECL_PATH, DB_PATH],
                            capture_output=True, text=True)
    print("phase1 stdout:", phase1.stdout.strip())
    print("phase1 returncode (forced kill):", phase1.returncode)

    phase2 = subprocess.run([sys.executable, PHASE2, DECL_PATH, DB_PATH],
                            capture_output=True, text=True)
    print(phase2.stdout)
    if phase2.returncode != 0:
        print(phase2.stderr, file=sys.stderr)
        raise SystemExit(phase2.returncode)


if __name__ == "__main__":
    main()
