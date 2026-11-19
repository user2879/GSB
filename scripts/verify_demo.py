"""Run the deterministic demo twice and compare every output byte-for-byte."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FILES = (
    "index.json",
    "summary.json",
    "hash_manifest.json",
    "invalidation_ledger.json",
    "recovery_report.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    runs = (
        (ROOT / "artifacts" / "verify_run1_workspace", ROOT / "artifacts" / "verify_run1"),
        (ROOT / "artifacts" / "verify_run2_workspace", ROOT / "artifacts" / "verify_run2"),
    )
    for workspace, output in runs:
        for path in (workspace, output):
            if path.exists():
                shutil.rmtree(path)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "diffrender.demo",
                "--workspace",
                str(workspace),
                "--out",
                str(output),
            ],
            cwd=str(ROOT),
            check=True,
        )
    report = {
        "equal": True,
        "files": {},
        "run1": {},
        "run2": {},
    }
    for name in FILES:
        left = runs[0][1] / name
        right = runs[1][1] / name
        same = left.read_bytes() == right.read_bytes()
        left_hash = digest(left)
        report["files"][name] = {"equal": same, "sha256": left_hash}
        report["run1"][name] = left_hash
        report["run2"][name] = digest(right)
        report["equal"] = report["equal"] and same
    for workspace, _output in runs:
        shutil.rmtree(workspace, ignore_errors=True)
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    (ROOT / "artifacts" / "verification.json").write_bytes(payload)
    print(payload.decode("utf-8"), end="")
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
