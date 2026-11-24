from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    environment = run(
        [
            sys.executable,
            "-c",
            (
                "import platform, sys, numpy; "
                "print('python', platform.python_version()); "
                "print('numpy', numpy.__version__); "
                "print('executable', sys.executable)"
            ),
        ]
    )
    tests = run([sys.executable, "-m", "unittest", "discover", "-v"])
    demo = run(
        [
            sys.executable,
            "-m",
            "config_arbiter",
            "demo",
            "--ledger",
            str(ARTIFACTS / "demo-ledger.sqlite3"),
            "--report",
            str(ARTIFACTS / "acceptance_report.json"),
        ]
    )
    status = (
        "ACCEPTANCE OK"
        if not (environment.returncode or tests.returncode or demo.returncode)
        else "ACCEPTANCE FAILED"
    )
    combined = (
        "=== environment ===\n"
        + environment.stdout
        + "\n=== unittest ===\n"
        + tests.stdout
        + "\n=== acceptance demo ===\n"
        + demo.stdout
        + "\n=== result ===\n"
        + status
        + "\n"
    )
    (ARTIFACTS / "acceptance-output.txt").write_text(combined, encoding="utf-8")
    (ARTIFACTS / "test-results.txt").write_text(tests.stdout, encoding="utf-8")
    print(combined)
    if environment.returncode or tests.returncode or demo.returncode:
        print(
            f"acceptance failed: env={environment.returncode} "
            f"tests={tests.returncode} demo={demo.returncode}",
            file=sys.stderr,
        )
        return 1
    print(status)
    return 0 if status == "ACCEPTANCE OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
