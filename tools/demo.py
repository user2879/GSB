from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "artifacts" / "demo"
CERTS = ROOT / "samples" / "certificates.json"
MODIFIED_CERTS = ROOT / "samples" / "certificates_modified.json"
HANDSHAKE = ROOT / "samples" / "handshake.json"


def environment() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not existing else str(ROOT) + os.pathsep + existing
    return env


def command(*args: str) -> list[str]:
    return [sys.executable, "-m", "tlschain.cli", *args]


def free_port() -> int:
    for port in range(23771, 23999):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            probe.close()
    raise RuntimeError("no free port in 23771-23998")


def run(args: list[str], expected: int = 0, timeout: float = 40.0) -> dict[str, Any]:
    result = subprocess.run(
        args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment(),
    )
    if expected >= 0 and result.returncode != expected:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1]) if lines else None
    return {
        "returncode": result.returncode,
        "stdout_tail": payload,
        "stderr": result.stderr.strip(),
    }


def main() -> int:
    if DEMO.exists():
        shutil.rmtree(DEMO)
    DEMO.mkdir(parents=True)
    run([sys.executable, str(ROOT / "tools" / "generate_samples.py")])
    validation = run(command("validate", "--certs", str(CERTS), "--handshake", str(HANDSHAKE)))
    incremental = run(
        command(
            "incremental",
            "--base-certs",
            str(CERTS),
            "--base-handshake",
            str(HANDSHAKE),
            "--changed-certs",
            str(MODIFIED_CERTS),
            "--changed-handshake",
            str(HANDSHAKE),
        )
    )
    full_dir = DEMO / "full"
    full_dir.mkdir()
    full = run(
        command(
            "run",
            "--certs",
            str(CERTS),
            "--handshake",
            str(HANDSHAKE),
            "--ledger",
            str(full_dir / "ledger.sqlite"),
            "--port",
            str(free_port()),
            "--reset",
        )
    )
    full_recovery = run(
        command(
            "recover",
            "--certs",
            str(CERTS),
            "--handshake",
            str(HANDSHAKE),
            "--ledger",
            str(full_dir / "ledger.sqlite"),
            "--port",
            str(free_port()),
        )
    )
    graceful_dir = DEMO / "graceful"
    graceful_dir.mkdir()
    graceful = run(
        command(
            "run",
            "--certs",
            str(CERTS),
            "--handshake",
            str(HANDSHAKE),
            "--ledger",
            str(graceful_dir / "ledger.sqlite"),
            "--port",
            str(free_port()),
            "--reset",
            "--terminate-after",
            "4",
            "--termination",
            "graceful",
        )
    )
    graceful_recovery = run(
        command(
            "recover",
            "--certs",
            str(CERTS),
            "--handshake",
            str(HANDSHAKE),
            "--ledger",
            str(graceful_dir / "ledger.sqlite"),
            "--port",
            str(free_port()),
        )
    )
    hard_dir = DEMO / "hard"
    hard_dir.mkdir()
    hard = run(
        command(
            "run",
            "--certs",
            str(CERTS),
            "--handshake",
            str(HANDSHAKE),
            "--ledger",
            str(hard_dir / "ledger.sqlite"),
            "--port",
            str(free_port()),
            "--reset",
            "--terminate-after",
            "4",
            "--termination",
            "hard",
        ),
        expected=-1,
    )
    if hard["returncode"] == 0:
        raise RuntimeError("hard termination unexpectedly returned success")
    hard_exit_report_before_recovery = (hard_dir / "exit-report.json").exists()
    hard_run_report_before_recovery = (hard_dir / "run-report.json").exists()
    hard_recovery = run(
        command(
            "recover",
            "--certs",
            str(CERTS),
            "--handshake",
            str(HANDSHAKE),
            "--ledger",
            str(hard_dir / "ledger.sqlite"),
            "--port",
            str(free_port()),
        )
    )
    diagnostics_args = command(
        "diagnostics",
        "--certs",
        str(CERTS),
        "--handshake",
        str(HANDSHAKE),
        "--ledger",
        str(full_dir / "ledger.sqlite"),
    )
    diagnostics_one = run(diagnostics_args)
    diagnostics_two = run(diagnostics_args)
    summary = {
        "ok": True,
        "python": sys.version,
        "validation": validation["stdout_tail"],
        "incremental": {
            "ok": incremental["stdout_tail"]["ok"],
            "invalidated_certificate_ids": incremental["stdout_tail"]["invalidated_certificate_ids"],
            "invalidated_message_seqs": incremental["stdout_tail"]["invalidated_message_seqs"],
            "retained_evidence": incremental["stdout_tail"]["retained_evidence"],
            "comparison_equal": incremental["stdout_tail"]["comparison"]["equal"],
        },
        "full_run": {
            "status": full["stdout_tail"]["status"],
            "progress_cursor": full["stdout_tail"]["progress_cursor"],
            "decision_counts": full["stdout_tail"]["decision_counts"],
            "batches": [batch["through_seq"] for batch in full["stdout_tail"]["committed_batches"]],
        },
        "full_recovery": {
            "status": full_recovery["stdout_tail"]["status"],
            "replayed_message_seqs": full_recovery["stdout_tail"]["replayed_message_seqs"],
        },
        "graceful": {
            "status": graceful["stdout_tail"]["status"],
            "signal": graceful["stdout_tail"]["signal"],
            "progress_cursor": graceful["stdout_tail"]["progress_cursor"],
            "remaining_message_seqs": graceful["stdout_tail"]["remaining_message_seqs"],
            "exit_report": (graceful_dir / "exit-report.json").exists(),
        },
        "graceful_recovery": {
            "status": graceful_recovery["stdout_tail"]["status"],
            "initial_unknown_seqs": graceful_recovery["stdout_tail"]["initial_unknown_seqs"],
            "progress_cursor": graceful_recovery["stdout_tail"]["progress_cursor"],
        },
        "hard": {
            "returncode": hard["returncode"],
            "exit_report": hard_exit_report_before_recovery,
            "run_report": hard_run_report_before_recovery,
        },
        "hard_recovery": {
            "status": hard_recovery["stdout_tail"]["status"],
            "initial_unknown_seqs": hard_recovery["stdout_tail"]["initial_unknown_seqs"],
            "progress_cursor": hard_recovery["stdout_tail"]["progress_cursor"],
        },
        "diagnostics_refresh_byte_identical": diagnostics_one["stdout_tail"] == diagnostics_two["stdout_tail"],
        "diagnostics_model_digest": diagnostics_one["stdout_tail"]["model_digest"],
    }
    summary_path = DEMO / "demo-summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
