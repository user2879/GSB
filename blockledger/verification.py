from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .clockutil import monotonic_ms


def _kill_process_group(proc: subprocess.Popen[bytes]) -> int | None:
    try:
        group_id = os.getpgid(proc.pid)
    except ProcessLookupError:
        group_id = None
    if group_id is not None:
        try:
            os.killpg(group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    try:
        _, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return -signal.SIGKILL


def verify_block_with_subprocess(
    *,
    block_file: str | Path,
    volume_id: str,
    block_index: int,
    baseline_version: int,
    expected_digest: str,
    timeout_ms: int = 1000,
    force_kill: bool = False,
    corrupt_in_memory: bool = False,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Run one block check in a real child process.

    A timeout is handled by SIGKILL against the child's process group.  A
    Python thread timeout is never reported as termination.
    """

    started_ms = monotonic_ms()
    command = [
        python_executable or sys.executable,
        "-m",
        "blockledger.worker",
        "--block-file",
        str(block_file),
        "--volume-id",
        volume_id,
        "--block-index",
        str(block_index),
        "--baseline-version",
        str(baseline_version),
        "--expected-digest",
        expected_digest,
    ]
    if force_kill:
        command.extend(["--mode", "slow", "--duration-ms", "30000"])
        effective_timeout = min(max(1, int(timeout_ms)), 500) / 1000.0
    else:
        effective_timeout = max(1, int(timeout_ms)) / 1000.0
    if corrupt_in_memory:
        command.append("--corrupt-in-memory")

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    result: dict[str, Any] = {
        "sample_selected": True,
        "worker_pid": proc.pid,
        "worker_returncode": None,
        "force_killed": False,
        "verification_status": "unknown",
        "expected_digest": expected_digest,
        "restored_digest": None,
        "wait_started_ms": started_ms,
        "wait_ended_ms": None,
    }
    try:
        stdout, stderr = proc.communicate(timeout=effective_timeout)
        result["worker_returncode"] = proc.returncode
        result["wait_ended_ms"] = monotonic_ms()
        text = stdout.decode("utf-8", errors="replace").strip().splitlines()
        payload: dict[str, Any] | None = None
        for line in reversed(text):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
            result.update(
                verification_status="unknown",
                reason="worker_without_json",
                stderr=stderr.decode("utf-8", errors="replace")[-500:],
            )
        elif payload.get("status") == "pass":
            result.update(
                verification_status="pass",
                restored_digest=payload.get("actual_digest"),
            )
        elif payload.get("status") == "corrupt":
            result.update(
                verification_status="corrupt",
                restored_digest=payload.get("actual_digest"),
                reason="digest_mismatch",
            )
        else:
            result.update(
                verification_status="unknown",
                reason=str(payload.get("error", "worker_error")),
                restored_digest=payload.get("actual_digest"),
            )
        return result
    except subprocess.TimeoutExpired:
        returncode = _kill_process_group(proc)
        result["worker_returncode"] = returncode
        result["force_killed"] = True
        result["verification_status"] = "unknown"
        result["reason"] = "worker_timed_out_and_sigkilled"
        result["wait_ended_ms"] = monotonic_ms()
        return result


def current_millis_for_test() -> int:
    return int(time.time() * 1000)
