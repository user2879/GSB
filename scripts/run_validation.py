"""Run the complete local acceptance suite and record reproducible artifacts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--validation-dir", type=Path, default=ROOT / "validation")
    parser.add_argument("--samples-dir", type=Path, default=ROOT / "samples")
    args = parser.parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.validation_dir.mkdir(parents=True, exist_ok=True)

    freeze = _run([sys.executable, "-m", "pip", "freeze"])
    _write_text(args.artifacts_dir / "pip-freeze.txt", freeze.stdout)

    environment = _run(
        [
            sys.executable,
            str(ROOT / "examples" / "record_environment.py"),
            "--samples-dir",
            str(args.samples_dir),
            "--output",
            str(args.artifacts_dir / "environment.json"),
        ]
    )
    if environment.returncode != 0:
        sys.stderr.write(environment.stderr)
        return environment.returncode
    environment_payload = json.loads((args.artifacts_dir / "environment.json").read_text())
    manifest = json.loads((args.samples_dir / "manifest.json").read_text(encoding="utf-8"))
    sample_lines = [
        f"{manifest['files'][name]['sha256']}  {manifest['files'][name]['path']}"
        for name in sorted(manifest["files"])
    ]
    _write_text(args.artifacts_dir / "samples-sha256.txt", "\n".join(sample_lines) + "\n")

    pytest = _run([sys.executable, "-m", "pytest", "-q"])
    _write_text(args.artifacts_dir / "pytest.txt", pytest.stdout + pytest.stderr)
    test_passed = pytest.returncode == 0

    demo_summaries: dict[str, Any] = {}
    injection_summaries: dict[str, Any] = {}
    exit_codes: dict[str, int] = {}
    expected_codes = {
        "zero-weight": "E_SAMPLE",
        "depth-exceeded": "E_DEPTH",
        "missing-version": "E_VERSION",
    }
    for injection in ("none", "zero-weight", "depth-exceeded", "missing-version"):
        runtime_dir = args.validation_dir / (
            "runtime" if injection == "none" else f"runtime-{injection}"
        )
        completed = _run(
            [
                sys.executable,
                "-m",
                "slowprof",
                "demo",
                "--samples-dir",
                str(args.samples_dir),
                "--runtime-dir",
                str(runtime_dir),
                "--inject",
                injection,
            ]
        )
        exit_codes[injection] = completed.returncode
        if completed.returncode not in {0, 2, 3, 4}:
            sys.stderr.write(completed.stdout + completed.stderr)
            return completed.returncode
        payload = json.loads(completed.stdout)
        if injection == "none":
            demo_summaries = payload
            _write_json(args.artifacts_dir / "demo-summary.json", payload)
        else:
            injection_summaries[injection] = payload

    sessions_ok = [item["aggregation_path"] for item in demo_summaries["sessions"]] == [
        "full",
        "streaming",
        "full",
    ]
    parity_ok = all(item["byte_equal"] for item in demo_summaries["path_parity"])
    replay_ok = all(item["status"] == "reproducible" for item in demo_summaries["replay_checks"])
    injection_ok = all(
        injection_summaries[name]["injected_anomaly"]["code"] == expected
        for name, expected in expected_codes.items()
    )
    restore = demo_summaries["restore_mismatch"]
    restore_ok = (
        restore["strategy"] == "replace"
        and restore["archived_snapshot_read_only"] is True
        and bool(restore["changed_frames"])
        and bool(restore["rank_moves"])
    )
    all_passed = bool(
        test_passed and sessions_ok and parity_ok and replay_ok and injection_ok and restore_ok
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_passed": all_passed,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "wsl": environment_payload["validation_host"]["wsl"],
        },
        "sample_manifest_sha256": environment_payload["sample_manifest_sha256"],
        "sample_set_digest": environment_payload["sample_set_digest"],
        "checks": {
            "pytest_passed": test_passed,
            "pytest_output": str(args.artifacts_dir / "pytest.txt"),
            "three_session_paths_ok": sessions_ok,
            "full_streaming_byte_equal": parity_ok,
            "all_replays_reproducible": replay_ok,
            "injections_ok": injection_ok,
            "whole_replacement_and_readonly_archive_ok": restore_ok,
        },
        "sessions": [
            {
                "session": item["session"],
                "aggregation_path": item["aggregation_path"],
                "selected_row_count": item["selected_row_count"],
                "total_weight": item["total_weight"],
                "total_time_ms": item["total_time_ms"],
                "top_frame": item["top_hotspots"][0]["frame"],
                "conclusion_sha256": item["conclusion_sha256"],
            }
            for item in demo_summaries["sessions"]
        ],
        "path_parity": demo_summaries["path_parity"],
        "replay_checks": demo_summaries["replay_checks"],
        "restore_mismatch": {
            "strategy": restore["strategy"],
            "mismatch_fields": [item["field"] for item in restore["mismatches"]],
            "changed_frames": [item["frame"] for item in restore["changed_frames"]],
            "rank_moves": restore["rank_moves"],
            "archived_snapshot_path": restore["archived_snapshot_path"],
            "archived_snapshot_read_only": restore["archived_snapshot_read_only"],
        },
        "injection_exit_codes": exit_codes,
        "injections": {
            name: {
                "code": payload["injected_anomaly"]["code"],
                "family": payload["injected_anomaly"].get("family"),
                "line": payload["injected_anomaly"].get("line"),
                "field": payload["injected_anomaly"].get("field"),
                "log_path": payload["log_path"],
            }
            for name, payload in injection_summaries.items()
        },
        "environment": str(args.artifacts_dir / "environment.json"),
        "pip_freeze": str(args.artifacts_dir / "pip-freeze.txt"),
    }
    _write_json(args.artifacts_dir / "validation-report.json", report)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
