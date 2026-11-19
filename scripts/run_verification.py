"""Run end-to-end verification and save raw outputs under artifacts/verification."""

from __future__ import annotations

import json
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts" / "verification"
WORK = OUT / "work"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def session_state(session_path: Path) -> tuple[Path, Path]:
    payload = read_json(session_path)
    base = session_path.parent
    return base / payload["state_dir"], base / payload["snapshot_dir"]


def run_cli(session_path: Path, output_name: str, *extra: str) -> tuple[int, dict, str]:
    process = subprocess.run(
        [sys.executable, "-m", "streamsim", "run", str(session_path), *extra],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = process.stdout.strip()
    (OUT / output_name).write_text(output + "\n", encoding="utf-8")
    payload = json.loads(output) if output else {}
    return process.returncode, payload, process.stderr


def run_verify(session_path: Path, output_name: str) -> tuple[int, dict]:
    process = subprocess.run(
        [sys.executable, "-m", "streamsim", "verify-snapshots", str(session_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = process.stdout.strip()
    (OUT / output_name).write_text(output + "\n", encoding="utf-8")
    return process.returncode, json.loads(output) if output else {}


def journal_for(session_path: Path) -> list[dict]:
    state_dir, _ = session_state(session_path)
    return [
        json.loads(line)
        for line in (state_dir / "simulation.log.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def make_variant_session(source: Path, target: Path, state: str, snapshots: str) -> Path:
    payload = read_json(source)
    payload["state_dir"] = state
    payload["snapshot_dir"] = snapshots
    write_json(target, payload)
    return target


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for generated in OUT.glob("[0-9][0-9]_*.json"):
        generated.unlink()
    for generated in (OUT / "verification_summary.json",):
        if generated.exists():
            generated.unlink()
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    from streamsim.samples import generate_samples

    generate_samples(WORK, seed=20261119)
    baseline_session = WORK / "session.json"
    concurrency1_session = WORK / "session_concurrency1.json"
    interrupt_session = WORK / "session_interrupt.json"

    code_base, baseline, stderr_base = run_cli(baseline_session, "01_continuous.report.json")
    if code_base != 0:
        raise RuntimeError(f"baseline failed: {stderr_base}")
    baseline_log_equal = journal_for(baseline_session) == baseline["events"]

    code_c1, concurrency1, stderr_c1 = run_cli(
        concurrency1_session, "02_concurrency1.report.json"
    )
    if code_c1 != 0:
        raise RuntimeError(f"concurrency1 failed: {stderr_c1}")

    code_int, interrupt, stderr_int = run_cli(
        interrupt_session, "03_scripted_interrupt.exit.json"
    )
    if code_int != 75:
        raise RuntimeError(f"scripted interrupt did not exit 75: {stderr_int}")
    code_resume, resumed, stderr_resume = run_cli(
        interrupt_session, "04_scripted_resume.report.json"
    )
    if code_resume != 0:
        raise RuntimeError(f"scripted resume failed: {stderr_resume}")
    resume_log_equal = journal_for(interrupt_session) == resumed["events"]

    tamper_session = make_variant_session(
        interrupt_session,
        WORK / "session_tamper.json",
        "sample_runs/tamper/state",
        "sample_runs/tamper/snapshots",
    )
    code_tamper_interrupt, _, stderr_tamper = run_cli(
        tamper_session, "05_tamper_interrupt.exit.json"
    )
    if code_tamper_interrupt != 75:
        raise RuntimeError(f"tamper setup interrupt failed: {stderr_tamper}")
    tamper_state, _ = session_state(tamper_session)
    current_path = tamper_state / "snapshot.json"
    current = read_json(current_path)
    current["state"]["buffer_ms"] = int(current["state"]["buffer_ms"]) + 13
    write_json(current_path, current)
    verify_code, verify_payload = run_verify(tamper_session, "06_tamper_verify.json")
    code_tamper_resume, tamper_resumed, stderr_tamper_resume = run_cli(
        tamper_session, "07_tamper_fallback.report.json"
    )
    if code_tamper_resume != 0:
        raise RuntimeError(f"tamper fallback failed: {stderr_tamper_resume}")

    sigterm_session = make_variant_session(
        baseline_session,
        WORK / "session_sigterm.json",
        "sample_runs/sigterm/state",
        "sample_runs/sigterm/snapshots",
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamsim",
            "run",
            str(sigterm_session),
            "--event-delay-ms",
            "15",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sigterm_state, _ = session_state(sigterm_session)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not (sigterm_state / "run_meta.json").exists():
        if process.poll() is not None:
            break
        time.sleep(0.01)
    process.send_signal(signal.SIGTERM)
    sigterm_stdout, sigterm_stderr = process.communicate(timeout=20)
    (OUT / "08_sigterm.exit.json").write_text(sigterm_stdout.strip() + "\n", encoding="utf-8")
    code_sigterm_resume, sigterm_resumed, stderr_sigterm_resume = run_cli(
        sigterm_session, "09_sigterm_resume.report.json"
    )
    if code_sigterm_resume != 0:
        raise RuntimeError(f"SIGTERM resume failed: {stderr_sigterm_resume}")

    version_session = make_variant_session(
        baseline_session,
        WORK / "session_version2.json",
        "sample_runs/version/state",
        "sample_runs/version/snapshots",
    )
    code_version_old, version_old_report, stderr_version_old = run_cli(
        version_session, "10_version_old.report.json"
    )
    if code_version_old != 0:
        raise RuntimeError(f"version baseline failed: {stderr_version_old}")
    version_payload = read_json(version_session)
    version_payload["client"]["max_buffer_ms"] = 11_000
    write_json(version_session, version_payload)
    code_version, version_report, stderr_version = run_cli(
        version_session, "11_version_reset.report.json"
    )
    if code_version != 0:
        raise RuntimeError(f"version reset failed: {stderr_version}")
    _, version_snapshots = session_state(version_session)
    stale_files = sorted((version_snapshots / "stale" / "version_changed").glob("*"))

    summary = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "checks": {
            "continuous_finished": baseline["summary"]["finished"],
            "continuous_decisions": baseline["summary"]["decision_count"],
            "continuous_switch_count": baseline["summary"]["switch_count"],
            "continuous_drain_count": baseline["summary"]["drain_count"],
            "continuous_log_matches_report": baseline_log_equal,
            "concurrency_invariance": concurrency1["events"] == baseline["events"],
            "concurrency_peak": {
                "c1": concurrency1["prefetch"]["peak_active_tasks"],
                "c2": baseline["prefetch"]["peak_active_tasks"],
            },
            "scripted_interrupt_exit": code_int,
            "scripted_resume_mode": resumed["recovery_start"]["mode"],
            "scripted_resume_exact": resumed["events"] == baseline["events"],
            "scripted_resume_log_matches_report": resume_log_equal,
            "tamper_verify_exit": verify_code,
            "tamper_verify_code": verify_payload.get("error", {}).get("code"),
            "tamper_resume_mode": tamper_resumed["recovery_start"]["mode"],
            "tamper_fallback_exact": tamper_resumed["events"] == baseline["events"],
            "tamper_invalid_count": len(
                tamper_resumed["recovery_start"]["invalid_snapshots"]
            ),
            "sigterm_exit": process.returncode,
            "sigterm_resume_exact": sigterm_resumed["events"] == baseline["events"],
            "sigterm_marker_observed": sigterm_resumed["recovery_start"][
                "had_interruption_marker"
            ],
            "version_change_mode": version_report["recovery_start"]["mode"],
            "version_change_stale_files": len(stale_files),
            "version_ids_differ": version_report["version_id"] != version_old_report["version_id"],
        },
        "outputs": {
            "baseline_report": "01_continuous.report.json",
            "concurrency1_report": "02_concurrency1.report.json",
            "scripted_interrupt": "03_scripted_interrupt.exit.json",
            "scripted_resume": "04_scripted_resume.report.json",
            "tamper_verify": "06_tamper_verify.json",
            "tamper_fallback": "07_tamper_fallback.report.json",
            "sigterm": "08_sigterm.exit.json",
            "sigterm_resume": "09_sigterm_resume.report.json",
            "version_old": "10_version_old.report.json",
            "version_reset": "11_version_reset.report.json",
        },
    }
    write_json(OUT / "verification_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
