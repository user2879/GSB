"""End-to-end local demo: samples -> aggregate -> recover -> replay.

Run from the repository root with:  python examples/run_demo.py
No network access or external service is required.
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obsflow import (
    ObsFlowEngine,
    generate_samples,
)
from obsflow.diagnostics import build_window_rows, render_text_table
from obsflow.process_runner import launch_crashed_worker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "samples", "demo")
WORK_DIR = os.path.join(ROOT, "artifacts", "demo")


def main() -> int:
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    os.makedirs(WORK_DIR, exist_ok=True)
    generated = generate_samples(SAMPLE_DIR)
    ledger_path = os.path.join(WORK_DIR, "obsflow.db")

    engine = ObsFlowEngine.from_files(
        generated["declaration_path"], generated["window_path"], ledger_path)
    engine.read_records()
    report = engine.advance()
    print(render_text_table(build_window_rows(
        engine.query_windows(), engine.lease_manager)))
    print(f"first run: windows={len(report.windows)} "
          f"recomputed={len(report.recomputed)}")
    engine.close()

    crashed = launch_crashed_worker(ledger_path, kill_after_ms=200)
    print(f"force-killed worker returncode={crashed.returncode}")

    recovered = ObsFlowEngine.from_files(
        generated["declaration_path"], generated["window_path"], ledger_path)
    windows, unknown = recovered.recover()
    print(f"recovered={len(windows)} unknown={len(unknown)}")
    recovered.read_records()
    recovered.advance()
    replayed = recovered.replay_verify()
    print(f"rerun + replay verified {len(replayed)} windows field by field")
    recovered.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
