from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from cachelease.engine import Engine
from cachelease.ledger import Ledger
from cachelease.projection import build_projection, compare_projections
from cachelease.spec import load_config, load_events

from .helpers import write_sample


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_hard_kill_marks_open_batch_unknown_recover_and_resume(tmp_path: Path) -> None:
    config_path, events_path, _, _ = write_sample(tmp_path, seed=914)
    ledger_path = tmp_path / "crash.sqlite"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    crashed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cachelease",
            "run",
            "--config",
            str(config_path),
            "--events",
            str(events_path),
            "--ledger",
            str(ledger_path),
            "--work-dir",
            str(tmp_path / "crash-nodes"),
            "--crash-at-batch",
            "2",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert crashed.returncode != 0

    config = load_config(config_path)
    events = load_events(events_path, config)
    ledger = Ledger(ledger_path, config)
    try:
        open_row = ledger.connection.execute(
            "SELECT status FROM batches WHERE batch_id=2"
        ).fetchone()
        assert open_row["status"] == "open"
        recovered = Engine(config, events, ledger, tmp_path / "recover-nodes").recover()
        assert recovered["recovery"]["open_batches_marked_unknown"][0]["batch_id"] == 2
        actions = ledger.recovery_actions()
        assert actions[0]["kind"] == "UNKNOWN_UNCOMMITTED_BATCH"
        resumed = Engine(config, events, ledger, tmp_path / "resume-nodes").run(resume=True)
        assert resumed["progress"]["last_ordinal"] == len(events)
    finally:
        ledger.close()

    fresh_ledger = Ledger(tmp_path / "fresh.sqlite", config)
    try:
        Engine(config, events, fresh_ledger, tmp_path / "fresh-nodes").run()
    finally:
        fresh_ledger.close()
    recovered_ledger = Ledger(ledger_path, config)
    fresh_reader = Ledger(tmp_path / "fresh.sqlite", config)
    try:
        compare_projections(
            build_projection(recovered_ledger, config),
            build_projection(fresh_reader, config),
            config,
        )
    finally:
        recovered_ledger.close()
        fresh_reader.close()
