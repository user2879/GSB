from __future__ import annotations

import json
from pathlib import Path

from walcrash.engine import ReplayEngine
from walcrash.storage import Storage
from walcrash.verification import verify_replay

from .conftest import SAMPLE, run_cli


def test_forced_termination_marks_only_incomplete_tasks_unknown(tmp_path: Path) -> None:
    workspace = tmp_path / "crash-workspace"
    load = run_cli(
        "--workspace",
        str(workspace),
        "--action",
        "load",
        "--input",
        str(SAMPLE),
        expect=0,
    )
    assert json.loads(load.stdout)["wal_entries"] == 12
    crashed = run_cli(
        "--workspace",
        str(workspace),
        "--action",
        "run",
        "--delay-ms",
        "95",
        "--crash-after-batches",
        "1",
        expect=91,
    )
    assert crashed.stdout == ""

    storage = Storage(workspace)
    status = storage.status()
    assert status["cursor"] == 4
    assert status["counts"]["batches"] == 1
    assert status["last_run"]["status"] == "running"
    assert status["counts"]["active_tasks"] > 0
    assert {row["ledger_cursor"] for row in storage.checkpoint_rows()} == {4}
    initial_states = {task["state"] for task in storage.tasks()}
    assert initial_states & {"queued", "running", "partial", "staged"}

    resumed = ReplayEngine(storage).replay(delay_ms=0)
    assert resumed["cursor_start"] == 4
    assert resumed["cursor_end"] == 12
    assert resumed["recovered_orphan_tasks"] > 0
    states = {task["state"] for task in storage.tasks()}
    assert states == {"committed", "unknown"}
    assert storage.status()["counts"]["unknown_tasks"] > 0
    assert storage.status()["counts"]["page_versions"] == 12
    assert verify_replay(storage)["ok"]
    assert storage.verify_ledger()["ok"]
    assert any(row["event_type"] == "RECOVERY_INCOMPLETE" for row in storage.ledger_rows())


def test_resume_does_not_replay_committed_versions(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "resume")
    from walcrash.validation import load_sample

    storage.load_fresh(load_sample(SAMPLE))
    first = ReplayEngine(storage).replay()
    assert first["batch_count"] == 3
    before = {(row["page_id"], row["version"]): row["digest"] for row in storage.page_versions()}
    second = ReplayEngine(storage).replay()
    after = {(row["page_id"], row["version"]): row["digest"] for row in storage.page_versions()}
    assert second["replayed_pages"] == 0
    assert second["batches"] == []
    assert before == after
    assert len(storage.batches()) == 3

