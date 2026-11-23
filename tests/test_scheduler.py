from __future__ import annotations

from pathlib import Path

from walcrash.storage import Storage

from .conftest import SAMPLE, max_overlap, make_storage
from walcrash.engine import ReplayEngine


def test_global_slots_and_per_transaction_quota_are_enforced(tmp_path: Path) -> None:
    storage = make_storage(tmp_path)
    result = ReplayEngine(storage).replay(delay_ms=35)
    tasks = storage.tasks(result["run_id"])
    budget = storage.budget()
    assert len(tasks) == 11
    assert all(task["state"] == "committed" for task in tasks)
    assert max_overlap(tasks) <= budget.global_slots
    assert max_overlap(tasks) >= 2
    for txid in {task["txid"] for task in tasks}:
        assert max_overlap(tasks, txid=txid) <= budget.per_tx_quota

    waiting = [task for task in tasks if task["wait_started_ms"] is not None]
    assert waiting
    for task in waiting:
        assert task["wait_ended_ms"] is not None
        assert task["wait_ended_ms"] >= task["wait_started_ms"]
    for task in tasks:
        assert task["planned_bytes"] <= budget.page_byte_budget
        assert task["consumed_bytes"] == task["planned_bytes"]
        assert task["batch_id"] is not None
        assert task["started_ms"] is not None
        assert task["finished_ms"] is not None


def test_waiting_is_recorded_when_slots_are_exhausted(tmp_path: Path) -> None:
    sample = SAMPLE
    storage = Storage(tmp_path / "tight")
    from walcrash.validation import load_sample
    import json

    document = json.loads(sample.read_text(encoding="utf-8"))
    document["log_file"] = str((sample.parent / document["log_file"]).resolve())
    document["budget"] = {
        "global_slots": 1,
        "per_tx_quota": 1,
        "page_byte_budget": 512,
        "checkpoint_interval": 4,
    }
    path = tmp_path / "tight.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    storage.load_fresh(load_sample(path))
    result = ReplayEngine(storage).replay(delay_ms=8)
    tasks = storage.tasks(result["run_id"])
    assert len(tasks) > 1
    assert max_overlap(tasks) == 1
    assert sum(1 for task in tasks if task["wait_started_ms"] is not None) >= len(tasks) - 1

