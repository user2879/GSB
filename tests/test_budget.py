from __future__ import annotations

import copy
from pathlib import Path

from ancient_vlayout.engine import RecognitionEngine
from ancient_vlayout.util import atomic_write_json, read_json


def _low_budget_samples(samples_copy: Path) -> Path:
    config = read_json(samples_copy / "config.json")
    config["ma_point_budget"] = 26000
    atomic_write_json(samples_copy / "config.json", config)
    return samples_copy


def test_budget_pending_priority_ledger_and_settle_equivalence(
    samples_copy: Path, tmp_path: Path
) -> None:
    samples = _low_budget_samples(samples_copy)
    state = tmp_path / "state"
    first = RecognitionEngine(samples, state_dir=state).run()
    assert first.pending
    assert len(first.ledger_events) == 3
    assert [event["sequence"] for event in first.ledger_events] == [1, 2, 3]
    pending_chars = [
        char
        for image in first.recognition["images"]
        for column in image["columns"]
        for char in column["chars"]
        if char["status"] == "pending"
    ]
    assert pending_chars
    assert all(char["errors"] == ["E_MA_BUDGET_PENDING"] for char in pending_chars)

    next_pending_key = first.pending[0]["stable_key"]
    second = RecognitionEngine(samples, state_dir=state).run()
    assert second.ledger_events[0]["stable_key"] == next_pending_key
    assert second.report["budget"]["cumulative_event_count"] == 6

    settled = RecognitionEngine(samples, state_dir=state).settle(max_batches=100)
    assert settled.pending == []
    assert settled.report["budget"]["cumulative_event_count"] == 80
    fresh = RecognitionEngine(samples, state_dir=tmp_path / "fresh").settle()
    assert settled.recognition == fresh.recognition


def test_repeated_budget_sequence_is_identical(samples_copy: Path, tmp_path: Path) -> None:
    samples = _low_budget_samples(samples_copy)
    left_state = tmp_path / "left"
    left1 = RecognitionEngine(samples, state_dir=left_state).run()
    left2 = RecognitionEngine(samples, state_dir=left_state).run()

    right_samples = tmp_path / "right-samples"
    right_samples.mkdir()
    for path in samples.iterdir():
        if path.is_file():
            (right_samples / path.name).write_bytes(path.read_bytes())
    right_state = tmp_path / "right"
    right1 = RecognitionEngine(right_samples, state_dir=right_state).run()
    right2 = RecognitionEngine(right_samples, state_dir=right_state).run()
    assert left1.ledger_events == right1.ledger_events
    assert left2.ledger_events == right2.ledger_events
    assert left1.recognition == right1.recognition
