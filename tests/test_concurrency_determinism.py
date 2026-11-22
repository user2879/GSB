"""The threaded path must match a single-worker run field by field."""

import copy

from dialog_state_pipeline.loader import load_event_stream
from dialog_state_pipeline.report import comparison_view
from dialog_state_pipeline.config import load_fsm
from dialog_state_pipeline.pipeline import Pipeline
from dialog_state_pipeline.config import load_config


def test_worker_count_does_not_change_results(sample_dir, tmp_path):
    reports = {}
    for workers, tag in ((1, "single"), (4, "multi")):
        config = load_config(sample_dir / "pipeline_config.json")
        config.max_workers = workers
        config.snapshot_dir = str(tmp_path / f"snapshot_{tag}")
        fsm = load_fsm(sample_dir / "fsm.yaml")
        pipeline = Pipeline(config, fsm, {"injections": []})
        reports[tag] = pipeline.run(sample_dir / "events.jsonl", None)
    assert comparison_view(reports["single"]) == comparison_view(reports["multi"])


def test_ledger_order_is_session_stable(sample_dir, fsm, config):
    from dialog_state_pipeline.scheduler import Scheduler
    materialised, _ = load_event_stream(sample_dir / "events.jsonl")
    cfg = copy.deepcopy(config)
    cfg.max_workers = 4
    scheduler = Scheduler(fsm, cfg, cache={}, budget_remaining=8)
    scheduler.run(materialised)
    consume_rows = [row for row in scheduler.ledger if row["kind"] == "consume"]
    # First round visits sessions in lexicographic order, one entry each.
    first_round = consume_rows[: len(materialised)]
    assert [row["session_id"] for row in first_round] == sorted(materialised)
    # Remaining budget never goes negative under concurrency.
    assert all(row["remaining_after"] >= 0 for row in consume_rows)

