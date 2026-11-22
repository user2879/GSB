"""Requirement 6: global budget + per-session quota concurrent scheduling."""

import copy

from dialog_state_pipeline.config import load_config
from dialog_state_pipeline.injection import load_injection
from dialog_state_pipeline.pipeline import Pipeline
from dialog_state_pipeline.state_machine import build_transition_index
from dialog_state_pipeline.scheduler import Scheduler
from dialog_state_pipeline.loader import load_event_stream


def _pipeline(sample_dir, tmp_path, config_name, snapshot_name):
    config = load_config(sample_dir / config_name)
    config.snapshot_dir = str(tmp_path / snapshot_name)
    from dialog_state_pipeline.config import load_fsm
    fsm = load_fsm(sample_dir / "fsm.yaml")
    return Pipeline(config, fsm, {"injections": []},
                    budget_refill_on_resume=True), config


def test_budget_exhaustion_retains_todos_and_resumes(sample_dir, tmp_path):
    pipeline, _ = _pipeline(sample_dir, tmp_path, "budget_config.json", "snap")
    pipeline.store.reset()
    materialised, failures = load_event_stream(sample_dir / "events.jsonl")
    assert failures == []

    first = pipeline.run(sample_dir / "events.jsonl", None)
    # Tight budget (10) cannot cover all 32 state entries.
    assert first["budget"]["consumed"] == 10
    assert first["budget"]["remaining"] == 0
    assert first["budget"]["pending_sessions"], "expected retained todos"
    pending_codes = [f["code"] for f in first["failures"]
                     if f["code"] == "E_BUDGET_PENDING"]
    assert pending_codes

    # Resume: snapshot restores the consumed budget and continues.
    second = pipeline.run(sample_dir / "events.jsonl", None)
    assert second["budget"]["consumed"] > first["budget"]["consumed"]
    # Ledger is an auditable, gap-free deduction sequence.
    seqs = [row["seq"] for row in second["budget"]["deduction_sequence"]]
    assert seqs == list(range(1, len(seqs) + 1))
    consumes = [row for row in second["budget"]["deduction_sequence"]
                if row["kind"] == "consume"]
    assert len(consumes) == second["budget"]["consumed"]

    # Repeated resumes eventually finish every session.
    report = second
    guard = 0
    while report["budget"]["pending_sessions"] and guard < 20:
        report = pipeline.run(sample_dir / "events.jsonl", None)
        guard += 1
    assert not report["budget"]["pending_sessions"]
    assert all(s["status"] == "complete" for s in report["sessions"].values())


def test_quota_exceeded_aborts_one_session_only(sample_dir, fsm, config):
    materialised, _ = load_event_stream(sample_dir / "events.jsonl")
    cfg = copy.deepcopy(config)
    cfg.session_quotas = {"S001": 3}
    scheduler = Scheduler(fsm, cfg, cache={}, budget_remaining=1000)
    results = scheduler.run(materialised)
    assert results["S001"].status == "quota_exceeded"
    assert results["S001"].consumed_entries == 3
    assert any(f["code"] == "E_QUOTA_EXCEEDED" for f in results["S001"].failures)
    # Other sessions are untouched by the per-session quota.
    assert results["S002"].status == "complete"


def test_stable_session_ordering(sample_dir, fsm, config):
    materialised, _ = load_event_stream(sample_dir / "events.jsonl")
    cfg = copy.deepcopy(config)
    cfg.global_budget = 5
    scheduler = Scheduler(fsm, cfg, cache={}, budget_remaining=5)
    scheduler.run(materialised)
    consumed_sessions = [row["session_id"] for row in scheduler.ledger
                         if row["kind"] == "consume"]
    # First allocations follow lexicographic session order.
    assert consumed_sessions[:2] == sorted(consumed_sessions[:2])
