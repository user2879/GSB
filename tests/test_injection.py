"""Requirement 8: fault injection isolation and diagnostic report content."""

import copy

from dialog_state_pipeline.config import load_config, load_fsm
from dialog_state_pipeline.injection import load_injection
from dialog_state_pipeline.pipeline import Pipeline
from dialog_state_pipeline.report import comparison_view
from dialog_state_pipeline.loader import load_event_stream


def _pipeline(sample_dir, tmp_path, injection=True):
    config = load_config(sample_dir / "pipeline_config.json")
    config.snapshot_dir = str(tmp_path / "snapshot_inj")
    fsm = load_fsm(sample_dir / "fsm.yaml")
    injections = (load_injection(sample_dir / "injection_config.json")
                  if injection else {"injections": []})
    return Pipeline(config, fsm, injections, budget_refill_on_resume=True)


def test_all_four_fault_kinds_are_hit_and_isolated(sample_dir, tmp_path):
    pipeline = _pipeline(sample_dir, tmp_path)
    report = pipeline.run(sample_dir / "events.jsonl", None)

    hits = report["injections"]["hits"]
    # Every requested injection id is stamped on at least one failure.
    expected = set(report["injections"]["requested"])
    assert set(hits) == expected
    codes = {inj_id: [h["code"] for h in rows]
             for inj_id, rows in hits.items()}
    assert "E_SLOT_INVALID" in codes["INJ-BAD-SLOT-01"]
    assert "E_STREAM_INVALID" in codes["INJ-TURN-JUMP-01"]
    assert "E_TRANSITION_REJECTED" in codes["INJ-UNKNOWN-TX-01"]
    assert "E_QUOTA_EXCEEDED" in codes["INJ-QUOTA-01"]

    # Isolation: other sessions still reach their normal terminal state.
    assert report["sessions"]["S003"]["current_state"] == "confirming"
    # The injected bad slot did not corrupt the slot value.
    comfort = report["sessions"]["S003"]["slots"]["comfort"]
    assert isinstance(comfort["value"], float)


def test_report_contains_required_diagnostic_sections(sample_dir, tmp_path):
    pipeline = _pipeline(sample_dir, tmp_path)
    report = pipeline.run(sample_dir / "events.jsonl", None)
    for sid, session in report["sessions"].items():
        assert {"status", "current_state", "slots", "transition_count",
                "transition_trace", "rejected", "reused_turns",
                "replayed_turns", "consumed_entries"} <= set(session)
    assert {"global_budget", "consumed", "remaining", "pending_sessions",
            "deduction_sequence"} <= set(report["budget"])
    assert isinstance(report["failures"], list)
    assert report["report_fingerprint"]


def test_injected_run_is_still_equivalent_to_full_replay(sample_dir, tmp_path):
    pipeline = _pipeline(sample_dir, tmp_path)
    incremental = pipeline.run(sample_dir / "events.jsonl", None)

    config = load_config(sample_dir / "pipeline_config.json")
    config.snapshot_dir = str(tmp_path / "fresh")
    fsm = load_fsm(sample_dir / "fsm.yaml")
    injections = load_injection(sample_dir / "injection_config.json")
    fresh = Pipeline(config, fsm, injections).run(
        sample_dir / "events.jsonl", None, mode="full")
    assert comparison_view(incremental) == comparison_view(fresh)


def test_bad_event_does_not_stop_sibling_events(sample_dir, tmp_path):
    pipeline = _pipeline(sample_dir, tmp_path)
    report = pipeline.run(sample_dir / "events.jsonl", None)
    # S004 survives a turn jump and still processes the remaining dense turns.
    session = report["sessions"]["S004"]
    assert session["transition_count"] >= 1
    assert session["status"] == "complete"


def test_injection_resume_does_not_falsely_flag_stale(sample_dir, tmp_path):
    config = load_config(sample_dir / "budget_config.json")
    config.snapshot_dir = str(tmp_path / "snap_resume")
    fsm = load_fsm(sample_dir / "fsm.yaml")
    injections = load_injection(sample_dir / "injection_config.json")
    pipeline = Pipeline(config, fsm, injections, budget_refill_on_resume=True)
    events = sample_dir / "events.jsonl"
    first = pipeline.run(events, None)
    assert "snapshot_event" not in first
    second = pipeline.run(events, None)
    # Quota overrides must not change the declared config fingerprint.
    assert "snapshot_event" not in second
    assert second["sessions"]["S001"]["status"] == "quota_exceeded"
    # The injected quota failure keeps its injection id on resume.
    quota_hits = second["injections"]["hits"]["INJ-QUOTA-01"]
    assert any(h["code"] == "E_QUOTA_EXCEEDED" for h in quota_hits)
