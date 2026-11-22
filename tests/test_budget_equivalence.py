"""Budget-resumed incremental runs must match a single full replay."""

from dialog_state_pipeline.config import load_config, load_fsm
from dialog_state_pipeline.pipeline import Pipeline
from dialog_state_pipeline.report import comparison_view


def test_resumed_budget_run_matches_full_replay(sample_dir, tmp_path):
    tight = load_config(sample_dir / "budget_config.json")
    tight.snapshot_dir = str(tmp_path / "tight")
    fsm = load_fsm(sample_dir / "fsm.yaml")
    pipeline = Pipeline(tight, fsm, {"injections": []},
                        budget_refill_on_resume=True)

    report = pipeline.run(sample_dir / "events.jsonl", None)
    guard = 0
    while report["budget"]["pending_sessions"] and guard < 20:
        report = pipeline.run(sample_dir / "events.jsonl", None)
        guard += 1
    assert not report["budget"]["pending_sessions"]

    # The full replay is given enough budget to finish in one shot; the
    # business-level state (slots, FSM state, traces, failures other than the
    # scheduler's pending markers) must still match the resumed run.
    full_config = load_config(sample_dir / "pipeline_config.json")
    full_config.snapshot_dir = str(tmp_path / "full")
    full = Pipeline(full_config, fsm, {"injections": []}).run(
        sample_dir / "events.jsonl", None, mode="full")
    view_a = comparison_view(report)
    view_b = comparison_view(full)
    for sid in view_a["sessions"]:
        sa, sb = view_a["sessions"][sid], view_b["sessions"][sid]
        assert sa["current_state"] == sb["current_state"]
        assert sa["slots"] == sb["slots"]
        assert sa["transition_trace"] == sb["transition_trace"]
        assert sa["rejected"] == sb["rejected"]
        assert sa["consumed_entries"] == sb["consumed_entries"]
    business_codes = {"E_BUDGET_PENDING"}
    assert [f for f in view_a["failures"] if f["code"] not in business_codes] == \
           [f for f in view_b["failures"] if f["code"] not in business_codes]
