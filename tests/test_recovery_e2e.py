"""End-to-end interruption recovery with a tight, refillable budget."""

from dialog_state_pipeline.config import load_config, load_fsm
from dialog_state_pipeline.pipeline import Pipeline
from dialog_state_pipeline.report import comparison_view
from dialog_state_pipeline.snapshot import SnapshotStore


def test_interrupted_run_resumes_and_converges(sample_dir, tmp_path):
    config = load_config(sample_dir / "budget_config.json")
    config.snapshot_dir = str(tmp_path / "snap")
    fsm = load_fsm(sample_dir / "fsm.yaml")
    events = sample_dir / "events.jsonl"
    pipeline = Pipeline(config, fsm, {"injections": []},
                        budget_refill_on_resume=True)

    # Interrupt immediately after the advance stage commit.
    pipeline.run(events, None, stop_after_stage="advance")
    store = SnapshotStore(config.snapshot_dir)
    assert store.load().stage == "advance"

    report = pipeline.run(events, None)
    guard = 0
    while report["budget"]["pending_sessions"] and guard < 20:
        report = pipeline.run(events, None)
        guard += 1

    assert not report["budget"]["pending_sessions"]
    assert all(s["status"] == "complete"
               for s in report["sessions"].values())
    assert report["completed_batches"] == 5

    # A single full run with ample budget reaches the same business state.
    ample = load_config(sample_dir / "pipeline_config.json")
    ample.snapshot_dir = str(tmp_path / "ample")
    full = Pipeline(ample, fsm, {"injections": []}).run(events, None, mode="full")
    for sid, session in comparison_view(report)["sessions"].items():
        counterpart = comparison_view(full)["sessions"][sid]
        assert session["current_state"] == counterpart["current_state"]
        assert session["slots"] == counterpart["slots"]
        assert session["transition_trace"] == counterpart["transition_trace"]
        assert session["rejected"] == counterpart["rejected"]

