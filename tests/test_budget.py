"""Requirement 5: latency budget returns pending state and still converges."""

from asr_stream_editor.engine import StreamingEngine
from asr_stream_editor.errors import E_REDECODE_BUDGET_PENDING
from asr_stream_editor.runner import load_resources
from asr_stream_editor.snapshot import SnapshotStore
import os


def _engine(samples_dir, work_dir, budget):
    import dataclasses
    pm, ac, sc, clips, _ = load_resources(samples_dir)
    sc = dataclasses.replace(sc, redecode_budget_ms=budget)
    clip = clips[0]
    store = SnapshotStore(os.path.join(work_dir, "snap", clip.source))
    return StreamingEngine(clip, pm, ac, sc, store, {})


def test_pending_response_then_convergence(samples_dir, work_dir):
    # budget 4ms with 1ms per task => only four tasks per response
    engine = _engine(samples_dir, work_dir, budget=4)
    first = engine.process_command({"command": "initial_decode"})
    assert first.code == E_REDECODE_BUDGET_PENDING
    assert first.status == "pending"
    assert first.progress["pending_tasks"]
    assert first.within_budget is False
    assert engine.over_budget_responses >= 1

    rounds = 0
    response = first
    while response.code is not None and rounds < 100:
        response = engine.process_command({"command": "resume"})
        rounds += 1
    assert response.code is None
    assert response.status == "completed"
    assert response.progress["percent_complete"] == 100.0
    assert response.transcript
    # summary contains every required budget-bounded field
    assert "stable_prefix_text" in response.to_dict()
    assert "mutable_tail_frames" in response.to_dict()
    assert "affected_chunks" in response.to_dict()


def test_large_budget_always_returns_completed(samples_dir, work_dir):
    engine = _engine(samples_dir, work_dir, budget=100000)
    response = engine.process_command({"command": "initial_decode"})
    assert response.code is None
    assert response.within_budget is True
    assert engine.within_budget_responses == 1


def test_edits_interrupting_pending_work_still_converge(samples_dir, work_dir):
    """A new edit while tasks are pending must not strand or duplicate work."""
    import dataclasses
    import os

    from asr_stream_editor.full_decode import full_decode
    from asr_stream_editor.snapshot import SnapshotStore
    from tests.test_equality import _final_engine_state, _assert_reference_equal

    pm, ac, sc, clips, _ = load_resources(samples_dir)
    sc = dataclasses.replace(sc, redecode_budget_ms=3)
    clip = clips[0]
    store = SnapshotStore(os.path.join(work_dir, "snap", clip.source))
    engine = StreamingEngine(clip, pm, ac, sc, store, {})
    # deliberately leave work pending, then change parameters before draining
    first = engine.process_command({"command": "initial_decode"})
    assert first.code is not None
    engine.process_command({"command": "set_stable_frames",
                            "min_consistent_frames": 5})
    engine.process_command({"command": "set_backoff", "backoff_scale": 0.75})
    response = first
    rounds = 0
    while response.code is not None and rounds < 100:
        response = engine.process_command({"command": "resume"})
        rounds += 1
    assert response.code is None
    reference = full_decode(clip, engine.samples, pm, ac, sc,
                            overlap_ms=sc.overlap_ms,
                            min_consistent_frames=5, backoff_scale=0.75)
    _assert_reference_equal(_final_engine_state(engine), reference)
    # no duplicate task identities remain queued
    assert engine._pending == []
