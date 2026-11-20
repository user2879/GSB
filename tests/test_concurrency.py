"""Requirement 7: concurrency level must not change any result field."""

import dataclasses
import os

from asr_stream_editor.engine import StreamingEngine
from asr_stream_editor.runner import load_resources
from asr_stream_editor.snapshot import SnapshotStore


def _run(samples_dir, work_dir, concurrency, budget):
    pm, ac, sc, clips, _ = load_resources(samples_dir)
    sc = dataclasses.replace(sc, concurrency=concurrency,
                             concurrency_limit=max(8, concurrency),
                             redecode_budget_ms=budget)
    clip = clips[0]
    store = SnapshotStore(os.path.join(work_dir, f"snap-{concurrency}", clip.source))
    engine = StreamingEngine(clip, pm, ac, sc, store, {})
    response = engine.process_command({"command": "initial_decode"})
    while response.code is not None:
        response = engine.process_command({"command": "resume"})
    return engine, response.to_dict()


def test_concurrency_does_not_change_results(samples_dir, work_dir):
    engine_single, single = _run(samples_dir, os.path.join(work_dir, "s1"), 1, 6)
    engine_multi, multi = _run(samples_dir, os.path.join(work_dir, "s4"), 4, 6)
    assert single["transcript"] == multi["transcript"]
    assert single["alignment_conflicts"] == multi["alignment_conflicts"]
    assert single["failures"] == multi["failures"]
    assert single["budget_consumed_ms"] == multi["budget_consumed_ms"]
    assert engine_single.budget_consumed_total == engine_multi.budget_consumed_total
    # stable ordering keys are derived deterministically
    from asr_stream_editor.engine import stable_sort_key
    keys = [stable_sort_key(i, stage) for i in range(3)
            for stage in ("feature", "decode")]
    assert keys == sorted(keys)


def test_concurrency_never_exceeds_limit(samples_dir, work_dir):
    pm, ac, sc, clips, _ = load_resources(samples_dir)
    assert sc.concurrency <= sc.concurrency_limit
