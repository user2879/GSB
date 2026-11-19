from __future__ import annotations

import pytest

from glyphforge.errors import ErrorCode, GlyphForgeError
from glyphforge.raster import estimate_raster_work_bytes
from glyphforge.samples import sample_glyphs
from glyphforge.scheduler import RasterScheduler, RasterTask, RasterTaskKey


def _tasks():
    glyphs = sample_glyphs()
    return [
        RasterTask(glyphs["A"], 12, "nonzero", revision=2),
        RasterTask(glyphs["A"], 48, "evenodd", revision=2),
        RasterTask(glyphs["O"], 24, "nonzero", revision=1),
        RasterTask(glyphs["dense"], 16, "nonzero", revision=1),
    ]


def test_results_merge_by_key_not_completion_order():
    scheduler = RasterScheduler()
    result = scheduler.run(
        list(reversed(_tasks())),
        worker_budget=4,
        memory_budget_bytes=8_000_000,
    )
    keys = [key for key, _bitmap in result.results]
    assert keys == sorted(keys)
    assert len(keys) == 4
    assert len(set(keys)) == 4


def test_thread_count_does_not_change_bitmap_bytes():
    scheduler = RasterScheduler()
    single = scheduler.run(_tasks(), worker_budget=1, memory_budget_bytes=8_000_000)
    parallel = scheduler.run(_tasks(), worker_budget=4, memory_budget_bytes=8_000_000)
    assert single.by_key.keys() == parallel.by_key.keys()
    assert {key: result.pixels for key, result in single.by_key.items()} == {
        key: result.pixels for key, result in parallel.by_key.items()
    }


def test_memory_budget_reduces_concurrency_and_records_reason():
    scheduler = RasterScheduler()
    glyphs = sample_glyphs()
    max_estimate = max(
        estimate_raster_work_bytes(glyph, 48) for glyph in glyphs.values()
    )
    result = scheduler.run(
        _tasks(),
        worker_budget=4,
        memory_budget_bytes=max_estimate * 2,
    )
    assert result.report.effective_threads == 2
    assert any(
        decision.reason == "thread_memory_budget"
        for decision in result.report.degradations
    )


def test_duplicate_or_invalid_keys_and_tight_memory_are_errors():
    scheduler = RasterScheduler()
    tasks = _tasks()
    with pytest.raises(GlyphForgeError) as caught:
        scheduler.run(
            [tasks[0], tasks[0]], worker_budget=2, memory_budget_bytes=8_000_000
        )
    assert caught.value.code is ErrorCode.TASK_KEY

    with pytest.raises(GlyphForgeError) as caught:
        RasterTaskKey("A", 13, "nonzero", 1)
    assert caught.value.code is ErrorCode.TASK_KEY

    with pytest.raises(GlyphForgeError) as caught:
        scheduler.run(tasks, worker_budget=2, memory_budget_bytes=4096)
    assert caught.value.code is ErrorCode.BUDGET_DENIED


def test_expected_hash_mismatch_is_reported():
    scheduler = RasterScheduler()
    tasks = _tasks()
    expected = {tasks[0].key: "f" * 64}
    with pytest.raises(GlyphForgeError) as caught:
        scheduler.run(
            tasks,
            worker_budget=1,
            memory_budget_bytes=8_000_000,
            expected_hashes=expected,
        )
    assert caught.value.code is ErrorCode.BITMAP_HASH
