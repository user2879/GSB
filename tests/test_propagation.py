"""Invalidation propagation and only-affected-subset recomputation."""
from __future__ import annotations

from dataclasses import replace

from obsflow.config import load_source_set
from obsflow.engine import ObsFlowEngine
from obsflow.propagation import diff_source_sets, derive_propagation


def _engine(sample_dir, ledger_path):
    return ObsFlowEngine.from_files(
        sample_dir["declaration_path"], sample_dir["window_path"],
        ledger_path)


def test_metric_mapping_only_invalidates_dependent_windows(sample_dir,
                                                           workdir):
    engine = _engine(sample_dir, f"{workdir}/a.db")
    engine.read_records()
    engine.advance()
    before = {k: w.window_version for k, w in engine._windows.items()}

    source_set = load_source_set(sample_dir["declaration_path"])
    sources = source_set.sources
    changed = replace(sources[2],
                      metric_mapping={"raw_hum": "humidity_mapped"})
    new_set = source_set.__class__(
        source_set_version=2,
        sources=(sources[0], sources[1], changed),
        base_dir=source_set.base_dir)

    changes = diff_source_sets(source_set.by_id(), new_set.by_id())
    assert any(c.field == "metric_mapping" for c in changes)
    plan = engine.apply_source_set(new_set)
    assert [e.order for e in plan.events] == list(range(1, len(plan.events) + 1))
    assert plan.affected
    report = engine.advance(plan)
    after = {k: w.window_version for k, w in engine._windows.items()}
    for key, version in before.items():
        if key in plan.affected:
            assert after[key] == version + 1
        else:
            assert after[key] == version
    assert set(report.recomputed) == plan.affected
    engine.close()


def test_window_size_change_rebuilds_entire_grid(sample_dir, workdir):
    engine = _engine(sample_dir, f"{workdir}/b.db")
    engine.read_records()
    engine.advance()
    plan = engine.apply_window_params(window_size=7, slide_step=2)
    assert all(e.reason == "window_param_change" for e in plan.events)
    report = engine.advance(plan)
    assert all(w.count == 7 for w in report.windows)
    engine.close()


def test_format_version_change_invalidates_source(sample_dir, workdir):
    old = load_source_set(sample_dir["declaration_path"]).by_id()
    new_set = load_source_set(sample_dir["declaration_path"])
    changed_source = replace(new_set.sources[0], format_version=2)
    new_set = new_set.__class__(
        source_set_version=3,
        sources=(changed_source,) + new_set.sources[1:],
        base_dir=new_set.base_dir)
    changes = diff_source_sets(old, new_set.by_id())
    assert any(c.field == "format_version" for c in changes)
    plan = derive_propagation(old, new_set.by_id(), [], changes)
    assert plan.events[0].changed_field == "format_version"
