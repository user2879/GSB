from __future__ import annotations

import numpy as np

from rsseg.config import SegmentationConfig, load_config
from rsseg.edits import copy_sample_set
from rsseg.errors import ErrorCode
from rsseg.runner import run_incremental
from rsseg.snapshot import SnapshotManager


def _config_with_budget(sample_dir, budget, workers=1):
    cfg = load_config(sample_dir / "segmentation_config.json").to_dict()
    cfg["recompute_budget"] = budget
    cfg["workers"] = workers
    return SegmentationConfig.from_dict(cfg)


def test_zero_budget_defers_everything_with_pending_list(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    cfg = _config_with_budget(sources, 0)
    inputs = __import__("rsseg.runner", fromlist=["build_inputs"]).build_inputs(
        sources, work, config_override=cfg.to_dict())
    from rsseg.engine import IncrementalEngine

    report = IncrementalEngine(inputs).run()
    assert report.recomputed_tiles == []
    assert report.pending_tiles == [(r, c) for r in range(3) for c in range(3)]
    assert all(f.code == ErrorCode.E_RECOMPUTE_BUDGET_PENDING for f in report.failures)
    assert report.snapshot_version is None


def test_budgeted_batches_converge_and_match_full(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    full_work = tmp_path / "full"
    copy_sample_set(sample_set, sources)

    cfg = _config_with_budget(sources, 2)
    from rsseg.runner import build_inputs
    from rsseg.engine import IncrementalEngine

    pending = True
    rounds = 0
    while pending:
        inputs = build_inputs(sources, work, config_override=cfg.to_dict())
        report = IncrementalEngine(inputs).run()
        pending = bool(report.pending_tiles)
        rounds += 1
        assert len(report.recomputed_tiles) <= 2
        assert rounds < 10
    assert report.snapshot_version is not None

    full_cfg = load_config(sources / "segmentation_config.json")
    inputs_full = build_inputs(sources, full_work)
    IncrementalEngine(inputs_full).run(force_full=True)
    inc_grid = np.load(SnapshotManager(work).latest().path / "global_grid.npy")
    full_grid = np.load(SnapshotManager(full_work).latest().path / "global_grid.npy")
    assert np.array_equal(inc_grid, full_grid)


def test_pending_order_is_row_major_and_stable(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    cfg = _config_with_budget(sources, 3)
    from rsseg.runner import build_inputs
    from rsseg.engine import IncrementalEngine

    report = IncrementalEngine(build_inputs(sources, work, config_override=cfg.to_dict())).run()
    assert report.pending_tiles == sorted(report.pending_tiles)
    assert report.recomputed_tiles == [(0, 0), (0, 1), (0, 2)]


def test_injection_zero_budget_overrides_config(sample_set, tmp_path):
    import json

    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    zero_budget_injection = sources / "zero_budget_injection.json"
    zero_budget_injection.write_text(json.dumps({
        "injections": [{"id": "INJ-004", "kind": "zero_budget", "payload": {}}]
    }), encoding="utf-8")
    inputs = __import__("rsseg.runner", fromlist=["build_inputs"]).build_inputs(
        sources, work, injections_path=zero_budget_injection)
    from rsseg.engine import IncrementalEngine

    report = IncrementalEngine(inputs).run()
    assert report.budget == 0
    assert report.pending_tiles and report.failures
    assert all(f.code == ErrorCode.E_RECOMPUTE_BUDGET_PENDING for f in report.failures)
