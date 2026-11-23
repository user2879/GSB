import json

from workbench import errors
from workbench.pipeline import Pipeline
from workbench.verify import fusion_artifacts_equal, reports_equal

from .conftest import write_config


def strip_config_fields(report):
    rep = dict(report)
    rep.pop("config_hash", None)
    rep.pop("global_version", None)
    return rep


def _codes(report):
    return {f["code"] for f in report["failed"]}


def test_per_tile_quota_aborts_tile(small_samples, tmp_path):
    cfg = write_config(small_samples / "config.json", per_tile_quota=512)
    report = Pipeline(small_samples, tmp_path / "w", config_path=cfg).run()
    assert errors.E_TILE_BUDGET_EXHAUSTED in _codes(report)
    assert report["per_tile_class_counts"] == {}


def test_global_budget_parks_pending_and_resumes(small_samples, tmp_path):
    cfg = write_config(small_samples / "config.json", global_pixel_budget=1024 * 2)
    work = tmp_path / "w"
    r1 = Pipeline(small_samples, work, config_path=cfg).run()
    assert errors.E_GLOBAL_BUDGET_EXHAUSTED in _codes(r1)
    assert len(r1["pending_tiles"]) == 2
    assert len(r1["per_tile_class_counts"]) == 2
    # next run with a fresh budget continues from the pending queue
    r2 = Pipeline(small_samples, work, config_path=cfg).run()
    assert len(r2["per_tile_class_counts"]) == 4
    # final state equals a from-scratch full run with a sufficient budget
    cfg_big = write_config(small_samples / "config.json",
                           global_pixel_budget=10 ** 9)
    r_full = Pipeline(small_samples, tmp_path / "wf", config_path=cfg_big,
                      full=True).run()
    ok, diffs = reports_equal(strip_config_fields(r2), strip_config_fields(r_full))
    assert ok, diffs
    ok, diffs = fusion_artifacts_equal(work, tmp_path / "wf")
    assert ok, diffs


def test_merge_batch_limit_retains_pairs(small_samples, tmp_path):
    cfg = write_config(small_samples / "config.json", merge_batch_limit=1)
    work = tmp_path / "w"
    r1 = Pipeline(small_samples, work, config_path=cfg).run()
    assert len(r1["retained_fusions"]) == 3  # 2x2 grid -> 4 pairs, 1 per batch
    r2 = Pipeline(small_samples, work, config_path=cfg).run()
    assert len(r2["retained_fusions"]) == 2
    r3 = Pipeline(small_samples, work, config_path=cfg).run()
    r4 = Pipeline(small_samples, work, config_path=cfg).run()
    assert r4["retained_fusions"] == []
    cfg_big = write_config(small_samples / "config.json",
                           merge_batch_limit=10 ** 6)
    r_full = Pipeline(small_samples, tmp_path / "wf", config_path=cfg_big,
                      full=True).run()
    ok, diffs = reports_equal(strip_config_fields(r4), strip_config_fields(r_full))
    assert ok, diffs


def test_concurrency_independence(small_samples, tmp_path):
    cfg1 = write_config(small_samples / "config.json", thread_pool_size=1,
                        global_pixel_budget=1024 * 3, merge_batch_limit=2)
    r1 = Pipeline(small_samples, tmp_path / "w1", config_path=cfg1).run()
    cfg4 = write_config(small_samples / "config.json", thread_pool_size=4)
    r4 = Pipeline(small_samples, tmp_path / "w4", config_path=cfg4).run()
    # budget deduction sequence, abort/pending lists identical across concurrency
    assert r1["budget_trace"] == r4["budget_trace"]
    assert r1["pending_tiles"] == r4["pending_tiles"]
    assert r1["retained_fusions"] == r4["retained_fusions"]
    assert r1["failed"] == r4["failed"]
    ok, diffs = reports_equal(strip_config_fields(r1), strip_config_fields(r4))
    assert ok, diffs
