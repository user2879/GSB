import json

from workbench.edits import apply_edit
from workbench.pipeline import Pipeline
from workbench.verify import fusion_artifacts_equal, reports_equal


def test_edit_sequence_incremental_equals_full(small_samples, tmp_path):
    config = small_samples / "config.json"
    edits = json.loads((small_samples / "edits.json").read_text())
    work_inc = tmp_path / "inc"
    report = None
    for edit in edits:
        apply_edit(small_samples, config, edit)
        report = Pipeline(small_samples, work_inc, config_path=config).run()
    full = Pipeline(small_samples, tmp_path / "full", config_path=config,
                    full=True).run()
    ok, diffs = reports_equal(report, full)
    assert ok, diffs
    ok, diffs = fusion_artifacts_equal(work_inc, tmp_path / "full")
    assert ok, diffs


def test_band_patch_invalidates_only_dependents(small_samples, tmp_path):
    work = tmp_path / "w"
    Pipeline(small_samples, work).run()
    apply_edit(small_samples, small_samples / "config.json",
               {"op": "replace_band", "tile": "tile_r0_c0", "band": "G",
                "seed_delta": 3})
    report = Pipeline(small_samples, work).run()
    tiles = report["invalidation_ledger"]["tiles"]
    assert tiles["tile_r0_c0"]["origin"] == "recomputed"
    assert tiles["tile_r0_c1"]["origin"] == "reused"
    assert tiles["tile_r1_c1"]["origin"] == "reused"
    fusions = report["invalidation_ledger"]["fusions"]
    assert fusions["tile_r0_c0::tile_r0_c1"]["origin"] == "recomputed"
    assert fusions["tile_r0_c0::tile_r1_c0"]["origin"] == "recomputed"
    assert fusions["tile_r0_c1::tile_r1_c1"]["origin"] == "reused"


def test_weight_change_invalidates_tiles_not_geometry(small_samples, tmp_path):
    work = tmp_path / "w"
    Pipeline(small_samples, work).run()
    apply_edit(small_samples, small_samples / "config.json",
               {"op": "set_fusion_strategy", "version": "fs-v2"})
    report = Pipeline(small_samples, work).run()
    tiles = report["invalidation_ledger"]["tiles"]
    assert all(t["origin"] == "reused" for t in tiles.values())
    fusions = report["invalidation_ledger"]["fusions"]
    assert all(f["origin"] == "recomputed" for f in fusions.values())
