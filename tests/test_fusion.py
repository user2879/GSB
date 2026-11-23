import numpy as np

from workbench.config import load_config
from workbench.fixedpoint import FixedPoint
from workbench.fusion import fuse_pair, postprocess
from workbench.loader import TileSpec, load_manifest


def _mk_result(cls, prob, size=32):
    classes = np.full((size, size), cls, dtype=np.uint8)
    probs = np.zeros((size, size, 4), dtype=np.int64)
    probs[:, :, cls] = prob
    return {"classes": classes, "probs": probs,
            "score_summary": {"class_pixel_counts": [0, 0, 0, 0],
                              "mean_scores": [0, 0, 0, 0]},
            "prob_summary": {"mean_probs": [0, 0, 0, 0], "hash": "x"}}


def test_fusion_tie_breaks_by_tile_row_col(small_samples, config_path):
    manifest = load_manifest(small_samples)
    config = load_config(config_path)
    fp = FixedPoint(config.fixed_point_frac_bits, config.quantum)
    spec_a = manifest.tiles["tile_r0_c0"]
    spec_b = manifest.tiles["tile_r0_c1"]
    # both tiles spread probability evenly over classes 0 and 1 -> every vote
    # ties -> the tile with the smaller (row, col) decides via its own class
    res_a = _mk_result(1, 0)
    res_b = _mk_result(0, 0)
    for res in (res_a, res_b):
        res["probs"][:, :, 0] = 2048
        res["probs"][:, :, 1] = 2048
    out = fuse_pair(res_a, res_b, spec_a, spec_b, manifest, config, fp, "va", "vb")
    assert (out["fused"] == 1).all()
    assert all(r["tie_broken"] for r in out["records"])


def test_fusion_records_are_traceable(small_samples, config_path):
    manifest = load_manifest(small_samples)
    config = load_config(config_path)
    fp = FixedPoint(config.fixed_point_frac_bits, config.quantum)
    spec_a = manifest.tiles["tile_r0_c0"]
    spec_b = manifest.tiles["tile_r0_c1"]
    out = fuse_pair(_mk_result(0, 3000), _mk_result(1, 3000),
                    spec_a, spec_b, manifest, config, fp, "va", "vb")
    rec = out["records"][0]
    assert rec["tiles"] == ["tile_r0_c0", "tile_r0_c1"]
    assert set(rec["weights"]) == {"tile_r0_c0", "tile_r0_c1"}
    assert rec["prob_versions"] == {"tile_r0_c0": "va", "tile_r0_c1": "vb"}
    assert len(rec["votes"]) == 4
    # distance weights decrease away from the owning tile interior
    first_col = [r for r in out["records"] if r["global_col"] ==
                 min(x["global_col"] for x in out["records"])]
    assert first_col[0]["weights"]["tile_r0_c0"] > first_col[0]["weights"]["tile_r0_c1"]


def test_postprocess_refills_small_components():
    grid = np.zeros((8, 8), dtype=np.uint8)
    grid[3, 3] = 2  # isolated pixel
    cleaned, refills = postprocess(grid, min_connected=4)
    assert cleaned[3, 3] == 0
    assert len(refills) == 1
    rec = refills[0]
    assert rec["old_class"] == 2 and rec["new_class"] == 0
    assert rec["size"] == 1 and rec["reason"] == "below_min_connected"
    assert rec["neighbor_class_counts"] == {"0": 4}
