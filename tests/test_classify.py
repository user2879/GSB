import numpy as np

from workbench.classify import classify_tile, iter_windows
from workbench.config import CLASSES, load_config
from workbench.fixedpoint import FixedPoint


def _config(config_path):
    return load_config(config_path)


def test_windows_partition_tile_exactly():
    covered = np.zeros((32, 32), dtype=int)
    for core, ctx, offset in iter_windows(32, 16, 2):
        covered[core] += 1
        assert ctx[0].start <= core[0].start and ctx[0].stop >= core[0].stop
    assert (covered == 1).all()


def test_classification_deterministic(small_samples, config_path):
    from workbench.loader import load_tile_bands
    bands = load_tile_bands(small_samples, "tile_r0_c0", ["R", "G", "B", "ELEV"], 32)
    config = _config(config_path)
    fp = FixedPoint(config.fixed_point_frac_bits, config.quantum)
    r1 = classify_tile(bands, config, fp)
    r2 = classify_tile(bands, config, fp)
    assert np.array_equal(r1["classes"], r2["classes"])
    assert np.array_equal(r1["probs"], r2["probs"])
    assert r1["score_summary"] == r2["score_summary"]


def test_tie_breaks_to_lowest_class_code(config_path):
    config = _config(config_path)
    # identical weights for every class -> all scores tie -> class 0 wins
    for cls in CLASSES:
        config.index_weights[cls] = {"veg": 1, "water": 1, "texture": 1, "slope": 1}
    fp = FixedPoint(config.fixed_point_frac_bits, config.quantum)
    bands = {b: np.full((32, 32), 100, dtype=np.uint8) for b in ("R", "G", "B", "ELEV")}
    result = classify_tile(bands, config, fp)
    assert (result["classes"] == 0).all()


def test_scores_are_quantized(small_samples, config_path):
    from workbench.loader import load_tile_bands
    config = _config(config_path)
    fp = FixedPoint(config.fixed_point_frac_bits, config.quantum)
    bands = load_tile_bands(small_samples, "tile_r0_c0", ["R", "G", "B", "ELEV"], 32)
    result = classify_tile(bands, config, fp)
    assert (result["probs"] % 1 == 0).all()
    assert result["prob_summary"]["hash"]
