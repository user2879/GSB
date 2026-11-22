import numpy as np

from mlimgedit.datagen import generate_sample
from mlimgedit.features import (
    build_feature_artifact,
    color_histogram,
    decode_png,
    extract_float_vector,
    feature_vector_length,
    hog_features,
)
from mlimgedit.fixedpoint import FixedPointQuantizer
from mlimgedit.validation import validate_sample
from pathlib import Path


def _png_and_view(config, tmp_path, index=0):
    png, annotation = generate_sample(
        "synthetic",
        index,
        config.labels,
        config.positive_ratios,
        width=config.image_size[0],
        height=config.image_size[1],
        base_seed=config.seed,
    )
    image_path = tmp_path / "synthetic.png"
    json_path = tmp_path / "synthetic.json"
    image_path.write_bytes(png)
    import json

    json_path.write_text(json.dumps(annotation), encoding="utf-8")
    return validate_sample("synthetic", Path(tmp_path), config, {})


def test_decoded_image_is_uint8_and_expected_size(config, tmp_path):
    view = _png_and_view(config, tmp_path)
    pixels = decode_png(view.png_bytes)
    assert pixels.dtype == np.uint8
    assert pixels.shape == (config.image_size[1], config.image_size[0], 3)


def test_feature_length_matches_config(config):
    assert feature_vector_length(config) == config.bins * 3 + (
        (config.image_size[0] // config.hog_cell)
        * (config.image_size[1] // config.hog_cell)
        * config.bins
    )


def test_color_histogram_normalized(config, tmp_path):
    view = _png_and_view(config, tmp_path)
    hist = color_histogram(decode_png(view.png_bytes), config.bins)
    assert hist.shape == (config.bins * 3,)
    for channel in range(3):
        np.testing.assert_allclose(
            hist[channel * config.bins : (channel + 1) * config.bins].sum(), 1.0
        )


def test_hog_shape_and_determinism(config, tmp_path):
    view = _png_and_view(config, tmp_path)
    pixels = decode_png(view.png_bytes)
    first = hog_features(pixels, config.bins, config.hog_cell)
    second = hog_features(pixels, config.bins, config.hog_cell)
    np.testing.assert_array_equal(first, second)
    cells_x = config.image_size[0] // config.hog_cell
    cells_y = config.image_size[1] // config.hog_cell
    assert first.shape == (cells_x * cells_y * config.bins,)


def test_feature_artifact_byte_identical_across_runs(config, tmp_path):
    view = _png_and_view(config, tmp_path)
    quantizer = FixedPointQuantizer(config.quant_level, config.frac_bits, config.rounding)
    first = build_feature_artifact(view, config, quantizer).to_dict()
    second = build_feature_artifact(view, config, quantizer).to_dict()
    assert first["integers"] == second["integers"]
    assert first["residual"] == second["residual"]
    assert first["feature_mode"] == config.feature_mode
    assert first["quant_level"] == config.quant_level
    assert first["rounding"] == config.rounding


def test_feature_modes_are_distinct_and_supported(config, tmp_path):
    view = _png_and_view(config, tmp_path)
    pixels = decode_png(view.png_bytes)
    color = extract_float_vector(pixels, config.with_updates({"feature": {"mode": "color_hist"}}))
    hog = extract_float_vector(pixels, config.with_updates({"feature": {"mode": "hog"}}))
    both = extract_float_vector(pixels, config)
    assert len(both) == len(color) + len(hog)
    np.testing.assert_array_equal(both[: len(color)], color)
    np.testing.assert_array_equal(both[len(color) :], hog)
