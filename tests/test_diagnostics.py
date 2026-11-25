from __future__ import annotations

from pathlib import Path

import numpy as np

from fixedtex.config import load_material, load_render_config
from fixedtex.diagnostics import build_diagnostics, build_error_report, compare_quant_steps
from fixedtex.texture import load_texture

from .helpers import SAMPLE_CONFIG, SAMPLE_MATERIAL


def _bundle():
    config, _ = load_render_config(SAMPLE_CONFIG)
    material, _ = load_material(SAMPLE_MATERIAL, len(config.layers))
    texture, _ = load_texture(config.texture_path, expected_resolution=config.resolution)
    return config, material, texture


def test_stage_error_report_has_first_exceed_and_pixel_arrays(tmp_path: Path) -> None:
    config, material, texture = _bundle()
    report = build_error_report(config, material, texture, tmp_path)
    assert report["tolerance_fixed"] == 1
    assert report["first_exceed_stage"] in {
        "layer_noise_synthesis",
        "layer_quantization",
        "material_mixing",
        "channel_composite",
    }
    assert report["first_exceed_count"] > 0
    with np.load(tmp_path / "stage_diffs.npz", allow_pickle=False) as arrays:
        for index in range(len(config.layers)):
            assert arrays[f"synthesis_layer_{index}"].shape == (64, 64)
            assert arrays[f"quantization_layer_{index}"].shape == (64, 64)
        for channel in ("diffuse", "roughness", "metallic", "normal"):
            assert arrays[f"mix_{channel}"].shape == (64, 64)
        assert arrays["first_exceed_stage"].shape == (64, 64)
        assert int(arrays["first_exceed_stage"].max()) >= 1
    for stage in (
        "layer_noise_synthesis",
        "layer_quantization",
        "material_mixing",
        "channel_composite",
    ):
        assert stage in report["stages"]


def test_quant_step_comparison_classifies_every_mismatch(tmp_path: Path) -> None:
    config, material, texture = _bundle()
    report = compare_quant_steps(config, material, texture, (4, 8), tmp_path)
    assert report["mismatch_pixels"] > 0
    assert sum(report["classifications"].values()) == report["mismatch_pixels"]
    assert report["classifications"]["quantization_rounding"] > 0
    assert report["classifications"]["boundary_decision"] > 0
    assert report["samples"]
    sample = report["samples"][0]
    for key in (
        "synthesis_deviation",
        "quantization_deviation",
        "mixing_deviation",
        "composite_deviation",
        "classification",
    ):
        assert key in sample
    with np.load(tmp_path / "quant_compare.npz", allow_pickle=False) as arrays:
        assert set(np.unique(arrays["classification"]).tolist()) <= {0, 1, 2, 3, 4}
        assert arrays["quant_a"].shape == (64, 64, 4)
        assert arrays["quant_b"].shape == (64, 64, 4)


def test_combined_diagnostics_writes_report(tmp_path: Path) -> None:
    config, material, texture = _bundle()
    report = build_diagnostics(config, material, texture, tmp_path, quant_steps=(4, 8))
    assert report["ok"] is True
    assert (tmp_path / "diagnostics.json").exists()
    assert (tmp_path / "error_report.json").exists()
    assert (tmp_path / "quant_compare.json").exists()
