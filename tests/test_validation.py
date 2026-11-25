from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from fixedtex.config import load_material, load_render_config
from fixedtex.texture import load_texture

from .helpers import ROOT, SAMPLE_CONFIG, SAMPLE_MATERIAL, SAMPLE_TEXTURE, write_json


def test_valid_samples_load_without_issues() -> None:
    config, config_issues = load_render_config(SAMPLE_CONFIG)
    assert config is not None and not config_issues
    material, material_issues = load_material(SAMPLE_MATERIAL, len(config.layers))
    assert material is not None and not material_issues
    texture, texture_issues = load_texture(config.texture_path, expected_resolution=config.resolution)
    assert texture is not None and not texture_issues
    assert texture.rgb.dtype.name == "uint8"
    assert texture.width == texture.height == 64
    assert texture.mode == "RGB"


def test_invalid_config_keeps_valid_items(tmp_path: Path) -> None:
    raw = json.loads(SAMPLE_CONFIG.read_text(encoding="utf-8"))
    raw["texture"] = str(SAMPLE_TEXTURE)
    raw["resolution_tiers"] = [64, 32, 64]
    raw["fixed"]["fraction_bits"] = 9
    raw["fixed"]["quant_step"] = 3
    raw["budgets"] = {"layer_points": 0, "batch_points": -1}
    raw["noise"]["layers"][0]["frequency"] = -1
    raw["noise"]["layers"][1]["amplitude"] = 2
    raw["noise"]["layers"][2]["frequency"] = 2.25
    path = write_json(tmp_path / "bad_config.json", raw)
    config, issues = load_render_config(path)
    codes = {item["code"] for item in issues}
    assert config is not None
    assert {"E_RES_TIER", "E_FRAC_BITS", "E_QUANT_STEP", "E_LAYER_BUDGET", "E_BATCH_BUDGET", "E_FREQ"} <= codes
    assert len(config.layers) == 5
    assert config.layers[2].frequency_milli == 2250
    assert config.fixed.fraction_bits == 12
    assert config.fixed.quant_step == 4
    assert config.budgets.layer_points > 0


def test_invalid_material_keeps_other_channels(tmp_path: Path) -> None:
    raw = json.loads(SAMPLE_MATERIAL.read_text(encoding="utf-8"))
    del raw["channels"]["normal"]
    raw["channels"]["diffuse"]["layers"].append({"index": 99, "weight": 0.2})
    path = write_json(tmp_path / "bad_material.json", raw)
    material, issues = load_material(path, 5)
    codes = {item["code"] for item in issues}
    assert material is not None
    assert "E_MATERIAL_CHANNEL" in codes
    assert "E_WEIGHT" in codes
    assert set(material.channels) == {"diffuse", "roughness", "metallic", "normal"}
    assert material.channels["normal"].valid is False
    assert material.channels["roughness"].layer_weights


def test_png_header_resolution_and_mode_validation(tmp_path: Path) -> None:
    bad_header = tmp_path / "bad_header.png"
    bad_header.write_bytes(b"not a png")
    texture, issues = load_texture(bad_header)
    assert texture is None
    assert issues[0]["code"] == "E_PNG_HEADER"

    wrong_resolution = tmp_path / "wrong_resolution.png"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(wrong_resolution)
    texture, issues = load_texture(wrong_resolution, expected_resolution=64)
    assert texture is None
    assert issues[0]["code"] == "E_PNG_RESOLUTION"

    palette = tmp_path / "palette.png"
    Image.new("P", (64, 64)).save(palette)
    texture, issues = load_texture(palette, expected_resolution=64)
    assert texture is None
    assert issues[0]["code"] == "E_PNG_MODE"

    rgba = tmp_path / "rgba.png"
    Image.new("RGBA", (64, 64), (10, 20, 30, 255)).save(rgba)
    texture, issues = load_texture(rgba, expected_resolution=64)
    assert texture is not None and not issues
    assert texture.mode == "RGBA"
    assert texture.rgb.shape == (64, 64, 3)
