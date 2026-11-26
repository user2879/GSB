from __future__ import annotations

import json

import pytest

from mmfm.config import load_render_config
from mmfm.errors import BusinessError
from mmfm.exchange import parse_v1, v1_to_v2, v2_to_v1


def profile_bits(config):
    return {name: profile.fraction_bits for name, profile in config.quant_profiles.items()}


def test_v2_v1_v2_round_trip_restores_float_refs_and_residual(repo_root):
    config = load_render_config(repo_root / "configs" / "render.json")
    document = json.loads((repo_root / "samples" / "exchange" / "denoise_v2.json").read_text())
    text, sidecar = v2_to_v1(
        document,
        noise_schemes=config.noise_schemes,
        profiles=profile_bits(config),
        allow_missing=False,
    )
    data_lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    assert data_lines[0] == "0001|01235|01|08"
    assert all(len(line) == 16 for line in data_lines)
    assert sidecar["downgrade_residuals"][0]["guidance_residual"] != 0
    assert sidecar["downgrade_residuals"][0]["overridden_references"] == ["condition:1:text", "fusion:1"]
    restored = v1_to_v2(
        text,
        sidecar=sidecar,
        noise_code_map=config.noise_schemes,
        profile_bits=profile_bits(config),
    )
    for original, back in zip(document["records"], restored["records"]):
        assert back["guidance"] == original["guidance"]
        assert back["refs"] == original["refs"]
        assert back["noise_scheme"] == original["noise_scheme"]
        assert back["quant_profile"] == original["quant_profile"]
        assert back["metadata"]["sidecar_restored"] is True
    assert restored["provenance"]["sidecar_hash_verified"] is True


def test_v1_to_v2_maps_codes_and_marks_missing_fields(repo_root):
    config = load_render_config(repo_root / "configs" / "render.json")
    text = "\n".join(
        [
            "# MMFM-V1 fields=frame:4,guidance_permille:5,noise_code:2,quant_bits:2",
            "# stage=denoise",
            "0003|01250|02|12",
            "0004|     |  |  ",
            "",
        ]
    )
    document = v1_to_v2(
        text,
        noise_code_map=config.noise_schemes,
        profile_bits=profile_bits(config),
    )
    first, second = document["records"]
    assert first["guidance"] == 1.25
    assert first["noise_scheme"] == "counter"
    assert first["quant_profile"] == "q12"
    assert second["guidance"] is None
    assert set(second["missing_fields"]) == {"guidance", "noise_code", "quant_bits"}


def test_bad_v1_width_and_unknown_enum(repo_root, tmp_path):
    config = load_render_config(repo_root / "configs" / "render.json")
    bad_width = "# MMFM-V1\n# stage=denoise\n0001|010000|01|08\n"
    with pytest.raises(BusinessError) as caught:
        parse_v1(bad_width, noise_code_map=config.noise_schemes, profile_bits=profile_bits(config))
    assert caught.value.code == "E_V1_WIDTH"
    unknown_code = "# MMFM-V1\n# stage=denoise\n0001|01000|99|08\n"
    with pytest.raises(BusinessError) as caught:
        parse_v1(unknown_code, noise_code_map=config.noise_schemes, profile_bits=profile_bits(config))
    assert caught.value.code == "E_ENUM_INVALID"


def test_sidecar_hash_mismatch_is_rejected(repo_root):
    config = load_render_config(repo_root / "configs" / "render.json")
    document = json.loads((repo_root / "samples" / "exchange" / "denoise_v2.json").read_text())
    text, sidecar = v2_to_v1(
        document,
        noise_schemes=config.noise_schemes,
        profiles=profile_bits(config),
        allow_missing=False,
    )
    sidecar["v1_sha256"] = "0" * 64
    with pytest.raises(BusinessError) as caught:
        parse_v1(text, sidecar=sidecar, noise_code_map=config.noise_schemes, profile_bits=profile_bits(config))
    assert caught.value.code == "E_SIDECAR"
