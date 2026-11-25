from __future__ import annotations

import itertools
from pathlib import Path

from voice_diag.config import load_registration_config
from voice_diag.features import extract_acoustic_template
from voice_diag.quantization import quantize_acoustic
from voice_diag.similarity import compare_quantized_templates
from voice_diag.wavio import read_pcm16_wav


def _acoustics(demo: Path):
    failures: list = []
    config = load_registration_config(demo / "config.json", failures)
    assert config is not None and failures == []
    result = {}
    for key, spec in config.template_map().items():
        audio = read_pcm16_wav(spec["path"])
        result[key] = extract_acoustic_template(
            audio,
            config,
            command_word=spec["command_word"],
            template_id=spec["template_id"],
            relative_path=spec["relative_path"],
        )
    return config, result


def test_template_extraction_is_elementwise_repeatable(demo: Path) -> None:
    config, acoustics = _acoustics(demo)
    key = next(key for key, value in sorted(acoustics.items()) if value["effective_frames"]["fricative"] > 0)
    spec = config.template_map()[key]
    second = extract_acoustic_template(
        read_pcm16_wav(spec["path"]),
        config,
        command_word=spec["command_word"],
        template_id=spec["template_id"],
        relative_path=spec["relative_path"],
    )
    assert acoustics[key]["vector"] == second["vector"]
    assert acoustics[key]["dimensions"] == second["dimensions"]
    assert acoustics[key]["effective_frames"]["vowel"] > 0
    assert acoustics[key]["effective_frames"]["fricative"] > 0
    assert all(
        dimension["projection_seed_version"] == 1
        and dimension["valid_frames"] == acoustics[key]["effective_frames"]["total"]
        for dimension in acoustics[key]["dimensions"]
    )


def _quantized(acoustics, profile: str, fractional_bits: int):
    return {
        key: quantize_acoustic(value, profile, fractional_bits)
        for key, value in acoustics.items()
    }


def _comparisons(quantized, threshold: int = 850):
    return [
        compare_quantized_templates(quantized[first], quantized[second], threshold)
        for first, second in itertools.combinations(sorted(quantized), 2)
    ]


def test_profile_and_fractional_bit_switches_change_integers_similarities_and_candidates(demo: Path) -> None:
    _, acoustics = _acoustics(demo)
    q12 = _quantized(acoustics, "q12", 8)
    q16 = _quantized(acoustics, "q16", 8)
    q12_fb6 = _quantized(acoustics, "q12", 6)

    key = sorted(q12)[0]
    assert q12[key]["integers"] != q16[key]["integers"]
    assert q12[key]["integers"] != q12_fb6[key]["integers"]

    sims12 = _comparisons(q12)
    sims16 = _comparisons(q16)
    sims_fb6 = _comparisons(q12_fb6)
    assert [item["similarity_permille"] for item in sims12] != [
        item["similarity_permille"] for item in sims16
    ]
    assert [item["similarity_permille"] for item in sims12] != [
        item["similarity_permille"] for item in sims_fb6
    ]
    assert {item["pair_key"] for item in sims12 if item["candidate"]} != {
        item["pair_key"] for item in sims16 if item["candidate"]
    }
    assert {item["pair_key"] for item in sims12 if item["candidate"]} != {
        item["pair_key"] for item in sims_fb6 if item["candidate"]
    }


def test_quantization_residual_step_and_clamping_are_auditable(demo: Path) -> None:
    _, acoustics = _acoustics(demo)
    quantized = next(iter(_quantized(acoustics, "q12", 8).values()))
    assert quantized["step"] == "1/256"
    assert quantized["clamped"] is True
    assert quantized["clamped_dimensions"]
    for dimension in quantized["dimensions"]:
        expected = dimension["source"] - dimension["integer"] / 256.0
        assert abs(dimension["residual"] - expected) < 1e-10
