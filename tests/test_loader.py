from __future__ import annotations

from pathlib import Path

import pytest

from voice_diag.config import load_registration_config
from voice_diag.engine import DiagnosticEngine
from voice_diag.errors import AudioFormatFailure
from voice_diag.wavio import read_pcm16_wav, scan_wav_files

from .conftest import append_label, update_config, write_float_nan_wav


def test_demo_config_and_every_wav_are_valid(demo: Path) -> None:
    failures: list = []
    config = load_registration_config(demo / "config.json", failures)
    assert config is not None
    assert failures == []
    wavs = scan_wav_files(demo)
    assert len(wavs) == 46
    for wav in wavs:
        audio = read_pcm16_wav(wav)
        assert audio.sample_rate == 16000
        assert audio.channels == 1
        assert audio.bit_depth == 16
        assert audio.sample_count > 0


def test_truncated_wav_is_isolated_and_other_clips_continue(demo: Path) -> None:
    source = demo / "clips" / "pos_lin_01.wav"
    broken = demo / "clips" / "broken.wav"
    broken.write_bytes(source.read_bytes()[:40])
    append_label(
        demo,
        {
            "clip_id": "broken",
            "path": "clips/broken.wav",
            "true_command": "LIN",
            "confusion_with": "LING",
            "kind": "positive",
        },
    )
    result, _ = DiagnosticEngine(demo / "config.json", demo / ".state").run()
    assert any(item["code"] == "E_AUDIO_FORMAT" and item["source"] == "clips/broken.wav" for item in result["failures"])
    assert len(result["conclusions"]) == 28
    totals = result["confusion_matrix"]["totals"]
    assert totals["expected_clip_count"] == 29
    assert totals["unavailable_clip_count"] == 1
    assert result["confusion_matrix"]["consistency"]["consistent"] is True


def test_non_finite_float_wav_is_rejected_explicitly(demo: Path) -> None:
    path = demo / "clips" / "nonfinite.wav"
    write_float_nan_wav(path)
    with pytest.raises(AudioFormatFailure) as caught:
        read_pcm16_wav(path)
    assert caught.value.failure.code == "E_AUDIO_NONFINITE"


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda config: config["quantization"].__setitem__("profile", "q99"), "quantization.profile"),
        (lambda config: config["features"].__setitem__("frame_shift", 0), "features.frame_shift"),
        (lambda config: config["features"].__setitem__("dimension", 13), "features.dimension"),
        (lambda config: config["thresholds"].__setitem__("reject_permille", 12.5), "thresholds.reject_permille"),
        (
            lambda config: config["words"][2].__setitem__("templates", config["words"][2]["templates"][:1]),
            "words[2].templates",
        ),
    ],
)
def test_config_validation_reports_field_and_line(demo: Path, mutate, field: str) -> None:
    update_config(demo, mutate)
    failures: list = []
    config = load_registration_config(demo / "config.json", failures)
    assert config is None
    matching = [failure for failure in failures if failure.field == field]
    assert matching, [(failure.field, failure.message) for failure in failures]
    assert any(failure.line is not None for failure in matching)


def test_duplicate_word_and_template_ids_are_rejected(demo: Path) -> None:
    def mutate(config):
        config["words"][1]["command_word"] = config["words"][0]["command_word"]
        config["words"][0]["templates"][1]["template_id"] = config["words"][0]["templates"][0]["template_id"]

    update_config(demo, mutate)
    failures: list = []
    assert load_registration_config(demo / "config.json", failures) is None
    fields = {failure.field for failure in failures}
    assert any("command_word" in field for field in fields)
    assert any("template_id" in field for field in fields)
