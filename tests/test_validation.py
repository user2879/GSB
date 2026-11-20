import json
import os
import wave

import pytest

from asr_stream_editor.audio import load_wav
from asr_stream_editor.config import (load_acoustic_config, load_phoneme_map,
                                      load_stream_config)
from asr_stream_editor.errors import E_AUDIO_FORMAT, E_CONFIG_INVALID, BusinessError
from asr_stream_editor.runner import load_resources


def test_broken_wav_goes_to_failure_list(samples_dir):
    _, _, _, clips, failures = load_resources(samples_dir)
    sources = {item["source"] for item in failures}
    assert "broken_sample.wav" in sources
    assert failures[0]["code"] == E_AUDIO_FORMAT
    assert {clip.source for clip in clips} == {
        "clip_01.wav", "clip_02.wav", "clip_03.wav"}


def test_valid_wav_passes_strict_checks(samples_dir):
    clip = load_wav(os.path.join(samples_dir, "clip_01.wav"))
    assert clip.n_samples > 0
    assert len(clip.fingerprint) == 16


def _write(tmp_path, name, payload):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return str(path)


def test_reject_bad_sample_rate(tmp_path, samples_dir):
    path = tmp_path / "bad_rate.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 100)
    with pytest.raises(BusinessError) as exc:
        load_wav(str(path))
    assert exc.value.code == E_AUDIO_FORMAT


def test_phoneme_uniqueness_and_probability_bounds(tmp_path, samples_dir):
    phoneme = load_phoneme_map(os.path.join(samples_dir, "phoneme_map.json"))
    bad = {"mapping": dict(phoneme.mapping)}
    bad["mapping"]["p_a"] = 123
    with pytest.raises(BusinessError) as exc:
        load_phoneme_map(_write(tmp_path, "pm.json", bad))
    assert exc.value.code == E_CONFIG_INVALID

    acoustic_path = os.path.join(samples_dir, "acoustic_params.json")
    acoustic = load_acoustic_config(acoustic_path, phoneme)
    bad_acoustic = json.load(open(acoustic_path))
    bad_acoustic["bigram"]["a|bo"] = 1.7
    with pytest.raises(BusinessError) as exc:
        load_acoustic_config(_write(tmp_path, "ac.json", bad_acoustic), phoneme)
    assert exc.value.code == E_CONFIG_INVALID
    assert acoustic.log_smooth < 0


def test_stream_config_enum_and_overlap_rule(tmp_path, samples_dir):
    path = os.path.join(samples_dir, "stream_config.json")
    config = load_stream_config(path)
    bad = json.load(open(path))
    bad["overlap_ms"] = bad["chunk_ms"]
    with pytest.raises(BusinessError) as exc:
        load_stream_config(_write(tmp_path, "sc1.json", bad))
    assert exc.value.code == E_CONFIG_INVALID
    bad = json.load(open(path))
    bad["precision"] = "ultra"
    with pytest.raises(BusinessError) as exc:
        load_stream_config(_write(tmp_path, "sc2.json", bad))
    assert exc.value.code == E_CONFIG_INVALID
    bad = json.load(open(path))
    bad["concurrency"] = bad["concurrency_limit"] + 1
    with pytest.raises(BusinessError) as exc:
        load_stream_config(_write(tmp_path, "sc3.json", bad))
    assert exc.value.code == E_CONFIG_INVALID
    assert config.significant_digits == 6
