from __future__ import annotations

from pathlib import Path

import pytest

from audiofp.audio import decode_audio
from audiofp.config import DEFAULT_CONFIG
from audiofp.errors import AudioFormatError


def test_multiformat_provenance_and_normalization(fixture_root: Path, manifest: dict) -> None:
    track = manifest["tracks"][0]
    by_label = {entry["label"]: entry for entry in track["formats"]}
    wav16 = decode_audio(fixture_root / by_label["wav16_mono_44k"]["relative_path"], DEFAULT_CONFIG.sample_rate)
    wav24 = decode_audio(fixture_root / by_label["wav24_stereo_48k"]["relative_path"], DEFAULT_CONFIG.sample_rate)
    ogg = decode_audio(fixture_root / by_label["vorbis_stereo_48k"]["relative_path"], DEFAULT_CONFIG.sample_rate)
    assert (wav16.source.sample_rate, wav16.source.bit_depth, wav16.source.channels) == (44100, 16, 1)
    assert (wav24.source.sample_rate, wav24.source.bit_depth, wav24.source.channels) == (48000, 24, 2)
    assert (ogg.source.sample_rate, ogg.source.bit_depth, ogg.source.channels) == (48000, None, 2)
    assert wav16.source.container == "WAV"
    assert ogg.source.container == "OGG"
    assert wav16.source.declared_frames == wav16.source.decoded_frames
    assert wav24.source.declared_frames == wav24.source.decoded_frames
    assert abs(wav16.duration - wav24.duration) < 0.001
    assert abs(wav16.duration - ogg.duration) < 0.001


def test_declared_parameter_mismatch_is_not_silently_accepted(
    fixture_root: Path, manifest: dict
) -> None:
    entry = next(
        item
        for item in manifest["tracks"][0]["formats"]
        if item["label"] == "wav24_stereo_48k"
    )
    path = fixture_root / entry["relative_path"]
    with pytest.raises(AudioFormatError) as error:
        decode_audio(
            path,
            DEFAULT_CONFIG.sample_rate,
            expected={"sample_rate": 44100, "channels": 1, "bit_depth": 16},
        )
    assert error.value.code == "E_AUDIO_FORMAT"
    assert error.value.details["mismatches"]


@pytest.mark.parametrize("index", range(3))
def test_corrupt_samples_raise_audio_format(fixture_root: Path, manifest: dict, index: int) -> None:
    item = manifest["broken_samples"][index]
    with pytest.raises(AudioFormatError) as error:
        decode_audio(fixture_root / item["relative_path"], DEFAULT_CONFIG.sample_rate)
    assert error.value.code == item["expected_error"]
