from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from timbre_lab.audio import AudioFile, read_wav_strict, scan_wavs, validate_samples, write_wav
from timbre_lab.config import load_config
from timbre_lab.errors import E_AUDIO_FORMAT, E_AUDIO_SAMPLE, BusinessError
from timbre_lab.segmentation import build_document, segment_audio


def test_valid_wav_and_corrupt_headers(tmp_path: Path) -> None:
    valid = tmp_path / "valid.wav"
    samples = np.zeros(1600, dtype=np.int16)
    write_wav(valid, samples)
    loaded = read_wav_strict(valid)
    assert loaded.samples.size == 1600

    bad_riff = tmp_path / "bad_riff.wav"
    bad_riff.write_bytes(b"not a wave file at all")
    with pytest.raises(BusinessError) as error:
        read_wav_strict(bad_riff)
    assert error.value.code == E_AUDIO_FORMAT

    payload = bytearray(valid.read_bytes())
    data_offset = payload.find(b"data")
    struct.pack_into("<I", payload, data_offset + 4, 0xFFFF)
    truncated = tmp_path / "truncated.wav"
    truncated.write_bytes(bytes(payload))
    with pytest.raises(BusinessError) as error:
        read_wav_strict(truncated)
    assert error.value.code == E_AUDIO_FORMAT


def test_wrong_sample_rate_channels_and_width_are_rejected(tmp_path: Path) -> None:
    for name, channels, rate, width in (
        ("stereo.wav", 2, 16000, 2),
        ("rate.wav", 1, 8000, 2),
        ("width.wav", 1, 16000, 3),
    ):
        path = tmp_path / name
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(width)
            handle.setframerate(rate)
            handle.writeframes(bytes(channels * 1600 * width))
        with pytest.raises(BusinessError) as error:
            read_wav_strict(path)
        assert error.value.code == E_AUDIO_FORMAT


def test_non_finite_sample_values_are_rejected() -> None:
    with pytest.raises(BusinessError) as error:
        validate_samples(np.array([0.0, np.nan, 1.0]))
    assert error.value.code == E_AUDIO_SAMPLE
    with pytest.raises(BusinessError) as error:
        validate_samples(np.array([0.0, np.inf, 1.0]))
    assert error.value.code == E_AUDIO_SAMPLE


def test_scan_isolates_bad_wav_and_keeps_valid_file(tmp_path: Path, demo_workspace: dict[str, Path]) -> None:
    valid_audio = (demo_workspace["samples"] / "voice_dialogue.wav").read_bytes()
    (tmp_path / "voice_dialogue.wav").write_bytes(valid_audio)
    (tmp_path / "bad.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEbroken")
    scan = scan_wavs(tmp_path)
    assert sorted(scan.files) == ["voice_dialogue.wav"]
    assert [item["code"] for item in scan.failures] == [E_AUDIO_FORMAT]


def test_hysteresis_min_segment_and_min_gap(tmp_path: Path, demo_workspace: dict[str, Path]) -> None:
    config = load_config(demo_workspace["config"])
    rng = np.random.default_rng(7)
    samples = rng.normal(0, 4, 32000).astype(np.int16)
    bursts = [(3200, 8000), (8480, 12800), (19200, 22400)]
    t = np.arange(32000)
    for start, end in bursts:
        samples[start:end] = (5000 * np.sin(2 * np.pi * 180 * t[start:end] / 16000)).astype(np.int16)
    audio = AudioFile("bursts.wav", samples)
    segments, summary = segment_audio(audio, config)
    assert len(segments) == 2
    assert segments[0].end_sample >= 12700
    assert segments[1].start_sample >= 18800
    assert summary["min_gap_frames"] == 6


def test_outside_segment_audio_does_not_enter_features(demo_workspace: dict[str, Path]) -> None:
    from timbre_lab.features import extract_segment_features

    config = load_config(demo_workspace["config"])
    scan = scan_wavs(demo_workspace["samples"])
    audio = scan.files["voice_dialogue.wav"]
    document, _, _ = build_document(scan, config)
    segment = document.segments_for_file(audio.name)[0]
    untouched = extract_segment_features(audio, segment, config)
    samples = audio.samples.copy()
    samples[: segment.start_sample] = 12345
    samples[segment.end_sample :] = -12345
    changed_outside = extract_segment_features(AudioFile(audio.name, samples), segment, config)
    assert untouched["content_fingerprint"] == changed_outside["content_fingerprint"]
    assert untouched["statistics"] == changed_outside["statistics"]
    assert segment.end_frame - segment.start_frame == untouched["frame_count"]
