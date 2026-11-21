from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio import write_wav
from .config import canonical_json


@dataclass(frozen=True)
class SynthSegment:
    start_ms: float
    duration_ms: float
    speaker: str
    style: str
    f0_hz: float
    f1_hz: float
    f2_hz: float
    syllable_rate_sps: float
    f0_variation_hz: float

    @property
    def end_ms(self) -> float:
        return self.start_ms + self.duration_ms

    @property
    def midpoint_ms(self) -> float:
        return (self.start_ms + self.end_ms) / 2.0


SPEAKERS = {
    "spk_amber": {"f0": 112.0, "f1": 520.0, "f2": 1450.0},
    "spk_noah": {"f0": 218.0, "f1": 790.0, "f2": 1950.0},
}


STYLES = {
    "slow": {"syllable_rate_sps": 2.8, "f0_variation_hz": 3.0},
    "neutral": {"syllable_rate_sps": 4.2, "f0_variation_hz": 7.0},
    "fast": {"syllable_rate_sps": 6.4, "f0_variation_hz": 12.0},
}


def _segment(start_ms: float, duration_ms: float, speaker: str, style: str) -> SynthSegment:
    speaker_values = SPEAKERS[speaker]
    style_values = STYLES[style]
    return SynthSegment(
        start_ms=start_ms,
        duration_ms=duration_ms,
        speaker=speaker,
        style=style,
        f0_hz=speaker_values["f0"],
        f1_hz=speaker_values["f1"],
        f2_hz=speaker_values["f2"],
        syllable_rate_sps=style_values["syllable_rate_sps"],
        f0_variation_hz=style_values["f0_variation_hz"],
    )


def dialogue_schedule() -> list[SynthSegment]:
    return [
        _segment(200.0, 700.0, "spk_amber", "slow"),
        _segment(1060.0, 650.0, "spk_noah", "fast"),
        _segment(1880.0, 720.0, "spk_amber", "slow"),
        _segment(2760.0, 620.0, "spk_noah", "fast"),
    ]


def solo_schedule() -> list[SynthSegment]:
    return [
        _segment(180.0, 610.0, "spk_noah", "slow"),
        _segment(950.0, 680.0, "spk_noah", "neutral"),
        _segment(1790.0, 640.0, "spk_amber", "fast"),
        _segment(2500.0, 700.0, "spk_amber", "slow"),
    ]


def _resonance_amplitude(frequency: float, f1: float, f2: float) -> float:
    first = math.exp(-0.5 * ((frequency - f1) / 180.0) ** 2)
    second = 0.72 * math.exp(-0.5 * ((frequency - f2) / 260.0) ** 2)
    return 0.22 + first + second


def synthesize_file(schedule: list[SynthSegment], total_ms: float, seed: int) -> np.ndarray:
    sample_rate = 16000
    total_samples = int(round(total_ms * sample_rate / 1000.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 10 ** (-62.0 / 20.0), total_samples)
    output = noise.copy()
    for segment_index, segment in enumerate(schedule):
        start = int(round(segment.start_ms * sample_rate / 1000.0))
        count = int(round(segment.duration_ms * sample_rate / 1000.0))
        t = np.arange(count, dtype=np.float64) / sample_rate
        vibrato = segment.f0_variation_hz * np.sin(2.0 * math.pi * (3.1 + 0.17 * segment_index) * t + 0.4 * segment_index)
        instantaneous_f0 = segment.f0_hz + vibrato
        phase = 2.0 * math.pi * np.cumsum(instantaneous_f0) / sample_rate
        voiced = np.zeros(count, dtype=np.float64)
        max_harmonic = int((sample_rate / 2.0 - 100.0) // segment.f0_hz)
        for harmonic in range(1, max_harmonic + 1):
            frequency = harmonic * segment.f0_hz
            amplitude = _resonance_amplitude(frequency, segment.f1_hz, segment.f2_hz) / harmonic
            voiced += amplitude * np.sin(harmonic * phase + 0.19 * harmonic)
        voiced /= max(float(np.max(np.abs(voiced))), 1e-9)
        modulation_phase = 2.0 * math.pi * segment.syllable_rate_sps * t + 0.37 * segment_index
        envelope = 0.34 + 0.66 * (0.5 - 0.5 * np.cos(modulation_phase)) ** 1.4
        fade_samples = min(int(0.018 * sample_rate), count // 4)
        fade = np.ones(count, dtype=np.float64)
        if fade_samples > 0:
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, fade_samples))
            fade[:fade_samples] = ramp
            fade[-fade_samples:] = ramp[::-1]
        signal = voiced * envelope * fade * 10 ** (-17.0 / 20.0)
        output[start : start + count] += signal
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0.94:
        output *= 0.94 / peak
    return np.clip(np.round(output * 32767.0), -32768, 32767).astype(np.int16)


def synthesize_replacement(duration_ms: float = 250.0, seed: int = 991) -> np.ndarray:
    schedule = [
        SynthSegment(
            start_ms=0.0,
            duration_ms=duration_ms,
            speaker="spk_noah",
            style="fast",
            f0_hz=232.0,
            f1_hz=820.0,
            f2_hz=2020.0,
            syllable_rate_sps=6.7,
            f0_variation_hz=15.0,
        )
    ]
    return synthesize_file(schedule, duration_ms, seed)


def _base_config(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sample_rate": 16000,
        "channels": 1,
        "sample_width_bits": 16,
        "frame_length_ms": 25.0,
        "frame_shift_ms": 10.0,
        "energy": {"enter_db": -44.0, "exit_db": -54.0},
        "min_segment_ms": 90.0,
        "min_gap_ms": 60.0,
        "quantile_interpolation": "linear",
        "f0": {"min_hz": 60.0, "max_hz": 500.0, "voiced_autocorr": 0.24},
        "formants": {
            "lpc_order": 16,
            "f1_range_hz": [180.0, 1150.0],
            "f2_range_hz": [700.0, 3200.0],
        },
        "style": {
            "speech_rate_bins_sps": [3.7, 5.1],
            "labels": ["slow", "neutral", "fast"],
            "high_variation_f0_std_hz": 45.0,
            "high_variation_label": "expressive",
        },
        "timbre_dimensions": [
            {"name": "f0_mean", "metric": "f0_mean_hz", "dynamic_range": [70.0, 360.0], "target_step": 3.0, "bit_widths": [4, 6, 8, 10]},
            {"name": "f0_stability", "metric": "f0_std_hz", "dynamic_range": [0.0, 45.0], "target_step": 0.8, "bit_widths": [4, 6, 8]},
            {"name": "formant_f1", "metric": "f1_mean_hz", "dynamic_range": [180.0, 1150.0], "target_step": 10.0, "bit_widths": [4, 6, 8, 10]},
            {"name": "formant_f2", "metric": "f2_mean_hz", "dynamic_range": [700.0, 3200.0], "target_step": 15.0, "bit_widths": [4, 6, 8, 10]},
            {"name": "speech_rate", "metric": "speech_rate_sps", "dynamic_range": [0.0, 8.0], "target_step": 0.25, "bit_widths": [4, 5, 6, 8]},
            {"name": "voiced_ratio", "metric": "voiced_ratio", "dynamic_range": [0.0, 1.0], "target_step": 0.03, "bit_widths": [4, 5, 6, 8]},
            {"name": "level", "metric": "rms_db", "dynamic_range": [-80.0, 0.0], "target_step": 1.0, "bit_widths": [4, 6, 8, 10]},
        ],
        "similarity": {
            "threshold": 0.82,
            "weights": {
                "f0_mean": 2.4,
                "f0_stability": 0.6,
                "formant_f1": 1.6,
                "formant_f2": 1.8,
                "speech_rate": 0.5,
                "voiced_ratio": 0.4,
                "level": 0.2,
            },
        },
        "preview_budget_ms": 200.0,
        "max_workers": 4,
        "snapshot_commit_every": 1,
        "verify_incremental_vs_full": True,
        "initial_annotations": annotations,
    }


def _annotations(schedule: list[SynthSegment], file_name: str) -> list[dict[str, Any]]:
    return [
        {
            "file": file_name,
            "contains_ms": segment.midpoint_ms,
            "speaker": segment.speaker,
            "style_hint": segment.style,
        }
        for segment in schedule
    ]


def build_commands() -> list[dict[str, Any]]:
    return [
        {"id": "cmd-01-split", "op": "split", "target": {"file": "voice_dialogue.wav", "contains_ms": 430.0}, "at_ms": 550.0},
        {"id": "cmd-02-merge", "op": "merge", "target": {"file": "voice_dialogue.wav", "contains_ms": 300.0}, "direction": "next"},
        {"id": "cmd-03-annotation", "op": "set_speaker", "target": {"file": "voice_dialogue.wav", "contains_ms": 2200.0}, "speaker": "spk_noah"},
        {
            "id": "cmd-04-replace",
            "op": "replace",
            "target": {"file": "voice_dialogue.wav", "contains_ms": 3000.0},
            "interval": {"start_ms": 100.0, "end_ms": 350.0},
            "source": "assets/replacement_noah.wav",
        },
        {"id": "cmd-05-quant-4bit", "op": "set_quantization", "dimension": "f0_mean", "bit_width": 4},
        {"id": "cmd-06-quant-10bit", "op": "set_quantization", "dimension": "f0_mean", "bit_width": 10},
        {"id": "cmd-07-rate-4bit", "op": "set_quantization", "dimension": "speech_rate", "bit_width": 4},
    ]


def build_anomalies() -> dict[str, Any]:
    return {
        "description": "Deterministic anomaly fixtures for failure isolation and boundary diagnostics.",
        "items": [
            {
                "id": "a01-corrupt-riff",
                "kind": "corrupt_wav",
                "path": "anomalies/bad_riff.wav",
                "expected_code": "E_AUDIO_FORMAT",
                "expectation": "file enters failures and valid files continue",
            },
            {
                "id": "a02-truncated-data",
                "kind": "corrupt_wav",
                "path": "anomalies/truncated_data.wav",
                "expected_code": "E_AUDIO_FORMAT",
                "expectation": "RIFF data length exceeds file length",
            },
            {
                "id": "a03-nonfinite-sample",
                "kind": "non_finite_samples",
                "path": "anomalies/nonfinite_samples.json",
                "expected_code": "E_AUDIO_SAMPLE",
                "expectation": "validate_samples rejects NaN/Inf without contaminating caches",
            },
            {
                "id": "a04-quantization-out-of-range",
                "kind": "command",
                "path": "anomalies/quantization_out_of_range.json",
                "expected_code": "E_QUANTIZATION_LEVEL",
                "expectation": "invalid bit width is rejected and previous state is restored",
            },
            {
                "id": "a05-similarity-equal-threshold",
                "kind": "similarity_equal",
                "path": "anomalies/similarity_equal.json",
                "expected_code": None,
                "expectation": "set similarity.threshold to an adjacent score from report.segment_assignments, then verify inclusive decision, dual references, and start-frame ordering",
            },
        ],
    }


def generate_demo(root: str | Path, seed: int = 20261121) -> dict[str, Path]:
    target = Path(root)
    samples_dir = target / "samples"
    assets_dir = target / "assets"
    anomalies_dir = target / "anomalies"
    samples_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    anomalies_dir.mkdir(parents=True, exist_ok=True)

    dialogue = dialogue_schedule()
    solo = solo_schedule()
    write_wav(samples_dir / "voice_dialogue.wav", synthesize_file(dialogue, 3800.0, seed))
    write_wav(samples_dir / "voice_solo.wav", synthesize_file(solo, 3500.0, seed + 1))
    write_wav(assets_dir / "replacement_noah.wav", synthesize_replacement(250.0, seed + 2))

    annotations = _annotations(dialogue, "voice_dialogue.wav") + _annotations(solo, "voice_solo.wav")
    config = _base_config(annotations)
    commands = build_commands()
    (target / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "anomalies.json").write_text(json.dumps(build_anomalies(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (anomalies_dir / "bad_riff.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEthis-is-not-a-valid-chunk-stream")
    valid = (samples_dir / "voice_dialogue.wav").read_bytes()
    truncated = bytearray(valid[:80])
    truncated[4:8] = (len(truncated) - 8).to_bytes(4, "little")
    data_offset = truncated.find(b"data")
    if data_offset >= 0:
        truncated[data_offset + 4 : data_offset + 8] = (0xFFFFFF).to_bytes(4, "little")
    (anomalies_dir / "truncated_data.wav").write_bytes(bytes(truncated))
    (anomalies_dir / "nonfinite_samples.json").write_text(
        canonical_json({"samples": [0, 1, "NaN", "Infinity"]}) + "\n",
        encoding="utf-8",
    )
    (anomalies_dir / "quantization_out_of_range.json").write_text(
        json.dumps(
            {"id": "bad-quant", "op": "set_quantization", "dimension": "f0_mean", "bit_width": 99},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (anomalies_dir / "similarity_equal.json").write_text(
        json.dumps(
            {
                "description": "Set similarity.threshold to the measured adjacent score to exercise the inclusive boundary.",
                "source": "OUTPUT/report.json -> segment_assignments[].similarity",
                "config_patch": {"similarity": {"threshold": "<measured_similarity>"}},
                "expected": {"boundary": True, "inclusive_same": True, "exclusive_same": False},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "root": target,
        "samples": samples_dir,
        "config": target / "config.json",
        "commands": target / "commands.json",
        "anomalies": target / "anomalies.json",
    }
