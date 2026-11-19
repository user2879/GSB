#!/usr/bin/env python3
"""Generate the deterministic 16 kHz PCM sample set shipped with the project."""

from __future__ import annotations

import hashlib
import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
AUDIO = SAMPLES / "audio"
REPLACEMENTS = SAMPLES / "replacements"
SAMPLE_RATE = 16000


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def stable_seed(name: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def synthesize(
    name: str,
    output: Path,
    *,
    duration_s: float,
    f0_mean_hz: float,
    f0_std_hz: float,
    energy_dbfs: float,
    dynamic_range_db: float,
    speech_rate_sps: float,
    noise_floor_dbfs: float,
    seed: int,
) -> dict[str, Any]:
    count = int(round(duration_s * SAMPLE_RATE))
    time = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(stable_seed(name, seed))

    vibrato = np.sin(2.0 * math.pi * 0.73 * time + 0.4)
    drift = np.sin(2.0 * math.pi * 0.19 * time + 1.1)
    f0 = f0_mean_hz + f0_std_hz * (0.78 * vibrato + 0.22 * drift)
    f0 = np.maximum(f0, 45.0)
    phase = 2.0 * math.pi * np.cumsum(f0) / SAMPLE_RATE
    harmonic = np.zeros(count, dtype=np.float64)
    for harmonic_index in range(1, 7):
        harmonic += np.sin(phase * harmonic_index) / (harmonic_index**1.18)

    syllable = 0.5 + 0.5 * np.sin(2.0 * math.pi * speech_rate_sps * time + 0.35 * np.sin(2.0 * math.pi * 1.4 * time))
    envelope_shape = np.power(np.clip(syllable, 0.0, 1.0), 0.8)
    gain_db = -dynamic_range_db * (1.0 - envelope_shape)
    envelope = np.power(10.0, gain_db / 20.0) * (0.28 + 0.72 * envelope_shape)
    fade_samples = min(int(0.025 * SAMPLE_RATE), count // 4)
    if fade_samples > 0:
        fade = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, fade_samples))
        envelope[:fade_samples] *= fade
        envelope[-fade_samples:] *= fade[::-1]

    signal = harmonic * envelope
    current_rms = math.sqrt(float(np.mean(signal * signal)))
    target_rms = 10.0 ** (energy_dbfs / 20.0)
    signal *= target_rms / max(current_rms, 1e-12)
    noise_rms = 10.0 ** (noise_floor_dbfs / 20.0)
    signal += rng.normal(0.0, noise_rms, size=count)
    peak = float(np.max(np.abs(signal)))
    if peak > 0.98:
        signal *= 0.98 / peak
    pcm = np.clip(np.rint(signal * 32767.0), -32768, 32767).astype("<i2")
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return {
        "file": output.relative_to(SAMPLES).as_posix(),
        "sample_count": count,
        "duration_ms": count / SAMPLE_RATE * 1000.0,
        "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
        "targets": {
            "f0_mean_hz": f0_mean_hz,
            "f0_std_hz": f0_std_hz,
            "energy_dbfs": energy_dbfs,
            "dynamic_range_db": dynamic_range_db,
            "speech_rate_sps": speech_rate_sps,
            "noise_floor_dbfs": noise_floor_dbfs,
        },
    }


TURN_SPECS = [
    ("t0001", "conv-a", 1, "spk_a", "neutral", 1.37, 126.0, 13.0, -21.0, 13.0, 3.8, 0.0, "opening"),
    ("t0002", "conv-a", 2, "spk_b", "happy", 1.19, 218.0, 24.0, -18.5, 16.0, 4.9, 120.0, "response"),
    ("t0003", "conv-a", 3, "spk_a", "angry", 1.43, 151.0, 37.0, -16.5, 22.0, 5.1, 80.0, "response"),
    ("t0004", "conv-a", 4, "spk_b", "sad", 1.11, 194.0, 9.0, -27.0, 9.0, 2.7, 420.0, "continuation"),
    ("t0005", "conv-a", 5, "spk_a", "neutral", 1.31, 124.0, 11.0, -22.0, 12.0, 3.6, 95.0, "response"),
    ("t0006", "conv-b", 6, "spk_c", "neutral", 1.22, 166.0, 10.0, -23.0, 11.0, 3.5, 0.0, "opening"),
    ("t0007", "conv-b", 7, "spk_b", "happy", 1.03, 224.0, 21.0, -19.0, 15.0, 4.7, 105.0, "response"),
    ("t0008", "conv-b", 8, "spk_c", "sad", 1.35, 158.0, 8.0, -28.0, 8.0, 2.5, 115.0, "continuation"),
    ("t0009", "conv-b", 9, "spk_a", "angry", 1.09, 138.0, 31.0, -17.0, 20.0, 4.8, 230.0, "response"),
]


def main() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    REPLACEMENTS.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    for turn_id, *_rest in TURN_SPECS:
        conversation, turn_index, speaker, emotion, duration, f0_mean, f0_std, energy, dynamic, rate, gap, link = _rest
        metadata.append(
            synthesize(
                f"audio/{turn_id}.wav",
                AUDIO / f"{turn_id}.wav",
                duration_s=duration,
                f0_mean_hz=f0_mean,
                f0_std_hz=f0_std,
                energy_dbfs=energy,
                dynamic_range_db=dynamic,
                speech_rate_sps=rate,
                noise_floor_dbfs=-58.0,
                seed=20261120,
            )
        )
        metadata[-1].update(
            {
                "turn_id": turn_id,
                "conversation_id": conversation,
                "turn_index": turn_index,
                "speaker_id": speaker,
                "emotion_intent": emotion,
                "gap_ms": gap,
                "link_type": link,
            }
        )

    metadata.append(
        synthesize(
            "replacements/t0003_v2.wav",
            REPLACEMENTS / "t0003_v2.wav",
            duration_s=1.47,
            f0_mean_hz=285.0,
            f0_std_hz=42.0,
            energy_dbfs=-14.5,
            dynamic_range_db=25.0,
            speech_rate_sps=5.7,
            noise_floor_dbfs=-58.0,
            seed=20261121,
        )
    )
    metadata[-1].update({"turn_id": "t0003", "role": "replacement", "emotion_intent": "angry"})
    metadata.append(
        synthesize(
            "audio/t0010.wav",
            AUDIO / "t0010.wav",
            duration_s=1.28,
            f0_mean_hz=171.0,
            f0_std_hz=15.0,
            energy_dbfs=-21.0,
            dynamic_range_db=13.0,
            speech_rate_sps=3.9,
            noise_floor_dbfs=-58.0,
            seed=20261122,
        )
    )
    metadata[-1].update({"turn_id": "t0010", "role": "appended", "conversation_id": "conv-b", "speaker_id": "spk_c"})

    turns = []
    previous_by_conversation: dict[str, str | None] = {}
    for spec in TURN_SPECS:
        turn_id, conversation, turn_index, speaker, emotion, duration, f0_mean, f0_std, energy, dynamic, rate, gap, link = spec
        candidates = ["spk_a", "spk_b"] if conversation == "conv-a" else ["spk_b", "spk_c"]
        if speaker not in candidates:
            candidates.append(speaker)
        turns.append(
            {
                "turn_id": turn_id,
                "turn_index": turn_index,
                "conversation_id": conversation,
                "audio": f"audio/{turn_id}.wav",
                "speaker_id": speaker,
                "speaker_candidates": sorted(set(candidates)),
                "previous_turn_id": previous_by_conversation.get(conversation),
                "gap_ms": gap,
                "link_type": link,
                "synthetic_intent": emotion,
            }
        )
        previous_by_conversation[conversation] = turn_id
    write_json(
        SAMPLES / "dialogue.json",
        {"schema_version": 1, "dialogue_id": "deterministic-demo-dialogue", "turns": turns},
    )

    config = {
        "schema_version": 1,
        "frame": {
            "sample_rate": 16000,
            "length_ms": 32,
            "shift_ms": 16,
            "pitch_min_hz": 70,
            "pitch_max_hz": 360,
            "voiced_correlation": 0.52,
            "silence_floor_dbfs": -52.0,
            "onset_margin_db": 7.0,
            "min_onset_gap_ms": 70,
        },
        "evidence": {
            "emotions": ["angry", "happy", "neutral", "sad"],
            "feature_ranges": {
                "f0_mean": [80.0, 300.0],
                "f0_std": [0.0, 55.0],
                "energy_mean": [-40.0, -12.0],
                "energy_dynamic": [2.0, 32.0],
                "speech_rate": [1.5, 7.0],
                "rhythm_variability": [0.0, 1.5],
            },
            "initial_weights": {
                "angry": {"f0_mean": 0.03, "f0_std": 0.36, "energy_mean": 0.28, "energy_dynamic": 0.20, "speech_rate": 0.08, "rhythm_variability": 0.05},
                "happy": {"f0_mean": 0.31, "f0_std": 0.10, "energy_mean": 0.24, "energy_dynamic": 0.10, "speech_rate": 0.19, "rhythm_variability": 0.06},
                "neutral": {"f0_mean": 0.05, "f0_std": 0.08, "energy_mean": 0.09, "energy_dynamic": 0.04, "speech_rate": 0.09, "rhythm_variability": 0.65},
                "sad": {"f0_mean": 0.02, "f0_std": 0.04, "energy_mean": 0.05, "energy_dynamic": 0.27, "speech_rate": 0.04, "rhythm_variability": 0.58},
            },
            "normalization": "stable_softmax",
        },
        "calibration": {
            "gain": 0.37,
            "decay": 0.81,
            "same_speaker_factor": 1.0,
            "different_speaker_factor": 0.73,
            "speaker_switch_penalty": 0.04,
        },
        "attribution": {
            "min_segment_ms": 180.0,
            "min_silence_ms": 220.0,
            "max_gap_turns": 1,
            "score_weights": {"emotion_affinity": 0.28, "pitch_match": 0.30, "script_prior": 0.42},
            "speaker_profiles": {
                "spk_a": {
                    "pitch_center_hz": 123.0,
                    "pitch_tolerance_hz": 42.0,
                    "emotion_affinity": {"angry": 0.72, "happy": 0.10, "neutral": 0.46, "sad": 0.30},
                },
                "spk_b": {
                    "pitch_center_hz": 211.0,
                    "pitch_tolerance_hz": 48.0,
                    "emotion_affinity": {"angry": 0.22, "happy": 0.84, "neutral": 0.31, "sad": 0.40},
                },
                "spk_c": {
                    "pitch_center_hz": 165.0,
                    "pitch_tolerance_hz": 37.0,
                    "emotion_affinity": {"angry": 0.18, "happy": 0.24, "neutral": 0.66, "sad": 0.62},
                },
            },
        },
        "fixed_point": {"decimal_places": 6, "bit_width": 32, "quantization_levels": 10001},
        "concurrency": 4,
        "snapshot_commit_interval": 2,
        "batch_budget": 3,
    }
    write_json(SAMPLES / "calibration.json", config)

    edits = {
        "schema_version": 1,
        "batches": [
            {
                "batch_id": "round-1-audio-and-gain",
                "commands": [
                    {"op": "replace_audio", "turn_id": "t0003", "audio": "replacements/t0003_v2.wav"},
                    {"op": "adjust_gain", "turn_id": "t0005", "gain": 0.52},
                ],
            },
            {
                "batch_id": "round-2-speaker-and-structure",
                "commands": [
                    {"op": "modify_speaker", "turn_id": "t0006", "speaker_id": "spk_b", "speaker_candidates": ["spk_b", "spk_c"]},
                    {"op": "delete_turn", "turn_id": "t0007"},
                    {
                        "op": "append_turn",
                        "turn": {
                            "turn_id": "t0010",
                            "turn_index": 10,
                            "conversation_id": "conv-b",
                            "audio": "audio/t0010.wav",
                            "speaker_id": "spk_c",
                            "speaker_candidates": ["spk_c", "spk_a"],
                            "previous_turn_id": "t0009",
                            "gap_ms": 210.0,
                            "link_type": "continuation",
                        },
                    },
                ],
            },
        ],
    }
    write_json(SAMPLES / "edits.json", edits)
    write_json(SAMPLES / "no_edits.json", {"schema_version": 1, "batches": []})

    injections = {
        "schema_version": 1,
        "injections": [
            {"injection_id": "INJ-AUDIO-001", "turn_id": "t0004", "stage": "extract", "kind": "corrupt_format", "enabled": True, "message": "truncate RIFF payload before parsing"},
            {"injection_id": "INJ-CAL-001", "turn_id": "t0006", "stage": "calibration", "kind": "force_exception", "enabled": True, "message": "deterministic calibration fault"},
        ],
    }
    write_json(SAMPLES / "injections_validation.json", injections)
    write_json(
        SAMPLES / "expected_business_codes.json",
        {
            "schema_version": 1,
            "codes": {
                "E_AUDIO_FORMAT": "RIFF/WAVE container is syntactically invalid or not 16 kHz mono 16-bit PCM.",
                "E_AUDIO_MISSING": "A referenced audio file is absent.",
                "E_CONFIG_INVALID": "Calibration or injection configuration is invalid.",
                "E_CONFIG_MISMATCH": "Snapshot version or base configuration hash does not match the runtime.",
                "E_CALIB_BUDGET_PENDING": "The per-batch turn budget was exhausted; unresolved turns remain queued.",
                "E_INJECTED_FAULT": "An enabled deterministic injection fault was raised.",
                "E_SEGMENT_TOO_SHORT": "A segment violates attribution.min_segment_ms.",
                "E_DIALOGUE_INVALID": "Dialogue schema or turn linkage is invalid.",
                "E_EDIT_INVALID": "An edit command is malformed or cannot be applied.",
                "E_SNAPSHOT_INVALID": "Snapshot files are missing or unreadable.",
                "E_INTERNAL": "An unexpected stage exception was isolated and reported.",
            },
            "injection_expectations": [
                {"injection_id": "INJ-AUDIO-001", "turn_id": "t0004", "stage": "extract", "expected_code": "E_AUDIO_FORMAT"},
                {"injection_id": "INJ-CAL-001", "turn_id": "t0006", "stage": "calibration", "expected_code": "E_INJECTED_FAULT"},
            ],
        },
    )
    write_json(
        SAMPLES / "generation_manifest.json",
        {
            "schema_version": 1,
            "seed": 20261120,
            "sample_rate": SAMPLE_RATE,
            "encoding": "pcm_s16le",
            "noise_floor_dbfs": -58.0,
            "artifacts": metadata,
        },
    )
    print(f"generated {len(metadata)} WAV files and inputs under {SAMPLES}")


if __name__ == "__main__":
    main()
