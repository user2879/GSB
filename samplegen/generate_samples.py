"""Generate deterministic 16 kHz mono 16-bit PCM samples and config files.

The generator is the only component allowed to use a seeded RNG. Audio is a
sequence of phoneme-driven harmonic tones plus stationary noise; every frame
stays close to one phoneme centroid so the deterministic decoder reproduces
the intended word sequence.
"""

from __future__ import annotations

import argparse
import json
import os
import wave
from typing import Dict, List, Tuple

import numpy as np

SAMPLE_RATE = 16000

# phoneme -> (word, base frequency)
VOCAB: Dict[str, Tuple[str, int]] = {
    "p_a": ("a", 220),
    "p_b": ("bo", 277),
    "p_c": ("ci", 349),
    "p_d": ("duo", 440),
    "p_e": ("e", 554),
    "p_f": ("fo", 698),
    "p_g": ("ge", 880),
    "p_h": ("hao", 1109),
}

# word sequences used to synthesize each clip (phoneme labels)
SEQUENCES: Dict[str, List[str]] = {
    "clip_01": ["p_a", "p_b", "p_c", "p_d", "p_e", "p_f", "p_g", "p_h",
                "p_a", "p_c", "p_e", "p_g", "p_b", "p_d", "p_f", "p_h"],
    "clip_02": ["p_h", "p_g", "p_f", "p_e", "p_d", "p_c", "p_b", "p_a",
                "p_h", "p_f", "p_d", "p_b", "p_g", "p_e", "p_c", "p_a"],
    "clip_03": ["p_a", "p_a", "p_b", "p_b", "p_c", "p_d", "p_d", "p_e",
                "p_f", "p_g", "p_g", "p_h", "p_e", "p_c", "p_b", "p_a"],
}


def harmonic_tone(frequency: int, n_samples: int, rng: np.random.Generator,
                  amplitude: float = 0.35) -> np.ndarray:
    timeline = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
    tone = np.zeros(n_samples, dtype=np.float64)
    for harmonic in (1, 2, 3):
        phase = rng.uniform(0.0, 2 * np.pi)
        tone += (amplitude / harmonic) * np.sin(
            2 * np.pi * frequency * harmonic * timeline + phase)
    envelope = np.minimum(1.0, np.linspace(0.0, 8.0, n_samples))
    envelope *= np.minimum(1.0, np.linspace(8.0, 0.0, n_samples))
    return tone * envelope


def stationary_noise(n_samples: int, rng: np.random.Generator,
                     scale: float = 0.01) -> np.ndarray:
    return rng.normal(0.0, scale, size=n_samples)


def synthesize(sequence: List[str], unit_ms: int, gap_ms: int,
                seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unit_len = SAMPLE_RATE * unit_ms // 1000
    gap_len = SAMPLE_RATE * gap_ms // 1000
    pieces: List[np.ndarray] = []
    for phoneme in sequence:
        frequency = VOCAB[phoneme][1]
        tone = harmonic_tone(frequency, unit_len, rng)
        noise = stationary_noise(unit_len, rng)
        pieces.append(tone + noise)
        pieces.append(stationary_noise(gap_len, rng))
    return np.concatenate(pieces).astype(np.float64)


def write_wav(path: str, samples: np.ndarray) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def build_centroids(mel_bands: int, unit_ms: int, gap_ms: int) -> Dict[str, List[float]]:
    """Estimate phoneme centroids empirically from known synthesis segments.

    Uses the exact same mel pipeline as the application, so the deterministic
    nearest-centroid frame classifier recovers the intended phoneme sequence.
    """
    from asr_stream_editor.audio import plan_frames
    from asr_stream_editor.features import log_mel_energies

    phonemes = sorted(VOCAB)
    accumulator: Dict[str, List[np.ndarray]] = {phoneme: [] for phoneme in phonemes}
    unit_frames = unit_ms // 10
    gap_frames = gap_ms // 10
    for clip_index, (_, sequence) in enumerate(SEQUENCES.items()):
        samples = synthesize(sequence, unit_ms, gap_ms, seed=20260101 + clip_index)
        plan = plan_frames(samples.shape[0], 25, 10)
        log_mel = log_mel_energies(samples, plan, mel_bands)
        for index, phoneme in enumerate(sequence):
            start = index * (unit_frames + gap_frames) + 2
            end = index * (unit_frames + gap_frames) + unit_frames
            accumulator[phoneme].append(log_mel[start:end].mean(axis=0))
    centroids: Dict[str, List[float]] = {}
    for phoneme in phonemes:
        centroid = np.mean(np.stack(accumulator[phoneme]), axis=0)
        centroids[phoneme] = [round(float(value), 6) for value in centroid]
    return centroids


def build_emission_table(phonemes: List[str]) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    for phoneme in phonemes:
        rows: Dict[str, float] = {}
        for other in phonemes:
            word = VOCAB[other][0]
            if other == phoneme:
                rows[word] = -0.1
            else:
                # leave two distractors explicit; remaining words are "missing"
                # and must be filled with the log-domain smoothing value
                if (phoneme_index(phonemes, other) - phoneme_index(phonemes, phoneme)) % len(phonemes) in (1, 2):
                    rows[word] = -6.0
        table[phoneme] = rows
    return table


def phoneme_index(phonemes: List[str], phoneme: str) -> int:
    return phonemes.index(phoneme)


def build_language_model(phonemes: List[str], seed: int) -> Tuple[Dict[str, float],
                                                                    Dict[str, float]]:
    rng = np.random.default_rng(seed)
    words = [VOCAB[phoneme][0] for phoneme in phonemes]
    bigram: Dict[str, float] = {}
    for left in words:
        weights = rng.uniform(0.2, 1.0, size=len(words))
        # encourage staying / moving smoothly rather than hard deterministic bias
        weights = weights / weights.sum()
        for right, probability in zip(words, weights):
            bigram[f"{left}|{right}"] = round(float(probability), 6)
    backoff = {word: round(float(rng.uniform(0.05, 0.3)), 6) for word in words}
    return bigram, backoff


def write_json(path: str, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)


def generate(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    replacement_dir = os.path.join(output_dir, "replacements")
    os.makedirs(replacement_dir, exist_ok=True)
    phonemes = sorted(VOCAB)
    for clip_index, (name, sequence) in enumerate(SEQUENCES.items()):
        samples = synthesize(sequence, unit_ms=160, gap_ms=40,
                             seed=20260101 + clip_index)
        write_wav(os.path.join(output_dir, f"{name}.wav"), samples)
    # replacement audio for clip_01 chunk 2: another deterministic tone pattern
    replacement_seed = 20261120
    rng = np.random.default_rng(replacement_seed)
    replacement_ms = 240  # equals chunk_ms - overlap_ms (ownership region)
    replacement = harmonic_tone(VOCAB["p_e"][1],
                                SAMPLE_RATE * replacement_ms // 1000, rng)
    replacement += stationary_noise(replacement.shape[0], rng)
    write_wav(os.path.join(replacement_dir, "replacement_clip01_chunk02.wav"),
              replacement)

    write_json(os.path.join(output_dir, "phoneme_map.json"), {
        "version": 1,
        "mapping": {phoneme: VOCAB[phoneme][0] for phoneme in phonemes},
    })
    centroids = build_centroids(mel_bands=12, unit_ms=160, gap_ms=40)
    bigram, backoff = build_language_model(phonemes, seed=4242)
    write_json(os.path.join(output_dir, "acoustic_params.json"), {
        "frame_length_ms": 25,
        "frame_shift_ms": 10,
        "mel_bands": 12,
        "emission_table": build_emission_table(phonemes),
        "bigram": bigram,
        "backoff": backoff,
        "log_smooth": -12.0,
        "centroids": centroids,
    })
    write_json(os.path.join(output_dir, "stream_config.json"), {
        "chunk_ms": 320,
        "overlap_ms": 80,
        "min_consistent_frames": 3,
        "redecode_budget_ms": 20,
        "concurrency": 4,
        "concurrency_limit": 8,
        "snapshot_commit_interval_chunks": 4,
        "precision": "medium",
        "simulated_task_cost_ms": 1,
    })
    write_json(os.path.join(output_dir, "command_stream.json"), [
        {"command": "initial_decode"},
        {"command": "replace_chunk", "chunk_index": 2,
         "replacement_wav": "replacements/replacement_clip01_chunk02.wav"},
        {"command": "set_overlap", "overlap_ms": 120},
        {"command": "set_stable_frames", "min_consistent_frames": 4},
        {"command": "set_backoff", "backoff_scale": 0.5},
        {"command": "resume"},
        {"command": "finalize"},
    ])
    write_json(os.path.join(output_dir, "fault_injection.example.json"), {
        "fail_chunks": {
            "3": {"stage": "decode", "message": "injected decode failure"}
        },
        "crash_chunks": {
            "5": {"stage": "feature", "exit_code": 9}
        },
        "exit_after_chunk": 6
    })
    # deliberately malformed WAV to exercise E_AUDIO_FORMAT
    with open(os.path.join(output_dir, "broken_sample.wav"), "wb") as handle:
        handle.write(b"NOT_RIFF_DATA")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="samples")
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
