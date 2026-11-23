"""Generate deterministic 16kHz mono 16bit PCM WAV samples and config files.

Each utterance concatenates harmonic speech-like segments (driven by a
controlled word sequence), silence gaps and stationary noise. All randomness
comes from a seeded numpy Generator so outputs are byte-identical per run.
"""

import argparse
import json
import os
import wave

import numpy as np

SR = 16000
VOCAB = ["a", "b", "c", "d", "e"]
PHONEMES = ["P_A", "P_B", "P_C", "P_D", "P_E"]
MEL_BANDS = 6


def _write_wav(path, samples):
    pcm = np.clip(np.rint(samples), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(pcm.tobytes())


def _harmonic_segment(duration_ms, base_freq, rng, amplitude=0.35):
    n = SR * duration_ms // 1000
    t = np.arange(n, dtype=np.float64) / SR
    signal = np.zeros(n, dtype=np.float64)
    for harmonic in range(1, 4):
        signal += (amplitude / harmonic) * np.sin(
            2 * np.pi * base_freq * harmonic * t + rng.uniform(0, 2 * np.pi)
        )
    envelope = 0.5 - 0.5 * np.cos(2 * np.pi * np.linspace(0, 1, n))
    return signal * envelope * 32767.0


def _silence(duration_ms):
    return np.zeros(SR * duration_ms // 1000, dtype=np.float64)


def _noise(duration_ms, rng, amplitude=0.03):
    n = SR * duration_ms // 1000
    return rng.normal(0.0, amplitude, size=n) * 32767.0


def synthesize(seed, word_sequence):
    rng = np.random.default_rng(seed)
    pieces = []
    for idx, char in enumerate(word_sequence):
        phoneme_index = VOCAB.index(char)
        base_freq = 180.0 + phoneme_index * 70.0
        pieces.append(_harmonic_segment(220, base_freq, rng))
        if idx % 2 == 0:
            pieces.append(_silence(120))
        else:
            pieces.append(_noise(120, rng))
    pieces.append(_noise(200, rng))
    return np.concatenate(pieces).astype(np.float64)


def build_score_config():
    rng = np.random.default_rng(2026)
    table = []
    for band in range(MEL_BANDS):
        row = []
        for char_idx in range(len(VOCAB)):
            # Each Mel band favors one character, giving a learnable emission map.
            peak = -2.0 if char_idx == band % len(VOCAB) else -8.0
            row.append(round(float(rng.normal(peak, 0.25)), 4))
        table.append(row)

    bigram = {}
    unigram = {}
    for idx, char in enumerate(VOCAB):
        unigram[char] = round(1.0 / len(VOCAB) + 0.01 * idx, 5)
        probs = {}
        weights = rng.uniform(0.5, 1.5, size=len(VOCAB))
        for next_idx, next_char in enumerate(VOCAB):
            probs[next_char] = round(float(weights[next_idx] / weights.sum()), 5)
        bigram[char] = probs
    return {
        "frame_length_ms": 25,
        "frame_shift_ms": 10,
        "mel_bands": MEL_BANDS,
        "vocab": VOCAB,
        "emission_table": table,
        "bigram_probs": bigram,
        "unigram_probs": unigram,
        "backoff_weight": -0.8,
        "log_smooth": -12.0,
        "significant_digits": 6,
    }


def build_schedule_config():
    return {
        "max_concurrency": 4,
        "global_point_budget": 4000,
        "shard_point_quota": 900,
        "shards_per_batch": 3,
        "snapshot_commit_shards": 2,
        "quant_mode": "fp64",
        "quant_frac_bits": 16,
        "shard_duration_ms": 600,
        "overlap_ms": 100,
    }


def build_injections(audio_name):
    return [
        {"id": "INJ-01", "type": "truncate_wav", "audio": audio_name,
         "shard_index": 0, "keep_ratio": 0.55},
        {"id": "INJ-02", "type": "non_finite_samples", "audio": audio_name,
         "shard_index": 1, "sample_index": 17},
        {"id": "INJ-03", "type": "emission_dim_mismatch", "audio": audio_name,
         "shard_index": 2},
        {"id": "INJ-04", "type": "quota_below_min_points", "audio": audio_name,
         "shard_index": 3, "forced_allowance": 1},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    wav_dir = os.path.join(args.out, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    sequences = {
        "utter_001.wav": list("abcdeabc"),
        "utter_002.wav": list("edcbaedc"),
        "utter_003.wav": list("aabbccddee"),
        "utter_004.wav": list("ceadbead"),
    }
    ground_truth = {}
    for idx, (name, sequence) in enumerate(sequences.items()):
        samples = synthesize(1000 + idx, sequence)
        _write_wav(os.path.join(wav_dir, name), samples)
        ground_truth[name] = "".join(sequence)

    with open(os.path.join(args.out, "phoneme_map.json"), "w", encoding="utf-8") as handle:
        json.dump(dict(zip(PHONEMES, VOCAB)), handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "score_params.json"), "w", encoding="utf-8") as handle:
        json.dump(build_score_config(), handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "schedule.json"), "w", encoding="utf-8") as handle:
        json.dump(build_schedule_config(), handle, indent=2)
    with open(os.path.join(args.out, "injections.example.json"), "w", encoding="utf-8") as handle:
        json.dump(build_injections("utter_004.wav"), handle, indent=2)
    with open(os.path.join(args.out, "ground_truth.json"), "w", encoding="utf-8") as handle:
        json.dump(ground_truth, handle, ensure_ascii=False, indent=2)
    print(f"generated {len(sequences)} WAV files and configs in {args.out}")


if __name__ == "__main__":
    main()
