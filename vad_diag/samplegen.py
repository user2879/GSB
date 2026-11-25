"""Deterministic local sample generation (no network, no real recordings).

A fixed integer seed drives a NumPy ``RandomState`` that synthesizes a
16 kHz / mono / 16-bit PCM signal alternating between low-energy noise
(silence-like) and amplitude-modulated tonal bursts (speech-like).  The signal
is split into fixed-size segments, each written as a standards-compliant WAV
file and described in a segment manifest with capture / arrival / process
timestamps.  An optional permutation makes the *arrival* order differ from the
*capture* order to exercise late and out-of-order delivery.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np

from .segments import write_wav

DEFAULT_SEED = 20261126
DEFAULT_RATE = 16000
DEFAULT_SEGMENT_SAMPLES = 3200  # 200 ms per segment


def synthesize(seed: int = DEFAULT_SEED,
               total_samples: int = 16000 * 4,
               rate: int = DEFAULT_RATE) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(total_samples, dtype=np.float64) / rate
    # Four alternating 1-second blocks: silence, speech, silence, speech.
    signal = np.zeros(total_samples, dtype=np.float64)
    block = rate
    for b in range(total_samples // block):
        lo, hi = b * block, (b + 1) * block
        if b % 2 == 1:  # speech-like blocks
            fundamental = 180.0 + 40.0 * (b // 2)
            tone = (0.55 * np.sin(2 * math.pi * fundamental * t[lo:hi])
                    + 0.25 * np.sin(2 * math.pi * fundamental * 2 *
                                    t[lo:hi])
                    + 0.12 * np.sin(2 * math.pi * fundamental * 3 *
                                    t[lo:hi]))
            envelope = 0.5 + 0.5 * np.sin(
                2 * math.pi * 6.0 * t[lo:hi] + b)
            signal[lo:hi] = tone * (0.6 + 0.4 * envelope)
        # Low-energy background noise in every block.
        signal[lo:hi] += rng.normal(0.0, 0.012, size=block)
    # Deterministic soft limiter then 16-bit quantization.
    signal = np.tanh(signal)
    pcm = np.clip(np.round(signal * 32767.0), -32768, 32767).astype(np.int16)
    return pcm


def build_segments(seed: int = DEFAULT_SEED,
                   segment_count: int = 20,
                   segment_samples: int = DEFAULT_SEGMENT_SAMPLES,
                   rate: int = DEFAULT_RATE,
                   arrival_perm: bool = True
                   ) -> Tuple[List[Dict], List[str]]:
    total = segment_count * segment_samples
    pcm = synthesize(seed=seed, total_samples=total, rate=rate)
    manifest_entries: List[Dict] = []
    ids: List[str] = []
    for i in range(segment_count):
        seg_id = f"S{i:04d}"
        ids.append(seg_id)
        lo, hi = i * segment_samples, (i + 1) * segment_samples
        samples = [int(v) for v in pcm[lo:hi]]
        capture_ms = int(round(lo / rate * 1000))
        # Arrival is generally after capture, but may be reordered.
        arrival_ms = capture_ms + 25
        process_ms = 5
        manifest_entries.append({
            "segment_id": seg_id,
            "capture_ms": capture_ms,
            "arrival_ms": arrival_ms,
            "process_ms": process_ms,
            "samples": samples,
        })
    if arrival_perm:
        # Fixed deterministic permutation: swap a couple of adjacent arrivals.
        order = list(range(segment_count))
        for a, b in ((2, 3), (7, 8), (13, 12)):
            if b < segment_count:
                order[a], order[b] = order[b], order[a]
        permuted = [None] * segment_count
        for new_pos, old_idx in enumerate(order):
            entry = manifest_entries[old_idx]
            entry = dict(entry)
            entry["arrival_ms"] = new_pos * 200 + 25
            permuted[new_pos] = entry
        manifest_entries = permuted
    return manifest_entries, ids


def write_dataset(out_dir: str, seed: int = DEFAULT_SEED,
                  segment_count: int = 20,
                  segment_samples: int = DEFAULT_SEGMENT_SAMPLES,
                  rate: int = DEFAULT_RATE,
                  write_wav_files: bool = True) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    wav_dir = os.path.join(out_dir, "wav")
    if write_wav_files:
        os.makedirs(wav_dir, exist_ok=True)
    entries, ids = build_segments(
        seed=seed, segment_count=segment_count,
        segment_samples=segment_samples, rate=rate, arrival_perm=True)
    manifest_entries = []
    for entry in entries:
        out = dict(entry)
        if write_wav_files:
            wav_name = entry["segment_id"] + ".wav"
            write_wav(os.path.join(wav_dir, wav_name), entry["samples"], rate)
            out.pop("samples", None)
            out["wav"] = os.path.join("wav", wav_name)
        manifest_entries.append(out)
    manifest = {
        "sample_rate": rate,
        "channels": 1,
        "sample_width_bytes": 2,
        "seed": seed,
        "segment_count": segment_count,
        "segment_samples": segment_samples,
        "segments": manifest_entries,
    }
    manifest_path = os.path.join(out_dir, "segments_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
    return {"manifest_path": manifest_path, "segment_ids": ids,
            "entries": entries}
