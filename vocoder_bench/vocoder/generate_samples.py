"""Deterministic sample generator.

All randomness comes from a local, seeded LCG - no global RNG state is used.
The reference WAVs are produced by the *float64 phase control path* with the
same sinusoidal / overlap-add model used by the vocoder, so they act as a
waveform-level ground truth for phase diagnostics.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import yaml_lite
from .dsp import ENERGY_SCALE, overlap_add, synthesize_frame
from .fixedpoint import Float64Phase
from .hashing import canonical_hash
from .wav_io import write_wav

TWO_PI = 2.0 * math.pi


class Lcg:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.state

    def uniform(self, low: float, high: float) -> float:
        return low + (self.next_u64() / (1 << 64)) * (high - low)


def f0_contour(kind: str, frame_index: int, n_frames: int) -> float:
    phase = frame_index / max(n_frames - 1, 1)
    if kind == "rising":
        return 120.0 + 80.0 * phase
    if kind == "falling":
        return 220.0 - 90.0 * phase
    if kind == "vibrato":
        return 160.0 + 18.0 * math.sin(TWO_PI * 2.0 * phase)
    raise ValueError(kind)


def build_frames(utter_id: str, kind: str, n_frames: int,
                 harmonic_limit: int, seed: int) -> list[dict]:
    rng = Lcg(seed)
    frames = []
    for idx in range(n_frames):
        f0 = f0_contour(kind, idx, n_frames)
        amps = []
        for harmonic in range(harmonic_limit):
            decay = 1.0 / (harmonic + 1)
            amps.append(round(0.55 * decay * rng.uniform(0.75, 1.25), 6))
        gain = round(0.45 + 0.2 * math.sin(math.pi * idx / max(n_frames - 1, 1)), 6)
        frames.append({
            "frame_index": idx,
            "f0": round(f0, 4),
            "harmonic_amplitudes": amps,
            "energy_gain": gain,
        })
    return frames


def render_reference(utter_id: str, frames: list[dict], cfg: dict) -> np.ndarray:
    sr = cfg["sample_rate"]
    hop = int(round(cfg["frame_shift_ms"] * sr / 1000.0))
    length = int(round(cfg["frame_length_ms"] * sr / 1000.0))
    frame_fps = [canonical_hash(f) for f in frames]
    phase_model = Float64Phase()
    phases = phase_model.accumulate(
        frames, hop, sr, utter_id, frame_fps, cfg["harmonic_limit"]
    )
    time_frames = []
    for idx, frame in enumerate(frames):
        time_frames.append(
            synthesize_frame(
                np.asarray(frame["harmonic_amplitudes"], dtype=np.float64),
                phases[idx],
                frame["f0"],
                frame["energy_gain"],
                length,
                sr,
            )
        )
    wave_float = overlap_add(
        time_frames, hop, cfg["window_type"], cfg["normalization_rule"]
    )
    pcm = np.clip(np.rint(wave_float * ENERGY_SCALE), -32768, 32767).astype(np.int16)
    return pcm


def write_jsonl(path: Path, frames: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for frame in frames:
            fh.write(json.dumps(frame, sort_keys=True) + "\n")


def corrupt_wav(path: Path) -> None:
    raw = bytearray(path.read_bytes())
    raw[8:12] = b"DATA"
    path.write_bytes(bytes(raw))


def generate(output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "sample_rate": 16000,
        "frame_shift_ms": 10.0,
        "frame_length_ms": 20.0,
        "harmonic_limit": 12,
        "phase_bits": 24,
        "phase_quant_step": 1,
        "f0_ratio": 1.0,
        "energy_rounding": "half_up",
        "window_type": "hann",
        "normalization_rule": "cola",
        "max_workers": 2,
        "snapshot_interval": 8,
        "align_search_ms": 12.0,
        "interactive_budget_ms": 50.0,
        "spectral_tol": 0.25,
        "rmse_tol": 0.12,
        "ncc_tol": 0.9,
        "phase_attribution_threshold": 1.5,
    }
    yaml_lite.dump(cfg, out / "vocoder_config.yaml")

    utters = [
        ("utt_rising", "rising", 40, 101),
        ("utt_falling", "falling", 36, 202),
        ("utt_vibrato", "vibrato", 44, 303),
    ]
    manifest = []
    for utter_id, kind, n_frames, seed in utters:
        frames = build_frames(utter_id, kind, n_frames, cfg["harmonic_limit"], seed)
        write_jsonl(out / f"{utter_id}.frames.jsonl", frames)
        pcm = render_reference(utter_id, frames, cfg)
        write_wav(out / f"{utter_id}.wav", pcm, cfg["sample_rate"])
        manifest.append({
            "utterance_id": utter_id,
            "contour": kind,
            "frames": n_frames,
            "seed": seed,
            "wav": f"{utter_id}.wav",
            "frames_file": f"{utter_id}.frames.jsonl",
        })

    # Anomaly injection: bad RIFF container and invalid feature sequence.
    bad_frames = build_frames("utt_badfeat", "rising", 8, cfg["harmonic_limit"], 404)
    bad_frames[4]["frame_index"] = 3  # duplicate index: strict sequence violated
    write_jsonl(out / "utt_badfeat.frames.jsonl", bad_frames)
    bad_ref = build_frames("utt_badaudio", "falling", 8, 12, 505)
    write_jsonl(out / "utt_badaudio.frames.jsonl", bad_ref)
    bad_pcm = render_reference("utt_badaudio", bad_ref, cfg)
    write_wav(out / "utt_badaudio.wav", bad_pcm, cfg["sample_rate"])
    corrupt_wav(out / "utt_badaudio.wav")

    commands = [
        {"seq": 0, "type": "snapshot", "note": "baseline full run"},
        {"seq": 1, "type": "phase_bits", "value": 8},
        {"seq": 2, "type": "phase_bits", "value": 6},
        {"seq": 3, "type": "phase_quant_step", "value": 4},
        {"seq": 4, "type": "phase_bits", "value": 5},
        {"seq": 5, "type": "harmonic_limit", "value": 8},
        {"seq": 6, "type": "window_type", "value": "hamming"},
        {"seq": 7, "type": "energy_rounding", "value": "half_even"},
        {"seq": 8, "type": "f0_ratio", "value": 1.05},
        {"seq": 9, "type": "f0_ratio", "value": 1.0},
        {"seq": 10, "type": "window_type", "value": "hann"},
        {"seq": 11, "type": "energy_rounding", "value": "half_up"},
        {"seq": 12, "type": "harmonic_limit", "value": 12},
        {"seq": 13, "type": "phase_quant_step", "value": 1},
        {"seq": 14, "type": "phase_bits", "value": 24},
        {"seq": 15, "type": "shutdown", "note": "graceful exit"},
    ]
    (out / "edit_commands.json").write_text(
        json.dumps(commands, indent=2, sort_keys=True), encoding="utf-8"
    )

    anomaly = {
        "injections": [
            {"utterance_id": "utt_badaudio", "file": "utt_badaudio.wav",
             "fault": "riff_container_corrupt", "expected_code": "E_AUDIO_FORMAT"},
            {"utterance_id": "utt_badfeat", "file": "utt_badfeat.frames.jsonl",
             "fault": "frame_index_duplicate_gap", "expected_code": "E_FEATURE_INVALID"},
        ]
    }
    (out / "anomaly_injection.json").write_text(
        json.dumps(anomaly, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "sample_manifest.json").write_text(
        json.dumps({"utterances": manifest, "config": "vocoder_config.yaml"},
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"out": str(out), "utterances": manifest}
