"""Generate fixed-seed 16 kHz mono PCM WAVs, config and command streams."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import wave
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_RATE = 16000
SEED = 20261121


def _write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -32768, 32767).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(clipped.tobytes())


def _voice(
    *,
    duration_seconds: float,
    base_f0: float,
    vibrato_hz: float,
    vibrato_depth: float,
    f0_trend: float,
    formants: tuple[tuple[float, float, float], ...],
    noise_amplitude: float,
    seed: int,
    phase_offset: float = 0.0,
) -> np.ndarray:
    count = int(round(duration_seconds * SAMPLE_RATE))
    t = np.arange(count, dtype=np.float64) / SAMPLE_RATE
    f0 = base_f0 + f0_trend * (t / max(duration_seconds, 1e-9)) + vibrato_depth * np.sin(
        2.0 * math.pi * vibrato_hz * t + phase_offset
    )
    instantaneous_phase = 2.0 * math.pi * np.cumsum(f0) / SAMPLE_RATE
    waveform = np.zeros(count, dtype=np.float64)
    max_harmonic = int(5000.0 / max(base_f0 - vibrato_depth, 1.0))
    for harmonic in range(1, max_harmonic + 1):
        frequency = harmonic * base_f0
        amplitude = 0.0
        for center, bandwidth, weight in formants:
            amplitude += weight * math.exp(-0.5 * ((frequency - center) / bandwidth) ** 2)
        amplitude = amplitude / (harmonic**0.72)
        if amplitude < 1e-5:
            continue
        waveform += amplitude * np.sin(harmonic * instantaneous_phase + 0.19 * harmonic * harmonic)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_amplitude, size=count)
    # A stationary deterministic high-pass-ish floor keeps silence from being digital zero.
    noise[1:] = 0.72 * noise[1:] + 0.28 * noise[:-1]
    waveform += noise
    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform = waveform / peak * 0.31
    return np.rint(waveform * 32767.0).astype(np.int16)


def _truncate_wav(source: Path, target: Path, fraction: float = 0.58) -> None:
    data = source.read_bytes()
    length = max(64, int(len(data) * fraction))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data[:length])


def _config(max_frames: int = 64) -> dict[str, Any]:
    return {
        "format_version": 1,
        "sample_rate": 16000,
        "frame_length_ms": 25.0,
        "frame_shift_ms": 10.0,
        "mel_bands": 24,
        "mel_max_hz": 7600.0,
        "cepstrum_order": 12,
        "f0_min_hz": 60.0,
        "f0_max_hz": 400.0,
        "f0_quantization_levels": 64,
        "thread_pool_workers": 4,
        "speech_frame_budget": 1000,
        "snapshot_interval_segments": 1,
        "max_frames_per_segment": max_frames,
        "fixed_point_decimals": 6,
        "energy_threshold_db": -45.0,
        "quantiles": [0.1, 0.5, 0.9],
        "quantile_interpolation": "linear",
        "output_gain_db": -1.0,
    }


def _commands() -> dict[str, Any]:
    return {
        "format_version": 1,
        "defaults": {
            "target_speaker": "target_a",
            "pitch_offset_semitones": 0.0,
            "timbre_strength": 0.65,
            "formant_warp": 1.0,
            "smoothing": 0.20,
        },
        "initial_segments": [
            {
                "segment_id": "seg_001",
                "path": "sources/session.wav",
                "start_sample": 0,
                "end_sample": 8000,
            },
            {
                "segment_id": "seg_002",
                "path": "sources/session.wav",
                "start_sample": 8000,
                "end_sample": 14880,
            },
            {
                "segment_id": "seg_003",
                "path": "sources/session.wav",
                "start_sample": 14880,
                "end_sample": 22880,
            },
        ],
        "commands": [
            {
                "seq": 1,
                "op": "set_params",
                "target_speaker": "target_a",
                "pitch_offset_semitones": 1.5,
                "timbre_strength": 0.72,
                "smoothing": 0.20,
            },
            {
                "seq": 2,
                "op": "override_segment",
                "segment_id": "seg_001",
                "timbre_strength": 0.55,
            },
            {
                "seq": 3,
                "op": "set_formant_warp",
                "value": 1.08,
            },
            {
                "seq": 4,
                "op": "replace_segment",
                "segment_id": "seg_002",
                "path": "sources/replacement.wav",
            },
            {
                "seq": 5,
                "op": "set_smoothing",
                "value": 0.35,
            },
            {
                "seq": 6,
                "op": "append_segment",
                "segment_id": "seg_004",
                "path": "sources/append.wav",
            },
            {
                "seq": 7,
                "op": "override_segment",
                "segment_id": "seg_002",
                "pitch_offset_semitones": -0.75,
            },
            {
                "seq": 8,
                "op": "set_target",
                "target_speaker": "target_b",
            },
        ],
    }


def generate_samples(root: Path, *, force: bool = False) -> dict[str, Any]:
    marker = root / ".generated"
    if marker.exists() and not force:
        return {"root": str(root), "created": False, "marker": str(marker)}
    if root.exists() and force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    target_a = ((620.0, 155.0, 1.0), (1320.0, 230.0, 0.62), (2550.0, 330.0, 0.34), (3550.0, 420.0, 0.15))
    target_b = ((430.0, 120.0, 1.0), (1850.0, 260.0, 0.70), (2940.0, 360.0, 0.39), (4100.0, 520.0, 0.17))
    source_shape = ((720.0, 170.0, 1.0), (1480.0, 240.0, 0.55), (2700.0, 350.0, 0.28), (3900.0, 460.0, 0.12))

    reference_specs = {
        "target_a": (118.0, 5.1, 3.0, 7.0),
        "target_b": (196.0, 4.3, 4.5, -5.0),
    }
    for speaker, (base_f0, vibrato_hz, depth, trend) in reference_specs.items():
        shape = target_a if speaker == "target_a" else target_b
        for index in (1, 2):
            samples = _voice(
                duration_seconds=1.0,
                base_f0=base_f0 + index * 1.7,
                vibrato_hz=vibrato_hz + index * 0.11,
                vibrato_depth=depth,
                f0_trend=trend,
                formants=shape,
                noise_amplitude=0.0026,
                seed=SEED + (11 if speaker == "target_a" else 29) + index,
                phase_offset=0.31 * index,
            )
            _write_wav(root / "references" / speaker / f"ref_{index:02d}.wav", samples)

    segment_shapes = [
        (0.50, 112.0, 4.8, 3.2, 9.0),
        (0.43, 127.0, 5.5, 2.4, -3.0),
        (0.50, 103.0, 4.2, 4.0, 5.0),
    ]
    segments: list[np.ndarray] = []
    for index, (duration, base_f0, vibrato_hz, depth, trend) in enumerate(segment_shapes):
        segments.append(
            _voice(
                duration_seconds=duration,
                base_f0=base_f0,
                vibrato_hz=vibrato_hz,
                vibrato_depth=depth,
                f0_trend=trend,
                formants=source_shape,
                noise_amplitude=0.0032,
                seed=SEED + 100 + index,
                phase_offset=0.17 * index,
            )
        )
    session = np.concatenate(segments)
    _write_wav(root / "sources" / "session.wav", session)
    replacement = _voice(
        duration_seconds=0.43,
        base_f0=138.0,
        vibrato_hz=5.9,
        vibrato_depth=2.2,
        f0_trend=-6.0,
        formants=source_shape,
        noise_amplitude=0.0030,
        seed=SEED + 211,
        phase_offset=0.73,
    )
    _write_wav(root / "sources" / "replacement.wav", replacement)
    append = _voice(
        duration_seconds=0.37,
        base_f0=110.0,
        vibrato_hz=4.6,
        vibrato_depth=2.8,
        f0_trend=4.0,
        formants=source_shape,
        noise_amplitude=0.0028,
        seed=SEED + 223,
        phase_offset=0.43,
    )
    _write_wav(root / "sources" / "append.wav", append)

    config = _config()
    (root / "config.json").write_text(json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    commands = _commands()
    (root / "commands.json").write_text(
        json.dumps(commands, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    errors = root / "errors"
    errors.mkdir(parents=True, exist_ok=True)
    _truncate_wav(root / "sources" / "session.wav", errors / "truncated.wav")
    overflow_config = _config(max_frames=8)
    (errors / "overflow_config.json").write_text(
        json.dumps(overflow_config, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    overflow_commands = {
        "format_version": 1,
        "defaults": {"target_speaker": "target_a"},
        "initial_segments": [
            {"segment_id": "seg_001", "path": "sources/session.wav", "start_sample": 0, "end_sample": 8000}
        ],
        "commands": [{"seq": 1, "op": "set_timbre_strength", "value": 0.7}],
    }
    (errors / "overflow_commands.json").write_text(
        json.dumps(overflow_commands, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    invalid_commands = {
        "format_version": 1,
        "defaults": {"target_speaker": "target_a"},
        "initial_segments": commands["initial_segments"],
        "commands": [
            {"seq": 2, "op": "set_timbre_strength", "value": 0.7},
            {"seq": 2, "op": "unknown_operation", "value": 1},
            {"seq": 3, "op": "set_smoothing", "value": 0.3},
        ],
    }
    (errors / "invalid_commands.json").write_text(
        json.dumps(invalid_commands, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    invalid_root = errors / "audio_format"
    invalid_ref = invalid_root / "references" / "target_a" / "ref_bad.wav"
    invalid_ref.parent.mkdir(parents=True, exist_ok=True)
    _truncate_wav(root / "references" / "target_a" / "ref_01.wav", invalid_ref, 0.51)
    invalid_source = invalid_root / "sources" / "bad.wav"
    _truncate_wav(root / "sources" / "append.wav", invalid_source, 0.49)
    invalid_commands = {
        "format_version": 1,
        "defaults": {"target_speaker": "target_a"},
        "initial_segments": [
            {"segment_id": "seg_bad", "path": "sources/bad.wav", "start_sample": 0, "end_sample": 4000}
        ],
        "commands": [{"seq": 1, "op": "set_timbre_strength", "value": 0.6}],
    }
    (invalid_root / "commands.json").write_text(
        json.dumps(invalid_commands, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (invalid_root / "config.json").write_text(
        json.dumps(_config(), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "fixed_seed": SEED,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "sample_width_bits": 16,
        "references": [
            "references/target_a/ref_01.wav",
            "references/target_a/ref_02.wav",
            "references/target_b/ref_01.wav",
            "references/target_b/ref_02.wav",
        ],
        "sources": {
            "session": {"path": "sources/session.wav", "sample_count": int(session.size)},
            "replacement": {"path": "sources/replacement.wav", "sample_count": int(replacement.size)},
            "append": {"path": "sources/append.wav", "sample_count": int(append.size)},
        },
        "error_fixtures": [
            "errors/truncated.wav",
            "errors/overflow_config.json",
            "errors/overflow_commands.json",
            "errors/invalid_commands.json",
            "errors/audio_format/",
        ],
    }
    (root / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    marker.write_text("generated by tools/generate_samples.py\n", encoding="utf-8")
    return {
        "root": str(root),
        "created": True,
        "seed": SEED,
        "session_samples": int(session.size),
        "reference_count": len(reference_specs) * 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="samples")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = generate_samples(Path(args.dir).resolve(), force=args.force)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
