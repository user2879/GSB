"""Deterministic WAV corpus and retrieval-task generator for local validation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .wavio import write_pcm16_wav


def _synthesize_reference(sample_rate: int, duration_seconds: float, seed: int, transpose: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(round(sample_rate * duration_seconds))
    t = np.arange(count, dtype=np.float64) / sample_rate
    signal = np.zeros(count, dtype=np.float64)
    notes = np.asarray([0, 3, 7, 10, 7, 3, 12, 10, 7, 3, 5, 7, 0, 7, 12, 10], dtype=np.float64)
    roots = np.asarray([196.00, 220.00, 261.63, 293.66, 329.63], dtype=np.float64)
    cursor = 0
    note_index = 0
    while cursor < count:
        step = int((0.18 + 0.025 * (note_index % 5)) * sample_rate)
        end = min(count, cursor + step)
        frequency = roots[note_index % roots.size] * (2.0 ** ((notes[note_index % notes.size] + transpose) / 12.0))
        local_n = end - cursor
        if local_n <= 0:
            break
        local_t = np.arange(local_n, dtype=np.float64) / sample_rate
        attack = np.minimum(1.0, local_t / 0.012)
        release = np.minimum(1.0, np.maximum(0.0, (local_n / sample_rate - local_t) / 0.05))
        envelope = attack * release
        note = np.zeros(local_n, dtype=np.float64)
        for harmonic in range(1, 7):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            note += (0.68 ** harmonic) * np.sin(
                2.0 * np.pi * frequency * harmonic * local_t + phase
            )
        signal[cursor:end] += 0.16 * envelope * note
        click_n = min(local_n, int(0.018 * sample_rate))
        if click_n > 0:
            click_t = np.arange(click_n, dtype=np.float64) / sample_rate
            click = rng.standard_normal(click_n) * np.exp(-click_t * 95.0)
            signal[cursor : cursor + click_n] += 0.018 * click
        cursor = end
        note_index += 1
    bass = 0.045 * np.sin(2.0 * np.pi * (65.41 + 1.7 * np.sin(2.0 * np.pi * 0.09 * t)) * t)
    texture = 0.006 * rng.standard_normal(count)
    signal = signal + bass + texture
    peak = float(np.max(np.abs(signal)))
    return signal / max(peak, 1e-9) * 0.82


def _resample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(signal, dtype=np.float64).copy()
    output_count = int(round(signal.size * target_rate / source_rate))
    positions = np.arange(output_count, dtype=np.float64) * source_rate / target_rate
    return np.interp(positions, np.arange(signal.size, dtype=np.float64), signal)


def _time_stretch(segment: np.ndarray, speed_factor: float) -> np.ndarray:
    if speed_factor <= 0:
        raise ValueError("speed factor must be positive")
    output_count = int(round(segment.size / speed_factor))
    positions = np.arange(output_count, dtype=np.float64) * speed_factor
    return np.interp(positions, np.arange(segment.size, dtype=np.float64), segment)


def _add_background(signal: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(signal * signal))) if signal.size else 0.0
    if rms <= 1e-12:
        return signal
    noise_sigma = rms / (10.0 ** (snr_db / 20.0))
    return signal + rng.standard_normal(signal.size) * noise_sigma


def _mix_pieces(
    sample_rate: int,
    reference_at_rate: np.ndarray,
    *,
    speed_factor: float,
    gain: float,
    prefix_seconds: float,
    insertion_seconds: float,
    suffix_seconds: float,
    snr_db: float,
    seed: int,
    first_range: tuple[float, float],
    second_range: tuple[float, float],
) -> tuple[np.ndarray, list[dict[str, int]]]:
    rng = np.random.default_rng(seed)
    first_start = int(round(first_range[0] * sample_rate))
    first_end = int(round(first_range[1] * sample_rate))
    second_start = int(round(second_range[0] * sample_rate))
    second_end = int(round(second_range[1] * sample_rate))
    first = _time_stretch(reference_at_rate[first_start:first_end], speed_factor)
    second = _time_stretch(reference_at_rate[second_start:second_end], speed_factor)
    prefix_n = int(round(prefix_seconds * sample_rate))
    insertion_n = int(round(insertion_seconds * sample_rate))
    suffix_n = int(round(suffix_seconds * sample_rate))
    output = np.zeros(prefix_n + first.size + insertion_n + second.size + suffix_n, dtype=np.float64)
    cursor = 0
    cursor += prefix_n
    query_first_start = cursor
    output[cursor : cursor + first.size] = first * gain
    cursor += first.size
    query_first_end = cursor
    if insertion_n:
        insertion_t = np.arange(insertion_n, dtype=np.float64) / sample_rate
        insertion = 0.06 * np.sin(2.0 * np.pi * 740.0 * insertion_t)
        insertion += 0.035 * rng.standard_normal(insertion_n)
        output[cursor : cursor + insertion_n] = insertion
    cursor += insertion_n
    query_second_start = cursor
    output[cursor : cursor + second.size] = second * gain
    cursor += second.size
    query_second_end = cursor
    output = _add_background(output, rng, snr_db)
    peak = float(np.max(np.abs(output)))
    if peak > 0.96:
        output *= 0.96 / peak
    truth = [
        {
            "reference_start_sample": first_start,
            "reference_end_sample": first_end,
            "query_start_sample": query_first_start,
            "query_end_sample": query_first_end,
            "speed_factor": round(speed_factor, 9),
        },
        {
            "reference_start_sample": second_start,
            "reference_end_sample": second_end,
            "query_start_sample": query_second_start,
            "query_end_sample": query_second_end,
            "speed_factor": round(speed_factor, 9),
        },
    ]
    return output, truth


def _make_plain_query(
    sample_rate: int,
    reference_at_rate: np.ndarray,
    *,
    start_seconds: float,
    end_seconds: float,
    prefix_seconds: float,
    suffix_seconds: float,
    gain: float,
    snr_db: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    rng = np.random.default_rng(seed)
    start = int(round(start_seconds * sample_rate))
    end = int(round(end_seconds * sample_rate))
    segment = reference_at_rate[start:end] * gain
    prefix_n = int(round(prefix_seconds * sample_rate))
    suffix_n = int(round(suffix_seconds * sample_rate))
    output = np.zeros(prefix_n + segment.size + suffix_n, dtype=np.float64)
    output[prefix_n : prefix_n + segment.size] = segment
    output = _add_background(output, rng, snr_db)
    peak = float(np.max(np.abs(output)))
    if peak > 0.96:
        output *= 0.96 / peak
    return output, [
        {
            "reference_start_sample": start,
            "reference_end_sample": end,
            "query_start_sample": prefix_n,
            "query_end_sample": prefix_n + segment.size,
            "speed_factor": 1.0,
        }
    ]


def _truncate_wav(path: Path, fraction: float) -> None:
    size = path.stat().st_size
    keep = max(44, int(size * fraction))
    with path.open("r+b") as handle:
        handle.truncate(keep)


def generate_demo(workspace: str | Path, *, clean: bool = True) -> Path:
    root = Path(workspace).resolve()
    samples_dir = root / "samples"
    output_dir = root / "output"
    state_dir = root / "state"
    if clean:
        for directory in (samples_dir, output_dir, state_dir):
            if directory.exists():
                shutil.rmtree(directory)
    for directory in (samples_dir, output_dir, state_dir):
        directory.mkdir(parents=True, exist_ok=True)

    reference_rate = 44100
    reference = _synthesize_reference(reference_rate, 11.5, seed=20260301, transpose=0.0)
    other = _synthesize_reference(reference_rate, 8.0, seed=20260302, transpose=-5.0)
    write_pcm16_wav(samples_dir / "reference_song.wav", reference, reference_rate)
    write_pcm16_wav(samples_dir / "reference_other.wav", other, reference_rate)

    speed_rate = 48000
    speed_reference = _resample(reference, reference_rate, speed_rate)
    speed_query, speed_truth = _mix_pieces(
        speed_rate,
        speed_reference,
        speed_factor=1.025,
        gain=0.72,
        prefix_seconds=1.15,
        insertion_seconds=0.36,
        suffix_seconds=0.9,
        snr_db=18.0,
        seed=20260303,
        first_range=(1.25, 4.95),
        second_range=(5.20, 8.85),
    )
    write_pcm16_wav(samples_dir / "query_speed_noise.wav", speed_query, speed_rate)

    gain_rate = 16000
    gain_reference = _resample(reference, reference_rate, gain_rate)
    gain_query, gain_truth = _make_plain_query(
        gain_rate,
        gain_reference,
        start_seconds=2.0,
        end_seconds=7.4,
        prefix_seconds=0.75,
        suffix_seconds=0.6,
        gain=0.24,
        snr_db=12.0,
        seed=20260304,
    )
    write_pcm16_wav(samples_dir / "query_gain_noise.wav", gain_query, gain_rate)

    cross_rate = 44100
    cross_query, cross_truth = _make_plain_query(
        cross_rate,
        reference,
        start_seconds=3.2,
        end_seconds=8.3,
        prefix_seconds=0.55,
        suffix_seconds=0.45,
        gain=0.46,
        snr_db=8.0,
        seed=20260305,
    )
    write_pcm16_wav(samples_dir / "query_cross_rate.wav", cross_query, cross_rate)

    write_pcm16_wav(samples_dir / "query_truncated.wav", gain_query, gain_rate)
    _truncate_wav(samples_dir / "query_truncated.wav", 0.61)
    silent = np.zeros(int(2.5 * gain_rate), dtype=np.float64)
    write_pcm16_wav(samples_dir / "query_silent.wav", silent, gain_rate)
    write_pcm16_wav(samples_dir / "query_rate_mismatch.wav", cross_query, cross_rate)
    band_query, _ = _make_plain_query(
        cross_rate,
        reference,
        start_seconds=4.0,
        end_seconds=6.2,
        prefix_seconds=0.4,
        suffix_seconds=0.3,
        gain=0.5,
        snr_db=15.0,
        seed=20260306,
    )
    write_pcm16_wav(samples_dir / "query_bad_band.wav", band_query, cross_rate)

    task: dict[str, Any] = {
        "schema_version": 1,
        "allowed_sample_rates": [16000, 44100, 48000],
        "references": [
            {"id": "ref_song", "path": "samples/reference_song.wav"},
            {"id": "ref_other", "path": "samples/reference_other.wav"},
        ],
        "queries": [
            {"id": "q_speed_noise", "path": "samples/query_speed_noise.wav"},
            {"id": "q_gain_noise", "path": "samples/query_gain_noise.wav"},
            {"id": "q_cross_rate", "path": "samples/query_cross_rate.wav"},
            {"id": "q_truncated", "path": "samples/query_truncated.wav"},
            {"id": "q_silent", "path": "samples/query_silent.wav"},
            {
                "id": "q_rate_mismatch",
                "path": "samples/query_rate_mismatch.wav",
                "declared_sample_rate": 48000,
            },
            {
                "id": "q_bad_band",
                "path": "samples/query_bad_band.wav",
                "fingerprint_override": {"bands": [[100.0, 7900.0], [7900.0, 9000.0]]},
            },
        ],
        "fingerprint": {
            "frame_length": 1024,
            "hop_length": 256,
            "base_sample_rate": 16000,
            "bands": [[80.0, 600.0], [600.0, 1400.0], [1400.0, 2600.0], [2600.0, 4500.0], [4500.0, 7600.0]],
            "peak_neighborhood": {"frequency_bins": 3, "time_frames": 2},
            "min_peak_db": -42.0,
            "fanout": 5,
            "max_pair_delta_frames": 24,
            "peaks_per_band": 2,
            "window": "hann",
        },
        "retrieval": {
            "bucket_capacity": 256,
            "voting_window_frames": 16,
            "min_matches": 40,
            "max_candidates": 10,
            "overlap_policy": "keep_highest",
            "boundary_gap_frames": 21,
            "boundary_padding_frames": 2,
        },
        "tolerance": {
            "score_scale": 1000,
            "score_tolerance": 1e-12,
            "time_offset_tolerance_samples": 1024.0,
        },
        "concurrency": {
            "max_workers": 3,
            "spectrum_buffer_bytes": 524288,
        },
        "state_dir": "state",
        "outputs": {"report": "output/report.json", "results": "output/results.json"},
        "injections": [
            {"target_id": "q_truncated", "kind": "truncated_file", "expected_code": "E_TRUNC"},
            {"target_id": "q_silent", "kind": "silent_track", "expected_code": "E_SILENT"},
            {
                "target_id": "q_rate_mismatch",
                "kind": "rate_declaration_mismatch",
                "expected_code": "E_RATE",
                "declared_sample_rate": 48000,
            },
            {
                "target_id": "q_bad_band",
                "kind": "band_override",
                "expected_code": "E_BAND",
                "bands": [[100.0, 7900.0], [7900.0, 9000.0]],
            },
        ],
        "ground_truth": {
            "q_speed_noise": {"segments": speed_truth, "rate": speed_rate},
            "q_gain_noise": {"segments": gain_truth, "rate": gain_rate},
            "q_cross_rate": {"segments": cross_truth, "rate": cross_rate},
        },
    }
    task_path = root / "task.json"
    task_path.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return task_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate deterministic audio-fingerprint demo corpus")
    parser.add_argument("--workspace", default=".", help="workspace directory")
    parser.add_argument("--no-clean", action="store_true", help="keep previous generated files")
    args = parser.parse_args(argv)
    task_path = generate_demo(args.workspace, clean=not args.no_clean)
    print(task_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
