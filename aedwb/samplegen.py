"""Offline, deterministic AED workbench sample generation."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CLASSES
from .util import write_json

SAMPLE_RATE = 16_000
CLIP_MS = 5_000
CLIP_COUNT = 3
GLOBAL_CODEBOOK_VERSION = "global-v2"
LEGACY_CODEBOOK_VERSION = "legacy-v1"

CLIP_EVENTS: tuple[tuple[tuple[str, int, int], ...], ...] = (
    (
        ("speech", 200, 1_200),
        ("music", 1_400, 2_500),
        ("noise", 2_700, 3_300),
        ("alarm", 3_500, 4_600),
    ),
    (
        ("music", 100, 1_100),
        ("speech", 1_300, 2_200),
        ("alarm", 2_400, 3_500),
        ("noise", 3_700, 4_600),
    ),
    (
        ("alarm", 200, 1_000),
        ("noise", 1_200, 2_100),
        ("speech", 2_300, 3_500),
        ("music", 3_700, 4_700),
    ),
)

GLOBAL_CODEBOOK = [
    {"code": 0, "name": "speech"},
    {"code": 1, "name": "music"},
    {"code": 2, "name": "noise"},
    {"code": 3, "name": "alarm"},
]

LEGACY_CODEBOOK = [
    {"code": 0, "name": "music"},
    {"code": 1, "name": "noise"},
    {"code": 2, "name": "alarm"},
    {"code": 3, "name": "speech"},
]


def _write_wav(path: Path, samples: np.ndarray) -> None:
    import wave

    clipped = np.clip(samples, -0.98, 0.98)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def _event_wave(kind: str, time_s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if kind == "speech":
        formant = (
            0.55 * np.sin(2.0 * np.pi * 178.0 * time_s)
            + 0.25 * np.sin(2.0 * np.pi * 372.0 * time_s + 0.4)
            + 0.12 * np.sin(2.0 * np.pi * 811.0 * time_s + 1.1)
        )
        syllable = 0.55 + 0.45 * np.sin(2.0 * np.pi * 3.2 * time_s) ** 2
        return formant * syllable
    if kind == "music":
        return (
            0.46 * np.sin(2.0 * np.pi * 261.63 * time_s)
            + 0.28 * np.sin(2.0 * np.pi * 329.63 * time_s + 0.3)
            + 0.18 * np.sin(2.0 * np.pi * 392.00 * time_s + 0.7)
        )
    if kind == "noise":
        white = rng.normal(0.0, 1.0, size=time_s.size)
        smooth = np.convolve(white, np.ones(7) / 7.0, mode="same")
        return 0.82 * smooth
    if kind == "alarm":
        gate = (np.sin(2.0 * np.pi * 2.2 * time_s) > 0.08).astype(float)
        return gate * (
            0.62 * np.sin(2.0 * np.pi * 880.0 * time_s)
            + 0.24 * np.sin(2.0 * np.pi * 1_760.0 * time_s)
        )
    raise ValueError(f"unknown event kind: {kind}")


def _render_clip(clip_index: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    seed = 10_000 + clip_index * 101
    rng = np.random.default_rng(seed)
    sample_count = SAMPLE_RATE * CLIP_MS // 1_000
    time_s = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    # A deterministic, low-level stationary bed remains present in silence gaps.
    audio = rng.normal(0.0, 0.0045, size=sample_count)
    truth: list[dict[str, Any]] = []

    for event_index, (kind, start_ms, end_ms) in enumerate(CLIP_EVENTS[clip_index]):
        start_sample = start_ms * SAMPLE_RATE // 1_000
        end_sample = end_ms * SAMPLE_RATE // 1_000
        local_time = time_s[start_sample:end_sample] - start_ms / 1_000.0
        duration_s = (end_ms - start_ms) / 1_000.0
        envelope = np.sin(np.pi * np.clip(local_time / duration_s, 0.0, 1.0)) ** 1.7
        event = _event_wave(kind, local_time, rng) * envelope
        audio[start_sample:end_sample] += 0.34 * event
        truth.append(
            {
                "event_id": f"clip{clip_index}-evt{event_index}",
                "class": kind,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "envelope": "raised_sine^1.7",
            }
        )

    # Make silence gaps explicit while preserving the stationary bed.
    for gap_start, gap_end in ((0, 200), (1_200, 1_400), (2_500, 2_700), (3_300, 3_500)):
        start = gap_start * SAMPLE_RATE // 1_000
        end = gap_end * SAMPLE_RATE // 1_000
        audio[start:end] *= 0.22
    return audio, truth


def _event_strength(events: tuple[tuple[str, int, int], ...], t_ms: float) -> dict[str, float]:
    result = {name: 0.0 for name in CLASSES}
    for kind, start_ms, end_ms in events:
        if start_ms <= t_ms < end_ms:
            position = (t_ms - start_ms) / max(1, end_ms - start_ms)
            result[kind] = max(result[kind], math.sin(math.pi * position) ** 1.7)
    return result


def _frame_probabilities(
    events: tuple[tuple[str, int, int], ...],
    start_ms: int,
    end_ms: int,
    shift_ms: int,
    seed: int,
    mode: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = (end_ms - start_ms) // shift_ms
    rows = np.zeros((count, len(CLASSES)), dtype=np.float64)
    for index in range(count):
        t_ms = start_ms + index * shift_ms
        strength = _event_strength(events, t_ms + shift_ms / 2.0)
        for class_index, name in enumerate(CLASSES):
            base = 0.018 + 0.010 * float(rng.random())
            rows[index, class_index] = base + 0.92 * strength[name]
        # Smooth measurement-like variation, deterministic per frame and class.
        rows[index] *= 1.0 + rng.normal(0.0, 0.035, size=len(CLASSES))
        rows[index] = np.clip(rows[index], 1e-5, None)
        rows[index] /= rows[index].sum()

    if mode == "drift":
        sharpened = np.power(rows, 0.36)
        bias = np.array([0.018, -0.010, 0.030, -0.028], dtype=np.float64)
        rows = np.clip(sharpened + bias, 1e-6, None)
        rows /= rows.sum(axis=1, keepdims=True)
    return rows


def _reorder_legacy(probabilities: np.ndarray) -> np.ndarray:
    legacy_positions = {entry["name"]: index for index, entry in enumerate(LEGACY_CODEBOOK)}
    reordered = np.zeros_like(probabilities)
    for global_index, name in enumerate(CLASSES):
        reordered[:, legacy_positions[name]] = probabilities[:, global_index]
    return reordered


def _packet_scores(
    events: tuple[tuple[str, int, int], ...],
    start_ms: int,
    end_ms: int,
    shift_ms: int,
    seed: int,
    score_type: str,
    codebook_version: str,
    mode: str,
) -> list[list[float | int]]:
    probabilities = _frame_probabilities(events, start_ms, end_ms, shift_ms, seed, mode)
    if score_type == "probability":
        return [[round(float(value), 9) for value in row] for row in probabilities]

    rng = np.random.default_rng(seed + 70_001)
    reordered = (
        _reorder_legacy(probabilities)
        if codebook_version == LEGACY_CODEBOOK_VERSION
        else probabilities
    )
    logits = np.log(reordered) + rng.normal(0.0, 0.055, size=reordered.shape)
    integer_logits = np.rint(logits * 12.0).astype(int)
    return integer_logits.tolist()


def _packet_definition(
    audio_index: int,
    name: str,
    start_ms: int,
    end_ms: int,
    shift_ms: int,
    score_type: str,
    codebook_version: str,
    grid_version: str,
    mode: str,
) -> dict[str, Any]:
    seed = 20_000 + audio_index * 1_000 + start_ms * 3 + end_ms + shift_ms
    return {
        "schema_version": "aed-packet/1",
        "packet_id": f"a{audio_index}_{name}",
        "audio_file": f"clip_{audio_index:03d}.wav",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "frame_shift_ms": shift_ms,
        "score_type": score_type,
        "class_codebook_version": codebook_version,
        "frame_grid_version": grid_version,
        "detector_version": (
            "legacy-logit-1.4" if score_type == "logit" else f"probabilistic-2.{audio_index}"
        ),
        "frames": _packet_scores(
            CLIP_EVENTS[audio_index],
            start_ms,
            end_ms,
            shift_ms,
            seed,
            score_type,
            codebook_version,
            mode,
        ),
    }


def _base_packets(audio_index: int) -> list[dict[str, Any]]:
    return [
        _packet_definition(
            audio_index,
            "base_new",
            0,
            2_500,
            20,
            "probability",
            GLOBAL_CODEBOOK_VERSION,
            "fg20-v2",
            "regular",
        ),
        _packet_definition(
            audio_index,
            "base_legacy",
            2_000,
            5_000,
            20,
            "logit",
            LEGACY_CODEBOOK_VERSION,
            "fg20-v1",
            "regular",
        ),
        _packet_definition(
            audio_index,
            "fine_legacy",
            0,
            1_500,
            10,
            "logit",
            LEGACY_CODEBOOK_VERSION,
            "fg10-v1",
            "regular",
        ),
        _packet_definition(
            audio_index,
            "timeline_new",
            1_500,
            3_000,
            30,
            "probability",
            GLOBAL_CODEBOOK_VERSION,
            "fg30-v2",
            "regular",
        ),
        _packet_definition(
            audio_index,
            "coarse_new",
            0,
            1_500,
            250,
            "probability",
            GLOBAL_CODEBOOK_VERSION,
            "fg250-v2",
            "regular",
        ),
        _packet_definition(
            audio_index,
            "logit_direct_drift",
            3_000,
            5_000,
            20,
            "logit",
            GLOBAL_CODEBOOK_VERSION,
            "fg20-v2",
            "drift",
        ),
        _packet_definition(
            audio_index,
            "drift_new",
            2_500,
            5_000,
            20,
            "probability",
            GLOBAL_CODEBOOK_VERSION,
            "fg20-v2",
            "drift",
        ),
    ]


def _injection_probes(audio_index: int) -> list[dict[str, Any]]:
    if audio_index != 0:
        return []
    return [
        _packet_definition(
            audio_index,
            "probe_truncated",
            3_000,
            3_500,
            20,
            "probability",
            GLOBAL_CODEBOOK_VERSION,
            "fg20-v2",
            "regular",
        ),
        _packet_definition(
            audio_index,
            "probe_frame_count",
            3_600,
            4_100,
            20,
            "probability",
            GLOBAL_CODEBOOK_VERSION,
            "fg20-v2",
            "regular",
        ),
        _packet_definition(
            audio_index,
            "probe_missing_type",
            4_200,
            4_700,
            20,
            "probability",
            GLOBAL_CODEBOOK_VERSION,
            "fg20-v2",
            "regular",
        ),
        _packet_definition(
            audio_index,
            "probe_nonfinite",
            2_600,
            3_100,
            20,
            "logit",
            LEGACY_CODEBOOK_VERSION,
            "fg20-v1",
            "regular",
        ),
        _packet_definition(
            audio_index,
            "probe_unmapped",
            3_200,
            3_700,
            20,
            "logit",
            LEGACY_CODEBOOK_VERSION,
            "fg20-v1",
            "regular",
        ),
    ]


def _global_config() -> dict[str, Any]:
    return {
        "schema_version": "aed-global-config/1",
        "global_version": "g-2026-11-21.1",
        "frame_shift_ms": 20,
        "classes": list(CLASSES),
        "class_codebook_version": GLOBAL_CODEBOOK_VERSION,
        "class_codebook": GLOBAL_CODEBOOK,
        "class_priority": ["alarm", "speech", "music", "noise"],
        "calibration_quantiles": [0.10, 0.90],
        "calibration_tolerance": 1e-9,
        "drift_alert_threshold": 0.05,
        "decision_threshold": 0.40,
        "probability_quantum": 0.05,
        "thread_pool_workers": 4,
        "snapshot_commit_interval": 2,
        "interaction_budget_ms": 3_000,
    }


def _injections(packet_files: dict[str, str]) -> list[dict[str, str]]:
    specifications = [
        {
            "injection_id": "INJ-001",
            "packet_id": "a0_probe_truncated",
            "kind": "truncated_packet",
            "expected_code": "E_PACKET_INVALID",
        },
        {
            "injection_id": "INJ-002",
            "packet_id": "a0_probe_frame_count",
            "kind": "frame_count_mismatch",
            "expected_code": "E_PACKET_INVALID",
        },
        {
            "injection_id": "INJ-003",
            "packet_id": "a0_probe_missing_type",
            "kind": "missing_score_type",
            "expected_code": "E_PACKET_INVALID",
        },
        {
            "injection_id": "INJ-004",
            "packet_id": "a0_probe_nonfinite",
            "kind": "nonfinite_logit",
            "expected_code": "E_PACKET_INVALID",
        },
        {
            "injection_id": "INJ-005",
            "packet_id": "a0_probe_unmapped",
            "kind": "unmappable_codebook",
            "expected_code": "E_CLASS_UNMAPPED",
        },
    ]
    return [
        {**item, "packet_file": packet_files[item["packet_id"]]}
        for item in specifications
    ]


def generate_sample(out_dir: str | Path, injected: bool = False, clean: bool = True) -> Path:
    """Generate a complete local sample directory.

    The same seed and package version always produce byte-identical inputs.  When
    ``injected`` is true, five extra probe packets and ``injections.json`` are
    added; the base packets and calibration baselines remain unchanged.
    """

    out_dir = Path(out_dir)
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth_clips: dict[str, Any] = {}
    for audio_index in range(CLIP_COUNT):
        audio, events = _render_clip(audio_index)
        audio_name = f"clip_{audio_index:03d}.wav"
        _write_wav(out_dir / "audio" / audio_name, audio)
        truth_clips[audio_name] = {
            "duration_ms": CLIP_MS,
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "bits_per_sample": 16,
            "events": events,
        }

    write_json(
        out_dir / "codebooks.json",
        {
            "schema_version": "aed-codebooks/1",
            "codebooks": {
                GLOBAL_CODEBOOK_VERSION: {"entries": GLOBAL_CODEBOOK},
                LEGACY_CODEBOOK_VERSION: {"entries": LEGACY_CODEBOOK},
            },
        },
    )
    write_json(out_dir / "config.json", _global_config())
    write_json(
        out_dir / "truth.json",
        {
            "schema_version": "aed-truth/1",
            "clips": truth_clips,
            "background_label": "none",
        },
    )

    packets: list[dict[str, Any]] = []
    for audio_index in range(CLIP_COUNT):
        packets.extend(_base_packets(audio_index))
    if injected:
        packets.extend(_injection_probes(0))

    baseline_ids: list[str] = []
    for audio_index in range(CLIP_COUNT):
        baseline_ids.extend([f"a{audio_index}_base_new", f"a{audio_index}_base_legacy"])

    write_json(
        out_dir / "baseline_packets.json",
        {
            "schema_version": "aed-baseline/1",
            "baseline_packet_ids": baseline_ids,
        },
    )

    packet_files: dict[str, str] = {}
    for index, packet in enumerate(packets):
        file_name = f"{index:03d}_{packet['packet_id']}.json"
        write_json(out_dir / "packets" / file_name, packet)
        packet_files[packet["packet_id"]] = f"packets/{file_name}"

    write_json(
        out_dir / "injections.json",
        {
            "schema_version": "aed-injections/1",
            "injections": _injections(packet_files) if injected else [],
        },
    )
    return out_dir
