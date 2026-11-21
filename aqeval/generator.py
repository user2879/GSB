"""Deterministic synthetic speech, exchange fixtures and edit scripts."""

from __future__ import annotations

import io
import shutil
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .audio import decode_wav_bytes, frame_ranges
from .config import Config, DIMENSIONS
from .conversion import convert_batch
from .formats import write_exchange
from .levels import LevelEntry, LevelMap
from .models import UnifiedResult
from .scoring import (
    compute_frame_features,
    frame_summaries_from_features,
    grade_total_micro,
    score_dimensions,
    summarize_frames,
)
from .util import atomic_write_json, sha256_bytes, sha256_json


@dataclass(frozen=True)
class SampleSpec:
    file_id: str
    seed: int
    snr_db: float | None
    clipping_ratio: float
    dropout_interval: int
    dropout_length: int
    noise_floor_dbfs: float | None
    boundary_zero_start: int | None = None
    boundary_zero_length: int = 0


DEFAULT_SAMPLE_SPECS = (
    SampleSpec("sample_clean", 101, None, 0.0, 0, 0, None),
    SampleSpec("sample_snr6", 102, 6.0, 0.0, 0, 0, -70.0),
    SampleSpec("sample_clip", 103, None, 0.02, 0, 0, -70.0),
    SampleSpec("sample_dropout", 104, None, 0.0, 640, 48, -70.0),
    SampleSpec("sample_floor", 105, None, 0.0, 0, 0, -34.0),
    SampleSpec(
        "sample_boundary",
        106,
        30.0,
        0.0,
        0,
        0,
        -75.0,
        boundary_zero_start=400,
        boundary_zero_length=20,
    ),
)

REPLACEMENT_SPEC = SampleSpec("sample_snr6", 202, 14.0, 0.0, 0, 0, -75.0)


def default_config() -> dict[str, Any]:
    return {
        "schema": "aqeval.config.v1",
        "sample_rate": 16000,
        "frame_length_ms": 25,
        "hop_ms": 10,
        "silence_threshold_dbfs": -45.0,
        "noise_floor_percentile": 20,
        "clipping_amplitude_threshold": 0.98,
        "consecutive_zero_samples": 20,
        "weights": {
            "level": 0.2,
            "noise": 0.3,
            "clipping": 0.25,
            "continuity": 0.25,
        },
        "fixed_decimal_bits": 5,
        "quantization_levels": list(range(0, 1001, 100)),
        "max_concurrency": 4,
        "concurrency_cap": 8,
        "snapshot_commit_interval": 2,
        "verify_serial_consistency": True,
    }


def default_level_map() -> LevelMap:
    return LevelMap(
        [
            LevelEntry("0", "poor", 0),
            LevelEntry("1", "fair", 600),
            LevelEntry("2", "good", 780),
            LevelEntry("3", "excellent", 900),
        ]
    )


def generate_sample_bytes(
    spec: SampleSpec,
    *,
    sample_rate: int = 16000,
    duration_ms: int = 1200,
) -> bytes:
    sample_count = int(round(sample_rate * duration_ms / 1000))
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    f0 = 118.0 + 7.0 * np.sin(2.0 * np.pi * 1.3 * time)
    phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate
    gate = (np.sin(2.0 * np.pi * 3.1 * time + 0.2) > -0.35).astype(np.float64)
    envelope = gate * (0.45 + 0.55 * np.abs(np.sin(2.0 * np.pi * 2.1 * time + 0.3)))
    signal = envelope * (
        0.68 * np.sin(phase)
        + 0.22 * np.sin(2.0 * phase + 0.2)
        + 0.10 * np.sin(3.0 * phase + 0.7)
    )
    signal -= float(np.mean(signal))
    target_rms = 10.0 ** (-18.0 / 20.0)
    signal *= target_rms / max(float(np.sqrt(np.mean(signal * signal))), 1e-12)
    rng = np.random.default_rng(spec.seed)
    if spec.snr_db is not None:
        noise_sigma = target_rms / (10.0 ** (spec.snr_db / 20.0))
        signal += rng.standard_normal(sample_count) * noise_sigma
    if spec.noise_floor_dbfs is not None:
        floor_sigma = 10.0 ** (spec.noise_floor_dbfs / 20.0)
        signal += rng.standard_normal(sample_count) * floor_sigma
    elif spec.snr_db is None:
        dither_sigma = 10.0 ** (-75.0 / 20.0)
        signal += rng.standard_normal(sample_count) * dither_sigma
    if spec.clipping_ratio > 0:
        clip_count = int(round(sample_count * spec.clipping_ratio))
        indices = np.sort(rng.choice(sample_count, size=clip_count, replace=False))
        signal[indices] *= 12.0
    signal = np.clip(signal, -1.0, 1.0)
    if spec.dropout_interval > 0 and spec.dropout_length > 0:
        for start in range(0, sample_count, spec.dropout_interval):
            signal[start : start + spec.dropout_length] = 0.0
    if spec.boundary_zero_start is not None and spec.boundary_zero_length > 0:
        start = spec.boundary_zero_start
        signal[start : start + spec.boundary_zero_length] = 0.0
    pcm = np.clip(np.rint(signal * 32767.0), -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.setcomptype("NONE", "not compressed")
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _evaluate_to_unified(audio_bytes: bytes, file_id: str, config: Config, level_map: LevelMap) -> UnifiedResult:
    samples, sample_rate = decode_wav_bytes(
        audio_bytes, expected_sample_rate=config.sample_rate, location=file_id
    )
    frames: list[dict[str, Any]] = []
    for frame_index, (start, end) in enumerate(
        frame_ranges(samples.size, config.frame_length_samples, config.hop_samples)
    ):
        frames.append(
            compute_frame_features(
                file_id,
                frame_index,
                samples[start:end],
                sample_rate,
                config.clipping_amplitude_threshold,
            )
        )
    summary = summarize_frames(file_id, frames, config)
    scored = score_dimensions(file_id, summary, config)
    code, name, at_threshold = grade_total_micro(scored["total_micro"], level_map)
    bypass: dict[str, Any] = {
        "generator": "aqeval.synthetic.v1",
        "sample_seed": None,
        "boundary_at_level_threshold": at_threshold,
    }
    result = UnifiedResult(
        file_id=file_id,
        source_version=2,
        total_micro=scored["total_micro"],
        dimensions_micro=scored["dimensions_micro"],
        level_code=code,
        level_name=name,
        precision_tag="reeval_float6",
        frame_summaries=frame_summaries_from_features(frames),
        bypass=bypass,
        location=file_id,
    )
    return result


def _write_sample_manifest(root: Path, specs: list[SampleSpec], hashes: dict[str, str]) -> None:
    atomic_write_json(
        root / "samples" / "manifest.json",
        {
            "schema": "aqeval.samples.v1",
            "sample_rate": 16000,
            "channels": 1,
            "sample_width_bits": 16,
            "specs": [asdict(spec) for spec in specs],
            "sha256": hashes,
        },
    )


def generate_workspace(root: str | Path, *, clean: bool = True) -> dict[str, Any]:
    root = Path(root)
    if clean and root.exists():
        shutil.rmtree(root)
    for directory in ("samples", "replacements", "eval/v1", "eval/v2", "eval/v3", "bypass", "edits"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    config_dict = default_config()
    config = Config.from_dict(config_dict)
    level_map = default_level_map()
    atomic_write_json(root / "config.json", config_dict)
    atomic_write_json(root / "level_map.json", level_map.to_dict())
    sample_hashes: dict[str, str] = {}
    generated_records: list[UnifiedResult] = []
    for spec in DEFAULT_SAMPLE_SPECS:
        data = generate_sample_bytes(spec, sample_rate=config.sample_rate)
        (root / "samples" / f"{spec.file_id}.wav").write_bytes(data)
        sample_hashes[spec.file_id] = sha256_bytes(data)
        record = _evaluate_to_unified(data, spec.file_id, config, level_map)
        record.bypass["sample_seed"] = spec.seed
        generated_records.append(record)
    replacement = generate_sample_bytes(REPLACEMENT_SPEC, sample_rate=config.sample_rate)
    (root / "replacements" / "sample_snr6.wav").write_bytes(replacement)
    _write_sample_manifest(root, list(DEFAULT_SAMPLE_SPECS), sample_hashes)
    write_exchange(1, generated_records, config.fixed_decimal_bits, root / "eval/v1/results_v1.txt")
    write_exchange(2, generated_records, config.fixed_decimal_bits, root / "eval/v2/results_v2.json")
    write_exchange(3, generated_records, config.fixed_decimal_bits, root / "eval/v3/results_v3.bin")
    bypass_outcome = convert_batch(generated_records, 1, level_map, config.fixed_decimal_bits)
    atomic_write_json(root / "bypass" / "initial_v2_to_v1.json", bypass_outcome.bypass_document())
    atomic_write_json(
        root / "bypass" / "initial_conversion_degradations.json",
        {
            "schema": "aqeval.degradations.v1",
            "entries": [item.to_dict() for item in bypass_outcome.degradations],
        },
    )
    atomic_write_json(
        root / "state.json",
        {
            "schema": "aqeval.state.v1",
            "active_version": 2,
            "config_revision": 0,
            "global_run": 0,
            "last_edits": [],
            "initial_config_hash": config.config_hash,
            "initial_level_map_hash": level_map.digest,
        },
    )
    edit_script = [
        {"op": "set_weight", "dimension": "noise", "value": 0.35},
        {"op": "set_threshold", "key": "silence_threshold_dbfs", "value": -42.0},
        {
            "op": "set_quantization_levels",
            "value": list(range(0, 1001, 50)),
        },
        {"op": "set_fixed_bits", "value": 2},
        {"op": "replace_audio", "file_id": "sample_snr6", "source": "replacements/sample_snr6.wav"},
        {"op": "add_level", "code": "4", "name": "outstanding", "min_total_permille": 950},
        {"op": "set_level_threshold", "code": "2", "value": 800},
        {"op": "remove_level", "code": "4"},
        {"op": "set_concurrency", "value": 4},
        {"op": "switch_version", "value": 3},
        {"op": "switch_version", "value": 1},
        {"op": "switch_version", "value": 2},
    ]
    import json

    (root / "edits" / "engineering_sequence.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in edit_script) + "\n",
        encoding="utf-8",
    )
    return {
        "workspace": str(root),
        "samples": len(DEFAULT_SAMPLE_SPECS),
        "sample_hashes": sample_hashes,
        "exchange_files": {
            "v1": str(root / "eval/v1/results_v1.txt"),
            "v2": str(root / "eval/v2/results_v2.json"),
            "v3": str(root / "eval/v3/results_v3.bin"),
        },
        "config_hash": config.config_hash,
        "level_map_hash": level_map.digest,
        "initial_results": [
            {
                "file_id": record.file_id,
                "total_permille": record.total_permille,
                "level_code": record.level_code,
                "level_name": record.level_name,
            }
            for record in generated_records
        ],
    }


def sample_generation_digest(specs: list[SampleSpec] | None = None) -> str:
    specs = specs or list(DEFAULT_SAMPLE_SPECS)
    return sha256_json(
        {
            "specs": [asdict(spec) for spec in specs],
            "sample_rate": 16000,
            "duration_ms": 1200,
        }
    )
