from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from .snapshot import atomic_write_json


def _repo_relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


SEED = 250
CHANNELS = ["acc_x_g", "acc_y_g", "acc_z_g"]
BATCH_SIZE = 8
BATCH_COUNT = 8
MISSING_RATE = 0.05
NULL_BITMAP = [[7, 1], [18, 2], [31, 0], [44, 1], [57, 2]]


def build_example_config(samples_path: str = "samples.jsonl") -> dict[str, Any]:
    scale = 10**12
    return {
        "schema_version": "1.0",
        "device_id": "rotating-pump-A",
        "samples_path": samples_path,
        "data_source": {
            "seed": SEED,
            "channels": CHANNELS,
            "samples_per_batch": BATCH_SIZE,
            "batch_count": BATCH_COUNT,
            "missing_rate": MISSING_RATE,
            "null_bitmap": NULL_BITMAP,
        },
        "input_scale": scale,
        "fixed_point": {
            "decimal_width": 12,
            "quantization_gear": "Q6",
            "attribution_tolerance_q": 1000,
        },
        "concurrency": {"max_workers": 4},
        "point_budget_per_batch": 10_000,
        "snapshot_interval_batches": 2,
        "drift_denominator_floor_q": 1_000_000,
        "windows": {
            "reference": {"start_seq": 1, "end_seq": 24},
            "observation": {"start_seq": 25, "end_seq": 48},
        },
        "stages": [
            {
                "name": "standardize",
                "type": "standardize",
                "upstream": "source",
                "scale": scale,
                "params": {
                    "mean_q": [0, 0, 0],
                    "inv_std_q": [4 * scale, 3 * scale, 2 * scale],
                },
            },
            {
                "name": "bucket",
                "type": "bucket",
                "upstream": "standardize",
                "scale": scale,
                "params": {
                    "edges_q": [-2 * scale, -scale, -4 * scale // 10, 4 * scale // 10, scale, 2 * scale],
                    "centers_q": [1 * scale, 2 * scale, 3 * scale, 4 * scale, 5 * scale, 6 * scale, 7 * scale],
                },
            },
            {
                "name": "log_stage",
                "type": "log",
                "upstream": "bucket",
                "scale": scale,
                "params": {"epsilon_q": 1_000, "iterations": 14},
            },
            {
                "name": "interaction",
                "type": "interaction",
                "upstream": "log_stage",
                "scale": scale,
                "params": {
                    "pairs": [[0, 1], [1, 2], [0, 2]],
                    "output_names": ["xy_energy", "yz_energy", "xz_energy"],
                },
            },
            {
                "name": "clip",
                "type": "clip",
                "upstream": "interaction",
                "scale": scale,
                "params": {"min_q": -10 * scale, "max_q": 10 * scale},
            },
        ],
        "feature_stage": "clip",
        "thresholds_bp": {
            "clip:xy_energy": 430,
            "clip:yz_energy": 1000,
            "clip:xz_energy": 400,
        },
    }


def generate_samples() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    total = BATCH_SIZE * BATCH_COUNT
    missing_coords = {(int(seq), int(ch)) for seq, ch in NULL_BITMAP}
    rows: list[dict[str, Any]] = []
    for seq in range(1, total + 1):
        batch = (seq - 1) // BATCH_SIZE
        observation_shift = 1.0 if 25 <= seq <= 48 else (0.35 if seq >= 49 else 0.0)
        values: list[float | None] = []
        missing: list[bool] = []
        for channel_index in range(3):
            phase = channel_index * 0.9
            amplitude = (0.22 + 0.05 * channel_index) * (1 + 0.55 * observation_shift)
            frequency = 0.55 + 0.13 * channel_index + 0.12 * observation_shift
            deterministic_wave = amplitude * math.sin(seq * frequency + phase)
            noise = rng.uniform(-0.045, 0.045)
            offset = observation_shift * (0.05 + 0.025 * channel_index)
            value = round(deterministic_wave + noise + offset, 9)
            is_missing = (seq, channel_index) in missing_coords
            if is_missing:
                values.append(None)
                missing.append(True)
            else:
                values.append(value)
                missing.append(False)
        rows.append({"seq": seq, "batch": batch, "values": values, "missing": missing})
    return rows


def write_example_bundle(target_dir: str | Path) -> tuple[Path, Path]:
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    config_path = target / "pipeline_config.json"
    samples_path = target / "samples.jsonl"
    config = build_example_config("samples.jsonl")
    atomic_write_json(config_path, config)
    with samples_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in generate_samples():
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return config_path, samples_path


def write_injection_bundle(example_dir: str | Path, injection_dir: str | Path) -> list[Path]:
    example_dir = Path(example_dir)
    injection_dir = Path(injection_dir)
    injection_dir.mkdir(parents=True, exist_ok=True)
    base_config = json.loads((example_dir / "pipeline_config.json").read_text(encoding="utf-8"))
    base_samples = (example_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    written: list[Path] = []

    cycle = json.loads(json.dumps(base_config))
    cycle["injection_id"] = "INJ-CONFIG-CYCLE-001"
    cycle["stages"][0]["upstream"] = "clip"
    cycle_path = injection_dir / "INJ-CONFIG-CYCLE-001.json"
    atomic_write_json(cycle_path, cycle)
    written.append(cycle_path)

    width_zero = json.loads(json.dumps(base_config))
    width_zero["injection_id"] = "INJ-WIDTH-ZERO-002"
    width_zero["fixed_point"]["decimal_width"] = 0
    width_zero_path = injection_dir / "INJ-WIDTH-ZERO-002.json"
    atomic_write_json(width_zero_path, width_zero)
    written.append(width_zero_path)

    width_range = json.loads(json.dumps(base_config))
    width_range["injection_id"] = "INJ-WIDTH-RANGE-003"
    width_range["fixed_point"]["decimal_width"] = 99
    width_range_path = injection_dir / "INJ-WIDTH-RANGE-003.json"
    atomic_write_json(width_range_path, width_range)
    written.append(width_range_path)

    bad_gear = json.loads(json.dumps(base_config))
    bad_gear["injection_id"] = "INJ-GEAR-ENUM-004"
    bad_gear["fixed_point"]["quantization_gear"] = "Q-BOGUS"
    bad_gear_path = injection_dir / "INJ-GEAR-ENUM-004.json"
    atomic_write_json(bad_gear_path, bad_gear)
    written.append(bad_gear_path)

    bad_threshold = json.loads(json.dumps(base_config))
    bad_threshold["injection_id"] = "INJ-THRESHOLD-RANGE-005"
    first_key = next(iter(bad_threshold["thresholds_bp"]))
    bad_threshold["thresholds_bp"][first_key] = 100_000
    threshold_path = injection_dir / "INJ-THRESHOLD-RANGE-005.json"
    atomic_write_json(threshold_path, bad_threshold)
    written.append(threshold_path)

    nonfinite_rows = list(base_samples)
    row = json.loads(nonfinite_rows[10])
    row["values"][0] = float("nan")
    nonfinite_rows[10] = json.dumps(row, ensure_ascii=False)
    nonfinite_path = injection_dir / "INJ-SAMPLE-NONFINITE-006.jsonl"
    nonfinite_path.write_text("\n".join(nonfinite_rows) + "\n", encoding="utf-8", newline="\n")
    written.append(nonfinite_path)

    duplicate_rows = list(base_samples)
    duplicate_rows[11] = duplicate_rows[10]
    duplicate_path = injection_dir / "INJ-SAMPLE-DUPLICATE-007.jsonl"
    duplicate_path.write_text("\n".join(duplicate_rows) + "\n", encoding="utf-8", newline="\n")
    written.append(duplicate_path)

    regressed_rows = list(base_samples)
    row = json.loads(regressed_rows[20])
    row["batch"] = 0
    regressed_rows[20] = json.dumps(row, ensure_ascii=False)
    regressed_path = injection_dir / "INJ-SAMPLE-BATCH-REGRESS-008.jsonl"
    regressed_path.write_text("\n".join(regressed_rows) + "\n", encoding="utf-8", newline="\n")
    written.append(regressed_path)

    runtime_faults = {
        "injection_id": "INJ-RUNTIME-BATCH-009",
        "compute_faults": [
            {
                "injection_id": "INJ-BATCH-COMPUTE-009A",
                "batch": 6,
                "stage": "log_stage",
                "code": "E_BATCH_COMPUTE",
            }
        ],
        "disk_faults": [
            {
                "injection_id": "INJ-SNAPSHOT-DISK-009B",
                "batch": 7,
                "stage": "clip",
                "code": "E_SNAPSHOT_WRITE",
            }
        ],
    }
    runtime_path = injection_dir / "INJ-RUNTIME-FAULTS-009.json"
    atomic_write_json(runtime_path, runtime_faults)
    written.append(runtime_path)

    manifest = [
        {
            "injection_id": "INJ-CONFIG-CYCLE-001",
            "kind": "config",
            "config": _repo_relative(cycle_path),
            "samples": _repo_relative(example_dir / "samples.jsonl"),
            "expected_code": "E_CONFIG_INVALID",
        },
        {
            "injection_id": "INJ-WIDTH-ZERO-002",
            "kind": "config",
            "config": _repo_relative(width_zero_path),
            "samples": _repo_relative(example_dir / "samples.jsonl"),
            "expected_code": "E_CONFIG_INVALID",
        },
        {
            "injection_id": "INJ-WIDTH-RANGE-003",
            "kind": "config",
            "config": _repo_relative(width_range_path),
            "samples": _repo_relative(example_dir / "samples.jsonl"),
            "expected_code": "E_CONFIG_INVALID",
        },
        {
            "injection_id": "INJ-GEAR-ENUM-004",
            "kind": "config",
            "config": _repo_relative(bad_gear_path),
            "samples": _repo_relative(example_dir / "samples.jsonl"),
            "expected_code": "E_CONFIG_INVALID",
        },
        {
            "injection_id": "INJ-THRESHOLD-RANGE-005",
            "kind": "config",
            "config": _repo_relative(threshold_path),
            "samples": _repo_relative(example_dir / "samples.jsonl"),
            "expected_code": "E_CONFIG_INVALID",
        },
        {
            "injection_id": "INJ-SAMPLE-NONFINITE-006",
            "kind": "sample",
            "config": _repo_relative(example_dir / "pipeline_config.json"),
            "samples": _repo_relative(nonfinite_path),
            "expected_code": "E_SAMPLE_INVALID",
        },
        {
            "injection_id": "INJ-SAMPLE-DUPLICATE-007",
            "kind": "sample",
            "config": _repo_relative(example_dir / "pipeline_config.json"),
            "samples": _repo_relative(duplicate_path),
            "expected_code": "E_SAMPLE_INVALID",
        },
        {
            "injection_id": "INJ-SAMPLE-BATCH-REGRESS-008",
            "kind": "sample",
            "config": _repo_relative(example_dir / "pipeline_config.json"),
            "samples": _repo_relative(regressed_path),
            "expected_code": "E_SAMPLE_INVALID",
        },
        {
            "injection_id": "INJ-RUNTIME-BATCH-009",
            "kind": "runtime",
            "runtime_config": _repo_relative(runtime_path),
            "config": _repo_relative(example_dir / "pipeline_config.json"),
            "samples": _repo_relative(example_dir / "samples.jsonl"),
            "expected_codes": ["E_BATCH_COMPUTE", "E_SNAPSHOT_WRITE"],
        },
    ]
    manifest_path = injection_dir / "injection_manifest.json"
    atomic_write_json(manifest_path, manifest)
    written.append(manifest_path)
    return written
