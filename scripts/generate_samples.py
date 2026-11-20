"""Generate the fixed-seed local demonstration package.

Usage: python scripts/generate_samples.py [output_dir]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantcal.calibration import (  # noqa: E402
    calibrate_quantiles,
    derive_scale_zero,
    export_quantile_range,
    parse_samples,
    parse_tensor_specs,
    tensor_values_and_refs,
)
from quantcal.config import parse_config  # noqa: E402
from quantcal.format_v2 import downgrade_v2_to_v1, to_v2_document  # noqa: E402
from quantcal.models import ExchangeSet, TensorParam, scheme_saturation_max  # noqa: E402
from quantcal.util import atomic_write_json, atomic_write_text, canonical_json_text  # noqa: E402


SEED = 20261121
TENSOR_SPECS = [
    {"name": "attn.q.weight", "shape": [8], "distribution": "normal", "value_scale": 0.9},
    {"name": "attn.k.weight", "shape": [8], "distribution": "normal", "value_scale": 1.2},
    {"name": "mlp.up.weight", "shape": [10], "distribution": "heavy_tail", "value_scale": 2.0},
    {"name": "mlp.down.weight", "shape": [6], "distribution": "uniform", "value_scale": 1.0},
    {"name": "norm.weight", "shape": [5], "distribution": "constant", "value_scale": 0.75},
    {"name": "zero.weight", "shape": [4], "distribution": "zero", "value_scale": 0.0},
    {"name": "probe.nonfinite", "shape": [4], "distribution": "normal", "value_scale": 0.8},
    {"name": "probe.bits", "shape": [4], "distribution": "normal", "value_scale": 1.0},
    {"name": "probe.range", "shape": [4], "distribution": "normal", "value_scale": 1.0},
]


def _sample_rng(sample_id: str, tensor_name: str) -> np.random.Generator:
    token = hashlib.sha256(f"{SEED}:{sample_id}:{tensor_name}".encode("utf-8")).digest()
    derived_seed = int.from_bytes(token[:8], "little", signed=False)
    return np.random.default_rng(derived_seed)


def _values_for(spec: dict[str, Any], sample_id: str) -> list[float]:
    count = int(np.prod(spec["shape"]))
    rng = _sample_rng(sample_id, spec["name"])
    distribution = spec["distribution"]
    scale = float(spec["value_scale"])
    if distribution == "normal":
        values = rng.normal(0.0, scale, count)
    elif distribution == "heavy_tail":
        values = rng.standard_t(df=3, size=count) * scale
        values[0] += 18.0 * scale
        values[-1] -= 12.0 * scale
    elif distribution == "uniform":
        values = rng.uniform(-1.5 * scale, 1.5 * scale, count)
    elif distribution == "constant":
        values = np.full(count, 0.75)
    elif distribution == "zero":
        values = np.zeros(count)
    else:
        values = rng.normal(0.0, scale, count)
    return [round(float(value), 10) for value in values]


def build_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = {
        "format": "quantcal.tensor-manifest.v1",
        "seed": SEED,
        "tensors": [
            {
                "name": spec["name"],
                "shape": spec["shape"],
                "distribution": spec["distribution"],
                "special_positions": (
                    [{"index": 0, "kind": "tail_outlier"}, {"index": spec["shape"][0] - 1, "kind": "tail_outlier"}]
                    if spec["distribution"] == "heavy_tail"
                    else []
                ),
            }
            for spec in TENSOR_SPECS
        ],
    }
    config = {
        "format": "quantcal.config.v1",
        "quantile_lower_permille": 10,
        "quantile_upper_permille": 990,
        "quantization_bits_candidates": [8, 12],
        "quantization_bits": 8,
        "fixed_fractional_bits": 16,
        "rounding_mode_candidates": [
            "nearest_even",
            "nearest_away",
            "floor",
            "ceil",
            "truncate",
        ],
        "rounding_mode": "nearest_even",
        "saturation_strategy": "saturate",
        "max_concurrency": 3,
        "batch_point_budget": 100000,
        "snapshot_commit_interval": 2,
        "quantization_scheme": "symmetric",
        "tensor_overrides": {
            "mlp.up.weight": {
                "quantization_scheme": "asymmetric",
                "rounding_mode": "nearest_away",
            },
            "norm.weight": {"quantization_scheme": "asymmetric"},
        },
    }
    count = len(TENSOR_SPECS)
    sample_ids = [f"cal-{index:04d}" for index in range(count * 3)]
    samples: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids):
        assigned = {index % count, (index + 4) % count}
        features: dict[str, list[float]] = {}
        for tensor_index in sorted(assigned):
            spec = TENSOR_SPECS[tensor_index]
            features[spec["name"]] = _values_for(spec, sample_id)
        samples.append({"sample_id": sample_id, "features": features})
    return manifest, config, samples


def _build_exchanges(
    specs: list[Any], config: Any, samples: list[Any]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    tensors: list[TensorParam] = []
    for spec in sorted(specs, key=lambda item: item.name):
        resolved = config.resolve(spec.name)
        values, refs, layout = tensor_values_and_refs(spec.name, samples)
        quantiles = calibrate_quantiles(spec.name, values, refs, layout, resolved)
        scale_result = derive_scale_zero(spec.name, quantiles, resolved)
        range_lower, range_upper, range_expanded = export_quantile_range(
            float(quantiles["quantile_lower"]), float(quantiles["quantile_upper"])
        )
        bits = int(scale_result["quantization_bits"])
        scheme = str(scale_result["quantization_scheme"])
        tensors.append(
            TensorParam(
                name=spec.name,
                scale=float(scale_result["scale"]),
                zero_point=int(scale_result["zero_point"]),
                scheme=scheme,
                rounding=resolved.rounding_mode,
                precision_tag=f"seed_generated_q{bits}",
                quantization_bits=bits,
                saturation_max=scheme_saturation_max(scheme, bits),
                quantile_lower=range_lower,
                quantile_upper=range_upper,
                source={
                    "generator": "scripts/generate_samples.py",
                    "seed": SEED,
                    "scale_guard": scale_result.get("scale_guard"),
                    "measured_quantile_range": [
                        float(quantiles["quantile_lower"]),
                        float(quantiles["quantile_upper"]),
                    ],
                    "quantile_range_expanded_for_zero_span": range_expanded,
                },
            )
        )
    v2_exchange = ExchangeSet(
        version="v2", quantization_bits=tensors[0].quantization_bits, tensors=tensors
    )
    v2_document = to_v2_document(v2_exchange)
    _, v1_text, sidecar = downgrade_v2_to_v1(v2_exchange)
    return v1_text, v2_document, sidecar


def generate(output_dir: str | Path) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest, config, samples = build_inputs()
    specs = parse_tensor_specs(manifest)
    parsed_config = parse_config(config)
    parsed_samples = parse_samples(samples, specs)
    v1_text, v2_document, downgrade_sidecar = _build_exchanges(specs, parsed_config, parsed_samples)

    nonfinite_sample = next(
        row["sample_id"]
        for row in samples
        if "probe.nonfinite" in row["features"]
    )
    injections = {
        "format": "quantcal.injections.v1",
        "seed": SEED,
        "injections": [
            {
                "id": "INJ-001",
                "kind": "nonfinite_sample",
                "input_file": "samples.jsonl",
                "jsonl_sample_id": nonfinite_sample,
                "tensor": "probe.nonfinite",
                "element_index": 0,
                "value": "NaN",
                "expected_code": "E_NON_FINITE_SAMPLE",
                "description": "inject one real NaN into an existing calibration vector",
            },
            {
                "id": "INJ-002",
                "kind": "quantile_bounds_inverted",
                "input_file": "config.json",
                "tensor": "probe.range",
                "lower_permille": 900,
                "upper_permille": 100,
                "expected_code": "E_QUANT_RANGE_INVALID",
                "description": "per-tensor lower quantile is greater than upper",
            },
            {
                "id": "INJ-003",
                "kind": "quantization_bits_out_of_range",
                "input_file": "config.json",
                "tensor": "probe.bits",
                "quantization_bits": 1,
                "expected_code": "E_QUANT_RANGE_INVALID",
                "description": "per-tensor bit width is below the supported minimum",
            },
            {
                "id": "INJ-004",
                "kind": "snapshot_version_tamper",
                "runtime_file": "state/snapshot.json",
                "field": "global_version",
                "expected_code": "E_SNAPSHOT_STALE",
                "description": "modify a committed snapshot version while preserving its old hash",
            },
        ],
    }

    atomic_write_json(target / "tensors.json", manifest)
    atomic_write_json(target / "config.json", config)
    sample_lines = [
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        for row in samples
    ]
    atomic_write_text(target / "samples.jsonl", "\n".join(sample_lines) + "\n")
    atomic_write_text(target / "calibration_v1.txt", v1_text)
    atomic_write_text(target / "calibration_v2.json", canonical_json_text(v2_document))
    atomic_write_json(target / "calibration_v2_to_v1_sidecar.json", downgrade_sidecar)
    atomic_write_json(target / "injections.json", injections)
    return target


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sample_data"
    generate(output)
    print(f"generated sample package: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
