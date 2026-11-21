#!/usr/bin/env python3
"""Generate the deterministic local dataset used by the example and tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcal.calibration import V1_HEADER, render_v1_record
from qcal.models import CalibrationRecord


LAYERS = {
    "layer.00": {"shape": (4, 8), "bit_width": 8, "thousandths": 8, "zero": -1, "v2_scale": 0.007563522},
    "layer.01": {"shape": (8, 8), "bit_width": 8, "thousandths": 8, "zero": 0, "v2_scale": 0.008177156},
    "layer.02": {"shape": (6, 8), "bit_width": 4, "thousandths": 16, "zero": 2, "v2_scale": 0.015708587},
    "layer.03": {"shape": (9, 8), "bit_width": 8, "thousandths": 8, "zero": 0, "v2_scale": 0.007528604},
}

V2_META = {
    "layer.00": {"scheme": "symmetric", "rounding_mode": "nearest_even", "precision_label": "fp16-ref", "per_channel": False},
    "layer.01": {"scheme": "asymmetric", "rounding_mode": "nearest_away", "precision_label": "int8-per-channel", "per_channel": True},
    "layer.02": {"scheme": "asymmetric", "rounding_mode": "nearest_even", "precision_label": "int4-group", "per_channel": False},
    "layer.03": {"scheme": "symmetric", "rounding_mode": "floor", "precision_label": "int8-outlier", "per_channel": False},
}


def make_weights(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}
    for index, (layer, spec) in enumerate(LAYERS.items()):
        shape = spec["shape"]
        # Layer 0 is exactly representable by the full-precision bypass. Other
        # layers have increasingly heavy tails so clipping and distillation are
        # visible without relying on external data.
        if layer == "layer.00":
            values = np.rint(rng.uniform(-0.75, 0.75, size=shape) * 1024.0) / 1024.0
        elif layer == "layer.01":
            values = rng.normal(0.0, 0.34, size=shape)
        elif layer == "layer.02":
            values = rng.normal(0.0, 0.46, size=shape)
        else:
            values = rng.normal(0.18, 0.52, size=shape)
            flat = values.reshape(-1)
            outlier_indices = rng.choice(flat.size, size=10, replace=False)
            flat[outlier_indices] += rng.normal(1.05, 0.25, size=outlier_indices.size)
        weights[layer] = np.asarray(values, dtype=np.float64).reshape(shape)
    return weights


def make_samples(seed: int, weights: dict[str, np.ndarray], count: int = 6) -> list[dict]:
    rng = np.random.default_rng(seed + 100_003)
    samples = []
    for index in range(count):
        sample_id = f"sample-{index:02d}"
        vector = rng.normal(0.0, 0.55, size=8)
        reference = {
            layer: [float(value) for value in (weights[layer] @ vector).tolist()]
            for layer in sorted(weights)
        }
        samples.append(
            {
                "id": sample_id,
                "input": [float(value) for value in vector.tolist()],
                "reference": reference,
            }
        )
    return samples


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def generate(out_dir: Path, seed: int = 20261121) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = make_weights(seed)
    with (out_dir / "weights.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for layer in sorted(weights):
            row = {
                "layer": layer,
                "shape": list(weights[layer].shape),
                "weights": weights[layer].tolist(),
                "content_version": f"seed-{seed}-{layer}-r1",
            }
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False))
            handle.write("\n")
    samples = make_samples(seed, weights)
    write_json(
        out_dir / "calibration_samples.json",
        {
            "schema_version": "qcal-samples/1.0",
            "seed": seed,
            "samples": samples,
        },
    )

    v1_records: dict[str, CalibrationRecord] = {}
    for layer, spec in LAYERS.items():
        v1_records[layer] = CalibrationRecord(
            layer=layer,
            scale=float(spec["thousandths"]) / 1000.0,
            zero_point=int(spec["zero"]),
            bit_width=int(spec["bit_width"]),
            scheme="affine",
            rounding_mode="nearest_even",
            precision_label="v1-thousandth-implicit",
            per_channel=False,
            source_version="v1",
            provenance={"scale_thousandths": int(spec["thousandths"])},
        )
    v1_lines = [V1_HEADER]
    for layer in sorted(v1_records):
        v1_lines.append(render_v1_record(v1_records[layer], int(LAYERS[layer]["thousandths"])))
    (out_dir / "calibration_v1.txt").write_text("\n".join(v1_lines) + "\n", encoding="utf-8", newline="\n")

    v2_layers = []
    for layer in sorted(LAYERS):
        spec = LAYERS[layer]
        meta = V2_META[layer]
        v2_layers.append(
            {
                "layer": layer,
                "scale": float(spec["v2_scale"]),
                "zero_point": int(spec["zero"]),
                "bit_width": int(spec["bit_width"]),
                **meta,
            }
        )
    write_json(
        out_dir / "calibration_v2.json",
        {
            "schema_version": "qcal-calibration/2.0",
            "exchange_version": "v2",
            "layers": v2_layers,
        },
    )

    write_json(
        out_dir / "exchange_spec.json",
        {
            "schema_version": "qcal-exchange-spec/1.0",
            "seed": seed,
            "versions": {
                "v1": {
                    "encoding": "fixed_width_text",
                    "field_widths": {"layer": 20, "scale_thousandths": 8, "zero_point": 8, "bit_width": 4},
                    "scale_semantics": "integer_thousandths",
                },
                "v2": {
                    "encoding": "json",
                    "scale_semantics": "binary_float64",
                    "named_fields": ["scheme", "rounding_mode", "precision_label", "per_channel"],
                },
            },
            "lossy_v2_to_v1": ["binary_float_scale", "rounding_mode", "precision_label", "per_channel"],
            "sidecar_required_for_reverse": True,
        },
    )

    config = {
        "schema_version": "qcal-config/1.0",
        "seed": seed,
        "exchange_version": "v1",
        "calibration": {
            "quantile_lower": 0.02,
            "quantile_upper": 0.98,
            "interpolation": "linear",
            "tie_break": "lower_index",
            "scheme": "asymmetric",
            "per_channel": False,
            "group_size": 0,
        },
        "fixed_point": {"fraction_bits": 10, "rounding_mode": "nearest_even", "saturation": "saturate"},
        "distillation": {"temperature": 1.5, "soft_label_weight": 0.35},
        "scheduler": {"workers": 4, "quant_points_budget": 0},
        "snapshot": {"commit_every_stages": 1},
        "full_precision_layers": ["layer.00"],
        "layers": {
            "layer.00": {
                "bit_width": 8,
                "group_size": 0,
                "group_size_candidates": [0, 4, 8],
                "scheme": "asymmetric",
                "per_channel": False,
            },
            "layer.01": {
                "bit_width": 8,
                "group_size": 0,
                "group_size_candidates": [0, 4, 8],
                "scheme": "asymmetric",
                "per_channel": False,
            },
            "layer.02": {
                "bit_width": 4,
                "group_size": 0,
                "group_size_candidates": [0, 4, 8],
                "scheme": "asymmetric",
                "per_channel": False,
            },
            "layer.03": {
                "bit_width": 8,
                "group_size": 0,
                "group_size_candidates": [0, 4, 8],
                "scheme": "asymmetric",
                "per_channel": False,
            },
        },
        "bit_width_candidates": [4, 8],
        "fixed_point_fraction_bits_candidates": [6, 8, 10, 12],
    }
    write_json(out_dir / "config.yaml", config)

    write_json(
        out_dir / "injections.yaml",
        {
            "schema_version": "qcal-injections/1.0",
            "injections": [
                {
                    "id": "INJ-001",
                    "type": "non_finite_weight",
                    "layer": "layer.01",
                    "index": [0, 0],
                    "value": "nan",
                    "expected_code": "E_WEIGHT_NONFINITE",
                },
                {
                    "id": "INJ-002",
                    "type": "zero_scale",
                    "layer": "layer.02",
                    "expected_code": "E_CALIB_PARSE",
                },
                {
                    "id": "INJ-003",
                    "type": "zero_point_out_of_range",
                    "layer": "layer.03",
                    "value": 999,
                    "expected_code": "E_CALIB_PARSE",
                },
                {
                    "id": "INJ-004",
                    "type": "sidecar_missing",
                    "layer": "layer.00",
                    "expected_code": "E_SIDECAR_MISSING",
                },
            ],
        },
    )

    write_json(
        out_dir / "edit_sequence.yaml",
        {
            "schema_version": "qcal-edit-sequence/1.0",
            "steps": [
                {"id": "edit-01", "set": {"exchange_version": "v2"}},
                {"id": "edit-02", "layer_set": {"layer.02": {"bit_width": 8}}},
                {"id": "edit-03", "layer_set": {"layer.03": {"group_size": 4}}},
                {"id": "edit-04", "set": {"fixed_point.rounding_mode": "nearest_away"}},
                {
                    "id": "edit-05",
                    "replace_weight": {
                        "layer": "layer.01",
                        "seed": seed + 5001,
                        "scale": 0.29,
                        "content_version": "replacement-01",
                    },
                },
                {"id": "edit-06", "set": {"distillation.temperature": 2.0}},
            ],
        },
    )

    write_json(
        out_dir / "manifest.json",
        {
            "schema_version": "qcal-sample-bundle/1.0",
            "seed": seed,
            "layers": sorted(LAYERS),
            "layer_shapes": {layer: list(spec["shape"]) for layer, spec in LAYERS.items()},
            "sample_count": len(samples),
            "generator": "scripts/generate_samples.py",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples")
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()
    generate(Path(args.out), args.seed)


if __name__ == "__main__":
    main()
