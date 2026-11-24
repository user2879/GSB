"""Deterministic sample generation for the local lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from .loaders import make_v1_line


LAYER_SPECS = [
    {
        "layer": "layer_alpha",
        "seed": 101,
        "rows": 4,
        "dim": 6,
        "weights": [(1200, -30), (1150, -10), (1050, 0), (950, 20)],
    },
    {
        "layer": "layer_beta",
        "seed": 202,
        "rows": 3,
        "dim": 5,
        "weights": [(1300, 10), (1250, -20), (1100, 5)],
    },
    {
        "layer": "layer_gamma",
        "seed": 303,
        "rows": 4,
        "dim": 4,
        "weights": [(900, 0), (1000, 15), (1100, -15), (1200, 25)],
    },
]


def _canonical_line(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_sample_files(root: Path) -> Dict[str, Path]:
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    activation_path = samples / "activations.jsonl"
    v1_path = samples / "weights_v1.txt"
    v2_path = samples / "weights_v2.jsonl"
    config_path = samples / "config.json"
    commands_path = samples / "commands.jsonl"
    injections_path = samples / "injections.json"
    widths = {
        "layer": 12,
        "row": 5,
        "scale_permille": 8,
        "bias_code": 8,
        "quant_bits": 3,
    }
    activation_lines = []
    v1_lines = []
    v2_values = []
    for spec in LAYER_SPECS:
        rng = np.random.default_rng(int(spec["seed"]))
        matrix = rng.normal(0.0, 0.85, size=(spec["rows"], spec["dim"]))
        # Deterministic, finite, sub-percent perturbations make quantization
        # level changes observable without relying on random runtime choices.
        matrix += rng.uniform(-0.04, 0.04, size=matrix.shape)
        matrix = np.round(matrix, 9)
        activation = [[float(value) for value in row] for row in matrix]
        reference_scale = []
        reference_bias = []
        for scale_permille, bias_code in spec["weights"]:
            reference_scale.append([scale_permille / 1000.0] * spec["dim"])
            reference_bias.append([bias_code / 1000.0] * spec["dim"])
        activation_lines.append(
            _canonical_line(
                {
                    "layer": spec["layer"],
                    "seed": spec["seed"],
                    "rows": spec["rows"],
                    "dim": spec["dim"],
                    "activation": activation,
                    "layernorm": {
                        "scale": reference_scale,
                        "bias": reference_bias,
                    },
                }
            )
        )
        for row_index, (scale_permille, bias_code) in enumerate(spec["weights"]):
            v1_lines.append(
                make_v1_line(
                    layer=str(spec["layer"]),
                    row=row_index,
                    scale_permille=scale_permille,
                    bias_code=bias_code,
                    quant_bits=16,
                    widths=widths,
                )
            )
            v2_values.append(
                {
                    "layer": spec["layer"],
                    "row": row_index,
                    "scale": scale_permille / 1000.0,
                    "bias": bias_code / 1000.0,
                    "scheme": "layernorm_affine",
                    "precision": "int16",
                }
            )
    activation_path.write_text("\n".join(activation_lines) + "\n", encoding="utf-8")
    v1_path.write_text("\n".join(v1_lines) + "\n", encoding="utf-8")
    v2_path.write_text("\n".join(_canonical_line(value) for value in v2_values) + "\n", encoding="utf-8")

    config = {
        "schema": "opexlab.config.v1",
        "exchange_version": "v1",
        "paths": {
            "activations": "samples/activations.jsonl",
            "weights_v1": "samples/weights_v1.txt",
            "weights_v2": "samples/weights_v2.jsonl",
            "commands": "samples/commands.jsonl",
            "injections": "samples/injections.json",
            "runtime": "runtime",
        },
        "fixed_point": {
            "fractional_bits": 12,
            "quant_level": 1,
            "allowed_quant_levels": [0, 1, 2, 3],
        },
        "softmax": {
            "exp_approximation": "range_reduced_ln2_taylor",
            "polynomial_base_order": 4,
            "max_tie_rule": "first_smallest_column_index",
            "zero_denominator_rule": "uniform_with_remainder_at_column_zero",
        },
        "layernorm": {"variance_floor": 0.000001},
        "limits": {
            "max_workers": 4,
            "batch_element_budget": 1000,
            "per_layer_element_quota": 1000,
        },
        "allowed_norm_schemes": ["layernorm_affine", "layernorm_scale_only"],
        "scheme_bias_policy": {
            "layernorm_affine": True,
            "layernorm_scale_only": False,
        },
        "v1_upgrade_scheme": "layernorm_affine",
        "v1_field_widths": widths,
        "quant_codebook": {"int8": 8, "int16": 16, "int32": 32},
    }
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    commands = [
        {"id": "cmd-001", "type": "set_exchange_version", "value": "v2"},
        {"id": "cmd-002", "type": "set_fractional_bits", "value": 10},
        {"id": "cmd-003", "type": "set_quant_level", "value": 2},
        {
            "id": "cmd-004",
            "type": "replace_activation_row",
            "layer": "layer_beta",
            "row": 1,
            "values": [1.1, -0.4, 0.2, 0.9, -1.3],
        },
        {
            "id": "cmd-005",
            "type": "replace_layer_params",
            "layer": "layer_gamma",
            "scale": 1.35,
            "bias": 0.02,
            "scheme": "layernorm_affine",
            "precision": "int16",
        },
        {"id": "cmd-006", "type": "set_max_workers", "value": 2},
        {"id": "cmd-007", "type": "set_exchange_version", "value": "v1"},
        {"id": "cmd-008", "type": "set_batch_element_budget", "value": 1000},
    ]
    commands_path.write_text(
        "".join(_canonical_line(command) + "\n" for command in commands), encoding="utf-8"
    )
    injections = {
        "schema": "opexlab.injections.v1",
        "injections": [
            {
                "id": "INJ-001",
                "kind": "truncated_weight_line",
                "target": {"layer": "layer_beta", "row": 1},
                "expected_code": "E_FIELD_TRUNCATED",
            },
            {
                "id": "INJ-002",
                "kind": "nonfinite_activation",
                "target": {"layer": "layer_alpha", "row": 2},
                "expected_code": "E_NONFINITE",
            },
            {
                "id": "INJ-003",
                "kind": "unknown_scheme",
                "target": {"layer": "layer_gamma", "row": 0},
                "expected_code": "E_SCHEME_UNKNOWN",
            },
            {
                "id": "INJ-004",
                "kind": "width_out_of_range",
                "target": {"layer": "layer_alpha", "row": 1},
                "expected_code": "E_WIDTH_OUT_OF_RANGE",
            },
            {
                "id": "INJ-005",
                "kind": "budget_smaller_than_row",
                "target": {"layer": "layer_alpha", "row": 0},
                "expected_code": "E_BUDGET_TOO_SMALL",
            },
        ],
    }
    injections_path.write_text(
        json.dumps(injections, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "config": config_path,
        "activations": activation_path,
        "weights_v1": v1_path,
        "weights_v2": v2_path,
        "commands": commands_path,
        "injections": injections_path,
    }
