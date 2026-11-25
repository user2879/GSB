"""固定种子确定性样本生成：多层激活张量包、算子配置、v1/v2 权重、命令流。"""

from __future__ import annotations

import json
import os

import numpy as np

from .models import (
    BIT_WIDTH_TO_SCHEME,
    EditCommand,
    InjectionItem,
    OpConfig,
    TensorPackage,
    V1WeightRow,
    V2WeightRow,
)
from .weights import write_v1, write_v2

SEED = 20261125
LAYER_SPECS = [
    ("embedding", 4, 6, []),
    ("attention", 4, 6, ["embedding"]),
    ("projection", 3, 6, ["attention"]),
]


def generate_package(seed: int = SEED) -> TensorPackage:
    rng = np.random.default_rng(seed)
    layers: dict[str, list[list[float]]] = {}
    depends_on: dict[str, list[str]] = {}
    for name, rows, cols, deps in LAYER_SPECS:
        matrix = rng.uniform(-2.5, 2.5, size=(rows, cols))
        # 在 attention 层制造一行零方差（全部相同）用于退化行验证
        if name == "attention":
            matrix[2, :] = 1.25
        layers[name] = [[float(value) for value in row] for row in matrix]
        depends_on[name] = deps
    return TensorPackage(layers=layers, depends_on=depends_on, seed=seed)


def generate_config() -> OpConfig:
    return OpConfig(
        frac_bits=6,
        tier="standard",
        rounding_mode="HALF_EVEN",
        max_workers=3,
        element_budget=24,
        layer_row_quota=3,
        tolerances={"abs": 5e-2, "rel": 5e-2},
        config_version=1,
    )


def generate_weights(package: TensorPackage):
    rng = np.random.default_rng(SEED + 1)
    v1_rows: list[V1WeightRow] = []
    v2_rows: list[V2WeightRow] = []
    bit_widths = (8, 16, 32)
    for layer, matrix in package.layers.items():
        for row_index in range(len(matrix)):
            bit_width = bit_widths[(row_index + len(layer)) % 3]
            scale_permille = int(rng.integers(500, 2000))
            bias_code = int(rng.integers(-200, 200))
            v1_rows.append(
                V1WeightRow(layer, row_index, scale_permille, bias_code, bit_width)
            )
            scheme = BIT_WIDTH_TO_SCHEME[bit_width]
            # v2 保留三位以上小数，降级 v1 时必然产生千分比截断残差
            v2_scale = (scale_permille + 0.37) / 1000.0
            v2_rows.append(
                V2WeightRow(
                    layer=layer,
                    row_index=row_index,
                    scale=v2_scale,
                    scheme=scheme.value,
                    rounding_mode="HALF_EVEN",
                )
            )
    return v1_rows, v2_rows


def generate_commands() -> list[EditCommand]:
    return [
        EditCommand(1, "switch_version", {"version": "v2"}),
        EditCommand(
            2,
            "replace_layer_params",
            {
                "layer": "attention",
                "scale": 1.333,
                "scheme": "Q8.8_NORM_NO_BIAS",
                "rounding_mode": "HALF_UP",
            },
        ),
        EditCommand(
            3,
            "replace_row_activation",
            {
                "layer": "embedding",
                "row_index": 1,
                "activation": [0.4, -0.8, 1.6, -2.0, 0.25, 1.1],
            },
        ),
        EditCommand(4, "set_frac_bits", {"frac_bits": 8}),
        EditCommand(5, "set_tier", {"tier": "fine"}),
        EditCommand(6, "switch_version", {"version": "v1"}),
    ]


def generate_injections() -> list[InjectionItem]:
    return [
        InjectionItem("INJ-01", "non_finite_scale", "embedding", 0,
                      {"scale": float("inf")}),
        InjectionItem("INJ-02", "zero_variance", "attention", 2),
        InjectionItem("INJ-03", "bad_bitwidth", "projection", 0,
                      {"bit_width": 13}),
        InjectionItem("INJ-04", "duplicate_layer", "embedding"),
    ]


def write_samples(samples_dir: str = "samples") -> dict[str, str]:
    os.makedirs(samples_dir, exist_ok=True)
    package = generate_package()
    config = generate_config()
    v1_rows, v2_rows = generate_weights(package)
    commands = generate_commands()

    package_path = os.path.join(samples_dir, "tensor_package.json")
    with open(package_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "seed": package.seed,
                "layers": package.layers,
                "depends_on": package.depends_on,
            },
            handle, ensure_ascii=False, indent=2, sort_keys=True,
        )
        handle.write("\n")

    config_path = os.path.join(samples_dir, "op_config.json")
    with open(config_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(config.__dict__, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    v1_path = os.path.join(samples_dir, "weights_v1.txt")
    v2_path = os.path.join(samples_dir, "weights_v2.json")
    write_v1(v1_path, v1_rows)
    write_v2(v2_path, v2_rows)

    commands_path = os.path.join(samples_dir, "commands.json")
    with open(commands_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            [{"seq": cmd.seq, "kind": cmd.kind, "payload": cmd.payload} for cmd in commands],
            handle, ensure_ascii=False, indent=2, sort_keys=True,
        )
        handle.write("\n")

    injections_path = os.path.join(samples_dir, "injections.json")
    with open(injections_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            [
                {
                    "inject_id": item.inject_id,
                    "fault": item.fault,
                    "target_layer": item.target_layer,
                    "target_row": item.target_row,
                    "payload": item.payload,
                }
                for item in generate_injections()
            ],
            handle, ensure_ascii=False, indent=2, sort_keys=True,
        )
        handle.write("\n")

    return {
        "package": package_path,
        "config": config_path,
        "v1": v1_path,
        "v2": v2_path,
        "commands": commands_path,
        "injections": injections_path,
    }
