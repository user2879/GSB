"""按固定种子生成本地确定性样本（无网络、无 GPU）。

产物：
  samples/operator_desc.json
  samples/tensor_pack.jsonl
  samples/calibration.json
  samples/injections.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fxpconv.models import load_descriptor
from fxpconv.reference import reference_position
from fxpconv.tensorpack import Tensor, TensorPack, expected_main_shapes

SEED = 20261121
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "samples"


def _write_jsonl_tensor(handle, tensor: Tensor) -> None:
    handle.write(
        json.dumps(
            {
                "name": tensor.name,
                "shape": list(tensor.shape),
                "dtype": tensor.dtype,
                "signed": tensor.signed,
                "data": list(tensor.data),
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    desc_payload = {
        "name": "fixed-point-conv2d-demo",
        "batch_n": 2,
        "in_channels": 2,
        "in_h": 5,
        "in_w": 5,
        "out_channels": 4,
        "k_h": 3,
        "k_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 1,
        "acc_bits": 16,
        "qbits": 8,
        "saturation": "saturate",
        "rounding": "round_nearest_even",
        "frac_bits": 6,
        "out_low": -2.0,
        "out_high": 2.0,
        "concurrency": 4,
        "mac_budget": 60,
        "snapshot_every": 16,
        "act_scale": 0.015625,
        "weight_scales": [0.03125, 0.0625, 0.046875, 0.09375],
        "scale_sources": {
            "act_scale": "samples/tensor_pack.jsonl#activation:dtype=int8",
            "weight_scales": "per-channel calibration table weight_scale[oc]",
        },
        "tolerance": {
            "accum": {"abs": 0, "rel": 0.0},
            "trunc": {"abs": 2, "rel": 0.05},
            "deq": {"abs": 0.25, "rel": 0.1},
        },
    }
    (SAMPLE_DIR / "operator_desc.json").write_text(
        json.dumps(desc_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    desc, ledger = load_descriptor(SAMPLE_DIR / "operator_desc.json")
    assert desc is not None, [item.to_dict() for item in ledger]
    shapes = expected_main_shapes(desc)

    activation = Tensor(
        name="activation",
        shape=shapes["activation"],
        dtype="int8",
        signed=True,
        data=tuple(int(v) for v in rng.integers(-128, 128, size=shapes["activation"], dtype=np.int64).flat),
        row_ids=(),
    )
    weight = Tensor(
        name="weight",
        shape=shapes["weight"],
        dtype="int8",
        signed=True,
        data=tuple(int(v) for v in rng.integers(-128, 128, size=shapes["weight"], dtype=np.int64).flat),
        row_ids=(),
    )
    # 重新生成带行标识的张量。
    from fxpconv.tensorpack import _row_ids
    activation = Tensor(
        activation.name, activation.shape, activation.dtype, True,
        activation.data, _row_ids(activation.name, activation.shape),
    )
    weight = Tensor(
        weight.name, weight.shape, weight.dtype, True,
        weight.data, _row_ids(weight.name, weight.shape),
    )

    with (SAMPLE_DIR / "tensor_pack.jsonl").open("w", encoding="utf-8") as handle:
        _write_jsonl_tensor(handle, activation)
        _write_jsonl_tensor(handle, weight)

    pack = TensorPack(
        tensors={"activation": activation, "weight": weight},
        package_fingerprint="",
    )

    # 校准样本：固定种子的确定性输入 + float64 参考输出。
    calib_rng = np.random.default_rng(SEED + 1)
    samples = []
    for sample_index in range(3):
        calib_act = tuple(
            int(v)
            for v in calib_rng.integers(-128, 128, size=shapes["activation"], dtype=np.int64).flat
        )
        override = Tensor(
            "activation",
            activation.shape,
            activation.dtype,
            True,
            calib_act,
            activation.row_ids,
        )
        reference = []
        for position in range(desc.total_positions):
            _, _, truth = reference_position(desc, pack, position, activation_override=override)
            reference.append(truth)
        samples.append(
            {
                "sample_id": f"calib-{sample_index:02d}",
                "activation": list(calib_act),
                "reference_output": reference,
            }
        )
    (SAMPLE_DIR / "calibration.json").write_text(
        json.dumps({"samples": samples}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # 异常注入清单：四类真实故障，目标位置取自确定性输出布局。
    target_a = desc.positions_per_sample  # 第二批次首个位置
    target_b = target_a + 5
    target_c = target_a + 10
    target_d = 3
    injections = {
        "injections": [
            {
                "id": "INJ-01",
                "kind": "nonfinite_scale",
                "target_position": target_a,
                "target_tensor": "weight",
                "stage": "dequant",
                "params": {"value": float("nan")},
            },
            {
                "id": "INJ-02",
                "kind": "qbits_out_of_range",
                "target_position": target_b,
                "target_tensor": None,
                "stage": "truncate",
                "params": {"qbits": 3},
            },
            {
                "id": "INJ-03",
                "kind": "data_out_of_shape",
                "target_position": None,
                "target_tensor": "activation",
                "stage": "load",
                "params": {"bad_index": len(activation.data), "bad_value": 999999},
            },
            {
                "id": "INJ-04",
                "kind": "illegal_strategy_combo",
                "target_position": None,
                "target_tensor": None,
                "stage": "validate",
                "params": {"saturation": "wrap", "acc_bits": 8, "qbits": 8},
            },
            {
                "id": "INJ-05",
                "kind": "nonfinite_scale",
                "target_position": target_c,
                "target_tensor": "weight",
                "stage": "dequant",
                "params": {"value": float("inf")},
            },
            {
                "id": "INJ-06",
                "kind": "qbits_out_of_range",
                "target_position": target_d,
                "target_tensor": None,
                "stage": "truncate",
                "params": {"qbits": 12},
            },
        ]
    }
    # NaN/Infinity 在 JSON 中以字符串显式标注，加载侧转回浮点。
    text = json.dumps(injections, ensure_ascii=False, indent=2, sort_keys=True)
    text = text.replace('"NaN"', "NaN").replace('"Infinity"', "Infinity")
    (SAMPLE_DIR / "injections.json").write_text(text, encoding="utf-8")

    print(f"样本已生成到 {SAMPLE_DIR}")
    print(f"输出布局: positions={desc.total_positions}, "
          f"macs/position={desc.macs_per_position}, out={desc.out_h}x{desc.out_w}")


if __name__ == "__main__":
    main()
