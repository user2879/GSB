"""加载校验层负样本：验证 E_DESC_INVALID / E_SHAPE_MISMATCH / E_BUDGET_INVALID。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fxpconv.errors import (
    E_BUDGET_INVALID,
    E_DESC_INVALID,
    E_SHAPE_MISMATCH,
)
from fxpconv.models import load_descriptor
from fxpconv.tensorpack import expected_main_shapes, load_tensor_pack

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    base = json.loads((SAMPLES / "operator_desc.json").read_text(encoding="utf-8"))
    work = Path(tempfile.mkdtemp())

    bad_desc = dict(base, acc_bits=13, rounding="bankers", mac_budget=-2,
                    stride_h=-1)
    desc_path = work / "bad_desc.json"
    write(desc_path, bad_desc)
    desc, ledger = load_descriptor(desc_path)
    codes = set(ledger.codes())
    assert desc is None
    assert E_DESC_INVALID in codes and E_BUDGET_INVALID in codes, codes
    print(f"[OK] 非法枚举/负维度 -> {E_DESC_INVALID}；预算非正 -> {E_BUDGET_INVALID}")

    good_desc, _ = load_descriptor(SAMPLES / "operator_desc.json")
    bad_pack = work / "bad_pack.jsonl"
    act_shape = [good_desc.batch_n, good_desc.in_channels, good_desc.in_h, good_desc.in_w]
    wrong_len = good_desc.batch_n * good_desc.in_channels * good_desc.in_h * good_desc.in_w + 3
    bad_pack.write_text(
        json.dumps({"name": "activation", "shape": act_shape, "dtype": "int8",
                    "signed": True, "data": [1] * wrong_len}) + "\n" +
        json.dumps({"name": "weight",
                    "shape": [good_desc.out_channels, good_desc.in_channels,
                              good_desc.k_h, good_desc.k_w],
                    "dtype": "int8", "signed": True,
                    "data": [1] * (good_desc.out_channels * good_desc.in_channels
                                   * good_desc.k_h * good_desc.k_w)}) + "\n",
        encoding="utf-8",
    )
    pack, pack_ledger = load_tensor_pack(bad_pack, good_desc,
                                         expected_main_shapes(good_desc))
    assert pack is None and E_SHAPE_MISMATCH in pack_ledger.codes()
    print(f"[OK] 数据长度与形状不符 -> {E_SHAPE_MISMATCH}，非法张量未进入后续阶段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
