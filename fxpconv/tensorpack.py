"""整数激活/权重包（JSONL）数据模型与加载校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import E_SHAPE_MISMATCH, FailureLedger
from .fingerprint import combine_fingerprints, ints_fingerprint
from .models import OpDesc, int_range


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    signed: bool
    data: tuple[int, ...]
    row_ids: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return ints_fingerprint(self.data)

    def get(self, *indices) -> int:
        index = 0
        stride = 1
        for dim, coord in zip(reversed(self.shape), reversed(indices)):
            if coord < 0 or coord >= dim:
                raise IndexError(f"张量 {self.name} 坐标 {indices} 越界 shape={self.shape}")
            index += coord * stride
            stride *= dim
        return self.data[index]


@dataclass
class TensorPack:
    tensors: dict[str, Tensor]
    package_fingerprint: str

    def get(self, name: str) -> Tensor:
        return self.tensors[name]


def _dtype_bounds(dtype: str, signed: bool) -> tuple[int, int]:
    bits = int(dtype.replace("int", "").replace("uint", ""))
    if signed:
        return int_range(bits)
    return 0, (1 << bits) - 1


def _row_ids(name: str, shape: tuple[int, ...]) -> tuple[str, ...]:
    if len(shape) < 2:
        return (f"{name}:row0",)
    row_count = 1
    for dim in shape[:-1]:
        row_count *= dim
    return tuple(f"{name}:r{row}:c0..{shape[-1] - 1}" for row in range(row_count))


def load_tensor_pack(
    path: str | Path,
    desc: OpDesc,
    expected: dict[str, tuple[int, ...]],
) -> tuple[TensorPack | None, FailureLedger]:
    ledger = FailureLedger()
    rows: list[dict] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    ledger.add(
                        E_SHAPE_MISMATCH,
                        f"JSONL 第 {line_no} 行无法解析: {exc}",
                        "load",
                        (line_no, "jsonl"),
                    )
    except OSError as exc:
        ledger.add(E_SHAPE_MISMATCH, f"无法读取张量包: {exc}", "load", (0, "io"))
        return None, ledger

    tensors: dict[str, Tensor] = {}
    for line_no, row in enumerate(rows, start=1):
        name = row.get("name")
        if not isinstance(name, str) or not name:
            ledger.add(
                E_SHAPE_MISMATCH,
                f"第 {line_no} 行缺少张量名",
                "validate",
                (line_no, "name"),
            )
            continue
        if name in tensors:
            ledger.add(
                E_SHAPE_MISMATCH,
                f"张量名重复: {name}",
                "validate",
                (line_no, "dup", name),
            )
            continue
        shape = row.get("shape")
        data = row.get("data")
        dtype = row.get("dtype", "int8")
        signed = bool(row.get("signed", True))
        if (
            not isinstance(shape, list)
            or not shape
            or not all(isinstance(dim, int) and dim > 0 for dim in shape)
        ):
            ledger.add(
                E_SHAPE_MISMATCH,
                f"张量 {name} 的形状非法: {shape!r}",
                "validate",
                (line_no, name, "shape"),
            )
            continue
        shape_tuple = tuple(int(dim) for dim in shape)
        if name not in expected:
            ledger.add(
                E_SHAPE_MISMATCH,
                f"未声明的张量 {name}，期望集合={sorted(expected)}",
                "validate",
                (line_no, name, "undeclared"),
            )
            continue
        if tuple(expected[name]) != shape_tuple:
            ledger.add(
                E_SHAPE_MISMATCH,
                f"张量 {name} 形状 {shape_tuple} 与描述 {tuple(expected[name])} 不一致",
                "validate",
                (line_no, name, "shape_mismatch"),
            )
            continue
        if not isinstance(data, list) or len(data) != _product(shape_tuple):
            ledger.add(
                E_SHAPE_MISMATCH,
                f"张量 {name} 数据长度 {len(data) if isinstance(data, list) else 'NA'} "
                f"与形状乘积 {_product(shape_tuple)} 不一致",
                "validate",
                (line_no, name, "data_len"),
            )
            continue
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in data):
            ledger.add(
                E_SHAPE_MISMATCH,
                f"张量 {name} 数据必须全部为整数",
                "validate",
                (line_no, name, "data_type"),
            )
            continue
        try:
            lo, hi = _dtype_bounds(dtype, signed)
        except (ValueError, AttributeError):
            ledger.add(
                E_SHAPE_MISMATCH,
                f"张量 {name} 的 dtype 非法: {dtype!r}",
                "validate",
                (line_no, name, "dtype"),
            )
            continue
        out_of_range = [value for value in data if value < lo or value > hi]
        if out_of_range:
            sample = out_of_range[0]
            ledger.add(
                E_SHAPE_MISMATCH,
                f"张量 {name} 整数数据 {sample} 超出 {dtype}(signed={signed}) 区间 [{lo},{hi}]",
                "validate",
                (line_no, name, "data_range"),
            )
            continue
        tensors[name] = Tensor(
            name=name,
            shape=shape_tuple,
            dtype=dtype,
            signed=signed,
            data=tuple(int(value) for value in data),
            row_ids=_row_ids(name, shape_tuple),
        )

    missing = sorted(set(expected) - set(tensors))
    for name in missing:
        ledger.add(
            E_SHAPE_MISMATCH,
            f"缺少声明张量: {name}（期望形状 {expected[name]}）",
            "validate",
            (0, name, "missing"),
        )

    if len(ledger) > 0:
        return None, ledger

    package_fp = combine_fingerprints(
        *[f"{name}:{tensors[name].fingerprint}" for name in sorted(tensors)]
    )
    return TensorPack(tensors=tensors, package_fingerprint=package_fp), ledger


def expected_main_shapes(desc: OpDesc) -> dict[str, tuple[int, ...]]:
    return {
        "activation": (desc.batch_n, desc.in_channels, desc.in_h, desc.in_w),
        "weight": (desc.out_channels, desc.in_channels, desc.k_h, desc.k_w),
    }


def _product(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total
