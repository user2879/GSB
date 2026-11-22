"""逐列标定配置加载与校验 (业务要求 1)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace

from .errors import (
    CqError,
    E_CONFIG_INVALID,
    E_QUANT_WIDTH_INVALID,
    E_GROUP_MISMATCH,
)

# 量化位宽允许枚举
ALLOWED_QUANT_BITS = (2, 4, 8, 16)
SCHEMES = ("symmetric", "asymmetric")
SATURATION_STRATEGIES = ("clamp", "mask")


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    column_axis: int

    @property
    def num_columns(self) -> int:
        return self.shape[self.column_axis]

    @property
    def column_len(self) -> int:
        n = 1
        for i, d in enumerate(self.shape):
            if i != self.column_axis:
                n *= d
        return n


@dataclass(frozen=True)
class Config:
    tensors: tuple[TensorSpec, ...]
    quantile_lo_permille: int
    quantile_hi_permille: int
    scheme: str
    quant_bits_candidates: tuple[int, ...]
    quant_bits: int
    frac_bits: int
    saturation: str
    abs_tol: float
    rel_tol: float
    groups: dict[str, tuple[int, ...]]
    group_oor_limit: float
    max_workers: int
    element_budget: int

    def tensor(self, name: str) -> TensorSpec:
        for t in self.tensors:
            if t.name == name:
                return t
        raise CqError(E_CONFIG_INVALID, f"未知张量 {name!r}", {"tensor": name})

    def with_changes(self, **kw) -> "Config":
        return replace(self, **kw)


def _fail(code: str, msg: str, **ctx) -> None:
    raise CqError(code, msg, ctx)


def _validate_tensor_spec(raw: dict, idx: int) -> TensorSpec:
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        _fail(E_CONFIG_INVALID, "张量缺少合法 name", tensor_index=idx)
    shape = raw.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in shape)
    ):
        _fail(E_CONFIG_INVALID, "张量形状维度必须为正整数列表", tensor=name, shape=shape)
    axis = raw.get("column_axis")
    if not isinstance(axis, int) or isinstance(axis, bool) or not (0 <= axis < len(shape)):
        _fail(
            E_CONFIG_INVALID,
            "列轴必须在维度范围内",
            tensor=name,
            column_axis=axis,
            ndim=len(shape),
        )
    return TensorSpec(name=name, shape=tuple(shape), column_axis=axis)


def _validate_finite_positive(value, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _fail(E_CONFIG_INVALID, f"{field_name} 必须为数值", value=value)
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        _fail(E_CONFIG_INVALID, f"{field_name} 必须为有限正数", value=value)
    return v


def validate_config(raw: dict) -> Config:
    if not isinstance(raw, dict):
        _fail(E_CONFIG_INVALID, "配置必须是 JSON 对象")

    tensors_raw = raw.get("tensors")
    if not isinstance(tensors_raw, list) or not tensors_raw:
        _fail(E_CONFIG_INVALID, "tensors 必须为非空列表")
    tensors = tuple(_validate_tensor_spec(t, i) for i, t in enumerate(tensors_raw))
    names = [t.name for t in tensors]
    if len(set(names)) != len(names):
        _fail(E_CONFIG_INVALID, "张量名重复", names=names)

    lo = raw.get("quantile_lo_permille")
    hi = raw.get("quantile_hi_permille")
    for label, v in (("quantile_lo_permille", lo), ("quantile_hi_permille", hi)):
        if not isinstance(v, int) or isinstance(v, bool) or not (0 <= v <= 1000):
            _fail(E_CONFIG_INVALID, f"{label} 必须在 0..1000", value=v)
    if lo >= hi:
        _fail(E_CONFIG_INVALID, "分位下限千分比必须小于上限", lo=lo, hi=hi)

    scheme = raw.get("scheme")
    if scheme not in SCHEMES:
        _fail(E_CONFIG_INVALID, "标定方案必须为 symmetric 或 asymmetric", scheme=scheme)

    candidates = raw.get("quant_bits_candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(c not in ALLOWED_QUANT_BITS for c in candidates)
    ):
        _fail(
            E_QUANT_WIDTH_INVALID,
            "量化位宽候选集合必须落在允许枚举内",
            candidates=candidates,
            allowed=list(ALLOWED_QUANT_BITS),
        )
    quant_bits = raw.get("quant_bits")
    if quant_bits not in ALLOWED_QUANT_BITS:
        _fail(
            E_QUANT_WIDTH_INVALID,
            "量化位宽不在允许枚举内",
            quant_bits=quant_bits,
            allowed=list(ALLOWED_QUANT_BITS),
        )
    if quant_bits not in candidates:
        _fail(
            E_QUANT_WIDTH_INVALID,
            "当前量化位宽不在候选集合内",
            quant_bits=quant_bits,
            candidates=candidates,
        )

    frac_bits = raw.get("frac_bits")
    if not isinstance(frac_bits, int) or isinstance(frac_bits, bool) or frac_bits <= 0:
        _fail(E_CONFIG_INVALID, "定点小数位宽必须为正整数", frac_bits=frac_bits)

    saturation = raw.get("saturation")
    if saturation not in SATURATION_STRATEGIES:
        _fail(
            E_CONFIG_INVALID,
            "饱和截断策略非法",
            saturation=saturation,
            allowed=list(SATURATION_STRATEGIES),
        )

    abs_tol = _validate_finite_positive(raw.get("abs_tol"), "abs_tol")
    rel_tol = _validate_finite_positive(raw.get("rel_tol"), "rel_tol")

    group_oor_limit = raw.get("group_oor_limit", 1.0)
    if (
        not isinstance(group_oor_limit, (int, float))
        or not math.isfinite(float(group_oor_limit))
        or not (0 < float(group_oor_limit) <= 1.0)
    ):
        _fail(E_CONFIG_INVALID, "group_oor_limit 必须为 (0,1] 内有限数", value=group_oor_limit)

    max_workers = raw.get("max_workers", 1)
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
        _fail(E_CONFIG_INVALID, "max_workers 必须为正整数", max_workers=max_workers)

    element_budget = raw.get("element_budget")
    if (
        not isinstance(element_budget, int)
        or isinstance(element_budget, bool)
        or element_budget <= 0
    ):
        _fail(E_CONFIG_INVALID, "element_budget 必须为正整数", element_budget=element_budget)

    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, dict):
        _fail(E_GROUP_MISMATCH, "groups 必须为按张量名的组标号映射")
    groups: dict[str, tuple[int, ...]] = {}
    for spec in tensors:
        labels = groups_raw.get(spec.name)
        if (
            not isinstance(labels, list)
            or len(labels) != spec.num_columns
            or any(not isinstance(g, int) or isinstance(g, bool) or g < 0 for g in labels)
        ):
            _fail(
                E_GROUP_MISMATCH,
                "列组标号必须覆盖全部列且为非负整数",
                tensor=spec.name,
                expected=spec.num_columns,
                got=None if not isinstance(labels, list) else len(labels),
            )
        groups[spec.name] = tuple(labels)

    return Config(
        tensors=tensors,
        quantile_lo_permille=lo,
        quantile_hi_permille=hi,
        scheme=scheme,
        quant_bits_candidates=tuple(sorted(set(candidates))),
        quant_bits=quant_bits,
        frac_bits=frac_bits,
        saturation=saturation,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
        groups=groups,
        group_oor_limit=float(group_oor_limit),
        max_workers=max_workers,
        element_budget=element_budget,
    )


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return validate_config(raw)
