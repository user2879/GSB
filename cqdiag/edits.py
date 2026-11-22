"""编辑计划 JSON 的加载与应用 (业务要求 7 的输入侧)."""

from __future__ import annotations

import json

import numpy as np

from .config import Config, validate_config
from .errors import (
    CqError,
    E_CONFIG_INVALID,
    E_GROUP_MISMATCH,
    E_QUANT_WIDTH_INVALID,
    E_SAMPLE_INVALID,
    FailureList,
)
from .samples import SampleSet
from .config import ALLOWED_QUANT_BITS, SATURATION_STRATEGIES


def config_to_raw(cfg: Config) -> dict:
    return {
        "tensors": [{"name": t.name, "shape": list(t.shape), "column_axis": t.column_axis}
                    for t in cfg.tensors],
        "quantile_lo_permille": cfg.quantile_lo_permille,
        "quantile_hi_permille": cfg.quantile_hi_permille,
        "scheme": cfg.scheme,
        "quant_bits_candidates": list(cfg.quant_bits_candidates),
        "quant_bits": cfg.quant_bits,
        "frac_bits": cfg.frac_bits,
        "saturation": cfg.saturation,
        "abs_tol": cfg.abs_tol,
        "rel_tol": cfg.rel_tol,
        "groups": {k: list(v) for k, v in cfg.groups.items()},
        "group_oor_limit": cfg.group_oor_limit,
        "max_workers": cfg.max_workers,
        "element_budget": cfg.element_budget,
    }


def revalidate(cfg: Config) -> Config:
    return validate_config(config_to_raw(cfg))


def load_edit_plan(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    batches = raw.get("batches")
    if not isinstance(batches, list):
        raise CqError(E_CONFIG_INVALID, "编辑计划必须包含 batches 列表")
    return batches


def _op_replace_column(config: Config, samples: SampleSet, op: dict) -> dict:
    tensor = op.get("tensor")
    column = op.get("column")
    spec = config.tensor(tensor)
    if not isinstance(column, int) or not (0 <= column < spec.num_columns):
        raise CqError(E_CONFIG_INVALID, "replace_column 列号越界",
                      tensor=tensor, column=column)
    st = samples.tensor(tensor)
    n = spec.column_len
    if "values" in op:
        values = op["values"]
        if not isinstance(values, list) or len(values) != n:
            raise CqError(E_SAMPLE_INVALID, "replace_column values 长度与列长不符",
                          tensor=tensor, column=column, expected=n,
                          got=None if not isinstance(values, list) else len(values))
        vals = np.array([float(v) for v in values], dtype=np.float64)
    else:
        seed = op.get("seed", 0)
        scale = float(op.get("scale", 1.0))
        rng = np.random.default_rng(int(seed))
        vals = rng.standard_normal(n) * scale
    missing = np.zeros(n, dtype=bool)
    for idx in op.get("missing_indices", []):
        if not isinstance(idx, int) or not (0 <= idx < n):
            raise CqError(E_SAMPLE_INVALID, "missing_indices 越界",
                          tensor=tensor, column=column, index=idx)
        missing[idx] = True
    st.set_column(column, vals, missing)
    return {"op": "replace_column", "tensor": tensor, "column": column}


def apply_batch(config: Config, samples: SampleSet, batch: dict,
                failures: FailureList) -> tuple[Config, list[dict]]:
    """应用一批编辑; 非法操作进入失败清单并跳过, 其余照常生效."""
    events: list[dict] = []
    for op in batch.get("ops", []):
        kind = op.get("op")
        try:
            if kind == "replace_column":
                events.append(_op_replace_column(config, samples, op))
            elif kind == "set_quantile":
                config = config.with_changes(
                    quantile_lo_permille=op["lo_permille"],
                    quantile_hi_permille=op["hi_permille"],
                )
                config = revalidate(config)
                events.append({"op": kind, "lo": op["lo_permille"], "hi": op["hi_permille"]})
            elif kind == "set_quant_bits":
                bits = op["bits"]
                if bits not in ALLOWED_QUANT_BITS or bits not in config.quant_bits_candidates:
                    raise CqError(E_QUANT_WIDTH_INVALID,
                                  "切换目标量化位宽非法", bits=bits,
                                  candidates=list(config.quant_bits_candidates))
                config = config.with_changes(quant_bits=bits)
                events.append({"op": kind, "bits": bits})
            elif kind == "set_frac_bits":
                config = config.with_changes(frac_bits=op["bits"])
                config = revalidate(config)
                events.append({"op": kind, "bits": op["bits"]})
            elif kind == "set_saturation":
                strategy = op["strategy"]
                if strategy not in SATURATION_STRATEGIES:
                    raise CqError(E_CONFIG_INVALID, "非法饱和策略", strategy=strategy)
                config = config.with_changes(saturation=strategy)
                events.append({"op": kind, "strategy": strategy})
            elif kind == "set_groups":
                tensor = op["tensor"]
                spec = config.tensor(tensor)
                labels = op["labels"]
                if not isinstance(labels, list) or len(labels) != spec.num_columns:
                    raise CqError(E_GROUP_MISMATCH, "列组重新划分未覆盖全部列",
                                  tensor=tensor, expected=spec.num_columns,
                                  got=None if not isinstance(labels, list) else len(labels))
                groups = dict(config.groups)
                groups[tensor] = tuple(labels)
                config = config.with_changes(groups=groups)
                config = revalidate(config)
                events.append({"op": kind, "tensor": tensor})
            elif kind == "set_tolerances":
                config = config.with_changes(abs_tol=op["abs_tol"], rel_tol=op["rel_tol"])
                config = revalidate(config)
                events.append({"op": kind})
            else:
                raise CqError(E_CONFIG_INVALID, "未知编辑操作", op=kind)
        except CqError as exc:
            failures.add_error(exc)
    return config, events
