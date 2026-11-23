"""异常注入清单加载与真实故障施加。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .errors import Code, OperatorError

# 注入类型 -> 业务码
INJECTION_CODE_MAP = {
    "missing_tensor": Code.E_TENSOR_MISSING,
    "head_dim_mismatch": Code.E_HEAD_DIM_MISMATCH,
    "nonfinite": Code.E_NONFINITE,
    "budget_below_min_block": Code.E_NODE_BUDGET_EXHAUSTED,
    "frac_bits_out_of_range": Code.E_FRAC_BITS_OUT_OF_RANGE,
    "block_fault": Code.E_INJECTED_FAULT,
}


@dataclass(frozen=True)
class Injection:
    id: str
    type: str
    target: dict = field(default_factory=dict)

    @property
    def code(self):
        return INJECTION_CODE_MAP.get(self.type, Code.E_INJECTED_FAULT)


def load_manifest(path) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [Injection(id=item["id"], type=item["type"],
                      target=item.get("target", {}))
            for item in data.get("injections", [])]


def apply_load_time(tensors: dict, meta: dict, cfg, injections: list):
    """对张量包/配置施加加载期真实故障。返回 (tensors, meta, cfg, hits)。"""
    tensors = dict(tensors)
    meta = dict(meta)
    hits = []
    for inj in injections:
        if inj.type == "missing_tensor":
            name = inj.target["tensor"]
            if name in tensors:
                del tensors[name]
                hits.append(inj)
        elif inj.type == "head_dim_mismatch":
            name = inj.target["tensor"]
            wrong_dim = int(inj.target["head_dim"])
            if name in tensors:
                arr = tensors[name]
                if wrong_dim <= arr.shape[1]:
                    tensors[name] = np.ascontiguousarray(arr[:, :wrong_dim])
                else:
                    pad = np.zeros((arr.shape[0], wrong_dim - arr.shape[1]),
                                   dtype=arr.dtype)
                    tensors[name] = np.concatenate([arr, pad], axis=1)
                hits.append(inj)
        elif inj.type == "nonfinite":
            name = inj.target["tensor"]
            row = int(inj.target["row"])
            col = int(inj.target["col"])
            if name in tensors and row < tensors[name].shape[0]:
                tensors[name] = tensors[name].copy()
                tensors[name][row, col] = np.nan
                hits.append(inj)
        elif inj.type == "budget_below_min_block":
            min_block_points = cfg.query_block_len * cfg.key_block_len
            cfg = cfg.with_overrides(
                point_budget_per_batch=min_block_points - 1)
            hits.append(inj)
        elif inj.type == "frac_bits_out_of_range":
            cfg = cfg.with_overrides(
                fixed_point_frac_bits=int(inj.target["value"]))
            hits.append(inj)
    return tensors, meta, cfg, hits


def block_fault_for(injections: list, stage: str, query_block: int,
                    key_block: int):
    """查找命中指定阶段与块坐标的块级故障注入项。"""
    for inj in injections:
        if inj.type != "block_fault":
            continue
        t = inj.target
        if (t.get("stage") == stage
                and int(t.get("query_block", -1)) == query_block
                and int(t.get("key_block", -1)) == key_block):
            return inj
    return None


def raise_block_fault(injection: Injection, stage: str, query_block: int,
                      key_block: int):
    raise OperatorError(
        Code.E_INJECTED_FAULT,
        f"注入故障 {injection.id} 命中阶段 {stage} "
        f"块 (q={query_block}, k={key_block})",
        stage=stage, query_block=query_block, key_block=key_block,
        injection_id=injection.id)
