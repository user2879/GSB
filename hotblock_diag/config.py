"""Prefilter / fixed-point configuration and edit-command validation."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .errors import E_COMMAND_REF, E_CONFIG_INVALID, E_TENSOR_LAYOUT, FailureList
from .util import canonical_json_bytes, json_hash

GEAR_FRAC = {"q4": 4, "q8": 8, "q12": 12, "q16": 16}
ALLOWED_FRAC = tuple(sorted(set(GEAR_FRAC.values())))
ALLOWED_ACC = (16, 32, 64)
ALLOWED_SAT = ("saturate", "wrap")
ALLOWED_ROUND = ("truncate", "nearest_even", "nearest_up", "floor")
ALLOWED_PADDING = ("last_zero", "reject")
COMMAND_TYPES = ("replace_qblock", "replace_kblock", "update_prefilter",
                 "set_quant", "set_hot_capacity")


@dataclass
class PrefilterConfig:
    q_block: int
    k_block: int
    candidates: int
    score_floor_ratio: float
    temperature: float
    padding: str = "last_zero"

    def fingerprint(self) -> str:
        return json_hash(self.__dict__)


@dataclass
class FixedPointConfig:
    frac_bits: int
    quant_gear: str
    acc_bits: int
    saturation: str
    rounding: str
    hot_capacity: int
    hot_keep: int
    max_rounds: int
    threads: int
    batch_budget: int
    qblock_quota: int
    snapshot_interval: int
    tolerance: float

    def fingerprint(self) -> str:
        return json_hash(self.__dict__)


@dataclass
class LayoutInfo:
    seq_q: int
    seq_k: int
    dim: int
    padded_q: int
    padded_k: int
    q_blocks: int
    k_blocks: int
    q_pad: int
    k_pad: int


@dataclass
class Command:
    cmd: str
    index: int
    payload: dict[str, Any] = field(default_factory=dict)


def _load_json(path: str, failures: FailureList, stage: str) -> Any | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        failures.add(E_CONFIG_INVALID, stage, path, exc, detail=str(exc))
        return None


def _require(cond: bool, target: str, detail: str, failures: FailureList,
             stage: str = "config_validate") -> None:
    if not cond:
        failures.add(E_CONFIG_INVALID, stage, target, ValueError(detail), detail=detail)


def load_prefilter(path: str, failures: FailureList) -> PrefilterConfig | None:
    raw = _load_json(path, failures, "prefilter_load")
    if raw is None:
        return None
    target = path
    for key in ("q_block", "k_block", "candidates", "score_floor_ratio", "temperature"):
        _require(key in raw, target, f"missing key {key!r}", failures)
    if failures.failures:
        return None
    q_block, k_block = raw["q_block"], raw["k_block"]
    _require(isinstance(q_block, int) and q_block > 0, target, "q_block must be positive int", failures)
    _require(isinstance(k_block, int) and k_block > 0, target, "k_block must be positive int", failures)
    _require(isinstance(raw["candidates"], int) and raw["candidates"] > 0, target,
             "candidates must be positive int", failures)
    ratio = raw["score_floor_ratio"]
    temp = raw["temperature"]
    _require(isinstance(ratio, (int, float)) and math.isfinite(ratio) and 0.0 <= ratio <= 1.0,
             target, "score_floor_ratio must be finite in [0,1]", failures)
    _require(isinstance(temp, (int, float)) and math.isfinite(temp) and temp > 0.0,
             target, "temperature must be a positive finite number", failures)
    padding = raw.get("padding", "last_zero")
    _require(padding in ALLOWED_PADDING, target, f"padding must be one of {ALLOWED_PADDING}", failures)
    if failures.failures:
        return None
    return PrefilterConfig(
        q_block=int(q_block), k_block=int(k_block),
        candidates=int(raw["candidates"]), score_floor_ratio=float(ratio),
        temperature=float(temp), padding=padding,
    )


def load_fixedpoint(path: str, failures: FailureList) -> FixedPointConfig | None:
    raw = _load_json(path, failures, "fixedpoint_load")
    if raw is None:
        return None
    target = path
    required = ("frac_bits", "quant_gear", "acc_bits", "saturation", "rounding",
                "hot_capacity", "hot_keep", "max_rounds", "threads",
                "batch_budget", "qblock_quota", "snapshot_interval", "tolerance")
    for key in required:
        _require(key in raw, target, f"missing key {key!r}", failures)
    if failures.failures:
        return None
    frac = raw["frac_bits"]
    gear = raw["quant_gear"]
    acc = raw["acc_bits"]
    _require(frac in ALLOWED_FRAC, target, f"frac_bits must be in {ALLOWED_FRAC}", failures)
    _require(gear in GEAR_FRAC, target, f"quant_gear must be in {tuple(GEAR_FRAC)}", failures)
    if gear in GEAR_FRAC:
        _require(GEAR_FRAC[gear] == frac, target,
                 f"quant_gear {gear} requires frac_bits={GEAR_FRAC[gear]}", failures)
    _require(acc in ALLOWED_ACC, target, f"acc_bits must be in {ALLOWED_ACC}", failures)
    if acc in ALLOWED_ACC:
        _require(acc > frac, target, "acc_bits must be greater than frac_bits", failures)
    _require(raw["saturation"] in ALLOWED_SAT, target, f"saturation in {ALLOWED_SAT}", failures)
    _require(raw["rounding"] in ALLOWED_ROUND, target, f"rounding in {ALLOWED_ROUND}", failures)
    int_keys = ("hot_capacity", "hot_keep", "max_rounds", "threads",
                "batch_budget", "qblock_quota", "snapshot_interval")
    for key in int_keys:
        _require(isinstance(raw[key], int) and raw[key] >= 1, target,
                 f"{key} must be int >= 1", failures)
    tol = raw["tolerance"]
    _require(isinstance(tol, (int, float)) and math.isfinite(tol) and tol >= 0.0,
             target, "tolerance must be finite and >= 0", failures)
    if failures.failures:
        return None
    return FixedPointConfig(
        frac_bits=int(frac), quant_gear=gear, acc_bits=int(acc),
        saturation=raw["saturation"], rounding=raw["rounding"],
        hot_capacity=int(raw["hot_capacity"]), hot_keep=int(raw["hot_keep"]),
        max_rounds=int(raw["max_rounds"]), threads=int(raw["threads"]),
        batch_budget=int(raw["batch_budget"]), qblock_quota=int(raw["qblock_quota"]),
        snapshot_interval=int(raw["snapshot_interval"]), tolerance=float(tol),
    )


def validate_layout(seq_q: int, seq_k: int, seq_v: int, dim_q: int, dim_k: int,
                    dim_v: int, pre: PrefilterConfig, failures: FailureList,
                    target: str = "tensor_pack") -> LayoutInfo | None:
    if dim_q != dim_k or dim_q != dim_v:
        failures.add(E_TENSOR_LAYOUT, "tensor_parse", target,
                     ValueError("embedding dims differ"),
                     detail=f"dims q={dim_q} k={dim_k} v={dim_v}")
        return None
    if seq_v != seq_k:
        failures.add(E_TENSOR_LAYOUT, "tensor_parse", target,
                     ValueError("value/key sequence lengths differ"),
                     detail=f"k={seq_k} v={seq_v}")
        return None

    def padded(seq: int, block: int, name: str) -> tuple[int, int, int] | None:
        if seq % block == 0:
            return seq, seq // block, 0
        if pre.padding == "reject":
            failures.add(E_TENSOR_LAYOUT, "tensor_parse", target,
                         ValueError(f"{name} length {seq} not divisible by block {block}"),
                         detail="padding policy is reject")
            return None
        pad = block - (seq % block)
        return seq + pad, (seq + pad) // block, pad

    q_res = padded(seq_q, pre.q_block, "query")
    k_res = padded(seq_k, pre.k_block, "key")
    if q_res is None or k_res is None:
        return None
    padded_q, q_blocks, q_pad = q_res
    padded_k, k_blocks, k_pad = k_res
    return LayoutInfo(seq_q, seq_k, dim_q, padded_q, padded_k, q_blocks,
                      k_blocks, q_pad, k_pad)


def validate_candidates(pre: PrefilterConfig, layout: LayoutInfo,
                        failures: FailureList, target: str = "prefilter") -> None:
    if pre.candidates > layout.k_blocks:
        failures.add(
            E_CONFIG_INVALID, "config_validate", target,
            ValueError("candidates exceeds key block count"),
            detail=f"candidates={pre.candidates} k_blocks={layout.k_blocks}",
        )


def load_commands(path: str | None, layout: LayoutInfo, pre: PrefilterConfig,
                  fp: FixedPointConfig, failures: FailureList) -> list[Command]:
    if not path:
        return []
    raw = _load_json(path, failures, "command_load")
    if raw is None:
        return []
    if not isinstance(raw, list):
        failures.add(E_CONFIG_INVALID, "command_validate", path,
                     ValueError("command stream must be a JSON array"))
        return []
    commands: list[Command] = []
    dim = layout.dim
    for pos, item in enumerate(raw):
        target = f"{path}#command[{pos}]"
        if not isinstance(item, dict) or "cmd" not in item:
            failures.add(E_COMMAND_REF, "command_validate", target,
                         ValueError("command must be object with 'cmd'"))
            continue
        cmd = item["cmd"]
        if cmd not in COMMAND_TYPES:
            failures.add(E_COMMAND_REF, "command_validate", target,
                         ValueError(f"unknown command {cmd!r}"))
            continue
        index = item.get("index", -1)
        if cmd == "replace_qblock":
            if not isinstance(index, int) or not (0 <= index < layout.q_blocks):
                failures.add(E_COMMAND_REF, "command_validate", target,
                             IndexError(f"qblock {index} out of [0,{layout.q_blocks})"))
                continue
            data = item.get("data")
            expected = pre.q_block * dim
            if not _is_data_block(data, expected):
                failures.add(E_COMMAND_REF, "command_validate", target,
                             ValueError(f"data must be {expected} finite numbers"))
                continue
        elif cmd == "replace_kblock":
            if not isinstance(index, int) or not (0 <= index < layout.k_blocks):
                failures.add(E_COMMAND_REF, "command_validate", target,
                             IndexError(f"kblock {index} out of [0,{layout.k_blocks})"))
                continue
            data = item.get("data")
            expected = pre.k_block * dim
            if not _is_data_block(data, expected):
                failures.add(E_COMMAND_REF, "command_validate", target,
                             ValueError(f"data must be {expected} finite numbers"))
                continue
        elif cmd == "update_prefilter":
            patch = item.get("params", {})
            if not isinstance(patch, dict) or not _valid_prefilter_patch(patch, layout):
                failures.add(E_COMMAND_REF, "command_validate", target,
                             ValueError("invalid prefilter patch"))
                continue
        elif cmd == "set_quant":
            gear = item.get("quant_gear")
            if gear not in GEAR_FRAC:
                failures.add(E_COMMAND_REF, "command_validate", target,
                             ValueError(f"quant_gear must be in {tuple(GEAR_FRAC)}"))
                continue
        elif cmd == "set_hot_capacity":
            cap = item.get("capacity")
            if not isinstance(cap, int) or cap < 1:
                failures.add(E_COMMAND_REF, "command_validate", target,
                             ValueError("capacity must be int >= 1"))
                continue
        commands.append(Command(cmd=cmd, index=int(index) if isinstance(index, int) else -1,
                                payload={k: v for k, v in item.items() if k != "cmd"}))
    return commands


def _is_data_block(data: Any, expected: int) -> bool:
    if not isinstance(data, list) or len(data) != expected:
        return False
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in data)


def _valid_prefilter_patch(patch: dict[str, Any], layout: LayoutInfo) -> bool:
    allowed = {"q_block", "k_block", "candidates", "score_floor_ratio", "temperature"}
    if set(patch) - allowed:
        return False
    if "q_block" in patch and not (isinstance(patch["q_block"], int) and patch["q_block"] >= 1):
        return False
    if "k_block" in patch and not (isinstance(patch["k_block"], int) and patch["k_block"] >= 1):
        return False
    if "candidates" in patch:
        cand = patch["candidates"]
        if not (isinstance(cand, int) and 1 <= cand <= layout.k_blocks):
            return False
    if "score_floor_ratio" in patch:
        ratio = patch["score_floor_ratio"]
        if not (isinstance(ratio, (int, float)) and math.isfinite(ratio) and 0 <= ratio <= 1):
            return False
    if "temperature" in patch:
        temp = patch["temperature"]
        if not (isinstance(temp, (int, float)) and math.isfinite(temp) and temp > 0):
            return False
    return True


def config_blob_hash(pre: PrefilterConfig, fp: FixedPointConfig) -> str:
    import hashlib
    blob = canonical_json_bytes({"prefilter": pre.__dict__, "fixedpoint": fp.__dict__})
    return hashlib.sha256(blob).hexdigest()[:16]
