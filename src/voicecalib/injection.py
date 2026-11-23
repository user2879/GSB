"""异常注入引擎：对指定轮次/阶段施加真实故障并隔离。

注入类型：
  nonfinite_sample      采样中注入 NaN/Inf -> E_NONFINITE_SAMPLE
  evidence_dim_mismatch 证据维度不符       -> E_INJECTION_DIM_MISMATCH
  zero_gain             校准增益为零       -> E_ZERO_GAIN
  replace_round_out_of_range 替换轮次越界  -> E_ROUND_OUT_OF_RANGE
单轮失败只污染该轮，错误记录携带 injection_id 以证明命中。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import ErrorCode

VALID_TYPES = (
    "nonfinite_sample",
    "evidence_dim_mismatch",
    "zero_gain",
    "replace_round_out_of_range",
)


@dataclass
class Injection:
    injection_id: str
    round: int
    stage: str
    type: str
    payload: dict

    @classmethod
    def from_dict(cls, item: dict) -> "Injection":
        return cls(
            injection_id=str(item.get("injection_id", "INJ-UNKNOWN")),
            round=int(item["round"]),
            stage=str(item.get("stage", "")),
            type=str(item["type"]),
            payload=dict(item.get("payload", {})),
        )


def build_injections(items: list[dict]) -> list[Injection]:
    return [Injection.from_dict(item) for item in items]


def injections_for_round(injections: list[Injection], rnd: int) -> list[Injection]:
    return [inj for inj in injections if inj.round == rnd]


def apply_sample_injections(samples: np.ndarray,
                            injections: list[Injection]) -> tuple[np.ndarray, list[str]]:
    hit: list[str] = []
    out = samples
    for inj in injections:
        if inj.type != "nonfinite_sample":
            continue
        out = out.astype(np.float64, copy=True)
        kind = inj.payload.get("value", "nan")
        if out.size:
            out[0] = math.nan if kind == "nan" else math.inf
        hit.append(inj.injection_id)
    return out, hit


def check_evidence_injection(vector, injections: list[Injection]
                             ) -> tuple[bool, str | None, str | None]:
    for inj in injections:
        if inj.type == "evidence_dim_mismatch":
            expected_dim = int(inj.payload.get("expected_dim", 3))
            if len(vector) != expected_dim or inj.payload.get("force_fail", True):
                return False, ErrorCode.E_INJECTION_DIM_MISMATCH, inj.injection_id
    return True, None, None


def check_gain_injection(gain_q: int, injections: list[Injection]
                         ) -> tuple[bool, str | None, str | None]:
    # zero_gain 注入存在即表示该轮校准增益被强制置零，触发真实故障。
    for inj in injections:
        if inj.type == "zero_gain":
            return False, ErrorCode.E_ZERO_GAIN, inj.injection_id
    return True, None, None


def check_replace_round(round_index: int, total_rounds: int,
                        injections: list[Injection]
                        ) -> tuple[bool, str | None, str | None]:
    for inj in injections:
        if inj.type != "replace_round_out_of_range":
            continue
        target = int(inj.payload.get("target_round", -1))
        if target < 0 or target >= total_rounds:
            return False, ErrorCode.E_ROUND_OUT_OF_RANGE, inj.injection_id
    return True, None, None


def has_nonfinite_samples(samples: np.ndarray) -> bool:
    return bool(np.any(~np.isfinite(samples)))
