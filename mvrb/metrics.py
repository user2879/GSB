"""整数度量码 <-> 命名匹配度量 码表。"""
from __future__ import annotations

from .errors import E_UNKNOWN_METRIC, MVRBError

METRIC_CODE_TO_NAME = {
    0: "ncc",
    1: "ssd",
    2: "sift_ratio",
    3: "orb_hamming",
}
METRIC_NAME_TO_CODE = {name: code for code, name in METRIC_CODE_TO_NAME.items()}


def code_to_name(code: int) -> str:
    if code not in METRIC_CODE_TO_NAME:
        raise MVRBError(E_UNKNOWN_METRIC, f"unknown integer metric code {code}")
    return METRIC_CODE_TO_NAME[code]


def name_to_code(name: str) -> int:
    if name not in METRIC_NAME_TO_CODE:
        raise MVRBError(E_UNKNOWN_METRIC, f"unknown named metric {name!r}")
    return METRIC_NAME_TO_CODE[name]
