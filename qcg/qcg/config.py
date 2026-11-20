"""Scheduler configuration loaded from YAML with deterministic defaults."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from typing import Tuple

import yaml

from .errors import QCGError, E_GRAPH_INVALID


@dataclass
class Config:
    max_workers: int = 1
    global_mac_budget: int = 10 ** 12
    node_mac_quota: int = 10 ** 12
    cache_resident_bytes: int = 10 ** 12
    quant_tiers: Tuple[int, ...] = (4, 8, 16)
    frac_bits: int = 8
    snapshot_commit_interval: int = 1
    batch_recompute_budget: int = 10 ** 12
    accumulator_bits: int = 32
    overflow_policy: str = "SATURATE"
    output_min: float = -128.0
    output_max: float = 127.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quant_tiers"] = list(self.quant_tiers)
        return d

    def hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def int_limits(self) -> Tuple[int, int]:
        bits = self.accumulator_bits
        if bits not in (8, 16, 32, 64):
            raise QCGError(E_GRAPH_INVALID,
                           "accumulator_bits must be one of 8/16/32/64",
                           field="accumulator_bits")
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    allowed = set(Config.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        raise QCGError(E_GRAPH_INVALID,
                       "unknown config keys: " + ",".join(sorted(unknown)),
                       field=sorted(unknown)[0])
    data = {}
    for key, value in raw.items():
        if key == "quant_tiers":
            value = tuple(int(v) for v in value)
        elif key in ("frac_bits", "accumulator_bits", "max_workers",
                     "global_mac_budget", "node_mac_quota",
                     "cache_resident_bytes", "snapshot_commit_interval",
                     "batch_recompute_budget"):
            if not isinstance(value, int) or value < 0:
                raise QCGError(E_GRAPH_INVALID,
                               f"{key} must be a non-negative integer",
                               field=key)
        elif key in ("output_min", "output_max"):
            value = float(value)
            if not math.isfinite(value):
                raise QCGError(E_GRAPH_INVALID,
                               f"{key} must be finite", field=key)
        elif key == "overflow_policy":
            if value not in ("SATURATE", "WRAP", "ERROR"):
                raise QCGError(E_GRAPH_INVALID,
                               "overflow_policy must be SATURATE/WRAP/ERROR",
                               field=key)
        data[key] = value
    cfg = Config(**data)
    if cfg.output_min >= cfg.output_max:
        raise QCGError(E_GRAPH_INVALID,
                       "output_min must be smaller than output_max",
                       field="output_min")
    return cfg
