"""Deterministic NumPy fixed-point verification of shard samples.

All arithmetic that feeds a digest is performed on *integer* arrays.  A shard
sample is generated from its fixed seed, quantized with the node's declared
fixed-point width and quantile step, then summarized as:

* ``check_rate_permille`` - integer permille pass rate (c1 native form)
* ``residual``             - truncated integer residual of the exact rate
* ``digest``               - sha256 over the canonical integer payload

The same pure function runs inside every node subprocess and inside the
centre's fingerprinting layer, so cached digests can be reused safely.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .config import ShardConfig

_DT = np.int64


@dataclass(frozen=True)
class ComputeResult:
    shard_id: str
    node_id: str
    quota_version: str
    fixed_point: int
    quant_level: int
    check_rate_permille: int
    residual: int
    check_rate_float: float
    digest: str
    sample_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "quota_version": self.quota_version,
            "fixed_point": self.fixed_point,
            "quant_level": self.quant_level,
            "check_rate_permille": self.check_rate_permille,
            "residual": self.residual,
            "check_rate_float": self.check_rate_float,
            "digest": self.digest,
            "sample_count": self.sample_count,
        }


def generate_samples(shard: ShardConfig) -> np.ndarray:
    """Fixed-seed uniform samples in [0, 1) as float64."""

    rng = np.random.default_rng(shard.seed)
    return rng.random(size=shard.sample_size, dtype=np.float64)


def quantize(samples: np.ndarray, fixed_point: int, quant_level: int) -> np.ndarray:
    """Scale to integer fixed point and snap to the declared quantile step."""

    scale = _DT(10 ** fixed_point)
    scaled = np.floor(samples.astype(np.float64) * float(scale) + 0.5).astype(_DT)
    step = _DT(quant_level)
    return (scaled // step) * step


def compute_shard(
    shard: ShardConfig,
    node_id: str,
    quota_version: str,
    fixed_point: int,
    quant_level: int,
) -> ComputeResult:
    samples = generate_samples(shard)
    quantized = quantize(samples, fixed_point, quant_level)
    scale = 10 ** fixed_point
    # Integer acceptance band [0, scale): every quantized value inside it is a
    # valid fixed-point code; outside it fails verification.
    accepted_mask = (quantized >= 0) & (quantized < _DT(scale))
    accepted = int(np.count_nonzero(accepted_mask))
    total = int(quantized.size)

    exact_num = accepted * 1000 * 1000
    rate_thousandths = exact_num // total
    check_rate_permille = rate_thousandths // 1000
    residual = rate_thousandths % 1000
    check_rate_float = round(accepted / total, 6)

    canonical = {
        "shard_id": shard.shard_id,
        "shard_seed": shard.seed,
        "sample_size": shard.sample_size,
        "size_units": shard.size_units,
        "node_id": node_id,
        "quota_version": quota_version,
        "fixed_point": fixed_point,
        "quant_level": quant_level,
        "quantized": [int(value) for value in quantized.tolist()],
        "accepted": accepted,
        "total": total,
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    return ComputeResult(
        shard_id=shard.shard_id,
        node_id=node_id,
        quota_version=quota_version,
        fixed_point=fixed_point,
        quant_level=quant_level,
        check_rate_permille=int(check_rate_permille),
        residual=int(residual),
        check_rate_float=check_rate_float,
        digest=digest,
        sample_count=total,
    )


def shard_content_fingerprint(shard: ShardConfig) -> str:
    """Fingerprint of the shard's own content (seed, size, units)."""

    payload = {
        "shard_id": shard.shard_id,
        "index": shard.index,
        "size_units": shard.size_units,
        "seed": shard.seed,
        "sample_size": shard.sample_size,
        "upstream": list(shard.upstream),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
