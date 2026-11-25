"""Integer fixed-point helpers and the deterministic NumPy executor.

Failure ratios use basis points (1/10000) so the entire breaker decision stays
in integer arithmetic. The executor runs a small integer fixed-point forward
pass on a fixed-seed sample and returns a stable conclusion summary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASIS_POINTS = 10_000


def mul_div(numerator: int, multiplier: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (int(numerator) * int(multiplier)) // int(denominator)


def ratio_bp(part: int, whole: int) -> int:
    """Return part/whole in basis points using floor integer division."""
    if whole <= 0:
        return 0
    return mul_div(max(0, int(part)), BASIS_POINTS, int(whole))


def ceil_div(left: int, right: int) -> int:
    return -((-int(left)) // int(right))


@dataclass(frozen=True)
class ExecResult:
    digest: str
    vector: tuple[int, ...]
    mac_count: int
    sample_size: int

    def to_dict(self) -> dict:
        return {
            "digest": self.digest,
            "vector": list(self.vector),
            "mac_count": self.mac_count,
            "sample_size": self.sample_size,
        }


def deterministic_execute(
    *,
    seed: int,
    service_id: str,
    batch_id: str,
    arrival_seq: int,
    sample_size: int = 16,
    width: int = 8,
    quant_bits: int = 8,
) -> ExecResult:
    """Deterministic integer fixed-point compute over a fixed-seed sample.

    The RNG key is derived only from stable identifiers, so a given
    (service, batch, arrival) tuple always produces identical bytes on any
    machine and under any thread interleaving.
    """
    if sample_size <= 0 or width <= 0:
        raise ValueError("sample_size and width must be positive")
    service_mix = int.from_bytes(
        hashlib_blake(service_id + "|" + batch_id), "little") % 997
    key = int(seed) * 1_000_003 + arrival_seq + service_mix * 7919
    rng = np.random.default_rng(key)
    x = rng.integers(0, 32, size=(sample_size, width), dtype=np.int64)
    limit = max(2, (1 << max(2, min(quant_bits, 14))) - 2)
    weights = rng.integers(-limit, limit + 1, size=(width, width), dtype=np.int64)
    hidden = np.matmul(x, weights, dtype=np.int64)
    bias = rng.integers(-8, 9, size=width, dtype=np.int64)
    hidden = np.clip(hidden + bias, -1 << 30, 1 << 30)
    # Integer fixed-point scaling by 1/256 (Q8) keeps everything integral.
    scaled = (hidden * (1 << quant_bits)) // 256
    reduced = np.bitwise_xor.reduce(scaled.astype(np.int64), axis=0)
    vector = tuple(int(v & 0xFFFF) for v in reduced)
    mac_count = int(sample_size * width * width)
    hasher_input = np.concatenate([np.array(vector, dtype=np.int64),
                                   np.array([key & 0xFFFF], dtype=np.int64)])
    digest = "sha1:" + hashlib_int(hasher_input)
    return ExecResult(
        digest=digest,
        vector=vector,
        mac_count=mac_count,
        sample_size=sample_size,
    )


def hashlib_int(values: np.ndarray) -> str:
    import hashlib

    raw = np.asarray(values, dtype=np.int64).tobytes()
    return hashlib.sha1(raw).hexdigest()[:16]


def hashlib_blake(text: str) -> bytes:
    import hashlib

    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
