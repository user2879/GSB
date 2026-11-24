"""Integer fixed-point and deterministic NumPy helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Q = 1_000_000
PERMILLE_Q = 1_000


def mul_div(numerator: int, multiplier: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (int(numerator) * int(multiplier)) // int(denominator)


def ratio_q(part: int, whole: int) -> int:
    if whole <= 0:
        return 0
    return mul_div(max(0, int(part)), Q, int(whole))


def ratio_permille(part: int, whole: int) -> int:
    if whole <= 0:
        return 0
    return mul_div(max(0, int(part)), PERMILLE_Q, int(whole))


def ceil_div(left: int, right: int) -> int:
    return -((-int(left)) // int(right))


@dataclass(frozen=True)
class ForwardResult:
    digest: str
    vector: list[int]
    mac_count: int
    param_count: int


def deterministic_forward(
    *,
    seed: int,
    arrival_seq: int,
    model_dim: int,
    hidden_dim: int,
    quant_bits: int = 8,
) -> ForwardResult:
    """Run a small deterministic integer fixed-point forward pass on CPU."""

    if model_dim <= 0 or hidden_dim <= 0:
        raise ValueError("model_dim and hidden_dim must be positive")
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(arrival_seq))
    x = rng.integers(0, 32, size=model_dim, dtype=np.int64)
    limit = max(2, 1 << max(2, min(quant_bits, 15)) - 2)
    w1 = rng.integers(-limit, limit + 1, size=(model_dim, hidden_dim), dtype=np.int64)
    b1 = rng.integers(-8, 9, size=hidden_dim, dtype=np.int64)
    w2 = rng.integers(-limit, limit + 1, size=(hidden_dim, model_dim), dtype=np.int64)
    b2 = rng.integers(-8, 9, size=model_dim, dtype=np.int64)

    hidden = np.maximum(0, x @ w1 + b1)
    output = hidden @ w2 + b2
    output = np.mod(output, 1_000_003).astype(np.int64)
    mac_count = int(model_dim * hidden_dim + hidden_dim * model_dim)
    param_count = int(w1.size + b1.size + w2.size + b2.size)
    payload = np.ascontiguousarray(output, dtype="<i8").tobytes()
    digest = "sha256:" + __import__("hashlib").sha256(payload).hexdigest()
    return ForwardResult(
        digest=digest,
        vector=[int(v) for v in output],
        mac_count=mac_count,
        param_count=param_count,
    )
