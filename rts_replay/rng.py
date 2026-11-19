"""Deterministic per-event random substream derivation.

Every event gets an *independent* :class:`numpy.random.Generator` whose seed is
derived only from ``(chain_version, match_seed, tick, rng_seed_delta)``. Because
the seed never depends on iteration order, thread scheduling or how many random
numbers earlier events consumed, replaying the same event stream reproduces the
exact same random sequence even when events are processed via snapshots/jumps.

Changing :data:`RNG_CHAIN_VERSION` changes every derived seed, which moves the
state hashes of every tick that consumed randomness; the verifier reports the
first such tick as ``E_RNG_DRIFT``.
"""
from __future__ import annotations

import hashlib

import numpy as np

RNG_CHAIN_VERSION = 1


def derive_seed(
    match_seed: int,
    tick: int,
    rng_seed_delta: int,
    chain_version: int = RNG_CHAIN_VERSION,
) -> int:
    """Derive a stable 64-bit substream seed from the identity tuple."""
    payload = hashlib.sha256(
        repr((chain_version, int(match_seed), int(tick), int(rng_seed_delta))).encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(payload[:8], "big")


def make_event_rng(
    match_seed: int,
    tick: int,
    rng_seed_delta: int,
    chain_version: int = RNG_CHAIN_VERSION,
) -> np.random.Generator:
    """Create the independent random substream for a single event."""
    seed = derive_seed(match_seed, tick, rng_seed_delta, chain_version)
    return np.random.default_rng(seed)


def draw_bounded(rng: np.random.Generator, low: int, high: int) -> int:
    """Draw an integer in the inclusive ``[low, high]`` range."""
    if high < low:
        raise ValueError("empty draw range")
    return int(rng.integers(low=low, high=high + 1, size=1, dtype=np.int64)[0])
