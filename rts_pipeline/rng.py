"""Independent deterministic event substreams."""

from __future__ import annotations

import hashlib

from .constants import RNG_VERSION
from .errors import PipelineError


def derive_u64(game_seed: int, tick: int, rng_seed_delta: int) -> int:
    payload = f"{RNG_VERSION}|{game_seed}|{tick}|{rng_seed_delta}".encode("ascii")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def derive_record(game_seed: int, tick: int, rng_seed_delta: int) -> dict[str, object]:
    value = derive_u64(game_seed, tick, rng_seed_delta)
    return {
        "version": RNG_VERSION,
        "seed": game_seed,
        "tick": tick,
        "rng_seed_delta": rng_seed_delta,
        "value_u64": value,
        "value_hex": f"{value:016x}",
        "roll_10000": value % 10000,
    }


def verify_record(record: dict[str, object], game_seed: int | None = None) -> None:
    required = ("version", "seed", "tick", "rng_seed_delta", "value_u64")
    missing = [field for field in required if field not in record]
    if missing:
        raise PipelineError(
            "E_RNG_DRIFT",
            "random derivation record is incomplete",
            missing=missing,
        )
    seed = record["seed"]
    if game_seed is not None:
        seed = game_seed
    if record["version"] != RNG_VERSION:
        raise PipelineError(
            "E_RNG_DRIFT",
            "random derivation version mismatch",
            expected=RNG_VERSION,
            actual=record["version"],
        )
    tick = record["tick"]
    delta = record["rng_seed_delta"]
    if not isinstance(seed, int) or not isinstance(tick, int) or not isinstance(delta, int):
        raise PipelineError("E_RNG_DRIFT", "random derivation inputs are not integers")
    expected = derive_u64(seed, tick, delta)
    if record["value_u64"] != expected:
        raise PipelineError(
            "E_RNG_DRIFT",
            "random derivation value does not match its independent substream",
            event={"tick": tick, "rng_seed_delta": delta, "seed": seed},
            expected=expected,
            actual=record["value_u64"],
        )


def verify_records(records: list[dict[str, object]], game_seed: int) -> int:
    for record in records:
        verify_record(record, game_seed)
    return len(records)

