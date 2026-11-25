"""Deterministic in-memory sample generation.

All randomness comes from the standard-library ``random.Random`` seeded with
the environment's integer seed. No external data set or network is used, and
the generated stream is independent of dictionary iteration order.

An observation record is a plain tuple so it can be serialized over the
line-oriented TCP protocol without relying on object hashing order.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal

from .spec import Declarations, InstanceSpec

# Fixed configuration key universe shared by every environment.
DEFAULT_KEYS: tuple[str, ...] = (
    "conn.timeout_ms",
    "pool.max_size",
    "cache.ttl_ms",
    "retry.backoff_ms",
    "rate.limit_qps",
)


@dataclass(frozen=True)
class Observation:
    env_id: str
    instance_id: str
    key: str
    revision: int
    ts_ms: int
    raw_value: str          # exact decimal rendered as string
    lease_capacity: int

    def as_dict(self) -> dict:
        return {
            "env_id": self.env_id,
            "instance_id": self.instance_id,
            "key": self.key,
            "revision": self.revision,
            "ts_ms": self.ts_ms,
            "raw_value": self.raw_value,
            "lease_capacity": self.lease_capacity,
        }


def _value_for(rng: random.Random, key: str, revision: int) -> str:
    # Bounded integer-or-half decimal values keep the samples human readable
    # while still exercising truncation at exponents 1..3.
    base = {
        "conn.timeout_ms": (50, 5000),
        "pool.max_size": (1, 256),
        "cache.ttl_ms": (100, 60000),
        "retry.backoff_ms": (10, 3000),
        "rate.limit_qps": (10, 10000),
    }[key]
    low, high = base
    integer_part = rng.randint(low, high)
    if rng.random() < 0.5:
        fractional = rng.randint(0, 999)
        return f"{integer_part}.{fractional:03d}"
    return str(integer_part)


def generate_for_instance(
    decls: Declarations, inst: InstanceSpec
) -> list[Observation]:
    env = decls.environments[inst.env_ref]
    rng = random.Random()
    # Combined deterministic seed so instances never share a stream.
    rng.seed(f"{env.seed}:{inst.instance_id}", version=2)
    observations: list[Observation] = []
    revision = rng.randint(1, 5)
    ts = inst.base_ts_ms
    for seq in range(inst.obs_count):
        revision += rng.randint(1, 3)
        ts += rng.randint(1, 25)
        for key in DEFAULT_KEYS:
            observations.append(
                Observation(
                    env_id=env.env_id,
                    instance_id=inst.instance_id,
                    key=key,
                    revision=revision,
                    ts_ms=ts,
                    raw_value=_value_for(rng, key, revision),
                    lease_capacity=env.lease_capacity,
                )
            )
    return observations


def generate_all(decls: Declarations) -> dict[str, list[Observation]]:
    """Observations grouped by environment id in declaration order."""
    grouped: dict[str, list[Observation]] = {
        env_id: [] for env_id in decls.environments
    }
    for inst in decls.instances:
        for obs in generate_for_instance(decls, inst):
            grouped[obs.env_id].append(obs)
    for env_id in grouped:
        # Canonical ordering makes batch splitting order-independent downstream.
        grouped[env_id].sort(
            key=lambda o: (o.key, o.revision, o.ts_ms, o.instance_id)
        )
    return grouped


def decimal_value(obs: Observation) -> Decimal:
    return Decimal(obs.raw_value)
