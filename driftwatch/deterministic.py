from __future__ import annotations

import hashlib
import time

from .models import AgentSpec, Observation

_MASK64 = (1 << 64) - 1
_DEFAULT_PROFILE = {
    "cpu_quota": (90000, 4500),
    "memory_pages": (60000, 3000),
    "processes": (480, 24),
}


def stable_u64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _profile(agent: AgentSpec, host_id: str, metric_id: str) -> tuple[int, int]:
    explicit = agent.profiles.get(host_id, {}).get(metric_id)
    if explicit:
        return int(explicit["base"]), int(explicit["spread"])
    base, spread = _DEFAULT_PROFILE[metric_id]
    host_offset = stable_u64(agent.seed, host_id, metric_id, "host-offset") % (spread + 1)
    return base + host_offset - spread // 2, spread


def aggregate_integers(kind: str, values: list[int]) -> int:
    if not values:
        raise ValueError("aggregation requires at least one sample")
    ordered = sorted(int(value) for value in values)
    if kind == "minimum":
        return ordered[0]
    if kind == "maximum":
        return ordered[-1]
    if kind == "median":
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) // 2
    if kind == "mean_floor":
        return sum(ordered) // len(ordered)
    if kind == "trimmed_mean":
        if len(ordered) < 3:
            return sum(ordered) // len(ordered)
        trimmed = ordered[1:-1]
        return sum(trimmed) // len(trimmed)
    raise ValueError(f"unsupported aggregation kind: {kind}")


def generate_observation(
    agent: AgentSpec,
    snapshot_seq: int,
    host_id: str,
    metric_id: str,
    observation_seq: int,
    monotonic_ms: int | None = None,
) -> Observation:
    if not agent.covers(host_id, metric_id):
        raise ValueError(f"agent {agent.agent_id} does not cover {host_id}/{metric_id}")
    base, spread = _profile(agent, host_id, metric_id)
    key = stable_u64(
        agent.seed,
        host_id,
        metric_id,
        snapshot_seq,
        agent.aggregation.version,
        agent.aggregation.kind,
        agent.aggregation.window,
    )
    samples: list[int] = []
    width = spread * 2 + 1
    for index in range(agent.aggregation.window):
        noise = splitmix64(key + index * 0xD1B54A32D192ED03) % width - spread
        samples.append(max(0, base + noise))
    value = aggregate_integers(agent.aggregation.kind, samples)
    if monotonic_ms is None:
        monotonic_ms = time.monotonic_ns() // 1_000_000
    return Observation(
        agent_id=agent.agent_id,
        observation_seq=observation_seq,
        host_id=host_id,
        metric_id=metric_id,
        value=value,
        aggregation_version=agent.aggregation.version,
        monotonic_ms=int(monotonic_ms),
    )
