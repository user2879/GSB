"""Deterministic sample generation from explicit rules (no external data).

For every instance we emit revision-ordered observations across a fixed key
set. Values follow declared per-key rules so the three arbitration policies all
have meaningful, reproducible inputs:

* scalar keys ``cache.ttl`` and ``pool.size`` use integer/float scalar values;
* interval key ``rate.window`` emits ``[lo, hi]`` integer intervals.

Observation identifiers are stable: ``{env_id}:{instance_id}:{seq}``. Replacing
an observation reuses the same id with a new value.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .determinism import stable_dumps_line
from .spec import Manifest

SCALAR_KEYS = ("cache.ttl", "pool.size")
INTERVAL_KEYS = ("rate.window",)
ALL_KEYS = SCALAR_KEYS + INTERVAL_KEYS


@dataclass(frozen=True)
class Observation:
    obs_id: str
    env_id: str
    instance_id: str
    seq: int
    gen_time_ms: int
    key: str
    value: Any

    def to_event(self) -> dict[str, Any]:
        return {
            "type": "observation",
            "obs_id": self.obs_id,
            "env_id": self.env_id,
            "instance_id": self.instance_id,
            "seq": self.seq,
            "gen_time_ms": self.gen_time_ms,
            "key": self.key,
            "value": self.value,
        }


def _scalar_value(rng: np.random.Generator, key: str, seq: int,
                  env_index: int) -> int | float:
    # Explicit rule: base offset by environment plus bounded deterministic drift.
    if key == "cache.ttl":
        base = 30 + env_index * 5
        return int(base + int(rng.integers(0, 9))) + (seq % 3)
    base = 100 + env_index * 20
    return base + int(rng.integers(-4, 5)) + (seq % 4)


def _interval_value(rng: np.random.Generator, seq: int,
                    env_index: int) -> list[int]:
    center = 50 + env_index * 10 + (seq % 5)
    half = 3 + int(rng.integers(0, 4))
    return [center - half, center + half]


def generate_observations(manifest: Manifest) -> list[Observation]:
    """Produce the full deterministic observation set for a manifest."""
    env_index = {env.env_id: i for i, env in enumerate(manifest.environments)}
    out: list[Observation] = []
    for inst in manifest.instances:
        env = manifest.env(inst.env_ref)
        rng = np.random.default_rng(env.seed + inst.seq_low)
        seq = inst.seq_low
        while seq <= inst.seq_high:
            gen_time = inst.time_base_ms + (seq - inst.seq_low) * 100
            for key in ALL_KEYS:
                obs_id = f"{inst.env_ref}:{inst.instance_id}:{seq}:{key}"
                if key in INTERVAL_KEYS:
                    value = _interval_value(rng, seq, env_index[inst.env_ref])
                else:
                    value = _scalar_value(
                        rng, key, seq, env_index[inst.env_ref]
                    )
                out.append(
                    Observation(
                        obs_id=obs_id,
                        env_id=inst.env_ref,
                        instance_id=inst.instance_id,
                        seq=seq,
                        gen_time_ms=gen_time,
                        key=key,
                        value=value,
                    )
                )
            seq += 1
    # Stable global order independent of callers.
    out.sort(key=lambda o: (o.env_id, o.instance_id, o.seq, o.key))
    return out


def write_ndjson(observations: list[Observation], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for obs in observations:
            handle.write(stable_dumps_line(obs.to_event()))
    return path


def read_ndjson(path: str | Path) -> list[dict[str, Any]]:
    import json

    events: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad NDJSON on line {line_no}: {exc}") from exc
    return events
