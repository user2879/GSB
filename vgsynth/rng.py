from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

MASK64 = (1 << 64) - 1


def derive_seed(root_seed: int, label: str) -> int:
    payload = f"vgsynth/v1|{root_seed}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK64
        self.draws = 0

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        value = value ^ (value >> 31)
        self.draws += 1
        return value & MASK64

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be positive")
        return self.next_u64() % upper

    def uniform(self, low: float, high: float) -> float:
        scale = self.next_u64() / float(1 << 64)
        return low + (high - low) * scale


@dataclass
class SeedAudit:
    root_seed: int
    entries: list[dict[str, Any]] = field(default_factory=list)

    def draw_int(self, decision_id: str, purpose: str, upper: int) -> int:
        sub_seed = derive_seed(self.root_seed, decision_id)
        rng = SplitMix64(sub_seed)
        value = rng.randbelow(upper)
        self.entries.append(
            {
                "decision_id": decision_id,
                "purpose": purpose,
                "sub_seed": f"{sub_seed:016x}",
                "algorithm": "splitmix64-mod",
                "upper": upper,
                "draws": 1,
                "value": value,
            }
        )
        return value

    def draw_float(
        self,
        decision_id: str,
        purpose: str,
        low: float,
        high: float,
        digits: int = 6,
    ) -> float:
        sub_seed = derive_seed(self.root_seed, decision_id)
        rng = SplitMix64(sub_seed)
        raw = rng.uniform(low, high)
        value = round(raw, digits)
        self.entries.append(
            {
                "decision_id": decision_id,
                "purpose": purpose,
                "sub_seed": f"{sub_seed:016x}",
                "algorithm": "splitmix64-unit-interval-round",
                "low": low,
                "high": high,
                "digits": digits,
                "draws": 1,
                "raw": f"{raw:.17g}",
                "value": value,
            }
        )
        return value

    def manifest(self) -> dict[str, Any]:
        return {
            "root_seed": self.root_seed,
            "derivation": "sha256('vgsynth/v1|<root_seed>|<decision_id>')[0:8]",
            "entry_count": len(self.entries),
            "entries": self.entries,
        }


def replay_int(root_seed: int, decision_id: str, upper: int) -> int:
    return SplitMix64(derive_seed(root_seed, decision_id)).randbelow(upper)


def replay_float(root_seed: int, decision_id: str, low: float, high: float, digits: int = 6) -> float:
    value = SplitMix64(derive_seed(root_seed, decision_id)).uniform(low, high)
    if not math.isfinite(value):
        raise ValueError("non-finite replay value")
    return round(value, digits)
