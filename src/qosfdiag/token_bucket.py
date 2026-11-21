"""Integer-only token bucket primitives."""

from __future__ import annotations

from dataclasses import dataclass

from .fixed import FixedPointSpec


@dataclass
class TokenBucket:
    spec: FixedPointSpec
    rate_units_per_tick: int
    capacity_units: int
    tokens: int
    last_tick: int = 0

    @classmethod
    def create(
        cls,
        *,
        scale: int,
        committed_rate_kbps: int,
        burst_bytes: int,
        start_tick: int = 0,
    ) -> "TokenBucket":
        spec = FixedPointSpec(scale)
        capacity = spec.bytes_to_units(burst_bytes)
        return cls(
            spec=spec,
            rate_units_per_tick=spec.rate_units_per_tick(committed_rate_kbps),
            capacity_units=capacity,
            tokens=capacity,
            last_tick=start_tick,
        )

    def refill(self, now_tick: int) -> int:
        if now_tick < self.last_tick:
            raise ValueError("time cannot move backwards")
        elapsed_ticks = now_tick - self.last_tick
        if elapsed_ticks:
            added = self.rate_units_per_tick * elapsed_ticks
            self.tokens = min(self.capacity_units, self.tokens + added)
            self.last_tick = now_tick
        return self.tokens

    def consume(self, cost_units: int, now_tick: int) -> tuple[bool, int, int]:
        before = self.refill(now_tick)
        if cost_units > self.tokens:
            return False, before, self.tokens
        self.tokens -= cost_units
        return True, before, self.tokens

    def release_delay_ticks(self, cost_units: int) -> int:
        deficit = max(0, cost_units - self.tokens)
        return self.spec.queue_delay_ticks(deficit, self.rate_units_per_tick)

    def to_state(self) -> dict[str, int]:
        return {
            "rate_units_per_tick": self.rate_units_per_tick,
            "capacity_units": self.capacity_units,
            "tokens": self.tokens,
            "last_tick": self.last_tick,
        }

    @classmethod
    def from_state(cls, scale: int, state: dict[str, int]) -> "TokenBucket":
        return cls(
            spec=FixedPointSpec(scale),
            rate_units_per_tick=int(state["rate_units_per_tick"]),
            capacity_units=int(state["capacity_units"]),
            tokens=int(state["tokens"]),
            last_tick=int(state["last_tick"]),
        )
