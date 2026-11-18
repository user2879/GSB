from __future__ import annotations

from typing import Iterable

from .errors import ConfigFormatError, OrderError
from .models import (
    AccessEvent,
    DECISION_ALLOW,
    DECISION_OVER_QUOTA,
    DECISION_RATE_LIMITED,
    EffectiveVolume,
    Judgment,
    LIMIT_DIMENSION_BYTES,
    TOKEN_SCALE,
    VolumeUsage,
)


class JudgmentEngine:
    """Deterministic, integer-only quota and token-bucket evaluator.

    Invariants:

    * Capacity is inclusive: ``cumulative + write_bytes <= capacity`` passes.
    * Quota is decided before rate limiting; a request that would cross the
      capacity is ``OVER_QUOTA`` even if the token bucket is also empty.
    * Tokens are stored in thousandths of the configured dimension unit.
      With integer milliseconds, ``delta_ms * rate_per_sec`` scaled units is
      exact, so both the single-event and aggregate quantization error bounds
      are zero.
    * A per-volume rate limit with both rate and burst equal to zero is
      disabled, preserving v1 semantics after normalization.
    """

    def __init__(
        self,
        effective_volumes: dict[str, EffectiveVolume],
        initial_usage: dict[str, VolumeUsage] | None = None,
    ) -> None:
        self.effective_volumes = dict(effective_volumes)
        self.usage: dict[str, VolumeUsage] = {}
        initial_usage = initial_usage or {}
        for volume_id, volume in self.effective_volumes.items():
            if volume_id in initial_usage:
                usage = VolumeUsage.from_json(initial_usage[volume_id].to_json())
            else:
                usage = VolumeUsage(volume_id=volume_id)
                usage.tokens_scaled = volume.token_capacity_scaled
            self.usage[volume_id] = usage

    def evaluate(self, event: AccessEvent) -> Judgment:
        volume = self.effective_volumes.get(event.volume_id)
        if volume is None:
            raise ConfigFormatError(
                f"event references unknown volume {event.volume_id!r}",
                volume_id=event.volume_id,
                field_path=f"event:{event.seq}.volume_id",
                details={"seq": event.seq},
            )
        usage = self.usage[volume.volume_id]

        if usage.last_refill_ms is not None and event.timestamp_ms < usage.last_refill_ms:
            raise OrderError(
                "event timestamp moved backwards",
                volume_id=event.volume_id,
                field_path=f"event:{event.seq}.timestamp_ms",
                details={
                    "previous_timestamp_ms": usage.last_refill_ms,
                    "timestamp_ms": event.timestamp_ms,
                },
            )

        if volume.rate_limit_enabled:
            if usage.last_refill_ms is None:
                refill_scaled = 0
            else:
                delta_ms = event.timestamp_ms - usage.last_refill_ms
                refill_scaled = delta_ms * volume.ops_per_sec
            usage.tokens_scaled = min(
                volume.token_capacity_scaled,
                usage.tokens_scaled + refill_scaled,
            )
        usage.last_refill_ms = event.timestamp_ms

        projected_bytes = usage.cumulative_write_bytes + event.write_bytes
        if projected_bytes > volume.capacity_bytes:
            usage.over_quota_events += 1
            return self._judgment(
                event,
                volume,
                usage,
                DECISION_OVER_QUOTA,
                (
                    f"projected_write_bytes={projected_bytes} > "
                    f"capacity_bytes={volume.capacity_bytes}"
                ),
            )

        if volume.rate_limit_enabled:
            cost_units = (
                event.write_bytes
                if volume.limit_dimension == LIMIT_DIMENSION_BYTES
                else event.operation_count
            )
            cost_scaled = cost_units * TOKEN_SCALE
            if cost_scaled > usage.tokens_scaled:
                usage.rate_limited_events += 1
                return self._judgment(
                    event,
                    volume,
                    usage,
                    DECISION_RATE_LIMITED,
                    (
                        f"cost_scaled={cost_scaled} > "
                        f"token_balance_scaled={usage.tokens_scaled}"
                    ),
                )
            usage.tokens_scaled -= cost_scaled

        usage.cumulative_write_bytes = projected_bytes
        usage.allowed_events += 1
        usage.allowed_write_bytes += event.write_bytes
        usage.allowed_operations += event.operation_count
        return self._judgment(
            event,
            volume,
            usage,
            DECISION_ALLOW,
            "within capacity and token budget",
        )

    def _judgment(
        self,
        event: AccessEvent,
        volume: EffectiveVolume,
        usage: VolumeUsage,
        decision: str,
        reason: str,
    ) -> Judgment:
        return Judgment(
            seq=event.seq,
            timestamp_ms=event.timestamp_ms,
            volume_id=event.volume_id,
            decision=decision,
            reason=reason,
            operation_count=event.operation_count,
            write_bytes=event.write_bytes,
            cumulative_write_bytes=usage.cumulative_write_bytes,
            tokens_scaled=usage.tokens_scaled,
            tokens_capacity_scaled=volume.token_capacity_scaled,
            last_refill_ms=usage.last_refill_ms,
        )

    def snapshot_usage(self) -> dict[str, VolumeUsage]:
        return dict(self.usage)


def replay_events(
    effective_volumes: dict[str, EffectiveVolume],
    events: Iterable[AccessEvent],
    initial_usage: dict[str, VolumeUsage] | None = None,
) -> tuple[list[Judgment], dict[str, VolumeUsage]]:
    engine = JudgmentEngine(effective_volumes, initial_usage)
    judgments = [engine.evaluate(event) for event in events]
    return judgments, engine.snapshot_usage()
