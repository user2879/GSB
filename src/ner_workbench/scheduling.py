from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .config import InferenceConfig
from .errors import E_GLOBAL_BUDGET_EXHAUSTED, E_SAMPLE_QUOTA_EXCEEDED
from .injector import InjectionSet, InjectionSpec
from .types import CorpusSample


@dataclass(frozen=True)
class BatchPlan:
    sample_id: str
    batch_index: int
    positions: tuple[int, ...]
    stable_key: str
    block_keys: tuple[str, ...]


@dataclass(frozen=True)
class BudgetDecision:
    sequence: int
    sample_id: str
    requested_chars: int
    effective_quota: int
    allocated: int
    before: int
    after: int
    admitted: bool
    code: str | None
    reason: str
    injection_id: str | None = None

    @property
    def charge(self) -> int:
        return int(self.allocated)

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "sample_id": self.sample_id,
            "requested_chars": self.requested_chars,
            "effective_quota": self.effective_quota,
            "allocated": self.allocated,
            "before": self.before,
            "after": self.after,
            "admitted": self.admitted,
            "code": self.code,
            "reason": self.reason,
            "injection_id": self.injection_id,
        }


def _quota_injection(injections: InjectionSet, sample_id: str) -> InjectionSpec | None:
    for injection in injections.for_sample(sample_id, "schedule"):
        if injection.kind == "quota_lt_min_chars":
            return injection
    return None


def plan_budget(
    samples: Sequence[CorpusSample],
    cfg: InferenceConfig,
    injections: InjectionSet,
    *,
    initial_remaining: int,
    sequence_start: int = 1,
) -> tuple[BudgetDecision, ...]:
    remaining = max(0, int(initial_remaining))
    decisions: list[BudgetDecision] = []
    sequence = int(sequence_start) - 1
    for sample in sorted(samples, key=lambda item: item.sample_id):
        sequence += 1
        before = remaining
        required = len(sample.chars)
        quota_injection = _quota_injection(injections, sample.sample_id)
        effective_quota = cfg.per_sample_quota
        if quota_injection is not None:
            effective_quota = min(cfg.per_sample_quota, max(0, required - 1))

        if effective_quota < required:
            allocation = min(effective_quota, remaining)
            remaining -= allocation
            decisions.append(
                BudgetDecision(
                    sequence=sequence,
                    sample_id=sample.sample_id,
                    requested_chars=required,
                    effective_quota=effective_quota,
                    allocated=allocation,
                    before=before,
                    after=remaining,
                    admitted=False,
                    code=E_SAMPLE_QUOTA_EXCEEDED,
                    reason=(
                        f"sample needs {required} chars but effective quota is {effective_quota}; "
                        f"{allocation} chars charged before abort"
                    ),
                    injection_id=quota_injection.injection_id if quota_injection else None,
                )
            )
            continue

        if remaining < required:
            decisions.append(
                BudgetDecision(
                    sequence=sequence,
                    sample_id=sample.sample_id,
                    requested_chars=required,
                    effective_quota=effective_quota,
                    allocated=0,
                    before=before,
                    after=before,
                    admitted=False,
                    code=E_GLOBAL_BUDGET_EXHAUSTED,
                    reason=f"global budget has {remaining} chars, sample requires {required}",
                )
            )
            continue

        remaining -= required
        decisions.append(
            BudgetDecision(
                sequence=sequence,
                sample_id=sample.sample_id,
                requested_chars=required,
                effective_quota=effective_quota,
                allocated=required,
                before=before,
                after=remaining,
                admitted=True,
                code=None,
                reason="sample admitted with atomic character-budget allocation",
            )
        )
    return tuple(decisions)


def _positions_for_sample(chars: Sequence[str], cfg: InferenceConfig) -> tuple[int, ...]:
    length = len(chars)
    if cfg.processing_unit == "sentence":
        return tuple(range(length))
    positions: list[int] = []
    seen: set[int] = set()
    start = 0
    while start < length:
        end = min(length, start + cfg.window_chars)
        for position in range(start, end):
            if position not in seen:
                seen.add(position)
                positions.append(position)
        if end >= length:
            break
        start += cfg.window_stride
    return tuple(positions)


def build_batch_plans(sample: CorpusSample, cfg: InferenceConfig) -> tuple[BatchPlan, ...]:
    positions = _positions_for_sample(sample.chars, cfg)
    batches: list[BatchPlan] = []
    for batch_index, start in enumerate(range(0, len(positions), cfg.char_batch_size)):
        batch_positions = tuple(positions[start : start + cfg.char_batch_size])
        blocks = tuple(
            sorted({f"{sample.sample_id}:block:{position // cfg.char_block_size:06d}" for position in batch_positions})
        )
        stable_key = f"{sample.sample_id}|b{batch_index:06d}|pipeline"
        batches.append(BatchPlan(sample.sample_id, batch_index, batch_positions, stable_key, blocks))
    return tuple(batches)


@dataclass(frozen=True)
class CacheEvent:
    event_id: str
    sequence: int
    sample_id: str
    batch_index: int
    stage: str
    block_key: str

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "sample_id": self.sample_id,
            "batch_index": self.batch_index,
            "stage": self.stage,
            "block_key": self.block_key,
        }


def build_cache_events(samples: Sequence[CorpusSample], cfg: InferenceConfig) -> tuple[CacheEvent, ...]:
    events: list[CacheEvent] = []
    sequence = 0
    for sample in sorted(samples, key=lambda item: item.sample_id):
        batches = build_batch_plans(sample, cfg)
        for stage in ("emission", "decode"):
            for batch in batches:
                for block_key in batch.block_keys:
                    sequence += 1
                    event_id = cache_event_id(sample.sample_id, batch.batch_index, stage, block_key)
                    events.append(
                        CacheEvent(
                            event_id=event_id,
                            sequence=sequence,
                            sample_id=sample.sample_id,
                            batch_index=batch.batch_index,
                            stage=stage,
                            block_key=block_key,
                        )
                    )
    return tuple(events)


def cache_event_id(sample_id: str, batch_index: int, stage: str, block_key: str) -> str:
    return f"{sample_id}|b{int(batch_index):06d}|{stage}|{block_key.rsplit(':', 1)[-1]}"


@dataclass(frozen=True)
class CacheSimulation:
    accesses: int
    hits: int
    misses: int
    recomputations: int
    evictions: tuple[dict[str, object], ...]
    recompilation_log: tuple[dict[str, object], ...]
    resident_blocks: tuple[str, ...]
    checksum_mismatches: int

    def as_dict(self) -> dict[str, object]:
        return {
            "accesses": self.accesses,
            "hits": self.hits,
            "misses": self.misses,
            "recomputations": self.recomputations,
            "eviction_count": len(self.evictions),
            "evictions": list(self.evictions),
            "recompilation_count": len(self.recompilation_log),
            "recompilation_log": list(self.recompilation_log),
            "resident_blocks": list(self.resident_blocks),
            "checksum_mismatches": self.checksum_mismatches,
        }


def simulate_lru_cache(
    events: Sequence[CacheEvent],
    cache_budget: int,
    signature_for_block: Callable[[str], str],
) -> CacheSimulation:
    last_access: dict[str, int] = {}
    computed_once: set[str] = set()
    checksum_first: dict[str, str] = {}
    evictions: list[dict[str, object]] = []
    recomputation_log: list[dict[str, object]] = []
    hits = 0
    misses = 0
    recomputations = 0
    mismatches = 0
    logical_sequence = 0

    for event in sorted(events, key=lambda item: item.sequence):
        logical_sequence += 1
        checksum = signature_for_block(event.block_key)
        if event.block_key in last_access:
            hits += 1
            last_access[event.block_key] = logical_sequence
            continue

        misses += 1
        if event.block_key in computed_once:
            recomputations += 1
            first = checksum_first.get(event.block_key)
            if first != checksum:
                mismatches += 1
            recomputation_log.append(
                {
                    "sequence": logical_sequence,
                    "event_id": event.event_id,
                    "block_key": event.block_key,
                    "checksum": checksum,
                    "matches_first_compute": first == checksum,
                }
            )
        else:
            computed_once.add(event.block_key)
            checksum_first[event.block_key] = checksum

        if cache_budget <= 0:
            continue
        if len(last_access) >= cache_budget:
            victim = min(last_access.items(), key=lambda item: (item[1], item[0]))[0]
            victim_last_access = last_access.pop(victim)
            evictions.append(
                {
                    "sequence": logical_sequence,
                    "event_id": event.event_id,
                    "evicted_block": victim,
                    "incoming_block": event.block_key,
                    "victim_last_access": victim_last_access,
                    "reason": "LRU; lexicographic block-key tie-break",
                }
            )
        last_access[event.block_key] = logical_sequence

    resident = tuple(sorted(last_access))
    return CacheSimulation(
        accesses=logical_sequence,
        hits=hits,
        misses=misses,
        recomputations=recomputations,
        evictions=tuple(evictions),
        recompilation_log=tuple(recomputation_log),
        resident_blocks=resident,
        checksum_mismatches=mismatches,
    )


def cache_events_for_executed(
    samples: Sequence[CorpusSample],
    events: Sequence[CacheEvent],
    executed_event_ids: Iterable[str],
) -> tuple[CacheEvent, ...]:
    executed = set(executed_event_ids)
    allowed_samples = {sample.sample_id for sample in samples}
    return tuple(
        event
        for event in sorted(events, key=lambda item: item.sequence)
        if event.event_id in executed and event.sample_id in allowed_samples
    )
