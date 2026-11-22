from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import AppConfig
from .loader import NormalizedDoc
from .util import failure, sort_failures


@dataclass(frozen=True)
class ScheduleUnit:
    shard: str
    batch_index: int
    doc_ids: tuple[str, ...]
    token_cost: int
    order_index: int

    @property
    def stable_key(self) -> str:
        return f"{self.shard}:{self.batch_index:08d}:schedule"


@dataclass
class ScheduleResult:
    units: list[ScheduleUnit]
    state: dict[str, Any]
    accepted_doc_ids: list[str]
    completed_shards: list[str]
    partial_shards: list[str]
    pending_shards: list[str]
    quota_exhausted_shards: list[str]
    ledger: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    interrupted: bool
    global_remaining: int


def build_units(
    documents_by_shard: dict[str, tuple[NormalizedDoc, ...]],
    batch_max_docs: int,
) -> list[ScheduleUnit]:
    units: list[ScheduleUnit] = []
    order_index = 0
    for shard in sorted(documents_by_shard):
        docs = documents_by_shard[shard]
        for batch_index, start in enumerate(range(0, len(docs), batch_max_docs)):
            batch = docs[start : start + batch_max_docs]
            units.append(
                ScheduleUnit(
                    shard=shard,
                    batch_index=batch_index,
                    doc_ids=tuple(doc.doc_id for doc in batch),
                    token_cost=sum(doc.token_count for doc in batch),
                    order_index=order_index,
                )
            )
            order_index += 1
    return units


def initial_state(config: AppConfig, units: list[ScheduleUnit]) -> dict[str, Any]:
    return {
        "schema": 1,
        "cursor": 0,
        "global_remaining": config.global_token_budget,
        "shard_used": {},
        "accepted_doc_ids": [],
        "exhausted_shards": [],
        "pending_shards": [],
        "global_exhausted": False,
        "stopped_gracefully": False,
        "ledger": [],
        "failures": [],
        "batches_since_commit": 0,
        "unit_count": len(units),
    }


def _coerce_state(raw: dict[str, Any], config: AppConfig, units: list[ScheduleUnit]) -> dict[str, Any]:
    state = initial_state(config, units)
    for key in (
        "cursor",
        "global_remaining",
        "shard_used",
        "accepted_doc_ids",
        "exhausted_shards",
        "pending_shards",
        "global_exhausted",
        "stopped_gracefully",
        "ledger",
        "failures",
    ):
        if key in raw:
            state[key] = raw[key]
    if not isinstance(state["cursor"], int) or not 0 <= state["cursor"] <= len(units):
        raise ValueError("checkpoint cursor is outside the schedule")
    if not isinstance(state["global_remaining"], int) or state["global_remaining"] < 0:
        raise ValueError("checkpoint global_remaining is invalid")
    if not isinstance(state["shard_used"], dict):
        state["shard_used"] = {}
    for key in ("accepted_doc_ids", "exhausted_shards", "pending_shards", "ledger", "failures"):
        if not isinstance(state[key], list):
            state[key] = []
    state["stopped_gracefully"] = False
    state["unit_count"] = len(units)
    return state


def run_scheduler(
    documents_by_shard: dict[str, tuple[NormalizedDoc, ...]],
    config: AppConfig,
    *,
    resume_state: dict[str, Any] | None = None,
    stop_check: Callable[[], bool] | None = None,
    checkpoint_callback: Callable[[dict[str, Any], str], None] | None = None,
) -> ScheduleResult:
    units = build_units(documents_by_shard, config.batch_max_docs)
    state = _coerce_state(resume_state, config, units) if resume_state else initial_state(config, units)
    state["stopped_gracefully"] = False
    shard_doc_counts = {shard: len(docs) for shard, docs in documents_by_shard.items()}
    shard_names = sorted(documents_by_shard)
    shard_of_doc = {
        doc.doc_id: shard for shard, docs in documents_by_shard.items() for doc in docs
    }
    interrupted = False
    commit_every = max(1, config.snapshot_commit_interval_batches)

    while state["cursor"] < len(units):
        unit = units[state["cursor"]]
        if state.get("global_exhausted"):
            break
        if unit.shard in state["pending_shards"] or unit.shard in state["exhausted_shards"]:
            state["cursor"] += 1
            continue
        used = int(state["shard_used"].get(unit.shard, 0))
        if used + unit.token_cost > config.per_shard_token_quota:
            if unit.shard not in state["exhausted_shards"]:
                state["exhausted_shards"].append(unit.shard)
                state["failures"].append(
                    failure(
                        "scheduling",
                        "E_SHARD_QUOTA_EXHAUSTED",
                        unit.stable_key,
                        "shard token quota would be exceeded; shard processing is stopped",
                        shard=unit.shard,
                        batch_index=unit.batch_index,
                        quota=config.per_shard_token_quota,
                        used=used,
                        attempted_tokens=unit.token_cost,
                    )
                )
            state["cursor"] += 1
            continue
        remaining = int(state["global_remaining"])
        if unit.token_cost > remaining:
            state["global_exhausted"] = True
            for shard in shard_names:
                if shard in state["exhausted_shards"] or shard in state["pending_shards"]:
                    continue
                accepted_in_shard = sum(
                    1 for doc_id in state["accepted_doc_ids"] if shard_of_doc.get(doc_id) == shard
                )
                if accepted_in_shard < shard_doc_counts[shard]:
                    state["pending_shards"].append(shard)
                    state["failures"].append(
                        failure(
                            "scheduling",
                            "E_BUDGET_PENDING",
                            f"{shard}:00000000:pending",
                            "global token budget exhausted; unfinished shard remains queued",
                            shard=shard,
                            global_remaining=remaining,
                            attempted_tokens=unit.token_cost if shard == unit.shard else None,
                        )
                    )
            state["cursor"] = len(units)
            if checkpoint_callback:
                checkpoint_callback(state, "budget_exhausted")
            break

        new_remaining = remaining - unit.token_cost
        state["global_remaining"] = new_remaining
        state["shard_used"][unit.shard] = used + unit.token_cost
        state["accepted_doc_ids"].extend(unit.doc_ids)
        state["ledger"].append(
            {
                "deduction_index": len(state["ledger"]) + 1,
                "stable_order": unit.order_index + 1,
                "shard": unit.shard,
                "batch_index": unit.batch_index,
                "tokens": unit.token_cost,
                "remaining": new_remaining,
                "remaining_before": remaining,
                "stable_key": unit.stable_key,
            }
        )
        state["cursor"] += 1
        state["batches_since_commit"] = int(state.get("batches_since_commit", 0)) + 1

        if stop_check and stop_check():
            state["stopped_gracefully"] = True
            interrupted = True
            if checkpoint_callback:
                checkpoint_callback(state, "graceful_stop")
            break
        if checkpoint_callback and state["batches_since_commit"] >= commit_every:
            state["batches_since_commit"] = 0
            checkpoint_callback(state, "interval")

    if not interrupted and state["cursor"] >= len(units) and not state.get("global_exhausted"):
        if checkpoint_callback:
            checkpoint_callback(state, "scheduler_complete")

    accepted_set = set(state["accepted_doc_ids"])
    completed_shards: list[str] = []
    partial_shards: list[str] = []
    for shard in shard_names:
        docs = documents_by_shard[shard]
        accepted = sum(1 for doc in docs if doc.doc_id in accepted_set)
        if accepted == len(docs):
            completed_shards.append(shard)
        elif accepted > 0:
            partial_shards.append(shard)
    pending_shards = sorted(set(state["pending_shards"]))
    exhausted_shards = sorted(set(state["exhausted_shards"]))
    state["failures"] = sort_failures(state["failures"])
    return ScheduleResult(
        units=units,
        state=state,
        accepted_doc_ids=sorted(accepted_set),
        completed_shards=sorted(completed_shards),
        partial_shards=sorted(partial_shards),
        pending_shards=pending_shards,
        quota_exhausted_shards=exhausted_shards,
        ledger=list(state["ledger"]),
        failures=list(state["failures"]),
        interrupted=interrupted,
        global_remaining=int(state["global_remaining"]),
    )


def serialize_schedule_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": 1,
        "cursor": state["cursor"],
        "global_remaining": state["global_remaining"],
        "shard_used": dict(state["shard_used"]),
        "accepted_doc_ids": list(state["accepted_doc_ids"]),
        "exhausted_shards": list(state["exhausted_shards"]),
        "pending_shards": list(state["pending_shards"]),
        "global_exhausted": bool(state.get("global_exhausted", False)),
        "stopped_gracefully": bool(state.get("stopped_gracefully", False)),
        "ledger": list(state["ledger"]),
        "failures": list(state["failures"]),
        "batches_since_commit": int(state.get("batches_since_commit", 0)),
    }
