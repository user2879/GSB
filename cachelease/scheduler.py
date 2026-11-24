from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterator

from .content import generate_content
from .node import mono_ms
from .spec import ArtifactSpec, Config, Event, NodeSpec


@dataclass(frozen=True)
class QueuePermit:
    node_id: str
    artifact_key: str
    stage: str
    wait_start_ms: int
    wait_end_ms: int
    wait_ms: int
    global_in_flight: int
    node_in_flight: int
    stage_in_flight: int


@lru_cache(maxsize=4096)
def _expected_digest(node_id: str, artifact_key: str, seed: int, shards: int, shard_bytes: int) -> str:
    return generate_content(node_id, artifact_key, seed, shards, shard_bytes).digest


def _candidate_entry(snapshot: dict[str, Any], artifact_key: str) -> dict[str, Any] | None:
    for entry in snapshot.get("entries", []):
        if entry.get("artifact_key") == artifact_key:
            return entry
    return None


def rank_candidates(
    event: Event,
    config: Config,
    snapshots: dict[str, dict[str, Any]],
    node_in_flight: dict[str, int],
    excluded: set[str] | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    excluded = excluded or set()
    artifact = config.artifact_by_key[event.artifact]  # type: ignore[index]
    candidates: list[dict[str, Any]] = []
    for node_spec in config.nodes:
        snapshot = snapshots.get(node_spec.node_id, {})
        watermark = snapshot.get("watermark", {})
        entry = _candidate_entry(snapshot, artifact.key)
        expected_digest = _expected_digest(
            node_spec.node_id,
            artifact.key,
            node_spec.seed,
            artifact.shards,
            artifact.shard_bytes,
        )
        digest_match = entry is not None and entry.get("digest") == expected_digest
        free_slots = int(watermark.get("free_slots", 0))
        lease_remaining = int(watermark.get("lease_remaining", 0))
        at_node_limit = node_in_flight.get(node_spec.node_id, 0) >= node_spec.request_concurrency
        quota_version = int(snapshot.get("quota_version", -1))

        reason: str | None = None
        eligible = True
        if event.event_type == "read" and not digest_match:
            eligible = False
            reason = "digest_missing"
        if free_slots <= 0:
            eligible = False
            reason = "no_free_slot"
        if lease_remaining <= 0:
            eligible = False
            reason = "lease_limit"
        if at_node_limit:
            eligible = False
            reason = "node_concurrency_limit"
        if node_spec.node_id in excluded:
            eligible = False
            reason = "reservation_rejected"

        candidates.append(
            {
                "node_id": node_spec.node_id,
                "artifact_key": artifact.key,
                "quota_version": quota_version,
                "digest_match": digest_match,
                "existing_digest": None if entry is None else entry.get("digest"),
                "expected_digest": expected_digest,
                "free_slots": free_slots,
                "lease_remaining": lease_remaining,
                "node_in_flight": node_in_flight.get(node_spec.node_id, 0),
                "node_concurrency_limit": node_spec.request_concurrency,
                "eligible": eligible,
                "reason": reason,
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            0 if item["digest_match"] else 1,
            -int(item["free_slots"]),
            str(item["node_id"]),
        ),
    )
    winner = next((item for item in ranked if item["eligible"]), None)
    chosen_node = None if winner is None else str(winner["node_id"])
    for item in ranked:
        if item is winner:
            item["reason"] = "selected"
            item["selected"] = True
        else:
            item["selected"] = False
            if item["reason"] is None:
                item["reason"] = "lower_priority" if item["eligible"] else "ineligible"
    return chosen_node, ranked


class NoEligibleRoute(Exception):
    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        super().__init__("no eligible cache node")
        self.candidates = candidates


class SchedulerAborted(Exception):
    def __init__(self, cause: BaseException) -> None:
        super().__init__(f"scheduler aborted: {cause}")
        self.cause = cause


class RequestScheduler:
    """Tracks real in-flight limits and records condition-variable queue waits."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.condition = threading.Condition(threading.RLock())
        self.global_in_flight = 0
        self.node_in_flight: dict[str, int] = {node.node_id: 0 for node in config.nodes}
        self.stage_in_flight: dict[str, int] = {stage.stage: 0 for stage in config.stages}
        self.max_global_observed = 0
        self.max_node_observed = {node.node_id: 0 for node in config.nodes}
        self.max_stage_observed = {stage.stage: 0 for stage in config.stages}
        self.abort_error: BaseException | None = None

    def inflight_snapshot(self) -> tuple[int, dict[str, int], dict[str, int]]:
        with self.condition:
            return self.global_in_flight, dict(self.node_in_flight), dict(self.stage_in_flight)

    def acquire(
        self,
        stage: str,
        select: Callable[[dict[str, int]], tuple[str | None, list[dict[str, Any]]]],
    ) -> QueuePermit:
        started = mono_ms()
        with self.condition:
            while True:
                if self.abort_error is not None:
                    raise SchedulerAborted(self.abort_error)
                chosen, candidates = select(dict(self.node_in_flight))
                if chosen is None:
                    if self.global_in_flight == 0:
                        raise NoEligibleRoute(candidates)
                    self.condition.wait()
                    continue
                stage_limit = self.config.stage_by_name[stage].concurrency
                node_limit = self.config.node_by_id[chosen].request_concurrency
                if self.global_in_flight >= self.config.runtime.global_concurrency:
                    self.condition.wait()
                    continue
                if self.stage_in_flight.get(stage, 0) >= stage_limit:
                    self.condition.wait()
                    continue
                if self.node_in_flight.get(chosen, 0) >= node_limit:
                    self.condition.wait()
                    continue
                self.global_in_flight += 1
                self.node_in_flight[chosen] = self.node_in_flight.get(chosen, 0) + 1
                self.stage_in_flight[stage] = self.stage_in_flight.get(stage, 0) + 1
                self.max_global_observed = max(self.max_global_observed, self.global_in_flight)
                self.max_node_observed[chosen] = max(self.max_node_observed[chosen], self.node_in_flight[chosen])
                self.max_stage_observed[stage] = max(self.max_stage_observed[stage], self.stage_in_flight[stage])
                ended = mono_ms()
                return QueuePermit(
                    node_id=chosen,
                    artifact_key=next(
                        str(item["artifact_key"])
                        for item in candidates
                        if item["node_id"] == chosen and item.get("selected")
                    ),
                    stage=stage,
                    wait_start_ms=started,
                    wait_end_ms=ended,
                    wait_ms=max(0, ended - started),
                    global_in_flight=self.global_in_flight,
                    node_in_flight=self.node_in_flight[chosen],
                    stage_in_flight=self.stage_in_flight[stage],
                )

    def release(self, permit: QueuePermit) -> None:
        with self.condition:
            self.global_in_flight = max(0, self.global_in_flight - 1)
            self.node_in_flight[permit.node_id] = max(0, self.node_in_flight.get(permit.node_id, 0) - 1)
            self.stage_in_flight[permit.stage] = max(0, self.stage_in_flight.get(permit.stage, 0) - 1)
            self.condition.notify_all()

    def abort(self, cause: BaseException) -> None:
        with self.condition:
            if self.abort_error is None:
                self.abort_error = cause
            self.condition.notify_all()


class DecisionSequencer:
    def __init__(self) -> None:
        self.condition = threading.Condition(threading.RLock())
        self.next_order = 0

    @contextmanager
    def turn(self, order: int) -> Iterator[None]:
        with self.condition:
            while self.next_order != order:
                self.condition.wait()
            try:
                yield
            finally:
                self.next_order += 1
                self.condition.notify_all()
