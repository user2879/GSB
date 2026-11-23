"""Stable client-key routing and deterministic batch planning."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .errors import SpecError
from .model import Backend, RequestSample, RouteRule, Spec

ROUTING_ALGORITHM_VERSION = 1
BUCKET_COUNT = 10000


@dataclass(frozen=True)
class RouteDecision:
    request_id: int
    client_key: str
    arrival_ms: int
    rule_id: str
    target: str
    backend_id: str
    bucket: int
    weight_bp: int
    weight_version: int
    routing_version: int = ROUTING_ALGORITHM_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "client_key": self.client_key,
            "arrival_ms": self.arrival_ms,
            "rule_id": self.rule_id,
            "target": self.target,
            "backend_id": self.backend_id,
            "bucket": self.bucket,
            "weight_bp": self.weight_bp,
            "weight_version": self.weight_version,
            "routing_version": self.routing_version,
        }


@dataclass(frozen=True)
class PlannedRequest:
    sample: RequestSample
    decision: RouteDecision


@dataclass(frozen=True)
class BatchPlan:
    batch_id: str
    revision: int
    rule_id: str
    backend_id: str
    target: str
    batch_ordinal: int
    items: tuple[PlannedRequest, ...]

    @property
    def request_ids(self) -> tuple[int, ...]:
        return tuple(item.sample.request_id for item in self.items)

    @property
    def first_arrival_ms(self) -> int:
        return min(item.sample.arrival_ms for item in self.items)

    @property
    def batch_size(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "revision": self.revision,
            "rule_id": self.rule_id,
            "backend_id": self.backend_id,
            "target": self.target,
            "batch_ordinal": self.batch_ordinal,
            "request_ids": list(self.request_ids),
        }


@dataclass(frozen=True)
class RoutingPlan:
    revision: int
    decisions: tuple[RouteDecision, ...]
    batches: tuple[BatchPlan, ...]

    @property
    def request_ids(self) -> tuple[int, ...]:
        return tuple(decision.request_id for decision in self.decisions)

    @property
    def batch_by_request(self) -> dict[int, BatchPlan]:
        return {
            request_id: batch
            for batch in self.batches
            for request_id in batch.request_ids
        }


def stable_bucket(rule_id: str, client_key: str) -> int:
    material = (
        f"gateway-route-v{ROUTING_ALGORITHM_VERSION}\0{rule_id}\0{client_key}".encode(
            "utf-8"
        )
    )
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big") % BUCKET_COUNT


def select_rule(spec: Spec, request: RequestSample) -> RouteRule:
    selector = request.selector
    if selector is not None:
        for rule in spec.rules:
            if rule.id == selector:
                return rule
        candidates = [rule for rule in spec.rules if selector.startswith(rule.prefix)]
        source = selector
    else:
        candidates = [rule for rule in spec.rules if request.client_key.startswith(rule.prefix)]
        source = request.client_key

    if not candidates and len(spec.rules) == 1:
        return spec.rules[0]
    if not candidates:
        raise SpecError(
            f"client key {source!r} does not match any rule prefix",
            entry=f"requests[{request.source_index}]",
            field="client_key" if selector is None else "prefix",
            request_id=request.request_id,
        )
    return sorted(candidates, key=lambda rule: (-len(rule.prefix), rule.id))[0]


def decide_route(spec: Spec, request: RequestSample) -> RouteDecision:
    rule = select_rule(spec, request)
    bucket = stable_bucket(rule.id, request.client_key)
    target = "canary" if bucket < rule.canary_weight_bp else "stable"
    backend_id = rule.canary_backend if target == "canary" else rule.stable_backend
    return RouteDecision(
        request_id=request.request_id,
        client_key=request.client_key,
        arrival_ms=request.arrival_ms,
        rule_id=rule.id,
        target=target,
        backend_id=backend_id,
        bucket=bucket,
        weight_bp=rule.canary_weight_bp,
        weight_version=rule.version,
    )


def _batch_id(revision: int, rule_id: str, backend_id: str, ordinal: int) -> str:
    group_digest = hashlib.sha256(
        f"{rule_id}\0{backend_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"b{revision:04d}-{group_digest}-{ordinal:06d}"


def plan_routes(spec: Spec, requests: Iterable[RequestSample]) -> RoutingPlan:
    ordered_requests = sorted(
        requests, key=lambda item: (item.arrival_ms, item.request_id)
    )
    planned_requests = [
        PlannedRequest(sample=request, decision=decide_route(spec, request))
        for request in ordered_requests
    ]
    groups: dict[tuple[str, str], list[PlannedRequest]] = defaultdict(list)
    for item in planned_requests:
        groups[(item.decision.rule_id, item.decision.backend_id)].append(item)

    rules = spec.rules_by_id
    batches: list[BatchPlan] = []
    for rule_id, backend_id in sorted(groups):
        group = groups[(rule_id, backend_id)]
        batch_size = rules[rule_id].batch_size
        for ordinal, start in enumerate(range(0, len(group), batch_size)):
            chunk = tuple(group[start : start + batch_size])
            batches.append(
                BatchPlan(
                    batch_id=_batch_id(spec.revision, rule_id, backend_id, ordinal),
                    revision=spec.revision,
                    rule_id=rule_id,
                    backend_id=backend_id,
                    target=(
                        chunk[0].decision.target
                        if all(item.decision.target == chunk[0].decision.target for item in chunk)
                        else "mixed"
                    ),
                    batch_ordinal=ordinal,
                    items=chunk,
                )
            )
    batches.sort(key=lambda batch: (batch.first_arrival_ms, batch.batch_id))
    decisions = tuple(item.decision for item in planned_requests)
    return RoutingPlan(
        revision=spec.revision,
        decisions=decisions,
        batches=tuple(batches),
    )


def batch_for_request(plan: RoutingPlan, request_id: int) -> BatchPlan:
    for batch in plan.batches:
        if request_id in batch.request_ids:
            return batch
    raise KeyError(request_id)


def validate_target_dimension(backend: Backend, request: RequestSample) -> None:
    if len(request.input_vector) != backend.model_dim:
        raise SpecError(
            f"vector length {len(request.input_vector)} does not match backend {backend.id!r} model_dim {backend.model_dim}",
            entry=f"requests[{request.source_index}]",
            field="input",
            request_id=request.request_id,
            backend_id=backend.id,
        )
