from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .backend_client import BackendClient
from .errors import E_REPLAY, GatewayError
from .graph import topological_order
from .models import Request, Rule, Spec
from .routing import RouteDecision, Router, distribution_from_receipts
from .util import canonical_json, digest_value

RECEIPT_SCHEMA = "route-receipt.v1"


@dataclass(frozen=True)
class BatchPlan:
    rule_id: str
    batch_index: int
    destination: str
    backend_id: str
    backend_param_digest: str
    requests: tuple[Request, ...]

    @property
    def request_ids(self) -> list[str]:
        return sorted(request.request_id for request in self.requests)


@dataclass(frozen=True)
class RuleExecution:
    rule_id: str
    rule_version: int
    receipt_version: int
    receipts: tuple[dict[str, Any], ...]
    receipt_digest: str
    distribution: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "receipt_version": self.receipt_version,
            "batch_count": len(self.receipts),
            "receipt_digest": self.receipt_digest,
            "distribution": dict(self.distribution),
            "receipts": [dict(receipt) for receipt in self.receipts],
        }


def receipt_set_digest(receipts: Iterable[dict[str, Any]]) -> str:
    ordered = sorted(receipts, key=lambda item: int(item["batch_index"]))
    return digest_value([receipt["receipt_digest"] for receipt in ordered])


def build_batch_plans(spec: Spec) -> dict[str, list[BatchPlan]]:
    router = Router(spec)
    by_rule: dict[str, list[tuple[Request, RouteDecision]]] = defaultdict(list)
    for request in sorted(spec.requests, key=lambda item: (item.arrival_seq, item.request_id)):
        rule = router.matching_rule(request)
        decision = router.decide(rule.id, request.client_key)
        backend = spec.backends[decision.backend_id]
        if len(request.vector) != backend.model_dim:
            raise GatewayError(
                "E_SPEC",
                "input vector length does not match the selected backend model dimension",
                rule_id=rule.id,
                field="vector",
                entry=request.request_id,
                details={"expected": backend.model_dim, "actual": len(request.vector)},
            )
        by_rule[rule.id].append((request, decision))

    plans: dict[str, list[BatchPlan]] = {}
    for rule_id in topological_order(spec):
        rule = spec.rules[rule_id]
        grouped: dict[tuple[str, str], list[Request]] = defaultdict(list)
        for request, decision in by_rule.get(rule_id, []):
            grouped[(decision.destination, decision.backend_id)].append(request)
        rule_plans: list[BatchPlan] = []
        group_order = sorted(
            grouped,
            key=lambda item: (0 if item[0] == "stable" else 1, item[1]),
        )
        batch_index = 0
        for destination, backend_id in group_order:
            backend = spec.backends[backend_id]
            requests = sorted(grouped[(destination, backend_id)], key=lambda item: (item.arrival_seq, item.request_id))
            for start in range(0, len(requests), rule.batch_size):
                chunk = tuple(requests[start : start + rule.batch_size])
                rule_plans.append(
                    BatchPlan(
                        rule_id=rule_id,
                        batch_index=batch_index,
                        destination=destination,
                        backend_id=backend_id,
                        backend_param_digest=backend.param_digest,
                        requests=chunk,
                    )
                )
                batch_index += 1
        plans[rule_id] = rule_plans
    return plans


def build_receipt(plan: BatchPlan, rule: Rule, response: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "rule_id": rule.id,
        "rule_version": rule.version,
        "receipt_version": rule.version,
        "batch_index": plan.batch_index,
        "destination": plan.destination,
        "backend_id": plan.backend_id,
        "backend_param_digest": plan.backend_param_digest,
        "request_ids": plan.request_ids,
        "output_digest": response["output_digest"],
    }
    body["receipt_digest"] = digest_value(body)
    return body


def compare_receipt_expected(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    rule_id = actual["rule_id"]
    fields = [
        "schema",
        "rule_id",
        "rule_version",
        "receipt_version",
        "batch_index",
        "destination",
        "backend_id",
        "backend_param_digest",
        "request_ids",
        "output_digest",
        "receipt_digest",
    ]
    for field in fields:
        if actual.get(field) != expected.get(field):
            raise GatewayError(
                E_REPLAY,
                "recomputed receipt does not match the committed recovery record",
                rule_id=rule_id,
                field=field,
                details={
                    "batch_index": actual.get("batch_index"),
                    "expected": expected.get(field),
                    "actual": actual.get(field),
                },
            )


def execute_rule(
    spec: Spec,
    rule_id: str,
    plans: list[BatchPlan],
    client: BackendClient,
    *,
    on_batch: Callable[[dict[str, Any], int], None] | None = None,
    expected_partial: dict[int, dict[str, Any]] | None = None,
) -> RuleExecution:
    rule = spec.rules[rule_id]
    receipts: list[dict[str, Any]] = []
    expected = dict(expected_partial or {})
    for plan in plans:
        backend = spec.backends[plan.backend_id]
        response = client.infer_batch(backend, list(plan.requests))
        receipt = build_receipt(plan, rule, response)
        if plan.batch_index in expected:
            compare_receipt_expected(receipt, expected[plan.batch_index])
            expected.pop(plan.batch_index, None)
        receipts.append(receipt)
        if on_batch is not None:
            on_batch(receipt, len(receipts))
    if expected:
        first_index = min(expected)
        raise GatewayError(
            E_REPLAY,
            "the ledger contains a committed batch that is absent from the recomputed plan",
            rule_id=rule_id,
            field="batch_index",
            details={
                "batch_index": first_index,
                "missing_from_recomputed_plan": sorted(expected),
            },
        )
    distribution = distribution_from_receipts(receipts)
    digest = receipt_set_digest(receipts)
    return RuleExecution(
        rule_id=rule_id,
        rule_version=rule.version,
        receipt_version=rule.version,
        receipts=tuple(receipts),
        receipt_digest=digest,
        distribution=distribution,
    )


def execute_full(spec: Spec, client: BackendClient) -> dict[str, RuleExecution]:
    plans = build_batch_plans(spec)
    results: dict[str, RuleExecution] = {}
    for rule_id in topological_order(spec):
        results[rule_id] = execute_rule(spec, rule_id, plans[rule_id], client)
    return results


def receipt_json(receipt: dict[str, Any]) -> str:
    return canonical_json(receipt)
