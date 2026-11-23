from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from .errors import E_SPEC, GatewayError
from .models import Request, Rule, Spec

HASH_DOMAIN = "gateway-route-v1"
BUCKET_COUNT = 10001


class RoutingError(GatewayError):
    pass


@dataclass(frozen=True)
class RouteDecision:
    destination: str
    backend_id: str
    bucket: int


class Router:
    """Stable, deterministic decision and derived-target resolver."""

    def __init__(self, spec: Spec) -> None:
        self.spec = spec
        self._decision_cache: dict[tuple[str, str], RouteDecision] = {}
        self._canary_cache: dict[tuple[str, str], str] = {}

    @staticmethod
    def bucket(rule_id: str, client_key: str) -> int:
        material = f"{HASH_DOMAIN}\0{rule_id}\0{client_key}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)
        return value % BUCKET_COUNT

    @staticmethod
    def choose_destination(bucket: int, canary_weight_bp: int) -> str:
        if canary_weight_bp <= 0:
            return "stable"
        if canary_weight_bp >= 10000:
            return "canary"
        return "canary" if bucket < canary_weight_bp else "stable"

    def canary_backend(self, rule_id: str, client_key: str) -> str:
        key = (rule_id, client_key)
        if key in self._canary_cache:
            return self._canary_cache[key]
        rule = self.spec.rules[rule_id]
        target = rule.canary_target
        if target.kind == "backend":
            result = target.backend_id
            assert result is not None
        else:
            upstream = target.from_rule
            assert upstream is not None
            if target.select == "conclusion":
                result = self.decide(upstream, client_key).backend_id
            elif target.select == "stable_target":
                result = self.spec.rules[upstream].stable_backend
            elif target.select == "canary_target":
                result = self.canary_backend(upstream, client_key)
            else:  # guarded by the loader; kept as a defensive failure
                raise RoutingError(
                    E_SPEC,
                    f"unsupported derived-target selector {target.select!r}",
                    rule_id=rule_id,
                    field="canary_backend.select",
                )
        self._canary_cache[key] = result
        return result

    def decide(self, rule_id: str, client_key: str) -> RouteDecision:
        key = (rule_id, client_key)
        if key in self._decision_cache:
            return self._decision_cache[key]
        rule = self.spec.rules[rule_id]
        bucket = self.bucket(rule_id, client_key)
        destination = self.choose_destination(bucket, rule.canary_weight_bp)
        if destination == "stable":
            backend_id = rule.stable_backend
        else:
            backend_id = self.canary_backend(rule_id, client_key)
        result = RouteDecision(destination=destination, backend_id=backend_id, bucket=bucket)
        self._decision_cache[key] = result
        return result

    def matching_rule(self, request: Request) -> Rule:
        explicit = getattr(request, "rule_id", None)
        if explicit:
            if explicit not in self.spec.rules:
                raise GatewayError(
                    E_SPEC,
                    "request names an unknown rule",
                    rule_id=explicit,
                    field="rule_id",
                    entry=request.request_id,
                )
            rule = self.spec.rules[explicit]
            if not request.prefix.startswith(rule.request_prefix):
                raise GatewayError(
                    E_SPEC,
                    "request prefix does not match the explicitly named rule",
                    rule_id=explicit,
                    field="prefix",
                    entry=request.request_id,
                )
            return rule
        candidates = [
            rule
            for rule in self.spec.rules.values()
            if request.prefix.startswith(rule.request_prefix)
        ]
        if not candidates:
            raise GatewayError(
                E_SPEC,
                "request prefix does not match any rule",
                field="prefix",
                entry=request.request_id,
            )
        candidates.sort(key=lambda rule: (-len(rule.request_prefix), rule.id))
        return candidates[0]


def validate_request_routes(spec: Spec) -> None:
    router = Router(spec)
    for request in sorted(spec.requests, key=lambda item: (item.arrival_seq, item.request_id)):
        rule = router.matching_rule(request)
        decision = router.decide(rule.id, request.client_key)
        backend = spec.backends.get(decision.backend_id)
        if backend is None:
            raise GatewayError(
                E_SPEC,
                "resolved backend is not declared",
                rule_id=rule.id,
                field="canary_backend" if decision.destination == "canary" else "stable_backend",
                entry=request.request_id,
            )
        if len(request.vector) != backend.model_dim:
            raise GatewayError(
                E_SPEC,
                "input vector length does not match the selected backend model dimension",
                rule_id=rule.id,
                field="vector",
                entry=request.request_id,
                details={
                    "expected": backend.model_dim,
                    "actual": len(request.vector),
                    "backend_id": backend.id,
                    "destination": decision.destination,
                },
            )


def distribution_from_receipts(receipts: Iterable[dict]) -> dict[str, int]:
    result = {"stable": 0, "canary": 0}
    for receipt in receipts:
        destination = receipt["destination"]
        result[destination] = result.get(destination, 0) + len(receipt["request_ids"])
    return result
