"""Gray routing, mirror sampling and deterministic request simulation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .fixedpoint import (
    GRID_BASIS,
    QuantizedRatio,
    complement_fixed,
    quantize,
)
from .hashing import MIRROR_DOMAIN, ROUTE_DOMAIN, stable_bucket
from .loader import match_rule
from .models import Request, Rule

STABLE = "stable"
GRAY = "gray"


def _sim_digest(rule_id: str, key: str, attempt: int, channel: str) -> int:
    material = f"sim|{channel}|{rule_id}|{key}|{attempt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


@dataclass
class QuantizedRule:
    rule: Rule
    scale: int
    gray: QuantizedRatio
    mirror: QuantizedRatio

    @property
    def gray_fixed(self) -> int:
        return self.gray.fixed

    @property
    def stable_fixed(self) -> int:
        return complement_fixed(self.gray_fixed, self.scale)

    def to_report(self) -> dict[str, Any]:
        data = self.rule.__dict__.copy()
        data.update({
            "gray_quantization": self.gray.to_report(),
            "stable_fixed_point": self.stable_fixed,
            "mirror_quantization": self.mirror.to_report(),
            "partition_sums_to_full_scale": (
                self.gray_fixed + self.stable_fixed == self.scale
            ),
        })
        return data


def quantize_rules(rules: list[Rule], scale: int) -> dict[str, QuantizedRule]:
    return {
        rule.rule_id: QuantizedRule(
            rule=rule, scale=scale,
            gray=QuantizedRatio(rule.gray_weight, scale,
                                quantize(rule.gray_weight, scale)),
            mirror=QuantizedRatio(rule.mirror_rate, scale,
                                  quantize(rule.mirror_rate, scale)),
        )
        for rule in rules
    }


@dataclass
class RouteDecision:
    request_id: str
    rule_id: str
    path: str
    client_key: str
    bucket: int
    gray_fixed: int
    scale: int
    target: str
    channel: str
    basis: str
    attempts: int
    status: str

    def to_report(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "rule_id": self.rule_id,
            "path": self.path, "client_key": self.client_key,
            "bucket": self.bucket, "gray_fixed_point": self.gray_fixed,
            "scale_factor": self.scale, "target": self.target,
            "channel": self.channel, "basis": self.basis,
            "attempts": self.attempts, "status": self.status,
        }


@dataclass
class MirrorRecord:
    request_id: str
    rule_id: str
    bucket: int
    mirror_fixed: int
    scale: int
    target: str
    timeout_ms: int
    status: str
    basis: str

    def to_report(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "rule_id": self.rule_id,
            "bucket": self.bucket, "mirror_fixed_point": self.mirror_fixed,
            "scale_factor": self.scale, "target": self.target,
            "timeout_ms": self.timeout_ms, "status": self.status,
            "basis": self.basis,
        }


class RoutingEngine:
    """Runs a fully deterministic, order independent routing session."""

    def __init__(self, rules: list[Rule], scale: int) -> None:
        self.rules = rules
        self.scale = scale
        self.quantized = quantize_rules(rules, scale)

    def decide_route(self, request: Request) -> tuple[Rule, RouteDecision]:
        rule = match_rule(self.rules, request.path)
        assert rule is not None  # loader already guarantees a match
        qrule = self.quantized[rule.rule_id]
        bucket = stable_bucket(ROUTE_DOMAIN, rule.rule_id, request.client_key)
        is_gray = qrule.gray.contains_bucket(bucket)
        channel = GRAY if is_gray else STABLE
        target = rule.gray_target if is_gray else rule.stable_target
        basis = (
            f"integer cross multiply: bucket {bucket} * scale {self.scale} "
            f"{'<' if is_gray else '>='} gray_fixed {qrule.gray_fixed} "
            f"* basis {GRID_BASIS}"
        )
        attempts, status = self._simulate(rule, request.client_key, channel,
                                          rule.gray_retries if is_gray else 0)
        return rule, RouteDecision(
            request_id=request.request_id, rule_id=rule.rule_id, path=request.path,
            client_key=request.client_key, bucket=bucket,
            gray_fixed=qrule.gray_fixed, scale=self.scale, target=target,
            channel=channel, basis=basis, attempts=attempts, status=status,
        )

    def decide_mirror(self, request: Request, rule: Rule) -> MirrorRecord | None:
        qrule = self.quantized[rule.rule_id]
        bucket = stable_bucket(MIRROR_DOMAIN, rule.rule_id, request.request_id)
        if not qrule.mirror.contains_bucket(bucket):
            return None
        digest = _sim_digest(rule.rule_id, request.request_id, 0, MIRROR_DOMAIN)
        status = "SUCCESS" if digest % 100 >= 15 else "TIMEOUT"
        return MirrorRecord(
            request_id=request.request_id, rule_id=rule.rule_id, bucket=bucket,
            mirror_fixed=qrule.mirror.fixed, scale=self.scale,
            target=rule.mirror_target, timeout_ms=rule.mirror_timeout_ms,
            status=status,
            basis=(
                f"integer cross multiply: bucket {bucket} * scale {self.scale} "
                f"< mirror_fixed {qrule.mirror.fixed} * basis {GRID_BASIS}"
            ),
        )

    @staticmethod
    def _simulate(rule: Rule, key: str, channel: str,
                  retries: int) -> tuple[int, str]:
        max_attempts = 1 + max(0, retries)
        for attempt in range(max_attempts):
            digest = _sim_digest(rule.rule_id, key, attempt, channel)
            if digest % 100 >= 20:  # deterministic 80% per-attempt success
                return attempt + 1, "SUCCESS"
        return max_attempts, "FAILED"

    def run(self, requests: list[Request]) -> dict[str, Any]:
        decisions: list[RouteDecision] = []
        mirrors: list[MirrorRecord] = []
        for request in requests:
            rule, decision = self.decide_route(request)
            decisions.append(decision)
            mirrored = self.decide_mirror(request, rule)
            if mirrored is not None:
                mirrors.append(mirrored)
        return {
            "decisions": decisions,
            "mirrors": mirrors,
            "quantized_rules": self.quantized,
        }
