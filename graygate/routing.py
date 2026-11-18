from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .errors import RequestError
from .spec import GatewaySpec, RouteSpec, StageSpec


HASH_BUCKETS = 10_000


@dataclass(frozen=True)
class RouteDecision:
    rule_id: str
    stage_id: str
    stage_weight_bp: int
    hash_bucket: int
    side: str
    backend_version: str

    def canonical(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "stage_id": self.stage_id,
            "stage_weight_bp": self.stage_weight_bp,
            "hash_bucket": self.hash_bucket,
            "side": self.side,
            "backend_version": self.backend_version,
        }


def stable_bucket(client_key: str, rule_id: str) -> int:
    digest = hashlib.sha256(
        rule_id.encode("utf-8") + b"\x1f" + client_key.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % HASH_BUCKETS


def stage_for_request(route: RouteSpec, request_seq: int) -> StageSpec:
    stage = route.stage_for(request_seq)
    if stage is None:
        raise RequestError(
            "E_STAGE",
            "$.request_seq",
            f"request sequence {request_seq} is outside every stage for rule {route.rule_id!r}",
            {"rule_id": route.rule_id, "request_seq": request_seq},
        )
    return stage


def decide_route(
    spec: GatewaySpec,
    route: RouteSpec,
    request_seq: int,
    client_key: str,
) -> RouteDecision:
    stage = stage_for_request(route, request_seq)
    bucket = stable_bucket(client_key, route.rule_id)
    side = "canary" if bucket < stage.gray_weight_bp else "stable"
    backend_version = (
        route.canary_version if side == "canary" else route.stable_version
    )
    return RouteDecision(
        rule_id=route.rule_id,
        stage_id=stage.stage_id,
        stage_weight_bp=stage.gray_weight_bp,
        hash_bucket=bucket,
        side=side,
        backend_version=backend_version,
    )

