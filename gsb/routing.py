"""基于稳定哈希的灰度分流与逐请求命中账本记录。

分桶值 = sha256("gsb-route|<rule_id>|<client_key>") 前 8 字节大端整数 mod 10000。
结论只依赖 (rule_id, client_key, weight)，不依赖字典或哈希遍历顺序。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .spec import RouteRule

BUCKET_SPACE = 10000


def bucket_for(rule_id: str, client_key: str) -> int:
    digest = hashlib.sha256(
        f"gsb-route|{rule_id}|{client_key}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % BUCKET_SPACE


@dataclass(frozen=True)
class HitRecord:
    request_id: str
    batch_id: str
    rule_id: str
    rule_version: int
    client_key: str
    bucket: int
    weight: int
    route: str
    backend_id: str
    param_version: int = 0


def route_request(rule: RouteRule, request_id: str, batch_id: str, client_key: str) -> HitRecord:
    bucket = bucket_for(rule.rule_id, client_key)
    route = "gray" if bucket < rule.weight else "stable"
    backend_id = rule.gray_backend if route == "gray" else rule.stable_backend
    return HitRecord(
        request_id=request_id,
        batch_id=batch_id,
        rule_id=rule.rule_id,
        rule_version=rule.rule_version,
        client_key=client_key,
        bucket=bucket,
        weight=rule.weight,
        route=route,
        backend_id=backend_id,
    )
