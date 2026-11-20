"""Stable-hash canary routing and per-request hit ledger rows."""

from __future__ import annotations

import hashlib
from typing import List, Optional

from .models import Declaration, Request, RoutedRequest


def stable_bucket(client_key: str, rule_id: str) -> int:
    """Hash client key + rule id into a bucket in [0, 10000)."""
    digest = hashlib.sha256(f"{rule_id}#{client_key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10000


def match_rule(decl: Declaration, request: Request):
    """Longest-prefix match; ties broken by rule_id (canonical order)."""
    best: Optional[object] = None
    for rule in decl.rules:  # sorted by rule_id
        if request.client_key.startswith(rule.prefix):
            if best is None or len(rule.prefix) > len(best.prefix):
                best = rule
    return best


def route_requests(decl: Declaration) -> List[RoutedRequest]:
    routed: List[RoutedRequest] = []
    for request in decl.requests:  # sorted by seq
        rule = match_rule(decl, request)
        if rule is None:
            continue  # no matching rule: request is not routed
        bucket = stable_bucket(request.client_key, rule.rule_id)
        if bucket < rule.canary_weight_bp:
            backend_id, variant = rule.canary_backend, "canary"
        else:
            backend_id, variant = rule.stable_backend, "stable"
        routed.append(RoutedRequest(request, rule.rule_id, backend_id,
                                    bucket, variant))
    return routed
