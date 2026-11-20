from __future__ import annotations

import json
from typing import Any

from meshgov.rules import RequestSpec, RuleSet, parse_requests, parse_rules


def make_rules(
    *,
    services: list[dict[str, Any]] | None = None,
    routes: list[dict[str, Any]] | None = None,
    global_concurrency: int = 3,
    default_service_quota: int = 2,
) -> RuleSet:
    document = {
        "schema_version": 1,
        "limits": {
            "global_concurrency": global_concurrency,
            "default_service_quota": default_service_quota,
        },
        "services": services
        or [
            {
                "name": "alpha",
                "concurrency_quota": 2,
                "instances": [
                    instance("alpha-a", 28001, 1),
                    instance("alpha-b", 28002, 1),
                ],
            }
        ],
        "routes": routes
        or [
            {
                "prefix": "/api/alpha",
                "service": "alpha",
                "timeout_ms": 100,
                "max_retries": 2,
            }
        ],
    }
    return parse_rules(
        json.dumps(document, ensure_ascii=False).encode("utf-8"),
        "<memory:rules>",
    )


def instance(
    name: str,
    port: int,
    weight: int,
    *,
    threshold: int = 2,
    cooldown_ms: int = 100,
) -> dict[str, Any]:
    return {
        "name": name,
        "host": "127.0.0.1",
        "port": port,
        "weight": weight,
        "failure_threshold": threshold,
        "cooldown_ms": cooldown_ms,
    }


def make_requests(
    rules: RuleSet,
    count: int,
    *,
    prefix: str = "/api/alpha",
) -> tuple[RequestSpec, ...]:
    document = {
        "requests": [
            {"id": f"req-{index + 1:04d}", "path": f"{prefix}/{index + 1}"}
            for index in range(count)
        ]
    }
    return parse_requests(
        json.dumps(document).encode("utf-8"),
        "<memory:requests>",
        rules,
    )

