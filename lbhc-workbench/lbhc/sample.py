"""Deterministic sample cluster declaration (loopback only)."""

from __future__ import annotations


def sample_spec(base_port=18601, version=1):
    port = base_port

    def bk(iid, weight, role="active", behavior="ok"):
        nonlocal port
        b = {"id": iid, "host": "127.0.0.1", "port": port,
             "weight": weight, "role": role, "behavior": behavior}
        port += 1
        return b

    return {
        "version": version,
        "global": {"concurrency_budget": 4, "rate_limit_per_sec": 40},
        "pools": [
            {
                "id": "pool-http", "protocol": "http",
                "timeout_ms": 250, "interval_ms": 250,
                "success_threshold": 2, "failure_threshold": 2,
                "concurrency_quota": 3, "expect_body": "OK",
                "backends": [
                    bk("h1", 100), bk("h2", 80), bk("h3", 60),
                    bk("h4", 40, behavior="mismatch"),
                    bk("h5", 20, behavior="refused"),
                    bk("h6", 10, role="draining"),
                ],
            },
            {
                "id": "pool-tcp", "protocol": "tcp",
                "timeout_ms": 200, "interval_ms": 250,
                "success_threshold": 2, "failure_threshold": 3,
                "concurrency_quota": 2, "expect_body": "OK",
                "backends": [
                    bk("t1", 90), bk("t2", 70),
                    bk("t3", 50, behavior="delay"),
                    bk("t4", 30, behavior="refused"),
                ],
            },
        ],
    }
