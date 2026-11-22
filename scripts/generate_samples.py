"""Generate deterministic sample declarations and NDJSON event streams.

Three scenarios are produced under ``examples/``:

* contention   - capacity fill, stable priority queue and drain
* expiry       - real monotonic lease expiry and reclamation
* replay       - contention-style workload used for crash/recovery replay

Event timestamps are monotonically non-decreasing integers (milliseconds).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def write_ndjson(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False,
                                separators=(",", ":")) + "\n")


def base_session(scale: int = 10000) -> dict:
    return {
        "domain_set_version": 1,
        "scale_factor": scale,
        "global_concurrency": 3,
        "per_holder_concurrency": 2,
        "retry_backoff_ms": [10, 25, 50],
    }


def domains_for(scenario: str) -> list[dict]:
    if scenario == "expiry":
        return [
            {"domain_id": "locks-alpha", "capacity": 1,
             "lease_duration_ms": 400, "renew_interval_ms": 120,
             "queue_limit": 2, "version": 1},
            {"domain_id": "locks-beta", "capacity": 2,
             "lease_duration_ms": 500, "renew_interval_ms": 150,
             "queue_limit": 2, "version": 1},
        ]
    return [
        {"domain_id": "locks-alpha", "capacity": 2,
         "lease_duration_ms": 800, "renew_interval_ms": 200,
         "queue_limit": 3, "version": 1},
        {"domain_id": "locks-beta", "capacity": 1,
         "lease_duration_ms": 900, "renew_interval_ms": 250,
         "queue_limit": 2, "version": 1},
    ]


def holders() -> list[dict]:
    return [
        {"holder_id": "svc-order", "priority": 10,
         "target_domains": ["locks-alpha", "locks-beta"]},
        {"holder_id": "svc-pay", "priority": 20,
         "target_domains": ["locks-alpha", "locks-beta"]},
        {"holder_id": "svc-inventory", "priority": 5,
         "target_domains": ["locks-alpha"]},
        {"holder_id": "svc-notify", "priority": 15,
         "target_domains": ["locks-beta", "locks-alpha"]},
    ]


def contention_events() -> list[dict]:
    # alpha capacity=2: first two granted, then deterministic queue + drain.
    rows = [
        ("svc-order", "locks-alpha", "申请", 0),
        ("svc-pay", "locks-alpha", "申请", 1),
        ("svc-inventory", "locks-alpha", "申请", 2),   # queued (prio 5)
        ("svc-notify", "locks-alpha", "申请", 3),      # queued ahead (prio 15)
        ("svc-order", "locks-alpha", "续约", 4),       # renewed before expiry
        ("svc-order", "locks-alpha", "释放", 5),       # drain: notify(15) first
        ("svc-pay", "locks-alpha", "释放", 6),         # drain: inventory(5)
        # beta capacity=1: queue full rejection path
        ("svc-pay", "locks-beta", "申请", 7),
        ("svc-order", "locks-beta", "申请", 8),        # queued
        ("svc-notify", "locks-beta", "申请", 9),       # queued
        ("svc-order", "locks-beta", "申请", 10),      # rejected queue full
        ("svc-pay", "locks-beta", "释放", 11),         # drain order first
    ]
    return [{"seq": i + 1, "timestamp_ms": ts * 10, "holder_id": h,
             "domain_id": d, "action": a}
            for i, (h, d, a, ts) in enumerate(rows)]


def expiry_events() -> list[dict]:
    rows = [
        ("svc-order", "locks-alpha", "申请", 0),
        ("svc-pay", "locks-alpha", "申请", 1),         # queued
        ("svc-order", "locks-alpha", "续约", 2),       # renewed
        # long gap produced by the runner; next event lands after expiry
        ("svc-order", "locks-alpha", "续约", 3),       # RENEW_EXPIRED
        ("svc-pay", "locks-alpha", "申请", 4),         # granted from freed slot
        ("svc-pay", "locks-alpha", "释放", 5),
    ]
    return [{"seq": i + 1, "timestamp_ms": ts, "holder_id": h,
             "domain_id": d, "action": a}
            for i, (h, d, a, ts) in enumerate(rows)]


def main() -> None:
    for scenario, events in (
        ("contention", contention_events()),
        ("expiry", expiry_events()),
        ("replay", contention_events()),
    ):
        spec = {"session": base_session(),
                "domains": domains_for(scenario), "holders": holders()}
        write_json(EX / scenario / "spec.json", spec)
        write_ndjson(EX / scenario / "events.ndjson", events)
        print(f"wrote examples/{scenario}/spec.json and events.ndjson "
              f"({len(events)} events)")


if __name__ == "__main__":
    main()
