"""Deterministically generate all sample input files (stdlib only)."""
from __future__ import annotations

import json
from pathlib import Path

from lease_middleware.spec import canonical_json, load_spec
from hashlib import sha256

HERE = Path(__file__).resolve().parent


def digest(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


CONFIG = {
    "configs": [
        {
            "key": "feature.flags",
            "ttl_seconds": 30,
            "values": [
                {"version": 1, "value": {"dark_mode": False, "rollout": 10}},
                {"version": 2, "value": {"dark_mode": True, "rollout": 50}},
                {"version": 3, "value": {"dark_mode": True, "rollout": 100}},
            ],
        },
        {
            "key": "service.endpoint",
            "ttl_seconds": 60,
            "values": [
                {"version": 1, "value": {"url": "https://v1.example.local", "retries": 2}},
                {"version": 2, "value": {"url": "https://v2.example.local", "retries": 3}},
            ],
        },
        {
            "key": "cache.policy",
            "ttl_seconds": 45,
            "values": [
                {"version": 1, "value": {"max_age": 60, "stale": 5}},
                {"version": 2, "value": {"max_age": 30, "stale": 2}},
            ],
        },
    ],
    "clients": [
        {"id": "client-r1", "subscribed_keys": ["feature.flags", "service.endpoint"],
         "lease_duration_ms": 400, "format_version": "r1"},
        {"id": "client-r2", "subscribed_keys": ["feature.flags", "cache.policy"],
         "lease_duration_ms": 600, "format_version": "r2"},
        {"id": "client-r3", "subscribed_keys": ["service.endpoint", "cache.policy"],
         "lease_duration_ms": 800, "format_version": "r3"},
    ],
}

# Deterministic change script consumed by the demo runner.
CHANGES = {
    "steps": [
        {"action": "grant", "client_id": "client-r1", "config_key": "feature.flags"},
        {"action": "grant", "client_id": "client-r2", "config_key": "feature.flags"},
        {"action": "grant", "client_id": "client-r3", "config_key": "service.endpoint"},
        {"action": "renew", "client_id": "client-r1", "config_key": "feature.flags",
         "expected_version": 1},
        {"action": "publish", "config_key": "feature.flags", "version": 2,
         "value": CONFIG["configs"][0]["values"][1]["value"]},
        {"action": "publish", "config_key": "service.endpoint", "version": 2,
         "value": CONFIG["configs"][1]["values"][1]["value"]},
        {"action": "sleep_ms", "duration_ms": 500},
        {"action": "sweep"},
        {"action": "grant", "client_id": "client-r3", "config_key": "cache.policy"},
        {"action": "publish", "config_key": "cache.policy", "version": 2,
         "value": CONFIG["configs"][2]["values"][1]["value"]},
    ]
}


def legacy_records(spec) -> dict[str, list[dict]]:
    flags = spec.configs["feature.flags"]
    endpoint = spec.configs["service.endpoint"]
    base = 1_900_000_000_000  # fixed epoch so files are deterministic
    r1 = [
        {"format": "r1", "lease_id": "L:client-r1:feature.flags:0001",
         "config_key": "feature.flags", "expires_at_ms": base + 400},
        {"format": "r1", "lease_id": "L:client-r1:service.endpoint:0002",
         "config_key": "service.endpoint", "expires_at_ms": base + 400},
    ]
    r2 = [
        {"format": "r2", "lease_id": "L:client-r2:feature.flags:0001",
         "config_key": "feature.flags", "expires_at_ms": base + 600,
         "lease_version": 2, "client_id": "client-r2"},
        {"format": "r2", "lease_id": "L:client-r2:cache.policy:0002",
         "config_key": "cache.policy", "expires_at_ms": base + 600,
         "lease_version": 1, "client_id": "client-r2"},
    ]
    r3 = [
        {"format": "r3", "lease_id": "L:client-r3:service.endpoint:0001",
         "config_key": "service.endpoint", "expires_at_ms": base + 800,
         "lease_version": 1, "client_id": "client-r3", "renewal_count": 2,
         "value_digest": endpoint.digest_at(1), "parent_lease_id": None},
        {"format": "r3", "lease_id": "L:client-r3:cache.policy:0002",
         "config_key": "cache.policy", "expires_at_ms": base + 800,
         "lease_version": 1, "client_id": "client-r3", "renewal_count": 0,
         "value_digest": spec.configs["cache.policy"].digest_at(1),
         "parent_lease_id": "L:client-r3:service.endpoint:0001"},
    ]
    # Same logical lease represented in all three formats: normalized views
    # must be identical (proves no version/digest drift across formats).
    same = {
        "r1": {"format": "r1", "lease_id": "L:client-r3:feature.flags:0009",
               "config_key": "feature.flags", "expires_at_ms": base + 800},
        "r2": {"format": "r2", "lease_id": "L:client-r3:feature.flags:0009",
               "config_key": "feature.flags", "expires_at_ms": base + 800,
               "lease_version": 2, "client_id": "client-r3"},
        "r3": {"format": "r3", "lease_id": "L:client-r3:feature.flags:0009",
               "config_key": "feature.flags", "expires_at_ms": base + 800,
               "lease_version": 2, "client_id": "client-r3", "renewal_count": 0,
               "value_digest": flags.digest_at(2), "parent_lease_id": None},
    }
    return {"r1": r1, "r2": r2, "r3": r3, "same_lease": [same["r1"], same["r2"], same["r3"]]}


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "config.json").write_text(
        json.dumps(CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "changes.json").write_text(
        json.dumps(CHANGES, indent=2, ensure_ascii=False), encoding="utf-8")
    spec = load_spec(HERE / "config.json")
    records = legacy_records(spec)
    for name, payload in records.items():
        (HERE / f"leases_{name}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("generated samples in", HERE)


if __name__ == "__main__":
    main()
