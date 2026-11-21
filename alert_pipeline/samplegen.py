from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .util import write_json_file


SERVICES = (
    "auth",
    "checkout",
    "database",
    "gateway",
    "inventory",
    "notifier",
    "payment",
)

DEPENDENCIES = (
    ("auth", "database"),
    ("checkout", "auth"),
    ("checkout", "inventory"),
    ("checkout", "payment"),
    ("gateway", "checkout"),
    ("inventory", "database"),
    ("notifier", "checkout"),
    ("payment", "database"),
)


def _graph(version: int, include_search: bool = False) -> dict[str, Any]:
    services = list(SERVICES)
    dependencies = [
        {"from": source, "to": target} for source, target in DEPENDENCIES
    ]
    if include_search:
        services.append("search")
        dependencies.append({"from": "search", "to": "database"})
        dependencies.append({"from": "gateway", "to": "search"})
    return {
        "schema": "alert-pipeline/service-graph",
        "version": version,
        "services": [{"service_id": service} for service in sorted(services)],
        "dependencies": sorted(
            dependencies, key=lambda edge: (edge["from"], edge["to"])
        ),
    }


def _rules(version: int, include_search: bool = False, checkout_depth: int = 3) -> dict[str, Any]:
    rules: list[dict[str, Any]] = [
        {
            "rule_id": "checkout-both",
            "services": ["checkout"],
            "directions": ["upstream", "downstream"],
            "max_depth": checkout_depth,
            "include_self": True,
        },
        {
            "rule_id": "gateway-upstream",
            "services": ["gateway"],
            "directions": ["upstream"],
            "max_depth": 2,
            "include_self": True,
        },
        {
            "rule_id": "database-upstream",
            "services": ["database"],
            "directions": ["upstream"],
            "max_depth": 0,
            "include_self": True,
        },
        {
            "rule_id": "storage-both",
            "services": ["payment", "inventory"],
            "directions": ["upstream", "downstream"],
            "max_depth": 2,
            "include_self": True,
        },
    ]
    if include_search:
        rules.append(
            {
                "rule_id": "search-upstream",
                "services": ["search"],
                "directions": ["upstream"],
                "max_depth": 2,
                "include_self": True,
            }
        )
    return {
        "schema": "alert-pipeline/root-rules",
        "version": version,
        "defaults": {
            "directions": ["upstream", "downstream"],
            "max_depth": 2,
            "include_self": True,
        },
        "rules": sorted(rules, key=lambda rule: rule["rule_id"]),
    }


def _direct_dependencies(service_id: str) -> list[str]:
    return sorted(target for source, target in DEPENDENCIES if source == service_id)


def generate_alert_records(seed: int, count: int = 40) -> list[dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64(seed))
    base_timestamp = 1_760_000_000_000
    templates = [
        ("checkout", "critical", "checkout:critical:payment-failure"),
        ("payment", "critical", "payment:critical:database-timeout"),
        ("inventory", "warning", "inventory:warning:stock-latency"),
        ("gateway", "warning", "gateway:warning:upstream-5xx"),
        ("auth", "critical", "auth:critical:token-store"),
        ("database", "critical", "database:critical:connection-pool"),
        ("notifier", "info", "notifier:info:queue-delay"),
    ]
    records: list[dict[str, Any]] = []
    timestamp = base_timestamp

    records.append(
        {
            "version": 1,
            "timestamp_ms": timestamp,
            "service_id": "checkout",
            "level": "critical",
        }
    )
    records.append(
        {
            "version": 3,
            "timestamp_ms": timestamp,
            "service_id": "checkout",
            "level": "critical",
            "fingerprint": "derived:checkout:critical",
            "silence_window_ms": 30000,
            "root_candidates": ["payment", "database"],
            "strategy_id": "policy-payment-failure",
        }
    )

    while len(records) < count:
        timestamp += int(rng.integers(0, 4200))
        service, level, fingerprint = templates[int(rng.integers(0, len(templates)))]
        version = (len(records) % 3) + 1
        if version == 3 and fingerprint.startswith("derived:"):
            fingerprint = f"{service}:{level}:derived"
        window = int((15000, 30000, 45000)[int(rng.integers(0, 3))])
        record: dict[str, Any] = {
            "version": version,
            "timestamp_ms": timestamp,
            "service_id": service,
            "level": level,
        }
        if version >= 2:
            record["fingerprint"] = fingerprint
            record["silence_window_ms"] = window
            record["tags"] = {
                "region": ("cn-east", "cn-north")[int(rng.integers(0, 2))],
                "source": "synthetic",
            }
        if version == 3:
            record["root_candidates"] = _direct_dependencies(service)
            record["strategy_id"] = f"policy-{service}-{level}"
        records.append(record)
    return records


def _manifest(
    declaration_version: int,
    format_version: int,
    silence_window_ms: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema": "alert-pipeline/manifest",
        "declaration_version": declaration_version,
        "format_version": format_version,
        "default_silence_window_ms": silence_window_ms,
        "seed": seed,
        "batch_size": 8,
        "global_concurrency": 4,
        "per_service_concurrency": 1,
        "services_path": "services.json",
        "rules_path": "rules.json",
        "alerts_path": "alerts.ndjson",
        "database_path": "../data/sample.db",
        "log_path": "../logs/pipeline.jsonl",
        "output_dir": "../artifacts/sample-run",
    }


def generate_samples(output_dir: str | Path, seed: int = 20261122) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    base_graph = _graph(version=3)
    evolved_graph = _graph(version=4, include_search=True)
    base_rules = _rules(version=2)
    evolved_rules = _rules(version=3, include_search=True, checkout_depth=2)

    write_json_file(output / "services.json", base_graph)
    write_json_file(output / "services_v2.json", evolved_graph)
    write_json_file(output / "rules.json", base_rules)
    write_json_file(output / "rules_v2.json", evolved_rules)

    records = generate_alert_records(seed)
    alert_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )
    (output / "alerts.ndjson").write_text(alert_text, encoding="utf-8", newline="\n")

    manifest_v1 = _manifest(1, 3, 30000, seed)
    write_json_file(output / "manifest.json", manifest_v1)
    manifest_v2 = _manifest(2, 3, 45000, seed)
    manifest_v2.update(
        {
            "services_path": "services_v2.json",
            "rules_path": "rules_v2.json",
            "database_path": "../data/sample.db",
            "log_path": "../logs/pipeline.jsonl",
            "output_dir": "../artifacts/sample-run-v2",
        }
    )
    write_json_file(output / "manifest_v2.json", manifest_v2)
    for name in (
        "services.json",
        "services_v2.json",
        "rules.json",
        "rules_v2.json",
        "alerts.ndjson",
        "manifest.json",
        "manifest_v2.json",
    ):
        paths[name] = str((output / name).resolve())
    return paths
