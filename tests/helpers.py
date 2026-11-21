from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alert_pipeline.config import PipelineConfig
from alert_pipeline.util import write_json_file


def write_graph(path: Path, services: list[str], dependencies: list[tuple[str, str]], version: int = 1) -> None:
    write_json_file(
        path,
        {
            "version": version,
            "services": [{"service_id": service} for service in services],
            "dependencies": [
                {"from": source, "to": target} for source, target in dependencies
            ],
        },
    )


def write_rules(
    path: Path,
    version: int = 1,
    rules: list[dict[str, Any]] | None = None,
    defaults: dict[str, Any] | None = None,
) -> None:
    write_json_file(
        path,
        {
            "version": version,
            "defaults": defaults
            or {"directions": ["upstream", "downstream"], "max_depth": 2, "include_self": True},
            "rules": rules or [],
        },
    )


def write_alerts(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )


def make_config(
    root: Path,
    *,
    services: list[str] | None = None,
    dependencies: list[tuple[str, str]] | None = None,
    rules: list[dict[str, Any]] | None = None,
    records: list[dict[str, Any]] | None = None,
    declaration_version: int = 1,
    format_version: int = 3,
    silence_window_ms: int = 30000,
    batch_size: int = 2,
    seed: int = 7,
    global_limit: int = 3,
    per_service_limit: int = 1,
) -> PipelineConfig:
    root.mkdir(parents=True, exist_ok=True)
    services = services or ["svc-a", "svc-b", "svc-c"]
    dependencies = dependencies if dependencies is not None else [("svc-a", "svc-b")]
    records = records or [
        {
            "version": 1,
            "timestamp_ms": 1000,
            "service_id": "svc-a",
            "level": "critical",
        }
    ]
    services_path = root / "services.json"
    rules_path = root / "rules.json"
    alerts_path = root / "alerts.ndjson"
    write_graph(services_path, services, dependencies)
    write_rules(rules_path, rules=rules)
    write_alerts(alerts_path, records)
    return PipelineConfig(
        declaration_version=declaration_version,
        format_version=format_version,
        default_silence_window_ms=silence_window_ms,
        seed=seed,
        batch_size=batch_size,
        global_concurrency=global_limit,
        per_service_concurrency=per_service_limit,
        services_path=services_path,
        rules_path=rules_path,
        alerts_path=alerts_path,
        database_path=root / "ledger.db",
        log_path=root / "pipeline.jsonl",
        output_dir=root / "output",
    )


def cli_config(manifest: Path, database: Path, log: Path, output: Path) -> PipelineConfig:
    from alert_pipeline.cli import build_config, build_parser

    args = build_parser().parse_args(
        [
            "run",
            "--manifest",
            str(manifest),
            "--database",
            str(database),
            "--log",
            str(log),
            "--output-dir",
            str(output),
        ]
    )
    return build_config(args)
