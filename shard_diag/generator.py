from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .routing import compare_route
from .spec import validate_spec


DEFAULT_SEED = 20261126
DEFAULT_PORTS = (49311, 49312, 49313)


TABLES: dict[str, dict[int, list[int]]] = {
    "orders": {
        1: [0, 100, 200, 400, 800],
        2: [0, 80, 220, 450, 800],
        3: [0, 70, 200, 460, 850],
    },
    "audit": {
        1: [0, 50, 150, 300],
        2: [0, 40, 180, 300],
        3: [0, 35, 170, 320],
    },
}


EVENT_ROWS = [
    (1, "write", "db-a", "orders", 210),
    (2, "query", "db-a", "audit", 45),
    (3, "write", "db-b", "orders", 90),
    (4, "query", "db-b", "orders", 210),
    (5, "write", "db-a", "orders", 700),
    (6, "query", "db-c", "orders", 430),
    (7, "write", "db-c", "audit", 250),
    (8, "query", "db-a", "audit", 170),
    (9, "write", "db-b", "audit", 250),
    (10, "query", "db-a", "orders", 430),
    (11, "write", "db-c", "audit", 45),
    (12, "query", "db-b", "orders", 60),
    (13, "write", "db-a", "orders", 450),
    (14, "query", "db-a", "audit", 35),
]


def _database_declarations(route_version: int) -> list[dict[str, Any]]:
    defaults = [
        ("db-a", DEFAULT_PORTS[0], 3, 2, 101),
        ("db-b", DEFAULT_PORTS[1], 2, 1, 202),
        ("db-c", DEFAULT_PORTS[2], 2, 2, 303),
    ]
    return [
        {
            "id": db_id,
            "host": "127.0.0.1",
            "port": port,
            "lease_limit": lease_limit,
            "concurrency": concurrency,
            "seed": seed,
            "route_version": route_version,
        }
        for db_id, port, lease_limit, concurrency, seed in defaults
    ]


def _table_declarations() -> list[dict[str, Any]]:
    return [
        {
            "name": table,
            "key_column": "order_id" if table == "orders" else "audit_id",
            "route_version": version,
            "boundaries": boundaries,
        }
        for table, history in TABLES.items()
        for version, boundaries in history.items()
    ]


def _events(edited: bool = False) -> list[dict[str, Any]]:
    rows = list(EVENT_ROWS)
    if edited:
        rows = [(seq, kind, db_id, table, key) for seq, kind, db_id, table, key in rows]
        # Editing the event stream changes its content digest without changing schema validity.
        rows[1] = (2, "write", "db-a", "audit", 45)
        rows[6] = (7, "write", "db-c", "audit", 260)
        rows.append((15, "query", "db-b", "orders", 75))
    return [
        {
            "seq": seq,
            "kind": kind,
            "database": db_id,
            "table": table,
            "route_key": key,
        }
        for seq, kind, db_id, table, key in rows
    ]


def build_spec_values(route_version: int = 2, *, edited: bool = False) -> dict[str, Any]:
    return {
        "format_version": 1,
        "global_lease_limit": 6,
        "databases": _database_declarations(route_version),
        "tables": _table_declarations(),
        "events": _events(edited=edited),
    }


def generate_workspace(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    configs = root / "configs"
    samples = root / "samples"
    ledger = root / "ledger"
    for directory in (configs, samples, ledger):
        directory.mkdir(parents=True, exist_ok=True)

    spec_values = build_spec_values(2)
    evolved_values = build_spec_values(3)
    edited_values = build_spec_values(2, edited=True)

    # Validate before writing so generated fixtures cannot drift into an invalid schema.
    old_spec = validate_spec(spec_values)
    validate_spec(evolved_values)
    validate_spec(edited_values)

    files: dict[str, Path] = {
        "spec": configs / "local.json",
        "evolved_spec": configs / "local_v3.json",
        "edited_spec": configs / "local_edited.json",
        "batch_plans": samples / "batch_plans.json",
        "expected_drift": samples / "expected_drift.json",
    }
    files["spec"].write_text(json.dumps(spec_values, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files["evolved_spec"].write_text(
        json.dumps(evolved_values, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files["edited_spec"].write_text(
        json.dumps(edited_values, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    batch_plans = {
        "format_version": 1,
        "seed": DEFAULT_SEED,
        "sample_count": 2,
        "plans": {
            "layout_a": {"batch_size": 3, "startup_order": ["db-c", "db-a", "db-b"]},
            "layout_b": {"batch_size": 5, "startup_order": ["db-b", "db-c", "db-a"]},
            "layout_c": {"batch_size": 1, "startup_order": ["db-a", "db-b", "db-c"]},
        },
    }
    files["batch_plans"].write_text(
        json.dumps(batch_plans, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    drift_records: list[dict[str, Any]] = []
    for event in old_spec.events:
        db = old_spec.database(event.database)
        rule = old_spec.event_rule(event)
        previous = old_spec.previous_rule(event.table, db.route_version)
        drift_records.append(
            compare_route(
                event_seq=event.seq,
                db_id=event.database,
                table=event.table,
                key_column=rule.key_column,
                route_key=event.route_key,
                previous_rule=previous,
                current_rule=rule,
            )
        )
    files["expected_drift"].write_text(
        json.dumps(
            {
                "format_version": 1,
                "seed": DEFAULT_SEED,
                "routing_digest": old_spec.spec_digest(),
                "records": drift_records,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "ok": True,
        "generator_seed": DEFAULT_SEED,
        "files": {name: str(path) for name, path in files.items()},
        "databases": [database.id for database in old_spec.databases],
        "tables": old_spec.table_versions(),
        "events": len(old_spec.events),
        "spec_digest": old_spec.spec_digest(),
        "event_digest": old_spec.event_digest(),
    }
