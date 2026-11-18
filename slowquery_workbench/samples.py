"""Deterministic, offline sample data and query catalog loading."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .errors import fail
from .session import Session, _split_sql_statements


DEFAULT_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise fail("E_DATA", f"cannot read sample input: {path}", {"error": str(exc)}) from exc


def generate_customers(spec: dict[str, Any]) -> list[tuple[Any, ...]]:
    count = int(spec["count"])
    seed = int(spec.get("seed", 157))
    cities = [str(item) for item in spec["cities"]]
    segments = [str(item) for item in spec["segments"]]
    if count < 1 or not cities or not segments:
        raise fail("E_DATA", "customer sample specification is invalid")
    base = date.fromisoformat(str(spec.get("base_date", "2024-01-01")))
    rows: list[tuple[Any, ...]] = []
    for number in range(1, count + 1):
        city = cities[(number * 7 + seed) % len(cities)]
        segment = segments[(number * 5 + seed) % len(segments)]
        created = base + timedelta(days=(number * 13) % 365)
        rows.append((number, f"Customer {number:05d}", city, segment, created.isoformat()))
    return rows


def generate_orders(spec: dict[str, Any], customer_count: int) -> list[tuple[Any, ...]]:
    count = int(spec["count"])
    seed = int(spec.get("seed", 157))
    statuses = [str(item) for item in spec["statuses"]]
    if count < 1 or customer_count < 1 or not statuses:
        raise fail("E_DATA", "order sample specification is invalid")
    base = date.fromisoformat(str(spec.get("base_date", "2024-01-01")))
    rows: list[tuple[Any, ...]] = []
    for number in range(1, count + 1):
        customer_id = ((number * 37 + seed) % customer_count) + 1
        status = statuses[(number * 11 + seed) % len(statuses)]
        amount = round((((number * 7919 + seed) % 200000) / 100.0) + 1.0, 2)
        created = base + timedelta(days=(number * 17) % 700)
        rows.append((number, customer_id, status, amount, created.isoformat()))
    return rows


def load_queries(samples_dir: str | Path = DEFAULT_SAMPLES_DIR) -> list[dict[str, Any]]:
    root = Path(samples_dir)
    payload = _read_json(root / "queries.json")
    if not isinstance(payload, list):
        raise fail("E_DATA", "queries.json must contain a list")
    return payload


def load_samples(
    session: Session,
    samples_dir: str | Path = DEFAULT_SAMPLES_DIR,
    *,
    corrupt_input: bool = False,
) -> dict[str, Any]:
    """Load the deterministic offline dataset into a fresh session.

    ``corrupt_input`` switches the seed input to samples/corrupt_data.sql. It
    does not fabricate an error: SQLite executes the broken file and its genuine
    integrity failure is translated to E_DATA.
    """

    root = Path(samples_dir)
    try:
        schema_script = (root / "schema.sql").read_text(encoding="utf-8")
        dataset = _read_json(root / "dataset.json")
    except OSError as exc:
        raise fail("E_DATA", f"cannot read sample files from {root}", {"error": str(exc)}) from exc
    schema_statements = _split_sql_statements(schema_script)
    if not schema_statements:
        raise fail("E_DATA", "sample schema is empty")

    if corrupt_input:
        for statement in schema_statements:
            session.create_domain_table(statement)
        corrupt_path = root / "corrupt_data.sql"
        try:
            corrupt_script = corrupt_path.read_text(encoding="utf-8")
            statements = _split_sql_statements(corrupt_script)
            with session.conn:
                for statement in statements:
                    session.conn.execute(statement)
        except sqlite3.Error as exc:
            session.logger.error(
                "sample_data_corrupt",
                "corrupt sample input failed integrity checks",
                code="E_DATA",
                details={"file": str(corrupt_path), "error": str(exc)},
            )
            raise fail(
                "E_DATA",
                f"corrupt sample data rejected: {exc}",
                {"file": str(corrupt_path), "sqlite_error": str(exc)},
            ) from exc
        except OSError as exc:
            raise fail("E_DATA", f"cannot read corrupt sample input: {corrupt_path}") from exc
        raise fail(
            "E_DATA",
            "corrupt sample input unexpectedly executed without an integrity error",
            {"file": str(corrupt_path)},
        )

    customer_spec = dict(dataset["customers"])
    order_spec = dict(dataset["orders"])
    data_path = root / "data.sql"
    try:
        seed_statements = _split_sql_statements(data_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise fail("E_DATA", f"cannot read deterministic data input: {data_path}", {"error": str(exc)}) from exc
    if not seed_statements:
        raise fail("E_DATA", "deterministic data input is empty", {"file": str(data_path)})
    rows_by_table: dict[str, tuple[tuple[str, ...], list[tuple[Any, ...]]]] = {}
    index_specs = [dict(item) for item in dataset.get("indexes", [])]
    thresholds = dict(dataset.get("thresholds", {}))
    return session.bootstrap_domain(
        schema_statements,
        rows_by_table,
        index_specs,
        seed_statements=seed_statements,
        threshold_ms=int(thresholds.get("slow_ms", 1)),
        threshold_rows=int(thresholds.get("scan_rows", 1000)),
    )
