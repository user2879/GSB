from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

import numpy as np

from .database import (
    canonical_value,
    connect,
    database_fingerprint,
    table_content_digest,
    table_row_count,
    write_json_atomic,
)


DEFAULT_SEED = 20261118
DEFAULT_ROWS = 48


def build_sample_database(
    output: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    rows: int = DEFAULT_ROWS,
    overwrite: bool = True,
) -> dict[str, Any]:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise FileExistsError(path)
        path.unlink()
    rng = np.random.default_rng(seed)
    customers = np.array(["alice", "bob", "carol", "dave", "erin"])
    notes = np.array(["standard", "priority", "hold"])
    quantity = rng.integers(1, 9, size=rows)
    unit_price_cents = rng.integers(500, 3000, size=rows)
    discount_rate = np.round(rng.uniform(0.0, 0.20, size=rows), 6)
    customer_values = customers[rng.integers(0, len(customers), size=rows)]
    note_values = notes[rng.integers(0, len(notes), size=rows)]
    connection = connect(path)
    try:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute(
            """
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                unit_price_cents INTEGER NOT NULL,
                tax_cents INTEGER NOT NULL,
                total_cents INTEGER NOT NULL,
                net_cents INTEGER NOT NULL,
                discount_rate REAL NOT NULL,
                note TEXT NOT NULL
            )
            """
        )
        records: list[tuple[Any, ...]] = []
        for index in range(rows):
            qty = int(quantity[index])
            price_cents = int(unit_price_cents[index])
            discount = Decimal(str(float(discount_rate[index])))
            discount_at_scale4 = Decimal(int(discount * 10000)) / Decimal(10000)
            tax_cents = int(Decimal(price_cents * qty) * Decimal("0.08"))
            exact_total_units = Decimal(price_cents * qty) * (
                Decimal(1) - discount_at_scale4
            )
            total_cents = int(exact_total_units)  # truncates toward zero at scale 2
            net_cents = total_cents - tax_cents
            records.append(
                (
                    index + 1,
                    str(customer_values[index]),
                    qty,
                    price_cents,
                    tax_cents,
                    total_cents,
                    net_cents,
                    float(discount_rate[index]),
                    str(note_values[index]),
                )
            )
        connection.executemany(
            """
            INSERT INTO orders(
                id, customer, quantity, unit_price_cents, tax_cents,
                total_cents, net_cents, discount_rate, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        fingerprint = database_fingerprint(connection)
        seed_payload = {
            "seed": seed,
            "rows": rows,
            "schema": connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='orders'"
            ).fetchone()[0],
            "semantic_digest": fingerprint["data_version"],
            "table_content_digest": table_content_digest(connection, "orders"),
            "table_rows": table_row_count(connection, "orders"),
        }
    finally:
        connection.close()
    seed_path = path.with_name(path.name + ".seed.json")
    write_json_atomic(seed_path, seed_payload)
    return seed_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the deterministic dbml sample database")
    parser.add_argument(
        "--output",
        default="artifacts/sample_base.sqlite3",
        help="SQLite output path",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args(argv)
    payload = build_sample_database(
        args.output,
        seed=args.seed,
        rows=args.rows,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
