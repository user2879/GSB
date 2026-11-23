from __future__ import annotations

import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
SEED = 20240601


def main() -> None:
    rng = random.Random(SEED)
    regions = []
    for region_id, tax_bps in enumerate((500, 650, 800, 950), start=1):
        regions.append([region_id, tax_bps])

    customers = []
    for customer_id in range(1, 61):
        region_id = 1 + ((customer_id * 7 + 3) % 4)
        tier = customer_id % 4
        active = 1 if customer_id % 5 != 0 else 0
        credit = round(100.0 + rng.random() * 900.0, 2)
        customers.append([customer_id, region_id, tier, active, credit])

    orders = []
    for order_id in range(1, 481):
        customer_id = 1 + ((order_id * 37 + 11) % 60)
        amount = round(1.0 + rng.random() * 99.0, 2)
        status = order_id % 3
        qty = 1 + (order_id % 10)
        discount = None if order_id % 37 == 0 else round(rng.random() * 0.25, 4)
        orders.append([order_id, customer_id, amount, status, qty, discount])

    tables = [
        table(
            "regions",
            [("id", "integer"), ("tax_bps", "integer")],
            regions,
            null_columns=set(),
        ),
        table(
            "customers",
            [
                ("id", "integer"),
                ("region_id", "integer"),
                ("tier", "integer"),
                ("active", "integer"),
                ("credit", "float"),
            ],
            customers,
            null_columns=set(),
        ),
        table(
            "orders",
            [
                ("id", "integer"),
                ("customer_id", "integer"),
                ("amount", "float"),
                ("status", "integer"),
                ("qty", "integer"),
                ("discount", "float"),
            ],
            orders,
            null_columns={"discount"},
        ),
    ]
    workload = {
        "format_version": 1,
        "tuning": {
            "seed": SEED,
            "fixed_point_scale": 1000,
            "exact_scan_threshold_rows": 100,
            "sample_size": 128,
            "confidence_level": 0.95,
            "max_candidates": 18,
            "page_size_bytes": 4096,
            "row_overhead_bytes": 16,
            "global_concurrency": 4,
            "per_table_scan_quota": 1,
            "memory_budget_bytes": 64 * 1024 * 1024,
            "bias_threshold_ratio": 1000000000.0,
            "comparison_weight": 1,
            "io_page_weight": 25,
            "memory_weight": 1,
        },
        "tables": tables,
        "query": {
            "sql": (
                "SELECT customers.region_id, SUM(orders.amount) AS total_amount, "
                "COUNT(*) AS order_count FROM orders "
                "INNER JOIN customers ON orders.customer_id = customers.id "
                "INNER JOIN regions ON customers.region_id = regions.id "
                "WHERE orders.amount >= 10.0 AND orders.status = 1 AND customers.active = 1 "
                "GROUP BY customers.region_id ORDER BY customers.region_id"
            )
        },
    }
    destination = ROOT / "samples" / "demo_workload.json"
    destination.write_text(json.dumps(workload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    whitespace_variant = json.loads(destination.read_text(encoding="utf-8"))
    whitespace_variant["query"]["sql"] = (
        " SELECT  customers.region_id, SUM(orders.amount) AS total_amount, COUNT(*) AS order_count\n"
        " FROM orders INNER JOIN customers ON ((orders.customer_id = customers.id))\n"
        " INNER JOIN regions ON (((customers.region_id = regions.id)))\n"
        " WHERE (((orders.amount >= 10.0)) AND ((orders.status = 1)) AND ((customers.active = 1)))\n"
        " GROUP BY customers.region_id ORDER BY customers.region_id ; "
    )
    (ROOT / "samples" / "demo_workload_whitespace.json").write_text(
        json.dumps(whitespace_variant, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    bad_overlap = json.loads(destination.read_text(encoding="utf-8"))
    bad_overlap["tables"][2]["histograms"]["amount"]["bins"][1]["lo"] -= 1
    (ROOT / "samples" / "bad_overlap_histogram.json").write_text(
        json.dumps(bad_overlap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    bad_schema = json.loads(destination.read_text(encoding="utf-8"))
    bad_schema["tables"][1]["null_bitmap"][0] = [0]
    (ROOT / "samples" / "bad_schema_bitmap.json").write_text(
        json.dumps(bad_schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def table(name, columns, rows, *, null_columns):
    column_names = [column[0] for column in columns]
    bitmap = []
    for row in rows:
        bitmap.append([1 if value is None and column in null_columns else 0 for value, column in zip(row, column_names)])
    histograms = {}
    for index, (column_name, column_type) in enumerate(columns):
        values = [row[index] for row in rows if row[index] is not None]
        histograms[column_name] = {
            "version": 1,
            "bins": bins(values, column_type),
        }
    return {
        "name": name,
        "columns": [{"name": name_, "type": type_} for name_, type_ in columns],
        "rows": rows,
        "null_bitmap": bitmap,
        "histograms": histograms,
    }


def bins(values, column_type):
    minimum = min(values)
    maximum = max(values)
    if column_type == "integer":
        start = int(minimum)
        stop = int(maximum) + 1
        bucket_count = min(8, max(1, stop - start))
        width = max(1, math.ceil((stop - start) / bucket_count))
        boundaries = list(range(start, stop, width))
        if boundaries[-1] != stop:
            boundaries.append(stop)
        result = []
        for lo, hi in zip(boundaries, boundaries[1:]):
            count = sum(1 for value in values if lo <= value < hi)
            result.append({"lo": lo, "hi": hi, "count": count})
        return result
    bucket_count = 8
    start = float(minimum)
    stop = float(maximum) + 1.0
    width = (stop - start) / bucket_count
    boundaries = [start + width * index for index in range(bucket_count + 1)]
    boundaries[-1] = stop
    result = []
    for lo, hi in zip(boundaries, boundaries[1:]):
        count = sum(1 for value in values if lo <= value < hi)
        result.append({"lo": round(lo, 6), "hi": round(hi, 6), "count": count})
    return result


if __name__ == "__main__":
    main()
