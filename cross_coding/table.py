from __future__ import annotations

import hashlib
import random
from typing import Any

from .hashing import fingerprint, stable_json


def column_seed(global_seed: int, column_name: str) -> int:
    digest = hashlib.sha256(
        f"{global_seed}:{column_name}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def generate_column_values(spec: dict[str, Any], row_count: int, global_seed: int):
    """Generate one column and its length-row_count missing bitmap."""
    name = str(spec["name"])
    rng = random.Random(column_seed(int(global_seed), name))
    missing_rate = float(spec.get("missing_rate", 0.0))
    missing_bitmap = [rng.random() < missing_rate for _ in range(row_count)]

    if "missing_bitmap" in spec and spec["missing_bitmap"] is not None:
        explicit = [bool(x) for x in spec["missing_bitmap"]]
        if len(explicit) != row_count:
            raise ValueError("missing_bitmap length mismatch")
        missing_bitmap = explicit

    if "values" in spec and spec["values"] is not None:
        values = list(spec["values"])
        if len(values) != row_count:
            raise ValueError("values length mismatch")
        for index, is_missing in enumerate(missing_bitmap):
            if is_missing:
                values[index] = None
            elif values[index] is None:
                missing_bitmap[index] = True
        return values, missing_bitmap

    column_type = spec["type"]
    values: list[Any] = []
    if column_type == "categorical":
        categories = list(spec.get("categories", []))
        for is_missing in missing_bitmap:
            values.append(None if is_missing else rng.choice(categories))
    elif column_type == "numeric":
        low = float(spec.get("min", 0))
        high = float(spec.get("max", 100))
        integer_values = bool(spec.get("integer", False))
        for is_missing in missing_bitmap:
            if is_missing:
                values.append(None)
            elif integer_values:
                values.append(rng.randint(int(low), int(high)))
            else:
                values.append(round(rng.uniform(low, high), 6))
    else:
        raise ValueError(f"unsupported column type: {column_type}")
    return values, missing_bitmap


def column_fingerprint(spec: dict[str, Any], values: list[Any], bitmap: list[bool]) -> str:
    return fingerprint(
        "col",
        {
            "name": spec["name"],
            "type": spec["type"],
            "values": values,
            "missing_bitmap": [bool(x) for x in bitmap],
        },
    )


def numeric_quantization(value: float | None, fixed_point_bits: int, levels: int):
    """Return both fixed-point integer and quantized bin for one value."""
    if value is None:
        return None, None
    scale = 1 << max(0, int(fixed_point_bits))
    fixed_int = int(round(float(value) * scale))
    low = 0.0
    high = max(1.0, float(levels))
    return fixed_int, min(int(levels) - 1, max(0, int(float(value) / high * levels)))


def build_table(feature_doc: dict[str, Any]) -> dict[str, Any]:
    table_spec = feature_doc.get("table", feature_doc)
    seed = int(table_spec["seed"])
    row_count = int(table_spec["row_count"])
    columns: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for spec in table_spec["columns"]:
        name = str(spec["name"])
        values, bitmap = generate_column_values(spec, row_count, seed)
        item = dict(spec)
        item.update(
            {
                "name": name,
                "values": values,
                "missing_bitmap": bitmap,
                "fingerprint": column_fingerprint(spec, values, bitmap),
            }
        )
        columns[name] = item
        order.append(name)
    return {
        "seed": seed,
        "row_count": row_count,
        "order": order,
        "columns": columns,
        "table_fingerprint": fingerprint(
            "table",
            {
                "seed": seed,
                "row_count": row_count,
                "columns": [columns[name]["fingerprint"] for name in order],
            },
        ),
    }


def table_digest(table: dict[str, Any]) -> str:
    return fingerprint(
        "table-digest",
        {
            "row_count": table["row_count"],
            "columns": {
                name: {
                    "fingerprint": table["columns"][name]["fingerprint"],
                    "missing_count": sum(table["columns"][name]["missing_bitmap"]),
                }
                for name in table["order"]
            },
        },
    )


def canonical_feature_hash(feature_doc: dict[str, Any]) -> str:
    return fingerprint("feature-config", feature_doc)


def canonical_policy_hash(policy: dict[str, Any]) -> str:
    return fingerprint("policy-config", policy)


def json_digest(payload: Any) -> str:
    return fingerprint("artifact", stable_json(payload))
