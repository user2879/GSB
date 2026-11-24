from __future__ import annotations

from typing import Any

from .hashing import fingerprint


CACHE_CODE_VERSION = "fixedpoint-v1"


def stage_cache_key(
    upstream_fingerprint: str | list[str] | dict[str, Any],
    parameter_fingerprint: str,
    decimal_width: int,
    gear: str,
) -> str:
    return fingerprint(
        {
            "code_version": CACHE_CODE_VERSION,
            "upstream": upstream_fingerprint,
            "parameters": parameter_fingerprint,
            "decimal_width": int(decimal_width),
            "gear": gear,
        }
    )


def make_ledger_record(
    stage: str,
    cache_key: str,
    upstream_fingerprint: Any,
    parameter_fingerprint: str,
    product: dict[str, Any] | None,
    status: str,
) -> dict[str, Any]:
    sample_sets: list[int] = []
    feature_ids: list[str] = []
    product_fingerprint = None
    if product is not None:
        feature_ids = [f["id"] for f in product.get("features", [])]
        product_fingerprint = product.get("product_fingerprint")
        if product.get("rows"):
            sample_sets = sorted({int(row["seq"]) for row in product["rows"]})
    return {
        "stage": stage,
        "status": status,
        "cache_key": cache_key,
        "upstream_fingerprint": upstream_fingerprint,
        "parameter_fingerprint": parameter_fingerprint,
        "decimal_width": None if product is None else product.get("decimal_width"),
        "gear": None if product is None else product.get("gear"),
        "product_fingerprint": product_fingerprint,
        "feature_ids": feature_ids,
        "sample_set": sample_sets,
        "sample_set_fingerprint": fingerprint(sample_sets),
    }


def canonical_ledger(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda row: (row.get("stage_rank", 0), row["stage"]))
