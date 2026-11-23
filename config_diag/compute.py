from __future__ import annotations

import sqlite3
from typing import Any, Mapping

from .canonical import canonical_json, digest_value
from .graph import topological_order
from .model import ConfigSpec

FORMAT_VERSION = 1


def trunc_div(numerator: int, denominator: int) -> int:
    """Integer division truncated toward zero, independent of Python flooring."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


def _record_digest_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "key": record["key"],
        "raw_value": record["raw_value"],
        "value": record["value"],
        "version": record["version"],
        "refs": list(record["refs"]),
        "formula": record["formula"],
        "ttl_ms": record["ttl_ms"],
        "expires_mono_ns": record["expires_mono_ns"],
        "status": record["status"],
        "basis": record["basis"],
    }


def seal_record(record: dict[str, Any]) -> dict[str, Any]:
    record["refs"] = list(sorted(record["refs"]))
    record["content_digest"] = digest_value(_record_digest_payload(record))
    return record


def record_payload(record: Mapping[str, Any], include_digest: bool = True) -> dict[str, Any]:
    payload = _record_digest_payload(record)
    if include_digest:
        payload["content_digest"] = record["content_digest"]
    return payload


def record_bytes(record: Mapping[str, Any]) -> bytes:
    return canonical_json(record_payload(record)).encode("utf-8")


def _lookup_value(records: Mapping[str, Mapping[str, Any]], ref: str) -> int | None:
    target = records.get(ref)
    if target is None or target["status"] != "active" or not isinstance(target["value"], int):
        return None
    return int(target["value"])


def derive_record(record: dict[str, Any], records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if record["status"] == "expired":
        record["value"] = None
        record["basis"] = {"op": "expired", "reason": "ttl_elapsed"}
        return seal_record(record)

    formula = record["formula"]
    if formula["op"] == "identity":
        record["value"] = int(record["raw_value"])
        record["status"] = "active"
        record["basis"] = {"op": "identity", "input": record["raw_value"], "result": record["value"]}
        return seal_record(record)

    terms: list[dict[str, Any]] = []
    missing: list[str] = []
    for term in sorted(formula["terms"], key=lambda item: item["ref"]):
        ref = term["ref"]
        value = _lookup_value(records, ref)
        if value is None:
            missing.append(ref)
            continue
        terms.append(
            {
                "ref": ref,
                "value": value,
                "weight": term["weight"],
                "product": value * int(term["weight"]),
            }
        )
    if missing:
        record["value"] = None
        record["status"] = "invalid"
        record["basis"] = {
            "op": "invalid",
            "reason": "unavailable_dependency",
            "missing": sorted(missing),
            "formula": formula,
        }
        return seal_record(record)

    if formula["op"] == "linear":
        aggregate = sum(item["product"] for item in terms)
        aggregate_name = "sum"
    elif formula["op"] == "product":
        aggregate = 1
        for item in terms:
            aggregate *= item["product"]
        aggregate_name = "product"
    else:  # structural validation should make this unreachable
        raise ValueError(f"unsupported formula op: {formula['op']}")

    result = trunc_div(aggregate, int(formula["divisor"])) + int(formula["bias"])
    record["value"] = result
    record["status"] = "active"
    record["basis"] = {
        "op": formula["op"],
        "terms": terms,
        aggregate_name: aggregate,
        "divisor": formula["divisor"],
        "bias": formula["bias"],
        "truncation": "toward_zero",
        "result": result,
    }
    return seal_record(record)


def compute_all(records: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Deterministically recompute every record from its declared formula."""

    refs_by_key = {key: tuple(record["refs"]) for key, record in records.items()}
    order = topological_order(refs_by_key)
    working: dict[str, dict[str, Any]] = {}
    for key in order:
        record = dict(records[key])
        record["refs"] = list(record["refs"])
        record["formula"] = dict(record["formula"])
        working[key] = derive_record(record, working)
    return working


def build_initial_records(spec: ConfigSpec, now_mono_ns: int) -> dict[str, dict[str, Any]]:
    base: dict[str, dict[str, Any]] = {}
    for item in spec.keys:
        base[item.key] = {
            "key": item.key,
            "raw_value": item.value,
            "value": None,
            "version": item.version,
            "refs": list(item.refs),
            "formula": item.formula.to_obj(),
            "ttl_ms": item.ttl_ms,
            "expires_mono_ns": now_mono_ns + item.ttl_ms * 1_000_000,
            "status": "active",
            "basis": {},
            "content_digest": "",
        }
    return compute_all(base)


def meta_digest_payload(meta: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "format_version": meta["format_version"],
        "ledger_version": meta["ledger_version"],
        "config_set_version": meta["config_set_version"],
        "derivation_version": meta["derivation_version"],
        "change_cursor": meta["change_cursor"],
        "keys": [
            record_payload(records[key], include_digest=True)
            for key in sorted(records)
        ],
    }


def build_ledger_digest(meta: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]) -> str:
    return digest_value(meta_digest_payload(meta, records))


def verify_record_digest(record: Mapping[str, Any]) -> bool:
    expected = digest_value(_record_digest_payload(record))
    return expected == record.get("content_digest")


def build_view(
    records: Mapping[str, Mapping[str, Any]],
    view_version: int,
    config_set_version: int,
    derivation_version: int,
    created_mono_ns: int,
    previous_view: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active = [records[key] for key in sorted(records) if records[key]["status"] == "active"]
    invalid = [records[key] for key in sorted(records) if records[key]["status"] == "invalid"]
    expired = [records[key] for key in sorted(records) if records[key]["status"] == "expired"]
    active_keys = {record["key"] for record in active}
    previous_active = set()
    if previous_view:
        previous_active = {item["key"] for item in previous_view.get("active", [])}
    added = sorted(active_keys - previous_active)
    deleted = sorted(previous_active - active_keys)
    invalidated = sorted({record["key"] for record in invalid + expired})
    payload: dict[str, Any] = {
        "view_version": view_version,
        "config_set_version": config_set_version,
        "derivation_version": derivation_version,
        "created_mono_ns": created_mono_ns,
        "active": [
            {
                "key": record["key"],
                "value": record["value"],
                "version": record["version"],
                "content_digest": record["content_digest"],
                "basis": record["basis"],
            }
            for record in active
        ],
        "invalid": [
            {
                "key": record["key"],
                "version": record["version"],
                "basis": record["basis"],
                "content_digest": record["content_digest"],
            }
            for record in invalid
        ],
        "expired": [
            {
                "key": record["key"],
                "version": record["version"],
                "basis": record["basis"],
                "content_digest": record["content_digest"],
            }
            for record in expired
        ],
        "added": added,
        "deleted": deleted,
        "invalidated": invalidated,
    }
    payload["view_digest"] = digest_value(payload)
    return payload


def snapshot_payload(meta: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], view: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "meta": {
            "format_version": meta["format_version"],
            "ledger_version": meta["ledger_version"],
            "config_set_version": meta["config_set_version"],
            "derivation_version": meta["derivation_version"],
            "change_cursor": meta["change_cursor"],
            "content_digest": meta["content_digest"],
            "initialized_mono_ns": meta["initialized_mono_ns"],
            "last_mutation_mono_ns": meta["last_mutation_mono_ns"],
            "initial_spec_json": meta["initial_spec_json"],
            "initial_clients_json": meta["initial_clients_json"],
        },
        "records": [record_payload(records[key], include_digest=True) for key in sorted(records)],
        "view": dict(view),
    }


def record_from_sql_row(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    import json

    return {
        "key": row["key"],
        "raw_value": row["raw_value"],
        "value": row["value"],
        "version": row["version"],
        "refs": json.loads(row["refs_json"]),
        "formula": json.loads(row["formula_json"]),
        "ttl_ms": row["ttl_ms"],
        "expires_mono_ns": row["expires_mono_ns"],
        "status": row["status"],
        "basis": json.loads(row["basis_json"]),
        "content_digest": row["content_digest"],
    }


def record_to_sql_params(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "key": record["key"],
        "raw_value": record["raw_value"],
        "value": record["value"],
        "version": record["version"],
        "refs_json": canonical_json(list(record["refs"])),
        "formula_json": canonical_json(record["formula"]),
        "ttl_ms": record["ttl_ms"],
        "expires_mono_ns": record["expires_mono_ns"],
        "status": record["status"],
        "basis_json": canonical_json(record["basis"]),
        "content_digest": record["content_digest"],
    }
