from __future__ import annotations

from typing import Any

from .codes import E_RECEIPT_VERSION, E_WORKER_INVALID
from .model import MISSING, CanonicalReceipt, Failure

VERSIONS = ("r1", "r2", "r3")


def _value(obj: dict[str, Any], key: str) -> Any:
    if key not in obj or obj[key] is None:
        return MISSING
    if obj[key] == MISSING:
        return MISSING
    return obj[key]


def _require_str(obj: dict[str, Any], key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_non_negative_int(value: Any, key: str) -> Any:
    if value is MISSING:
        return MISSING
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def parse_r1(obj: dict[str, Any]) -> CanonicalReceipt:
    return CanonicalReceipt(
        source_version="r1",
        batch_id=_require_str(obj, "batch_id"),
        request_id=_require_str(obj, "request_id"),
        output_digest=_require_str(obj, "output_digest"),
    )


def parse_r2(obj: dict[str, Any]) -> CanonicalReceipt:
    base = parse_r1(obj)
    return CanonicalReceipt(
        source_version="r2",
        batch_id=base.batch_id,
        request_id=base.request_id,
        output_digest=base.output_digest,
        param_version=_optional_non_negative_int(_value(obj, "param_version"), "param_version"),
        elapsed_ms=_optional_non_negative_int(_value(obj, "elapsed_ms"), "elapsed_ms"),
        worker_id=_value(obj, "worker_id"),
    )


def parse_r3(obj: dict[str, Any]) -> CanonicalReceipt:
    base = parse_r2(obj)
    rank = _optional_non_negative_int(_value(obj, "rank_in_batch"), "rank_in_batch")
    supersedes = _value(obj, "supersedes_ref")
    param_digest = _value(obj, "param_digest")
    if supersedes is not MISSING and (not isinstance(supersedes, str) or not supersedes):
        raise ValueError("supersedes_ref must be a non-empty string or missing")
    if param_digest is not MISSING and (not isinstance(param_digest, str) or not param_digest):
        raise ValueError("param_digest must be a non-empty string or missing")
    return CanonicalReceipt(
        source_version="r3",
        batch_id=base.batch_id,
        request_id=base.request_id,
        output_digest=base.output_digest,
        param_version=base.param_version,
        elapsed_ms=base.elapsed_ms,
        worker_id=base.worker_id,
        rank_in_batch=rank,
        supersedes_ref=supersedes,
        param_digest=param_digest,
    )


def parse_receipt(obj: dict[str, Any], expected_version: str | None = None) -> CanonicalReceipt:
    version = obj.get("v", obj.get("version", expected_version))
    if expected_version is not None and version != expected_version:
        raise ValueError(f"expected {expected_version}, found {version!r}")
    if version == "r1":
        return parse_r1(obj)
    if version == "r2":
        return parse_r2(obj)
    if version == "r3":
        return parse_r3(obj)
    raise ValueError(f"unsupported receipt version {version!r}")


def parse_and_validate_receipt(
    raw: dict[str, Any],
    *,
    known_worker_ids: set[str],
    source: str,
    record_index: int,
) -> tuple[CanonicalReceipt | None, Failure | None]:
    obj = {key: value for key, value in raw.items() if not key.startswith("_")}
    version = raw.get("v", raw.get("version"))
    if version not in VERSIONS:
        return None, Failure(
            E_RECEIPT_VERSION,
            "receipt version code must be r1, r2, or r3",
            source,
            record_index,
            {"v": version},
        )
    try:
        receipt = parse_receipt(obj)
    except (ValueError, TypeError) as exc:
        return None, Failure(
            "E_RECEIPT_SCHEMA",
            str(exc),
            source,
            record_index,
            {"v": version},
        )

    if receipt.worker_id is not MISSING and receipt.worker_id not in known_worker_ids:
        return None, Failure(
            E_WORKER_INVALID,
            "receipt references an unknown worker_id",
            source,
            record_index,
            {"worker_id": receipt.worker_id, "receipt_ref": receipt.receipt_ref},
        )
    return receipt, None


def receipt_to_version_dict(receipt: CanonicalReceipt, version: str, *, include_missing: bool = True) -> dict[str, Any]:
    if version not in VERSIONS:
        raise ValueError(f"unsupported target version {version}")
    data: dict[str, Any] = {"v": version}
    for key in ("batch_id", "request_id", "output_digest"):
        data[key] = getattr(receipt, key)

    if version in ("r2", "r3"):
        for key in ("param_version", "elapsed_ms", "worker_id"):
            value = getattr(receipt, key)
            if value is MISSING:
                if include_missing:
                    data[key] = MISSING
            else:
                data[key] = value

    if version == "r3":
        for key in ("rank_in_batch", "supersedes_ref", "param_digest"):
            value = getattr(receipt, key)
            if value is MISSING:
                if include_missing:
                    data[key] = MISSING
            else:
                data[key] = value
    return data
