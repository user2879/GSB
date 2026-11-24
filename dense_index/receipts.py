from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .canonical import canonical_dumps, fingerprint
from .errors import E_RECEIPT_INVALID, DiagnosticError, Failure


VERSION_ALIASES = {
    1: "v1",
    "1": "v1",
    "v1": "v1",
    "receipt/v1": "v1",
    2: "v2",
    "2": "v2",
    "v2": "v2",
    "receipt/v2": "v2",
    3: "v3",
    "3": "v3",
    "v3": "v3",
    "receipt/v3": "v3",
}

VERSION_FIELDS = {
    "v1": ["bucket_id", "hit_rate_permille"],
    "v2": ["bucket_id", "hit_rate_permille", "param_version", "elapsed_ms"],
    "v3": [
        "bucket_id",
        "hit_rate_permille",
        "param_version",
        "elapsed_ms",
        "chunk_digest",
        "overridden_duplicate_refs",
        "projection_params",
    ],
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def detect_version(obj: Dict[str, Any], forced: Optional[str] = None) -> Optional[str]:
    if forced:
        return VERSION_ALIASES.get(forced, forced if forced in {"v1", "v2", "v3"} else None)
    for key in ("format", "receipt_version", "schema_version", "version"):
        if key in obj:
            return VERSION_ALIASES.get(obj[key])
    return None


def normalize_receipt(
    obj: Any,
    source: str,
    index: int = 0,
    forced_version: Optional[str] = None,
) -> Dict[str, Any]:
    failures: List[Failure] = []
    if not isinstance(obj, dict):
        raise DiagnosticError(
            E_RECEIPT_INVALID,
            "receipt must be an object",
            [Failure(E_RECEIPT_INVALID, "receipt is not an object", path=source, line=index)],
        )

    version = detect_version(obj, forced_version)
    if version not in VERSION_FIELDS:
        failure = Failure(E_RECEIPT_INVALID, "receipt version is missing or unsupported", path=source, line=index, field="version")
        raise DiagnosticError(E_RECEIPT_INVALID, "unsupported receipt version", [failure])

    bucket_id = obj.get("bucket_id", obj.get("bucket"))
    hit_rate = obj.get("hit_rate_permille", obj.get("hit_rate_per_mille", obj.get("hit_rate_per_thousand")))

    if not isinstance(bucket_id, str) or not bucket_id:
        failures.append(Failure(E_RECEIPT_INVALID, "bucket_id must be a non-empty string", path=source, line=index, field="bucket_id"))
    if not _is_int(hit_rate) or not (0 <= int(hit_rate) <= 1000):
        failures.append(Failure(E_RECEIPT_INVALID, "hit_rate_permille must be an integer in [0, 1000]", path=source, line=index, field="hit_rate_permille"))

    param_version = obj.get("param_version")
    elapsed_ms = obj.get("elapsed_ms")
    chunk_digest = obj.get("chunk_digest")
    duplicate_refs = obj.get("overridden_duplicate_refs")
    projection_params = obj.get("projection_params")

    if version in {"v2", "v3"}:
        if not _is_int(param_version) or param_version < 0:
            failures.append(Failure(E_RECEIPT_INVALID, "param_version must be a non-negative integer", path=source, line=index, field="param_version"))
        if not _is_int(elapsed_ms) or elapsed_ms < 0:
            failures.append(Failure(E_RECEIPT_INVALID, "elapsed_ms must be a non-negative integer", path=source, line=index, field="elapsed_ms"))

    if version == "v3":
        if not isinstance(chunk_digest, str) or not chunk_digest:
            failures.append(Failure(E_RECEIPT_INVALID, "chunk_digest must be a non-empty string", path=source, line=index, field="chunk_digest"))
        if not isinstance(duplicate_refs, list) or not all(isinstance(item, str) for item in duplicate_refs):
            failures.append(Failure(E_RECEIPT_INVALID, "overridden_duplicate_refs must be a string list", path=source, line=index, field="overridden_duplicate_refs"))
        if not isinstance(projection_params, dict) or not _is_int(projection_params.get("projection_dim")) or projection_params["projection_dim"] <= 0:
            failures.append(Failure(E_RECEIPT_INVALID, "projection_params.projection_dim must be a positive integer", path=source, line=index, field="projection_params"))

    if failures:
        raise DiagnosticError(E_RECEIPT_INVALID, f"{len(failures)} receipt error(s)", failures)

    if version == "v1":
        missing = [
            "param_version",
            "elapsed_ms",
            "chunk_digest",
            "overridden_duplicate_refs",
            "projection_params",
        ]
    elif version == "v2":
        missing = ["chunk_digest", "overridden_duplicate_refs", "projection_params"]
    else:
        missing = []

    logical_material = {"bucket_id": bucket_id, "hit_rate_permille": int(hit_rate)}
    logical_id = fingerprint(logical_material, 20)
    return {
        "entry_id": f"LED-{logical_id[:16]}",
        "logical_id": logical_id,
        "bucket_id": bucket_id,
        "hit_rate_permille": int(hit_rate),
        "hit_rate": int(hit_rate) / 1000.0,
        "param_version": int(param_version) if _is_int(param_version) else None,
        "elapsed_ms": int(elapsed_ms) if _is_int(elapsed_ms) else None,
        "chunk_digest": chunk_digest if isinstance(chunk_digest, str) else None,
        "overridden_duplicate_refs": list(duplicate_refs) if isinstance(duplicate_refs, list) else [],
        "projection_params": dict(projection_params) if isinstance(projection_params, dict) else None,
        "source_version": version,
        "missing_fields": missing,
        "source_file": source,
        "source_index": index,
        "authoritative": False,
        "authority_reason": None,
        "ledger_cache_key": None,
    }


def load_receipt_file(path: str | Path, forced_version: Optional[str] = None) -> List[Dict[str, Any]]:
    path = Path(path)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failure = Failure(E_RECEIPT_INVALID, f"cannot read receipt: {exc}", path=str(path))
        raise DiagnosticError(E_RECEIPT_INVALID, str(exc), [failure]) from exc
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        failure = Failure(E_RECEIPT_INVALID, f"truncated or invalid JSON: {exc.msg}", path=str(path), line=exc.lineno, details={"column": exc.colno})
        raise DiagnosticError(E_RECEIPT_INVALID, str(exc), [failure]) from exc

    file_version = forced_version
    if isinstance(raw, dict):
        file_version = detect_version(raw.get("format") and {"version": raw.get("format")} or raw, forced_version) or forced_version
        items = raw.get("receipts", raw.get("items"))
        if items is None and all(field in raw for field in ("bucket_id", "hit_rate_permille")):
            items = [raw]
    else:
        items = raw
    if not isinstance(items, list):
        failure = Failure(E_RECEIPT_INVALID, "receipt file must contain a receipts list", path=str(path))
        raise DiagnosticError(E_RECEIPT_INVALID, "invalid receipt envelope", [failure])

    detected = file_version or forced_version
    return [normalize_receipt(item, str(path), index, detected) for index, item in enumerate(items)]


def strict_receipt(entry: Dict[str, Any], target_version: str) -> Dict[str, Any]:
    if target_version == "v1":
        return {
            "bucket_id": entry["bucket_id"],
            "hit_rate_permille": entry["hit_rate_permille"],
        }
    if target_version == "v2":
        return {
            "bucket_id": entry["bucket_id"],
            "hit_rate_permille": entry["hit_rate_permille"],
            "param_version": entry.get("param_version") if entry.get("param_version") is not None else 0,
            "elapsed_ms": entry.get("elapsed_ms") if entry.get("elapsed_ms") is not None else 0,
        }
    projection = entry.get("projection_params") or {"projection_dim": 1, "synthetic_default": True}
    digest = entry.get("chunk_digest")
    if not digest:
        seed = hashlib.sha256(("missing-digest:" + entry["logical_id"]).encode("utf-8")).hexdigest()
        digest = "sha256:" + seed
    return {
        "bucket_id": entry["bucket_id"],
        "hit_rate_permille": entry["hit_rate_permille"],
        "param_version": entry.get("param_version") if entry.get("param_version") is not None else 0,
        "elapsed_ms": entry.get("elapsed_ms") if entry.get("elapsed_ms") is not None else 0,
        "chunk_digest": digest,
        "overridden_duplicate_refs": list(entry.get("overridden_duplicate_refs") or []),
        "projection_params": projection,
    }


def convert_with_sidecar(entry: Dict[str, Any], target_version: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    source_version = entry["source_version"]
    receipt = strict_receipt(entry, target_version)
    target_rank = int(target_version[1])
    source_rank = int(source_version[1])
    dropped: List[str] = []
    if target_rank < 3:
        dropped.extend(["chunk_digest", "overridden_duplicate_refs", "projection_params"])
    if target_rank < 2:
        dropped.extend(["param_version", "elapsed_ms"])
    dropped = [field for field in dropped if entry.get(field) not in (None, []) or field == "projection_params"]
    degradation = {
        "from": source_version,
        "to": target_version,
        "dropped_fields": dropped,
        "sidecar_fields": dropped,
        "rule": "lossless-restore-requires-sidecar",
    }
    sidecar = {
        "schema": "receipt-sidecar/v1",
        "entry_id": entry["entry_id"],
        "logical_id": entry["logical_id"],
        "target_version": target_version,
        "normalized_entry": json.loads(canonical_dumps(entry)),
        "degradations": [degradation] if dropped else [],
    }
    return receipt, sidecar


def restore_from_sidecar(receipt: Dict[str, Any], sidecar: Dict[str, Any]) -> Dict[str, Any]:
    normalized = sidecar["normalized_entry"]
    if receipt.get("bucket_id") != normalized["bucket_id"] or receipt.get("hit_rate_permille") != normalized["hit_rate_permille"]:
        raise DiagnosticError(E_RECEIPT_INVALID, "sidecar does not match receipt logical fields")
    return normalized


def _digest_matches(entry: Dict[str, Any], digests: Optional[Dict[str, str]]) -> bool:
    if not digests or not entry.get("chunk_digest"):
        return False
    wanted = entry["chunk_digest"]
    candidates = set()
    bucket_digest = digests.get(entry["bucket_id"])
    if bucket_digest:
        candidates.add(bucket_digest)
    candidates.update(digests.get("blocks", {}).values() if isinstance(digests.get("blocks"), dict) else [])
    return wanted in candidates


def build_unified_ledger(
    receipts: List[Dict[str, Any]],
    current_digests: Optional[Dict[str, Any]] = None,
    quant_level: Optional[str] = None,
    param_fingerprint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in receipts:
        entry = dict(entry)
        entry["authoritative"] = False
        entry["authority_reason"] = None
        grouped.setdefault(entry["bucket_id"], []).append(entry)

    result: List[Dict[str, Any]] = []
    for bucket_id in sorted(grouped):
        group = grouped[bucket_id]
        matching_v3 = [item for item in group if item["source_version"] == "v3" and _digest_matches(item, current_digests)]
        if matching_v3:
            authority = sorted(matching_v3, key=lambda item: (int(item.get("param_version") or 0), item["entry_id"]))[-1]
            reason = "v3_content_digest_match"
        else:
            v3 = [item for item in group if item["source_version"] == "v3"]
            if v3:
                authority = sorted(v3, key=lambda item: (int(item.get("param_version") or 0), item["entry_id"]))[-1]
                reason = "v3_highest_version_stale_digest"
            else:
                authority = sorted(group, key=lambda item: int(item["source_version"][1]))[-1]
                reason = f"{authority['source_version']}_highest_available"
        for entry in sorted(group, key=lambda item: item["entry_id"]):
            if entry["entry_id"] == authority["entry_id"] and entry["source_file"] == authority.get("source_file") and entry["source_index"] == authority.get("source_index"):
                entry["authoritative"] = True
                entry["authority_reason"] = reason
            entry["ledger_cache_key"] = "ck_ledger_" + fingerprint(
                {
                    "entry_id": entry["entry_id"],
                    "bucket_id": entry["bucket_id"],
                    "upstream": current_digests.get("bucket_cache_keys", {}).get(bucket_id) if current_digests else None,
                    "param_fingerprint": param_fingerprint,
                    "quant_level": quant_level,
                    "source_version": entry["source_version"],
                },
                24,
            )
            result.append(entry)
    return sorted(result, key=lambda item: (item["bucket_id"], item["source_version"], item["entry_id"], item["source_index"]))
