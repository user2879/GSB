"""Content fingerprints, cache keys and dependency fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .models import Cell, MergeConfig


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_dumps(payload) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def config_fingerprint(config: MergeConfig) -> str:
    payload = config.to_dict()
    # Execution-only knobs must not invalidate data/rule-derived cache keys.
    payload.pop("max_workers", None)
    payload.pop("batch_size", None)
    return sha256_text(stable_json_dumps(payload))


def metadata_fingerprint(metadata: dict) -> str:
    payload = {
        feature: {
            "semantic_type": meta.semantic_type,
            "allowed_precisions": list(meta.allowed_precisions),
        }
        for feature, meta in sorted(metadata.items())
    }
    return sha256_text(stable_json_dumps(payload))


def source_fingerprint(version: str, path: str | Path) -> str:
    """File content fingerprint prefixed with the parser/format version.

    The version prefix makes a format change (e.g. fixed-width layout bump)
    distinguishable even when raw bytes hash the same, and embedding the file
    hash detects replacement with different content.
    """
    return f"{version}:{sha256_file(path)}"


def cell_cache_key(cell: Cell, config_hash: str) -> str:
    """Per (entity, feature) cache key.

    Derived from source bytes/raw text, parser version and config fingerprint,
    as required by the incremental invalidation contract.
    """
    if cell.raw_bytes is not None:
        source_token = "bytes:" + sha256_bytes(cell.raw_bytes)
    elif cell.raw_text is not None:
        source_token = "text:" + sha256_text(cell.raw_text)
    else:
        source_token = "value:" + repr(float(cell.value))
    payload = {
        "entity_id": cell.entity_id,
        "feature": cell.feature,
        "source_version": cell.source_version,
        "source_token": source_token,
        "missing": cell.missing,
        "precision_label": cell.precision_label,
        "partition_id": cell.partition_id,
        "config_hash": config_hash,
    }
    return sha256_text(stable_json_dumps(payload))


def dependency_fingerprint(cell_keys: Iterable[str], rule_fingerprint: str) -> str:
    """Merge-result fingerprint derived from participating cache keys + rules."""
    material = "\n".join(sorted(cell_keys))
    return sha256_text(rule_fingerprint + "\n" + material)


def rule_fingerprint(config: MergeConfig) -> str:
    payload = {
        "conflict_priority": config.conflict_priority,
        "source_order": config.source_order,
        "precision_rank": config.precision_rank,
        "numeric_tolerance": config.numeric_tolerance,
        "scaling_method": config.scaling_method,
        "missing_fill": config.missing_fill,
        "percentile_tie_rule": config.percentile_tie_rule,
    }
    return sha256_text(stable_json_dumps(payload))
