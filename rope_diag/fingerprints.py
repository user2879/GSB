"""Deterministic content fingerprints used for cache keys and audit records."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Serialize with canonical key ordering so hashes are reproducible."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: Any) -> str:
    """SHA-256 over the canonical JSON form, truncated to 24 hex chars."""
    digest = hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
    return digest[:24]


def sample_fingerprint(sample: dict) -> str:
    """Fingerprint of one attention scoring sample's full content."""
    return stable_hash(
        {
            "sample_id": sample["sample_id"],
            "query_pos": sample["query_pos"],
            "key_positions": list(sample["key_positions"]),
            "query": [repr(float(v)) for v in sample["query"]],
            "keys": [[repr(float(v)) for v in row] for row in sample["keys"]],
        }
    )


def scheme_param_fingerprint(params: dict) -> str:
    """Fingerprint of the scheme-relevant parameter subset of a config."""
    return stable_hash(
        {
            "scheme": params["scheme"],
            "base": repr(float(params["base"])),
            "scale": repr(float(params["scale"])),
            "dim": int(params["dim"]),
            "train_len": int(params["train_len"]),
            "target_len": int(params["target_len"]),
        }
    )


def pruning_fingerprint(policy: dict) -> str:
    """Fingerprint of the active pruning policy (the pruning spec)."""
    return stable_hash(
        {
            "policy_id": policy["policy_id"],
            "dim_range": list(policy["dim_range"]),
            "score_chunk": int(policy["score_chunk"]),
        }
    )


def cache_key(
    scheme_params: dict,
    sample_fp: str,
    precision: str,
    pruning_fp: str,
    stage: str,
) -> str:
    """Cache key derived from scheme params, sample content, precision, pruning."""
    return stable_hash(
        {
            "scheme_params": scheme_param_fingerprint(scheme_params),
            "sample": sample_fp,
            "precision": precision,
            "pruning": pruning_fp,
            "stage": stage,
        }
    )
