"""Canonical serialization and content digests.

``canonical_json`` sorts object keys and uses fixed separators so that equal
logical content always yields byte-identical output, independent of dict
insertion or traversal order. All persisted digests are SHA-256 over this form.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def canonical_json(payload: Any) -> bytes:
    def default(obj: Any):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, (set, tuple)):
            return list(obj)
        raise TypeError(f"cannot canonically serialize {type(obj)!r}")

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=default,
    ).encode("utf-8")


def content_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def fingerprint_view(view) -> str:
    """Stable structural fingerprint used to detect definition edits."""
    return content_digest(
        {
            "view_id": view.view_id,
            "env_ids": list(view.env_ids),
            "policies": dict(sorted(view.policies.items())),
        }
    )


def fingerprint_environments(spec) -> str:
    envs = {}
    for env_id in sorted(spec.environments):
        env = spec.environments[env_id]
        envs[env_id] = {
            "weight": env.weight,
            "observations": sorted(
                (
                    {
                        "key": o.key,
                        "seq": o.seq,
                        "timestamp_ms": o.timestamp_ms,
                        "value": o.value,
                    }
                    for o in env.observations
                ),
                key=lambda row: (row["key"], row["seq"]),
            ),
        }
    return content_digest({"scale_exponent": spec.scale_exponent, "environments": envs})
