"""Content fingerprints and stage cache keys.

Keys are derived from upstream content fingerprints, parameter fingerprints,
the quantization tier and the exchange version. The derivation is domain
separated by stage name and stable across processes and Python hash seeds.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def digest(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def fingerprint(obj: Any, domain: str = "content") -> str:
    payload = json.dumps(
        {"d": domain, "o": obj},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")
    return digest(payload)


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return sorted(value)
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return str(value)


def cache_key(stage: str, upstream: str, params: Any, quant: str,
              exchange_version: int) -> str:
    material = {
        "stage": stage,
        "upstream": upstream,
        "params": params,
        "quant": quant,
        "exchange_version": exchange_version,
    }
    return f"{stage}:{fingerprint(material, domain='cache-key')}"
