"""Content fingerprints and hierarchical cache keys.

Fingerprints are derived only from *content*, never from wall-clock time or
object identity, so the same logical input always yields the same key.  BLAKE2b
is used (256-bit digest, hex encoded) because it is in the Python standard
library, deterministic and free of platform-dependent OpenSSL behaviour.

Cache keys form an explicit dependency graph::

    segment_fingerprint            (raw bytes + timing fields)
    timeline_fingerprint           (ordered set of accepted segments)
    param_fingerprint              (frame/window geometry + thresholds)
    quant_fingerprint              (frac_bits + quant_mode)
    region_key  = f(timeline, params)
    window_feature_key = f(window segment set, segment fps, params, quant)
    posterior_key = f(feature_key, projection, quant)
    endpoint_key  = f(posterior keys covering frames, threshold, quant)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

DIGEST_BYTES = 32


def canonical(obj: Any) -> bytes:
    """Serialize any JSON-able object to a canonical, key-sorted byte string."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(obj: Any) -> str:
    """Return the BLAKE2b hex digest of the canonical form of ``obj``."""
    return hashlib.blake2b(canonical(obj), digest_size=DIGEST_BYTES).hexdigest()


def digest_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=DIGEST_BYTES).hexdigest()


def combine(parts: Iterable[Any]) -> str:
    """Fingerprint over the ordered list of already-serializable parts."""
    return digest([p for p in parts])


def fingerprint_int_sequence(values: Sequence[int]) -> str:
    return digest(["ints", [int(v) for v in values]])


def fingerprint_str_set(values: Iterable[str]) -> str:
    # Sets are unordered; sort to make the fingerprint order independent.
    return digest(["strset", sorted(str(v) for v in values)])


def make_cache_key(kind: str, upstream: Iterable[str], **material: Any) -> str:
    """Build a namespaced cache key.

    ``kind`` names the producing stage, ``upstream`` lists direct dependency
    fingerprints (ordered) and ``material`` captures the parameters that must
    force a recompute when changed.
    """
    payload = {
        "kind": kind,
        "upstream": [str(u) for u in upstream],
        "material": material,
    }
    return kind + ":" + digest(payload)


def short(key: str, length: int = 16) -> str:
    """Truncate a key for human-readable reports while staying unambiguous."""
    tail = key.split(":", 1)[-1]
    return tail[:length]
