"""Content-derived cache keys and dependency fingerprints."""
from __future__ import annotations

import hashlib

import numpy as np

from .config import canonical_json


def _sha(*parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            part = part.encode("utf-8")
        h.update(part)
        h.update(b"\x00")
    return h.hexdigest()


def band_content_fingerprint(bands: dict) -> str:
    parts = []
    for name in sorted(bands):
        arr = np.ascontiguousarray(bands[name])
        parts.append(name.encode())
        parts.append(str(arr.dtype).encode())
        parts.append(arr.tobytes())
    return _sha(*parts)


def tile_cache_key(bands_fp: str, band_set_version: str, weights_fp: str,
                   fp_params: str) -> str:
    return _sha(bands_fp, band_set_version, weights_fp, fp_params)


def fusion_fingerprint(key_a: str, key_b: str, prob_hash_a: str, prob_hash_b: str,
                       geometry: dict, fusion_version: str, fp_params: str) -> str:
    return _sha(key_a, key_b, prob_hash_a, prob_hash_b,
                canonical_json(geometry), fusion_version, fp_params)


def result_version(cache_key: str) -> str:
    return cache_key[:16]
