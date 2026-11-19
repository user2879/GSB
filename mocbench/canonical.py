import hashlib
import json

import numpy as np


def _canonical(value):
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def canonical_json(value):
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def version_digest(raw_bundle):
    skeleton = raw_bundle.get("skeleton")
    curves = raw_bundle.get("curves")
    constraints = raw_bundle.get("constraints")
    params = raw_bundle.get("params")
    payload = {
        "skeleton": skeleton,
        "curves": curves,
        "constraints": constraints,
        "params": params,
    }
    return sha256_text(canonical_json(payload))
