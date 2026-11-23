"""Content fingerprints and per-stage cache keys."""

import hashlib
import json


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def fingerprint(obj) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def combine(parts) -> str:
    return fingerprint([str(p) for p in parts])


def stage_key(stage: str, *parts) -> str:
    return f"{stage}:{combine(list(parts))}"
