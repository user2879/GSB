"""Byte-stable canonical serialization and content digests.

Conclusions never embed wall-clock timestamps or host-specific values.
Dictionaries are always emitted with sorted keys and without whitespace, so
identical logical content is byte identical regardless of insertion or
dictionary iteration order.
"""

import hashlib
import json


def canonical_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_text(value):
    return canonical_bytes(value).decode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
