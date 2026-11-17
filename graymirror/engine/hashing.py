"""Deterministic stable hashing (order independent, salt free, stdlib only).

Python's built in ``hash`` is salted per process and dict iteration order is
irrelevant here: every decision derives from an explicit SHA-256 digest of a
domain separated key string, so the same client/request key, rule id and scale
factor always produce the same bucket on every machine and run.
"""
from __future__ import annotations

import hashlib

from .fixedpoint import GRID_BASIS

ROUTE_DOMAIN = "route"
MIRROR_DOMAIN = "mirror"


def stable_bucket(domain: str, rule_id: str, key: str) -> int:
    """Map a domain/rule/key triple to an integer bucket in ``0..G`` inclusive."""
    material = f"{domain}|{rule_id}|{key}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return integer % (GRID_BASIS + 1)
