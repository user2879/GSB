"""Deterministic, order-independent weighted instance selection.

The selection key is ``(rule_version, route_id, client_key)``; retries reuse
the same key and only carry an incremented ``attempt`` number in the request
payload, so the chosen instance and the per-attempt timeout budget never change.

Hashing uses SHA-256 (not Python's randomized ``hash``) and instances are
ordered by their stable ``instance_id``; therefore results never depend on dict
or filesystem iteration order.
"""

import hashlib

from .config import BUCKET_SPAN, InstanceConfig


def stable_bucket(rule_version: int, route_id: str, client_key: str) -> int:
    """Map a request deterministically to an integer bucket in [0, 10000)."""
    namespace = "\x1f".join(["meshbench/v1", str(rule_version), route_id, client_key])
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % BUCKET_SPAN


def _eligible(instances):
    # Explicit stable ordering; zero-weight instances can never be selected.
    return sorted((inst for inst in instances if inst.weight > 0),
                  key=lambda inst: inst.instance_id)


def select_instance(instances, bucket: int) -> InstanceConfig:
    """Place ``bucket`` into cumulative-weight intervals and pick an instance.

    The cut points scale bucket space [0, 10000) onto total weight using exact
    integer floor division, which keeps the mapping fully reproducible.
    """
    ordered = _eligible(instances)
    if not ordered:
        raise ValueError("no eligible (positive-weight) instances")
    total_weight = sum(inst.weight for inst in ordered)
    # point is measured on the integer weight scale [0, total_weight).
    point = (bucket * total_weight) // BUCKET_SPAN
    cumulative = 0
    for inst in ordered:
        cumulative += inst.weight
        if point < cumulative:
            return inst
    # Defensive fallback (point < total_weight always holds mathematically).
    return ordered[-1]


def choose(rule_version: int, route_id: str, client_key: str, instances):
    """Convenience: hash the request and return the selected instance."""
    bucket = stable_bucket(rule_version, route_id, client_key)
    return select_instance(instances, bucket)
