"""Deterministic prefix/range index construction.

Entries are produced by explicit sorting only; positions never depend on
dict or hash iteration order, so any input row order yields a byte-identical
entry set for the same shard sample.
"""

from .canon import canonical_json, content_digest, key_digest
from .errors import SpecError


def _rule_sort_key(rule, row):
    value = row[rule.column]
    if rule.kind == "prefix":
        return (value[: rule.prefix_len], row["key"])
    return (value // rule.width, row["key"])


def ensure_key_compatible(shard, key, buckets=0):
    """A key digest may only enter the index of a shard that owns the key."""
    if not shard.contains_key(key, buckets):
        raise SpecError(
            "key digest incompatible with shard range/bucket",
            shard_id=shard.shard_id, field="rows.key", details={"key": key},
        )


def build_index_entries(shard, rows, index_version, buckets=0):
    """Build the deterministic entry list for one shard.

    Returns a list of dicts: shard_id, rule, key_digest, position,
    index_version. Positions are assigned per rule after sorting.
    """
    entries = []
    for rule in shard.index_rules:
        ordered = sorted(rows, key=lambda row: _rule_sort_key(rule, row))
        for position, row in enumerate(ordered):
            key = row["key"]
            ensure_key_compatible(shard, key, buckets)
            entries.append({
                "shard_id": shard.shard_id,
                "rule": rule.name,
                "key_digest": key_digest(key),
                "position": position,
                "index_version": index_version,
            })
    entries.sort(key=lambda e: (e["rule"], e["position"], e["key_digest"]))
    return entries


def entries_digest(entries):
    return content_digest([dict(e) for e in entries])


def entries_bytes(entries):
    return canonical_json([dict(e) for e in entries])
