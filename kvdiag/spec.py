"""Keyspace declaration loading and strong validation (business code E_SPEC).

The declaration is a local JSON file describing shard partitioning
(high-byte ranges or stable hash buckets), per-shard sample files, index
rules and lease policy. Validation is all-or-nothing: any violation raises
SpecError naming the first offending shard id and field, and nothing is
partially loaded.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from .canon import content_digest
from .errors import SpecError

MODE_HIGH_BYTE = "high_byte"
MODE_HASH_BUCKET = "hash_bucket"
INDEX_KINDS = ("prefix", "range")


@dataclass(frozen=True)
class LeaseSpec:
    ttl_ms: int
    renew_interval_ms: int
    max_inflight_renewals: int


@dataclass(frozen=True)
class IndexRule:
    column: str
    kind: str  # "prefix" | "range"
    prefix_len: int = 0  # prefix kind: prefix length
    width: int = 0       # range kind: bucket width on numeric column

    @property
    def name(self):
        return "%s:%s" % (self.kind, self.column)


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    lo: int            # high_byte mode: inclusive lower bound
    hi: int            # high_byte mode: inclusive upper bound
    bucket: int        # hash_bucket mode: bucket id, -1 otherwise
    sample_file: str
    index_rules: tuple
    lease: LeaseSpec
    tolerance: int

    def contains_key(self, key, buckets=0):
        if self.bucket >= 0:
            return bucket_of(key, buckets) == self.bucket
        return self.lo <= key <= self.hi


@dataclass(frozen=True)
class Spec:
    format_version: int
    seed: int
    mode: str
    buckets: int
    shards: tuple
    base_dir: str
    digest: str

    def shard_ids(self):
        return sorted(s.shard_id for s in self.shards)

    def shard(self, shard_id):
        for s in self.shards:
            if s.shard_id == shard_id:
                return s
        raise KeyError(shard_id)


def bucket_of(key, buckets):
    """Stable hash bucket for an integer key (FNV-1a over decimal text)."""
    h = 1469598103934665603
    for ch in str(key).encode("ascii"):
        h ^= ch
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h % buckets


def _fail(message, *, shard_id=None, field_name=None, **details):
    raise SpecError(message, shard_id=shard_id, field=field_name, details=details)


def _require_int(value, *, shard_id, field_name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("field must be an integer", shard_id=shard_id, field_name=field_name, value=value)
    if minimum is not None and value < minimum:
        _fail("field below minimum", shard_id=shard_id, field_name=field_name,
              value=value, minimum=minimum)
    if maximum is not None and value > maximum:
        _fail("field above maximum", shard_id=shard_id, field_name=field_name,
              value=value, maximum=maximum)
    return value


def _parse_lease(raw, shard_id):
    if not isinstance(raw, dict):
        _fail("lease must be an object", shard_id=shard_id, field_name="lease")
    ttl = _require_int(raw.get("ttl_ms"), shard_id=shard_id,
                       field_name="lease.ttl_ms", minimum=1)
    interval = _require_int(raw.get("renew_interval_ms"), shard_id=shard_id,
                            field_name="lease.renew_interval_ms", minimum=1)
    if interval >= ttl:
        _fail("renew_interval_ms must be smaller than ttl_ms", shard_id=shard_id,
              field_name="lease.renew_interval_ms", ttl_ms=ttl)
    max_inflight = _require_int(raw.get("max_inflight_renewals"), shard_id=shard_id,
                                field_name="lease.max_inflight_renewals", minimum=0)
    return LeaseSpec(ttl_ms=ttl, renew_interval_ms=interval,
                     max_inflight_renewals=max_inflight)


def _parse_index_rule(raw, shard_id):
    if not isinstance(raw, dict):
        _fail("index rule must be an object", shard_id=shard_id, field_name="index_rules")
    column = raw.get("column")
    if not isinstance(column, str) or not column:
        _fail("index rule column must be a non-empty string", shard_id=shard_id,
              field_name="index_rules.column")
    kind = raw.get("kind")
    if kind not in INDEX_KINDS:
        _fail("index rule kind must be one of %s" % (INDEX_KINDS,), shard_id=shard_id,
              field_name="index_rules.kind", value=kind)
    if kind == "prefix":
        prefix_len = _require_int(raw.get("prefix_len"), shard_id=shard_id,
                                  field_name="index_rules.prefix_len", minimum=1)
        return IndexRule(column=column, kind=kind, prefix_len=prefix_len)
    width = _require_int(raw.get("width"), shard_id=shard_id,
                         field_name="index_rules.width", minimum=1)
    return IndexRule(column=column, kind=kind, width=width)


def load_spec(path):
    """Load and fully validate a keyspace declaration. Raises SpecError."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("declaration is not readable JSON: %s" % exc, field_name="<file>")
    if not isinstance(raw, dict):
        _fail("declaration root must be an object", field_name="<root>")

    format_version = _require_int(raw.get("format_version"), shard_id=None,
                                  field_name="format_version", minimum=1)
    seed = _require_int(raw.get("seed"), shard_id=None, field_name="seed", minimum=0)
    mode = raw.get("mode")
    if mode not in (MODE_HIGH_BYTE, MODE_HASH_BUCKET):
        _fail("mode must be high_byte or hash_bucket", field_name="mode", value=mode)
    buckets = 0
    if mode == MODE_HASH_BUCKET:
        buckets = _require_int(raw.get("buckets"), shard_id=None,
                               field_name="buckets", minimum=1)
    raw_shards = raw.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        _fail("shards must be a non-empty array", field_name="shards")

    # Uniqueness of shard ids, in declaration order.
    seen = set()
    for entry in raw_shards:
        sid = entry.get("shard_id") if isinstance(entry, dict) else None
        if not isinstance(sid, str) or not sid:
            _fail("shard_id must be a non-empty string", field_name="shard_id")
        if sid in seen:
            _fail("duplicate shard_id", shard_id=sid, field_name="shard_id")
        seen.add(sid)

    shards = []
    for entry in raw_shards:
        sid = entry["shard_id"]
        lo, hi, bucket = -1, -1, -1
        if mode == MODE_HIGH_BYTE:
            rng = entry.get("range")
            if (not isinstance(rng, list)) or len(rng) != 2:
                _fail("range must be [lo, hi]", shard_id=sid, field_name="range")
            lo = _require_int(rng[0], shard_id=sid, field_name="range[0]", minimum=0)
            hi = _require_int(rng[1], shard_id=sid, field_name="range[1]", minimum=0)
            if lo > hi:
                _fail("range lo must not exceed hi", shard_id=sid, field_name="range")
        else:
            bucket = _require_int(entry.get("bucket"), shard_id=sid,
                                  field_name="bucket", minimum=0, maximum=buckets - 1)
        sample_file = entry.get("sample_file")
        if not isinstance(sample_file, str) or not sample_file:
            _fail("sample_file must be a non-empty string", shard_id=sid,
                  field_name="sample_file")
        raw_rules = entry.get("index_rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            _fail("index_rules must be a non-empty array", shard_id=sid,
                  field_name="index_rules")
        rules = tuple(_parse_index_rule(r, sid) for r in raw_rules)
        lease = _parse_lease(entry.get("lease"), sid)
        tolerance = _require_int(entry.get("tolerance"), shard_id=sid,
                                 field_name="tolerance", minimum=0)
        shards.append(ShardSpec(shard_id=sid, lo=lo, hi=hi, bucket=bucket,
                                sample_file=sample_file, index_rules=rules,
                                lease=lease, tolerance=tolerance))

    if mode == MODE_HIGH_BYTE:
        # Ranges must be contiguous and non-overlapping when sorted by lo.
        ordered = sorted(shards, key=lambda s: (s.lo, s.shard_id))
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.lo <= prev.hi:
                _fail("shard ranges overlap", shard_id=cur.shard_id, field_name="range",
                      prev_shard=prev.shard_id)
            if cur.lo != prev.hi + 1:
                _fail("shard ranges are not contiguous", shard_id=cur.shard_id,
                      field_name="range", prev_shard=prev.shard_id,
                      expected_lo=prev.hi + 1)
    else:
        used = sorted(s.bucket for s in shards)
        if used != list(range(buckets)):
            missing = sorted(set(range(buckets)) - set(used))
            dup = next((b for b in used if used.count(b) > 1), None)
            sid = next(s.shard_id for s in shards
                       if s.bucket == (dup if dup is not None else missing[0]))
            _fail("buckets must cover 0..buckets-1 exactly once", shard_id=sid,
                  field_name="bucket", missing=missing, duplicate=dup)

    spec = Spec(format_version=format_version, seed=seed, mode=mode, buckets=buckets,
                shards=tuple(shards), base_dir=str(path.parent),
                digest=content_digest(raw))
    return spec


def load_shard_rows(spec, shard):
    """Load and validate one shard sample file. Raises SpecError."""
    path = Path(spec.base_dir) / shard.sample_file
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("sample file is not readable JSON: %s" % exc,
              shard_id=shard.shard_id, field_name="sample_file")
    if not isinstance(raw, dict) or raw.get("shard_id") != shard.shard_id:
        _fail("sample shard_id mismatch", shard_id=shard.shard_id,
              field_name="shard_id", value=raw.get("shard_id") if isinstance(raw, dict) else None)
    rows = raw.get("rows")
    if not isinstance(rows, list):
        _fail("sample rows must be an array", shard_id=shard.shard_id, field_name="rows")
    seen_keys = set()
    for row in rows:
        if not isinstance(row, dict):
            _fail("sample row must be an object", shard_id=shard.shard_id, field_name="rows")
        key = row.get("key")
        _require_int(key, shard_id=shard.shard_id, field_name="rows.key")
        if key in seen_keys:
            _fail("duplicate sample key", shard_id=shard.shard_id,
                  field_name="rows.key", key=key)
        seen_keys.add(key)
        if not shard.contains_key(key, spec.buckets):
            _fail("sample key outside shard range/bucket", shard_id=shard.shard_id,
                  field_name="rows.key", key=key)
        for rule in shard.index_rules:
            if rule.column not in row:
                _fail("index rule references missing column", shard_id=shard.shard_id,
                      field_name="index_rules.column", column=rule.column, key=key)
            value = row[rule.column]
            if rule.kind == "prefix" and not isinstance(value, str):
                _fail("prefix index column must be a string", shard_id=shard.shard_id,
                      field_name="rows.%s" % rule.column, key=key)
            if rule.kind == "range" and (isinstance(value, bool) or not isinstance(value, int)):
                _fail("range index column must be an integer", shard_id=shard.shard_id,
                      field_name="rows.%s" % rule.column, key=key)
    return rows


def load_all_rows(spec):
    """Load every shard sample; any failure aborts before returning anything."""
    return {s.shard_id: load_shard_rows(spec, s) for s in spec.shards}
