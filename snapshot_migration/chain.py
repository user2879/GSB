"""JSON loading and strong validation of snapshot generation chains.

Validation is purely in-memory: nothing is persisted until the whole chain
passes, therefore an invalid declaration can never leave a half-loaded store.
All failures raise :class:`ChainError` (``E_CHAIN``) with a precise location.
"""

import json
from typing import Any, Dict, List

from .errors import ChainError
from .model import Generation, KeyRange, Limits, SnapshotChain

VALID_ACTIONS = ("inherit", "rewrite", "discard", "new")


def load_chain(path) -> SnapshotChain:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise ChainError(f"chain file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ChainError(f"chain file is not valid JSON: {exc}")
    return parse_chain(raw)


def _require(condition, message, **location):
    if not condition:
        raise ChainError(message, **location)


def _is_nonneg_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_pos_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def parse_chain(raw: Dict[str, Any]) -> SnapshotChain:
    _require(isinstance(raw, dict), "chain root must be a JSON object")

    title = raw.get("title", "snapshot-chain")
    _require(isinstance(title, str), "title must be a string")

    limits_raw = raw.get("limits")
    _require(isinstance(limits_raw, dict), "limits must be an object")
    lower, upper = limits_raw.get("lower"), limits_raw.get("upper")
    _require(_is_pos_int(lower), "limits.lower must be a positive integer")
    _require(_is_pos_int(upper), "limits.upper must be a positive integer")
    _require(lower < upper, "limits.lower must be strictly less than limits.upper")
    limits = Limits(lower=lower, upper=upper)

    gens_raw = raw.get("generations")
    _require(isinstance(gens_raw, list) and gens_raw,
             "generations must be a non-empty array")

    parsed_gens: List[Generation] = []
    known_ids = set()
    # range_id -> (lower, upper); bounds must be stable across the whole chain
    range_bounds: Dict[str, tuple] = {}

    for gen_index, gen_raw in enumerate(gens_raw):
        _require(isinstance(gen_raw, dict), "each generation must be an object")
        gid = gen_raw.get("generation_id")
        _require(isinstance(gid, str) and gid, "generation_id must be a non-empty string",
                 generation_id=gid)
        _require(gid not in known_ids, f"duplicate generation_id: {gid}",
                 generation_id=gid)
        known_ids.add(gid)

        seq = gen_raw.get("seq")
        _require(_is_nonneg_int(seq), "seq must be a non-negative integer",
                 generation_id=gid)

        parent_id = gen_raw.get("parent_id")
        _require(parent_id is None or isinstance(parent_id, str),
                 "parent_id must be null or a string", generation_id=gid)
        if gen_index == 0:
            _require(parent_id is None, "root generation must have parent_id null",
                     generation_id=gid)
            _require(seq == 0, "root generation must have seq 0", generation_id=gid)
        else:
            _require(parent_id is not None,
                     "non-root generation must declare a parent_id",
                     generation_id=gid)

        ranges_raw = gen_raw.get("ranges")
        _require(isinstance(ranges_raw, list) and ranges_raw,
                 "ranges must be a non-empty array", generation_id=gid)

        rules_map: Dict[str, str] = {}
        for rule in gen_raw.get("rules", []):
            _require(isinstance(rule, dict) and "range_id" in rule and "action" in rule,
                     "each rule must be an object with range_id and action",
                     generation_id=gid)
            rid, action = rule["range_id"], rule["action"]
            _require(isinstance(rid, str) and rid, "rule.range_id must be a string",
                     generation_id=gid)
            _require(action in VALID_ACTIONS,
                     f"rule.action must be one of {VALID_ACTIONS}, got {action!r}",
                     generation_id=gid, range_id=rid)
            _require(rid not in rules_map, f"duplicate rule for range {rid}",
                     generation_id=gid, range_id=rid)
            rules_map[rid] = action

        ranges: List[KeyRange] = []
        seen_range_ids = set()
        for range_raw in ranges_raw:
            _require(isinstance(range_raw, dict), "each range must be an object",
                     generation_id=gid)
            rid = range_raw.get("range_id")
            _require(isinstance(rid, str) and rid, "range_id must be a non-empty string",
                     generation_id=gid)
            _require(rid not in seen_range_ids,
                     f"duplicate range_id {rid} in generation {gid}",
                     generation_id=gid, range_id=rid)
            seen_range_ids.add(rid)

            lo, hi = range_raw.get("lower"), range_raw.get("upper")
            _require(isinstance(lo, str) and isinstance(hi, str),
                     "range lower/upper must be strings",
                     generation_id=gid, range_id=rid)
            _require(lo < hi, "range lower must be strictly less than upper",
                     generation_id=gid, range_id=rid)
            if rid in range_bounds:
                _require(range_bounds[rid] == (lo, hi),
                         f"range {rid} bounds differ across generations; "
                         "range bounds are immutable",
                         generation_id=gid, range_id=rid)
            else:
                range_bounds[rid] = (lo, hi)

            keys = range_raw.get("keys", [])
            _require(isinstance(keys, list), "keys must be an array",
                     generation_id=gid, range_id=rid)
            for key in keys:
                _require(isinstance(key, str), "each key must be a string",
                         generation_id=gid, range_id=rid)
            tombstone = bool(range_raw.get("tombstone", False))
            if tombstone:
                _require(not keys, f"discarded (tombstone) range {rid} must have no keys",
                         generation_id=gid, range_id=rid)
            else:
                previous = None
                for key in keys:
                    _require(lo <= key < hi,
                             f"key {key!r} outside half-open bounds [{lo!r}, {hi!r})",
                             generation_id=gid, range_id=rid)
                    if previous is not None:
                        _require(previous < key,
                                 f"keys must be unique and strictly ordered; "
                                 f"{previous!r} then {key!r}",
                                 generation_id=gid, range_id=rid)
                    previous = key

            declared_action = rules_map.get(rid)
            if gen_index == 0:
                action = declared_action or "new"
                _require(action in ("new", "rewrite"),
                         "root ranges must use action new or rewrite",
                         generation_id=gid, range_id=rid)
            else:
                _require(declared_action is not None,
                         f"non-root generation must declare a rule for range {rid}",
                         generation_id=gid, range_id=rid)
                action = declared_action
            if action == "discard":
                _require(tombstone or not keys,
                         f"discarded range {rid} must be a tombstone with no keys",
                         generation_id=gid, range_id=rid)
                tombstone = True

            ranges.append(KeyRange(rid, lo, hi, tuple(keys), action, tombstone))

        # ranges tile the full (possibly partly tombstone) key space without gaps
        ordered = sorted(ranges, key=lambda item: item.lower)
        _require([item.range_id for item in ordered] ==
                 [item.range_id for item in ranges],
                 f"ranges in generation {gid} must be declared in lower-bound order",
                 generation_id=gid)
        for left, right in zip(ordered, ordered[1:]):
            _require(left.upper <= right.lower,
                     f"ranges {left.range_id} and {right.range_id} overlap or are "
                     "out of order", generation_id=gid, range_id=right.range_id)
            _require(left.upper == right.lower,
                     f"ranges {left.range_id} and {right.range_id} are not contiguous: "
                     f"upper bound {left.upper!r} must meet next lower bound "
                     f"{right.lower!r}", generation_id=gid, range_id=right.range_id)

        # rules may only reference declared ranges
        for rid in rules_map:
            _require(rid in seen_range_ids,
                     f"rule references undeclared range {rid}",
                     generation_id=gid, range_id=rid)

        parsed_gens.append(Generation(
            gid, seq, parent_id, tuple(ranges),
            tuple(sorted(rules_map.items()))))

    # parent references and acyclicity / sequence ordering
    ids = {gen.generation_id for gen in parsed_gens}
    for gen in parsed_gens:
        if gen.parent_id is not None:
            _require(gen.parent_id in ids,
                     f"parent_id {gen.parent_id!r} does not exist",
                     generation_id=gen.generation_id)
            parent = next(item for item in parsed_gens
                          if item.generation_id == gen.parent_id)
            _require(parent.seq < gen.seq,
                     "parent generation must have a smaller seq",
                     generation_id=gen.generation_id)
    visiting, settled = set(), set()

    def visit(gid):
        if gid in settled:
            return
        _require(gid not in visiting, f"cycle detected at generation {gid}",
                 generation_id=gid)
        visiting.add(gid)
        gen = next(item for item in parsed_gens if item.generation_id == gid)
        if gen.parent_id:
            visit(gen.parent_id)
        visiting.discard(gid)
        settled.add(gid)

    for gen in parsed_gens:
        visit(gen.generation_id)

    universe = tuple(sorted(((rid, lo, hi) for rid, (lo, hi) in range_bounds.items()),
                            key=lambda item: item[1]))
    return SnapshotChain(title, limits, tuple(parsed_gens), universe)

