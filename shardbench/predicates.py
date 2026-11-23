"""Predicate normalization and shard-key extraction.

Predicates arrive as JSON trees:
  {"eq": ["col", value]}
  {"in": ["col", [v1, v2, ...]]}
  {"range": ["col", low, high]}        low/high may be null (unbounded)
  {"and": [p1, p2, ...]}

Normalization is semantic: whitespace/case of identifiers and string literals,
IN-list order, and AND-clause order all collapse to one canonical form, so the
plan digest is byte-identical for equivalent predicates regardless of input
order or dict iteration order.
"""

from __future__ import annotations

from .canon import canonical_bytes, digest
from .errors import BizError, E_PREDICATE


def _fail(message, field=None):
    raise BizError(E_PREDICATE, message, field=field)


def _norm_value(value):
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    if value is None:
        return None
    _fail(f"unsupported literal type {type(value).__name__}")


def _norm_col(col, field):
    if not isinstance(col, str) or not col.strip():
        _fail("column reference must be a non-empty string", field)
    return col.strip().lower()


def _sort_key(value):
    return canonical_bytes(value)


def normalize(pred):
    """Return the canonical form of a predicate tree (plain JSON-safe data)."""
    if not isinstance(pred, dict) or len(pred) != 1:
        _fail("predicate node must be an object with exactly one operator")
    (op, body), = pred.items()
    op = op.strip().lower() if isinstance(op, str) else op
    if op == "eq":
        col, value = _pair(body, "eq")
        return {"eq": [_norm_col(col, "eq[0]"), _norm_value(value)]}
    if op == "in":
        col, values = _pair(body, "in")
        if not isinstance(values, list) or not values:
            _fail("IN list must be a non-empty array", "in[1]")
        frozen = sorted({_freeze(_norm_value(v)) for v in values})
        normed = [_thaw(f) for f in frozen]
        if len(normed) == 1:
            return {"eq": [_norm_col(col, "in[0]"), normed[0]]}
        return {"in": [_norm_col(col, "in[0]"), normed]}
    if op == "range":
        if not isinstance(body, list) or len(body) != 3:
            _fail("range must be [column, low, high]", "range")
        col = _norm_col(body[0], "range[0]")
        low = _norm_value(body[1])
        high = _norm_value(body[2])
        if low is not None and high is not None and low > high:
            _fail("range low must not exceed high", "range")
        return {"range": [col, low, high]}
    if op == "and":
        if not isinstance(body, list) or not body:
            _fail("AND must be a non-empty array of predicates", "and")
        children = [normalize(p) for p in body]
        children.sort(key=_sort_key)
        deduped = []
        for child in children:
            if not deduped or canonical_bytes(child) != canonical_bytes(deduped[-1]):
                deduped.append(child)
        if len(deduped) == 1:
            return deduped[0]
        return {"and": deduped}
    _fail(f"unsupported operator '{op}'")


def _pair(body, op):
    if not isinstance(body, list) or len(body) != 2:
        _fail(f"{op} must be [column, value]", op)
    return body[0], body[1]


def _freeze(value):
    return canonical_bytes(value)


def _thaw(frozen):
    import json

    return json.loads(frozen.decode("utf-8"))


def predicate_digest(pred) -> str:
    return digest(normalize(pred))


def _flatten_and(pred):
    if "and" in pred:
        out = []
        for child in pred["and"]:
            out.extend(_flatten_and(child))
        return out
    return [pred]


def extract_constraints(pred):
    """Map column -> constraint after intersecting all AND clauses.

    Constraint forms:
      {"eq": value}
      {"in": [v1, ...]}            (always length >= 1 after intersection)
      {"range": (low, high)}
    A column may end with an empty "in" list meaning a contradiction.
    """
    constraints = {}
    for node in _flatten_and(normalize(pred)):
        if "eq" in node:
            col, value = node["eq"]
            constraints[col] = _intersect(constraints.get(col), {"eq": value})
        elif "in" in node:
            col, values = node["in"]
            constraints[col] = _intersect(constraints.get(col), {"in": list(values)})
        elif "range" in node:
            col, low, high = node["range"]
            constraints[col] = _intersect(constraints.get(col), {"range": (low, high)})
    return constraints


def _intersect(old, new):
    if old is None:
        return new
    if "eq" in old and "eq" in new:
        return old if old["eq"] == new["eq"] else {"in": []}
    if "eq" in old:
        return _intersect(new, old)
    if "eq" in new:
        value = new["eq"]
        if "in" in old:
            return {"eq": value} if value in old["in"] else {"in": []}
        low, high = old["range"]
        ok = (low is None or value >= low) and (high is None or value <= high)
        return {"eq": value} if ok else {"in": []}
    if "in" in old and "in" in new:
        inter = sorted(set(old["in"]) & set(new["in"]), key=_sort_key)
        return {"in": inter}
    if "in" in old and "range" in new:
        return _intersect(new, old)
    if "range" in old and "in" in new:
        low, high = old["range"]
        kept = [v for v in new["in"]
                if (low is None or v >= low) and (high is None or v <= high)]
        return {"in": kept}
    low1, high1 = old["range"]
    low2, high2 = new["range"]
    low = max(x for x in (low1, low2) if x is not None) if (low1 is not None or low2 is not None) else None
    high = min(x for x in (high1, high2) if x is not None) if (high1 is not None or high2 is not None) else None
    if low is not None and high is not None and low > high:
        return {"in": []}
    return {"range": (low, high)}


def shard_key_tuples(unit, constraints):
    """Return (covered, key_tuples, intervals).

    covered: True when every shard-key column is pinned by eq/IN.
    key_tuples: sorted list of concrete key tuples (cartesian product).
    intervals: per-column (low, high) for range-constrained columns.
    """
    per_col = []
    intervals = {}
    covered = True
    for col in unit.shard_keys:
        cons = constraints.get(col)
        if cons is None:
            covered = False
            per_col.append(None)
        elif "eq" in cons:
            per_col.append([cons["eq"]])
        elif "in" in cons:
            per_col.append(list(cons["in"]))
        else:
            covered = False
            intervals[col] = cons["range"]
            per_col.append(None)
    if not covered:
        return False, [], intervals
    tuples = [()]
    for values in per_col:
        tuples = [t + (v,) for t in tuples for v in values]
    tuples.sort(key=_sort_key)
    return True, tuples, intervals
