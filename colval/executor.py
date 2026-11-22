"""Vectorized in-batch evaluation and deterministic merge.
Semantics (explicit rules):
  * All arithmetic runs on fixed-point int64 quanta (see quant.py):
    truncation toward zero, saturation to int64 on overflow.
  * Null propagation: any null operand yields a null result (arithmetic,
    comparison, ternary branches). Logic uses three-valued rules:
      false AND null -> false; true OR null -> true; NOT null -> null;
    otherwise null propagates.
  * Division/modulo by zero: result row becomes null and the row is flagged
    DIV_BY_ZERO; the flag propagates to the final output.
  * Ternary: null condition -> null; otherwise the chosen branch (including
    its nulls) is selected.
  * Merge: batch outputs are concatenated in row-range order; aggregations
    (none/sum/count/avg) are exact integer operations, so the final result
    is byte-identical for any batch size and any batch execution order.
"""

import hashlib
import struct

import numpy as np

from .expr import Bin, Col, Num, Ternary, Un
from .formats import make_bitmap
from .quant import saturate, trunc_div

FLAG_DIV_BY_ZERO = "DIV_BY_ZERO"


class Vec:
    """A vectorized column segment: quantized int values + nulls + flags."""

    def __init__(self, values, nulls, div_zero=None):
        self.values = np.asarray(values, dtype=object)
        self.nulls = np.asarray(nulls, dtype=bool)
        self.div_zero = (np.zeros(len(self.values), dtype=bool)
                         if div_zero is None else np.asarray(div_zero, dtype=bool))

    def __len__(self):
        return len(self.values)

    @staticmethod
    def const(value, n):
        return Vec([value] * n, np.zeros(n, dtype=bool))


def _elementwise(a, b, fn):
    n = len(a.values)
    out = np.zeros(n, dtype=object)
    nulls = a.nulls | b.nulls
    div_zero = a.div_zero | b.div_zero
    for i in range(n):
        if not nulls[i]:
            out[i] = fn(a.values[i], b.values[i])
    return Vec(out, nulls, div_zero)


def _divmod(a, b, op, ctx):
    n = len(a.values)
    out = np.zeros(n, dtype=object)
    nulls = (a.nulls | b.nulls).copy()
    div_zero = a.div_zero | b.div_zero
    for i in range(n):
        if nulls[i]:
            continue
        if b.values[i] == 0:
            nulls[i] = True
            div_zero[i] = True
            continue
        if op == "/":
            out[i] = saturate(trunc_div(a.values[i] * ctx.scale, b.values[i]))
        else:
            r = abs(a.values[i]) % abs(b.values[i])
            out[i] = r if a.values[i] >= 0 else -r
    return Vec(out, nulls, div_zero)


def _eval(node, env, ctx):
    if isinstance(node, Num):
        q = (ctx.quantize_int(node.value) if isinstance(node.value, int)
             else ctx.quantize(node.value))
        return Vec.const(q, env["__n__"])
    if isinstance(node, Col):
        if node.name not in env:
            raise KeyError(f"unknown column {node.name!r}")
        return env[node.name]
    if isinstance(node, Un):
        v = _eval(node.expr, env, ctx)
        if node.op == "neg":
            out = [0 if isn else saturate(-x) for isn, x in zip(v.nulls, v.values)]
            return Vec(out, v.nulls, v.div_zero)
        if node.op == "not":
            out = [0 if isn else (0 if x else 1) for isn, x in zip(v.nulls, v.values)]
            return Vec(out, v.nulls, v.div_zero)
        raise ValueError(node.op)
    if isinstance(node, Ternary):
        cond = _eval(node.cond, env, ctx)
        then = _eval(node.then, env, ctx)
        other = _eval(node.otherwise, env, ctx)
        n = len(cond.values)
        out = np.zeros(n, dtype=object)
        nulls = np.zeros(n, dtype=bool)
        for i in range(n):
            if cond.nulls[i]:
                nulls[i] = True
                continue
            src = then if cond.values[i] else other
            if src.nulls[i]:
                nulls[i] = True
            else:
                out[i] = src.values[i]
        return Vec(out, nulls, cond.div_zero | then.div_zero | other.div_zero)
    if isinstance(node, Bin):
        a = _eval(node.left, env, ctx)
        b = _eval(node.right, env, ctx)
        op = node.op
        if op == "+":
            return _elementwise(a, b, lambda x, y: saturate(x + y))
        if op == "-":
            return _elementwise(a, b, lambda x, y: saturate(x - y))
        if op == "*":
            return _elementwise(a, b, lambda x, y: saturate(trunc_div(x * y, ctx.scale)))
        if op in ("/", "%"):
            return _divmod(a, b, op, ctx)
        if op in ("==", "!=", "<", "<=", ">", ">="):
            cmp_fn = {"==": lambda x, y: x == y, "!=": lambda x, y: x != y,
                      "<": lambda x, y: x < y, "<=": lambda x, y: x <= y,
                      ">": lambda x, y: x > y, ">=": lambda x, y: x >= y}[op]
            return _elementwise(a, b, lambda x, y: 1 if cmp_fn(x, y) else 0)
        if op in ("and", "or"):
            return _logic(a, b, op)
        raise ValueError(op)
    raise TypeError(node)


def _logic(a, b, op):
    n = len(a.values)
    out = np.zeros(n, dtype=object)
    nulls = np.zeros(n, dtype=bool)
    for i in range(n):
        av = None if a.nulls[i] else bool(a.values[i])
        bv = None if b.nulls[i] else bool(b.values[i])
        if op == "and":
            r = False if (av is False or bv is False) else (
                None if (av is None or bv is None) else True)
        else:
            r = True if (av is True or bv is True) else (
                None if (av is None or bv is None) else False)
        if r is None:
            nulls[i] = True
        else:
            out[i] = 1 if r else 0
    return Vec(out, nulls, a.div_zero | b.div_zero)


def batch_env(block, ctx, row_start, row_end):
    """Quantized column environment for rows [row_start, row_end)."""
    env = {"__n__": row_end - row_start}
    for col in block.columns:
        vals, nulls = [], []
        for r in range(row_start, row_end):
            is_null = col.null_bitmap[r // 8] >> (r % 8) & 1
            nulls.append(bool(is_null))
            if is_null:
                vals.append(0)
            elif col.dtype == "int64":
                vals.append(ctx.quantize_int(col.values[r]))
            else:
                vals.append(ctx.quantize(col.values[r]))
        env[col.name] = Vec(vals, nulls)
    return env


def evaluate_batch(block, ast, ctx, row_start, row_end):
    """Evaluate one batch; returns the result Vec."""
    env = batch_env(block, ctx, row_start, row_end)
    return _eval(ast, env, ctx)


def merge_batches(batch_results):
    """batch_results: iterable of (row_start, Vec). Deterministic: sorts by
    row_start and concatenates -> identical for any execution order."""
    ordered = sorted(batch_results, key=lambda item: item[0])
    values, nulls, div_zero = [], [], []
    for _, vec in ordered:
        values.extend(int(v) for v in vec.values)
        nulls.extend(bool(x) for x in vec.nulls)
        div_zero.extend(bool(x) for x in vec.div_zero)
    return Vec(values, nulls, div_zero)


def aggregate(vec, agg):
    """Explicit aggregation semantics (exact integer ops)."""
    valid = [int(v) for v, is_null in zip(vec.values, vec.nulls) if not is_null]
    if agg == "none":
        return None
    if agg == "sum":
        return saturate(sum(valid))
    if agg == "count":
        return len(valid)
    if agg == "avg":
        if not valid:
            return None  # null
        return saturate(trunc_div(sum(valid), len(valid)))
    raise ValueError(f"unknown aggregation {agg!r}")


def result_digest(normalized, scale_exp, vec, agg="none", agg_value=None):
    """Canonical digest of a final result (byte-identity criterion)."""
    out = bytearray(b"COLVALRES\x00")
    out += normalized.encode() + b"\x00"
    out += struct.pack("<i", scale_exp)
    out += struct.pack("<q", len(vec.values))
    for v in vec.values:
        out += struct.pack("<q", int(v))
    out += make_bitmap([i for i, x in enumerate(vec.nulls) if x], len(vec.values))
    out += make_bitmap([i for i, x in enumerate(vec.div_zero) if x], len(vec.values))
    out += agg.encode() + b"\x00"
    out += struct.pack("<q", -2 ** 63 if agg_value is None else int(agg_value))
    return hashlib.sha256(bytes(out)).hexdigest()
