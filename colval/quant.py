"""Fixed-point quantization with a unified precision policy.

Policy (declared, deterministic):
  - Session scale factor S = 10 ** scale_exp (integer power of ten).
  - quantize(x) = trunc_toward_zero(x * S), saturated to int64.
    Truncation direction: toward zero. Overflow rule: saturate to int64.
  - dequantize(q) = q / S.
  - Fixed-point multiply: (qa * qb) trunc-div S, saturate.
  - Fixed-point divide: (qa * S) trunc-div qb, saturate; a zero divisor is
    handled by the executor's null/flag rules, not here.
  - All trunc-divisions are toward zero (never floor).

Error model (real units):
  - Single-value quantization error: |x - dequantize(quantize(x))| <= 1/S.
  - add/sub: e = ea + eb.
  - mul:     e = |b|*ea + |a|*eb + ea*eb + 1/S (output re-quantization).
  - div:     e = (ea + (|a|+ea)*eb/den) / den + 1/S with den = |b| - eb,
             valid only when den > 0; otherwise the bound is +inf.
  - Aggregated batch bound: rows * per-row bound.
"""

from .expr import Bin, Col, Num, Ternary, Un
from .formats import INT64_MAX, INT64_MIN


def trunc_div(a, b):
    """Integer division truncated toward zero (b != 0)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def saturate(q):
    if q > INT64_MAX:
        return INT64_MAX
    if q < INT64_MIN:
        return INT64_MIN
    return q


class QuantContext:
    def __init__(self, scale_exp):
        if scale_exp < 0:
            raise ValueError("scale_exp must be non-negative")
        self.scale_exp = scale_exp
        self.scale = 10 ** scale_exp

    def quantize(self, x):
        x = float(x)
        q = int(abs(x) * self.scale)  # int() truncates toward zero
        return saturate(q if x >= 0 else -q)

    def quantize_int(self, n):
        return saturate(int(n) * self.scale)

    def dequantize(self, q):
        return q / self.scale

    @property
    def unit(self):
        return 1.0 / self.scale

    @property
    def single_value_bound(self):
        return self.unit


class _Bounds:
    """Error-bound evaluator over an expression AST (real units)."""

    def __init__(self, ctx, col_stats):
        self.ctx = ctx
        self.col_stats = col_stats

    def eval(self, node):
        """Return (max_abs_value, error_bound) for the expression."""
        u = self.ctx.unit
        if isinstance(node, Num):
            v = abs(float(node.value))
            if isinstance(node.value, float):
                err = abs(float(node.value) - self.ctx.dequantize(
                    self.ctx.quantize(float(node.value))))
            else:
                err = 0.0
            return v, err
        if isinstance(node, Col):
            return self.col_stats.get(node.name, (float("inf"), u))
        if isinstance(node, Un):
            v, e = self.eval(node.expr)
            return v, e
        if isinstance(node, Ternary):
            self.eval(node.cond)
            va, ea = self.eval(node.then)
            vb, eb = self.eval(node.otherwise)
            return max(va, vb), max(ea, eb)
        if isinstance(node, Bin):
            va, ea = self.eval(node.left)
            vb, eb = self.eval(node.right)
            if node.op in ("+", "-"):
                return va + vb, ea + eb
            if node.op == "*":
                return va * vb, vb * ea + va * eb + ea * eb + u
            if node.op == "/":
                den = vb - eb
                if den <= 0:
                    return float("inf"), float("inf")
                err = (ea + (va + ea) * eb / den) / den + u
                return va / den, err
            if node.op == "%":
                return vb, ea + u
            return 1.0, 0.0  # comparisons / logic produce exact 0/1
        raise TypeError(node)


def expression_error_bound(ast, ctx, col_stats):
    """Per-row error bound (real units) for an expression AST."""
    return _Bounds(ctx, col_stats).eval(ast)[1]


def batch_error_bound(ast, ctx, col_stats, rows):
    """Aggregated error bound over a batch of rows (sum of per-row bounds)."""
    return rows * expression_error_bound(ast, ctx, col_stats)


def rank_candidates(candidates, ctx, col_stats):
    """Rank candidate expressions by fixed-point result with stable tie-break.

    candidates: list of (expr_text, ast, quantized_result:int).
    Two results whose difference falls within the combined error bound are
    ordered by the stable rule (normalized digest, then normalized text) and
    flagged error_affected=True.
    """
    from .expr import expr_digest, normalize

    entries = []
    for text, ast, result in candidates:
        bound = expression_error_bound(ast, ctx, col_stats)
        entries.append({
            "expr": text,
            "normalized": normalize(ast),
            "digest": expr_digest(text),
            "result": result,
            "bound": bound,
            "error_affected": False,
        })
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            combined = entries[i]["bound"] + entries[j]["bound"]
            if abs(entries[i]["result"] - entries[j]["result"]) <= combined * ctx.scale:
                entries[i]["error_affected"] = True
                entries[j]["error_affected"] = True
    entries.sort(key=lambda e: (e["result"], e["digest"], e["normalized"]))
    return entries
