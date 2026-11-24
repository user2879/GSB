"""Integer fixed-point stage transformations.

Every stage is element-wise and stateless: given identical input blocks it
produces byte-identical outputs regardless of block partition or scheduling
order. Nulls (``None``) propagate through every stage.

Result metadata used by the error-chain engine:
* ``boundary_uncertain`` rows are those whose bucket membership can flip under
  the propagated input error bound;
* ``saturated`` rows hit the signed 32-bit envelope and are excluded from the
  finite error-bound guarantee;
* ``nonpos_skipped`` rows are log inputs that are null/non-positive.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import fixedpoint as fp
from .spec import StageSpec


@dataclass
class BlockResult:
    values: list[int | None]
    scale_exp: int
    saturated: list[bool]
    boundary_uncertain: list[bool]
    nonpos_skipped: list[bool]
    max_abs: int = 0  # max absolute non-null value; drives tight product bounds


def _empty_meta(n: int):
    return ([False] * n, [False] * n, [False] * n)


def _rescale_block(values, in_exp: int, out_exp: int):
    """Rescale a block; saturation detected via the pre-clamp threshold."""
    out, sat = [], []
    for v in values:
        if v is None:
            out.append(None)
            sat.append(False)
            continue
        if out_exp > in_exp:
            did_sat = abs(v) > fp.INT32_MAX // fp.POW10[out_exp - in_exp]
        else:
            did_sat = False  # downscaling shrinks magnitude, cannot overflow
        out.append(fp.rescale_int(v, in_exp, out_exp))
        sat.append(did_sat)
    return out, sat


def run_standardize(stage: StageSpec, values, in_exp: int) -> BlockResult:
    """z = sat32(round_half_away((x-mean) * 10**scale / sd)); nulls propagate."""
    p = stage.params
    mean = fp.rescale_int(p["mean"], p["sd_scale_exp"], in_exp)
    sd = fp.rescale_int(p["sd"], p["sd_scale_exp"], in_exp)
    n = len(values)
    saturated, uncertain, skipped = _empty_meta(n)
    out = []
    for i, x in enumerate(values):
        if x is None:
            out.append(None)
            continue
        numer = (x - mean) * fp.POW10[stage.scale_exp]
        z = fp.sat32(fp.div_half_away(numer, sd))
        out.append(z)
        saturated[i] = z in (fp.INT32_MIN, fp.INT32_MAX)
    max_abs = max((abs(v) for v in out if v is not None), default=0)
    return BlockResult(out, stage.scale_exp, saturated, uncertain, skipped, max_abs)


def run_log(stage: StageSpec, values, in_exp: int) -> BlockResult:
    base_map = {0: "e", 1: "2", 2: "10"}
    base = base_map[stage.params["base_flag"]]
    n = len(values)
    saturated, uncertain, skipped = _empty_meta(n)
    out = []
    for i, x in enumerate(values):
        if x is None:
            out.append(None)
            continue
        if x <= 0:
            out.append(None)
            skipped[i] = True
            continue
        y = fp.sat32(fp.ilog(x, in_exp, base, stage.scale_exp))
        out.append(y)
        saturated[i] = y in (fp.INT32_MIN, fp.INT32_MAX)
    max_abs = max((abs(v) for v in out if v is not None), default=0)
    return BlockResult(out, stage.scale_exp, saturated, uncertain, skipped, max_abs)


def _bucket_id(x: int, edges) -> int:
    bid = 0
    for e in edges:
        if x <= e:
            break
        bid += 1
    return bid  # 0..len(edges)


def run_bucket(stage: StageSpec, values, in_exp: int,
               input_bound_ulp: int) -> BlockResult:
    """Bucket by integer edges (rule x<=edge joins the left bucket).

    Output is the bucket center at the stage scale. A row is boundary
    uncertain when the propagated input bound could move it across an edge.
    """
    edges = list(stage.params["edges"])
    n = len(values)
    saturated, uncertain, skipped = _empty_meta(n)
    out = []
    for i, x in enumerate(values):
        if x is None:
            out.append(None)
            continue
        bid = _bucket_id(x, edges)
        lo = fp.INT32_MIN if bid == 0 else edges[bid - 1]
        hi = fp.INT32_MAX if bid == len(edges) else edges[bid]
        # Unbounded outer buckets collapse to the finite edge (no int32 center).
        lo_c = lo if lo != fp.INT32_MIN else edges[0]
        hi_c = hi if hi != fp.INT32_MAX else edges[-1]
        center_in = fp.div_half_away(lo_c + hi_c, 2)
        center = fp.rescale_int(center_in, in_exp, stage.scale_exp)
        out.append(center)
        uncertain[i] = any(abs(x - e) <= input_bound_ulp for e in edges)
        saturated[i] = center in (fp.INT32_MIN, fp.INT32_MAX)
    max_abs = max((abs(v) for v in out if v is not None), default=0)
    return BlockResult(out, stage.scale_exp, saturated, uncertain, skipped, max_abs)


def run_product(stage: StageSpec, left, right, in_exp: int) -> BlockResult:
    n = len(left)
    saturated, uncertain, skipped = _empty_meta(n)
    out = []
    for i in range(n):
        y, did_sat = fp.mul_quant(left[i], right[i], in_exp, stage.scale_exp)
        out.append(y)
        saturated[i] = bool(did_sat)
    max_abs = max((abs(v) for v in out if v is not None), default=0)
    return BlockResult(out, stage.scale_exp, saturated, uncertain, skipped, max_abs)


def run_truncate(stage: StageSpec, values, in_exp: int) -> BlockResult:
    """Clamp to bounds declared at the stage's own scale, then rescale output."""
    lo = fp.rescale_int(stage.params["lower"], stage.scale_exp, in_exp)
    hi = fp.rescale_int(stage.params["upper"], stage.scale_exp, in_exp)
    n = len(values)
    saturated, uncertain, skipped = _empty_meta(n)
    out = []
    for i, x in enumerate(values):
        if x is None:
            out.append(None)
            continue
        if x < lo:
            out.append(lo)
            saturated[i] = True
        elif x > hi:
            out.append(hi)
            saturated[i] = True
        else:
            out.append(x)
    if in_exp != stage.scale_exp:
        out, sat2 = _rescale_block(out, in_exp, stage.scale_exp)
        saturated = [a or b for a, b in zip(saturated, sat2)]
    max_abs = max((abs(v) for v in out if v is not None), default=0)
    return BlockResult(out, stage.scale_exp, saturated, uncertain, skipped, max_abs)
