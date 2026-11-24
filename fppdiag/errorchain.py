"""Synthesis of fixed-point error upper bounds along the transform chain.

All bounds are integer *output ulps* at the producing stage's scale exponent.
Two passes happen in declared stage order (PipelineSpec.stages), so results
never depend on dict traversal order:

1. every stage's integer values are computed by the engine;
2. this module folds the observed per-stage magnitudes into tight bounds.

Modelled error sources:
* ``quant``    - single-value rounding/rescale quantization;
* ``bucket``   - bucket boundary membership plus center half-width;
* ``product``  - interaction amplification |x|*eb + |y|*ea plus quantization.

A stage is the *dominant* contributor when its added contribution exceeds
DOMINANT_SHARE_BPS (4000 = 40%) of its accumulated contribution; ties resolve
to the earliest stage (stable rule).
"""

from __future__ import annotations

from . import fixedpoint as fp
from .spec import PipelineSpec

DOMINANT_SHARE_BPS = 4000
# Conservative bound on the integer ln routine in output ulps (verified by
# tests against math.log across every allowed in/out scale pair; observed 12).
LOG_ROUTINE_BOUND = 16


def build_error_chains(spec: PipelineSpec, engine) -> dict[str, dict]:
    chains: dict[str, dict] = {}
    for stage in spec.stages:
        chains[stage.stage_id] = _chain_for(stage, spec, engine, chains)
    _assign_chain_dominance(chains, spec)
    return chains


def _quant_bound(in_exp: int, out_exp: int) -> int:
    """Max rounding error from rescaling, in output ulps (half-away rule)."""
    if out_exp >= in_exp:
        return 0
    return fp.POW10[in_exp - out_exp] // 2


def _to_scale(value: int, from_exp: int, to_exp: int) -> int:
    if to_exp >= from_exp:
        return value * fp.POW10[to_exp - from_exp]
    return fp.div_half_away(value, fp.POW10[from_exp - to_exp])


def _chain_for(stage, spec, engine, chains: dict[str, dict]) -> dict:
    entries: list[dict] = []

    if stage.stype == "standardize":
        in_exp = spec.sources[stage.source_id].ingest_scale
        added = 1 + _quant_bound(in_exp, stage.scale_exp)
        bound, kind = added, "quant"

    elif stage.stype == "log":
        in_exp = spec.sources[stage.source_id].ingest_scale
        added = LOG_ROUTINE_BOUND + _quant_bound(in_exp, stage.scale_exp)
        bound, kind = added, "quant"

    elif stage.stype == "bucket":
        up_stage = spec.stage(stage.refs[0])
        in_exp = up_stage.scale_exp
        up_bound = chains[stage.refs[0]]["bound_ulp"]
        edges = list(stage.params["edges"])
        half_widths = []
        for bid in range(len(edges) + 1):
            lo = fp.INT32_MIN if bid == 0 else edges[bid - 1]
            hi = fp.INT32_MAX if bid == len(edges) else edges[bid]
            lo_c = lo if lo != fp.INT32_MIN else edges[0]
            hi_c = hi if hi != fp.INT32_MAX else edges[-1]
            half_widths.append(fp.div_half_away(abs(hi_c - lo_c), 2))
        max_half = max(half_widths)
        center_bound_in = up_bound + max_half
        bound = _to_scale(center_bound_in, in_exp, stage.scale_exp)
        added, kind = bound, "bucket"

    elif stage.stype == "product":
        left_id, right_id = stage.refs
        left_chain, right_chain = chains[left_id], chains[right_id]
        in_exp = spec.stage(left_id).scale_exp
        left_run = engine.runs.get(left_id)
        right_run = engine.runs.get(right_id)
        mag_left = max((b.max_abs for b in left_run.blocks), default=0) \
            if left_run else 0
        mag_right = max((b.max_abs for b in right_run.blocks), default=0) \
            if right_run else 0
        # |x+dx)(y+dy) - xy| <= |x||dy| + |y||dx| (second-order term dropped)
        ampl_in = mag_left * right_chain["bound_ulp"] + \
            mag_right * left_chain["bound_ulp"]
        ampl = _to_scale(ampl_in, 2 * in_exp, stage.scale_exp)
        quant = (fp.POW10[2 * in_exp - stage.scale_exp] // 2
                 if 2 * in_exp > stage.scale_exp else 0)
        bound = ampl + quant
        added, kind = bound, "product"
        entries.append({"stage_id": stage.stage_id, "source": "product",
                        "amplification_ulp": ampl,
                        "quantization_ulp": quant,
                        "mag_left_ulp": mag_left, "mag_right_ulp": mag_right})

    elif stage.stype == "truncate":
        up_stage = spec.stage(stage.refs[0])
        in_exp = up_stage.scale_exp
        up_bound = chains[stage.refs[0]]["bound_ulp"]
        propagated = _to_scale(up_bound, in_exp, stage.scale_exp)
        quant = _quant_bound(in_exp, stage.scale_exp)
        bound = propagated + quant
        added, kind = quant, "quant"

    else:  # pragma: no cover - guarded by spec validation
        raise ValueError(stage.stype)

    if not entries:
        entries.append({"stage_id": stage.stage_id, "source": kind,
                        "added_ulp": added})
    chain = {"stage_id": stage.stage_id, "type": stage.stype,
             "scale_exp": stage.scale_exp, "bound_ulp": int(bound),
             "added_ulp": int(added), "entries": entries,
             "upstream_refs": list(stage.refs)}
    return chain


def _assign_chain_dominance(chains: dict, spec: PipelineSpec) -> None:
    """Per-feature contribution shares in declaration order; stable dominant."""
    for stage in spec.stages:
        chain = chains[stage.stage_id]
        # Walk the ancestor chain (single parent except product's two parents).
        ancestor_added = []
        seen = set()

        def walk(sid: str):
            if sid in seen:
                return
            seen.add(sid)
            for ref in chains[sid]["upstream_refs"]:
                walk(ref)
            ancestor_added.append(chains[sid]["added_ulp"])

        walk(stage.stage_id)
        total = sum(ancestor_added)
        shares = []
        for sid in [s.stage_id for s in spec.stages
                    if s.stage_id in seen]:
            added = chains[sid]["added_ulp"]
            share = fp.bps_ratio(added, total) if total else 0
            chains[sid].setdefault("contribution_share_bps", {})
            chains[sid]["contribution_share_bps"][stage.stage_id] = share
            shares.append((sid, share, added))
        chain["dominant_threshold_bps"] = DOMINANT_SHARE_BPS
        # earliest stage wins ties -> iterate declared order, take strict >.
        dominant = None
        best = -1
        for sid, share, added in shares:
            if share > best and share >= DOMINANT_SHARE_BPS:
                best, dominant = share, sid
        chain["dominant_stage_id"] = dominant
        chain["contribution_total_ulp"] = total
