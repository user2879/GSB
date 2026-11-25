"""v1 fixed-width lines and v2 JSON conclusions: parsing, serialization,
and sidecar-backed round trips that never re-derive folded information."""
from __future__ import annotations

from .core import BizError, E_BAD_ID, TIERS, to_fixed

SEL_CODE_TO_NAME = {"0": "DROPPED", "1": "SELECTED"}
SEL_NAME_TO_CODE = {v: k for k, v in SEL_CODE_TO_NAME.items()}
SCHEME_NAME = "stratified_permille_v1"

V1_WIDTH = 2 + 12 + 4 + 12 + 7 + 1  # kind|req|ordinal|cid|score_permille|sel


def format_v1_line(request_id, ordinal, candidate_id, score_permille, sel_code):
    rid = str(request_id)
    if len(rid) > 12:
        raise BizError(E_BAD_ID, "request_id %r exceeds v1 width 12" % rid, rid)
    return "V1%12s%04d%012d%+07d%d" % (rid, ordinal, candidate_id, score_permille, sel_code)


def parse_v1_line(line):
    """Parse a v1 fixed-width line into request id, ordinal, integer candidate
    id, permille integer score and integer selection code."""
    if len(line) != V1_WIDTH or not line.startswith("V1"):
        raise BizError(E_BAD_ID, "malformed v1 line (width %d)" % len(line))
    return {
        "request_id": line[2:14].strip(),
        "ordinal": int(line[14:18]),
        "candidate_id": int(line[18:30]),
        "score_permille": int(line[30:37]),
        "selection_code": int(line[37:38]),
    }


def picked_set(concl):
    out = set()
    for layer in concl["layers"].values():
        out.update(layer["picked"])
    return out


def build_sidecar_entry(concl):
    return {
        "precision": {cid: dict(v) for cid, v in concl["scores"].items()},
        "boundary": {str(b["candidate_id"]): {"diff_ref": b["diff_ref"], "tier": b["tier"]}
                     for b in concl["boundary"]},
        "sel_code_map": dict(SEL_CODE_TO_NAME),
        "scheme": SCHEME_NAME,
        "quant_tier": concl["quant_tier"],
    }


def export_v1(concl):
    """Serialize a conclusion to v1 lines; returns (lines, sidecar_entry).
    Folding losses (ref residual, boundary, scheme name) go to the sidecar."""
    frac = TIERS[concl["quant_tier"]]
    picked = picked_set(concl)
    lines = []
    for ordinal, cid in enumerate(concl["truncated"]):
        q = concl["scores"][str(cid)]["q"]
        permille = to_fixed(q * 1000.0 / (1 << frac), 0)
        lines.append(format_v1_line(concl["request_id"], ordinal, cid,
                                    permille, 1 if cid in picked else 0))
    return lines, build_sidecar_entry(concl)


def export_v2(concl):
    frac = TIERS[concl["quant_tier"]]
    picked = picked_set(concl)
    bmap = {b["candidate_id"]: b for b in concl["boundary"]}
    cands = []
    for cid in concl["truncated"]:
        b = bmap.get(cid)
        cands.append({
            "candidate_id": cid,
            "score": concl["scores"][str(cid)]["q"] / (1 << frac),
            "selection": "SELECTED" if cid in picked else "DROPPED",
            "boundary": b is not None,
            "boundary_diff_ref": b["diff_ref"] if b else None,
            "quant_tier": concl["quant_tier"],
        })
    return {"version": "v2", "request_id": concl["request_id"],
            "scheme": SCHEME_NAME, "candidates": cands}


def parse_v2(record):
    """Parse a v2 JSON record into float scores, named selection and named
    sampling scheme."""
    if record.get("version") != "v2":
        raise BizError(E_BAD_ID, "not a v2 record")
    cands = []
    for c in record["candidates"]:
        if c["selection"] not in SEL_NAME_TO_CODE:
            raise BizError(E_BAD_ID, "unknown selection name %r" % c["selection"])
        cands.append({
            "candidate_id": int(c["candidate_id"]),
            "score": float(c["score"]),
            "selection": c["selection"],
            "boundary": bool(c.get("boundary", False)),
            "boundary_diff_ref": c.get("boundary_diff_ref"),
            "quant_tier": c.get("quant_tier"),
        })
    return {"request_id": str(record["request_id"]), "scheme": str(record["scheme"]),
            "candidates": cands}


def v1_lines_to_v2(lines, sidecar, ledger=None):
    """Convert v1 lines back to v2 records. Boundary flags/diffs and exact
    fixed-point scores are restored from the sidecar, never re-derived from
    folded permille values."""
    by_req = {}
    for line in lines:
        p = parse_v1_line(line)
        by_req.setdefault(p["request_id"], []).append(p)
    records = []
    for rid in sorted(by_req):
        sc = sidecar.get(rid)
        tier = sc["quant_tier"] if sc else "Q12"
        frac = TIERS[tier]
        cands = []
        for p in sorted(by_req[rid], key=lambda p: p["ordinal"]):
            cid_key = str(p["candidate_id"])
            if sc and cid_key in sc["precision"]:
                q = sc["precision"][cid_key]["q"]
                score = q / (1 << frac)
                b = sc["boundary"].get(cid_key)
                boundary = b is not None
                diff = b["diff_ref"] if b else None
            else:
                score = p["score_permille"] / 1000.0  # degraded folded value
                boundary, diff = False, None
                if ledger is not None:
                    ledger.append({"type": "degraded_restore", "request_id": rid,
                                   "candidate_id": p["candidate_id"],
                                   "reason": "sidecar_missing"})
            cands.append({"candidate_id": p["candidate_id"], "score": score,
                          "selection": SEL_CODE_TO_NAME[str(p["selection_code"])],
                          "boundary": boundary, "boundary_diff_ref": diff,
                          "quant_tier": tier})
        records.append({"version": "v2", "request_id": rid,
                        "scheme": sc["scheme"] if sc else SCHEME_NAME,
                        "candidates": cands})
    return records


def downgrade_entries(concl, direction):
    """Ledger entries describing what a conversion folds away."""
    folded = ["ref_residual", "boundary_marks", "boundary_diff_ref"]
    if direction == "v2->v1":
        folded.append("scheme_name")
    return {"type": "downgrade", "direction": direction,
            "request_id": concl["request_id"], "folded": folded,
            "sidecar_fields": ["precision", "boundary", "sel_code_map", "scheme"]}
