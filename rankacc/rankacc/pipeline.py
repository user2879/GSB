"""Deterministic pipeline: per-channel normalization, fixed-point TopK
truncation with boundary detection, stratified sampling with seeded
shuffle, and stage-by-stage error attribution against a wider reference."""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .core import REF_EXTRA, BizError, E_INVALID_RATIO, half_quantum_ref, to_fixed


# --------------------------------------------------------------- HashPRNG
class HashPRNG:
    """Counter-based PRNG derived from SHA-256; never touches global state."""

    def __init__(self, seed):
        self._seed = str(seed).encode("utf-8")
        self._ctr = 0
        self._buf = b""

    def _fill(self):
        self._buf = hashlib.sha256(self._seed + self._ctr.to_bytes(8, "big")).digest()
        self._ctr += 1

    def randbelow(self, n):
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            if len(self._buf) < 8:
                self._fill()
            chunk = int.from_bytes(self._buf[:8], "big")
            self._buf = self._buf[8:]
            if chunk < limit:
                return chunk % n


def shuffle_ids(ids, seed):
    ids = list(ids)
    rng = HashPRNG(seed)
    for i in range(len(ids) - 1, 0, -1):
        j = rng.randbelow(i + 1)
        ids[i], ids[j] = ids[j], ids[i]
    return ids


# ------------------------------------------------------------ normalize
def normalize_channel(cands, weight, frac, method):
    """Normalize one channel. `cands` must be sorted by candidate_id so the
    accumulation order is fixed (channel name asc, candidate id asc overall)."""
    xs = np.array([c.score for c in cands], dtype=np.float64)
    if method == "minmax":
        lo, hi = xs.min(), xs.max()
        denom = hi - lo
        normed = np.zeros_like(xs) if denom == 0 else (xs - lo) / denom
    else:  # zscore, fixed-order summation
        mean = np.float64(0.0)
        for x in xs:
            mean += x
        mean /= len(xs)
        var = np.float64(0.0)
        for x in xs:
            var += (x - mean) * (x - mean)
        std = math.sqrt(var / len(xs))
        normed = np.zeros_like(xs) if std == 0 else (xs - mean) / std
    out = {}
    for c, v in zip(cands, normed):
        val = float(weight) * float(v)
        out[c.candidate_id] = {"q": to_fixed(val, frac), "ref": to_fixed(val, frac + REF_EXTRA)}
    return out


def group_by_channel(req):
    by_ch = {}
    for c in req.candidates:
        by_ch.setdefault(c.channel, []).append(c)
    return {ch: sorted(cs, key=lambda c: c.candidate_id) for ch, cs in sorted(by_ch.items())}


def normalize_request(req, cfg, frac):
    scores = {}
    for ch, cands in group_by_channel(req).items():
        scores.update(normalize_channel(cands, cfg.channel_weights[ch], frac, cfg.norm_method))
    return scores


# ------------------------------------------------------------- truncate
def truncate(scores, topk, tier):
    """Sort by normalized fixed score desc, candidate id asc; cut at topk.
    Boundary candidates: within half a quantum (reference units) of the cut."""
    order = sorted(scores, key=lambda cid: (-scores[cid]["q"], cid))
    selected = order[:topk]
    boundary = []
    if len(order) > topk and selected:
        cutoff_ref = scores[selected[-1]]["ref"]
        half = half_quantum_ref()
        for cid in order:
            diff = scores[cid]["ref"] - cutoff_ref
            if abs(diff) <= half:
                boundary.append({"candidate_id": cid, "diff_ref": diff, "tier": tier})
    return {"order": order, "selected": selected, "boundary": boundary}


# --------------------------------------------------------------- sampling
def layer_quotas(layer_sizes, ratio_permille):
    """Largest-remainder allocation; remainders filled in channel-name order."""
    channels = sorted(layer_sizes)
    total = sum(layer_sizes.values())
    total_quota = ratio_permille * total // 1000
    base, rem = {}, {}
    for ch in channels:
        raw = ratio_permille * layer_sizes[ch]
        base[ch], rem[ch] = raw // 1000, raw % 1000
    deficit = total_quota - sum(base.values())
    for ch in channels:  # remainder fill, channel name ascending
        if deficit <= 0:
            break
        if rem[ch] > 0 and base[ch] < layer_sizes[ch]:
            base[ch] += 1
            deficit -= 1
    for ch in channels:  # safety pass if remainders were all zero
        if deficit <= 0:
            break
        if base[ch] < layer_sizes[ch]:
            base[ch] += 1
            deficit -= 1
    return total_quota, base


def shuffle_layer(member_ids, aux_of, config_fp, request_id, channel):
    """Group by aux integer (missing aux -> single trailing group), then
    shuffle each group with a PRNG seeded from config fingerprint + request."""
    groups = {}
    for cid in sorted(member_ids):
        groups.setdefault(aux_of(cid), []).append(cid)
    int_keys = sorted(k for k in groups if k is not None)
    keys = int_keys + ([None] if None in groups else [])
    order = []
    for key in keys:
        seed = f"shuffle|{config_fp}|{request_id}|{channel}|{key}"
        order.extend(shuffle_ids(groups[key], seed))
    return order


def stratified_sample(req, selected, ratio_permille, config_fp):
    if not (0 <= ratio_permille <= 1000):
        raise BizError(E_INVALID_RATIO, f"ratio {ratio_permille} out of [0,1000]", req.request_id)
    table = {c.candidate_id: c for c in req.candidates}
    layer_sizes = {}
    for cid in selected:
        ch = table[cid].channel
        layer_sizes[ch] = layer_sizes.get(ch, 0) + 1
    total_quota, base = layer_quotas(layer_sizes, ratio_permille)
    layers = {}
    for ch in sorted(layer_sizes):
        members = [cid for cid in selected if table[cid].channel == ch]
        order = shuffle_layer(members, lambda cid: table[cid].aux, config_fp, req.request_id, ch)
        quota = base[ch]
        layers[ch] = {"quota": quota, "order": order,
                      "picked": order[:quota], "dropped": order[quota:]}
    return {"total_quota": total_quota, "layers": layers}


# ------------------------------------------------------------- compute all
def compute_all(req, cfg, frac, config_fp, ratio_permille=None):
    """Full per-request pipeline at a given fractional bitwidth."""
    ratio = cfg.sample_ratio_permille if ratio_permille is None else ratio_permille
    scores = normalize_request(req, cfg, frac)
    trunc = truncate(scores, cfg.topk, cfg.quant_tier)
    sample = stratified_sample(req, trunc["selected"], ratio, config_fp)
    return {"scores": scores, "trunc": trunc, "sample": sample}


# ------------------------------------------------------------- attribution
def attribute_request(req, cfg, config_fp):
    """Compare the pipeline at frac against a one-bit-wider reference and
    report per-stage deltas plus the first stage exceeding tolerance."""
    frac = cfg.frac
    main = compute_all(req, cfg, frac, config_fp)
    ref = compute_all(req, cfg, frac + 1, config_fp)
    tol2 = 2 * cfg.tolerance_units  # tolerance expressed in half-units of main scale

    norm_diff = 0
    for cid in main["scores"]:
        norm_diff = max(norm_diff, abs(2 * main["scores"][cid]["q"] - ref["scores"][cid]["q"]))
    norm_exceed = norm_diff > tol2

    main_b = {b["candidate_id"] for b in main["trunc"]["boundary"]}
    ref_b = {b["candidate_id"] for b in ref["trunc"]["boundary"]}
    flips = sorted(main_b ^ ref_b)
    quota_diff = {}
    for ch in main["sample"]["layers"]:
        mq = main["sample"]["layers"][ch]["quota"]
        rq = ref["sample"]["layers"].get(ch, {}).get("quota")
        if mq != rq:
            quota_diff[ch] = [mq, rq]
    flip_causes = {}
    for cid in flips:
        if 2 * main["scores"][cid]["q"] != ref["scores"][cid]["q"]:
            flip_causes[cid] = "norm_quantization"
        elif quota_diff:
            flip_causes[cid] = "quota_rounding"
        else:
            flip_causes[cid] = "truncation_tie"

    shuffle_diff = []
    for ch in main["sample"]["layers"]:
        ref_layer = ref["sample"]["layers"].get(ch)
        if ref_layer and main["sample"]["layers"][ch]["order"] != ref_layer["order"]:
            shuffle_diff.append(ch)

    first = None
    if norm_exceed:
        first = "normalize"
    elif flips:
        first = "truncate"
    elif quota_diff:
        first = "sample"
    elif shuffle_diff:
        first = "shuffle"
    return {
        "request_id": req.request_id,
        "norm_max_diff_half_units": norm_diff,
        "norm_exceeds_tolerance": norm_exceed,
        "boundary_flips": flips,
        "flip_causes": {str(k): v for k, v in flip_causes.items()},
        "quota_diff": quota_diff,
        "shuffle_diff_layers": shuffle_diff,
        "first_stage_exceeding": first,
    }
