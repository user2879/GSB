"""Generate a deterministic local sample: sources, windows and a 5-stage spec.

Standardize mean/sd are derived from the generated reference window with pure
integer statistics so the parameters are guaranteed valid and reproducible.
"""

from __future__ import annotations

import os

from . import fixedpoint as fp
from .sources import generate_column
from .spec import load_spec
from .util import canonical_json


def _integer_sd(values: list[int], mean: int, in_exp: int, sd_exp: int) -> int:
    # variance at scale 2*in_exp, then integer sqrt scaled to sd_exp
    sq_sum = sum((v - mean) ** 2 for v in values)
    n = len(values)
    var_scaled = fp.div_half_away(sq_sum, n)  # at 2*in_exp
    target_exp2 = 2 * sd_exp
    if target_exp2 >= 2 * in_exp:
        var_target = var_scaled * fp.POW10[target_exp2 - 2 * in_exp]
    else:
        var_target = fp.div_half_away(var_scaled,
                                      fp.POW10[2 * in_exp - target_exp2])
    return fp.sat32(_isqrt(var_target))


def _isqrt(n: int) -> int:
    if n < 0:
        raise ValueError("sqrt of negative")
    lo, hi = 0, 1
    while hi * hi <= n:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid * mid <= n:
            lo = mid
        else:
            hi = mid
    return lo


def build_sample_doc(rows: int = 240, version: int = 1,
                     session_scale_exp: int = 4) -> dict:
    split = rows // 2
    sources = [
        {"id": "src_price", "kind": "float", "rows": rows, "seed": 101,
         "low": 5000, "high": 20000, "ingest_scale": 2,
         "null_seed": 9001, "null_rate_bps": 800, "null_bitmap_len": rows},
        {"id": "src_vol", "kind": "integer", "rows": rows, "seed": 202,
         "low": 10, "high": 900, "ingest_scale": 0,
         "null_seed": 9002, "null_rate_bps": 500, "null_bitmap_len": rows},
        {"id": "src_size", "kind": "float", "rows": rows, "seed": 303,
         "low": 100, "high": 999, "ingest_scale": 1,
         "null_seed": 9003, "null_rate_bps": 300, "null_bitmap_len": rows},
    ]
    from .spec import _parse_source
    parsed_sources = [_parse_source(s) for s in sources]
    cols = {s.sid: generate_column(s) for s in parsed_sources}
    price_exp = 2
    ref_price = [c for c in cols["src_price"][:split] if c is not None]
    mean_price = fp.div_half_away(sum(ref_price), len(ref_price))
    sd_price = _integer_sd(ref_price, mean_price, price_exp, price_exp)

    stages = [
        {"id": "std_price", "type": "standardize", "scale_exp": 4,
         "source": "src_price", "refs": [],
         "params": {"mean": mean_price, "sd": max(1, sd_price),
                    "sd_scale_exp": price_exp}},
        {"id": "std_vol", "type": "standardize", "scale_exp": 4,
         "source": "src_vol", "refs": [],
         "params": {"mean": 0, "sd": 1000, "sd_scale_exp": 0}},
        {"id": "bucket_price", "type": "bucket", "scale_exp": 4,
         "source": None, "refs": ["std_price"],
         "params": {"edges": [-10000, -3000, 0, 3000, 10000]}},
        {"id": "log_size", "type": "log", "scale_exp": 5,
         "source": "src_size", "refs": [],
         "params": {"base_flag": 0}},
        {"id": "interact_pv", "type": "product", "scale_exp": 4,
         "source": None, "refs": ["std_price", "std_vol"], "params": {}},
        {"id": "clip_interact", "type": "truncate", "scale_exp": 4,
         "source": None, "refs": ["interact_pv"],
         "params": {"lower": -50000, "upper": 50000}},
    ]
    thresholds = {s["id"]: 1500 for s in stages}
    doc = {
        "version": version,
        "session_scale_exp": session_scale_exp,
        "block_size": 37,
        "sources": sources,
        "window": {"ref_start": 0, "ref_end": split,
                   "obs_start": split, "obs_end": rows},
        "stages": stages,
        "thresholds_bps": thresholds,
    }
    load_spec(doc)  # self-validate before writing
    return doc


def write_sample(outdir: str, rows: int = 240, version: int = 1) -> str:
    os.makedirs(outdir, exist_ok=True)
    doc = build_sample_doc(rows=rows, version=version)
    path = os.path.join(outdir, "pipeline.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(doc))
    return path
