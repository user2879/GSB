"""Deterministic local sample generation (fixed seed, no global random)."""
from __future__ import annotations

import json
import os

from .pipeline import HashPRNG

CHANNELS = ["ads", "news", "search", "social"]


def gen_requests(seed="rankacc-samples-v1", n_requests=12):
    rng = HashPRNG(seed)
    requests = []
    for i in range(1, n_requests + 1):
        rid = "req%04d" % i
        n_cand = 6 + rng.randbelow(19)          # 6..24 candidates
        if i == 7:
            n_cand = 30                          # exercises per-request quota
        cands = []
        for cid in range(1, n_cand + 1):
            score = round(rng.randbelow(100000) / 1000.0 + rng.randbelow(50), 3)
            aux = None if rng.randbelow(5) == 0 else rng.randbelow(6)
            ch = CHANNELS[rng.randbelow(len(CHANNELS))]
            cands.append({"candidate_id": cid, "score": score,
                          "aux": aux, "channel": ch})
        requests.append({"request_id": rid, "candidates": cands})
    return requests


def gen_config():
    return {
        "topk": 10,
        "sample_ratio_permille": 500,
        "quant_tier": "Q12",
        "norm_method": "minmax",
        "channel_weights": {"ads": 1.5, "news": 1.0, "search": 0.8, "social": 1.2},
        "output_version": "v2",
        "batch_budget": 40,
        "per_request_quota": 25,
        "threads": 4,
        "tolerance_units": 1,
    }


def gen_edits():
    return [
        {"op": "append_request", "request": {
            "request_id": "req9001",
            "candidates": [
                {"candidate_id": 1, "score": 3.25, "aux": 1, "channel": "ads"},
                {"candidate_id": 2, "score": 9.75, "aux": None, "channel": "news"},
                {"candidate_id": 3, "score": 5.5, "aux": 2, "channel": "ads"},
            ]}},
        {"op": "replace_candidate", "request_id": "req0002",
         "candidate": {"candidate_id": 3, "score": 42.125, "aux": 4, "channel": "ads"}},
        {"op": "set_channel_weight", "channel": "ads", "weight": 2.5},
        {"op": "set_sample_ratio", "permille": 300},
        {"op": "set_topk", "topk": 8},
        {"op": "set_version", "version": "v1"},
        {"op": "set_quant_tier", "tier": "Q16"},
        {"op": "set_version", "version": "v2"},
    ]


def gen_faults():
    return [
        {"request_id": "req0003", "stage": "normalize", "fault": "corrupt_score"},
        {"request_id": "req0005", "stage": "normalize", "fault": "unknown_channel"},
    ]


def config_to_yaml(cfg):
    lines = []
    for key, val in cfg.items():
        if isinstance(val, dict):
            lines.append(key + ":")
            for k, v in val.items():
                lines.append("  %s: %s" % (k, v))
        else:
            lines.append("%s: %s" % (key, val))
    return "\n".join(lines) + "\n"


def write_samples(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "candidates.jsonl"), "w", encoding="utf-8") as fh:
        for req in gen_requests():
            fh.write(json.dumps(req, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write(config_to_yaml(gen_config()))
    with open(os.path.join(out_dir, "edits.jsonl"), "w", encoding="utf-8") as fh:
        for op in gen_edits():
            fh.write(json.dumps(op, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "faults.jsonl"), "w", encoding="utf-8") as fh:
        for f in gen_faults():
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    return out_dir
