"""固定种子确定性样本生成器。

产出：
- candidates.jsonl：候选流（请求 ID、通道、物品 ID、原始得分、稠密特征、
  通道内名次、分片序号）；
- fusion_config.json：通道融合配置；
- candidates_invalid.jsonl：若干故意非法的记录，用于演示失败清单不中断加载。
不访问网络、账号或任何附件。
"""
from __future__ import annotations

import json
import os

import numpy as np

SEED = 20261123
CHANNELS = ["lexical", "collaborative", "hotness"]
NORMS = {"lexical": "minmax", "collaborative": "zscore", "hotness": "minmax"}
WEIGHTS = {"lexical": 0.5, "collaborative": 0.35, "hotness": 0.15}
FEATURE_DIM = 4
N_REQUESTS = 6
# 每个请求的候选数：含两个小请求制造近边界/并列候选。
ITEMS_PER_REQUEST = [12, 12, 8, 12, 6, 12]
SHARDS = 2


def _scores(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    if kind == "lexical":
        base = rng.uniform(0.1, 0.95, n)
    elif kind == "collaborative":
        base = rng.normal(0.0, 1.0, n)
    else:
        base = rng.exponential(1.0, n)
    # 制造近边界候选：最后一对得分刻意贴近，量化粗档位下可能翻转。
    if n >= 4:
        base[-1] = base[-2] + 1e-4
    return np.round(base, 6)


def generate(out_dir: str) -> dict:
    rng = np.random.default_rng(SEED)
    os.makedirs(out_dir, exist_ok=True)
    stream_path = os.path.join(out_dir, "candidates.jsonl")
    invalid_path = os.path.join(out_dir, "candidates_invalid.jsonl")
    config_path = os.path.join(out_dir, "fusion_config.json")

    lines = []
    invalid_lines = []
    for req_idx, n_items in enumerate(ITEMS_PER_REQUEST):
        request_id = f"req-{req_idx:03d}"
        item_ids = [f"item-{req_idx:03d}-{k:03d}" for k in range(n_items)]
        # 每个物品一份固定稠密特征（跨通道一致）。
        features = np.round(rng.uniform(-1.0, 1.0,
                                        size=(n_items, FEATURE_DIM)), 6)
        for chan in CHANNELS:
            scores = _scores(rng, n_items, chan)
            order = np.argsort(-scores, kind="stable")
            rank_of = {int(pos): rank for rank, pos in enumerate(order, 1)}
            for k, item_id in enumerate(item_ids):
                record = {
                    "request_id": request_id,
                    "channel": chan,
                    "item_id": item_id,
                    "raw_score": float(scores[k]),
                    "features": [float(v) for v in features[k]],
                    "channel_rank": int(rank_of[k]),
                    "shard": int(k % SHARDS),
                }
                lines.append(json.dumps(record, ensure_ascii=False,
                                        sort_keys=True))

    # 故意非法记录（独立文件，按需混入演示 E_RECORD_INVALID）。
    invalid_lines.append(json.dumps({
        "request_id": "", "channel": "lexical", "item_id": "x",
        "raw_score": 0.5, "features": [0.0] * FEATURE_DIM,
        "channel_rank": 1, "shard": 0}, sort_keys=True))
    invalid_lines.append(json.dumps({
        "request_id": "req-000", "channel": "lexical",
        "item_id": "item-000-000", "raw_score": float("nan"),
        "features": [0.0] * FEATURE_DIM,
        "channel_rank": 99, "shard": 0}, sort_keys=True))
    invalid_lines.append(json.dumps({
        "request_id": "req-000", "channel": "unknown",
        "item_id": "item-000-999", "raw_score": 0.5,
        "features": [0.0] * FEATURE_DIM,
        "channel_rank": 1, "shard": 0}, sort_keys=True))

    with open(stream_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(invalid_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(invalid_lines) + "\n")

    config = {
        "channels": {
            chan: {"weight": WEIGHTS[chan], "normalization": NORMS[chan]}
            for chan in CHANNELS
        },
        "fusion_temperature": 1.0,
        "feature_dim": FEATURE_DIM,
        "feature_weights": [0.25, -0.15, 0.1, 0.05],
        "score_clip": [-4.0, 4.0],
        "frac_bits_choices": [4, 8, 12],
        "gears": [1, 2, 4],
        "tolerance": 1e-6,
        "recompute_budget": 24,
        "concurrency": 4,
        "snapshot_commit_interval_batches": 2,
    }
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")

    return {"stream": stream_path, "invalid": invalid_path,
            "config": config_path, "records": len(lines)}


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "samples", "input")
    info = generate(target)
    print(json.dumps(info, ensure_ascii=False, indent=2))
