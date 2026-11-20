"""固定种子确定性生成样本输入、交互命令流与故障注入样例。

用法: python -m scripts.generate_samples --out samples/sample_input
不依赖随机网络资源；NumPy 仅用于可复现的样本数据生成。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 20261120
NUM_REQUESTS = 6
ITEMS_PER_REQUEST = 14
RECALL_CHANNELS = ["ann", "inverted", "hot"]
DENSE_DIM = 4
SPARSE_DIM = 3


def _clean(value: float, decimals: int = 6) -> float:
    rounded = round(float(value), decimals)
    if not np.isfinite(rounded):
        raise ValueError("样本生成出现非有限值")
    return rounded


def generate_candidates(out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    path = out_dir / "candidates.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for request_index in range(NUM_REQUESTS):
            request_id = f"req-{request_index:03d}"
            for item_index in range(ITEMS_PER_REQUEST):
                item_id = f"item-{request_index:03d}-{item_index:03d}"
                base = rng.normal(0, 1)
                recall = {
                    channel: _clean(base * 0.6 + rng.normal(0, 1))
                    for channel in RECALL_CHANNELS
                }
                # 刻意制造若干高相似候选，触发窗口内折减
                dense = [
                    _clean(base + rng.normal(0, 0.4)),
                    _clean(rng.normal(0, 1)),
                    _clean(rng.normal(5, 2)),
                    _clean(rng.normal(-3, 1.2)),
                ]
                if item_index in (1, 2):
                    dense[0] = _clean(base + 0.02 * item_index)
                sparse = [int(rng.integers(0, 8)) for _ in range(SPARSE_DIM)]
                record = {
                    "request_id": request_id,
                    "item_id": item_id,
                    "recall_scores": recall,
                    "dense": dense,
                    "sparse": sparse,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_config(out_dir: Path) -> None:
    feature_dim = len(RECALL_CHANNELS) + DENSE_DIM + SPARSE_DIM
    config = {
        "recall_channels": RECALL_CHANNELS,
        "dense_dim": DENSE_DIM,
        "sparse_dim": SPARSE_DIM,
        "weights": [0.30, 0.20, 0.15, 0.12, 0.08, 0.06, 0.04, 0.03, 0.01, 0.01],
        "normalization": "minmax",
        "fusion_temperature": 1.2,
        "score_clip": [-50, 50],
        "frac_bits": 4,
        "quant_levels": 256,
        "rerank_window": 6,
        "diversity_penalty": 0.25,
        "similarity_threshold": 0.9,
        "latency_budget_ms": 200,
        "concurrency": 3,
        "snapshot_every_commands": 2,
    }
    assert len(config["weights"]) == feature_dim
    (out_dir / "rerank_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    groups = {
        "groups": [
            {"name": "recall_group", "feature_indexes": [0, 1, 2], "conflict": False},
            {"name": "dense_group", "feature_indexes": [3, 4, 5, 6], "conflict": True},
            {"name": "sparse_group", "feature_indexes": [7, 8, 9], "conflict": True},
        ]
    }
    (out_dir / "feature_groups.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def generate_commands(out_dir: Path) -> None:
    commands = [
        {"type": "set_weight", "payload": {"feature_index": 3, "weight": 0.22}},
        {"type": "set_rerank_window", "payload": {"window": 4}},
        {"type": "set_quant", "payload": {"frac_bits": 6, "quant_levels": 128}},
        {"type": "set_penalty", "payload": {"penalty": 0.4}},
        {"type": "set_normalization", "payload": {"normalization": "zscore"}},
        {
            "type": "add_candidate",
            "payload": {
                "request_id": "req-001",
                "line_no": 9001,
                "record": {
                    "request_id": "req-001",
                    "item_id": "item-001-extra",
                    "recall_scores": {"ann": 1.25, "inverted": -0.4, "hot": 0.8},
                    "dense": [0.9, -1.1, 4.2, -2.0],
                    "sparse": [2, 0, 5],
                },
            },
        },
        {"type": "delete_candidate",
         "payload": {"request_id": "req-002", "item_id": "item-002-003"}},
        {"type": "set_quant", "payload": {"frac_bits": 3, "quant_levels": 64}},
        {"type": "set_rerank_window", "payload": {"window": 8}},
    ]
    (out_dir / "commands.json").write_text(
        json.dumps({"commands": commands}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate_fault_injection(base_dir: Path) -> None:
    out_dir = base_dir / "fault_injection"
    out_dir.mkdir(parents=True, exist_ok=True)

    bad_records = [
        {
            "request_id": "req-bad-1",
            "item_id": "nan-item",
            "recall_scores": {"ann": float("nan"), "inverted": 0.1, "hot": 0.2},
            "dense": [0.1, 0.2, 0.3, 0.4],
            "sparse": [1, 2, 3],
        },
        {
            "request_id": "req-bad-1",
            "item_id": "inf-item",
            "recall_scores": {"ann": 0.1, "inverted": float("inf"), "hot": 0.2},
            "dense": [0.1, 0.2, 0.3, 0.4],
            "sparse": [1, 2, 3],
        },
        {
            "request_id": "req-bad-2",
            "item_id": "neg-sparse",
            "recall_scores": {"ann": 0.1, "inverted": 0.2, "hot": 0.3},
            "dense": [0.1, 0.2, 0.3, 0.4],
            "sparse": [1, -2, 3],
        },
        {
            "request_id": "",
            "item_id": "empty-request",
            "recall_scores": {"ann": 0.1, "inverted": 0.2, "hot": 0.3},
            "dense": [0.1, 0.2, 0.3, 0.4],
            "sparse": [1, 2, 3],
        },
        {
            "request_id": "req-bad-3",
            "item_id": "dup-item",
            "recall_scores": {"ann": 0.1, "inverted": 0.2, "hot": 0.3},
            "dense": [0.1, 0.2, 0.3, 0.4],
            "sparse": [1, 2, 3],
        },
        {
            "request_id": "req-bad-3",
            "item_id": "dup-item",
            "recall_scores": {"ann": 0.9, "inverted": 0.8, "hot": 0.7},
            "dense": [0.9, 0.8, 0.7, 0.6],
            "sparse": [0, 0, 0],
        },
    ]
    with (out_dir / "bad_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for record in bad_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.write('{"request_id": "req-bad-4", broken json line\n')

    bad_weight_config = {
        "recall_channels": RECALL_CHANNELS,
        "dense_dim": DENSE_DIM,
        "sparse_dim": SPARSE_DIM,
        "weights": [0.1, 0.2],  # 长度不匹配
        "normalization": "minmax",
        "fusion_temperature": 1,
        "score_clip": [-10, 10],
        "frac_bits": 4,
        "quant_levels": 256,
        "rerank_window": 3,
        "diversity_penalty": 0.2,
        "similarity_threshold": 0.8,
        "latency_budget_ms": 100,
        "concurrency": 2,
        "snapshot_every_commands": 1,
    }
    bad_bits_config = dict(bad_weight_config)
    bad_bits_config["weights"] = [0.1] * (len(RECALL_CHANNELS) + DENSE_DIM + SPARSE_DIM)
    bad_bits_config["frac_bits"] = 99  # 位宽越界

    (out_dir / "bad_weights_config.json").write_text(
        json.dumps(bad_weight_config, indent=2), encoding="utf-8"
    )
    (out_dir / "bad_bits_config.json").write_text(
        json.dumps(bad_bits_config, indent=2), encoding="utf-8"
    )

    groups = {
        "groups": [
            {"name": "recall_group", "feature_indexes": [0, 1, 2], "conflict": False},
            {"name": "dense_group", "feature_indexes": [3, 4, 5, 6], "conflict": True},
            {"name": "sparse_group", "feature_indexes": [7, 8, 9], "conflict": True},
        ]
    }
    (out_dir / "feature_groups.json").write_text(
        json.dumps(groups, indent=2), encoding="utf-8"
    )

    # 相似度恰好等于门限的确定性样本：两条候选归一化后方向完全一致
    threshold_records = [
        {
            "request_id": "req-threshold",
            "item_id": "t-a",
            "recall_scores": {"ann": 0.5, "inverted": 0.5, "hot": 0.5},
            "dense": [1.0, 1.0, 1.0, 1.0],
            "sparse": [1, 1, 1],
        },
        {
            "request_id": "req-threshold",
            "item_id": "t-b",
            "recall_scores": {"ann": 0.5, "inverted": 0.5, "hot": 0.5},
            "dense": [1.0, 1.0, 1.0, 1.0],
            "sparse": [1, 1, 1],
        },
        {
            "request_id": "req-threshold",
            "item_id": "t-c",
            "recall_scores": {"ann": -0.5, "inverted": -0.2, "hot": 0.0},
            "dense": [-1.0, 0.4, -2.0, 0.8],
            "sparse": [3, 0, 2],
        },
        {
            "request_id": "req-threshold",
            "item_id": "t-d",
            "recall_scores": {"ann": 0.1, "inverted": 0.9, "hot": -0.3},
            "dense": [0.2, -1.5, 2.2, -0.6],
            "sparse": [0, 4, 1],
        },
    ]
    with (out_dir / "threshold_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for record in threshold_records:
            handle.write(json.dumps(record) + "\n")

    threshold_config = dict(bad_bits_config)
    threshold_config["frac_bits"] = 4
    threshold_config["similarity_threshold"] = 1.0
    threshold_config["rerank_window"] = 4
    threshold_config["diversity_penalty"] = 0.3
    (out_dir / "threshold_config.json").write_text(
        json.dumps(threshold_config, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples/sample_input")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_candidates(out_dir)
    generate_config(out_dir)
    generate_commands(out_dir)
    generate_fault_injection(Path("samples"))
    print(f"样本已生成到 {out_dir} 与 samples/fault_injection")


if __name__ == "__main__":
    main()
