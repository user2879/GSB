"""固定种子样本生成器（受控词表，控制长度）。

使用确定性 LCG 随机源（非 numpy/非全局 random），保证任何机器、任何次
运行生成完全相同的偏好对、配置与命令流。
"""
from __future__ import annotations

import json
import os
from typing import Any

import yaml

from .vocab import connector_words, slot_words

SEED = 20261124
LCG_A = 6364136223846793005
LCG_C = 1442695040888963407
LCG_M = 1 << 64


class DeterministicRng:
    def __init__(self, seed: int = SEED) -> None:
        self.state = seed & (LCG_M - 1)

    def next_u64(self) -> int:
        self.state = (LCG_A * self.state + LCG_C) & (LCG_M - 1)
        return self.state

    def below(self, n: int) -> int:
        return self.next_u64() % n

    def pick(self, seq):
        return seq[self.below(len(seq))]


def _make_sentence(rng: DeterministicRng, slot: str) -> str:
    words = slot_words(slot)
    connectors = connector_words()
    length = 3 + rng.below(4)  # 3..6 个受控词，控制长度
    tokens = []
    for i in range(length):
        if i % 2 == 1:
            tokens.append(rng.pick(connectors))
        else:
            tokens.append(rng.pick(words))
    if len(tokens) < 3:  # 保证非空且至少含一个槽位词
        tokens.insert(0, rng.pick(words))
    return " ".join(tokens)


def generate_pairs(rng: DeterministicRng, n_per_batch: int = 6,
                   batches: tuple[str, ...] = ("batch-0", "batch-1")) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    counter = 0
    base_ts = 1_700_000_000_000  # 固定基准毫秒时间戳
    annotators = ("ann-alpha", "ann-beta", "ann-gamma")
    for batch_id in batches:
        for _ in range(n_per_batch):
            counter += 1
            records.append({
                "pair_id": f"pair-{counter:04d}",
                "prompt": _make_sentence(rng, "prompt"),
                "chosen": _make_sentence(rng, "chosen"),
                "rejected": _make_sentence(rng, "rejected"),
                "annotator_id": rng.pick(annotators),
                "event_ms": base_ts + counter * 350 + rng.below(200),
                "batch_id": batch_id,
            })
    return records


def default_config() -> dict[str, Any]:
    return {
        "tokenizer_version": "v1",
        "normalize": True,
        "feature_dim": 64,
        "hash_buckets": 32,
        "weights": {"prompt": 0.25, "chosen": 1.0, "rejected": 0.75},
        "boundary_threshold": 0.5,
        "decimal_places": 6,
        "quant_mode": "half_even",
        "batch_size": 4,
        "learning_rate": 0.01,
        "global_pair_budget": 9,
        "per_batch_quota": 4,
        "max_workers": 4,
        "snapshot_commit_pairs": 4,
        "time_partition_ms": 1000,
    }


def generate_commands() -> list[dict[str, Any]]:
    """覆盖替换/删除/权重/阈值/档位/子词版本/批次/配额/预算的命令序列。"""
    rng = DeterministicRng(SEED ^ 0x9E3779B9)
    replacement = {
        "pair_id": "pair-0002",
        "prompt": "summarize the document for request",
        "chosen": "accurate and concise summary output",
        "rejected": "verbose filler repeat noise error",
        "annotator_id": "ann-beta",
        "event_ms": 1700000001000,
        "batch_id": "batch-0",
    }
    return [
        {"op": "replace_pair", "payload": {"record": replacement}},
        {"op": "set_weight", "payload": {"name": "chosen", "value": 1.25}},
        {"op": "set_boundary_threshold", "payload": {"value": 0.4}},
        {"op": "set_quant_mode", "payload": {"value": "trunc"}},
        {"op": "set_decimal_places", "payload": {"value": 4}},
        {"op": "delete_pair", "payload": {"pair_id": "pair-0005"}},
        {"op": "set_batch", "payload": {"pair_id": "pair-0007", "batch_id": "batch-0"}},
        {"op": "set_tokenizer_version", "payload": {"value": "v2"}},
        {"op": "set_per_batch_quota", "payload": {"value": 5}},
        {"op": "set_global_budget", "payload": {"value": 12}},
        {"op": "set_tokenizer_version", "payload": {"value": "v1"}},
        {"op": "set_quant_mode", "payload": {"value": "half_even"}},
        {"op": "set_decimal_places", "payload": {"value": 6}},
        # 末尾放大配额与预算，保证最终状态下全部存活对都被处理，数据产物与全量重建收敛
        {"op": "set_per_batch_quota", "payload": {"value": 8}},
        {"op": "set_global_budget", "payload": {"value": 20}},
    ]


def write_workspace(root: str) -> dict[str, str]:
    rng = DeterministicRng(SEED)
    pairs = generate_pairs(rng)
    cfg = default_config()
    commands = generate_commands()

    paths = {
        "pairs": os.path.join(root, "samples", "preference_pairs.jsonl"),
        "config": os.path.join(root, "config", "reward_config.yaml"),
        "commands": os.path.join(root, "commands", "command_stream.json"),
    }
    os.makedirs(os.path.dirname(paths["pairs"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["config"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["commands"]), exist_ok=True)

    with open(paths["pairs"], "w", encoding="utf-8", newline="\n") as fh:
        for rec in pairs:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    with open(paths["config"], "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=True)
    with open(paths["commands"], "w", encoding="utf-8", newline="\n") as fh:
        json.dump(commands, fh, ensure_ascii=False, indent=2)

    # 额外的非法样本文件，用于 E_PAIR_INVALID 演示（不影响主流水线）
    bad_path = os.path.join(root, "samples", "preference_pairs_invalid.jsonl")
    bad_records = [
        # 空提示词
        {"pair_id": "bad-1", "prompt": "", "chosen": "accurate output",
         "rejected": "verbose noise", "annotator_id": "ann-alpha", "event_ms": 1},
        # 负时间戳
        {"pair_id": "bad-2", "prompt": "summarize document",
         "chosen": "accurate output", "rejected": "verbose noise",
         "annotator_id": "ann-alpha", "event_ms": -5},
        # 受控词表外的词项 OUTSIDEWORD
        {"pair_id": "bad-3", "prompt": "summarize OUTSIDEWORD document",
         "chosen": "accurate output", "rejected": "verbose noise",
         "annotator_id": "ann-alpha", "event_ms": 2},
        # 重复 pair_id（与下一行同 ID）
        {"pair_id": "bad-dup", "prompt": "summarize document",
         "chosen": "accurate output", "rejected": "verbose noise",
         "annotator_id": "ann-alpha", "event_ms": 3},
        {"pair_id": "bad-dup", "prompt": "translate document",
         "chosen": "concise output", "rejected": "vague noise",
         "annotator_id": "ann-beta", "event_ms": 4},
    ]
    with open(bad_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("not-a-json-line\n")
        for rec in bad_records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    paths["invalid_pairs"] = bad_path
    return paths
