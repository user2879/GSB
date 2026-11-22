from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .io import write_json, write_jsonl_atomic
from .models import ACTION_NAMES


LEXICON: tuple[tuple[str, str], ...] = (
    ("我们", "r"), ("你", "r"), ("他们", "r"), ("学生", "n"), ("老师", "n"),
    ("代码", "n"), ("句子", "n"), ("模型", "n"), ("数据", "n"), ("工作台", "n"),
    ("解析", "v"), ("编辑", "v"), ("运行", "v"), ("恢复", "v"), ("检查", "v"),
    ("生成", "v"), ("保存", "v"), ("读取", "v"), ("喜欢", "v"), ("看见", "v"),
    ("分析", "v"), ("维护", "v"), ("提交", "v"), ("快照", "n"), ("今天", "t"),
    ("昨天", "t"), ("认真", "d"), ("快速", "d"), ("本地", "a"), ("稳定", "a"),
    ("在", "p"), ("向", "p"), ("把", "p"), ("的", "u"), ("和", "c"), ("了", "u"),
    ("吗", "y"), ("不", "d"), ("很", "d"), ("又", "d"), ("中国", "ns"), ("北京", "ns"),
)


FEATURE_DIM = 17
NUM_BUCKETS = 4093


def _feature_keys(vocabulary: list[str], pos_tags: list[str]) -> list[str]:
    keys: list[str] = []
    for word in vocabulary:
        keys.append(f"w:{word}")
    for pos in pos_tags:
        keys.append(f"p:{pos}")
    for action in ACTION_NAMES:
        keys.append(f"a:{action}|bias")
        for word in vocabulary:
            keys.extend((f"a:{action}|w:s0={word}", f"a:{action}|w:b0={word}"))
        for pos in pos_tags:
            keys.extend(
                (
                    f"a:{action}|p:s0={pos}",
                    f"a:{action}|p:s1={pos}",
                    f"a:{action}|p:b0={pos}",
                )
            )
        for left_pos in pos_tags:
            for right_pos in pos_tags:
                keys.append(f"a:{action}|pair={left_pos}_{right_pos}")
    for word in vocabulary:
        for action in ACTION_NAMES:
            keys.append(f"a:{action}|w:s1={word}")
    return sorted(set(keys), key=lambda item: (item.split("|", 1)[0], item))


def _gold_chain(length: int, root: int) -> tuple[list[int], list[str]]:
    heads: list[int] = []
    for position in range(length):
        if position == root:
            heads.append(-1)
        elif position < root:
            heads.append(position + 1)
        else:
            heads.append(position - 1)
    relations: list[str] = []
    for position, head in enumerate(heads):
        if head < 0:
            relations.append("ROOT")
        elif head > position:
            relations.append("L")
        else:
            relations.append("R")
    return heads, relations


def generate_samples(out_dir: str | Path, *, seed: int = 20261123, count: int = 10) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    vocabulary = sorted({word for word, _ in LEXICON})
    pos_tags = sorted({pos for _, pos in LEXICON})
    lexicon_by_pos: dict[str, list[str]] = {}
    for word, pos in LEXICON:
        lexicon_by_pos.setdefault(pos, []).append(word)
    for words in lexicon_by_pos.values():
        words.sort()

    rows: list[dict[str, Any]] = []
    for index in range(count):
        sid = f"S{index + 1:03d}"
        length = 4 + rng.randrange(0, 4)
        tokens: list[tuple[str, str]] = []
        while len(tokens) < length:
            pos = pos_tags[rng.randrange(len(pos_tags))]
            word = lexicon_by_pos[pos][rng.randrange(len(lexicon_by_pos[pos]))]
            if pos in {"u", "y"} and len(tokens) < 2:
                continue
            tokens.append((word, pos))
        words = " ".join(word for word, _ in tokens)
        tags = " ".join(pos for _, pos in tokens)
        root = rng.randrange(length)
        heads, relations = _gold_chain(length, root)
        rows.append(
            {
                "sentence_id": sid,
                "words": words,
                "pos_tags": tags,
                "gold_heads": heads,
                "gold_relations": relations,
            }
        )

    feature_keys = _feature_keys(vocabulary, pos_tags)
    if len(feature_keys) > NUM_BUCKETS:
        raise RuntimeError("featurization does not fit into the declared bucket count")
    buckets = rng.sample(range(NUM_BUCKETS), len(feature_keys))
    feature_hashes = dict(zip(feature_keys, buckets))

    rng_weights = random.Random(seed ^ 0x5EED)
    action_weights: dict[str, list[float]] = {}
    for action_index, action in enumerate(ACTION_NAMES):
        base = 0.28 if action == "SHIFT" else (-0.16 if action == "FINISH" else 0.06)
        weights = [round(rng_weights.uniform(-1.2, 1.2) + base, 6) for _ in range(FEATURE_DIM)]
        # The four action vectors intentionally differ, making both mode
        # comparison and tie handling observable in the generated sample.
        weights[0] = round(base + action_index * 0.07, 6)
        action_weights[action] = weights

    model = {
        "feature_dim": FEATURE_DIM,
        "num_buckets": NUM_BUCKETS,
        "action_weights": action_weights,
        "feature_hashes": feature_hashes,
        "vocabulary": vocabulary,
        "pos_tags": pos_tags,
    }
    config = {
        "mode": "greedy",
        "max_dependency_distance": 7,
        "max_transition_steps": 40,
        "max_threads": 4,
        "batch_parse_point_budget": 80,
        "snapshot_commit_interval_sentences": 2,
        "fixed_point_decimals": 3,
        "quantization_level": 1,
        "constraints": {
            "max_arc_distance": 6,
            "max_stack_size": 9,
            "max_dependents_per_head": 5,
            "forbid_left_dependent_pos": [],
            "forbid_right_dependent_pos": [],
            "finish_requires_single_root": True,
        },
        "max_exhaustive_nodes": 250000,
        "max_pruned_records": 200,
    }
    vocabulary_payload = {
        "words": vocabulary,
        "pos_tags": pos_tags,
        "seed": seed,
    }
    demo_injections = {
        "injections": [
            {
                "injection_id": "INJ-001",
                "sentence_id": "S002",
                "stage": "sentence_load",
                "kind": "empty_words",
            },
            {
                "injection_id": "INJ-002",
                "sentence_id": "S003",
                "stage": "parse",
                "kind": "unknown_pos",
                "token_index": 1,
            },
            {
                "injection_id": "INJ-003",
                "sentence_id": "S004",
                "stage": "parse",
                "kind": "nonfinite_weight",
                "weight_action": "LEFT_ARC",
                "weight_index": 2,
            },
            {
                "injection_id": "INJ-004",
                "sentence_id": "S005",
                "stage": "parse",
                "kind": "zero_max_steps",
            },
        ]
    }
    write_jsonl_atomic(out_path / "sentences.jsonl", rows)
    write_json(out_path / "vocabulary.json", vocabulary_payload)
    write_json(out_path / "model.json", model)
    write_json(out_path / "config.json", config)
    write_json(out_path / "injections.json", {"injections": []})
    write_json(out_path / "injections_demo.json", demo_injections)
    metadata = {
        "seed": seed,
        "sentence_count": count,
        "vocabulary_size": len(vocabulary),
        "pos_count": len(pos_tags),
        "feature_dim": FEATURE_DIM,
        "num_buckets": NUM_BUCKETS,
    }
    write_json(out_path / "generation.json", metadata)
    return metadata
