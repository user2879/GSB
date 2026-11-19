"""固定种子的确定性本地样本生成。

产物：句子 JSONL、转移解析模型参数 JSON、关系映射表 JSON、v1 定宽文本、v2 JSON。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .codecs import v1_serialize_document, v2_serialize_document
from .config import AppConfig
from .models import Arc, Sentence, Word
from .relations import load_mapping

RELATION_V1_TO_V2 = {
    "SBV": "nsubj",
    "VOB": "obj",
    "POB": "pobj",
    "ADV": "advmod",
    "ATT": "amod",
    "COO": "conj",
    "RAD": "mark",
    "HED": "root",
}

WORD_POOL = [
    ("经济", "n"), ("发展", "vn"), ("迅速", "adj"), ("市场", "n"), ("持续", "adv"),
    ("推动", "v"), ("技术", "n"), ("创新", "vn"), ("我们", "pron"), ("努力", "adv"),
    ("提升", "v"), ("产品", "n"), ("质量", "n"), ("的", "u"), ("和", "c"),
    ("在", "p"), ("公司", "n"), ("年", "q"), ("实现", "v"), ("增长", "vn"),
]


def _random_tree(rng: random.Random, n: int) -> list[int]:
    """生成一棵有根有向树：heads[i] 为词 i+1 的头词，0 表示根。"""
    order = list(range(1, n + 1))
    rng.shuffle(order)
    root = order[0]
    heads = [0] * n
    connected = {root}
    remaining = order[1:]
    for node in remaining:
        parent = rng.choice(sorted(connected))
        heads[node - 1] = parent
        connected.add(node)
    return heads


def generate_samples(base_dir: str | Path, *, seed: int = 20241120, n_sentences: int = 8) -> dict[str, str]:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # 1) 句子 JSONL
    sentences: list[Sentence] = []
    jsonl_path = base / "sentences.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for s in range(1, n_sentences + 1):
            n = rng.randint(5, 10)
            picks = [rng.choice(WORD_POOL) for _ in range(n)]
            tokens = [f"{form}/{pos}" for form, pos in picks]
            sid = f"S{s:03d}"
            fh.write(json.dumps({"sent_id": sid, "tokens": " ".join(tokens)}, ensure_ascii=False) + "\n")

            heads = _random_tree(rng, n)
            words = [Word(idx=i + 1, form=picks[i][0], pos=picks[i][1]) for i in range(n)]
            codes = list(RELATION_V1_TO_V2.keys())
            arcs: list[Arc] = []
            for i in range(n):
                head = heads[i]
                code = "HED" if head == 0 else rng.choice([c for c in codes if c != "HED"])
                named = RELATION_V1_TO_V2[code]
                permille = rng.randint(500, 1000)
                arcs.append(
                    Arc(
                        word_idx=i + 1,
                        head=head,
                        relation=named,
                        score=permille / 1000.0,
                        precision="permille",
                        source_version="v1",
                        raw={"relation_code": code, "permille": permille},
                        referenced_words=[i + 1] + ([head] if head else []),
                    )
                )
            sentences.append(Sentence(sent_id=sid, words=words, arcs=arcs, source_version="v1"))

    # 2) 模型参数 JSON（移进/左归约/右归约/结束四类动作权重 + 哈希表 + 维度 + 桶数）
    feature_dim = 64
    buckets = 256
    model_params = {
        "actions": ["shift", "left_reduce", "right_reduce", "terminate"],
        "weights": {
            action: [round(rng.uniform(-1.0, 1.0), 6) for _ in range(feature_dim)]
            for action in ["shift", "left_reduce", "right_reduce", "terminate"]
        },
        "word_feature_hash": {
            word: rng.randrange(buckets) for word, _ in WORD_POOL
        },
        "pos_feature_hash": {
            pos: rng.randrange(buckets) for _, pos in WORD_POOL
        },
        "feature_dim": feature_dim,
        "hash_buckets": buckets,
        "seed": seed,
    }
    model_path = base / "model_params.json"
    model_path.write_text(
        json.dumps(model_params, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    # 3) 关系映射表
    mapping_path = base / "relation_map.json"
    mapping_path.write_text(
        json.dumps({"v1_to_v2": RELATION_V1_TO_V2}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    cfg = AppConfig()
    mapping = load_mapping(mapping_path)

    # 4) v1 定宽文本
    v1_path = base / "parse_v1.txt"
    v1_text, _ = v1_serialize_document(sentences, cfg, mapping["v2_to_v1"])
    v1_path.write_text(v1_text, encoding="utf-8")

    # 5) v2 JSON + 旁路元数据
    v2_path = base / "parse_v2.json"
    v2_doc, sidecar = v2_serialize_document(sentences, cfg)
    v2_path.write_text(
        json.dumps(v2_doc, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    sidecar_path = base / "parse_v2_sidecar.json"
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    return {
        "sentences": str(jsonl_path),
        "model": str(model_path),
        "mapping": str(mapping_path),
        "v1": str(v1_path),
        "v2": str(v2_path),
        "sidecar": str(sidecar_path),
    }
