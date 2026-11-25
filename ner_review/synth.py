"""Deterministic local Chinese corpus and small model package synthesis."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .config import RuntimeConfig
from .io_utils import write_json, write_jsonl
from .labels import LABELS, TYPE_CODES


CONTROLLED_CHARS = list("我你他去来看听读写说讲走跑游玩吃喝买卖上下班" "街巷桥园河湖山海天地年月日时候朋友同事老师学生")


SEED_LEXICON: tuple[tuple[str, str], ...] = (
    ("LOC", "北京"),
    ("LOC", "上海"),
    ("LOC", "西安"),
    ("LOC", "杭州"),
    ("LOC", "西湖"),
    ("LOC", "南京"),
    ("PER", "王伟"),
    ("PER", "李娜"),
    ("PER", "张敏"),
    ("PER", "欧阳峰"),
    ("PER", "司马芳"),
    ("ORG", "清华大学"),
    ("ORG", "北京大学"),
    ("ORG", "华为公司"),
    ("ORG", "西湖大学"),
    ("ORG", "鼓楼医院"),
    ("TIME", "二零二六年"),
    ("TIME", "三月"),
    ("TIME", "十五日"),
    ("TIME", "星期一"),
    ("TIME", "早晨"),
)


def build_lexicon() -> list[dict[str, str]]:
    return [
        {"id": f"lex-{entity_type.lower()}-{index:03d}", "text": text, "type": entity_type}
        for index, (entity_type, text) in enumerate(SEED_LEXICON)
    ]


def _append_entity(chars: list[str], labels: list[str], text: str, entity_type: str) -> None:
    entity_chars = list(text)
    if len(entity_chars) == 1:
        chars.extend(entity_chars)
        labels.append(f"S-{entity_type}")
        return
    for offset, char in enumerate(entity_chars):
        chars.append(char)
        if offset == 0:
            prefix = "B"
        elif offset == len(entity_chars) - 1:
            prefix = "E"
        else:
            prefix = "I"
        labels.append(f"{prefix}-{entity_type}")


def make_sample(seed: int, index: int, lexicon: list[dict[str, str]] | None = None) -> dict[str, Any]:
    rng = random.Random((seed * 1_000_003) + index * 97 + 17)
    lexicon = lexicon or build_lexicon()
    chars: list[str] = []
    labels: list[str] = []
    block_count = rng.randint(4, 7)
    for block in range(block_count):
        if block > 0 or rng.random() < 0.7:
            filler_length = rng.randint(1, 3)
            for _ in range(filler_length):
                chars.append(rng.choice(CONTROLLED_CHARS))
                labels.append("O")
        entry = rng.choice(lexicon)
        _append_entity(chars, labels, entry["text"], entry["type"])
    if len(chars) > 48:
        chars = chars[:48]
        labels = labels[:48]
        if labels[-1] not in {"O"} and not labels[-1].startswith(("E-", "S-")):
            entity_type = labels[-1].split("-", 1)[1]
            labels[-1] = f"E-{entity_type}"
    return {
        "id": f"S{index:04d}",
        "char_sequence": " ".join(chars),
        "label_sequence": " ".join(labels),
    }


def build_corpus(config: RuntimeConfig | None = None) -> list[dict[str, Any]]:
    config = config or RuntimeConfig()
    lexicon = build_lexicon()
    return [make_sample(config.seed, index, lexicon) for index in range(1, config.corpus_size + 1)]


def _structurally_likely(from_label: str, to_label: str) -> bool:
    if from_label == "O":
        return to_label == "O" or to_label.startswith(("B-", "S-"))
    if from_label.startswith("B-"):
        entity_type = from_label.split("-", 1)[1]
        return to_label in {f"I-{entity_type}", f"E-{entity_type}"}
    if from_label.startswith("I-"):
        entity_type = from_label.split("-", 1)[1]
        return to_label in {f"I-{entity_type}", f"E-{entity_type}"}
    if from_label.startswith(("E-", "S-")):
        return to_label == "O" or to_label.startswith(("B-", "S-"))
    return False


def build_package(config: RuntimeConfig | None = None) -> dict[str, Any]:
    config = config or RuntimeConfig()
    rng = random.Random(config.seed + 777)
    feature_dim = config.feature_dim
    bucket_count = config.bucket_count
    hash_table = [
        [round(rng.uniform(-0.5, 0.5), 6) for _ in range(feature_dim)]
        for _ in range(bucket_count)
    ]
    emission_weights = []
    for label_index, label in enumerate(LABELS):
        feature_rows = []
        for feature_index in range(feature_dim):
            bucket_row = []
            for bucket in range(bucket_count):
                value = round(rng.uniform(-0.08, 0.08) + 0.01 * ((label_index + feature_index + bucket) % 3), 6)
                bucket_row.append(value)
            feature_rows.append(bucket_row)
        emission_weights.append(feature_rows)

    matrix = []
    for from_label in LABELS:
        row = []
        for to_label in LABELS:
            value = 1.15 if _structurally_likely(from_label, to_label) else -1.35
            value += round(rng.uniform(-0.08, 0.08), 6)
            row.append(value)
        matrix.append(row)
    start = [
        0.35 if label == "O" or label.startswith(("B-", "S-")) else -2.0
        for label in LABELS
    ]
    end = [
        0.35 if label == "O" or label.startswith(("E-", "S-")) else -2.0
        for label in LABELS
    ]
    return {
        "model_version": config.model_version,
        "schema": "ner-local-package/v1",
        "feature_dim": feature_dim,
        "bucket_count": bucket_count,
        "entity_types": list(TYPE_CODES),
        "label_encoding": list(LABELS),
        "hash_algorithm": "blake2b-64",
        "hash_table": hash_table,
        "emission_weights": emission_weights,
        "transition_weights": {"start": start, "end": end, "matrix": matrix},
    }


def build_command_scenario() -> list[dict[str, Any]]:
    return [
        {
            "op": "upsert_lexicon",
            "entry": {"id": "lex-custom-001", "text": "苏州", "type": "LOC"},
        },
        {
            "op": "replace_sample",
            "sample": {
                "id": "S0002",
                "char_sequence": "我 去 北 京 大 学 看 欧 阳 峰",
                "label_sequence": "O O B-LOC E-LOC B-ORG I-ORG E-ORG B-PER I-PER E-PER",
            },
        },
        {"op": "set_emission_weight", "label": "B-LOC", "feature_index": 0, "bucket": 3, "value": 0.42},
        {"op": "set_transition_weight", "from": "B-LOC", "to": "E-LOC", "value": 2.5},
        {"op": "set_quant", "decimal_bits": 8, "quant_level": 1},
        {"op": "upsert_lexicon", "entry": {"id": "lex-loc-000", "text": "北京市", "type": "LOC"}},
        {"op": "remove_lexicon", "entry_id": "lex-custom-001"},
        {"op": "set_quant", "decimal_bits": 12, "quant_level": 2},
    ]


def write_demo_bundle(root: str | Path, config: RuntimeConfig | None = None) -> dict[str, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = config or RuntimeConfig()
    lexicon_path = root / "lexicon.json"
    corpus_path = root / "corpus.jsonl"
    package_path = root / "model_params.json"
    commands_path = root / "command_stream.json"
    injections_path = root / "injection_manifest.json"
    lexicon = build_lexicon()
    write_json(lexicon_path, {"entries": lexicon})
    write_jsonl(corpus_path, build_corpus(config))
    write_json(package_path, build_package(config))
    write_json(commands_path, {"commands": build_command_scenario()})
    write_json(
        injections_path,
        {
            "injections": [
                {"id": "inj-truncated-package", "kind": "truncate_package", "truncate_bytes": 96},
                {"id": "inj-tag-length", "kind": "bad_tag_length", "sample_index": 0},
                {"id": "inj-unknown-tag", "kind": "unknown_tag", "sample_index": 1},
                {"id": "inj-duplicate-id", "kind": "duplicate_id", "sample_index": 2},
                {"id": "inj-feature-dim", "kind": "feature_dim_out_of_range"},
                {"id": "inj-nonfinite", "kind": "non_finite_weight"},
            ]
        },
    )
    return {
        "lexicon": lexicon_path,
        "corpus": corpus_path,
        "package": package_path,
        "commands": commands_path,
        "injections": injections_path,
    }
