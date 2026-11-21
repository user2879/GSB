from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .config import sha256_file
from .types import ENTITY_TYPES, required_tags


@dataclass(frozen=True)
class TemplatePart:
    kind: str
    value: str
    entity_type: str | None = None


PERSON_NAMES = ("张伟", "王芳", "李明", "赵敏", "陈晨", "刘洋", "周宁", "吴磊", "孙悦", "郑华")
LOCATION_NAMES = ("北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉", "长安", "江南")
ORGANIZATION_NAMES = ("清华大学", "北京大学", "科学院", "研究院", "交通局", "实验室", "科技公司", "数据中心")
TIME_NAMES = ("2024年", "2025年", "三月", "五月", "上午", "下午", "今天", "昨日", "本周", "下周")

FILLERS = (
    "在",
    "于",
    "与",
    "和",
    "从",
    "到",
    "的",
    "了",
    "将",
    "对",
    "表示",
    "发布",
    "访问",
    "位于",
    "参加",
    "会议",
    "项目",
    "团队",
    "记者",
    "消息",
    "计划",
    "正式",
    "共同",
    "完成",
    "调研",
    "报告",
)


def _literal(text: str) -> TemplatePart:
    return TemplatePart("literal", text)


def _entity(entity_type: str, text: str) -> TemplatePart:
    return TemplatePart("entity", text, entity_type)


def sample_templates(rng: random.Random) -> list[list[TemplatePart]]:
    person = rng.choice(PERSON_NAMES)
    location = rng.choice(LOCATION_NAMES)
    organization = rng.choice(ORGANIZATION_NAMES)
    time_value = rng.choice(TIME_NAMES)
    extra_verb = rng.choice(FILLERS)
    extra_noun = rng.choice(("项目", "会议", "报告", "团队", "消息"))
    return [
        [
            _entity("人物", person),
            _literal("在"),
            _entity("时间", time_value),
            _literal("访问了"),
            _entity("地点", location),
            _literal("的"),
            _entity("机构", organization),
            _literal("。"),
        ],
        [
            _literal("据"),
            _entity("机构", organization),
            _literal("消息，"),
            _entity("人物", person),
            _literal("于"),
            _entity("时间", time_value),
            _literal("在"),
            _entity("地点", location),
            _literal(f"{extra_verb}{extra_noun}。"),
        ],
        [
            _entity("机构", organization),
            _literal("与"),
            _entity("机构", rng.choice(ORGANIZATION_NAMES)),
            _literal("在"),
            _entity("地点", location),
            _literal("共同完成"),
            _entity("时间", time_value),
            _literal("调研报告。"),
        ],
        [
            _literal("记者从"),
            _entity("地点", location),
            _literal("获悉，"),
            _entity("人物", person),
            _literal("将参加"),
            _entity("机构", organization),
            _literal("于"),
            _entity("时间", time_value),
            _literal("举办的会议。"),
        ],
        [
            _entity("时间", time_value),
            _literal("，"),
            _entity("机构", organization),
            _literal("位于"),
            _entity("地点", location),
            _literal("的团队发布了"),
            _entity("人物", person),
            _literal("负责的项目报告。"),
        ],
    ]


def materialize_template(parts: Sequence[TemplatePart]) -> tuple[list[str], list[dict[str, object]]]:
    chars: list[str] = []
    entities: list[dict[str, object]] = []
    for part in parts:
        start = len(chars)
        chars.extend(part.value)
        if part.kind == "entity":
            end = len(chars) - 1
            entities.append(
                {
                    "start": start,
                    "end": end,
                    "type": part.entity_type,
                    "text": part.value,
                }
            )
    return chars, entities


def labels_from_entities(length: int, entities: Sequence[dict[str, object]]) -> list[str]:
    labels = ["O"] * length
    for entity in entities:
        start = int(entity["start"])
        end = int(entity["end"])
        entity_type = str(entity["type"])
        span = end - start + 1
        if span == 1:
            labels[start] = f"S-{entity_type}"
        else:
            labels[start] = f"B-{entity_type}"
            for position in range(start + 1, end):
                labels[position] = f"I-{entity_type}"
            labels[end] = f"E-{entity_type}"
    return labels


def _round6(value: float) -> float:
    return round(value, 6)


def _uniform(rng: random.Random, low: float, high: float) -> float:
    return _round6(rng.uniform(low, high))


def _transition_bias(previous_label: str, current_label: str) -> float:
    if previous_label == "O":
        symbol = current_label.split("-", 1)[0]
        return {"O": 0.22, "B": 0.14, "S": 0.12, "I": -0.72, "E": -0.68}[symbol]
    if current_label == "O":
        return 0.22
    prev_symbol, prev_type = previous_label.split("-", 1)
    cur_symbol, cur_type = current_label.split("-", 1)
    same_type = prev_type == cur_type
    if prev_symbol == "B":
        if cur_symbol == "I" and same_type:
            return 0.24
        if cur_symbol == "E" and same_type:
            return 0.20
        return -0.46 if cur_symbol in {"B", "S", "O"} else -0.58
    if prev_symbol == "I":
        if cur_symbol == "I" and same_type:
            return 0.20
        if cur_symbol == "E" and same_type:
            return 0.23
        return -0.52
    if prev_symbol in {"E", "S"}:
        if cur_symbol == "O":
            return 0.22
        if cur_symbol in {"B", "S"}:
            return 0.10
        return -0.55
    return -0.2


def build_parameter_package(chars: Iterable[str], vocab: Iterable[str], seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    labels = list(required_tags())
    feature_dim = 6
    hash_buckets = 31
    entity_vocab = sorted(set(vocab))
    max_vocab_length = max((len(word) for word in entity_vocab), default=1)

    controlled_chars = sorted(set(chars) | {"<BOS>", "<EOS>", "<UNK>"})
    char_embeddings = {
        char: [_uniform(rng, -1.0, 1.0) for _ in range(feature_dim)] for char in controlled_chars
    }
    position_embeddings = {
        str(bucket): [_uniform(rng, -1.0, 1.0) for _ in range(feature_dim)] for bucket in range(4)
    }
    vocab_embeddings = {
        str(length): [_uniform(rng, -1.0, 1.0) for _ in range(feature_dim)]
        for length in range(max_vocab_length + 1)
    }

    emission_weights: list[list[list[float]]] = []
    for _label in labels:
        label_buckets = []
        for _bucket in range(hash_buckets):
            label_buckets.append([_uniform(rng, -0.045, 0.045) for _ in range(feature_dim)])
        emission_weights.append(label_buckets)

    transition_weights: list[list[float]] = []
    for previous_label in labels:
        row = []
        for current_label in labels:
            bias = _transition_bias(previous_label, current_label)
            row.append(_round6(bias + _uniform(rng, -0.035, 0.035)))
        transition_weights.append(row)

    start_weights = []
    for label in labels:
        symbol = label.split("-", 1)[0]
        start_weights.append(_round6({"O": 0.18, "B": 0.10, "S": 0.12, "I": -0.66, "E": -0.66}[symbol] + _uniform(rng, -0.02, 0.02)))

    return {
        "model_version": f"hash-ner-v1-seed{seed}",
        "labels": labels,
        "feature_dim": feature_dim,
        "hash_buckets": hash_buckets,
        "hash_salt": f"local-ner-{seed}",
        "char_embeddings": char_embeddings,
        "position_embeddings": position_embeddings,
        "vocab_embeddings": vocab_embeddings,
        "vocab": entity_vocab,
        "emission_weights": emission_weights,
        "emission_weights_layout": "label,bucket,feature_dim",
        "transition_weights": transition_weights,
        "start_weights": start_weights,
        "notes": "All weights are quantized to fixed-point integers by the loader.",
    }


def generate_dataset(
    output_dir: Path,
    seed: int = 20261121,
    sample_count: int = 32,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    all_chars: set[str] = set()
    vocab: set[str] = set(PERSON_NAMES + LOCATION_NAMES + ORGANIZATION_NAMES + TIME_NAMES)

    for index in range(sample_count):
        parts = rng.choice(sample_templates(rng))
        if index % 7 == 0:
            parts = list(parts) + [_literal("相关"), _literal(rng.choice(FILLERS)), _literal("。")]
        chars, entities = materialize_template(parts)
        labels = labels_from_entities(len(chars), entities)
        entity_types = sorted({str(entity["type"]) for entity in entities})
        for entity in entities:
            vocab.add(str(entity["text"]))
        all_chars.update(chars)
        records.append(
            {
                "sample_id": f"SMP-{index + 1:04d}",
                "chars": " ".join(chars),
                "labels": " ".join(labels),
                "entity_types": entity_types,
                "entities": entities,
            }
        )

    corpus_path = output_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    params = build_parameter_package(all_chars, vocab, seed)
    params_path = output_dir / "model_params.json"
    params_path.write_text(json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    total_chars = sum(len(str(record["chars"]).split()) for record in records)
    longest = max(len(str(record["chars"]).split()) for record in records)
    target_sample_ids = [f"SMP-{number:04d}" for number in (5, 11, 17, 23)]
    injections = [
        {
            "injection_id": "INJ-001",
            "sample_id": target_sample_ids[0],
            "stage": "emission",
            "kind": "nonfinite_score",
        },
        {
            "injection_id": "INJ-002",
            "sample_id": target_sample_ids[1],
            "stage": "decode",
            "kind": "truncated_labels",
        },
        {
            "injection_id": "INJ-003",
            "sample_id": target_sample_ids[2],
            "stage": "emission",
            "kind": "hash_bucket_mismatch",
        },
        {
            "injection_id": "INJ-004",
            "sample_id": target_sample_ids[3],
            "stage": "schedule",
            "kind": "quota_lt_min_chars",
        },
    ]
    injections_path = output_dir / "injections.json"
    injections_path.write_text(
        json.dumps({"injections": injections}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config = {
        "snapshot_format_version": 1,
        "processing_unit": "sliding_window",
        "window_chars": 10,
        "window_stride": 8,
        "char_batch_size": 4,
        "max_iterations": 512,
        "fixed_decimals": 4,
        "quantization_level": 6,
        "max_workers": 4,
        "global_char_budget": max(1, int(total_chars * 0.66)),
        "per_sample_quota": max(8, longest),
        "cache_blocks": 5,
        "char_block_size": 4,
        "latency_budget_ms": 250,
        "snapshot_interval_batches": 4,
        "seed": seed,
    }
    config_path = output_dir / "inference.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    manifest = {
        "generator": "ner_workbench.generator",
        "seed": seed,
        "sample_count": sample_count,
        "total_chars": total_chars,
        "longest_sample_chars": longest,
        "global_char_budget": config["global_char_budget"],
        "expected_budget_exhaustion": total_chars > int(config["global_char_budget"]),
        "injection_ids": [item["injection_id"] for item in injections],
        "files": {
            "corpus.jsonl": sha256_file(corpus_path),
            "model_params.json": sha256_file(params_path),
            "inference.yaml": sha256_file(config_path),
            "injections.json": sha256_file(injections_path),
        },
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic local NER fixtures")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args(argv)
    manifest = generate_dataset(args.out, seed=args.seed, sample_count=args.samples)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
