from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .config import normalize_config
from .vocab import Vocabulary


CONCEPTS: list[tuple[str, str, str]] = [
    ("C01", "银行", "bank"),
    ("C02", "河流", "river"),
    ("C03", "模型", "model"),
    ("C04", "缓存", "cache"),
    ("C05", "索引", "index"),
    ("C06", "查询", "query"),
    ("C07", "文档", "document"),
    ("C08", "语言", "language"),
    ("C09", "分数", "score"),
    ("C10", "指纹", "fingerprint"),
    ("C11", "快照", "snapshot"),
    ("C12", "预算", "budget"),
    ("C13", "调度器", "scheduler"),
    ("C14", "词桶", "bucket"),
    ("C15", "词项", "term"),
    ("C16", "排序", "ranking"),
    ("C17", "边界", "boundary"),
    ("C18", "整数", "integer"),
    ("C19", "证据", "evidence"),
    ("C20", "融合", "fusion"),
    ("C21", "版本", "version"),
    ("C22", "恢复", "recovery"),
    ("C23", "并发", "concurrency"),
    ("C24", "账本", "ledger"),
]


def build_vocab(seed: int = 360) -> dict[str, Any]:
    terms = []
    for concept_id, original, translation in CONCEPTS:
        terms.append(
            {
                "concept_id": concept_id,
                "forms": {"original": original, "translation": translation},
                "aliases": {
                    "original": [original.casefold()],
                    "translation": [translation.casefold(), f"{translation}_t"],
                },
            }
        )
    return {"seed": seed, "terms": terms}


def build_config() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "views": [
            {"name": "original", "language": "zh-CN"},
            {"name": "translation", "language": "en-US"},
        ],
        "allowed_languages": ["zh-CN", "en-US"],
        "fields": ["title", "body"],
        "field_weights": {"title": 3, "body": 1},
        "field_order": ["title", "body"],
        "top_k": 3,
        "saturation_k": 2,
        "language_weights": {"original": "1.20", "translation": "0.90"},
        "fixed_point": {
            "frac_bits": 16,
            "quant_tier": "nearest",
            "reference_extra_bits": 8,
        },
        "budget": {"batch_points": 40, "query_quota_points": 40},
    }


def _terms(rng: random.Random, count: int, hot_pool: int) -> list[str]:
    result: list[str] = []
    for position in range(count):
        if rng.random() < 0.58:
            index = rng.randrange(hot_pool)
        else:
            index = rng.randrange(len(CONCEPTS))
        result.append(CONCEPTS[index][0])
    return result


def make_document(doc_number: int, version: int | None = None, seed: int = 360) -> dict[str, Any]:
    rng = random.Random(seed + doc_number * 1009)
    doc_id = f"D{doc_number:03d}"
    if version is None:
        version = 1 + (doc_number % 3)
    views: dict[str, Any] = {}
    for view_number, view_name in enumerate(("original", "translation")):
        view_rng = random.Random(seed + doc_number * 1009 + view_number * 7919)
        title = _terms(view_rng, view_rng.randint(2, 4), hot_pool=10)
        body = _terms(view_rng, view_rng.randint(10, 18), hot_pool=14)
        language = "zh-CN" if view_name == "original" else "en-US"
        views[view_name] = {
            "language": language,
            "fields": {"title": title, "body": body},
        }
    return {"doc_id": doc_id, "version": version, "views": views}


def build_documents_jsonl() -> list[str]:
    rows: list[Any] = [make_document(number) for number in range(1, 13)]
    invalid_duplicate = dict(rows[0])
    invalid_duplicate["version"] = 9
    bad_language = make_document(90, version=1)
    bad_language["doc_id"] = "D900"
    bad_language["views"]["translation"]["language"] = "fr-FR"
    bad_term = make_document(91, version=1)
    bad_term["doc_id"] = "D901"
    bad_term["views"]["original"]["fields"]["body"].append("NOT_A_TERM")
    bad_version = make_document(92, version=-2)
    bad_version["doc_id"] = "D902"

    ordered: list[Any] = []
    for index, row in enumerate(rows, start=1):
        ordered.append(row)
        if index == 2:
            ordered.append(invalid_duplicate)
        if index == 5:
            ordered.append(bad_language)
        if index == 8:
            ordered.append(bad_term)
        if index == 11:
            ordered.append(bad_version)
    lines = [json.dumps(item, ensure_ascii=False) for item in ordered]
    lines.insert(6, "{not-json")
    return lines


def build_queries_jsonl() -> list[str]:
    preferences = [
        {"original": "1.35", "translation": "0.70"},
        {"original": "1.10", "translation": "0.95"},
        {"original": "0.85", "translation": "1.30"},
        {"original": "1.00", "translation": "1.00"},
    ]
    rows: list[dict[str, Any]] = []
    for number in range(1, 11):
        rng = random.Random(9000 + number * 131)
        term_count = 2 + (number % 4)
        terms = _terms(rng, term_count, hot_pool=12)
        rows.append(
            {
                "query_id": f"Q{number:02d}",
                "terms": terms,
                "language_preference": preferences[(number - 1) % len(preferences)],
            }
        )
    ordered: list[Any] = []
    for index, row in enumerate(rows, start=1):
        ordered.append(row)
        if index == 3:
            ordered.append({"query_id": "QBAD1", "terms": []})
        if index == 7:
            ordered.append({"query_id": "QBAD2", "terms": ["C01", "NO_SUCH_TERM"]})
    return [json.dumps(item, ensure_ascii=False) for item in ordered]


def build_commands_jsonl() -> list[str]:
    add_13 = make_document(13, version=1)
    add_14 = make_document(14, version=2)
    add_15 = make_document(15, version=1)
    replace_4 = make_document(4, version=7, seed=777)
    replace_7 = make_document(7, version=9, seed=888)
    commands: list[dict[str, Any]] = [
        {"op": "add_document", "document": add_13},
        {"op": "add_document", "document": add_14},
        {"op": "replace_document", "document": replace_4},
        {"op": "set_language_weight", "view": "translation", "weight": "1.25"},
        {"op": "set_field_weight", "field": "title", "weight": 5},
        {"op": "set_saturation_k", "value": 3},
        {"op": "set_quant_tier", "value": "floor"},
        {"op": "set_frac_bits", "value": 8},
        {"op": "replace_document", "document": replace_7},
        {"op": "delete_document", "doc_id": "D009"},
        {"op": "add_document", "document": add_15},
        {"op": "set_language_weight", "view": "original", "weight": "1.10"},
        {"op": "set_frac_bits", "value": 16},
        {"op": "set_quant_tier", "value": "nearest"},
    ]
    lines = [json.dumps(item, ensure_ascii=False) for item in commands]
    lines.insert(5, json.dumps({"op": "set_field_weight", "field": "title", "weight": 0}))
    lines.insert(10, json.dumps({"op": "explode", "value": 1}))
    return lines


def generate_sample_files(root: str | Path) -> dict[str, str]:
    root = Path(root)
    configs_dir = root / "configs"
    samples_dir = root / "samples"
    configs_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = configs_dir / "vocab.json"
    config_path = configs_dir / "pipeline.json"
    docs_path = samples_dir / "documents.jsonl"
    queries_path = samples_dir / "queries.jsonl"
    commands_path = samples_dir / "commands.jsonl"

    vocab_path.write_text(json.dumps(build_vocab(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_path.write_text(json.dumps(build_config(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    docs_path.write_text("\n".join(build_documents_jsonl()) + "\n", encoding="utf-8")
    queries_path.write_text("\n".join(build_queries_jsonl()) + "\n", encoding="utf-8")
    commands_path.write_text("\n".join(build_commands_jsonl()) + "\n", encoding="utf-8")
    return {
        "vocab": str(vocab_path),
        "config": str(config_path),
        "documents": str(docs_path),
        "queries": str(queries_path),
        "commands": str(commands_path),
    }
