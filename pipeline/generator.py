from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Concept:
    concept_id: str
    zh: str
    en: str
    availability: str = "both"
    translated: bool = True


CONCEPTS = [
    Concept("c01", "人工智能", "artificial_intelligence", availability="en"),
    Concept("c02", "机器学习", "machine_learning"),
    Concept("c03", "检索系统", "retrieval_system"),
    Concept("c04", "向量数据库", "vector_database"),
    Concept("c05", "语言模型", "language_model"),
    Concept("c06", "数据分析", "data_analysis"),
    Concept("c07", "并行计算", "parallel_computing", availability="zh"),
    Concept("c08", "本地部署", "local_deployment"),
    Concept("c09", "网络安全", "cybersecurity"),
    Concept("c10", "知识图谱", "knowledge_graph"),
    Concept("c11", "信息检索", "information_retrieval"),
    Concept("c12", "相关性排序", "relevance_ranking"),
    Concept("c13", "跨语言", "cross_language"),
    Concept("c14", "文档库", "document_library"),
    Concept("c15", "查询优化", "query_optimization"),
    Concept("c16", "搜索引擎", "search_engine", availability="zh"),
    Concept("c17", "模型评估", "model_evaluation", availability="zh"),
    Concept("c18", "预算调度", "budget_scheduling", availability="en"),
    Concept("c19", "异常隔离", "anomaly_isolation", availability="en"),
    Concept("c20", "边角案例", "edge_case", availability="en", translated=False),
    Concept("c21", "审计线索", "audit_trail", translated=False),
]


TRANSLATION_VARIANTS = {
    "c01": [("ai", 0.71)],
    "c02": [("ml", 0.68)],
    "c03": [("ir_system", 0.64)],
    "c04": [("vector_store", 0.62)],
    "c05": [("llm", 0.58)],
    "c11": [("ir", 0.55)],
    "c12": [("ranking", 0.61)],
    "c13": [("multilingual", 0.57)],
}


def _term(concept: Concept, language: str) -> str:
    return concept.zh if language == "zh" else concept.en


def _concepts_for(concept: Concept) -> list[str]:
    if concept.availability == "zh":
        return ["zh"]
    if concept.availability == "en":
        return ["en"]
    return ["zh", "en"]


def _jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _translation_rows() -> list[dict]:
    rows = []
    for concept in CONCEPTS:
        if not concept.translated:
            continue
        base = 0.93 + (int(concept.concept_id[1:]) % 7) * 0.01
        rows.append(
            {
                "source_lang": "zh",
                "source_term": concept.zh,
                "target_lang": "en",
                "target_term": concept.en,
                "confidence": round(min(base, 1.0), 6),
            }
        )
        rows.append(
            {
                "source_lang": "en",
                "source_term": concept.en,
                "target_lang": "zh",
                "target_term": concept.zh,
                "confidence": round(min(base - 0.02, 1.0), 6),
            }
        )
        for variant, confidence in TRANSLATION_VARIANTS.get(concept.concept_id, []):
            rows.append(
                {
                    "source_lang": "zh",
                    "source_term": concept.zh,
                    "target_lang": "en",
                    "target_term": variant,
                    "confidence": confidence,
                }
            )
            rows.append(
                {
                    "source_lang": "en",
                    "source_term": variant,
                    "target_lang": "zh",
                    "target_term": concept.zh,
                    "confidence": round(max(confidence - 0.05, 0.0), 6),
                }
            )
    rows.sort(key=lambda item: (item["source_lang"], item["source_term"], -item["confidence"], item["target_term"]))
    return rows


def _documents(rng: random.Random, count: int, zh_ratio: float) -> list[dict]:
    rows = []
    concept_sets: list[set[int]] = []
    languages: list[str] = []
    for index in range(count):
        language = "zh" if rng.random() < zh_ratio else "en"
        eligible = [position for position, concept in enumerate(CONCEPTS) if language in _concepts_for(concept)]
        picks = rng.sample(eligible, k=min(7, len(eligible)))
        concept_sets.append(set(picks))
        languages.append(language)
    for concept_index, concept in enumerate(CONCEPTS):
        for language in _concepts_for(concept):
            applicable = [index for index, lang in enumerate(languages) if lang == language]
            if applicable and not any(concept_index in concept_sets[index] for index in applicable):
                concept_sets[applicable[concept_index % len(applicable)]].add(concept_index)
    for index, (language, concept_set) in enumerate(zip(languages, concept_sets), start=1):
        ordered = sorted(concept_set)
        title_len = 2 if len(ordered) >= 2 else len(ordered)
        title_concepts = ordered[:title_len]
        body_concepts = ordered[title_len:]
        rows.append(
            {
                "doc_id": f"d{index:04d}",
                "lang": language,
                "title": " ".join(_term(CONCEPTS[position], language) for position in title_concepts),
                "body": " ".join(_term(CONCEPTS[position], language) for position in body_concepts)
                or _term(CONCEPTS[ordered[0]], language),
            }
        )
    return rows


def _queries(rng: random.Random, count: int, zh_ratio: float, mixed_ratio: float) -> tuple[list[dict], list[set[int]]]:
    rows = []
    concept_sets = []
    for index in range(1, count + 1):
        roll = rng.random()
        if index == 2:
            language = "zh"
            concepts = [19, 10, 11]
            terms = [_term(CONCEPTS[position], language) for position in concepts]
        elif index == 3:
            language = "zh"
            concepts = [0, 13, 7]
            terms = [_term(CONCEPTS[position], language) for position in concepts]
        elif roll < mixed_ratio:
            language = "mixed"
            zh_candidates = [position for position, concept in enumerate(CONCEPTS) if concept.availability in ("both", "zh")]
            en_candidates = [position for position, concept in enumerate(CONCEPTS) if concept.availability in ("both", "en")]
            zh_picks = rng.sample(zh_candidates, k=min(2, len(zh_candidates)))
            en_picks = rng.sample(en_candidates, k=min(2, len(en_candidates)))
            concepts = zh_picks + en_picks
            terms = [_term(CONCEPTS[position], "zh") for position in zh_picks] + [
                _term(CONCEPTS[position], "en") for position in en_picks
            ]
        else:
            language = "zh" if rng.random() < zh_ratio else "en"
            eligible = [position for position, concept in enumerate(CONCEPTS) if language in _concepts_for(concept)]
            picks = rng.sample(eligible, k=min(3, len(eligible)))
            forced_pool = [position for position, concept in enumerate(CONCEPTS) if concept.availability == ("en" if language == "zh" else "zh")]
            if forced_pool and index % 3 == 0:
                picks[0] = forced_pool[(index // 3) % len(forced_pool)]
            concepts = sorted(set(picks))
            terms = [_term(CONCEPTS[position], language) for position in concepts]
        rng.shuffle(terms)
        rows.append(
            {
                "query_id": f"q{index:04d}",
                "lang": language,
                "query": " ".join(terms),
            }
        )
        concept_sets.append(set(concepts))
    return rows, concept_sets


def _qrels(documents: list[dict], queries: list[dict], query_concepts: list[set[int]]) -> list[dict]:
    document_concepts: dict[str, set[int]] = {}
    term_to_concept = {}
    for concept in CONCEPTS:
        term_to_concept[concept.zh] = concept.concept_id
        term_to_concept[concept.en] = concept.concept_id
    id_to_position = {concept.concept_id: index for index, concept in enumerate(CONCEPTS)}
    for document in documents:
        tokens = (document["title"] + " " + document["body"]).split()
        document_concepts[document["doc_id"]] = {
            id_to_position[term_to_concept[token]] for token in tokens if token in term_to_concept
        }
    rows = []
    for query, concepts in zip(queries, query_concepts):
        scored = []
        for document in documents:
            overlap = len(concepts & document_concepts[document["doc_id"]])
            if overlap:
                scored.append((overlap, document["doc_id"]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = scored[:10]
        for overlap, doc_id in selected:
            grade = min(3, overlap)
            rows.append({"query_id": query["query_id"], "doc_id": doc_id, "relevance": grade})
        negatives = [document["doc_id"] for document in documents if document["doc_id"] not in {item[1] for item in selected}]
        for doc_id in negatives[:2]:
            rows.append({"query_id": query["query_id"], "doc_id": doc_id, "relevance": 0})
    rows.sort(key=lambda item: (item["query_id"], item["doc_id"]))
    return rows


def default_config() -> dict:
    return {
        "pipeline_version": "1.0.0",
        "normalization": {
            "casefold": True,
            "nfkc": True,
            "collapse_whitespace": True,
            "token_pattern": r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+",
        },
        "allowed_languages": ["zh", "en", "mixed"],
        "index_languages": ["zh", "en"],
        "language_detection": {
            "enabled": True,
            "mixed_ratio_threshold": 0.2,
        },
        "translation": {"bridge_weight": 0.75},
        "bm25": {"k1": 1.2, "b": 0.75, "title_repeat": 2},
        "recall_limit": 20,
        "fusion": {
            "mono_weight": 1.0,
            "cross_weight": 2.0,
            "temperature": 2.0,
        },
        "fixed_point": {"bits": 64, "scale": 1000000},
        "relevance_thresholds": [
            {"min_score": 0.0, "grade": 0},
            {"min_score": 0.08, "grade": 1},
            {"min_score": 0.14, "grade": 2},
            {"min_score": 0.2, "grade": 3},
        ],
        "max_relevance_grade": 3,
        "evaluation": {
            "metrics": ["precision_at_k", "recall_at_k", "ndcg_at_k", "mrr_at_k", "grade_mae", "grade_accuracy"],
            "k": 10,
        },
        "threads": {"max_workers": 4},
        "budget": {"global_retrieval_points": 20000, "per_query_quota": 400},
        "snapshot": {"batch_size": 4},
    }


def _default_injections(queries: list[dict]) -> list[dict]:
    ids = [item["query_id"] for item in queries]
    if not ids:
        return []
    candidates = sorted(set([0, len(ids) // 3, (2 * len(ids)) // 3, len(ids) - 1]))
    fill = 0
    while len(candidates) < min(4, len(ids)):
        if fill not in candidates:
            candidates.append(fill)
        fill += 1
    candidates.sort()

    selected = [ids[index] for index in candidates[:4]]
    if len(selected) < 4:
        selected = (selected * 4)[:4]

    result = [
        {
            "injection_id": "INJ-001",
            "query_id": selected[0],
            "stage": "retrieval_scoring",
            "kind": "nonfinite_confidence",
            "enabled": True,
            "params": {},
        },
        {
            "injection_id": "INJ-002",
            "query_id": selected[1],
            "stage": "retrieval_scoring",
            "kind": "translation_term_missing",
            "enabled": True,
            "params": {"source_lang": "en", "term": "unmapped_probe_term"},
        },
        {
            "injection_id": "INJ-003",
            "query_id": selected[2],
            "stage": "retrieval_scoring",
            "kind": "recall_limit_zero",
            "enabled": True,
            "params": {},
        },
        {
            "injection_id": "INJ-004",
            "query_id": selected[3],
            "stage": "evaluation",
            "kind": "relevance_grade_out_of_range",
            "enabled": True,
            "params": {},
        },
    ]
    return result


def generate_sample(
    output_dir: str | Path,
    *,
    seed: int = 20261120,
    documents: int = 48,
    queries: int = 20,
    zh_ratio: float = 0.5,
    mixed_ratio: float = 0.25,
) -> dict:
    if documents <= 0 or queries <= 0:
        raise ValueError("documents and queries must be positive")
    if not 0.0 <= zh_ratio <= 1.0 or not 0.0 <= mixed_ratio <= 1.0:
        raise ValueError("language ratios must be in [0, 1]")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_rng = random.Random(seed)
    query_rng = random.Random(seed + 1)
    document_rows = _documents(doc_rng, documents, zh_ratio)
    query_rows, query_concepts = _queries(query_rng, queries, zh_ratio, mixed_ratio)
    translation_rows = _translation_rows()
    qrel_rows = _qrels(document_rows, query_rows, query_concepts)
    config = default_config()
    injection_rows = _default_injections(query_rows)
    _jsonl(output_dir / "documents.jsonl", document_rows)
    _jsonl(output_dir / "queries.jsonl", query_rows)
    _jsonl(output_dir / "translations.jsonl", translation_rows)
    _jsonl(output_dir / "qrels.jsonl", qrel_rows)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    with (output_dir / "injections.json").open("w", encoding="utf-8") as handle:
        json.dump({"injections": injection_rows}, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    metadata = {
        "generator": "pipeline.generator.generate_sample",
        "seed": seed,
        "documents": len(document_rows),
        "queries": len(query_rows),
        "translations": len(translation_rows),
        "qrels": len(qrel_rows),
        "injections": len(injection_rows),
        "zh_ratio": zh_ratio,
        "mixed_ratio": mixed_ratio,
        "controlled_concepts": [
            {
                "concept_id": concept.concept_id,
                "zh": concept.zh,
                "en": concept.en,
                "availability": concept.availability,
                "translated": concept.translated,
            }
            for concept in CONCEPTS
        ],
    }
    with (output_dir / "sample.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    return {
        "output_dir": str(output_dir),
        "seed": seed,
        "documents": len(document_rows),
        "queries": len(query_rows),
        "translations": len(translation_rows),
        "qrels": len(qrel_rows),
        "injections": len(injection_rows),
    }
