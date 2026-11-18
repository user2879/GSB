"""Deterministic sample generator (fixed seed, controlled vocabulary)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SEED = 20261118

CATEGORY_TERMS: dict[str, list[str]] = {
    "phone": ["phone", "smartphone", "mobile", "handset"],
    "laptop": ["laptop", "notebook", "ultrabook", "workstation"],
    "appliance": ["fridge", "washer", "appliance", "freezer"],
    "wearable": ["watch", "earbuds", "wearable", "headset"],
    "home": ["lamp", "vacuum", "humidifier", "purifier"],
    "camera": ["camera", "lens", "tripod", "camcorder"],
}

BRANDS = ["nimbus", "vertex", "halcyon", "orbit", "lumen"]
ATTRIBUTES = [
    "battery",
    "warranty",
    "shipping",
    "discount",
    "stock",
    "screen",
    "memory",
    "charger",
]
SOURCES = ["catalog", "manual", "review", "faq"]
# Intentionally outside the document vocabulary for citation-missing tests.
OOV_TOKEN = "zzx-quantum-widget"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def build_documents(rng: random.Random) -> list[dict[str, Any]]:
    categories = list(CATEGORY_TERMS)
    docs: list[dict[str, Any]] = []
    for index in range(60):
        category = categories[index % len(categories)]
        terms = CATEGORY_TERMS[category]
        brand = BRANDS[index % len(BRANDS)]
        head = terms[index % len(terms)]
        title = f"{brand} {head} guide {index + 1:03d}"
        sentences: list[str] = []
        for _ in range(4):
            picks = rng.sample(terms, k=min(2, len(terms)))
            attr = rng.choice(ATTRIBUTES)
            sentences.append(
                " ".join(
                    [
                        f"the {brand} {head}",
                        f"covers {' '.join(picks)}",
                        f"and explains {attr} policy.",
                    ]
                )
            )
        if index == 7:
            # Full-width folding case: title contains full-width ASCII chars
            # that normalization must map back to the regular vocabulary.
            title = "".join(
                chr(ord(ch) + 0xFEE0) if ch.isascii() and ch != " " else ch
                for ch in title
            ).replace(" ", "\u3000")
        docs.append(
            {
                "doc_id": f"DOC{index + 1:04d}",
                "title": title,
                "body": " ".join(sentences),
                "source": SOURCES[index % len(SOURCES)],
            }
        )
    return docs


def build_queries(rng: random.Random) -> list[dict[str, Any]]:
    categories = list(CATEGORY_TERMS)
    rows: list[dict[str, Any]] = []

    def add(session_id: str, turn_no: int, query: str, followup: bool, refs):
        rows.append(
            {
                "session_id": session_id,
                "turn_no": turn_no,
                "query": query,
                "is_followup": followup,
                "context_ref": list(refs),
            }
        )

    # Session S-A: 8 turns, follow-ups referencing earlier turns.
    for turn_no in range(1, 9):
        category = categories[(turn_no - 1) % len(categories)]
        term = CATEGORY_TERMS[category][0]
        attr = ATTRIBUTES[(turn_no - 1) % len(ATTRIBUTES)]
        query = f"{term} {attr} options"
        if turn_no == 1:
            add("S-A", 1, query, False, [])
        else:
            refs = [turn_no - 1] if turn_no % 2 == 0 else [turn_no - 2]
            add("S-A", turn_no, query, True, refs)
    # Session S-B: 6 turns.
    for turn_no in range(1, 7):
        category = categories[(turn_no + 2) % len(categories)]
        term = CATEGORY_TERMS[category][1]
        query = f"best {term} review and price"
        if turn_no == 1:
            add("S-B", 1, query, False, [])
        else:
            add("S-B", turn_no, query, True, [turn_no - 1])
    # Session S-C: a valid turn plus an OOV-only turn (E_CITATION_MISSING).
    add("S-C", 1, "phone warranty shipping", False, [])
    add("S-C", 2, OOV_TOKEN, True, [1])
    return rows


def build_commands() -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = [
        {
            "seq": 1,
            "op": "set_bm25",
            "args": {"k1": 1.8, "b": 0.85},
        },
        {
            "seq": 2,
            "op": "set_recall_top_k",
            "args": {"recall_top_k": 5},
        },
        {
            "seq": 3,
            "op": "set_fusion_weights",
            "args": {"title_weight": 0.35, "overlap_weight": 0.2},
        },
        {
            "seq": 4,
            "op": "set_template",
            "args": {"template": "concise"},
        },
        {
            "seq": 5,
            "op": "set_context",
            "args": {
                "session_id": "S-A",
                "turn_no": 3,
                "is_followup": True,
                "context_ref": [1, 2],
            },
        },
        {
            "seq": 6,
            "op": "replace_turn_query",
            "args": {
                "session_id": "S-B",
                "turn_no": 2,
                "query": "laptop memory warranty options",
            },
        },
        {
            "seq": 7,
            "op": "update_template",
            "args": {
                "template": "concise",
                "body": "Q:{query}|A:{top_snippet}|C:{citation_count}|D:{top_doc_id}",
            },
        },
        {
            "seq": 8,
            "op": "set_template",
            "args": {"template": "standard"},
        },
    ]
    return commands


def build_config() -> dict[str, Any]:
    return {
        "normalization": {
            "fold_full_half": True,
            "case_fold": True,
            "whitespace_merge": True,
        },
        "bm25": {"k1": 1.5, "b": 0.75},
        "recall_top_k": 8,
        "answer_window": 3,
        "min_citations": 4,
        "fusion_weights": {
            "bm25_weight": 1.0,
            "title_weight": 0.25,
            "overlap_weight": 0.15,
            "sentence_weight": 1.0,
            "doc_balance_weight": 0.2,
        },
        "first_response_budget_ms": 120,
        "max_concurrency": 4,
        "snapshot_commit_interval": 1,
    }


def build_rules() -> dict[str, Any]:
    return {
        "version": "rules-v1",
        "synonyms": {
            "mobile": "phone",
            "handset": "phone",
            "notebook": "laptop",
            "ultrabook": "laptop",
        },
        "category_expand": {
            "phone": ["smartphone"],
            "laptop": ["workstation"],
            "earbuds": ["headset"],
        },
        "templates": {
            "standard": (
                "Answer for '{query}': {top_snippet} "
                "(citations={citation_count}, top={top_doc_id}, sources={sources})"
            ),
            "concise": (
                "{query} => {top_snippet} [{citation_count} cites via {sources}]"
            ),
        },
    }


def build_faults() -> dict[str, Any]:
    return {
        "force_no_citation_turn": "S-C#0002",
        "invalid_doc_inject": False,
        "invalid_query_inject": False,
        "retrieval_delay_ms": 0,
        "retrieval_delay_targets": [],
    }


def generate(sample_dir: str | Path) -> dict[str, Path]:
    rng = random.Random(SEED)
    sample_dir = Path(sample_dir)
    docs = build_documents(rng)
    queries = build_queries(rng)
    commands = build_commands()
    paths = {
        "docs": sample_dir / "documents.jsonl",
        "queries": sample_dir / "queries.jsonl",
        "commands": sample_dir / "commands.jsonl",
        "config": sample_dir / "retrieval_config.json",
        "rules": sample_dir / "domain_rules.json",
        "faults": sample_dir / "fault_injection.json",
    }
    _write_jsonl(paths["docs"], docs)
    _write_jsonl(paths["queries"], queries)
    _write_jsonl(paths["commands"], commands)
    _write_json(paths["config"], build_config())
    _write_json(paths["rules"], build_rules())
    _write_json(paths["faults"], build_faults())
    return paths
