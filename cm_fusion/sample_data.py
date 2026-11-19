from __future__ import annotations

import json
import random
import shutil
from pathlib import Path


VOCABULARY = [
    "Alpha",
    "Beta",
    "Gamma",
    "Delta",
    "Epsilon",
    "Zeta",
    "Eta",
    "Theta",
    "Iota",
    "Kappa",
    "Lambda",
    "Mu",
    "Nu",
    "Xi",
    "Omicron",
    "Pi",
    "Rho",
    "Sigma",
    "Tau",
    "Upsilon",
    "Phi",
    "Chi",
    "Psi",
    "Omega",
    "Retrieval",
    "Fusion",
    "Index",
    "Snapshot",
    "Query",
    "Document",
    "Shard",
    "Token",
    "检索",
    "融合",
    "索引",
    "快照",
    "查询",
    "文档",
    "分片",
    "词项",
]


COMPAT_TABLE = {
    "schema_version": 1,
    "artifact_versions": {
        "v1": {
            "header": {
                "kind": "fixed",
                "fields": {
                    "term_ref": {"encoding": "uint", "width": 4},
                    "doc_count": {"encoding": "uint", "width": 2},
                    "flags": {"encoding": "uint", "width": 1},
                    "tier": {"encoding": "uint", "width": 1},
                    "payload_length": {"encoding": "uint", "width": 4},
                    "checksum": {"encoding": "uint", "width": 4},
                },
            },
            "compression_encoding": "delta_varint",
            "tier_semantics": {
                "1": "delta_varint",
                "2": "delta_varint_lowfreq_tail",
            },
            "supported_tiers": [1, 2],
            "features": {
                "block_checksum": True,
                "position_skiplist": False,
            },
        },
        "v2": {
            "header": {
                "kind": "varint",
                "fields": {
                    "term_ref": {"encoding": "uvarint"},
                    "doc_count": {"encoding": "uvarint"},
                    "flags": {"encoding": "uvarint"},
                    "tier": {"encoding": "uvarint"},
                    "payload_length": {"encoding": "uvarint"},
                },
            },
            "compression_encoding": "delta_varint",
            "tier_semantics": {
                "1": "delta_varint",
                "2": "delta_varint_lowfreq_sidecar",
            },
            "supported_tiers": [1, 2],
            "features": {
                "block_checksum": False,
                "position_skiplist": True,
            },
        },
    },
    "query_stream_versions": ["q1", "q2"],
    "tier_mapping": {
        "v1": {
            "1": "delta_varint",
            "2": "delta_varint_lowfreq_tail",
        },
        "v2": {
            "1": "delta_varint",
            "2": "delta_varint_lowfreq_sidecar",
        },
    },
}


INITIAL_CONFIG = {
    "schema_version": 1,
    "normalization": {
        "case_fold": True,
        "width_fold": True,
        "whitespace_collapse": True,
    },
    "controlled_vocabulary": VOCABULARY,
    "allowed_shards": ["shard-a", "shard-b", "shard-c"],
    "posting_block_size": 4,
    "compression_tiers": [
        {
            "id": 1,
            "codec": "delta_varint",
            "low_frequency_filter": False,
            "low_frequency_threshold": 2,
        },
        {
            "id": 2,
            "codec": "delta_varint_lowfreq_filter",
            "low_frequency_filter": True,
            "low_frequency_threshold": 3,
        },
    ],
    "active_compression_tier": 1,
    "keep_positions": True,
    "index_format_version": "v2",
    "channel_weights": {"lexical": 0.65, "cooccurrence": 0.35},
    "fusion_temperature": 1.0,
    "fixed_decimal_bits": 16,
    "fixed_tolerance_units": 0.5,
    "interaction_budget_ms": 250,
    "concurrency": 4,
    "snapshot_interval_batches": 2,
    "top_k": 5,
}


def _present(term: str, rng: random.Random) -> str:
    variant = rng.randrange(5)
    if variant == 0:
        return term
    if variant == 1:
        return term.upper()
    if variant == 2:
        return "".join(chr(ord(char) + 0xFEE0) if "!" <= char <= "~" else char for char in term)
    if variant == 3:
        return f"  {term.casefold()}  "
    return term


def _make_document(doc_id: str, rng: random.Random, shard: str) -> dict:
    title = [_present(term, rng) for term in rng.sample(VOCABULARY, rng.randint(3, 4))]
    body = [_present(term, rng) for term in rng.choices(VOCABULARY, k=rng.randint(9, 15))]
    return {"doc_id": doc_id, "title_terms": title, "body_terms": body, "shard": shard}


def _query(query_id: str, terms: list[str], version: str) -> dict:
    return {"query_id": query_id, "terms": terms, "stream_version": version}


def _line(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_samples(output_dir: str | Path, seed: int = 20261120) -> dict:
    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    documents = []
    for index in range(1, 25):
        rng = random.Random(seed + index * 101)
        documents.append(_make_document(f"D{index:04d}", rng, f"shard-{chr(97 + (index - 1) % 3)}"))

    documents.extend(
        [
            {"doc_id": "D9001", "title_terms": ["Alpha", "Beta"], "body_terms": [], "shard": "shard-a"},
            {"doc_id": "D9002", "title_terms": ["Gamma"], "body_terms": ["Delta", "Epsilon"], "shard": "shard-unknown"},
            {"doc_id": "D9003", "title_terms": ["NotInVocabulary"], "body_terms": ["Zeta"], "shard": "shard-b"},
        ]
    )
    docs_text = "\n".join(_line(document) for document in documents) + "\n"
    (out / "docs.jsonl").write_text(docs_text, encoding="utf-8")

    query_rng = random.Random(seed + 17)
    queries = []
    fixed_queries = [
        ["Alpha", "Beta", "Gamma"],
        ["Fusion", "Retrieval", "Index"],
        ["Snapshot", "Query", "Document"],
        ["检索", "融合", "索引"],
        ["Theta", "Iota", "Kappa", "Lambda"],
        ["Token", "Shard", "Document"],
        ["Omega", "Psi", "Chi"],
        ["快照", "查询", "词项"],
    ]
    for index, terms in enumerate(fixed_queries, start=1):
        presented = [_present(term, query_rng) for term in terms]
        queries.append(_query(f"Q{index:04d}", presented, "q1" if index % 2 else "q2"))
    queries.extend(
        [
            _query("Q9001", ["Alpha"], "q-undeclared"),
            _query("Q9002", [], "q1"),
        ]
    )
    queries_text = "\n".join(_line(query) for query in queries) + "\n"
    (out / "queries.jsonl").write_text(queries_text, encoding="utf-8")
    _write_json(out / "config.json", INITIAL_CONFIG)
    _write_json(out / "compat.json", COMPAT_TABLE)

    add_25 = _make_document("D0025", random.Random(seed + 2501), "shard-b")
    add_26 = _make_document("D0026", random.Random(seed + 2601), "shard-c")
    fixed_03 = _make_document("D0003", random.Random(seed + 3003), "shard-c")
    fixed_03["title_terms"] = ["Fusion", "Retrieval", "Alpha", "Index"]
    fixed_03["body_terms"] = ["Fusion", "Retrieval", "Index", "Alpha", "Beta", "Gamma", "Snapshot"]

    edits = [
        {"command_id": "E001", "op": "add_doc", "doc": add_25},
        {"command_id": "E002", "op": "fix_doc", "doc": fixed_03},
        {
            "command_id": "E003",
            "op": "set_config",
            "patch": {"posting_block_size": 3, "active_compression_tier": 2},
        },
        {"command_id": "E004", "op": "set_config", "patch": {"keep_positions": False}},
        {
            "command_id": "E005",
            "op": "set_config",
            "patch": {
                "channel_weights": {"lexical": 0.55, "cooccurrence": 0.45},
                "fusion_temperature": 0.9,
            },
        },
        {
            "command_id": "E006",
            "op": "set_config",
            "patch": {
                "index_format_version": "v1",
                "active_compression_tier": 1,
                "keep_positions": True,
            },
        },
        {"command_id": "E007", "op": "add_doc", "doc": add_26},
        {
            "command_id": "E008",
            "op": "set_config",
            "patch": {
                "index_format_version": "v2",
                "active_compression_tier": 2,
                "keep_positions": True,
                "posting_block_size": 3,
            },
        },
    ]
    (out / "edits.jsonl").write_text("\n".join(_line(command) for command in edits) + "\n", encoding="utf-8")

    final_lines = docs_text.rstrip("\n").split("\n")
    final_lines.append(_line(add_25))
    final_lines = [
        _line(fixed_03) if json.loads(line).get("doc_id") == "D0003" else line
        for line in final_lines
    ]
    final_lines.append(_line(add_26))
    final_config = json.loads(json.dumps(INITIAL_CONFIG))
    for command in edits:
        if command["op"] == "set_config":
            final_config = _deep_merge(final_config, command["patch"])

    final_dir = out / "final"
    final_dir.mkdir()
    (final_dir / "docs.jsonl").write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    (final_dir / "queries.jsonl").write_text(queries_text, encoding="utf-8")
    _write_json(final_dir / "config.json", final_config)
    _write_json(final_dir / "compat.json", COMPAT_TABLE)

    invalid_docs = [
        '{"doc_id":"D8001",',
        _line({"doc_id": "D8002", "title_terms": ["Alpha"], "body_terms": [], "shard": "shard-a"}),
        _line({"doc_id": "D8003", "title_terms": ["Alpha"], "body_terms": ["Beta"], "shard": "shard-z"}),
        _line({"doc_id": "D8004", "title_terms": ["Alpha"], "body_terms": ["Outside"], "shard": "shard-a"}),
        _line({"doc_id": "D8005", "title_terms": ["Alpha"], "body_terms": ["Beta"], "shard": "shard-a"}),
        _line({"doc_id": "D8005", "title_terms": ["Gamma"], "body_terms": ["Delta"], "shard": "shard-b"}),
    ]
    (out / "invalid_docs.jsonl").write_text("\n".join(invalid_docs) + "\n", encoding="utf-8")
    invalid_queries = [
        _line({"query_id": "Q8001", "terms": ["Alpha"], "stream_version": "q-unknown"}),
        _line({"query_id": "Q8002", "terms": [], "stream_version": "q1"}),
        _line({"query_id": "Q8003", "terms": ["Outside"], "stream_version": "q1"}),
        _line({"query_id": "Q8004", "terms": ["Alpha"], "stream_version": "q1"}),
        _line({"query_id": "Q8004", "terms": ["Beta"], "stream_version": "q1"}),
    ]
    (out / "invalid_queries.jsonl").write_text("\n".join(invalid_queries) + "\n", encoding="utf-8")

    return {
        "seed": seed,
        "documents": len(documents),
        "valid_documents": 24,
        "queries": len(queries),
        "valid_queries": 8,
        "edit_commands": len(edits),
        "final_documents": len(final_lines),
        "final_config_hash": _canonical_hash(final_config),
    }


def _deep_merge(base: dict, patch: dict) -> dict:
    result = json.loads(json.dumps(base))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _canonical_hash(value: dict) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

