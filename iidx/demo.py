from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .util import atomic_write_json, atomic_write_text


DEFAULT_SEED = 20261123


def _fullwidth(term: str) -> str:
    output: list[str] = []
    for char in term:
        code = ord(char)
        if 0x21 <= code <= 0x7E:
            output.append(chr(code + 0xFEE0))
        else:
            output.append(char)
    return "".join(output)


def _stylize(term: str, mode: int) -> str:
    if mode == 0:
        return term
    if mode == 1:
        return term.upper()
    if mode == 2:
        return term.title()
    if mode == 3:
        return _fullwidth(term)
    if mode == 4:
        return "  " + term + "  "
    return term.casefold()


def generate_sample(workspace: Path, *, seed: int = DEFAULT_SEED, force: bool = False, include_invalid: bool = False) -> dict[str, Any]:
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    documents_path = workspace / "documents.jsonl"
    queries_path = workspace / "queries.jsonl"
    vocabulary_path = workspace / "vocabulary.txt"
    versions_path = workspace / "query_versions.json"
    config_path = workspace / "config.json"
    metadata_path = workspace / "generator.json"
    existing = [documents_path, queries_path, vocabulary_path, versions_path, config_path]
    if all(path.exists() for path in existing) and not force:
        return {
            "workspace": str(workspace),
            "generated": False,
            "seed": seed,
            "files": [str(path) for path in existing],
        }

    rng = random.Random(seed)
    vocabulary = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
        "index",
        "search",
        "query",
        "token",
        "budget",
        "snapshot",
        "vector",
        "shard",
        "latency",
        "cache",
        "filter",
        "decode",
        "encode",
        "block",
        "stream",
        "manifest",
    ]
    shards = ["shard-a", "shard-b", "shard-c", "shard-d"]
    documents: list[dict[str, Any]] = []
    for number in range(36):
        title_terms = []
        body_terms = []
        for _ in range(rng.randint(3, 5)):
            title_terms.append(_stylize(rng.choice(vocabulary), rng.randrange(6)))
        for _ in range(rng.randint(8, 14)):
            body_terms.append(_stylize(rng.choice(vocabulary), rng.randrange(6)))
        documents.append(
            {
                "doc_id": f"doc-{number + 1:04d}",
                "title_terms": title_terms,
                "body_terms": body_terms,
                "shard": shards[number % len(shards)],
            }
        )
    queries: list[dict[str, Any]] = []
    for number in range(12):
        terms = sorted({_stylize(rng.choice(vocabulary), rng.randrange(6)) for _ in range(rng.randint(2, 4))})
        queries.append(
            {
                "query_id": f"q-{number + 1:03d}",
                "query_terms": terms,
                "query_stream_version": rng.choice(["v1", "v2"]),
            }
        )
    if include_invalid:
        documents.append(
            {
                "doc_id": "doc-bad-vocab",
                "title_terms": ["not-in-vocabulary"],
                "body_terms": ["alpha"],
                "shard": "shard-a",
            }
        )
        documents.append(
            {
                "doc_id": "doc-0001",
                "title_terms": ["alpha"],
                "body_terms": ["beta"],
                "shard": "shard-b",
            }
        )
        queries.append(
            {
                "query_id": "q-bad-version",
                "query_terms": ["alpha", "beta"],
                "query_stream_version": "v9",
            }
        )
        queries.append(
            {
                "query_id": "q-001",
                "query_terms": ["alpha"],
                "query_stream_version": "v1",
            }
        )
    config = {
        "chunk_size": 5,
        "batch_max_docs": 4,
        "thread_pool_size": 3,
        "concurrency_quota": 4,
        "global_token_budget": 1200,
        "per_shard_token_quota": 500,
        "snapshot_commit_interval_batches": 2,
        "interactive_latency_budget_ms": 180,
        "fixed_decimal_bits": 12,
        "quantization_tiers": [1, 2, 4, 8],
        "quantization_tier": 1,
        "compression_tier": "auto",
        "compression_switch_points": [3, 6],
        "low_frequency_threshold": 1,
        "query_top_k": 10,
    }
    version_table = {
        "schema": 1,
        "declared_versions": ["v1", "v2"],
        "compatibility": {"v1": "stable", "v2": "stable"},
    }
    document_lines = "\n".join(json.dumps(doc, ensure_ascii=False, sort_keys=True) for doc in documents) + "\n"
    query_lines = "\n".join(json.dumps(query, ensure_ascii=False, sort_keys=True) for query in queries) + "\n"
    atomic_write_text(documents_path, document_lines)
    atomic_write_text(queries_path, query_lines)
    atomic_write_text(vocabulary_path, "\n".join(vocabulary) + "\n")
    atomic_write_json(versions_path, version_table)
    atomic_write_json(config_path, config)
    metadata = {
        "schema": 1,
        "seed": seed,
        "vocabulary_size": len(vocabulary),
        "documents": len(documents),
        "queries": len(queries),
        "shards": shards,
        "include_invalid": include_invalid,
        "generator": "iidx.demo.generate_sample",
    }
    atomic_write_json(metadata_path, metadata)
    return {
        "workspace": str(workspace),
        "generated": True,
        "seed": seed,
        "files": [str(path) for path in existing],
        "metadata": str(metadata_path),
    }
