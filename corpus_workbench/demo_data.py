"""Generate the deterministic local corpus, configurations and fault injections."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .util import canonical_json, file_sha256, sha256_json, write_bytes_atomic, write_text_atomic

COMMON_WORDS = [
    "archive", "signal", "garden", "chapter", "river", "copper", "meadow", "orbit",
    "theory", "village", "machine", "winter", "market", "silent", "vector", "harbor",
    "engine", "forest", "memory", "planet", "studio", "camera", "letter", "bridge",
    "window", "island", "crystal", "battery", "pattern", "lantern", "journey", "puzzle",
    "compass", "library", "station", "field", "thread", "canvas", "summer", "mountain",
    "notebook", "weather", "stone", "paper", "city", "valley", "cloud", "ocean",
]

TOPIC_WORDS = {
    "books": ["novel", "narrator", "prologue", "metaphor", "chapter", "author", "fiction", "poem"],
    "papers": ["study", "dataset", "method", "baseline", "experiment", "evidence", "model", "result"],
    "web": ["portal", "headline", "browser", "article", "forum", "update", "column", "traffic"],
}

SOURCE_WEIGHTS = {"books": 0.30, "papers": 0.20, "web": 0.50}


def _write_json(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _generate_source_documents(source: str, count: int = 24) -> list[dict[str, Any]]:
    rng = random.Random(20261123 + sum(ord(char) for char in source))
    topic = TOPIC_WORDS[source]
    documents: list[dict[str, Any]] = []
    prior_tokens: list[str] | None = None
    for index in range(count):
        if index % 2 == 1 and prior_tokens is not None:
            tokens = list(prior_tokens)
            mutation_index = 5 + (index % 7)
            tokens[mutation_index] = rng.choice(COMMON_WORDS)
            relation = "near_duplicate"
        else:
            tokens = [rng.choice(topic)]
            while len(tokens) < 18:
                candidate = rng.choice(COMMON_WORDS)
                if candidate not in tokens or len(tokens) >= 12:
                    tokens.append(candidate)
            relation = "base"
        prior_tokens = list(tokens)
        documents.append(
            {
                "source": source,
                "doc_id": f"{source}-{index:04d}",
                "text": " ".join(tokens),
                "metadata": {
                    "topic": topic[index % len(topic)],
                    "relation": relation,
                    "generator_seed": 20261123 + index,
                },
            }
        )
    return documents


def generate_demo_workspace(project_root: str | Path = ".") -> dict[str, Any]:
    root = Path(project_root)
    data_root = root / "data" / "corpus"
    config_root = root / "configs"
    data_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    source_documents = {
        source: _generate_source_documents(source)
        for source in ("books", "papers", "web")
    }
    for source, documents in source_documents.items():
        lines = [canonical_json(document) for document in documents]
        write_text_atomic(data_root / f"{source}.jsonl", "\n".join(lines) + "\n")

    injected_valid = [
        {
            "source": "web",
            "doc_id": "web-injected-empty",
            "text": "",
            "metadata": {"injection_id": "inj-empty-text", "fault": "empty_text"},
        },
        {
            "source": "forum",
            "doc_id": "forum-injected-unknown",
            "text": "unknown source document",
            "metadata": {"injection_id": "inj-unknown-source", "fault": "unknown_source"},
        },
        {
            "source": "papers",
            "doc_id": "papers-0000",
            "text": "duplicate identifier injected document",
            "metadata": {"injection_id": "inj-duplicate-doc-id", "fault": "duplicate_doc_id"},
        },
    ]
    non_utf8_prefix = (
        b'{"source":"books","doc_id":"books-injected-non-utf8",'
        b'"metadata":{"injection_id":"inj-non-utf8","fault":"non_utf8"},"text":"'
    )
    injected_bytes = b"\n".join([canonical_json(item).encode("utf-8") for item in injected_valid])
    injected_bytes += b"\n" + non_utf8_prefix + b"broken \xff byte sequence" + b'"}\n'
    write_bytes_atomic(data_root / "zz_injected.jsonl", injected_bytes)

    _write_json(
        config_root / "balance.json",
        {
            "source_weights": SOURCE_WEIGHTS,
            "target_sample_count": 42,
            "sampling_salt": "local-corpus-demo-v1",
        },
    )
    _write_json(
        config_root / "dedup.json",
        {
            "shingle_length": 3,
            "signature_dim": 64,
            "jaccard_threshold": 0.72,
            "representative_rule": "source_weight_desc_doc_id_asc",
        },
    )
    _write_json(
        config_root / "dedup_threshold_090.json",
        {
            "shingle_length": 3,
            "signature_dim": 64,
            "jaccard_threshold": 0.90,
            "representative_rule": "source_weight_desc_doc_id_asc",
        },
    )
    _write_json(
        config_root / "dedup_shingle_4_dim_32.json",
        {
            "shingle_length": 4,
            "signature_dim": 32,
            "jaccard_threshold": 0.72,
            "representative_rule": "source_weight_desc_doc_id_asc",
        },
    )
    _write_json(
        config_root / "scheduler.json",
        {
            "max_workers": 4,
            "global_shingle_budget": 5000,
            "source_quotas": {"books": 5000, "papers": 5000, "web": 5000},
            "batch_size": 8,
            "snapshot_interval_batches": 2,
        },
    )
    _write_json(
        config_root / "scheduler_tight.json",
        {
            "max_workers": 4,
            "global_shingle_budget": 160,
            "source_quotas": {"books": 56, "papers": 96, "web": 999},
            "batch_size": 6,
            "snapshot_interval_batches": 2,
        },
    )
    _write_json(
        config_root / "scheduler_expanded.json",
        {
            "max_workers": 4,
            "global_shingle_budget": 5000,
            "source_quotas": {"books": 5000, "papers": 5000, "web": 5000},
            "batch_size": 8,
            "snapshot_interval_batches": 2,
        },
    )
    _write_json(
        config_root / "scheduler_serial.json",
        {
            "max_workers": 1,
            "global_shingle_budget": 5000,
            "source_quotas": {"books": 5000, "papers": 5000, "web": 5000},
            "batch_size": 8,
            "snapshot_interval_batches": 2,
        },
    )
    _write_json(
        config_root / "scheduler_parallel.json",
        {
            "max_workers": 8,
            "global_shingle_budget": 5000,
            "source_quotas": {"books": 5000, "papers": 5000, "web": 5000},
            "batch_size": 8,
            "snapshot_interval_batches": 2,
        },
    )
    _write_json(
        config_root / "injection.json",
        {
            "injections": [
                {
                    "injection_id": "inj-empty-text",
                    "source": "web",
                    "doc_id": "web-injected-empty",
                    "fault": "empty_text",
                    "business_code_hint": "E_ENCODING_INVALID",
                },
                {
                    "injection_id": "inj-non-utf8",
                    "source": "books",
                    "doc_id": "books-injected-non-utf8",
                    "fault": "non_utf8",
                    "business_code_hint": "E_ENCODING_INVALID",
                },
                {
                    "injection_id": "inj-unknown-source",
                    "source": "forum",
                    "doc_id": "forum-injected-unknown",
                    "fault": "unknown_source",
                    "business_code_hint": "E_SOURCE_UNKNOWN",
                },
                {
                    "injection_id": "inj-duplicate-doc-id",
                    "source": "papers",
                    "doc_id": "papers-0000",
                    "fault": "duplicate_doc_id",
                    "business_code_hint": "E_DUP_DOC_ID",
                },
            ]
        },
    )

    files = sorted(
        path for path in [*data_root.glob("*.jsonl"), *config_root.glob("*.json")]
    )
    manifest = {
        "generator": "corpus_workbench.demo_data.v1",
        "seed": 20261123,
        "files": {
            path.relative_to(root).as_posix(): file_sha256(path)
            for path in files
        },
    }
    manifest["digest"] = sha256_json(manifest["files"])
    _write_json(data_root / "generated_manifest.json", manifest)
    return manifest

