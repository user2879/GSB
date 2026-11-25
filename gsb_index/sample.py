"""Deterministic sample generation (no external data or services)."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

from .canonical import write_json


VOCABULARY_TERMS = (
    "archive",
    "buffer",
    "cache",
    "catalog",
    "cluster",
    "codec",
    "cursor",
    "delta",
    "document",
    "encoding",
    "fragment",
    "index",
    "ledger",
    "manifest",
    "merge",
    "offset",
    "partition",
    "posting",
    "query",
    "recall",
    "revision",
    "segment",
    "shard",
    "snapshot",
    "stream",
    "token",
    "tuple",
    "version",
    "window",
    "worker",
    "the",
    "of",
    "and",
    "to",
)
STOPWORDS = ("the", "of", "and", "to")


class _Lcg:
    """Tiny fixed integer generator so samples do not depend on library RNGs."""

    def __init__(self, seed: int):
        self.state = (int(seed) & 0xFFFFFFFF) or 0xC0FFEE

    def next(self, modulo: int) -> int:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state % modulo


def find_free_ports(start: int, count: int) -> list[int]:
    ports: list[int] = []
    candidate = start
    while len(ports) < count and candidate <= 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                candidate += 1
                continue
        ports.append(candidate)
        candidate += 1
    if len(ports) != count:
        raise RuntimeError("unable to find enough free loopback ports")
    return ports


def _make_document(seed: int, doc_number: int, shard_id: str) -> dict[str, Any]:
    rng = _Lcg(seed * 131 + doc_number * 17)
    title_vocab = VOCABULARY_TERMS[:30]
    title_length = 3 + rng.next(5)
    body_length = 6 + rng.next(7)
    title = [title_vocab[rng.next(len(title_vocab))] for _ in range(title_length)]
    body = [VOCABULARY_TERMS[rng.next(len(VOCABULARY_TERMS))] for _ in range(body_length)]
    return {
        "doc_id": f"doc-{doc_number:04d}",
        "version": 1,
        "shard_id": shard_id,
        "fields": {"title": title, "body": body},
        "field_stopwords": {"title": False, "body": True},
    }


def generate_sample(out_dir: str | Path, *, seed: int = 17, port_start: int = 19200) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ports = find_free_ports(port_start, 3)
    shards = [
        {
            "shard_id": f"shard-{label}",
            "port": ports[index],
            "term_bucket_capacity": 192,
            "concurrency_limit": 2,
            "seed": seed + index * 97,
            "revision": 2,
        }
        for index, label in enumerate(("a", "b", "c"))
    ]
    documents = [
        _make_document(
            seed + (position % len(shards)) * 41,
            position + 1,
            shards[position % len(shards)]["shard_id"],
        )
        for position in range(18)
    ]
    revision_1 = {
        "version": 1,
        "parent": None,
        "bucket_count": 8,
        "bucket_weights": [float(index) for index in range(1, 9)],
        "encoding": "delta-int-v1",
    }
    revision_2 = {
        "version": 2,
        "parent": 1,
        "bucket_count": 16,
        "bucket_weights": [float(index) for index in range(1, 17)],
        "encoding": "zigzag-varint-v1",
    }
    revision_3 = {
        "version": 3,
        "parent": 2,
        "bucket_count": 32,
        "bucket_weights": [float(index) for index in range(1, 33)],
        "encoding": "bucket-fixed-v1",
    }
    project = {
        "vocabulary": {"terms": list(VOCABULARY_TERMS), "stopwords": list(STOPWORDS)},
        "documents": documents,
        "layout": {"shards": shards},
        "compression": {
            "bucket_count": 16,
            "bucket_weights": [float(index) for index in range(1, 17)],
            "encoding": "zigzag-varint-v1",
        },
        "budget": {"global_term_entry_budget": 240, "per_shard_term_entry_limit": 80},
        "revisions": {
            "revisions": [revision_1, revision_2, revision_3],
            "current_revision": 2,
        },
        "scoring": {
            "bit_width": 28,
            "fraction_bits": 8,
            "quantum_step": 4,
            "title_weight": 3,
            "body_weight": 2,
            "tf_saturation_k": 4,
        },
        "queries": [
            {"query_id": "q-index", "terms": ["index", "shard"], "top_k": 4},
            {"query_id": "q-snapshot", "terms": ["snapshot", "revision", "the"], "top_k": 3},
        ],
    }
    extra_documents = [
        _make_document(seed + 500, 19, shards[0]["shard_id"]),
        _make_document(seed + 501, 20, shards[1]["shard_id"]),
    ]
    replacement = _make_document(seed + 502, 1, shards[0]["shard_id"])
    replacement["version"] = 2
    replacement["fields"]["title"].insert(0, "snapshot")
    replacement["fields"]["body"].extend(["revision", "manifest", "cache"])
    edits = [
        {"edit_id": "e01-append", "type": "append", "documents": extra_documents},
        {
            "edit_id": "e02-replace",
            "type": "replace",
            "document": replacement,
        },
        {"edit_id": "e03-delete", "type": "delete", "doc_id": "doc-0003"},
        {
            "edit_id": "e04-capacity",
            "type": "set_shard",
            "shard_id": "shard-a",
            "term_bucket_capacity": 256,
        },
        {"edit_id": "e05-compression", "type": "set_current_revision", "revision": 3},
        {"edit_id": "e06-rollback", "type": "rollback", "revision": 1},
    ]

    write_json(out_dir / "project.json", project)
    write_json(out_dir / "vocabulary.json", project["vocabulary"])
    write_json(out_dir / "documents.json", {"documents": documents})
    write_json(out_dir / "layout.json", project["layout"])
    write_json(out_dir / "compression.json", project["compression"])
    write_json(out_dir / "budget.json", project["budget"])
    write_json(out_dir / "revisions.json", project["revisions"])
    write_json(out_dir / "queries.json", {"scoring": project["scoring"], "queries": project["queries"]})
    write_json(out_dir / "edits.json", {"seed": seed, "ports": ports, "edits": edits})
    return out_dir / "project.json"
