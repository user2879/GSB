from __future__ import annotations

from pathlib import Path
from typing import Any

from .validation import (
    load_command_stream,
    load_json,
    validate_command_stream,
    validate_document_stream,
    validate_inference,
    validate_topology,
    validate_vocab,
)


def load_bundle(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    topology = validate_topology(load_json(root / "config" / "nodes.json"))
    sample_raw = load_json(root / "samples" / "documents.json")
    vocab = validate_vocab(sample_raw.get("vocab") if isinstance(sample_raw, dict) else None)
    inference = validate_inference(
        load_json(root / "config" / "inference.json"), vocab_size=len(vocab)
    )
    documents = validate_document_stream(
        sample_raw,
        shard_ids=topology["shards"],
        term_ids={item["term_id"] for item in vocab},
    )
    commands_path = root / "config" / "commands.json"
    commands = load_command_stream(commands_path) if commands_path.exists() else []
    validate_command_stream(
        commands,
        shard_ids=topology["shards"],
        term_ids={item["term_id"] for item in vocab},
        initial_documents=documents,
        node_ids={node["node_id"] for node in topology["nodes"]},
        vocab_size=len(vocab),
    )
    return {
        "topology": topology,
        "inference": inference,
        "vocab": vocab,
        "documents": documents,
        "commands": commands,
        "seed": sample_raw.get("seed") if isinstance(sample_raw, dict) else None,
    }
