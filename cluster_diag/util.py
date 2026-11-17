from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def stable_seed(*parts: object) -> int:
    raw = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def node_content_payload(
    *,
    task_id: str,
    cluster_id: str,
    revision: int,
    parent_revision: int | None,
    aggregate_summary_canonical: str,
) -> dict[str, Any]:
    return {
        "schema": "cluster-node-v1",
        "task_id": task_id,
        "cluster_id": cluster_id,
        "revision": revision,
        "parent_revision": parent_revision,
        "aggregate_summary": aggregate_summary_canonical,
    }


def node_content_digest(
    *,
    task_id: str,
    cluster_id: str,
    revision: int,
    parent_revision: int | None,
    aggregate_summary_canonical: str,
) -> str:
    return digest_json(
        node_content_payload(
            task_id=task_id,
            cluster_id=cluster_id,
            revision=revision,
            parent_revision=parent_revision,
            aggregate_summary_canonical=aggregate_summary_canonical,
        )
    )


def tree_snapshot_digest(root_revision: int, effective_nodes: list[dict[str, Any]]) -> str:
    nodes = [
        {
            "cluster_id": node["cluster_id"],
            "revision": node["revision"],
            "content_digest": node["content_digest"],
        }
        for node in sorted(effective_nodes, key=lambda item: item["cluster_id"])
    ]
    return digest_json(
        {
            "schema": "result-tree-snapshot-v1",
            "root_revision": root_revision,
            "nodes": nodes,
        }
    )


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def read_json(path: str | os.PathLike[str]) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON number {value!r} is not allowed")

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)
