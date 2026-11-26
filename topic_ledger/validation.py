from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .errors import invalid_input


def load_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise invalid_input(f"file not found: {path}", str(path)) from exc
    except json.JSONDecodeError as exc:
        raise invalid_input(f"invalid JSON: {exc.msg}", str(path)) from exc


def load_command_stream(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        commands: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    commands.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise invalid_input(
                        f"invalid JSONL at line {line_number}: {exc.msg}",
                        f"commands[{line_number - 1}]",
                    ) from exc
    else:
        commands = load_json(path)
    if not isinstance(commands, list):
        raise invalid_input("command stream must be a list", "commands")
    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            raise invalid_input("command must be an object", f"commands[{index}]")
        if not isinstance(command.get("op"), str) or not command["op"]:
            raise invalid_input("command requires a non-empty op", f"commands[{index}].op")
        if "command_id" in command and not isinstance(command["command_id"], str):
            raise invalid_input("command_id must be a string", f"commands[{index}].command_id")
    return commands


KNOWN_COMMAND_OPS = {
    "run_sampling",
    "upsert_document",
    "replace_document",
    "delete_document",
    "set_config",
    "set_node_lease_limit",
    "issue_lease",
    "renew_lease",
    "release_lease",
    "recover_leases",
    "advance_clock_ms",
    "sleep_ms",
    "snapshot",
    "restore",
}


def validate_command_stream(
    commands: list[dict[str, Any]],
    *,
    shard_ids: Iterable[str],
    term_ids: Iterable[int],
    initial_documents: list[dict[str, Any]] | None = None,
    node_ids: Iterable[str] | None = None,
    vocab_size: int | None = None,
) -> list[dict[str, Any]]:
    shard_set = set(shard_ids)
    term_set = set(term_ids)
    node_set = set(node_ids or [])
    current_doc_ids: set[str] = set()
    revisions: dict[str, int] = {}
    for document in initial_documents or []:
        doc_id = document["doc_id"]
        current_doc_ids.add(doc_id)
        revisions[doc_id] = int(document["revision"])

    for index, command in enumerate(commands):
        base = f"commands[{index}]"
        op = command.get("op")
        if op not in KNOWN_COMMAND_OPS:
            raise invalid_input("unknown or unsupported command op", f"{base}.op", op=op)
        if "command_id" in command and not isinstance(command["command_id"], str):
            raise invalid_input("command_id must be a string", f"{base}.command_id")

        if op in {"upsert_document", "replace_document"}:
            document = validate_document(
                command.get("document"),
                f"{base}.document",
                shard_ids=shard_set,
                term_ids=term_set,
                last_revisions=revisions,
            )
            current_doc_ids.add(document["doc_id"])
        elif op == "delete_document":
            doc_id = require_string(command.get("doc_id"), f"{base}.doc_id")
            if doc_id not in current_doc_ids:
                raise invalid_input("doc_id does not reference a stream-known document", f"{base}.doc_id")
            current_doc_ids.remove(doc_id)
        elif op == "set_config":
            patch = require_dict(command.get("patch"), f"{base}.patch")
            if "topic_count" in patch:
                topic_count = require_positive_int(patch["topic_count"], f"{base}.patch.topic_count")
                if vocab_size is not None and topic_count > vocab_size:
                    raise invalid_input(
                        "topic_count must not exceed vocabulary size",
                        f"{base}.patch.topic_count",
                    )
            for key in ("sample_rounds", "max_rounds_per_shard", "sample_point_budget", "lease_ttl_ms"):
                if key in patch:
                    require_nonnegative_int(patch[key], f"{base}.patch.{key}")
            if "interaction_budget_ms" in patch:
                require_positive_int(patch["interaction_budget_ms"], f"{base}.patch.interaction_budget_ms")
        elif op == "set_node_lease_limit":
            node_id = require_string(command.get("node_id"), f"{base}.node_id")
            if node_set and node_id not in node_set:
                raise invalid_input("node_id is not declared", f"{base}.node_id")
            require_positive_int(command.get("lease_limit"), f"{base}.lease_limit")
        elif op == "issue_lease":
            shard_id = require_string(command.get("shard_id"), f"{base}.shard_id")
            if shard_id not in shard_set:
                raise invalid_input("shard_id is not declared", f"{base}.shard_id")
            if "node_id" in command and node_set:
                node_id = require_string(command["node_id"], f"{base}.node_id")
                if node_id not in node_set:
                    raise invalid_input("node_id is not declared", f"{base}.node_id")
        elif op in {"renew_lease", "release_lease"}:
            require_string(command.get("shard_id"), f"{base}.shard_id")
            require_positive_int(command.get("generation"), f"{base}.generation")
        elif op in {"advance_clock_ms", "sleep_ms"}:
            require_nonnegative_int(command.get("milliseconds"), f"{base}.milliseconds")
        elif op == "run_sampling":
            if "force_full" in command and not isinstance(command["force_full"], bool):
                raise invalid_input("force_full must be boolean", f"{base}.force_full")
    return commands


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise invalid_input("value must be a JSON object", field)
    return value


def require_positive_int(value: Any, field: str) -> int:
    if not is_int(value) or value <= 0:
        raise invalid_input("value must be a positive integer", field)
    return value


def require_nonnegative_int(value: Any, field: str) -> int:
    if not is_int(value) or value < 0:
        raise invalid_input("value must be a non-negative integer", field)
    return value


def require_string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise invalid_input("value must be a non-empty string", field)
    return value


def validate_topology(data: Any) -> dict[str, Any]:
    data = require_dict(data, "topology")
    global_concurrency = require_positive_int(
        data.get("global_concurrency"), "topology.global_concurrency"
    )
    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise invalid_input("nodes must be a non-empty list", "topology.nodes")

    nodes: list[dict[str, Any]] = []
    seen_node_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, raw_node in enumerate(raw_nodes):
        base = f"topology.nodes[{index}]"
        node = require_dict(raw_node, base)
        node_id = require_string(node.get("node_id"), f"{base}.node_id")
        if node_id in seen_node_ids:
            raise invalid_input("duplicate node_id", f"{base}.node_id")
        host = require_string(node.get("host", "127.0.0.1"), f"{base}.host")
        if host != "127.0.0.1":
            raise invalid_input("worker host must be a 127.0.0.1 loopback address", f"{base}.host")
        port = require_positive_int(node.get("port"), f"{base}.port")
        if port > 65535:
            raise invalid_input("port must be <= 65535", f"{base}.port")
        endpoint = (host, port)
        if port in seen_ports:
            raise invalid_input("duplicate loopback port", f"{base}.port", endpoint=endpoint)
        quota = require_positive_int(
            node.get("concurrency_quota"), f"{base}.concurrency_quota"
        )
        lease_limit = require_positive_int(node.get("lease_limit"), f"{base}.lease_limit")
        seen_node_ids.add(node_id)
        seen_ports.add(port)
        nodes.append(
            {
                "node_id": node_id,
                "host": host,
                "port": port,
                "concurrency_quota": quota,
                "lease_limit": lease_limit,
            }
        )

    raw_shards = data.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise invalid_input("shards must be a non-empty list", "topology.shards")
    shards: list[str] = []
    seen_shards: set[str] = set()
    for index, shard_id in enumerate(raw_shards):
        field = f"topology.shards[{index}]"
        shard_id = require_string(shard_id, field)
        if shard_id in seen_shards:
            raise invalid_input("duplicate shard_id", field)
        seen_shards.add(shard_id)
        shards.append(shard_id)

    routing = data.get("routing") or {}
    if not isinstance(routing, dict):
        raise invalid_input("routing must be an object", "topology.routing")
    node_ids = {node["node_id"] for node in nodes}
    for shard_id, node_id in routing.items():
        if shard_id not in seen_shards:
            raise invalid_input("routing key is not a declared shard", f"topology.routing.{shard_id}")
        if not isinstance(node_id, str) or node_id not in node_ids:
            raise invalid_input(
                "routing value must reference a declared node",
                f"topology.routing.{shard_id}",
            )
    normalized_routing = {shard: routing.get(shard) for shard in shards}
    cursor = 0
    for shard in shards:
        if normalized_routing[shard] is None:
            normalized_routing[shard] = nodes[cursor % len(nodes)]["node_id"]
            cursor += 1

    return {
        "global_concurrency": global_concurrency,
        "nodes": nodes,
        "shards": shards,
        "routing": normalized_routing,
    }


def validate_inference(data: Any, *, vocab_size: int) -> dict[str, Any]:
    data = require_dict(data, "inference")
    topic_count = require_positive_int(data.get("topic_count"), "inference.topic_count")
    if topic_count > vocab_size:
        raise invalid_input(
            "topic_count must not exceed controlled vocabulary size",
            "inference.topic_count",
            topic_count=topic_count,
            vocab_size=vocab_size,
        )
    sample_rounds = require_nonnegative_int(
        data.get("sample_rounds"), "inference.sample_rounds"
    )
    max_rounds = require_nonnegative_int(
        data.get("max_rounds_per_shard"), "inference.max_rounds_per_shard"
    )
    if sample_rounds > max_rounds:
        raise invalid_input(
            "sample_rounds exceeds max_rounds_per_shard",
            "inference.sample_rounds",
            sample_rounds=sample_rounds,
            max_rounds_per_shard=max_rounds,
        )
    budget = require_nonnegative_int(
        data.get("sample_point_budget"), "inference.sample_point_budget"
    )
    lease_ttl_ms = require_nonnegative_int(data.get("lease_ttl_ms"), "inference.lease_ttl_ms")
    interaction_budget_ms = require_positive_int(
        data.get("interaction_budget_ms"), "inference.interaction_budget_ms"
    )
    version = require_positive_int(
        data.get("topic_parameter_version"), "inference.topic_parameter_version"
    )
    seed = data.get("topic_parameter_seed")
    if not is_int(seed):
        raise invalid_input(
            "topic_parameter_seed must be an integer",
            "inference.topic_parameter_seed",
        )
    fixed_point = require_dict(data.get("fixed_point"), "inference.fixed_point")
    tier = require_string(fixed_point.get("bit_width_tier"), "inference.fixed_point.bit_width_tier")
    if tier not in {"I32", "I64"}:
        raise invalid_input("bit_width_tier must be I32 or I64", "inference.fixed_point.bit_width_tier")
    scale = require_positive_int(fixed_point.get("scale"), "inference.fixed_point.scale")
    if tier == "I32" and scale > 1_000_000:
        raise invalid_input(
            "I32 fixed-point scale is too large for the declared safety bound",
            "inference.fixed_point.scale",
        )
    return {
        "topic_count": topic_count,
        "sample_rounds": sample_rounds,
        "max_rounds_per_shard": max_rounds,
        "sample_point_budget": budget,
        "lease_ttl_ms": lease_ttl_ms,
        "interaction_budget_ms": interaction_budget_ms,
        "topic_parameter_version": version,
        "topic_parameter_seed": seed,
        "fixed_point": {"bit_width_tier": tier, "scale": scale},
    }


def validate_vocab(raw_vocab: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_vocab, list) or not raw_vocab:
        raise invalid_input("vocab must be a non-empty list", "sample.vocab")
    vocab: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for index, item in enumerate(raw_vocab):
        base = f"sample.vocab[{index}]"
        entry = require_dict(item, base)
        term_id = require_nonnegative_int(entry.get("term_id"), f"{base}.term_id")
        if term_id in seen_ids:
            raise invalid_input("duplicate term_id", f"{base}.term_id")
        term = require_string(entry.get("term"), f"{base}.term")
        seen_ids.add(term_id)
        vocab.append({"term_id": term_id, "term": term})
    return sorted(vocab, key=lambda item: item["term_id"])


def validate_document(
    raw_doc: Any,
    field_prefix: str,
    *,
    shard_ids: set[str],
    term_ids: set[int],
    last_revisions: dict[str, int] | None = None,
    seen_doc_ids: set[str] | None = None,
) -> dict[str, Any]:
    doc = require_dict(raw_doc, field_prefix)
    doc_id = require_string(doc.get("doc_id"), f"{field_prefix}.doc_id")
    if seen_doc_ids is not None:
        if doc_id in seen_doc_ids:
            raise invalid_input("duplicate doc_id", f"{field_prefix}.doc_id", doc_id=doc_id)
        seen_doc_ids.add(doc_id)
    shard_id = require_string(doc.get("shard_id"), f"{field_prefix}.shard_id")
    if shard_id not in shard_ids:
        raise invalid_input("shard_id is not declared", f"{field_prefix}.shard_id", shard_id=shard_id)
    revision = require_positive_int(doc.get("revision"), f"{field_prefix}.revision")
    if last_revisions is not None:
        previous = last_revisions.get(doc_id)
        if previous is not None and revision < previous:
            raise invalid_input(
                "document revision must be non-decreasing",
                f"{field_prefix}.revision",
                doc_id=doc_id,
                previous_revision=previous,
            )
        last_revisions[doc_id] = revision
    raw_terms = doc.get("term_ids")
    if not isinstance(raw_terms, list):
        raise invalid_input("term_ids must be a list", f"{field_prefix}.term_ids")
    terms: list[int] = []
    for term_index, term_id in enumerate(raw_terms):
        term_field = f"{field_prefix}.term_ids[{term_index}]"
        if not is_int(term_id):
            raise invalid_input("term_id must be an integer", term_field)
        if term_id not in term_ids:
            raise invalid_input("term_id is outside controlled vocabulary", term_field, term_id=term_id)
        terms.append(term_id)
    return {
        "doc_id": doc_id,
        "shard_id": shard_id,
        "revision": revision,
        "term_ids": terms,
    }


def validate_document_stream(data: Any, *, shard_ids: Iterable[str], term_ids: Iterable[int]) -> list[dict[str, Any]]:
    data = require_dict(data, "sample")
    vocab = validate_vocab(data.get("vocab"))
    vocab_ids = {item["term_id"] for item in vocab}
    supplied_term_ids = set(term_ids)
    if supplied_term_ids and vocab_ids != supplied_term_ids:
        # Cross-check when caller supplies the vocabulary from another document.
        missing = supplied_term_ids - vocab_ids
        if missing:
            raise invalid_input("sample vocabulary does not include configured term ids", "sample.vocab")
    effective_term_ids = vocab_ids or set(term_ids)
    shard_set = set(shard_ids)
    raw_docs = data.get("documents")
    if not isinstance(raw_docs, list):
        raise invalid_input("documents must be a list", "sample.documents")
    seen: set[str] = set()
    revisions: dict[str, int] = {}
    documents: list[dict[str, Any]] = []
    for index, raw_doc in enumerate(raw_docs):
        doc = validate_document(
            raw_doc,
            f"sample.documents[{index}]",
            shard_ids=shard_set,
            term_ids=effective_term_ids,
            last_revisions=revisions,
            seen_doc_ids=seen,
        )
        documents.append(doc)
    return documents
