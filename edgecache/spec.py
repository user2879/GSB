"""Strong, deterministic loading and validation of local specifications."""

from __future__ import annotations

import binascii
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .errors import EdgeError, spec_error
from .model import (
    DERIVATION_RULES,
    EVICTION_RULES,
    KEY_FIELDS,
    ContentRecord,
    ContentSnapshot,
    Derivation,
    LoadedSpec,
    NodePolicy,
    Policy,
    b64decode,
    canonical_json_bytes,
    derive_bytes,
    sha256_hex,
)


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_NODE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _need_dict(value: Any, field_path: str, path: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise spec_error("value must be an object", path=path, field_path=field_path)
    return value


def _need_list(value: Any, field_path: str, path: str | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise spec_error("value must be an array", path=path, field_path=field_path)
    return value


def _need_str(
    value: Any,
    field_path: str,
    *,
    path: str | None = None,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise spec_error("value must be a non-empty string", path=path, field_path=field_path)
    return value


def read_json_file(path: str | Path) -> Any:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise EdgeError(
            code="E_SPEC",
            message="unable to read specification file",
            field_path="path",
            details={"file": str(source), "error": str(exc)},
        ) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise EdgeError(
            code="E_SPEC",
            message="specification is not valid JSON",
            field_path=f"json.line[{exc.lineno}].column[{exc.colno}]",
            details={"file": str(source), "error": exc.msg},
        ) from exc


def parse_content_records(
    raw_records: Sequence[Any],
    *,
    cycle_error_code: str = "E_SPEC",
) -> ContentSnapshot:
    if not isinstance(raw_records, list):
        raise spec_error("content must be an array", field_path="content")
    if not raw_records:
        raise spec_error("content must not be empty", field_path="content")

    prepared: list[tuple[str, int, dict[str, Any]]] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise spec_error("content entry must be an object", field_path=f"content[{index}]")
        path_value = raw.get("path")
        sort_key = path_value if isinstance(path_value, str) else f"~{index:08d}"
        prepared.append((sort_key, index, raw))
    prepared.sort(key=lambda item: (item[0], item[1]))

    seen_paths: dict[str, int] = {}
    for sort_key, index, raw in prepared:
        path_value = raw.get("path")
        if isinstance(path_value, str):
            if path_value in seen_paths:
                raise spec_error(
                    "duplicate content path",
                    path=path_value,
                    field_path="content[].path",
                    first_index=seen_paths[path_value],
                    duplicate_index=index,
                )
            seen_paths[path_value] = index

    parsed: list[ContentRecord] = []
    for _sort_key, index, raw in prepared:
        path = raw.get("path")
        if not isinstance(path, str) or not path:
            raise spec_error(
                "path must be a non-empty string",
                field_path=f"content[{index}].path",
            )
        if not path.startswith("/") or any(char in path for char in ("?", "#", "\x00")):
            raise spec_error(
                "path must be an absolute cache path without query or fragment",
                path=path,
                field_path=f"content[{index}].path",
            )
        prefix = f"content[{path}]"
        etag = _need_str(raw.get("etag"), f"{prefix}.etag", path=path)
        ttl_ms = raw.get("ttl_ms")
        if not _is_int(ttl_ms) or ttl_ms <= 0:
            raise spec_error(
                "ttl_ms must be a positive integer",
                path=path,
                field_path=f"{prefix}.ttl_ms",
            )

        content_b64 = raw.get("content_b64")
        if not isinstance(content_b64, str):
            raise EdgeError(
                code="E_CONTENT",
                message="content_b64 must be a base64 string",
                path=path,
                field_path=f"{prefix}.content_b64",
            )
        try:
            content_bytes = b64decode(content_b64)
        except (binascii.Error, ValueError) as exc:
            raise EdgeError(
                code="E_CONTENT",
                message="content_b64 is truncated or invalid base64",
                path=path,
                field_path=f"{prefix}.content_b64",
                details={"error": str(exc)},
            ) from exc

        declared_digest = raw.get("content_sha256")
        if not isinstance(declared_digest, str) or not _SHA256_RE.match(declared_digest):
            raise spec_error(
                "content_sha256 must be a 64-character hexadecimal digest",
                path=path,
                field_path=f"{prefix}.content_sha256",
            )
        actual_digest = sha256_hex(content_bytes)
        if actual_digest != declared_digest.lower():
            raise EdgeError(
                code="E_CONTENT",
                message="content bytes do not match content_sha256 (possibly truncated)",
                path=path,
                field_path=f"{prefix}.content_b64",
                details={
                    "declared_sha256": declared_digest.lower(),
                    "actual_sha256": actual_digest,
                },
            )

        derived_raw = raw.get("derived_from", [])
        derived_from = _need_list(derived_raw, f"{prefix}.derived_from", path=path)
        dependency_paths: list[str] = []
        for dep_index, dependency in enumerate(derived_from):
            if not isinstance(dependency, str) or not dependency:
                raise spec_error(
                    "derived_from entries must be non-empty path strings",
                    path=path,
                    field_path=f"{prefix}.derived_from[{dep_index}]",
                )
            if dependency in dependency_paths:
                raise spec_error(
                    "derived_from contains a duplicate path",
                    path=path,
                    field_path=f"{prefix}.derived_from[{dep_index}]",
                    dependency=dependency,
                )
            dependency_paths.append(dependency)

        derivation_raw = raw.get("derivation")
        derivation: Derivation | None = None
        if dependency_paths:
            derivation_dict = _need_dict(derivation_raw, f"{prefix}.derivation", path=path)
            rule = _need_str(
                derivation_dict.get("rule"),
                f"{prefix}.derivation.rule",
                path=path,
            )
            if rule not in DERIVATION_RULES:
                raise spec_error(
                    "unsupported derivation rule",
                    path=path,
                    field_path=f"{prefix}.derivation.rule",
                    allowed=sorted(DERIVATION_RULES),
                )
            separator = derivation_dict.get("separator", "")
            if not isinstance(separator, str):
                raise spec_error(
                    "derivation separator must be a string",
                    path=path,
                    field_path=f"{prefix}.derivation.separator",
                )
            derivation = Derivation(rule=rule, separator=separator)
        elif derivation_raw not in (None, {}):
            raise spec_error(
                "a leaf content entry must not declare a derivation",
                path=path,
                field_path=f"{prefix}.derivation",
            )

        parsed.append(
            ContentRecord(
                path=path,
                etag=etag,
                content=content_bytes,
                ttl_ms=ttl_ms,
                derived_from=tuple(dependency_paths),
                derivation=derivation,
            )
        )

    by_path = {record.path: record for record in parsed}
    for record in sorted(parsed, key=lambda item: item.path):
        for dep_index, dependency in enumerate(record.derived_from):
            if dependency not in by_path:
                raise spec_error(
                    "derived_from references a missing content path",
                    path=record.path,
                    field_path=f"content[{record.path}].derived_from[{dep_index}]",
                    dependency=dependency,
                )

    _validate_acyclic(parsed, by_path, cycle_error_code=cycle_error_code)

    for record in sorted(parsed, key=lambda item: item.path):
        if not record.derived_from:
            continue
        expected = derive_bytes(
            record.derivation.rule if record.derivation else "",
            record.derivation.separator if record.derivation else "",
            [by_path[path] for path in record.derived_from],
        )
        if expected != record.content:
            raise EdgeError(
                code="E_CONTENT",
                message="derived content bytes do not match the declared deterministic rule",
                path=record.path,
                field_path=f"content[{record.path}].content_b64",
                details={
                    "expected_sha256": sha256_hex(expected),
                    "actual_sha256": record.content_sha256,
                    "derived_from": list(record.derived_from),
                    "rule": record.derivation.rule if record.derivation else None,
                },
            )

    snapshot = ContentSnapshot(tuple(parsed))
    snapshot.topological_order()
    return snapshot


def _validate_acyclic(
    records: Sequence[ContentRecord],
    by_path: dict[str, ContentRecord],
    *,
    cycle_error_code: str,
) -> None:
    indegree = {record.path: len(record.derived_from) for record in records}
    dependents: dict[str, list[str]] = {record.path: [] for record in records}
    for record in records:
        for dependency in record.derived_from:
            dependents[dependency].append(record.path)
    ready = sorted(path for path, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        path = ready.pop(0)
        visited += 1
        for child in sorted(dependents[path]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if visited == len(records):
        return

    unresolved = {path for path, degree in indegree.items() if degree > 0}
    cycle = _first_cycle(unresolved, by_path)
    first_path = cycle[0] if cycle else sorted(unresolved)[0]
    raise EdgeError(
        code=cycle_error_code,
        message="derivation relationship contains a cycle",
        path=first_path,
        field_path=f"content[{first_path}].derived_from",
        details={"cycle": cycle, "spec_rejection": "E_SPEC" if cycle_error_code == "E_CYCLE" else None},
    )


def _first_cycle(paths: set[str], by_path: dict[str, ContentRecord]) -> list[str]:
    """Find a cycle using sorted DFS so the reported member is stable."""

    state: dict[str, int] = {}
    stack: list[str] = []
    position: dict[str, int] = {}

    def visit(path: str) -> list[str] | None:
        state[path] = 1
        position[path] = len(stack)
        stack.append(path)
        for dependency in by_path[path].derived_from:
            if dependency not in paths:
                continue
            if state.get(dependency, 0) == 0:
                found = visit(dependency)
                if found:
                    return found
            elif state.get(dependency) == 1:
                return stack[position[dependency] :] + [dependency]
        stack.pop()
        position.pop(path, None)
        state[path] = 2
        return None

    for path in sorted(paths):
        if state.get(path, 0) == 0:
            found = visit(path)
            if found:
                # Rotate so the lexicographically first member is reported first.
                core = found[:-1]
                if not core:
                    return found
                pivot = core.index(min(core))
                rotated = core[pivot:] + core[:pivot]
                return rotated + [rotated[0]]
    return []


def parse_policy(raw_policy: Any) -> Policy:
    policy = _need_dict(raw_policy, "policy")
    version = _need_str(policy.get("version"), "policy.version")
    key_fields_raw = _need_list(policy.get("key_fields"), "policy.key_fields")
    if not key_fields_raw:
        raise spec_error("policy.key_fields must not be empty", field_path="policy.key_fields")
    key_fields: list[str] = []
    for index, field in enumerate(key_fields_raw):
        if not isinstance(field, str) or not field:
            raise spec_error(
                "cache key fields must be non-empty strings",
                field_path=f"policy.key_fields[{index}]",
            )
        if field not in KEY_FIELDS:
            raise spec_error(
                "unsupported cache key field",
                field_path=f"policy.key_fields[{index}]",
                allowed=sorted(KEY_FIELDS),
                value=field,
            )
        if field in key_fields:
            raise spec_error(
                "cache key fields must be unique",
                field_path=f"policy.key_fields[{index}]",
                field=field,
            )
        key_fields.append(field)

    nodes_raw = _need_list(policy.get("nodes"), "policy.nodes")
    if len(nodes_raw) < 3:
        raise spec_error(
            "policy.nodes must declare at least three edge nodes",
            field_path="policy.nodes",
            count=len(nodes_raw),
        )
    nodes: list[NodePolicy] = []
    seen_nodes: set[str] = set()
    indexed_nodes: list[tuple[str, int, dict[str, Any]]] = []
    for index, raw_node in enumerate(nodes_raw):
        node_dict = _need_dict(raw_node, f"policy.nodes[{index}]")
        node_id = node_dict.get("node_id")
        sort_key = node_id if isinstance(node_id, str) else f"~{index:08d}"
        indexed_nodes.append((sort_key, index, node_dict))
    indexed_nodes.sort(key=lambda item: (item[0], item[1]))
    for _sort_key, index, node_dict in indexed_nodes:
        prefix = f"policy.nodes[{index}]"
        node_id = _need_str(node_dict.get("node_id"), f"{prefix}.node_id")
        if not _NODE_ID_RE.match(node_id):
            raise spec_error(
                "node_id contains unsupported characters",
                field_path=f"{prefix}.node_id",
                node_id=node_id,
            )
        if node_id in seen_nodes:
            raise spec_error(
                "node_id must be unique",
                field_path=f"{prefix}.node_id",
                node_id=node_id,
            )
        seen_nodes.add(node_id)
        capacity = node_dict.get("capacity")
        if not _is_int(capacity) or capacity <= 0:
            raise spec_error(
                "capacity must be a positive integer",
                field_path=f"{prefix}.capacity",
                node_id=node_id,
            )
        eviction_rule = _need_str(node_dict.get("eviction_rule"), f"{prefix}.eviction_rule")
        if eviction_rule not in EVICTION_RULES:
            raise spec_error(
                "unsupported eviction rule",
                field_path=f"{prefix}.eviction_rule",
                allowed=sorted(EVICTION_RULES),
                node_id=node_id,
            )
        nodes.append(NodePolicy(node_id=node_id, capacity=capacity, eviction_rule=eviction_rule))
    return Policy(version=version, key_fields=tuple(key_fields), nodes=tuple(sorted(nodes)))


def load_spec(path: str | Path, *, cycle_error_code: str = "E_SPEC") -> LoadedSpec:
    document = _need_dict(read_json_file(path), "$")
    if document.get("schema") not in (None, "edgecache.spec/v1"):
        raise spec_error(
            "unsupported specification schema",
            field_path="schema",
            expected="edgecache.spec/v1",
            actual=document.get("schema"),
        )
    policy = parse_policy(document.get("policy"))
    content = parse_content_records(document.get("content"), cycle_error_code=cycle_error_code)
    return LoadedSpec(content=content, policy=policy, source_path=str(Path(path).resolve()))


def load_content_document(document: dict[str, Any], *, cycle_error_code: str = "E_SPEC") -> ContentSnapshot:
    if not isinstance(document, dict) or "content" not in document:
        raise spec_error("content document must contain a content array", field_path="content")
    return parse_content_records(document["content"], cycle_error_code=cycle_error_code)


def spec_digest(spec: LoadedSpec) -> str:
    return sha256_hex(canonical_json_bytes(spec.to_raw()))
