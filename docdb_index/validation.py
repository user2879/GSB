"""Strict, all-or-nothing loading of collection/index declarations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .canonical import CanonicalError, canonical_json, is_number, loads_strict
from .errors import DocumentError
from .models import CollectionSpec, IndexSpec

_PATH_PART = re.compile(r"^[A-Za-z0-9_]+$")


def field_path_parts(path: Any, *, primary_key: Any = None) -> tuple[str, ...]:
    if not isinstance(path, str) or not path:
        raise DocumentError(
            "field path must be a non-empty dot-separated string",
            primary_key=primary_key,
            field_path=str(path) if path is not None else "$",
        )
    parts = tuple(path.split("."))
    if any(not part or not _PATH_PART.fullmatch(part) for part in parts):
        raise DocumentError(
            "field path fragments may contain only letters, digits and underscores",
            primary_key=primary_key,
            field_path=path,
        )
    return parts


def get_path(value: Any, path: str) -> tuple[bool, Any]:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return (False, None)
        current = current[part]
    return (True, current)


def set_path(document: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current: dict[str, Any] = document
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise DocumentError(
                "cannot set a nested field through a non-object value",
                field_path=path,
            )
        current = child
    current[parts[-1]] = value


def read_json_file(path: str | Path) -> Any:
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DocumentError(f"cannot read declaration file: {exc}", field_path="$") from exc
    try:
        return loads_strict(text)
    except (CanonicalError, ValueError) as exc:
        raise DocumentError(f"invalid JSON declaration: {exc}", field_path="$") from exc


def _collection_items(root: Any) -> list[dict[str, Any]]:
    if isinstance(root, list):
        items = root
    elif isinstance(root, dict) and isinstance(root.get("collections"), list):
        items = root["collections"]
    elif isinstance(root, dict) and (
        "documents" in root or "docs" in root
    ):
        items = [root]
    else:
        raise DocumentError(
            "declaration must contain a collections array or one collection object",
            field_path="$",
        )
    if not all(isinstance(item, dict) for item in items):
        raise DocumentError("each collection declaration must be an object", field_path="$")
    return items


def _single_index(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("indexes")
    if raw is None:
        raw = item.get("index_declarations")
    if raw is None:
        raw = item.get("index_declaration")
    if raw is None:
        raw = item.get("index")
    if raw is None and ("path" in item or "field_path" in item or "target_path" in item):
        raw = item
    if isinstance(raw, list):
        if len(raw) != 1:
            raise DocumentError(
                "the component supports exactly one index declaration per collection",
                field_path="indexes",
            )
        raw = raw[0]
    if not isinstance(raw, dict):
        raise DocumentError("missing index declaration", field_path="indexes")
    return raw


def _index_spec(raw: dict[str, Any], collection_name: str) -> IndexSpec:
    path = raw.get("path", raw.get("field_path", raw.get("target_path")))
    field_path_parts(path)
    capacity = raw.get(
        "segment_capacity",
        raw.get("capacity", raw.get("segment_size")),
    )
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise DocumentError(
            "segment capacity must be a positive integer",
            field_path=str(path) if path is not None else "segment_capacity",
        )
    allow_null = raw.get("allow_null", raw.get("allow_nulls", False))
    if not isinstance(allow_null, bool):
        raise DocumentError(
            "allow_null must be a boolean",
            field_path=str(path) if path is not None else "allow_null",
        )
    return IndexSpec(path=path, segment_capacity=capacity, allow_null=allow_null)


def _extract_pk(document: dict[str, Any], primary_key: str) -> tuple[bool, Any]:
    return get_path(document, primary_key)


def _validate_pk(value: Any, field_path: str) -> None:
    if value is None or isinstance(value, bool) or not (
        is_number(value) or (isinstance(value, str) and value != "")
    ):
        raise DocumentError(
            "primary key must be a non-empty string or a finite number",
            primary_key=value,
            field_path=field_path,
        )


def validate_document(
    document: Any,
    *,
    primary_key: str,
    indexes: tuple[IndexSpec, ...],
) -> tuple[Any, list[IndexEntry]]:
    from .models import IndexEntry

    if not isinstance(document, dict):
        raise DocumentError("document must be an object", field_path="$")
    present, pk = _extract_pk(document, primary_key)
    if not present:
        raise DocumentError(
            "primary key field is missing",
            primary_key=None,
            field_path=primary_key,
        )
    _validate_pk(pk, primary_key)
    entries: list[IndexEntry] = []
    for index in indexes:
        path_present, raw_value = get_path(document, index.path)
        if not path_present or raw_value is None:
            if index.allow_null:
                continue
            raise DocumentError(
                "indexed field is missing or null but allow_null is false",
                primary_key=pk,
                field_path=index.path,
            )
        if not is_number(raw_value):
            raise DocumentError(
                "indexed field must be a finite number",
                primary_key=pk,
                field_path=index.path,
            )
        entries.append(IndexEntry(raw_value, pk))
    return pk, entries


def validate_declarations(root: Any) -> tuple[CollectionSpec, ...]:
    items = _collection_items(root)
    if not items:
        raise DocumentError("at least one collection is required", field_path="$")
    seen_collections: set[str] = set()
    specs: list[CollectionSpec] = []

    for item in items:
        name = item.get("name", item.get("collection_name", item.get("collection")))
        if not isinstance(name, str) or not name:
            raise DocumentError("collection name must be a non-empty string", field_path="$")
        if name in seen_collections:
            raise DocumentError(
                "collection name must be unique",
                field_path="$",
                collection=name,
            )
        seen_collections.add(name)

        version = item.get("version", item.get("collection_version"))
        if not isinstance(version, int) or isinstance(version, bool):
            raise DocumentError(
                "collection version must be an integer",
                field_path="$",
                collection=name,
            )

        primary_key = item.get("primary_key", item.get("pk", "id"))
        field_path_parts(primary_key)

        documents = item.get("documents", item.get("docs"))
        if not isinstance(documents, list):
            raise DocumentError(
                "documents must be an array",
                field_path="$",
                collection=name,
            )

        index = _index_spec(_single_index(item), name)
        indexes = (index,)
        seen_pks: dict[str, Any] = {}
        validated_docs: list[dict[str, Any]] = []
        for document in documents:
            pk, _ = validate_document(
                document,
                primary_key=primary_key,
                indexes=indexes,
            )
            pk_json = canonical_json(pk)
            if pk_json in seen_pks:
                raise DocumentError(
                    "primary key must be unique in a collection",
                    primary_key=pk,
                    field_path=primary_key,
                    collection=name,
                )
            try:
                canonical_json(document)
            except CanonicalError as exc:
                raise DocumentError(
                    f"document is not canonically encodable: {exc}",
                    primary_key=pk,
                    field_path="$",
                ) from exc
            seen_pks[pk_json] = pk
            validated_docs.append(document)

        specs.append(
            CollectionSpec(
                name=name,
                version=version,
                primary_key=primary_key,
                documents=tuple(validated_docs),
                indexes=indexes,
            )
        )
    return tuple(specs)


def load_declarations(path: str | Path) -> tuple[CollectionSpec, ...]:
    return validate_declarations(read_json_file(path))
