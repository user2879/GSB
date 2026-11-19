"""Selection sequence parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .canonical import sha256_bytes
from .errors import E_SPEC_INVALID, BusinessError


def parse_sequence(data: str | bytes, source_name: str = "<memory>") -> tuple[tuple[str, ...], str]:
    raw_bytes = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    try:
        loaded: Any = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise BusinessError(
            E_SPEC_INVALID,
            "selection sequence parsing failed",
            {"source": source_name, "yaml_error": str(exc)},
        ) from exc

    if isinstance(loaded, dict):
        unknown = sorted(str(key) for key in loaded if key not in {"version", "selections"})
        if unknown:
            raise BusinessError(
                E_SPEC_INVALID,
                "selection sequence contains unknown keys",
                {"source": source_name, "unknown": unknown},
            )
        if loaded.get("version", 1) != 1:
            raise BusinessError(
                E_SPEC_INVALID,
                "unsupported selection sequence version",
                {"source": source_name, "version": loaded.get("version")},
            )
        loaded = loaded.get("selections")
    if not isinstance(loaded, list):
        raise BusinessError(
            E_SPEC_INVALID,
            "selection sequence must be a list or a mapping with selections",
            {"source": source_name},
        )
    selections: list[str] = []
    for index, item in enumerate(loaded):
        if not isinstance(item, str) or not item.strip():
            raise BusinessError(
                E_SPEC_INVALID,
                "selection token must be a non-empty string",
                {"source": source_name, "index": index},
            )
        selections.append(item.strip())
    return tuple(selections), sha256_bytes(raw_bytes)


def load_sequence(path: str | Path) -> tuple[tuple[str, ...], str]:
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise BusinessError(
            E_SPEC_INVALID,
            "cannot read selection sequence",
            {"path": str(source), "error": str(exc)},
        ) from exc
    return parse_sequence(data, source_name=str(source))


def sequence_json_bytes(selections: tuple[str, ...]) -> bytes:
    value = {"version": 1, "selections": list(selections)}
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")

