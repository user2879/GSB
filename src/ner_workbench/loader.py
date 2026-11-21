from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import sha256_file
from .errors import (
    E_CORPUS_INVALID,
    E_LABEL_LENGTH_MISMATCH,
    E_LABEL_UNKNOWN,
    failure_record,
    failure_sort_key,
)
from .types import CorpusSample, canonical_tag, canonical_type


@dataclass(frozen=True)
class CorpusLoadResult:
    samples: tuple[CorpusSample, ...]
    failures: tuple[dict[str, Any], ...]
    corpus_hash: str


def _split_sequence(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(value.split())
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    raise ValueError(f"{field} must be a space-separated string or list")


def _parse_entities(value: object, length: int, chars: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    if value is None:
        return tuple()
    if not isinstance(value, list):
        raise ValueError("entities must be a list")
    entities: list[dict[str, object]] = []
    last_end = -1
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("entity must be an object")
        start_raw = item.get("start")
        end_raw = item.get("end")
        if isinstance(start_raw, bool) or isinstance(end_raw, bool) or not isinstance(start_raw, int) or not isinstance(end_raw, int):
            raise ValueError("entity start/end must be integers")
        if start_raw < 0 or end_raw < start_raw or end_raw >= length:
            raise ValueError(f"entity span [{start_raw}, {end_raw}] is out of bounds for length {length}")
        if start_raw <= last_end:
            raise ValueError("entity spans must be ordered and non-overlapping")
        try:
            entity_type = canonical_type(str(item.get("type", "")))
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        text = "".join(chars[start_raw : end_raw + 1])
        declared_text = item.get("text")
        if declared_text is not None and str(declared_text) != text:
            raise ValueError(
                f"entity text mismatch at [{start_raw}, {end_raw}]: {declared_text!r} != {text!r}"
            )
        entities.append(
            {
                "start": start_raw,
                "end": end_raw,
                "type": entity_type,
                "text": text,
            }
        )
        last_end = end_raw
    return tuple(entities)


def _validate_gold_label_structure(labels: tuple[str, ...], entities: tuple[dict[str, object], ...]) -> None:
    if not entities:
        if any(label != "O" for label in labels):
            raise ValueError("non-O labels present without gold entities")
        return
    expected = ["O"] * len(labels)
    for entity in entities:
        start = int(entity["start"])
        end = int(entity["end"])
        entity_type = str(entity["type"])
        span = end - start + 1
        if span == 1:
            expected[start] = f"S-{entity_type}"
        else:
            expected[start] = f"B-{entity_type}"
            for position in range(start + 1, end):
                expected[position] = f"I-{entity_type}"
            expected[end] = f"E-{entity_type}"
    if tuple(expected) != labels:
        for index, (actual, wanted) in enumerate(zip(labels, expected)):
            if actual != wanted:
                raise ValueError(
                    f"gold label structure mismatch at position {index}: {actual!r} != {wanted!r}"
                )
        raise ValueError("gold label structure mismatch")


def load_corpus(path: Path) -> CorpusLoadResult:
    samples: list[CorpusSample] = []
    failures: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        failures.append(
            failure_record("@corpus", "load", E_CORPUS_INVALID, type(exc).__name__, f"cannot read corpus: {path}")
        )
        return CorpusLoadResult(tuple(), tuple(sorted(failures, key=failure_sort_key)), sha256_file(path) if path.exists() else "")

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            failures.append(
                failure_record(
                    f"line-{line_number:06d}",
                    "load",
                    E_CORPUS_INVALID,
                    "ValueError",
                    "blank corpus line",
                )
            )
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(
                failure_record(
                    f"line-{line_number:06d}",
                    "load",
                    E_CORPUS_INVALID,
                    type(exc).__name__,
                    f"invalid JSON at line {line_number}: {exc.msg}",
                )
            )
            continue
        if not isinstance(payload, dict):
            failures.append(
                failure_record(
                    f"line-{line_number:06d}",
                    "load",
                    E_CORPUS_INVALID,
                    "ValueError",
                    "corpus record must be an object",
                )
            )
            continue

        sample_id = str(payload.get("sample_id", payload.get("id", ""))).strip()
        if not sample_id:
            sample_id = f"line-{line_number:06d}"
            failures.append(
                failure_record(
                    sample_id,
                    "load",
                    E_CORPUS_INVALID,
                    "ValueError",
                    "missing sample_id",
                )
            )
            continue
        if sample_id in seen_ids:
            failures.append(
                failure_record(
                    sample_id,
                    "load",
                    E_CORPUS_INVALID,
                    "ValueError",
                    f"duplicate sample_id at line {line_number}",
                )
            )
            continue
        seen_ids.add(sample_id)

        try:
            chars = _split_sequence(payload.get("chars"), "chars")
        except ValueError as exc:
            failures.append(
                failure_record(sample_id, "load", E_CORPUS_INVALID, "ValueError", str(exc))
            )
            continue
        if not chars or any(char == "" for char in chars):
            failures.append(
                failure_record(sample_id, "load", E_CORPUS_INVALID, "ValueError", "empty character sequence")
            )
            continue

        raw_labels = payload.get("labels", payload.get("tags"))
        try:
            label_tokens = _split_sequence(raw_labels, "labels")
        except ValueError as exc:
            failures.append(
                failure_record(sample_id, "load", E_CORPUS_INVALID, "ValueError", str(exc))
            )
            continue
        if len(label_tokens) != len(chars):
            failures.append(
                failure_record(
                    sample_id,
                    "load",
                    E_LABEL_LENGTH_MISMATCH,
                    "ValueError",
                    f"chars length {len(chars)} != labels length {len(label_tokens)}",
                )
            )
            continue

        unknown_label: str | None = None
        labels: list[str] = []
        for raw_label in label_tokens:
            try:
                labels.append(canonical_tag(raw_label))
            except ValueError:
                unknown_label = raw_label
                break
        if unknown_label is not None:
            failures.append(
                failure_record(
                    sample_id,
                    "load",
                    E_LABEL_UNKNOWN,
                    "ValueError",
                    f"unknown label: {unknown_label!r}",
                )
            )
            continue

        raw_entity_types = payload.get("entity_types", [])
        if isinstance(raw_entity_types, str):
            raw_entity_types = raw_entity_types.split()
        if not isinstance(raw_entity_types, list):
            failures.append(
                failure_record(
                    sample_id,
                    "load",
                    E_CORPUS_INVALID,
                    "ValueError",
                    "entity_types must be a list or space-separated string",
                )
            )
            continue
        try:
            entity_types = tuple(sorted({canonical_type(str(item)) for item in raw_entity_types}))
            entities = _parse_entities(payload.get("entities"), len(chars), chars)
            _validate_gold_label_structure(tuple(labels), entities)
            declared_types = set(entity_types)
            observed_types = {str(entity["type"]) for entity in entities}
            if declared_types and declared_types != observed_types:
                raise ValueError(
                    f"declared entity_types {sorted(declared_types)} != observed {sorted(observed_types)}"
                )
        except ValueError as exc:
            failures.append(
                failure_record(sample_id, "load", E_CORPUS_INVALID, "ValueError", str(exc))
            )
            continue

        samples.append(
            CorpusSample(
                sample_id=sample_id,
                chars=chars,
                gold_labels=tuple(labels),
                entity_types=tuple(sorted(observed_types)),
                gold_entities=entities,
                source_line=line_number,
            )
        )

    samples.sort(key=lambda sample: sample.sample_id)
    failures.sort(key=failure_sort_key)
    return CorpusLoadResult(
        samples=tuple(samples),
        failures=tuple(failures),
        corpus_hash=sha256_file(path),
    )
