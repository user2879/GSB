"""Strict local loading and validation for corpus, model, config and commands."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from .constants import (
    COMMAND_ALIASES,
    E_COMMAND_INVALID,
    E_CONFIG_INVALID,
    E_CORPUS_INVALID,
    E_MODEL_PARAM_INVALID,
    LABELS,
    TYPE_ALIASES,
    TYPE_CODES,
)
from .models import (
    CommandSpec,
    CorpusSample,
    Failure,
    InteractionConfig,
    ModelParams,
    VocabEntry,
)
from .util import (
    digest,
    is_int,
    is_number,
    number_to_decimal,
    parse_json_text,
)


def _pick(mapping: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _split_sequence(value: Any, *, kind: str) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str):
        raise ValueError(f"{kind} must be a string or array")
    text = value.strip()
    if not text:
        return []
    if any(char.isspace() for char in text):
        return text.split()
    return list(text)


def _normalise_type(value: Any, *, allow_o: bool = False) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    code = value.strip().upper()
    if allow_o and code in {"O", "_", "", "NONE", "-"}:
        return "_"
    return TYPE_ALIASES.get(code) or TYPE_ALIASES.get(value.strip())


def _split_typed_label(value: str) -> tuple[str, str | None]:
    text = value.strip().upper()
    if not text:
        return "", None
    if len(text) == 1:
        return text, None
    for separator in ("-", "_", "/", ":"):
        if separator in text:
            label, possible_type = text.split(separator, 1)
            if len(label) == 1 and label in LABELS:
                return label, _normalise_type(possible_type)
    return text, None


def validate_corpus_sample(
    record: Any,
    *,
    locator: str,
) -> tuple[CorpusSample | None, list[Failure]]:
    failures: list[Failure] = []

    def invalid(message: str, details: dict[str, Any] | None = None) -> None:
        failures.append(
            Failure(
                code=E_CORPUS_INVALID,
                stage="load",
                message=message,
                locator=locator,
                details=details or {},
            )
        )

    if not isinstance(record, dict):
        invalid("corpus row must be a JSON object")
        return None, failures

    sample_value = _pick(record, "sample_id", "id", "sid")
    if not isinstance(sample_value, str) or not sample_value.strip():
        invalid("sample id must be a non-empty string")
        return None, failures
    sample_id = sample_value.strip()

    chars_value = _pick(record, "chars", "tokens", "text")
    if chars_value is None:
        invalid("missing chars field", {"sample_id": sample_id})
        return None, failures
    labels_value = _pick(record, "labels", "bieos", "tags", "label_sequence")
    if labels_value is None:
        invalid("missing labels field", {"sample_id": sample_id})
        return None, failures
    types_value = _pick(record, "types", "entity_types", "ner_types", "type_sequence")
    scalar_type = _pick(record, "entity_type", "type", default=None)

    try:
        chars = _split_sequence(chars_value, kind="chars")
        raw_labels = _split_sequence(labels_value, kind="labels")
    except ValueError as exc:
        invalid(str(exc), {"sample_id": sample_id})
        return None, failures

    if not chars:
        invalid("chars must not be empty", {"sample_id": sample_id})
        return None, failures
    non_chars = [item for item in chars if not isinstance(item, str)]
    if non_chars:
        invalid("every char token must be a string", {"sample_id": sample_id})
        return None, failures
    chars = [item for item in chars]
    if any(len(item) != 1 for item in chars):
        invalid("char tokens must each contain exactly one character", {"sample_id": sample_id})
        return None, failures

    if len(raw_labels) != len(chars):
        invalid(
            "chars and BIESO labels must have equal length",
            {"sample_id": sample_id, "chars": len(chars), "labels": len(raw_labels)},
        )
        return None, failures

    labels: list[str] = []
    typed_types: list[str | None] = []
    for item in raw_labels:
        if not isinstance(item, str):
            invalid("every label must be a string", {"sample_id": sample_id})
            return None, failures
        label, inferred_type = _split_typed_label(item)
        labels.append(label)
        typed_types.append(inferred_type)
    bad_labels = [label for label in labels if label not in LABELS]
    if bad_labels:
        invalid(
            "label outside BIESO enum",
            {"sample_id": sample_id, "labels": sorted(set(bad_labels))},
        )
        return None, failures

    if types_value is None:
        if scalar_type is not None:
            code = _normalise_type(scalar_type)
            if code is None:
                invalid("entity type is outside the supported enum", {"sample_id": sample_id})
                return None, failures
            raw_types: list[Any] = [code] * len(chars)
        elif any(item is not None for item in typed_types):
            raw_types = [item if item is not None else "O" for item in typed_types]
        else:
            invalid("missing types field", {"sample_id": sample_id})
            return None, failures
    else:
        try:
            raw_types = _split_sequence(types_value, kind="types")
        except ValueError as exc:
            invalid(str(exc), {"sample_id": sample_id})
            return None, failures
        if len(raw_types) != len(chars):
            invalid(
                "chars and entity types must have equal length",
                {"sample_id": sample_id, "chars": len(chars), "types": len(raw_types)},
            )
            return None, failures

    types: list[str] = []
    for index, (label, raw_type, inferred_type) in enumerate(
        zip(labels, raw_types, typed_types)
    ):
        if inferred_type is not None and _normalise_type(raw_type, allow_o=False) not in {
            None,
            inferred_type,
        }:
            invalid(
                "typed label conflicts with the entity type column",
                {"sample_id": sample_id, "position": index},
            )
            return None, failures
        if label == "O":
            normalised = _normalise_type(raw_type, allow_o=True)
            if normalised is None:
                invalid(
                    "O position must use O, _, or empty type",
                    {"sample_id": sample_id, "position": index},
                )
                return None, failures
            types.append("_")
            continue
        normalised = _normalise_type(raw_type)
        if normalised is None:
            invalid(
                "entity type outside the supported enum",
                {"sample_id": sample_id, "position": index, "value": str(raw_type)},
            )
            return None, failures
        if inferred_type is not None and normalised != inferred_type:
            invalid(
                "typed label conflicts with the entity type column",
                {"sample_id": sample_id, "position": index},
            )
            return None, failures
        types.append(normalised)

    # Validate exact BIESO structure and type continuity. B-E is accepted as a
    # two-character span; standalone I/E and unterminated B spans are rejected.
    index = 0
    while index < len(labels):
        label = labels[index]
        if label == "O":
            index += 1
            continue
        if label == "S":
            index += 1
            continue
        if label != "B":
            invalid(
                "I/E label has no preceding B",
                {"sample_id": sample_id, "position": index, "label": label},
            )
            return None, failures

        start = index
        entity_type = types[index]
        index += 1
        if index >= len(labels):
            invalid(
                "B span is not terminated by E",
                {"sample_id": sample_id, "position": start},
            )
            return None, failures
        if labels[index] == "E":
            end = index
            index += 1
        elif labels[index] == "I":
            while index < len(labels) and labels[index] == "I":
                if types[index] != entity_type:
                    invalid(
                        "entity type changes inside one span",
                        {"sample_id": sample_id, "position": index},
                    )
                    return None, failures
                index += 1
            if index >= len(labels) or labels[index] != "E":
                invalid(
                    "B/I span is not terminated by E",
                    {"sample_id": sample_id, "position": start},
                )
                return None, failures
            end = index
            index += 1
        else:
            invalid(
                "B must be followed by I or E",
                {"sample_id": sample_id, "position": start, "next_label": labels[index]},
            )
            return None, failures

        span_types = types[start : end + 1]
        if any(value != entity_type for value in span_types):
            invalid(
                "entity type changes inside one span",
                {"sample_id": sample_id, "start": start, "end": end},
            )
            return None, failures

    return (
        CorpusSample(
            sample_id=sample_id,
            chars=tuple(chars),
            labels=tuple(labels),
            types=tuple(types),
        ),
        failures,
    )


def load_corpus(path: Path | str) -> tuple[dict[str, CorpusSample], list[Failure]]:
    path = Path(path)
    samples: dict[str, CorpusSample] = {}
    failures: list[Failure] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {}, [
            Failure(
                code=E_CORPUS_INVALID,
                stage="load",
                message=f"cannot read corpus: {exc}",
                locator=str(path),
            )
        ]

    seen: dict[str, int] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        locator = f"{path}:{line_number}"
        try:
            record = parse_json_text(line)
        except (json.JSONDecodeError, ValueError) as exc:
            failures.append(
                Failure(
                    code=E_CORPUS_INVALID,
                    stage="load",
                    message=f"invalid JSONL row: {exc}",
                    locator=locator,
                )
            )
            continue
        sample, row_failures = validate_corpus_sample(record, locator=locator)
        failures.extend(row_failures)
        if sample is None:
            continue
        if sample.sample_id in seen:
            failures.append(
                Failure(
                    code=E_CORPUS_INVALID,
                    stage="load",
                    message="duplicate sample id",
                    sample_id=sample.sample_id,
                    locator=locator,
                    details={"first_line": seen[sample.sample_id]},
                )
            )
            continue
        seen[sample.sample_id] = line_number
        samples[sample.sample_id] = sample
    return samples, failures


def validate_vocab_entry(
    value: Any,
    *,
    locator: str,
    error_code: str = E_COMMAND_INVALID,
    stage: str = "command",
) -> tuple[VocabEntry | None, list[Failure]]:
    failures: list[Failure] = []
    if not isinstance(value, dict):
        return None, [
            Failure(
                code=error_code,
                stage=stage,
                message="vocabulary entry must be an object",
                locator=locator,
            )
        ]
    entry_id = _pick(value, "id", "entry_id", "vocab_id")
    text = _pick(value, "text", "word", "value")
    type_value = _pick(value, "type", "entity_type", "label_type")
    if not isinstance(entry_id, str) or not entry_id.strip():
        failures.append(
            Failure(
                code=error_code,
                stage=stage,
                message="vocabulary entry id must be a non-empty string",
                locator=locator,
            )
        )
    if not isinstance(text, str) or not text.strip() or any(ch.isspace() for ch in text):
        failures.append(
            Failure(
                code=error_code,
                stage=stage,
                message="vocabulary text must be a non-empty string without whitespace",
                locator=locator,
            )
        )
    type_code = _normalise_type(type_value)
    if type_code is None:
        failures.append(
            Failure(
                code=error_code,
                stage=stage,
                message="vocabulary type outside the supported enum",
                locator=locator,
            )
        )
    if failures:
        return None, failures
    return VocabEntry(entry_id.strip(), text, type_code), []


def _coerce_vector(value: Any, *, length: int, name: str) -> list[Decimal]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    if len(value) != length:
        raise ValueError(f"{name} has dimension {len(value)}, expected {length}")
    result: list[Decimal] = []
    for item in value:
        if not is_number(item):
            raise ValueError(f"{name} contains a non-numeric value")
        result.append(number_to_decimal(item))
    return result


def _normalise_vocabulary(
    value: Any,
    *,
    path: Path,
) -> tuple[dict[str, VocabEntry], list[Failure]]:
    failures: list[Failure] = []
    entries: dict[str, VocabEntry] = {}
    if value is None:
        return entries, failures
    values: list[Any]
    if isinstance(value, list):
        values = value
    elif isinstance(value, dict):
        values = []
        for entry_id, item in sorted(value.items()):
            if isinstance(item, dict):
                candidate = dict(item)
                candidate.setdefault("id", entry_id)
            else:
                candidate = {"id": entry_id, "text": str(item)}
            values.append(candidate)
    else:
        return {}, [
            Failure(
                code=E_MODEL_PARAM_INVALID,
                stage="load",
                message="vocabulary must be an array or object",
                locator=str(path),
            )
        ]

    for index, value in enumerate(values):
        entry, entry_failures = validate_vocab_entry(
            value,
            locator=f"{path}#vocabulary[{index}]",
            error_code=E_MODEL_PARAM_INVALID,
            stage="load",
        )
        failures.extend(entry_failures)
        if entry is None:
            continue
        if entry.entry_id in entries:
            failures.append(
                Failure(
                    code=E_MODEL_PARAM_INVALID,
                    stage="load",
                    message="duplicate vocabulary entry id",
                    locator=f"{path}#vocabulary[{index}]",
                    details={"entry_id": entry.entry_id},
                )
            )
            continue
        entries[entry.entry_id] = entry
    return entries, failures


def load_model(path: Path | str) -> tuple[ModelParams | None, list[Failure]]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
        raw = parse_json_text(text)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, [
            Failure(
                code=E_MODEL_PARAM_INVALID,
                stage="load",
                message=f"cannot parse model package: {exc}",
                locator=str(path),
            )
        ]
    if not isinstance(raw, dict):
        return None, [
            Failure(
                code=E_MODEL_PARAM_INVALID,
                stage="load",
                message="model package must be a JSON object",
                locator=str(path),
            )
        ]

    failures: list[Failure] = []

    def invalid(message: str, details: dict[str, Any] | None = None) -> None:
        failures.append(
            Failure(
                code=E_MODEL_PARAM_INVALID,
                stage="load",
                message=message,
                locator=str(path),
                details=details or {},
            )
        )

    version = _pick(raw, "model_version", "version", "parameter_version")
    if not isinstance(version, str) or not version.strip():
        invalid("model version must be a non-empty string")

    feature_dim = _pick(raw, "feature_dim", "feature_dimension", "dim")
    hash_buckets = _pick(raw, "hash_buckets", "hash_bucket_count", "buckets")
    if not is_int(feature_dim) or feature_dim <= 0:
        invalid("feature_dim must be a positive integer")
    if not is_int(hash_buckets) or hash_buckets <= 0:
        invalid("hash_buckets must be a positive integer")

    raw_labels = _pick(raw, "labels", "label_set", default=list(LABELS))
    label_order: list[str] = []
    if not isinstance(raw_labels, list) or not all(isinstance(item, str) for item in raw_labels):
        invalid("labels must be an array of strings")
    elif set(raw_labels) != set(LABELS) or len(raw_labels) != len(LABELS):
        invalid("labels must contain B, I, E, S and O exactly once")
    else:
        label_order = list(raw_labels)

    embeddings_value = _pick(
        raw, "char_embeddings", "embedding_hash", "embeddings", "char_embedding_hash"
    )
    embeddings: dict[str, list[Decimal]] = {}
    if not isinstance(embeddings_value, dict):
        invalid("char_embeddings must be an object")
    elif is_int(feature_dim) and feature_dim > 0:
        for key, vector in sorted(embeddings_value.items()):
            if not isinstance(key, str):
                invalid("char embedding keys must be strings")
                continue
            try:
                embeddings[key] = _coerce_vector(
                    vector,
                    length=feature_dim,
                    name=f"char_embeddings[{key!r}]",
                )
            except ValueError as exc:
                invalid(str(exc))

    emission_value = _pick(raw, "emission_weights", "emissions")
    emission_weights: list[list[Decimal]] = []
    if not isinstance(emission_value, list):
        invalid("emission_weights must be an array")
    elif is_int(feature_dim) and feature_dim > 0 and label_order:
        if len(emission_value) != feature_dim:
            invalid(
                "emission_weights row count must equal feature_dim"
                " and the embedding vector dimension",
                {"rows": len(emission_value), "feature_dim": feature_dim},
            )
        else:
            for row_index, row in enumerate(emission_value):
                try:
                    values = _coerce_vector(
                        row,
                        length=len(label_order),
                        name=f"emission_weights[{row_index}]",
                    )
                except ValueError as exc:
                    invalid(str(exc))
                    continue
                emission_weights.append(values)
            if len(emission_weights) == feature_dim:
                # Reorder columns from the package label order to canonical order.
                positions = [label_order.index(label) for label in LABELS]
                emission_weights = [
                    [row[index] for index in positions] for row in emission_weights
                ]

    transition_value = _pick(raw, "transition_weights", "transitions")
    transition_weights: list[list[Decimal]] = []
    if not isinstance(transition_value, list):
        invalid("transition_weights must be an array")
    elif label_order:
        if len(transition_value) != len(LABELS):
            invalid("transition_weights must be a square 5x5 matrix")
        else:
            matrix: list[list[Decimal]] = []
            for row_index, row in enumerate(transition_value):
                try:
                    matrix.append(
                        _coerce_vector(
                            row,
                            length=len(LABELS),
                            name=f"transition_weights[{row_index}]",
                        )
                    )
                except ValueError as exc:
                    invalid(str(exc))
                    break
            if len(matrix) == len(LABELS):
                old_positions = [label_order.index(label) for label in LABELS]
                transition_weights = [
                    [matrix[row][column] for column in old_positions]
                    for row in old_positions
                ]

    start_value = _pick(raw, "start_weights", "initial_weights", default=None)
    if start_value is None:
        start_weights = [Decimal(0)] * len(LABELS)
    elif label_order:
        try:
            raw_start = _coerce_vector(
                start_value,
                length=len(LABELS),
                name="start_weights",
            )
            old_positions = [label_order.index(label) for label in LABELS]
            start_weights = [raw_start[index] for index in old_positions]
        except ValueError as exc:
            invalid(str(exc))
            start_weights = []
    else:
        start_weights = []

    default_type = _normalise_type(
        _pick(raw, "default_entity_type", "default_type", default="ORG")
    )
    if default_type is None:
        invalid("default entity type outside the supported enum")
        default_type = "ORG"

    vocabulary, vocabulary_failures = _normalise_vocabulary(
        _pick(raw, "vocabulary", "vocab", "controlled_vocabulary", default=None),
        path=path,
    )
    failures.extend(vocabulary_failures)

    if failures:
        return None, sorted(failures, key=Failure.sort_key)

    assert isinstance(version, str)
    assert isinstance(feature_dim, int)
    assert isinstance(hash_buckets, int)
    return (
        ModelParams(
            version=version.strip(),
            feature_dim=feature_dim,
            hash_buckets=hash_buckets,
            labels=LABELS,
            char_embeddings=embeddings,
            emission_weights=emission_weights,
            transition_weights=transition_weights,
            start_weights=start_weights,
            default_entity_type=default_type,
            vocabulary=vocabulary,
        ),
        [],
    )


def load_config(
    path: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> tuple[InteractionConfig | None, list[Failure]]:
    path = Path(path)
    try:
        raw = parse_json_text(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return None, [
            Failure(
                code=E_CONFIG_INVALID,
                stage="load",
                message=f"cannot parse interaction config: {exc}",
                locator=str(path),
            )
        ]
    if not isinstance(raw, dict):
        return None, [
            Failure(
                code=E_CONFIG_INVALID,
                stage="load",
                message="interaction config must be a JSON object",
                locator=str(path),
            )
        ]
    failures: list[Failure] = []

    def invalid(message: str) -> None:
        failures.append(
            Failure(
                code=E_CONFIG_INVALID,
                stage="load",
                message=message,
                locator=str(path),
            )
        )

    budget = _pick(
        raw,
        "interaction_latency_budget_ms",
        "latency_budget_ms",
        "interaction_budget_ms",
    )
    if not is_number(budget) or not Decimal(str(budget)).is_finite() or budget < 0:
        invalid("interaction latency budget must be a finite non-negative number")

    decimals = _pick(raw, "fixed_decimals", "fixed_point_decimals", "decimal_places")
    if not is_int(decimals) or not 0 <= decimals <= 12:
        invalid("fixed decimals must be an integer in [0, 12]")

    quant_level = _pick(raw, "quantization_level", "quantization_tier", "quant_level")
    if not is_int(quant_level) or quant_level <= 0:
        invalid("quantization level must be a positive integer")

    levels_value = _pick(raw, "quantization_levels", "quant_levels", default=None)
    if levels_value is None:
        levels = [quant_level] if is_int(quant_level) and quant_level > 0 else []
    elif isinstance(levels_value, list) and all(
        is_int(item) and item > 0 for item in levels_value
    ):
        levels = sorted(set(levels_value))
        if is_int(quant_level) and quant_level > 0 and quant_level not in levels:
            invalid("active quantization level must occur in quantization_levels")
    else:
        invalid("quantization_levels must be an array of positive integers")
        levels = []

    workers = _pick(raw, "thread_pool_workers", "thread_pool_size", "workers", "concurrency")
    if not is_int(workers) or workers <= 0:
        invalid("thread pool workers must be a positive integer")

    interval = _pick(
        raw,
        "snapshot_commit_interval_commands",
        "snapshot_interval",
        "snapshot_commit_interval",
    )
    if not is_int(interval) or interval <= 0:
        invalid("snapshot commit interval must be a positive integer")

    tolerance = _pick(raw, "boundary_score_tolerance", "boundary_tolerance", "tolerance")
    if not is_number(tolerance) or not Decimal(str(tolerance)).is_finite() or tolerance < 0:
        invalid("boundary score tolerance must be a finite non-negative number")

    default_type = _normalise_type(
        _pick(raw, "default_entity_type", "default_type", default="ORG")
    )
    if default_type is None:
        invalid("default entity type outside LOC/ORG/PER/TIME")

    temp_dir = _pick(raw, "temp_dir", "temporary_dir", default="artifacts/tmp")
    if not isinstance(temp_dir, str) or not temp_dir.strip():
        invalid("temp_dir must be a non-empty string")
        temp_dir = "artifacts/tmp"
    temp_path = Path(temp_dir)
    if not temp_path.is_absolute() and base_dir is not None:
        temp_path = Path(base_dir) / temp_path

    if failures:
        return None, failures

    return (
        InteractionConfig(
            latency_budget_ms=float(budget),
            fixed_decimals=int(decimals),
            quantization_level=int(quant_level),
            quantization_levels=tuple(levels),
            thread_pool_workers=int(workers),
            snapshot_commit_interval_commands=int(interval),
            boundary_score_tolerance=Decimal(str(tolerance)),
            temp_dir=str(temp_path),
            default_entity_type=default_type or "ORG",
        ),
        [],
    )


def _command_weight(payload: dict[str, Any]) -> tuple[bool, Any]:
    for field in ("delta", "value", "weight", "weight_delta"):
        if field in payload:
            return True, payload[field]
    target = payload.get("target")
    if isinstance(target, dict):
        for field in ("delta", "value", "weight", "weight_delta"):
            if field in target:
                return True, target[field]
    return False, None


def _extract_command(raw: Any, position: int) -> tuple[CommandSpec | None, list[Failure]]:
    locator = f"commands[{position}]"
    if not isinstance(raw, dict):
        return None, [
            Failure(
                code=E_COMMAND_INVALID,
                stage="command",
                message="command must be a JSON object",
                command_position=position,
                locator=locator,
            )
        ]
    name = _pick(raw, "type", "command", "op", "action")
    if not isinstance(name, str):
        return None, [
            Failure(
                code=E_COMMAND_INVALID,
                stage="command",
                message="command type is required",
                command_position=position,
                locator=locator,
            )
        ]
    command_type = COMMAND_ALIASES.get(name.strip().lower())
    if command_type is None:
        return None, [
            Failure(
                code=E_COMMAND_INVALID,
                stage="command",
                message="command type outside the allowed enum",
                command_position=position,
                locator=locator,
                details={"type": name},
            )
        ]
    payload = dict(raw)
    payload["type"] = command_type
    sequence = raw.get("seq", raw.get("sequence"))
    if sequence is not None and not is_int(sequence):
        return None, [
            Failure(
                code=E_COMMAND_INVALID,
                stage="command",
                message="command sequence must be an integer when present",
                command_position=position,
                locator=locator,
            )
        ]

    failures: list[Failure] = []

    def invalid(message: str, details: dict[str, Any] | None = None) -> None:
        failures.append(
            Failure(
                code=E_COMMAND_INVALID,
                stage="command",
                message=message,
                command_position=position,
                locator=locator,
                details=details or {},
            )
        )

    if command_type == "vocab_add":
        entry_value = _pick(raw, "entry", "vocabulary_entry", "vocab")
        entry, entry_failures = validate_vocab_entry(
            entry_value,
            locator=f"{locator}.entry",
        )
        failures.extend(entry_failures)
        if entry is not None:
            payload["entry"] = entry.as_dict()
    elif command_type == "vocab_delete":
        entry_id = _pick(raw, "entry_id", "id", "target_entry_id", "target")
        if isinstance(entry_id, dict):
            entry_id = _pick(entry_id, "entry_id", "id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            invalid("vocabulary delete requires a target entry id")
        else:
            payload["entry_id"] = entry_id.strip()
    elif command_type == "vocab_replace":
        entry_id = _pick(raw, "entry_id", "id", "target_entry_id", "target")
        if isinstance(entry_id, dict):
            entry_id = _pick(entry_id, "entry_id", "id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            invalid("vocabulary replace requires a target entry id")
        entry_value = _pick(raw, "entry", "replacement", "vocabulary_entry")
        entry, entry_failures = validate_vocab_entry(
            entry_value,
            locator=f"{locator}.entry",
        )
        failures.extend(entry_failures)
        if isinstance(entry_id, str) and entry_id.strip() and entry is not None:
            payload["entry_id"] = entry_id.strip()
            payload["entry"] = entry.as_dict()
    elif command_type == "corpus_replace":
        sample_value = _pick(raw, "sample", "corpus_sample", "data", "record")
        if sample_value is None:
            invalid("corpus replace requires a sample object")
        else:
            target_id = _pick(raw, "sample_id", "target_sample_id", "target")
            if isinstance(target_id, dict):
                target_id = _pick(target_id, "sample_id", "id")
            sample, sample_failures = validate_corpus_sample(
                sample_value,
                locator=f"{locator}.sample",
            )
            if sample_failures:
                for failure in sample_failures:
                    failure.command_position = position
                    failure.details = dict(failure.details)
                    failure.details.setdefault("command_position", position)
                failures.extend(sample_failures)
            if sample is not None:
                if target_id is not None and target_id != sample.sample_id:
                    invalid("corpus replace target id does not match the sample id")
                else:
                    payload["sample"] = sample.as_dict()
                    payload["sample_id"] = sample.sample_id
    elif command_type == "corpus_delete":
        sample_id = _pick(raw, "sample_id", "id", "target_sample_id", "target")
        if isinstance(sample_id, dict):
            sample_id = _pick(sample_id, "sample_id", "id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            invalid("corpus delete requires a target sample id")
        else:
            payload["sample_id"] = sample_id.strip()
    elif command_type in {"emission_adjust", "transition_adjust"}:
        has_weight, weight = _command_weight(raw)
        if not has_weight:
            invalid("weight adjustment requires a delta/value/weight field")
        elif not is_number(weight):
            invalid("weight adjustment value must be numeric")
        else:
            payload["weight"] = weight
        if command_type == "emission_adjust":
            target = raw.get("target")
            if not isinstance(target, dict):
                target = raw
            dimension = _pick(target, "dimension", "row", "bucket", "feature_index")
            label = _pick(target, "label", "tag", "to_label")
            if not is_int(dimension):
                invalid("emission adjustment requires an integer dimension/row")
            if not isinstance(label, str) or label.strip().upper() not in LABELS:
                invalid("emission adjustment requires a BIESO label")
            if is_int(dimension) and isinstance(label, str):
                payload["dimension"] = dimension
                payload["label"] = label.strip().upper()
        else:
            target = raw.get("target")
            if not isinstance(target, dict):
                target = raw
            from_label = _pick(target, "from", "from_label", "source")
            to_label = _pick(target, "to", "to_label", "destination")
            if (
                not isinstance(from_label, str)
                or from_label.strip().upper() not in LABELS
            ):
                invalid("transition adjustment requires a source BIESO label")
            if not isinstance(to_label, str) or to_label.strip().upper() not in LABELS:
                invalid("transition adjustment requires a destination BIESO label")
            if isinstance(from_label, str) and isinstance(to_label, str):
                payload["from_label"] = from_label.strip().upper()
                payload["to_label"] = to_label.strip().upper()
    elif command_type == "fixed_point_set":
        decimals = _pick(raw, "fixed_decimals", "decimals", "width")
        if not is_int(decimals) or not 0 <= decimals <= 12:
            invalid("fixed point command requires an integer decimals value in [0, 12]")
        else:
            payload["fixed_decimals"] = decimals
    elif command_type == "quantization_set":
        level = _pick(raw, "quantization_level", "level", "tier")
        if not is_int(level) or level <= 0:
            invalid("quantization command requires a positive integer level")
        else:
            payload["quantization_level"] = level
    elif command_type == "quit":
        pass
    elif command_type == "noop":
        pass

    if failures:
        return None, failures
    return (
        CommandSpec(
            position=position,
            command_type=command_type,
            payload=payload,
            sequence=sequence,
        ),
        [],
    )


def load_commands(path: Path | str) -> tuple[list[CommandSpec], list[Failure]]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], [
            Failure(
                code=E_COMMAND_INVALID,
                stage="load",
                message=f"cannot read command stream: {exc}",
                locator=str(path),
            )
        ]

    raw_commands: list[Any]
    parse_error: Exception | None = None
    try:
        parsed = parse_json_text(text)
        if isinstance(parsed, list):
            raw_commands = parsed
        elif isinstance(parsed, dict) and isinstance(parsed.get("commands"), list):
            raw_commands = parsed["commands"]
        elif isinstance(parsed, dict):
            raw_commands = [parsed]
        else:
            raise ValueError("command file must be an array or an object with commands")
    except (json.JSONDecodeError, ValueError) as exc:
        parse_error = exc
        raw_commands = []
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                raw_commands.append(parse_json_text(line))
            except (json.JSONDecodeError, ValueError):
                raw_commands = []
                break

    if parse_error is not None and not raw_commands:
        return [], [
            Failure(
                code=E_COMMAND_INVALID,
                stage="load",
                message=f"cannot parse command stream: {parse_error}",
                locator=str(path),
            )
        ]

    commands: list[CommandSpec] = []
    failures: list[Failure] = []
    for position, raw in enumerate(raw_commands, start=1):
        command, command_failures = _extract_command(raw, position)
        failures.extend(command_failures)
        if command is not None:
            commands.append(command)
    return commands, failures


def model_fingerprint(model: ModelParams, *, include_vocabulary: bool = False) -> str:
    value: dict[str, Any] = {
        "schema": "model-v1",
        "version": model.version,
        "feature_dim": model.feature_dim,
        "hash_buckets": model.hash_buckets,
        "labels": list(model.labels),
        "char_embeddings": {
            key: list(vector) for key, vector in model.char_embeddings.items()
        },
        "emission_weights": model.emission_weights,
        "transition_weights": model.transition_weights,
        "start_weights": model.start_weights,
        "default_entity_type": model.default_entity_type,
    }
    if include_vocabulary:
        value["vocabulary"] = {
            key: entry.as_dict() for key, entry in model.vocabulary.items()
        }
    return digest(value)


def vocabulary_fingerprint(vocabulary: dict[str, VocabEntry]) -> str:
    return digest(
        {
            key: entry.as_dict()
            for key, entry in sorted(vocabulary.items())
        }
    )


def config_fingerprint(config: InteractionConfig) -> str:
    return digest(
        {
            "latency_budget_ms": str(config.latency_budget_ms),
            "fixed_decimals": config.fixed_decimals,
            "quantization_level": config.quantization_level,
            "quantization_levels": list(config.quantization_levels),
            "thread_pool_workers": config.thread_pool_workers,
            "snapshot_commit_interval_commands": config.snapshot_commit_interval_commands,
            "boundary_score_tolerance": config.boundary_score_tolerance,
            "default_entity_type": config.default_entity_type,
        }
    )


def corpus_fingerprint(samples: dict[str, CorpusSample]) -> str:
    return digest({key: sample.as_dict() for key, sample in sorted(samples.items())})

