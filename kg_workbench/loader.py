"""Input loading and schema validation for sentences, models, and configuration."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from .errors import (
    AnnotationRangeError,
    ConfigError,
    ModelShapeError,
    failure_from_exception,
)
from .util import (
    ZERO,
    as_decimal,
    canonical_bytes,
    fixed_hash_bucket,
    load_json,
    require_int,
    sha256_bytes,
    sha256_value,
)


ENTITY_LABEL_PRIORITY = ("B", "I", "E", "S", "O")
STAGE_ORDER = ("load", "entity", "relation", "fusion", "diagnosis")
ALLOWED_FIXED_DECIMALS = (2, 3, 4, 5, 6, 8)
ALLOWED_QUANTIZATION_DECIMALS = (0, 1, 2, 3, 4, 5)
INJECTION_KIND_STAGE = {
    "annotation_out_of_range": "load",
    "missing_pos": "entity",
    "feature_hash_bucket_mismatch": "relation",
    "threshold_equal": "fusion",
}


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} must be a JSON object")
    return value


def _sequence(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{field_name} must be a JSON array")
    return value


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{field_name} must be a non-empty string")
    return value


def _decimal_in_range(
    value: Any, field_name: str, *, lower: Decimal | None = None, upper: Decimal | None = None
) -> Decimal:
    result = as_decimal(value, field_name)
    if lower is not None and result < lower:
        raise ConfigError(f"{field_name} must be >= {lower}")
    if upper is not None and result > upper:
        raise ConfigError(f"{field_name} must be <= {upper}")
    return result


@dataclass(frozen=True)
class FixedPointConfig:
    decimals: int
    quantization_decimals: int
    temperature: Decimal
    confidence_threshold: Decimal
    entity_weight: Decimal
    relation_weight: Decimal
    bias: Decimal
    threshold_equal_policy: str
    tolerance_units: int

    @property
    def scale(self) -> int:
        return 10**self.decimals

    @property
    def quantum(self) -> int:
        return 10 ** (self.decimals - self.quantization_decimals)

    def reference(self) -> "FixedPointConfig":
        reference_decimals = self.decimals + 1
        reference_quantization = min(self.quantization_decimals + 1, reference_decimals)
        return FixedPointConfig(
            decimals=reference_decimals,
            quantization_decimals=reference_quantization,
            temperature=self.temperature,
            confidence_threshold=self.confidence_threshold,
            entity_weight=self.entity_weight,
            relation_weight=self.relation_weight,
            bias=self.bias,
            threshold_equal_policy=self.threshold_equal_policy,
            tolerance_units=self.tolerance_units,
        )


@dataclass(frozen=True)
class WorkbenchConfig:
    source_path: Path
    base_dir: Path
    raw: dict[str, Any]
    config_hash: str
    sentences_path: Path
    model_path: Path
    evidence_path: Path
    injections_path: Path | None
    snapshot_dir: Path
    report_path: Path
    entity_threshold: Decimal
    relation_types: tuple[str, ...]
    trigger_words: dict[str, tuple[str, ...]]
    fixed: FixedPointConfig
    workers: int
    snapshot_commit_interval: int


def _resolve_path(base_dir: Path, value: Any, field_name: str) -> Path:
    raw = _string(value, field_name)
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path)


def load_config(path: str | Path) -> WorkbenchConfig:
    config_path = Path(path).resolve()
    raw = _mapping(load_json(config_path), "config")
    base_dir = config_path.parent

    paths = _mapping(raw.get("paths"), "paths")
    sentences_path = _resolve_path(base_dir, paths.get("sentences"), "paths.sentences")
    model_path = _resolve_path(base_dir, paths.get("model"), "paths.model")
    evidence_path = _resolve_path(base_dir, paths.get("evidence"), "paths.evidence")
    injections_raw = paths.get("injections")
    injections_path = (
        _resolve_path(base_dir, injections_raw, "paths.injections")
        if injections_raw not in (None, "")
        else None
    )
    snapshot_dir = _resolve_path(base_dir, paths.get("snapshot_dir"), "paths.snapshot_dir")
    report_path = _resolve_path(base_dir, paths.get("report"), "paths.report")

    entity = _mapping(raw.get("entity"), "entity")
    entity_threshold = _decimal_in_range(
        entity.get("threshold"), "entity.threshold", lower=ZERO, upper=Decimal(1)
    )

    relation = _mapping(raw.get("relation"), "relation")
    relation_types_raw = _sequence(relation.get("relation_types"), "relation.relation_types")
    relation_types_list = [_string(item, "relation.relation_types[]") for item in relation_types_raw]
    if not relation_types_list:
        raise ConfigError("relation.relation_types must not be empty")
    if len(set(relation_types_list)) != len(relation_types_list):
        raise ConfigError("relation.relation_types contains duplicates")
    relation_types = tuple(sorted(relation_types_list))

    triggers_raw = _mapping(relation.get("trigger_words"), "relation.trigger_words")
    trigger_words: dict[str, tuple[str, ...]] = {}
    for relation_type in sorted(triggers_raw):
        if relation_type not in relation_types:
            raise ConfigError(f"relation.trigger_words contains unknown type {relation_type}")
        values = _sequence(triggers_raw[relation_type], f"relation.trigger_words.{relation_type}")
        words = tuple(sorted({_string(item, "trigger word") for item in values}))
        if not words:
            raise ConfigError(f"relation.trigger_words.{relation_type} must not be empty")
        trigger_words[relation_type] = words
    for relation_type in relation_types:
        trigger_words.setdefault(relation_type, ())

    fixed_raw = _mapping(raw.get("fixed"), "fixed")
    decimals = require_int(fixed_raw.get("decimals"), "fixed.decimals", minimum=0)
    if decimals not in ALLOWED_FIXED_DECIMALS:
        raise ConfigError(
            f"fixed.decimals must be one of {list(ALLOWED_FIXED_DECIMALS)}"
        )
    quantization_raw = fixed_raw.get(
        "quantization_decimals", fixed_raw.get("quantization_tier")
    )
    quantization_decimals = require_int(
        quantization_raw, "fixed.quantization_decimals", minimum=0
    )
    if quantization_decimals not in ALLOWED_QUANTIZATION_DECIMALS:
        raise ConfigError(
            "fixed.quantization_decimals must be one of "
            f"{list(ALLOWED_QUANTIZATION_DECIMALS)}"
        )
    if quantization_decimals > decimals:
        raise ConfigError("fixed.quantization_decimals cannot exceed fixed.decimals")
    threshold_equal_policy = _string(
        fixed_raw.get("threshold_equal_policy"), "fixed.threshold_equal_policy"
    )
    if threshold_equal_policy not in {"accept", "reject"}:
        raise ConfigError(
            "fixed.threshold_equal_policy must be 'accept' or 'reject'"
        )
    entity_weight = _decimal_in_range(
        fixed_raw.get("entity_weight"), "fixed.entity_weight", lower=ZERO, upper=Decimal(10)
    )
    relation_weight = _decimal_in_range(
        fixed_raw.get("relation_weight"), "fixed.relation_weight", lower=ZERO, upper=Decimal(10)
    )
    if entity_weight == 0 and relation_weight == 0:
        raise ConfigError("at least one fusion weight must be positive")
    temperature = _decimal_in_range(
        fixed_raw.get("temperature"), "fixed.temperature", lower=Decimal("0.000001")
    )
    confidence_threshold = _decimal_in_range(
        fixed_raw.get("confidence_threshold"),
        "fixed.confidence_threshold",
        lower=ZERO,
        upper=Decimal(1),
    )
    bias = _decimal_in_range(
        fixed_raw.get("bias"), "fixed.bias", lower=Decimal(-10), upper=Decimal(10)
    )

    diagnostic = _mapping(raw.get("diagnostic"), "diagnostic")
    tolerance_units = require_int(
        diagnostic.get("tolerance_units"), "diagnostic.tolerance_units", minimum=0
    )

    runtime = _mapping(raw.get("runtime"), "runtime")
    workers = require_int(runtime.get("workers"), "runtime.workers", minimum=1)
    if workers > 64:
        raise ConfigError("runtime.workers must be <= 64")
    snapshot_commit_interval = require_int(
        runtime.get("snapshot_commit_interval"),
        "runtime.snapshot_commit_interval",
        minimum=1,
    )

    fixed = FixedPointConfig(
        decimals=decimals,
        quantization_decimals=quantization_decimals,
        temperature=temperature,
        confidence_threshold=confidence_threshold,
        entity_weight=entity_weight,
        relation_weight=relation_weight,
        bias=bias,
        threshold_equal_policy=threshold_equal_policy,
        tolerance_units=tolerance_units,
    )
    return WorkbenchConfig(
        source_path=config_path,
        base_dir=base_dir,
        raw=raw,
        config_hash=sha256_value(raw),
        sentences_path=sentences_path,
        model_path=model_path,
        evidence_path=evidence_path,
        injections_path=injections_path,
        snapshot_dir=snapshot_dir,
        report_path=report_path,
        entity_threshold=entity_threshold,
        relation_types=relation_types,
        trigger_words=trigger_words,
        fixed=fixed,
        workers=workers,
        snapshot_commit_interval=snapshot_commit_interval,
    )


@dataclass(frozen=True)
class ModelPackage:
    raw: dict[str, Any]
    model_hash: str
    feature_dim: int
    hash_buckets: int
    labels: tuple[str, ...]
    transitions: dict[str, dict[str, Decimal]]
    entity_bias: dict[str, Decimal]
    entity_weights: dict[str, tuple[Decimal, ...]]
    entity_types: tuple[str, ...]
    entity_type_bias: dict[str, Decimal]
    entity_type_weights: dict[str, tuple[Decimal, ...]]
    relation_types: tuple[str, ...]
    relation_bias: dict[str, Decimal]
    relation_weights: dict[str, tuple[Decimal, ...]]
    feature_hashes: dict[str, int]
    lexicon: tuple[str, ...]

    def bucket_for(self, key: str) -> int:
        bucket = self.feature_hashes.get(key)
        if bucket is None:
            bucket = fixed_hash_bucket(key, self.hash_buckets)
        if not isinstance(bucket, int) or isinstance(bucket, bool):
            raise ModelShapeError(f"feature hash for {key!r} is not an integer")
        if bucket < 0 or bucket >= self.hash_buckets:
            raise ModelShapeError(
                f"feature hash bucket for {key!r} is outside [0, {self.hash_buckets})"
            )
        return bucket


def _model_decimal(value: Any, field_name: str) -> Decimal:
    try:
        return as_decimal(value, field_name)
    except ConfigError as exc:
        raise ModelShapeError(exc.message) from exc


def _model_vector(value: Any, field_name: str, size: int) -> tuple[Decimal, ...]:
    if not isinstance(value, list) or len(value) != size:
        actual = len(value) if isinstance(value, list) else "not-an-array"
        raise ModelShapeError(
            f"{field_name} must contain {size} weights (actual: {actual})"
        )
    return tuple(_model_decimal(item, f"{field_name}[]") for item in value)


def load_model(path: str | Path, config: WorkbenchConfig) -> ModelPackage:
    try:
        return _load_model_impl(path, config)
    except ModelShapeError:
        raise
    except (ConfigError, ValueError, TypeError) as exc:
        raise ModelShapeError(getattr(exc, "message", str(exc))) from exc


def _load_model_impl(path: str | Path, config: WorkbenchConfig) -> ModelPackage:
    raw = _mapping(load_json(Path(path)), "model")
    feature_dim = require_int(raw.get("feature_dim"), "model.feature_dim", minimum=1)
    hash_buckets = require_int(raw.get("hash_buckets"), "model.hash_buckets", minimum=1)
    if feature_dim != hash_buckets:
        raise ModelShapeError(
            f"feature_dim ({feature_dim}) must equal hash_buckets ({hash_buckets})"
        )

    labels_raw = _sequence(raw.get("labels"), "model.labels")
    labels = tuple(_string(item, "model.labels[]") for item in labels_raw)
    if len(labels) != 5 or set(labels) != set(ENTITY_LABEL_PRIORITY):
        raise ModelShapeError("model.labels must contain B, I, E, S and O exactly once")

    transitions_raw = _mapping(raw.get("transitions"), "model.transitions")
    transitions: dict[str, dict[str, Decimal]] = {}
    for from_label in ENTITY_LABEL_PRIORITY:
        row_raw = _mapping(
            transitions_raw.get(from_label), f"model.transitions.{from_label}"
        )
        row: dict[str, Decimal] = {}
        for to_label in ENTITY_LABEL_PRIORITY:
            row[to_label] = _model_decimal(
                row_raw.get(to_label), f"model.transitions.{from_label}.{to_label}"
            )
        transitions[from_label] = row

    entity_bias_raw = _mapping(raw.get("entity_bias"), "model.entity_bias")
    entity_weights_raw = _mapping(raw.get("entity_weights"), "model.entity_weights")
    entity_bias: dict[str, Decimal] = {}
    entity_weights: dict[str, tuple[Decimal, ...]] = {}
    for label in ENTITY_LABEL_PRIORITY:
        entity_bias[label] = _model_decimal(
            entity_bias_raw.get(label), f"model.entity_bias.{label}"
        )
        entity_weights[label] = _model_vector(
            entity_weights_raw.get(label), f"model.entity_weights.{label}", feature_dim
        )

    entity_types_raw = _sequence(raw.get("entity_types"), "model.entity_types")
    entity_types = tuple(_string(item, "model.entity_types[]") for item in entity_types_raw)
    if not entity_types or len(set(entity_types)) != len(entity_types):
        raise ModelShapeError("model.entity_types must be non-empty and unique")
    entity_type_bias_raw = _mapping(raw.get("entity_type_bias"), "model.entity_type_bias")
    entity_type_weights_raw = _mapping(
        raw.get("entity_type_weights"), "model.entity_type_weights"
    )
    entity_type_bias: dict[str, Decimal] = {}
    entity_type_weights: dict[str, tuple[Decimal, ...]] = {}
    for entity_type in entity_types:
        entity_type_bias[entity_type] = _model_decimal(
            entity_type_bias_raw.get(entity_type),
            f"model.entity_type_bias.{entity_type}",
        )
        entity_type_weights[entity_type] = _model_vector(
            entity_type_weights_raw.get(entity_type),
            f"model.entity_type_weights.{entity_type}",
            feature_dim,
        )

    relation_types_raw = _sequence(raw.get("relation_types"), "model.relation_types")
    relation_types = tuple(
        sorted({_string(item, "model.relation_types[]") for item in relation_types_raw})
    )
    if set(relation_types) != set(config.relation_types):
        raise ConfigError(
            "model.relation_types must match relation.relation_types in the config"
        )
    relation_bias_raw = _mapping(raw.get("relation_bias"), "model.relation_bias")
    relation_weights_raw = _mapping(raw.get("relation_weights"), "model.relation_weights")
    relation_bias: dict[str, Decimal] = {}
    relation_weights: dict[str, tuple[Decimal, ...]] = {}
    for relation_type in relation_types:
        relation_bias[relation_type] = _model_decimal(
            relation_bias_raw.get(relation_type), f"model.relation_bias.{relation_type}"
        )
        relation_weights[relation_type] = _model_vector(
            relation_weights_raw.get(relation_type),
            f"model.relation_weights.{relation_type}",
            feature_dim,
        )

    feature_hashes_raw = _mapping(raw.get("feature_hashes", {}), "model.feature_hashes")
    feature_hashes: dict[str, int] = {}
    for key, bucket in feature_hashes_raw.items():
        if not isinstance(key, str) or not isinstance(bucket, int) or isinstance(bucket, bool):
            raise ModelShapeError("model.feature_hashes must map strings to integers")
        if bucket < 0 or bucket >= hash_buckets:
            raise ModelShapeError(
                f"feature hash bucket for {key!r} is outside [0, {hash_buckets})"
            )
        feature_hashes[key] = bucket

    lexicon_raw = raw.get("lexicon", [])
    if not isinstance(lexicon_raw, list):
        raise ModelShapeError("model.lexicon must be an array")
    lexicon = tuple(
        sorted(
            {_string(item, "model.lexicon[]") for item in lexicon_raw},
            key=lambda item: (-len(item), item),
        )
    )

    return ModelPackage(
        raw=raw,
        model_hash=sha256_value(raw),
        feature_dim=feature_dim,
        hash_buckets=hash_buckets,
        labels=labels,
        transitions=transitions,
        entity_bias=entity_bias,
        entity_weights=entity_weights,
        entity_types=entity_types,
        entity_type_bias=entity_type_bias,
        entity_type_weights=entity_type_weights,
        relation_types=relation_types,
        relation_bias=relation_bias,
        relation_weights=relation_weights,
        feature_hashes=feature_hashes,
        lexicon=lexicon,
    )


@dataclass(frozen=True)
class EntityAnnotation:
    start: int
    end: int
    entity_type: str


@dataclass(frozen=True)
class RelationTruth:
    subject_entity: int
    object_entity: int
    relation_type: str


@dataclass(frozen=True)
class Sentence:
    sentence_id: str
    words: tuple[str, ...]
    pos: tuple[str, ...]
    entities: tuple[EntityAnnotation, ...]
    relations: tuple[RelationTruth, ...]
    source_line: int


@dataclass(frozen=True)
class EvidenceCandidate:
    sentence_id: str
    subject_span: tuple[int, int]
    object_span: tuple[int, int]
    relations: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceTable:
    raw: dict[str, Any]
    candidates_by_sentence: dict[str, tuple[EvidenceCandidate, ...]]
    exclusive_relations: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class Injection:
    injection_id: str
    sentence_id: str
    stage: str
    kind: str
    ordinal: int


@dataclass
class LoadBundle:
    sentences: list[Sentence]
    failures: list[dict[str, Any]]
    injection_events: list[dict[str, Any]]


def _parse_tokens(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = tuple(part for part in value.split() if part)
    elif isinstance(value, list):
        if not all(isinstance(item, str) and item for item in value):
            raise ConfigError(f"{field_name} must contain non-empty strings")
        tokens = tuple(value)
    else:
        raise ConfigError(f"{field_name} must be a whitespace string or string array")
    if not tokens:
        raise ConfigError(f"{field_name} must not be empty")
    return tokens


def _span_from_value(value: Any, field_name: str) -> tuple[int, int]:
    if isinstance(value, dict):
        start = value.get("start")
        end = value.get("end")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value
    else:
        raise ConfigError(f"{field_name} must be [start, end]")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end < start
    ):
        raise AnnotationRangeError(f"{field_name} has invalid integer bounds")
    return start, end


def _apply_load_injections(
    raw: dict[str, Any],
    sentence_id: str,
    injections: Iterable[Injection],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = copy.deepcopy(raw)
    events: list[dict[str, Any]] = []
    for injection in injections:
        if injection.sentence_id != sentence_id or injection.stage != "load":
            continue
        if injection.kind == "annotation_out_of_range":
            entities = result.get("entities")
            words = result.get("words", result.get("tokens"))
            if isinstance(words, str):
                count = len(words.split())
            elif isinstance(words, list):
                count = len(words)
            else:
                count = 0
            if not isinstance(entities, list) or not entities:
                raise AnnotationRangeError(
                    "annotation_out_of_range requires at least one entity annotation"
                )
            first = entities[0]
            if isinstance(first, dict):
                first["end"] = max(count, 1)
            elif isinstance(first, (list, tuple)) and len(first) >= 2:
                first[1] = max(count, 1)
            else:
                raise AnnotationRangeError("invalid first entity annotation for injection")
            events.append(
                {
                    "injection_id": injection.injection_id,
                    "sentence_id": sentence_id,
                    "stage": "load",
                    "kind": injection.kind,
                    "applied": True,
                    "business_code": "E_ANNOTATION_RANGE",
                }
            )
    return result, events


def load_sentences(
    path: str | Path,
    config: WorkbenchConfig,
    injections: Iterable[Injection],
) -> LoadBundle:
    sentence_path = Path(path)
    accepted: list[Sentence] = []
    failures: list[dict[str, Any]] = []
    injection_events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    injection_list = tuple(injections)

    with sentence_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line, parse_float=Decimal, parse_int=int)
            except Exception as exc:  # noqa: BLE001 - one bad line must not stop the run
                failures.append(
                    failure_from_exception(
                        f"line:{line_number}", "load", ConfigError(f"invalid JSON: {exc}")
                    )
                )
                continue
            if not isinstance(raw, dict):
                failures.append(
                    failure_from_exception(
                        f"line:{line_number}",
                        "load",
                        ConfigError("sentence record must be a JSON object"),
                    )
                )
                continue
            sentence_id = raw.get("id", raw.get("sentence_id"))
            if not isinstance(sentence_id, str) or not sentence_id:
                failures.append(
                    failure_from_exception(
                        f"line:{line_number}",
                        "load",
                        ConfigError("sentence id must be a non-empty string"),
                    )
                )
                continue

            if sentence_id in seen_ids:
                failures.append(
                    failure_from_exception(
                        sentence_id,
                        "load",
                        ConfigError(f"duplicate sentence id {sentence_id}"),
                    )
                )
                continue
            seen_ids.add(sentence_id)

            applied_events: list[dict[str, Any]] = []
            try:
                injected_raw, applied_events = _apply_load_injections(
                    raw, sentence_id, injection_list
                )
                words = _parse_tokens(
                    injected_raw.get("words", injected_raw.get("tokens")), "words"
                )
                pos = _parse_tokens(
                    injected_raw.get("pos", injected_raw.get("pos_tags")), "pos"
                )
                if len(words) != len(pos):
                    raise ConfigError(
                        f"word count ({len(words)}) does not match POS count ({len(pos)})"
                    )

                entities_raw = injected_raw.get("entities", [])
                if not isinstance(entities_raw, list):
                    raise AnnotationRangeError("entities must be an array")
                entities: list[EntityAnnotation] = []
                for index, item in enumerate(entities_raw):
                    if not isinstance(item, dict):
                        raise AnnotationRangeError(f"entities[{index}] must be an object")
                    start, end = _span_from_value(item, f"entities[{index}]")
                    if start >= len(words) or end >= len(words):
                        raise AnnotationRangeError(
                            f"entities[{index}] span [{start}, {end}] is outside 0..{len(words)-1}"
                        )
                    entity_type = item.get("type", item.get("entity_type"))
                    if not isinstance(entity_type, str) or not entity_type:
                        raise AnnotationRangeError(f"entities[{index}] type is missing")
                    entities.append(EntityAnnotation(start, end, entity_type))
                entities.sort(key=lambda item: (item.start, item.end, item.entity_type))
                for previous, current in zip(entities, entities[1:]):
                    if current.start <= previous.end:
                        raise AnnotationRangeError(
                            "entity annotation spans overlap; spans must be continuous and disjoint"
                        )

                relations_raw = injected_raw.get("relations", [])
                if not isinstance(relations_raw, list):
                    raise AnnotationRangeError("relations must be an array")
                relations: list[RelationTruth] = []
                for index, item in enumerate(relations_raw):
                    if not isinstance(item, dict):
                        raise AnnotationRangeError(f"relations[{index}] must be an object")
                    subject_index = item.get(
                        "subject_entity", item.get("subject", item.get("head"))
                    )
                    object_index = item.get(
                        "object_entity", item.get("object", item.get("tail"))
                    )
                    if (
                        isinstance(subject_index, bool)
                        or isinstance(object_index, bool)
                        or not isinstance(subject_index, int)
                        or not isinstance(object_index, int)
                        or subject_index < 0
                        or object_index < 0
                        or subject_index >= len(entities)
                        or object_index >= len(entities)
                    ):
                        raise AnnotationRangeError(
                            f"relations[{index}] references an invalid entity index"
                        )
                    relation_type = item.get("type", item.get("relation_type"))
                    if not isinstance(relation_type, str) or not relation_type:
                        raise AnnotationRangeError(f"relations[{index}] type is missing")
                    if relation_type not in config.relation_types:
                        raise ConfigError(
                            f"relations[{index}] unknown relation type {relation_type}"
                        )
                    relations.append(
                        RelationTruth(subject_index, object_index, relation_type)
                    )
                relations.sort(
                    key=lambda item: (
                        item.subject_entity,
                        item.object_entity,
                        item.relation_type,
                    )
                )
                accepted.append(
                    Sentence(
                        sentence_id=sentence_id,
                        words=words,
                        pos=pos,
                        entities=tuple(entities),
                        relations=tuple(relations),
                        source_line=line_number,
                    )
                )
                injection_events.extend(applied_events)
            except (ConfigError, AnnotationRangeError) as exc:
                failure = failure_from_exception(
                    sentence_id,
                    "load",
                    exc,
                    injection_ids=tuple(
                        event["injection_id"] for event in applied_events
                    ),
                )
                failures.append(failure)
                for event in applied_events:
                    event["failure_code"] = failure["code"]
                injection_events.extend(applied_events)

    accepted.sort(key=lambda item: item.sentence_id)
    failures.sort(
        key=lambda item: (
            item["sentence_id"],
            STAGE_ORDER.index(item["stage"]),
            item["code"],
            item["exception_type"],
        )
    )
    return LoadBundle(accepted, failures, injection_events)


def load_evidence(path: str | Path, config: WorkbenchConfig) -> EvidenceTable:
    raw = _mapping(load_json(Path(path)), "evidence")
    candidates_raw = _sequence(raw.get("candidate_pairs"), "evidence.candidate_pairs")
    candidates_by_sentence: dict[str, list[EvidenceCandidate]] = {}
    seen: set[tuple[str, tuple[int, int], tuple[int, int]]] = set()
    relation_set = set(config.relation_types)
    for index, item in enumerate(candidates_raw):
        if not isinstance(item, dict):
            raise ConfigError(f"evidence.candidate_pairs[{index}] must be an object")
        sentence_id = _string(item.get("sentence_id"), "candidate sentence_id")
        subject_span = _span_from_value(
            item.get("subject_span", item.get("subject")), "candidate subject_span"
        )
        object_span = _span_from_value(
            item.get("object_span", item.get("object")), "candidate object_span"
        )
        relations_raw = _sequence(item.get("relations"), "candidate relations")
        relations = tuple(
            sorted({_string(value, "candidate relation") for value in relations_raw})
        )
        if not relations:
            raise ConfigError("candidate pair relations must not be empty")
        unknown = set(relations) - relation_set
        if unknown:
            raise ConfigError(f"candidate pair contains unknown relations {sorted(unknown)}")
        key = (sentence_id, subject_span, object_span)
        if key in seen:
            raise ConfigError(f"duplicate evidence candidate pair {key}")
        seen.add(key)
        candidates_by_sentence.setdefault(sentence_id, []).append(
            EvidenceCandidate(sentence_id, subject_span, object_span, relations)
        )

    exclusive_raw = _sequence(raw.get("exclusive_relations", []), "evidence.exclusive_relations")
    exclusive: dict[str, set[str]] = {relation: set() for relation in config.relation_types}
    for index, item in enumerate(exclusive_raw):
        if not isinstance(item, dict):
            raise ConfigError(f"evidence.exclusive_relations[{index}] must be an object")
        relation = _string(
            item.get("relation", item.get("if")), "exclusive relation source"
        )
        targets_raw = _sequence(
            item.get("excludes", item.get("then")), "exclusive relation targets"
        )
        targets = [_string(value, "exclusive relation target") for value in targets_raw]
        if relation not in relation_set or any(target not in relation_set for target in targets):
            raise ConfigError("exclusive_relations references an unknown relation type")
        for target in targets:
            if target == relation:
                raise ConfigError("a relation cannot exclude itself")
            exclusive[relation].add(target)
            exclusive[target].add(relation)

    normalized_candidates = {
        sentence_id: tuple(
            sorted(items, key=lambda item: (item.subject_span, item.object_span, item.relations))
        )
        for sentence_id, items in candidates_by_sentence.items()
    }
    return EvidenceTable(
        raw=raw,
        candidates_by_sentence=normalized_candidates,
        exclusive_relations={
            relation: tuple(sorted(targets)) for relation, targets in sorted(exclusive.items())
        },
    )


def load_injections(path: str | Path | None, config: WorkbenchConfig) -> tuple[Injection, ...]:
    if path is None:
        return ()
    raw = load_json(Path(path))
    if isinstance(raw, list):
        items_raw = raw
    elif isinstance(raw, dict):
        items_raw = _sequence(raw.get("injections", []), "injections")
    else:
        raise ConfigError("injections must be a JSON object or array")
    injections: list[Injection] = []
    seen: set[str] = set()
    for ordinal, item in enumerate(items_raw):
        if not isinstance(item, dict):
            raise ConfigError("injection entry must be an object")
        injection_id = _string(item.get("id"), "injection id")
        if injection_id in seen:
            raise ConfigError(f"duplicate injection id {injection_id}")
        seen.add(injection_id)
        sentence_id = _string(item.get("sentence_id"), "injection sentence_id")
        stage = _string(item.get("stage"), "injection stage")
        kind = _string(item.get("kind"), "injection kind")
        if stage not in STAGE_ORDER:
            raise ConfigError(f"injection stage {stage} is not recognized")
        expected_stage = INJECTION_KIND_STAGE.get(kind)
        if expected_stage is None:
            raise ConfigError(f"injection kind {kind} is not recognized")
        if stage != expected_stage:
            raise ConfigError(
                f"injection kind {kind} must be applied at stage {expected_stage}"
            )
        injections.append(Injection(injection_id, sentence_id, stage, kind, ordinal))
    injections.sort(key=lambda item: (item.injection_id, item.ordinal))
    return tuple(injections)


def input_hashes(
    config: WorkbenchConfig, evidence: EvidenceTable, injections: tuple[Injection, ...]
) -> dict[str, str]:
    """Hash every input that can change the semantic result of a resumed run."""

    sentence_bytes = Path(config.sentences_path).read_bytes()
    injection_payload: Any = []
    if config.injections_path is not None:
        injection_payload = load_json(Path(config.injections_path))
    return {
        "sentences": sha256_bytes(sentence_bytes),
        "evidence": sha256_value(evidence.raw),
        "injections": sha256_value(injection_payload),
        "combined": sha256_bytes(
            canonical_bytes(
                {
                    "sentences": sha256_bytes(sentence_bytes),
                    "evidence": sha256_value(evidence.raw),
                    "injections": sha256_value(injection_payload),
                    "injection_ids": [item.injection_id for item in injections],
                }
            )
        ),
    }
