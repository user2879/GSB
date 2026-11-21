from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from .errors import E_CONFIG_INVALID, FailureRecord, WorkbenchError, failure_from_error
from .models import (
    Document,
    Entity,
    Event,
    InjectionSpec,
    SynonymRule,
    SynonymTable,
    WorkbenchConfig,
)


@dataclass(frozen=True)
class RawDocument:
    data: dict[str, Any]
    source_name: str
    source_line: int
    source_index: int


def _line_sort_key(source_name: str, line: int, index: int) -> str:
    return f"load|{source_name}|{line:08d}|{index:08d}"


def discover_news_files(path: Path, excluded_paths: set[Path] | None = None) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"news input does not exist: {path}")
    excluded_paths = {item.resolve() for item in (excluded_paths or set())}
    candidates = [
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl"}
        and candidate.resolve() not in excluded_paths
    ]
    named_news = [
        candidate
        for candidate in candidates
        if candidate.stem.lower().startswith(("news", "document", "doc_"))
    ]
    if named_news:
        return sorted(named_news, key=lambda item: item.name)
    excluded = {"injections", "synonym_table", "config", "run_state", "report"}
    files = [candidate for candidate in candidates if candidate.stem.lower() not in excluded]
    return sorted(files, key=lambda item: item.name)


def load_raw_news(
    path: Path, excluded_paths: set[Path] | None = None
) -> tuple[list[RawDocument], list[FailureRecord]]:
    documents: list[RawDocument] = []
    failures: list[FailureRecord] = []
    source_index = 0
    try:
        files = discover_news_files(path, excluded_paths)
    except Exception as exc:  # noqa: BLE001 - converted into a diagnosed failure
        failures.append(
            failure_from_error(
                exc,
                document_id=None,
                stage="load",
                sort_key="load|news|discovery",
                details={"input": path.name},
            )
        )
        return documents, failures

    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    source_index += 1
                    sort_key = _line_sort_key(file_path.name, line_number, source_index)
                    try:
                        payload = json.loads(line)
                    except Exception as exc:  # noqa: BLE001
                        failures.append(
                            FailureRecord(
                                document_id=None,
                                stage="load",
                                code=E_CONFIG_INVALID,
                                exception_type=exc.__class__.__name__,
                                message=f"news JSON parse failed: {exc}",
                                sort_key=sort_key,
                                details={"source": f"{file_path.name}:{line_number}"},
                            )
                        )
                        continue
                    if not isinstance(payload, dict):
                        failures.append(
                            FailureRecord(
                                document_id=None,
                                stage="load",
                                code=E_CONFIG_INVALID,
                                exception_type="TypeError",
                                message="news JSON entry must be an object",
                                sort_key=sort_key,
                                details={"source": f"{file_path.name}:{line_number}"},
                            )
                        )
                        continue
                    documents.append(
                        RawDocument(
                            data=payload,
                            source_name=file_path.name,
                            source_line=line_number,
                            source_index=source_index,
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                failure_from_error(
                    exc,
                    document_id=None,
                    stage="load",
                    sort_key=f"load|{file_path.name}|file",
                    details={"source": file_path.name},
                )
            )
    return documents, failures


def _parse_decimal(
    value: Any,
    field_name: str,
    violations: list[dict[str, Any]],
) -> Decimal | None:
    if isinstance(value, bool):
        violations.append({"field": field_name, "reason": "must be a finite number"})
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        violations.append({"field": field_name, "reason": "must be a finite number"})
        return None
    if not parsed.is_finite():
        violations.append({"field": field_name, "reason": "must be finite"})
        return None
    return parsed


def _parse_positive_int(
    value: Any,
    field_name: str,
    violations: list[dict[str, Any]],
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        violations.append({"field": field_name, "reason": "must be a positive integer"})
        return None
    if value <= 0:
        violations.append({"field": field_name, "reason": "must be greater than zero"})
        return None
    return value


def load_config(path: Path) -> tuple[WorkbenchConfig | None, list[FailureRecord]]:
    violations: list[dict[str, Any]] = []
    sort_key = f"config|{path.name}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except Exception as exc:  # noqa: BLE001
        return None, [
            failure_from_error(
                exc,
                document_id=None,
                stage="config",
                sort_key=sort_key,
                details={"file": path.name},
            )
        ]

    if not isinstance(payload, dict):
        return None, [
            FailureRecord(
                document_id=None,
                stage="config",
                code=E_CONFIG_INVALID,
                exception_type="TypeError",
                message="config root must be a mapping",
                sort_key=sort_key,
                details={"file": path.name},
            )
        ]

    max_sentences = _parse_positive_int(
        payload.get("max_summary_sentences"), "max_summary_sentences", violations
    )
    word_weight = _parse_decimal(
        payload.get("word_frequency_weight"), "word_frequency_weight", violations
    )
    entity_weight = _parse_decimal(
        payload.get("entity_coverage_weight"), "entity_coverage_weight", violations
    )
    decay = _parse_decimal(
        payload.get("position_decay_coefficient"), "position_decay_coefficient", violations
    )
    coverage_min = _parse_decimal(
        payload.get("entity_coverage_min"), "entity_coverage_min", violations
    )
    fixed_decimals = _parse_positive_int(
        payload.get("fixed_decimals"), "fixed_decimals", violations
    )
    quantization_levels = _parse_positive_int(
        payload.get("quantization_levels"), "quantization_levels", violations
    )
    max_workers = _parse_positive_int(payload.get("max_workers"), "max_workers", violations)

    if word_weight is not None and word_weight < 0:
        violations.append({"field": "word_frequency_weight", "reason": "must be >= 0"})
    if entity_weight is not None and entity_weight < 0:
        violations.append({"field": "entity_coverage_weight", "reason": "must be >= 0"})
    if decay is not None and not (Decimal("0") <= decay <= Decimal("1")):
        violations.append({"field": "position_decay_coefficient", "reason": "must be in [0, 1]"})
    if coverage_min is not None and not (Decimal("0") <= coverage_min <= Decimal("1")):
        violations.append({"field": "entity_coverage_min", "reason": "must be in [0, 1]"})

    if fixed_decimals is not None and not (1 <= fixed_decimals <= 12):
        violations.append({"field": "fixed_decimals", "reason": "must be in [1, 12]"})
    if fixed_decimals is not None and quantization_levels is not None:
        scale = 10**fixed_decimals
        if quantization_levels > scale:
            violations.append(
                {"field": "quantization_levels", "reason": "must not exceed 10**fixed_decimals"}
            )
        elif scale % quantization_levels != 0:
            violations.append(
                {
                    "field": "quantization_levels",
                    "reason": "must divide 10**fixed_decimals exactly",
                }
            )

    style_template_id = payload.get("style_template_id")
    if not isinstance(style_template_id, str) or not style_template_id.strip():
        violations.append({"field": "style_template_id", "reason": "must be a non-empty string"})

    synonym_table_version = payload.get("synonym_table_version")
    if not isinstance(synonym_table_version, str) or not synonym_table_version.strip():
        violations.append(
            {"field": "synonym_table_version", "reason": "must be a non-empty string"}
        )

    synonym_table_path = payload.get("synonym_table_path")
    if not isinstance(synonym_table_path, str) or not synonym_table_path.strip():
        violations.append({"field": "synonym_table_path", "reason": "must be a non-empty string"})

    if violations or any(
        value is None
        for value in (
            max_sentences,
            word_weight,
            entity_weight,
            decay,
            coverage_min,
            fixed_decimals,
            quantization_levels,
            max_workers,
        )
    ):
        return None, [
            FailureRecord(
                document_id=None,
                stage="config",
                code=E_CONFIG_INVALID,
                exception_type="ConfigValidationError",
                message="configuration validation failed",
                sort_key=sort_key,
                details={"file": path.name, "violations": violations},
            )
        ]

    assert max_sentences is not None
    assert word_weight is not None
    assert entity_weight is not None
    assert decay is not None
    assert coverage_min is not None
    assert fixed_decimals is not None
    assert quantization_levels is not None
    assert max_workers is not None
    assert isinstance(style_template_id, str)
    assert isinstance(synonym_table_version, str)
    assert isinstance(synonym_table_path, str)

    config = WorkbenchConfig(
        max_summary_sentences=max_sentences,
        word_frequency_weight=word_weight,
        entity_coverage_weight=entity_weight,
        position_decay_coefficient=decay,
        entity_coverage_min=coverage_min,
        style_template_id=style_template_id.strip(),
        synonym_table_version=synonym_table_version.strip(),
        fixed_decimals=fixed_decimals,
        quantization_levels=quantization_levels,
        max_workers=max_workers,
        synonym_table_path=synonym_table_path.strip(),
        source_name=path.name,
    )
    return config, []


def load_synonym_table(
    path: Path, config: WorkbenchConfig
) -> tuple[SynonymTable | None, list[FailureRecord]]:
    sort_key = f"config|synonym|{path.name}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except Exception as exc:  # noqa: BLE001
        return None, [
            failure_from_error(
                exc,
                document_id=None,
                stage="config",
                sort_key=sort_key,
                details={"file": path.name},
            )
        ]

    try:
        table = _validate_synonym_table(payload, path.name, config)
    except WorkbenchError as exc:
        return None, [
            failure_from_error(exc, document_id=None, stage="config", sort_key=sort_key)
        ]
    return table, []


def _validate_synonym_table(
    payload: Any, source_name: str, config: WorkbenchConfig
) -> SynonymTable:
    if not isinstance(payload, dict):
        raise WorkbenchError(E_CONFIG_INVALID, "config", "synonym table root must be a mapping")
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise WorkbenchError(E_CONFIG_INVALID, "config", "synonym table version must be non-empty")
    if version.strip() != config.synonym_table_version:
        raise WorkbenchError(
            E_CONFIG_INVALID,
            "config",
            "style template version does not match synonym table version",
            details={
                "config_version": config.synonym_table_version,
                "table_version": version.strip(),
            },
        )
    templates_payload = payload.get("templates")
    if not isinstance(templates_payload, dict) or not templates_payload:
        raise WorkbenchError(
            E_CONFIG_INVALID, "config", "synonym table templates must be a non-empty mapping"
        )
    templates: dict[str, dict[str, Any]] = {}
    for template_id, template_value in templates_payload.items():
        if not isinstance(template_id, str) or not template_id.strip():
            raise WorkbenchError(
                E_CONFIG_INVALID, "config", "synonym template id must be a non-empty string"
            )
        if not isinstance(template_value, dict):
            raise WorkbenchError(
                E_CONFIG_INVALID,
                "config",
                f"synonym template {template_id} must be a mapping",
            )
        templates[template_id] = dict(template_value)
    if config.style_template_id not in templates:
        raise WorkbenchError(
            E_CONFIG_INVALID,
            "config",
            "style template id is not declared in synonym table version",
            details={
                "style_template_id": config.style_template_id,
                "declared_templates": sorted(templates),
            },
        )

    rules_payload = payload.get("rules")
    if not isinstance(rules_payload, list):
        raise WorkbenchError(E_CONFIG_INVALID, "config", "synonym rules must be a list")
    rules: list[SynonymRule] = []
    seen_rule_ids: set[str] = set()
    for order, rule_payload in enumerate(rules_payload):
        if not isinstance(rule_payload, dict):
            raise WorkbenchError(
                E_CONFIG_INVALID, "config", f"synonym rule at index {order} must be a mapping"
            )
        rule_id = rule_payload.get("rule_id")
        template_id = rule_payload.get("template_id")
        source = rule_payload.get("from")
        target = rule_payload.get("to")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise WorkbenchError(
                E_CONFIG_INVALID, "config", f"synonym rule at index {order} has invalid rule_id"
            )
        if rule_id in seen_rule_ids:
            raise WorkbenchError(
                E_CONFIG_INVALID, "config", f"duplicate synonym rule id: {rule_id}"
            )
        if not isinstance(template_id, str) or not template_id.strip():
            raise WorkbenchError(
                E_CONFIG_INVALID, "config", f"synonym rule {rule_id} has invalid template_id"
            )
        if not isinstance(source, str) or not source:
            raise WorkbenchError(
                E_CONFIG_INVALID, "config", f"synonym rule {rule_id} has invalid from"
            )
        if not isinstance(target, str):
            raise WorkbenchError(
                E_CONFIG_INVALID, "config", f"synonym rule {rule_id} has invalid to"
            )
        seen_rule_ids.add(rule_id)
        rules.append(
            SynonymRule(
                order=order,
                rule_id=rule_id,
                template_id=template_id,
                source=source,
                target=target,
            )
        )
    return SynonymTable(
        version=version.strip(),
        templates=templates,
        rules=tuple(rules),
        source_name=source_name,
    )


def missing_template_rules(table: SynonymTable) -> list[SynonymRule]:
    declared = set(table.templates)
    return [rule for rule in table.rules if rule.template_id not in declared]


def load_injections(path: Path | None) -> tuple[list[InjectionSpec], list[FailureRecord]]:
    if path is None:
        return [], []
    sort_key = f"load|injection|{path.name}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:  # noqa: BLE001
        return [], [
            failure_from_error(
                exc,
                document_id=None,
                stage="load",
                sort_key=sort_key,
                details={"file": path.name},
            )
        ]
    if not isinstance(payload, dict) or not isinstance(payload.get("injections"), list):
        return [], [
            FailureRecord(
                document_id=None,
                stage="load",
                code=E_CONFIG_INVALID,
                exception_type="TypeError",
                message="injection file must contain an injections list",
                sort_key=sort_key,
                details={"file": path.name},
            )
        ]

    injections: list[InjectionSpec] = []
    failures: list[FailureRecord] = []
    seen_ids: set[str] = set()
    for ordinal, raw in enumerate(payload["injections"]):
        item_sort_key = f"load|injection|{path.name}|{ordinal:08d}"
        if not isinstance(raw, dict):
            failures.append(
                FailureRecord(
                    document_id=None,
                    stage="load",
                    code=E_CONFIG_INVALID,
                    exception_type="TypeError",
                    message="injection entry must be an object",
                    sort_key=item_sort_key,
                    details={"ordinal": ordinal},
                )
            )
            continue
        injection_id = raw.get("injection_id")
        errors: list[str] = []
        if not isinstance(injection_id, str) or not injection_id.strip():
            errors.append("injection_id must be a non-empty string")
            injection_id = f"INVALID-{ordinal:04d}"
        elif injection_id in seen_ids:
            errors.append(f"duplicate injection_id: {injection_id}")
        seen_ids.add(injection_id)
        kind = raw.get("type")
        allowed_kinds = {
            "frequency_delta",
            "entity_coverage_delta",
            "illegal_entity",
            "missing_template_ref",
        }
        if kind not in allowed_kinds:
            errors.append(f"unsupported injection type: {kind!r}")

        doc_id = raw.get("doc_id")
        sentence_index = raw.get("sentence_index")
        paragraph_index = raw.get("paragraph_index")
        delta: Decimal | None = None
        entity = raw.get("entity")
        rule_id = raw.get("rule_id")
        missing_template_id = raw.get("missing_template_id")

        if kind in {"frequency_delta", "entity_coverage_delta"}:
            if not isinstance(doc_id, str) or not doc_id.strip():
                errors.append("doc_id must be a non-empty string")
            if isinstance(sentence_index, bool) or not isinstance(sentence_index, int) or sentence_index < 0:
                errors.append("sentence_index must be a non-negative integer")
            if not isinstance(raw.get("delta"), (int, float)) or isinstance(raw.get("delta"), bool):
                errors.append("delta must be numeric")
            try:
                delta = Decimal(str(raw.get("delta")))
                if not delta.is_finite():
                    errors.append("delta must be finite")
            except (InvalidOperation, ValueError):
                errors.append("delta must be numeric")
            if kind == "entity_coverage_delta" and delta is not None and delta < 0:
                errors.append("entity_coverage_delta must be >= 0")
        elif kind == "illegal_entity":
            if not isinstance(doc_id, str) or not doc_id.strip():
                errors.append("doc_id must be a non-empty string")
            if isinstance(paragraph_index, bool) or not isinstance(paragraph_index, int) or paragraph_index < 0:
                errors.append("paragraph_index must be a non-negative integer")
            if not isinstance(entity, dict):
                errors.append("entity must be an object")
        elif kind == "missing_template_ref":
            if not isinstance(rule_id, str) or not rule_id.strip():
                errors.append("rule_id must be a non-empty string")
            if not isinstance(missing_template_id, str) or not missing_template_id.strip():
                errors.append("missing_template_id must be a non-empty string")

        if errors:
            failures.append(
                FailureRecord(
                    document_id=doc_id if isinstance(doc_id, str) else None,
                    stage="load",
                    code=E_CONFIG_INVALID,
                    exception_type="InjectionValidationError",
                    message="injection validation failed",
                    sort_key=item_sort_key,
                    injection_ids=[str(injection_id)],
                    details={"ordinal": ordinal, "violations": errors},
                )
            )
            continue

        injections.append(
            InjectionSpec(
                injection_id=str(injection_id),
                kind=str(kind),
                ordinal=ordinal,
                doc_id=str(doc_id) if isinstance(doc_id, str) else None,
                sentence_index=int(sentence_index)
                if isinstance(sentence_index, int) and not isinstance(sentence_index, bool)
                else None,
                paragraph_index=int(paragraph_index)
                if isinstance(paragraph_index, int) and not isinstance(paragraph_index, bool)
                else None,
                delta=delta,
                entity=dict(entity) if isinstance(entity, dict) else None,
                rule_id=str(rule_id) if isinstance(rule_id, str) else None,
                missing_template_id=str(missing_template_id)
                if isinstance(missing_template_id, str)
                else None,
                raw=dict(raw),
            )
        )
    return injections, failures


def preapply_illegal_entity_injections(
    raw_documents: list[RawDocument], injections: list[InjectionSpec]
) -> None:
    first_by_doc_id: dict[str, RawDocument] = {}
    for raw_document in raw_documents:
        doc_id = raw_document.data.get("doc_id")
        if isinstance(doc_id, str) and doc_id not in first_by_doc_id:
            first_by_doc_id[doc_id] = raw_document

    for injection in injections:
        if injection.kind != "illegal_entity":
            continue
        target = first_by_doc_id.get(injection.doc_id or "")
        if target is None:
            injection.status = "not_applied"
            injection.reason = "target document id was not found in parsed news input"
            continue
        paragraphs = target.data.get("paragraphs")
        index = injection.paragraph_index
        if (
            not isinstance(paragraphs, list)
            or index is None
            or index < 0
            or index >= len(paragraphs)
        ):
            injection.status = "not_applied"
            injection.reason = "target paragraph index is outside the document paragraph list"
            continue
        entities = target.data.get("entities")
        if not isinstance(entities, list):
            injection.status = "not_applied"
            injection.reason = "target document entities field is not a list"
            continue
        entity = dict(injection.entity or {})
        entity["$injection_id"] = injection.injection_id
        entity["$paragraph_index"] = index
        entities.append(entity)
        injection.status = "applied"
        injection.effect = {
            "document_id": injection.doc_id,
            "paragraph_index": index,
            "effect": "entity appended before validation; expected validation rejection",
        }


def _validate_document(raw: RawDocument) -> tuple[Document | None, list[str]]:
    payload = raw.data
    errors: list[str] = []
    doc_id = payload.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        errors.append("doc_id must be a non-empty string")
        doc_id = ""
    title = payload.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append("title must be a non-empty string")
    paragraphs_payload = payload.get("paragraphs")
    if not isinstance(paragraphs_payload, list) or not paragraphs_payload:
        errors.append("paragraphs must be a non-empty list")
        paragraphs_payload = []
    paragraphs: list[str] = []
    for index, paragraph in enumerate(paragraphs_payload):
        if not isinstance(paragraph, str) or not paragraph.strip():
            errors.append(f"paragraphs[{index}] must be a non-empty string")
        else:
            paragraphs.append(paragraph)

    entities_payload = payload.get("entities")
    entities: list[Entity] = []
    entity_ids: set[str] = set()
    if not isinstance(entities_payload, list) or not entities_payload:
        errors.append("entities must be a non-empty list")
        entities_payload = []
    for index, entity_payload in enumerate(entities_payload):
        if not isinstance(entity_payload, dict):
            errors.append(f"entities[{index}] must be an object")
            continue
        entity_id = entity_payload.get("entity_id")
        name = entity_payload.get("name")
        entity_type = entity_payload.get("type")
        aliases = entity_payload.get("aliases", [])
        local_errors: list[str] = []
        if not isinstance(entity_id, str) or not entity_id.strip():
            local_errors.append("entity_id must be a non-empty string")
        elif entity_id in entity_ids:
            local_errors.append(f"duplicate entity_id: {entity_id}")
        if not isinstance(name, str) or not name.strip():
            local_errors.append("name must be a non-empty string")
        if not isinstance(entity_type, str) or not entity_type.strip():
            local_errors.append("type must be a non-empty string")
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            local_errors.append("aliases must be a list of non-empty strings")
        if local_errors:
            injection_id = entity_payload.get("$injection_id")
            prefix = f"entities[{index}]"
            if isinstance(injection_id, str):
                prefix += f" injected_by={injection_id}"
            errors.extend(f"{prefix}: {message}" for message in local_errors)
            continue
        entity_ids.add(entity_id)
        entities.append(
            Entity(
                entity_id=entity_id,
                name=name.strip(),
                aliases=tuple(alias.strip() for alias in aliases),
                entity_type=entity_type.strip(),
            )
        )

    events_payload = payload.get("events")
    events: list[Event] = []
    event_ids: set[str] = set()
    if not isinstance(events_payload, list) or not events_payload:
        errors.append("events must be a non-empty list")
        events_payload = []
    for index, event_payload in enumerate(events_payload):
        if not isinstance(event_payload, dict):
            errors.append(f"events[{index}] must be an object")
            continue
        event_id = event_payload.get("event_id")
        subject = event_payload.get("subject")
        predicate = event_payload.get("predicate")
        obj = event_payload.get("object")
        local_errors = []
        if not isinstance(event_id, str) or not event_id.strip():
            local_errors.append("event_id must be a non-empty string")
        elif event_id in event_ids:
            local_errors.append(f"duplicate event_id: {event_id}")
        for field_name, field_value in (
            ("subject", subject),
            ("predicate", predicate),
            ("object", obj),
        ):
            if not isinstance(field_value, str) or not field_value.strip():
                local_errors.append(f"{field_name} must be a non-empty string")
        if isinstance(subject, str) and subject and subject not in entity_ids:
            local_errors.append(f"subject does not reference a declared entity: {subject}")
        if isinstance(obj, str) and obj and obj not in entity_ids:
            local_errors.append(f"object does not reference a declared entity: {obj}")
        if local_errors:
            errors.extend(f"events[{index}]: {message}" for message in local_errors)
            continue
        event_ids.add(event_id)
        events.append(
            Event(
                event_id=event_id,
                subject=subject,
                predicate=predicate,
                object=obj,
            )
        )

    if errors:
        return None, errors

    document_text = "\n".join(paragraphs)
    paragraph_spans: list[tuple[int, int]] = []
    cursor = 0
    for paragraph in paragraphs:
        start = cursor
        end = start + len(paragraph)
        paragraph_spans.append((start, end))
        cursor = end + 1
    return (
        Document(
            doc_id=doc_id,
            title=title,
            paragraphs=tuple(paragraphs),
            entities=tuple(entities),
            events=tuple(events),
            source_name=raw.source_name,
            source_line=raw.source_line,
            source_index=raw.source_index,
            text=document_text,
            paragraph_spans=tuple(paragraph_spans),
            sort_key=f"{doc_id}|00000000|load",
        ),
        [],
    )


def validate_raw_documents(
    raw_documents: list[RawDocument], injections: list[InjectionSpec]
) -> tuple[list[Document], list[FailureRecord]]:
    documents: list[Document] = []
    failures: list[FailureRecord] = []
    seen_ids: dict[str, RawDocument] = {}
    injections_by_doc: dict[str, list[str]] = {}
    for injection in injections:
        if injection.doc_id and injection.status == "applied":
            injections_by_doc.setdefault(injection.doc_id, []).append(injection.injection_id)

    for raw in raw_documents:
        raw_doc_id = raw.data.get("doc_id")
        doc_id = raw_doc_id if isinstance(raw_doc_id, str) else None
        sort_key = _line_sort_key(raw.source_name, raw.source_line, raw.source_index)
        if doc_id and doc_id in seen_ids:
            first = seen_ids[doc_id]
            failures.append(
                FailureRecord(
                    document_id=doc_id,
                    stage="load",
                    code=E_CONFIG_INVALID,
                    exception_type="DuplicateDocumentId",
                    message="document id must be unique",
                    sort_key=sort_key,
                    injection_ids=injections_by_doc.get(doc_id, []),
                    details={
                        "source": f"{raw.source_name}:{raw.source_line}",
                        "first_source": f"{first.source_name}:{first.source_line}",
                    },
                )
            )
            continue
        document, errors = _validate_document(raw)
        if doc_id:
            seen_ids.setdefault(doc_id, raw)
        if errors:
            failures.append(
                FailureRecord(
                    document_id=doc_id,
                    stage="load",
                    code=E_CONFIG_INVALID,
                    exception_type="DocumentValidationError",
                    message="document validation failed",
                    sort_key=sort_key,
                    injection_ids=injections_by_doc.get(doc_id or "", []),
                    details={
                        "source": f"{raw.source_name}:{raw.source_line}",
                        "violations": errors,
                    },
                )
            )
            continue
        assert document is not None
        documents.append(document)
    return documents, failures


def build_runtime_table(
    table: SynonymTable, injections: list[InjectionSpec]
) -> tuple[SynonymTable, list[str]]:
    runtime_table = table
    applied_ids: list[str] = []
    for injection in injections:
        if injection.kind != "missing_template_ref":
            continue
        rule_id = injection.rule_id or ""
        missing_template_id = injection.missing_template_id or ""
        if not any(rule.rule_id == rule_id for rule in runtime_table.rules):
            injection.status = "not_applied"
            injection.reason = "target synonym rule id was not found"
            continue
        runtime_table = runtime_table.with_rule_template(rule_id, missing_template_id)
        injection.status = "applied"
        applied_ids.append(injection.injection_id)
        injection.effect = {
            "rule_id": rule_id,
            "effective_template_id": missing_template_id,
            "declared_templates": sorted(runtime_table.templates),
        }
    return runtime_table, applied_ids


def resolve_synonym_table_path(config_path: Path, config: WorkbenchConfig) -> Path:
    candidate = Path(config.synonym_table_path)
    if candidate.is_absolute():
        return candidate
    return config_path.parent / candidate


def numeric_float(value: Any) -> float:
    """Small helper used by tests and report sanitization."""
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("numeric value must be finite")
    return converted
