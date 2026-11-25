from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PrototypeConfig
from .errors import E_INDEX_INPUT, Failure, InputValidationError
from .model import (
    FIELD_ORDER,
    ControlledVocabulary,
    Document,
    QuerySpec,
    StopwordTable,
)
from .util import read_json, sha256_file, sha256_json


@dataclass(frozen=True)
class EditCommand:
    op: str
    line: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "line": self.line, **self.payload}


@dataclass(frozen=True)
class LoadedInputs:
    config: PrototypeConfig
    vocabulary: ControlledVocabulary
    stopwords: StopwordTable
    documents: tuple[Document, ...]
    queries: tuple[QuerySpec, ...]
    edits: tuple[EditCommand, ...]
    failures: tuple[Failure, ...]
    input_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "vocabulary": self.vocabulary.to_dict(),
            "stopwords": self.stopwords.to_dict(),
            "documents": [doc.to_dict() for doc in self.documents],
            "queries": [query.to_dict() for query in self.queries],
            "edits": [edit.to_dict() for edit in self.edits],
            "failures": [failure.to_dict() for failure in self.failures],
            "input_fingerprint": self.input_fingerprint,
        }


def _failure(
    source: str,
    line: int,
    entity_id: str,
    message: str,
    detail_code: str,
    context: dict[str, Any] | None = None,
) -> Failure:
    return Failure(
        code=E_INDEX_INPUT,
        source=source,
        line=line,
        entity_id=entity_id,
        message=message,
        detail_code=detail_code,
        context=context,
    )


def _read_jsonl(path: Path, source: str) -> tuple[list[tuple[int, Any]], list[Failure]]:
    rows: list[tuple[int, Any]] = []
    failures: list[Failure] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    failures.append(
                        _failure(source, line_number, "", "空行不是合法 JSONL 记录", "BLANK_LINE")
                    )
                    continue
                try:
                    rows.append((line_number, json.loads(raw_line)))
                except json.JSONDecodeError as exc:
                    failures.append(
                        _failure(
                            source,
                            line_number,
                            "",
                            f"JSON 解析失败: {exc.msg}",
                            "JSON_PARSE",
                        )
                    )
    except OSError as exc:
        failures.append(
            _failure(source, 0, str(path), f"无法读取文件: {exc}", "FILE_READ")
        )
    return rows, failures


def load_vocabulary(path: Path) -> tuple[ControlledVocabulary, list[Failure]]:
    try:
        value = read_json(path)
    except (OSError, ValueError) as exc:
        return ControlledVocabulary.create("invalid", ()), [
            _failure("vocabulary", 0, str(path), f"无法读取受控词表: {exc}", "VOCAB_READ")
        ]
    failures: list[Failure] = []
    if isinstance(value, list):
        version = "unversioned"
        terms = value
    elif isinstance(value, dict):
        version = value.get("version", "unversioned")
        terms = value.get("terms", [])
    else:
        return ControlledVocabulary.create("invalid", ()), [
            _failure("vocabulary", 0, str(path), "受控词表格式非法", "VOCAB_SHAPE")
        ]
    if not isinstance(version, str) or not version:
        failures.append(
            _failure("vocabulary", 0, "vocabulary", "词表版本必须为非空字符串", "VOCAB_VERSION")
        )
    if not isinstance(terms, list) or not terms:
        failures.append(
            _failure("vocabulary", 0, "vocabulary", "受控词表不能为空", "VOCAB_EMPTY")
        )
        terms = []
    normalised: list[str] = []
    seen: set[str] = set()
    for index, term in enumerate(terms):
        if not isinstance(term, str) or not term or any(char.isspace() for char in term):
            failures.append(
                _failure(
                    "vocabulary",
                    index + 1,
                    str(term),
                    "受控词项必须为非空、无空白字符串",
                    "VOCAB_TERM",
                )
            )
            continue
        if term in seen:
            failures.append(
                _failure(
                    "vocabulary",
                    index + 1,
                    term,
                    "受控词项重复",
                    "VOCAB_DUPLICATE",
                )
            )
            continue
        seen.add(term)
        normalised.append(term)
    return ControlledVocabulary.create(str(version), normalised), failures


def load_stopwords(path: Path) -> tuple[StopwordTable, list[Failure]]:
    try:
        value = read_json(path)
    except (OSError, ValueError) as exc:
        return StopwordTable.create("invalid"), [
            _failure("stopwords", 0, str(path), f"无法读取停用词表: {exc}", "STOPWORD_READ")
        ]
    if not isinstance(value, dict):
        return StopwordTable.create("invalid"), [
            _failure("stopwords", 0, str(path), "停用词表根节点必须为对象", "STOPWORD_SHAPE")
        ]

    failures: list[Failure] = []
    version = value.get("version", "unversioned")
    if not isinstance(version, str) or not version:
        failures.append(
            _failure("stopwords", 0, "stopwords", "停用词表版本必须为非空字符串", "STOPWORD_VERSION")
        )
        version = "invalid"

    def _normalise_terms(raw: Any, label: str) -> list[str]:
        if raw is None:
            return []
        if not isinstance(raw, list):
            failures.append(
                _failure("stopwords", 0, label, "停用词集合必须为数组", "STOPWORD_TERMS")
            )
            return []
        result: list[str] = []
        for index, term in enumerate(raw):
            if not isinstance(term, str) or not term or any(char.isspace() for char in term):
                failures.append(
                    _failure(
                        "stopwords",
                        index + 1,
                        label,
                        "停用词必须为非空、无空白字符串",
                        "STOPWORD_TERM",
                    )
                )
                continue
            result.append(term)
        return result

    global_terms = _normalise_terms(value.get("global", []), "global")
    by_field_raw = value.get("by_field", {})
    if not isinstance(by_field_raw, dict):
        failures.append(
            _failure("stopwords", 0, "by_field", "by_field 必须为对象", "STOPWORD_BY_FIELD")
        )
        by_field_raw = {}
    by_field: dict[str, list[str]] = {}
    for field_name, raw_terms in by_field_raw.items():
        if field_name not in FIELD_ORDER:
            failures.append(
                _failure(
                    "stopwords",
                    0,
                    field_name,
                    "停用词字段必须为 title/body",
                    "STOPWORD_FIELD",
                )
            )
            continue
        by_field[field_name] = _normalise_terms(raw_terms, field_name)

    return StopwordTable.create(version, global_terms, by_field), failures


def validate_document(
    value: Any,
    config: PrototypeConfig,
    vocabulary: ControlledVocabulary,
    *,
    source: str,
    line: int,
    entity_hint: str = "",
) -> tuple[Document | None, list[Failure]]:
    failures: list[Failure] = []
    if not isinstance(value, dict):
        return None, [
            _failure(source, line, entity_hint, "文档必须为 JSON 对象", "DOC_SHAPE")
        ]

    doc_id = value.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id:
        failures.append(
            _failure(source, line, entity_hint, "doc_id 必须为非空字符串", "DOC_ID")
        )
        doc_id = entity_hint or "<unknown>"
    version = value.get("version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version < 0
    ):
        failures.append(
            _failure(source, line, doc_id, "文档版本号必须为非负整数", "DOC_VERSION")
        )

    fields_raw = value.get("fields")
    if fields_raw is None:
        fields_raw = {
            name: value.get(name)
            for name in FIELD_ORDER
            if value.get(name) is not None
        }
    if not isinstance(fields_raw, dict):
        failures.append(
            _failure(source, line, doc_id, "fields 必须为对象", "DOC_FIELDS_SHAPE")
        )
        fields_raw = {}

    normalised_fields: dict[str, tuple[str, ...]] = {}
    for field_name, terms_raw in fields_raw.items():
        if field_name not in FIELD_ORDER:
            failures.append(
                _failure(
                    source,
                    line,
                    doc_id,
                    "文档字段名必须限定为 title/body",
                    "DOC_FIELD_NAME",
                    {"field": field_name},
                )
            )
            continue
        if field_name not in config.fields:
            failures.append(
                _failure(
                    source,
                    line,
                    doc_id,
                    "文档字段未在配置字段集合中",
                    "DOC_FIELD_NOT_CONFIGURED",
                    {"field": field_name},
                )
            )
            continue
        if not isinstance(terms_raw, list):
            failures.append(
                _failure(
                    source,
                    line,
                    doc_id,
                    "字段词项必须为数组",
                    "DOC_FIELD_TERMS",
                    {"field": field_name},
                )
            )
            continue
        terms: list[str] = []
        for position, term in enumerate(terms_raw):
            if not isinstance(term, str) or not term:
                failures.append(
                    _failure(
                        source,
                        line,
                        doc_id,
                        "字段词项必须为非空字符串",
                        "DOC_TERM",
                        {"field": field_name, "position": position},
                    )
                )
                continue
            if term not in vocabulary.terms:
                failures.append(
                    _failure(
                        source,
                        line,
                        doc_id,
                        "字段词项不在受控词表内",
                        "DOC_TERM_NOT_IN_VOCAB",
                        {"field": field_name, "position": position, "term": term},
                    )
                )
                continue
            terms.append(term)
        normalised_fields[field_name] = tuple(terms)

    if not normalised_fields:
        failures.append(
            _failure(source, line, doc_id, "文档至少需要一个配置字段", "DOC_FIELDS_EMPTY")
        )

    marks_raw = value.get("field_stopword_marks", value.get("stopword_marks", {}))
    if marks_raw is None:
        marks_raw = {}
    if not isinstance(marks_raw, dict):
        failures.append(
            _failure(source, line, doc_id, "字段停用词标记必须为对象", "DOC_MARKS_SHAPE")
        )
        marks_raw = {}
    normalised_marks: dict[str, tuple[int, ...]] = {}
    for field_name, positions_raw in marks_raw.items():
        if field_name not in FIELD_ORDER:
            failures.append(
                _failure(
                    source,
                    line,
                    doc_id,
                    "停用词标记字段必须为 title/body",
                    "DOC_MARK_FIELD",
                    {"field": field_name},
                )
            )
            continue
        if not isinstance(positions_raw, list):
            failures.append(
                _failure(
                    source,
                    line,
                    doc_id,
                    "字段停用词标记必须为位置数组",
                    "DOC_MARK_POSITIONS",
                    {"field": field_name},
                )
            )
            continue
        positions: list[int] = []
        field_length = len(normalised_fields.get(field_name, ()))
        for position in positions_raw:
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or position < 0
                or position >= field_length
            ):
                failures.append(
                    _failure(
                        source,
                        line,
                        doc_id,
                        "停用词标记位置越界或不是非负整数",
                        "DOC_MARK_POSITION",
                        {"field": field_name, "position": position},
                    )
                )
                continue
            positions.append(position)
        normalised_marks[field_name] = tuple(sorted(set(positions)))

    if failures:
        return None, failures
    return (
        Document.create(doc_id, int(version), normalised_fields, normalised_marks),
        [],
    )


def validate_query(
    value: Any,
    config: PrototypeConfig,
    vocabulary: ControlledVocabulary,
    *,
    source: str,
    line: int,
    entity_hint: str = "",
) -> tuple[QuerySpec | None, list[Failure]]:
    failures: list[Failure] = []
    if not isinstance(value, dict):
        return None, [
            _failure(source, line, entity_hint, "查询必须为 JSON 对象", "QUERY_SHAPE")
        ]
    query_id = value.get("query_id")
    if not isinstance(query_id, str) or not query_id:
        failures.append(
            _failure(source, line, entity_hint, "query_id 必须为非空字符串", "QUERY_ID")
        )
        query_id = entity_hint or "<unknown>"

    terms_raw = value.get("terms")
    if not isinstance(terms_raw, list) or not terms_raw:
        failures.append(
            _failure(source, line, query_id, "查询词项必须为非空数组", "QUERY_TERMS_EMPTY")
        )
        terms_raw = []
    terms: list[str] = []
    for index, term in enumerate(terms_raw):
        if not isinstance(term, str) or not term:
            failures.append(
                _failure(
                    source,
                    line,
                    query_id,
                    "查询词项必须为非空字符串",
                    "QUERY_TERM",
                    {"index": index},
                )
            )
            continue
        if term not in vocabulary.terms:
            failures.append(
                _failure(
                    source,
                    line,
                    query_id,
                    "查询词项不在受控词表内",
                    "QUERY_TERM_NOT_IN_VOCAB",
                    {"index": index, "term": term},
                )
            )
            continue
        terms.append(term)

    preferences_raw = value.get("field_preferences", list(config.fields))
    if isinstance(preferences_raw, dict):
        preferences_raw = [
            field_name
            for field_name, enabled in preferences_raw.items()
            if enabled
        ]
    if not isinstance(preferences_raw, list) or not preferences_raw:
        failures.append(
            _failure(
                source,
                line,
                query_id,
                "字段偏好声明必须为非空数组",
                "QUERY_FIELD_PREFERENCES",
            )
        )
        preferences_raw = []
    preferences: list[str] = []
    for field_name in preferences_raw:
        if not isinstance(field_name, str) or field_name not in FIELD_ORDER:
            failures.append(
                _failure(
                    source,
                    line,
                    query_id,
                    "字段偏好声明只能包含 title/body",
                    "QUERY_FIELD_NAME",
                    {"field": field_name},
                )
            )
            continue
        if field_name not in config.fields:
            failures.append(
                _failure(
                    source,
                    line,
                    query_id,
                    "字段偏好声明不在配置字段集合内",
                    "QUERY_FIELD_NOT_CONFIGURED",
                    {"field": field_name},
                )
            )
            continue
        preferences.append(field_name)

    if failures:
        return None, failures
    return QuerySpec.create(query_id, terms, preferences), []


def _validate_positive_finite(
    value: Any,
) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def validate_edit_command(
    value: Any,
    config: PrototypeConfig,
    vocabulary: ControlledVocabulary,
    *,
    line: int,
) -> tuple[EditCommand | None, list[Failure]]:
    source = "edits"
    if not isinstance(value, dict):
        return None, [_failure(source, line, "", "编辑命令必须为对象", "EDIT_SHAPE")]
    op = value.get("op")
    if not isinstance(op, str) or not op:
        return None, [_failure(source, line, "", "编辑命令缺少 op", "EDIT_OP")]

    if op in {"append_doc", "replace_doc"}:
        doc, failures = validate_document(
            value.get("doc"),
            config,
            vocabulary,
            source=source,
            line=line,
        )
        if failures:
            return None, failures
        return EditCommand(op=op, line=line, payload={"doc": doc}), []

    if op == "delete_doc":
        doc_id = value.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            return None, [
                _failure(source, line, "", "delete_doc 需要非空 doc_id", "EDIT_DOC_ID")
            ]
        return EditCommand(op=op, line=line, payload={"doc_id": doc_id}), []

    if op == "set_field_weights":
        weights = value.get("weights")
        if not isinstance(weights, dict) or set(weights) != set(config.fields):
            return None, [
                _failure(
                    source,
                    line,
                    "config",
                    "字段权重必须覆盖当前全部配置字段",
                    "EDIT_FIELD_WEIGHTS",
                )
            ]
        failures: list[Failure] = []
        normalised: dict[str, float] = {}
        for field_name, weight in weights.items():
            if not _validate_positive_finite(weight):
                failures.append(
                    _failure(
                        source,
                        line,
                        "config",
                        "字段权重必须为有限正数",
                        "EDIT_FIELD_WEIGHT",
                        {"field": field_name},
                    )
                )
            else:
                normalised[field_name] = float(weight)
        if failures:
            return None, failures
        return EditCommand(op=op, line=line, payload={"weights": normalised}), []

    if op == "set_bm25":
        k1 = value.get("k1")
        b = value.get("b")
        if not _validate_positive_finite(k1) or not _validate_positive_finite(b):
            return None, [
                _failure(
                    source,
                    line,
                    "config",
                    "k1 与 b 必须为有限正数",
                    "EDIT_BM25",
                )
            ]
        return EditCommand(op=op, line=line, payload={"k1": float(k1), "b": float(b)}), []

    if op in {"add_stopword", "remove_stopword"}:
        field_name = value.get("field", "*")
        term = value.get("term")
        if field_name not in {"*", *config.fields}:
            return None, [
                _failure(
                    source,
                    line,
                    "stopwords",
                    "停用词作用字段必须为 */title/body",
                    "EDIT_STOPWORD_FIELD",
                )
            ]
        if not isinstance(term, str) or not term or term not in vocabulary.terms:
            return None, [
                _failure(
                    source,
                    line,
                    "stopwords",
                    "停用词必须为非空受控词项",
                    "EDIT_STOPWORD_TERM",
                    {"term": term},
                )
            ]
        return EditCommand(
            op=op, line=line, payload={"field": field_name, "term": term}
        ), []

    if op == "set_fixed":
        decimal_places = value.get("decimal_places")
        quantum = value.get("quantum")
        failures = []
        if (
            isinstance(decimal_places, bool)
            or not isinstance(decimal_places, int)
            or decimal_places not in config.allowed_decimal_places
        ):
            failures.append(
                _failure(
                    source,
                    line,
                    "config",
                    "定点小数位宽不在允许枚举内",
                    "EDIT_FIXED_DECIMAL_PLACES",
                )
            )
        if (
            isinstance(quantum, bool)
            or not isinstance(quantum, int)
            or quantum not in config.allowed_quantums
        ):
            failures.append(
                _failure(
                    source,
                    line,
                    "config",
                    "量化档位不在允许枚举内",
                    "EDIT_FIXED_QUANTUM",
                )
            )
        if failures:
            return None, failures
        return EditCommand(
            op=op,
            line=line,
            payload={"decimal_places": decimal_places, "quantum": quantum},
        ), []

    if op == "set_top_k":
        top_k = value.get("top_k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            return None, [
                _failure(source, line, "config", "TopK 必须为正整数", "EDIT_TOP_K")
            ]
        return EditCommand(op=op, line=line, payload={"top_k": top_k}), []

    return None, [
        _failure(source, line, "", f"未知编辑命令 op: {op}", "EDIT_UNKNOWN_OP")
    ]


def load_edits(
    path: Path | None,
    config: PrototypeConfig,
    vocabulary: ControlledVocabulary,
) -> tuple[list[EditCommand], list[Failure]]:
    if path is None:
        return [], []
    rows, failures = _read_jsonl(path, "edits")
    commands: list[EditCommand] = []
    for line, row in rows:
        command, command_failures = validate_edit_command(
            row, config, vocabulary, line=line
        )
        failures.extend(command_failures)
        if command is not None:
            commands.append(command)
    return commands, failures


def load_inputs(
    config_path: Path,
    vocabulary_path: Path,
    stopwords_path: Path,
    documents_path: Path,
    queries_path: Path,
    edits_path: Path | None = None,
) -> LoadedInputs:
    from .config import load_config

    config = load_config(config_path)
    vocabulary, vocabulary_failures = load_vocabulary(vocabulary_path)
    stopwords, stopword_failures = load_stopwords(stopwords_path)
    critical_failures = vocabulary_failures + stopword_failures
    if critical_failures:
        raise InputValidationError(critical_failures)

    failures: list[Failure] = []
    doc_rows, doc_failures = _read_jsonl(documents_path, "documents")
    failures.extend(doc_failures)
    documents: dict[str, Document] = {}
    for line, row in doc_rows:
        document, document_failures = validate_document(
            row, config, vocabulary, source="documents", line=line
        )
        failures.extend(document_failures)
        if document is None:
            continue
        if document.doc_id in documents:
            failures.append(
                _failure(
                    "documents",
                    line,
                    document.doc_id,
                    "文档标识重复",
                    "DOC_DUPLICATE",
                )
            )
            continue
        documents[document.doc_id] = document

    query_rows, query_failures = _read_jsonl(queries_path, "queries")
    failures.extend(query_failures)
    queries: dict[str, QuerySpec] = {}
    for line, row in query_rows:
        query, query_failures = validate_query(
            row, config, vocabulary, source="queries", line=line
        )
        failures.extend(query_failures)
        if query is None:
            continue
        if query.query_id in queries:
            failures.append(
                _failure(
                    "queries",
                    line,
                    query.query_id,
                    "查询标识重复",
                    "QUERY_DUPLICATE",
                )
            )
            continue
        queries[query.query_id] = query

    edits, edit_failures = load_edits(edits_path, config, vocabulary)
    failures.extend(edit_failures)

    file_paths = [config_path, vocabulary_path, stopwords_path, documents_path, queries_path]
    if edits_path is not None:
        file_paths.append(edits_path)
    input_fingerprint = sha256_json(
        {
            str(path): sha256_file(path) if path.exists() else "missing"
            for path in file_paths
        }
    )

    return LoadedInputs(
        config=config,
        vocabulary=vocabulary,
        stopwords=stopwords,
        documents=tuple(documents[key] for key in sorted(documents)),
        queries=tuple(queries[key] for key in sorted(queries)),
        edits=tuple(edits),
        failures=tuple(sorted(failures, key=Failure.sort_key)),
        input_fingerprint=input_fingerprint,
    )
