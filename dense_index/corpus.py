from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .errors import E_CORPUS_INVALID, E_COMMAND_INVALID, DiagnosticError, Failure


@dataclass
class CorpusBundle:
    records: List[Dict[str, Any]]
    vocabulary: List[str]
    path: Optional[str] = None


def load_vocabulary(path: str | Path) -> List[str]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failure = Failure(E_CORPUS_INVALID, f"invalid vocabulary file: {exc}", path=str(path))
        raise DiagnosticError(E_CORPUS_INVALID, str(exc), [failure]) from exc
    if isinstance(raw, dict):
        terms = raw.get("terms")
    else:
        terms = raw
    if not isinstance(terms, list) or not terms or not all(isinstance(term, str) and term for term in terms):
        failure = Failure(E_CORPUS_INVALID, "vocabulary.terms must be a non-empty string list", path=str(path))
        raise DiagnosticError(E_CORPUS_INVALID, "invalid vocabulary", [failure])
    if len(set(terms)) != len(terms):
        failure = Failure(E_CORPUS_INVALID, "vocabulary terms must be unique", path=str(path))
        raise DiagnosticError(E_CORPUS_INVALID, "duplicate vocabulary terms", [failure])
    return terms


def _nonempty_scalar(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def normalize_sequence(value: Any) -> Tuple[int, Any]:
    if isinstance(value, bool):
        return (2, str(value).lower())
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return (0, int(text))
        except ValueError:
            return (1, text)
    return (2, str(value))


def parse_corpus_line(
    line: str,
    line_number: int,
    path: str,
    vocab: Set[str],
    seen_paragraph_ids: Set[str],
) -> Tuple[Optional[Dict[str, Any]], List[Failure]]:
    failures: List[Failure] = []
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        return None, [
            Failure(
                E_CORPUS_INVALID,
                f"JSON parse error: {exc.msg}",
                path=path,
                line=line_number,
                details={"column": exc.colno},
            )
        ]

    if not isinstance(obj, dict):
        return None, [Failure(E_CORPUS_INVALID, "line must contain a JSON object", path=path, line=line_number)]

    paragraph_id = obj.get("paragraph_id", obj.get("id", obj.get("pid")))
    document_id = obj.get("document_id", obj.get("doc_id"))
    sequence_no = obj.get("sequence_no", obj.get("chunk_index", obj.get("paragraph_index", obj.get("seq"))))
    terms = obj.get("terms", obj.get("term_sequence"))

    if not _nonempty_scalar(paragraph_id):
        failures.append(Failure(E_CORPUS_INVALID, "paragraph_id is required", path=path, line=line_number, field="paragraph_id"))
    elif not isinstance(paragraph_id, str):
        paragraph_id = str(paragraph_id)

    if paragraph_id and paragraph_id in seen_paragraph_ids:
        failures.append(
            Failure(
                E_CORPUS_INVALID,
                f"duplicate paragraph_id {paragraph_id!r}",
                path=path,
                line=line_number,
                field="paragraph_id",
            )
        )
    elif paragraph_id:
        seen_paragraph_ids.add(paragraph_id)

    if not _nonempty_scalar(document_id):
        failures.append(Failure(E_CORPUS_INVALID, "document_id must be non-empty", path=path, line=line_number, field="document_id"))
    elif not isinstance(document_id, str):
        document_id = str(document_id)

    if not _nonempty_scalar(sequence_no):
        failures.append(Failure(E_CORPUS_INVALID, "sequence_no/chunk_index must be non-empty", path=path, line=line_number, field="sequence_no"))

    if not isinstance(terms, list):
        failures.append(Failure(E_CORPUS_INVALID, "terms must be a list", path=path, line=line_number, field="terms"))
        terms = []
    else:
        for index, term in enumerate(terms):
            if not isinstance(term, str) or term not in vocab:
                failures.append(
                    Failure(
                        E_CORPUS_INVALID,
                        f"term at position {index} is outside controlled vocabulary",
                        path=path,
                        line=line_number,
                        field=f"terms[{index}]",
                        details={"term": term},
                    )
                )

    if failures:
        return None, failures

    record = {
        "paragraph_id": paragraph_id,
        "document_id": document_id,
        "sequence_no": sequence_no,
        "sequence_key": list(normalize_sequence(sequence_no)),
        "terms": list(terms),
        "source_line": line_number,
    }
    return record, []


def load_corpus_file(path: str | Path, vocabulary: Iterable[str], strict: bool = True) -> CorpusBundle:
    path = Path(path)
    vocab = set(vocabulary)
    records: List[Dict[str, Any]] = []
    failures: List[Failure] = []
    seen: Set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        failure = Failure(E_CORPUS_INVALID, f"cannot read corpus: {exc}", path=str(path))
        raise DiagnosticError(E_CORPUS_INVALID, str(exc), [failure]) from exc

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        record, line_failures = parse_corpus_line(line, line_number, str(path), vocab, seen)
        failures.extend(line_failures)
        if record is not None:
            records.append(record)

    if failures and strict:
        raise DiagnosticError(E_CORPUS_INVALID, f"{len(failures)} corpus error(s)", failures)
    return CorpusBundle(records=records, vocabulary=sorted(vocab), path=str(path))


KNOWN_COMMANDS = {
    "append_batch",
    "replace_paragraph",
    "delete_paragraph",
    "replace_block",
    "delete_block",
    "set_config",
    "set_threshold",
    "set_point_budget",
    "switch_receipt_version",
}


def load_command_stream(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failure = Failure(E_COMMAND_INVALID, f"invalid command JSON: {exc}", path=str(path))
        raise DiagnosticError(E_COMMAND_INVALID, str(exc), [failure]) from exc
    if isinstance(raw, dict):
        commands = raw.get("commands")
    else:
        commands = raw
    if not isinstance(commands, list) or not commands:
        failure = Failure(E_COMMAND_INVALID, "command stream must contain a non-empty commands list", path=str(path))
        raise DiagnosticError(E_COMMAND_INVALID, "invalid command stream", [failure])

    failures: List[Failure] = []
    normalized: List[Dict[str, Any]] = []
    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            failures.append(Failure(E_COMMAND_INVALID, "command must be an object", path=str(path), line=index))
            continue
        op = command.get("op")
        if op not in KNOWN_COMMANDS:
            failures.append(Failure(E_COMMAND_INVALID, f"unknown command op {op!r}", path=str(path), line=index, field="op"))
            continue
        command_id = command.get("id", f"cmd-{index + 1:03d}")
        item = dict(command)
        item["id"] = command_id
        item["index"] = index
        if op in {"append_batch"} and not isinstance(command.get("paragraphs"), list):
            failures.append(Failure(E_COMMAND_INVALID, "append_batch requires paragraphs list", path=str(path), line=index, field="paragraphs"))
        if op in {"replace_paragraph"} and not isinstance(command.get("paragraph"), dict):
            failures.append(Failure(E_COMMAND_INVALID, "replace_paragraph requires paragraph object", path=str(path), line=index, field="paragraph"))
        if op in {"delete_paragraph", "replace_block", "delete_block"} and not command.get("target_id"):
            failures.append(Failure(E_COMMAND_INVALID, f"{op} requires target_id", path=str(path), line=index, field="target_id"))
        if op == "replace_block" and not isinstance(command.get("terms"), list):
            failures.append(Failure(E_COMMAND_INVALID, "replace_block requires terms list", path=str(path), line=index, field="terms"))
        if op == "set_config" and not isinstance(command.get("values"), dict):
            failures.append(Failure(E_COMMAND_INVALID, "set_config requires values mapping", path=str(path), line=index, field="values"))
        normalized.append(item)

    if failures:
        raise DiagnosticError(E_COMMAND_INVALID, f"{len(failures)} command error(s)", failures)
    return normalized
