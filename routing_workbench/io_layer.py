"""Input loading and per-record validation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from .config import ConfigValidationError, parse_config, parse_vocabulary_file
from .errors import ErrorCode, Failure
from .models import RoutingConfig, Utterance
from .util import canonical_json, fingerprint


def load_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: str | Path, vocabulary_path: str | Path | None = None) -> tuple[RoutingConfig | None, list[Failure]]:
    failures: list[Failure] = []
    fallback_vocabulary: tuple[str, ...] = ()
    if vocabulary_path is not None:
        try:
            fallback_vocabulary = parse_vocabulary_file(vocabulary_path)
        except OSError as exc:
            failures.append(
                Failure(
                    ErrorCode.ROUTE_INVALID.value,
                    "__config__",
                    "validation",
                    f"cannot read vocabulary file: {exc}",
                    field="vocabulary",
                )
            )
    try:
        raw = load_yaml(path)
    except Exception as exc:  # YAML parser and OS errors are input failures, never crashes.
        failures.append(
            Failure(
                ErrorCode.ROUTE_INVALID.value,
                "__config__",
                "validation",
                f"cannot parse routing configuration: {exc}",
                field="config",
            )
        )
        return None, failures
    try:
        return parse_config(raw, fallback_vocabulary), failures
    except ConfigValidationError as exc:
        failures.append(exc.as_failure())
        return None, failures


def _get(mapping: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _normalize_slots_truth(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, (list, tuple)):
        result: dict[str, str] = {}
        for item in raw:
            if isinstance(item, dict):
                name = _get(item, ("slot", "name", "key", "type"))
                value = _get(item, ("value", "text", "fill", "token"))
                if name is None or value is None:
                    raise ValueError("slot truth object must include slot and value")
                result[str(name)] = str(value)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                result[str(item[0])] = str(item[1])
            else:
                raise ValueError("slot truth must be a mapping or tuple list")
        return result
    raise ValueError("slots must be a mapping or tuple list")


def parse_utterance(raw: Any, line_number: int) -> Utterance:
    if not isinstance(raw, dict):
        raise ValueError("JSONL record must be an object")
    session_id_raw = _get(raw, ("session_id", "sessionId", "conversation_id", "conversationId", "session"))
    turn_raw = _get(raw, ("turn", "turn_index", "turnIndex", "round", "round_no", "sequence", "seq"))
    tokens_raw = _get(raw, ("tokens", "terms", "token_seq", "tokenSequence", "words", "items"))
    domain_raw = _get(raw, ("domain", "domain_label", "domainLabel", "domain_truth"))
    intent_raw = _get(raw, ("intent", "intent_label", "intentLabel", "intent_truth"))
    slots_raw = _get(raw, ("slots", "slot_truth", "slotTruth", "truth_slots", "slot_triples"))
    utterance_id_raw = _get(raw, ("utterance_id", "utteranceId", "id", "uid", "sample_id"))

    if session_id_raw is None or not str(session_id_raw).strip():
        raise ValueError("session_id must be a non-empty string")
    if turn_raw is None or isinstance(turn_raw, bool):
        raise ValueError("turn must be a positive integer")
    try:
        turn = int(turn_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("turn must be a positive integer") from exc
    if turn <= 0:
        raise ValueError("turn must be a positive integer")
    if not isinstance(tokens_raw, (list, tuple)) or not tokens_raw:
        raise ValueError("tokens must be a non-empty list")
    tokens: list[str] = []
    for index, token in enumerate(tokens_raw):
        if not isinstance(token, str) or not token:
            raise ValueError(f"tokens[{index}] must be a non-empty string")
        tokens.append(token)
    session_id = str(session_id_raw)
    utterance_id = str(utterance_id_raw) if utterance_id_raw is not None else f"{session_id}-U{turn}"
    if not utterance_id:
        raise ValueError("utterance_id must not be empty")
    domain = str(domain_raw) if domain_raw is not None else None
    intent = str(intent_raw) if intent_raw is not None else None
    try:
        slots = _normalize_slots_truth(slots_raw)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    return Utterance(
        utterance_id=utterance_id,
        session_id=session_id,
        turn=turn,
        tokens=tuple(tokens),
        domain_truth=domain,
        intent_truth=intent,
        slots_truth=slots,
        raw=dict(raw),
        source_line=line_number,
    )


def load_utterances(path: str | Path) -> tuple[list[Utterance], list[Failure]]:
    utterances: list[Utterance] = []
    failures: list[Failure] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(
                Failure(
                    ErrorCode.ROUTE_INVALID.value,
                    f"__line_{line_number}__",
                    "validation",
                    f"invalid JSON: {exc.msg}",
                    field=f"line:{line_number}",
                )
            )
            continue
        try:
            utterances.append(parse_utterance(raw, line_number))
        except ValueError as exc:
            utterance_id = "__unknown__"
            if isinstance(raw, dict):
                utterance_id = str(
                    _get(raw, ("utterance_id", "utteranceId", "id", "uid"), f"__line_{line_number}__")
                )
            failures.append(
                Failure(
                    ErrorCode.ROUTE_INVALID.value,
                    utterance_id,
                    "validation",
                    str(exc),
                    field=f"line:{line_number}",
                )
            )
    return utterances, failures


def validate_utterances(
    utterances: list[Utterance],
    config: RoutingConfig | None,
) -> tuple[dict[str, Utterance], list[Failure]]:
    failures: list[Failure] = []
    by_id: dict[str, Utterance] = {}
    duplicate_ids: set[str] = set()
    for utterance in utterances:
        if utterance.utterance_id in by_id:
            duplicate_ids.add(utterance.utterance_id)
        else:
            by_id[utterance.utterance_id] = utterance
    for utterance_id in sorted(duplicate_ids):
        failures.append(
            Failure(
                ErrorCode.ROUTE_INVALID.value,
                utterance_id,
                "validation",
                "utterance_id must be unique",
                field="utterance_id",
            )
        )

    # Session/turn uniqueness and continuity are checked over every parsed record, including records
    # whose other fields may be invalid.
    session_turns: dict[str, list[tuple[int, str]]] = {}
    pair_owner: dict[tuple[str, int], str] = {}
    for utterance in utterances:
        pair = (utterance.session_id, utterance.turn)
        if pair in pair_owner:
            failures.append(
                Failure(
                    ErrorCode.ROUTE_INVALID.value,
                    utterance.utterance_id,
                    "validation",
                    f"duplicate session/turn pair {utterance.session_id!r}/{utterance.turn}",
                    field="session_id+turn",
                    details={"other_utterance_id": pair_owner[pair]},
                )
            )
        else:
            pair_owner[pair] = utterance.utterance_id
        session_turns.setdefault(utterance.session_id, []).append((utterance.turn, utterance.utterance_id))
    for session_id, turns in session_turns.items():
        ordered = sorted(turns, key=lambda item: (item[0], item[1]))
        expected = 1
        for turn, utterance_id in ordered:
            if turn != expected:
                failures.append(
                    Failure(
                        ErrorCode.ROUTE_INVALID.value,
                        utterance_id,
                        "validation",
                        f"turn sequence for session {session_id!r} has a gap: expected {expected}, got {turn}",
                        field="turn",
                    )
                )
                # Continue from the observed value so only the first gap per stretch is reported.
                expected = turn
            expected += 1

    vocabulary = config.vocabulary if config is not None else frozenset()
    domains = config.domains if config is not None else {}
    for utterance in utterances:
        record_invalid = False
        if vocabulary:
            out_of_vocab = sorted({token for token in utterance.tokens if token not in vocabulary})
            if out_of_vocab:
                record_invalid = True
                failures.append(
                    Failure(
                        ErrorCode.ROUTE_INVALID.value,
                        utterance.utterance_id,
                        "validation",
                        f"tokens outside controlled vocabulary: {out_of_vocab!r}",
                        field="tokens",
                    )
                )
        if config is not None:
            if utterance.domain_truth is None or utterance.domain_truth not in domains:
                record_invalid = True
                failures.append(
                    Failure(
                        ErrorCode.ROUTE_INVALID.value,
                        utterance.utterance_id,
                        "validation",
                        f"domain label {utterance.domain_truth!r} is not in the domain set",
                        field="domain",
                    )
                )
            elif utterance.intent_truth is None or utterance.intent_truth not in domains[utterance.domain_truth].intents:
                record_invalid = True
                failures.append(
                    Failure(
                        ErrorCode.ROUTE_INVALID.value,
                        utterance.utterance_id,
                        "validation",
                        f"intent label {utterance.intent_truth!r} does not belong to domain {utterance.domain_truth!r}",
                        field="intent",
                    )
                )
            undeclared_slots = sorted(set(utterance.slots_truth) - set(config.slot_declarations))
            if undeclared_slots:
                record_invalid = True
                failures.append(
                    Failure(
                        ErrorCode.ROUTE_INVALID.value,
                        utterance.utterance_id,
                        "validation",
                        f"truth slots are not declared: {undeclared_slots!r}",
                        field="slots",
                    )
                )
        if record_invalid:
            by_id.pop(utterance.utterance_id, None)

    for utterance_id in duplicate_ids:
        by_id.pop(utterance_id, None)

    # A duplicate session/turn pair or an empty-string ID is also excluded from routing.
    blocked_ids = {
        failure.utterance_id
        for failure in failures
        if failure.field in {"utterance_id", "session_id+turn", "turn"}
    }
    for utterance_id in blocked_ids:
        by_id.pop(utterance_id, None)
    return by_id, _sort_failures(failures)


def _sort_failures(failures: Iterable[Failure]) -> list[Failure]:
    return sorted(
        failures,
        key=lambda failure: (
            failure.utterance_id,
            failure.stage,
            failure.business_code,
            failure.field or "",
            failure.message,
            failure.command_index if failure.command_index is not None else -1,
            failure.injection_id or "",
        ),
    )


def load_commands(path: str | Path) -> tuple[list[dict[str, Any]], list[Failure]]:
    failures: list[Failure] = []
    try:
        raw = load_json(path)
    except Exception as exc:
        failures.append(
            Failure(
                ErrorCode.ROUTE_INVALID.value,
                "__commands__",
                "commands",
                f"cannot parse command stream: {exc}",
                field="commands",
            )
        )
        return [], failures
    if isinstance(raw, dict):
        raw = raw.get("commands", raw.get("stream", raw.get("items")))
    if not isinstance(raw, list):
        failures.append(
            Failure(
                ErrorCode.ROUTE_INVALID.value,
                "__commands__",
                "commands",
                "command stream must be a JSON array or contain a commands array",
                field="commands",
            )
        )
        return [], failures
    commands: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            failures.append(
                Failure(
                    ErrorCode.ROUTE_INVALID.value,
                    f"__command_{index}__",
                    "commands",
                    "command must be an object",
                    field=f"commands[{index}]",
                    command_index=index,
                )
            )
            continue
        normalized = dict(item)
        normalized.setdefault("command_index", index)
        normalized.setdefault("command_id", str(item.get("id", f"C{index + 1:04d}")))
        commands.append(normalized)
    return commands, failures


def dataset_fingerprint(utterances: Iterable[Utterance]) -> str:
    return fingerprint(
        {
            "utterances": [
                utterance.canonical_content()
                for utterance in sorted(utterances, key=lambda item: item.utterance_id)
            ]
        }
    )


def raw_dataset_fingerprint(raw_records: Iterable[dict[str, Any]]) -> str:
    return fingerprint({"utterances": list(raw_records)})


def utterance_content_fingerprint(utterance: Utterance) -> str:
    return fingerprint(utterance.canonical_content())


def find_sample_file(samples_dir: str | Path, names: Iterable[str], globs: Iterable[str] = ()) -> Path | None:
    root = Path(samples_dir)
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    for pattern in globs:
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def resolve_inputs(
    samples_dir: str | Path | None,
    utterances_path: str | Path | None,
    config_path: str | Path | None,
    commands_path: str | Path | None,
    vocabulary_path: str | Path | None,
    injections_path: str | Path | None = None,
) -> dict[str, Path | None]:
    if samples_dir is not None:
        root = Path(samples_dir)
        utterances_path = utterances_path or find_sample_file(root, ("utterances.jsonl", "samples.jsonl", "tickets.jsonl"), ("*.jsonl",))
        config_path = config_path or find_sample_file(root, ("routing.yaml", "routing.yml", "config.yaml", "config.yml"), ("*.yaml", "*.yml"))
        commands_path = commands_path or find_sample_file(root, ("commands.json", "command_stream.json", "edits.json"))
        vocabulary_path = vocabulary_path or find_sample_file(root, ("vocabulary.txt", "vocab.txt"))
        injections_path = injections_path or find_sample_file(root, ("injections.yaml", "injections.yml", "injections.json"))
    return {
        "utterances": Path(utterances_path) if utterances_path is not None else None,
        "config": Path(config_path) if config_path is not None else None,
        "commands": Path(commands_path) if commands_path is not None else None,
        "vocabulary": Path(vocabulary_path) if vocabulary_path is not None else None,
        "injections": Path(injections_path) if injections_path is not None else None,
    }


def replace_utterance_record(old: Utterance, raw_patch: dict[str, Any]) -> Utterance:
    raw = dict(old.raw)
    raw.update(raw_patch)
    parsed = parse_utterance(raw, old.source_line)
    return replace(parsed, source_line=old.source_line)
