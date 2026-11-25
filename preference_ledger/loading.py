from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from .codes import (
    E_ANNOTATOR_WEIGHT_RANGE,
    E_NON_FINITE,
    E_WINDOW_MONOTONIC,
)
from .model import (
    ALLOWED_DECIMAL_PLACES,
    ALLOWED_EXCHANGE_VERSIONS,
    ALLOWED_LEDGER_GENERATIONS,
    ALLOWED_SUBWORD_RULES,
    DECLARED_QUANT_SCHEMES,
    Config,
    Issue,
    LoadedInput,
    PreferenceRecord,
)
from .util import fingerprint, is_strict_int, whitespace_terms


def read_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def parse_weights(raw: Any, stage: str = "load_weights") -> tuple[dict[str, int], list[Issue]]:
    issues: list[Issue] = []
    if raw is None:
        return {}, [Issue(E_ANNOTATOR_WEIGHT_RANGE, stage, "weights", "missing weights document")]
    entries = raw.get("weights", raw) if isinstance(raw, dict) else raw
    if isinstance(entries, list):
        normalized: dict[str, Any] = {}
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or "annotator_id" not in entry or "weight_bp" not in entry:
                issues.append(Issue(E_ANNOTATOR_WEIGHT_RANGE, stage, f"weights[{index}]", "weight entry requires annotator_id and weight_bp"))
                continue
            normalized[str(entry["annotator_id"])] = entry["weight_bp"]
        entries = normalized
    if not isinstance(entries, dict):
        return {}, [Issue(E_ANNOTATOR_WEIGHT_RANGE, stage, "weights", "weights must be a mapping or list")]

    valid: dict[str, int] = {}
    for annotator_id in sorted(entries, key=str):
        value = entries[annotator_id]
        target = f"annotator:{annotator_id}"
        if isinstance(value, float) and not math.isfinite(value):
            issues.append(Issue(E_NON_FINITE, stage, target, f"weight is non-finite: {value!r}"))
            continue
        if not is_strict_int(value):
            issues.append(Issue(E_ANNOTATOR_WEIGHT_RANGE, stage, target, f"weight must be an integer basis-point value, got {value!r}"))
            continue
        if value < 0 or value > 10_000:
            issues.append(Issue(E_ANNOTATOR_WEIGHT_RANGE, stage, target, f"weight {value} outside [0, 10000]"))
            continue
        valid[str(annotator_id)] = int(value)
    return valid, issues


def _positive_int_sequence(raw: Any, field_name: str, issues: list[Issue], code: str = E_WINDOW_MONOTONIC) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        issues.append(Issue(code, "load_config", field_name, f"{field_name} must be a non-empty list"))
        return ()
    values: list[int] = []
    for index, value in enumerate(raw):
        if not is_strict_int(value) or value <= 0:
            issues.append(Issue(code, "load_config", f"{field_name}[{index}]", "must be a positive integer"))
        values.append(int(value) if is_strict_int(value) else -1)
    if any(value <= 0 for value in values):
        return tuple(value for value in values if value > 0)
    if any(values[i] >= values[i + 1] for i in range(len(values) - 1)):
        issues.append(Issue(code, "load_config", field_name, f"{field_name} must be strictly increasing: {values}"))
    return tuple(values)


def parse_config(raw: Any, source: str = "<config>") -> tuple[Config | None, list[Issue]]:
    issues: list[Issue] = []
    if not isinstance(raw, dict):
        return None, [Issue("E_CONFIG_INVALID", "load_config", source, "configuration must be a mapping")]

    feature_dim = raw.get("feature_dim")
    hash_bucket_count = raw.get("hash_bucket_count")
    if not is_strict_int(feature_dim) or feature_dim <= 0:
        issues.append(Issue("E_CONFIG_INVALID", "load_config", "feature_dim", "must be a positive integer"))
        feature_dim = 0
    if not is_strict_int(hash_bucket_count) or hash_bucket_count <= 0:
        issues.append(Issue("E_CONFIG_INVALID", "load_config", "hash_bucket_count", "must be a positive integer"))
        hash_bucket_count = 0

    rule_version = raw.get("subword_rule_version", "sw-v1")
    if rule_version not in ALLOWED_SUBWORD_RULES:
        issues.append(Issue("E_CONFIG_INVALID", "load_config", "subword_rule_version", f"must be one of {sorted(ALLOWED_SUBWORD_RULES)}"))

    normalize = raw.get("normalize", True)
    if not isinstance(normalize, bool):
        issues.append(Issue("E_CONFIG_INVALID", "load_config", "normalize", "must be boolean"))

    vocab_raw = raw.get("controlled_vocabulary", [])
    if not isinstance(vocab_raw, list) or not vocab_raw or not all(isinstance(x, str) and x for x in vocab_raw):
        issues.append(Issue("E_CONFIG_INVALID", "load_config", "controlled_vocabulary", "must be a non-empty string list"))
        vocab: tuple[str, ...] = ()
    else:
        vocab = tuple(sorted({x.lower() if normalize else x for x in vocab_raw}))

    window_lengths = _positive_int_sequence(raw.get("window_lengths"), "window_lengths", issues)
    boundaries = _positive_int_sequence(raw.get("partition_boundaries"), "partition_boundaries", issues)

    decimal_places = raw.get("fixed_decimal_places", 4)
    if decimal_places not in ALLOWED_DECIMAL_PLACES:
        issues.append(Issue("E_CONFIG_INVALID", "load_config", "fixed_decimal_places", f"must be one of {sorted(ALLOWED_DECIMAL_PLACES)}"))
        decimal_places = 4

    quant_scheme = raw.get("quant_scheme", "Q_FINE")
    if quant_scheme not in DECLARED_QUANT_SCHEMES:
        issues.append(Issue("E_CONFIG_INVALID", "load_config", "quant_scheme", f"must be one of {sorted(DECLARED_QUANT_SCHEMES)}"))
        quant_scheme = "Q_FINE"

    exchange_version = raw.get("exchange_version", "g2")
    if exchange_version not in ALLOWED_EXCHANGE_VERSIONS:
        issues.append(Issue("E_VERSION_UNKNOWN", "load_config", "exchange_version", f"unknown exchange version {exchange_version!r}"))
        exchange_version = "g2"

    generation = raw.get("ledger_generation", "L2")
    if generation not in ALLOWED_LEDGER_GENERATIONS:
        issues.append(Issue("E_VERSION_UNKNOWN", "load_config", "ledger_generation", f"unknown ledger generation {generation!r}"))
        generation = "L2"

    threshold = raw.get("decision_threshold_permille", 0)
    if not is_strict_int(threshold):
        issues.append(Issue("E_CONFIG_INVALID", "load_config", "decision_threshold_permille", "must be integer permille"))
        threshold = 0

    score_weights_raw = raw.get("score_weights", [])
    score_weights: tuple[int, ...] = ()
    if score_weights_raw:
        if not isinstance(score_weights_raw, list) or not all(is_strict_int(x) for x in score_weights_raw):
            issues.append(Issue("E_CONFIG_INVALID", "load_config", "score_weights", "must be an integer list"))
        elif feature_dim and len(score_weights_raw) != feature_dim:
            issues.append(Issue("E_CONFIG_INVALID", "load_config", "score_weights", f"expected {feature_dim} weights"))
        else:
            score_weights = tuple(int(x) for x in score_weights_raw)
    elif feature_dim > 0:
        score_weights = tuple(((fingerprint({"i": i}) and int(fingerprint({"i": i})[:8], 16)) % 7) - 3 for i in range(feature_dim))

    window_budget = raw.get("window_budget", 10_000)
    session_quota = raw.get("session_quota", 10_000)
    if not is_strict_int(window_budget) or window_budget <= 0:
        issues.append(Issue("E_CONFIG_INVALID", "load_config", "window_budget", "must be positive integer"))
        window_budget = 10_000
    if not is_strict_int(session_quota) or session_quota <= 0:
        issues.append(Issue("E_CONFIG_INVALID", "load_config", "session_quota", "must be positive integer"))
        session_quota = 10_000

    if issues:
        return None, issues
    config = Config(
        feature_dim=feature_dim,
        hash_bucket_count=hash_bucket_count,
        subword_rule_version=rule_version,
        normalize=bool(normalize),
        controlled_vocabulary=vocab,
        window_lengths=window_lengths,
        partition_boundaries=boundaries,
        fixed_decimal_places=int(decimal_places),
        quant_scheme=quant_scheme,
        exchange_version=exchange_version,
        ledger_generation=generation,
        decision_threshold_permille=threshold,
        score_weights=score_weights,
        window_budget=window_budget,
        session_quota=session_quota,
    )
    return config, []


def load_config_file(path: str | Path) -> tuple[Config | None, list[Issue]]:
    raw = read_yaml(path)
    return parse_config(raw, str(path))


def parse_records_text(
    text: str,
    weights: dict[str, int],
    config: Config,
    stage: str = "load_records",
) -> tuple[list[PreferenceRecord], list[Issue]]:
    issues: list[Issue] = []
    records: list[PreferenceRecord] = []
    grouped: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        target = f"jsonl:{line_number}"
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(Issue("E_RECORD_JSON", stage, target, str(exc)))
            continue
        if not isinstance(obj, dict):
            issues.append(Issue("E_RECORD_INVALID", stage, target, "record must be an object"))
            continue

        session_id = obj.get("session_id")
        turn = obj.get("turn")
        prompt = obj.get("prompt")
        chosen = obj.get("chosen")
        rejected = obj.get("rejected")
        annotator_id = obj.get("annotator_id")
        event_ms = obj.get("event_ms")
        record_ok = True

        if not isinstance(session_id, str) or not session_id:
            issues.append(Issue("E_RECORD_INVALID", stage, f"{target}:session_id", "session_id must be non-empty"))
            record_ok = False
        if not is_strict_int(turn) or turn <= 0:
            issues.append(Issue("E_RECORD_INVALID", stage, f"{target}:turn", "turn must be a positive integer"))
            record_ok = False
        for field_name, value in (("prompt", prompt), ("chosen", chosen), ("rejected", rejected)):
            if not isinstance(value, str) or not value.strip():
                issues.append(Issue("E_RECORD_INVALID", stage, f"{target}:{field_name}", f"{field_name} must be non-empty"))
                record_ok = False
                continue
            terms = whitespace_terms(value, config.normalize)
            vocab = set(config.controlled_vocabulary)
            outside = [term for term in terms if (term.lower() if config.normalize else term) not in vocab]
            if outside:
                issues.append(Issue("E_VOCAB_OUTSIDE", stage, f"{target}:{field_name}", f"terms outside controlled vocabulary: {outside}"))
                record_ok = False
        if not is_strict_int(event_ms) or event_ms < 0:
            issues.append(Issue("E_RECORD_INVALID", stage, f"{target}:event_ms", "event_ms must be a non-negative integer"))
            record_ok = False
        if not isinstance(annotator_id, str) or not annotator_id:
            issues.append(Issue("E_RECORD_INVALID", stage, f"{target}:annotator_id", "annotator_id must be non-empty"))
            record_ok = False
        elif annotator_id not in weights:
            issues.append(Issue("E_ANNOTATOR_UNDECLARED", stage, f"{target}:annotator_id", f"annotator {annotator_id!r} is not declared in weights table"))
            record_ok = False

        if record_ok:
            assert isinstance(session_id, str) and is_strict_int(turn)
            assert isinstance(prompt, str) and isinstance(chosen, str) and isinstance(rejected, str)
            assert isinstance(annotator_id, str) and is_strict_int(event_ms)
            record = PreferenceRecord(session_id, int(turn), prompt, chosen, rejected, annotator_id, int(event_ms))
            records.append(record)
            grouped.setdefault(session_id, []).append((line_number, int(turn), obj))

    for session_id in sorted(grouped):
        turns = [turn for _, turn, _ in grouped[session_id]]
        if len(turns) != len(set(turns)):
            issues.append(Issue("E_TURN_HOLE", stage, f"session:{session_id}", f"duplicate turns: {turns}"))
            continue
        ordered = sorted(turns)
        expected = list(range(1, len(ordered) + 1))
        if ordered != expected:
            issues.append(Issue("E_TURN_HOLE", stage, f"session:{session_id}", f"turns must be 1..n without holes, got {ordered}"))

    records.sort(key=lambda record: (record.session_id, record.turn))
    return records, issues


def load_input(records_path: str | Path, weights_path: str | Path, config_path: str | Path) -> LoadedInput:
    raw_weights = read_yaml(weights_path)
    weights, weight_issues = parse_weights(raw_weights)
    config, config_issues = load_config_file(config_path)
    issues = [*weight_issues, *config_issues]
    if config is None:
        return LoadedInput(None, [], weights, issues)
    records, record_issues = parse_records_text(read_text(records_path), weights, config)
    issues.extend(record_issues)
    return LoadedInput(config, records, weights, issues)
