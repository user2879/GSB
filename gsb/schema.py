from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .errors import SchemaViolation


ALLOWED_DECIMAL_WIDTHS = (2, 4, 6)
ALLOWED_BIT_WIDTHS = (32, 64)
ALLOWED_QUANT_GEARS = ("coarse", "standard", "fine")
ALLOWED_ROUNDING = ("half_up", "half_even", "down")
ALLOWED_TICKET_STATUS = ("open", "closed")
ALLOWED_POLARITY = ("positive", "negative")
ALLOWED_VOCAB_KIND = ("sentiment", "discourse", "neutral", "punctuation")


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def issue(entity: str, entity_id: Any, field: str, message: str) -> dict[str, Any]:
    return {"entity": entity, "id": entity_id, "field": field, "message": message}


def load_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise SchemaViolation([issue("file", str(path), "path", "required sample file is missing")]) from exc
    except json.JSONDecodeError as exc:
        raise SchemaViolation(
            [issue("file", str(path), f"json:{exc.lineno}:{exc.colno}", f"invalid JSON: {exc.msg}")]
        ) from exc


def load_jsonl(path: str | Path) -> list[Any]:
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise SchemaViolation([issue("file", str(path), "path", "required command stream is missing")]) from exc
    rows: list[Any] = []
    issues: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            issues.append(issue("edit_command", line_no, "json", f"invalid JSONL: {exc.msg}"))
    if issues:
        raise SchemaViolation(issues)
    return rows


def _require_mapping(data: Any, entity: str) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return [issue(entity, None, "$", "top-level value must be an object")]
    return []


def validate_vocabulary(data: Any) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    issues.extend(_require_mapping(data, "controlled_vocabulary"))
    if issues:
        return {}, issues
    terms = data.get("terms")
    if not isinstance(terms, list) or not terms:
        return {}, [issue("controlled_vocabulary", data.get("version"), "terms", "must be a non-empty list")]

    by_term: dict[str, dict[str, Any]] = {}
    orders: set[int] = set()
    for index, entry in enumerate(terms):
        ident = entry.get("term") if isinstance(entry, dict) else index
        if not isinstance(entry, dict):
            issues.append(issue("vocab_term", index, "$", "term entry must be an object"))
            continue
        term = entry.get("term")
        if not isinstance(term, str) or not term:
            issues.append(issue("vocab_term", ident, "term", "must be a non-empty string"))
        elif term in by_term:
            issues.append(issue("vocab_term", term, "term", "controlled term must be unique"))
        order = entry.get("order")
        if not is_int(order) or order <= 0:
            issues.append(issue("vocab_term", ident, "order", "must be a positive integer"))
        elif order in orders:
            issues.append(issue("vocab_term", ident, "order", "term order must be unique"))
        else:
            orders.add(order)
        kind = entry.get("kind")
        if kind not in ALLOWED_VOCAB_KIND:
            issues.append(issue("vocab_term", ident, "kind", f"must be one of {ALLOWED_VOCAB_KIND}"))
        if kind == "sentiment":
            if entry.get("polarity") not in ALLOWED_POLARITY:
                issues.append(issue("vocab_term", ident, "polarity", f"must be one of {ALLOWED_POLARITY}"))
            evidence = entry.get("evidence_points")
            if not is_number(evidence) or float(evidence) <= 0:
                issues.append(issue("vocab_term", ident, "evidence_points", "must be a positive number"))
        if isinstance(term, str) and term and term not in by_term:
            by_term[term] = entry
    return by_term, issues


def validate_trigger_list(data: Any, vocab: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    issues = _require_mapping(data, "negation_triggers")
    if issues:
        return issues
    rows = data.get("triggers")
    if not isinstance(rows, list) or not rows:
        return [issue("negation_triggers", data.get("version"), "triggers", "must be a non-empty list")]
    ids: set[str] = set()
    orders: set[int] = set()
    terms: set[str] = set()
    for index, row in enumerate(rows):
        ident = row.get("trigger_id") if isinstance(row, dict) else index
        if not isinstance(row, dict):
            issues.append(issue("negation_trigger", index, "$", "entry must be an object"))
            continue
        trigger_id = row.get("trigger_id")
        if not isinstance(trigger_id, str) or not trigger_id:
            issues.append(issue("negation_trigger", ident, "trigger_id", "must be a non-empty string"))
        elif trigger_id in ids:
            issues.append(issue("negation_trigger", trigger_id, "trigger_id", "trigger_id must be unique"))
        else:
            ids.add(trigger_id)
        term = row.get("term")
        if not isinstance(term, str) or term not in vocab:
            issues.append(issue("negation_trigger", ident, "term", "must name a controlled-vocabulary term"))
        elif vocab[term].get("kind") == "punctuation":
            issues.append(issue("negation_trigger", ident, "term", "punctuation cannot be a negation trigger"))
        elif term in terms:
            issues.append(issue("negation_trigger", ident, "term", "negation trigger term must be unique"))
        else:
            terms.add(term)
        order = row.get("order")
        if not is_int(order) or order <= 0:
            issues.append(issue("negation_trigger", ident, "order", "must be a positive integer"))
        elif order in orders:
            issues.append(issue("negation_trigger", ident, "order", "trigger order must be unique"))
        else:
            orders.add(order)
    return issues


def validate_conflict_list(data: Any, vocab: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    issues = _require_mapping(data, "polarity_conflicts")
    if issues:
        return issues
    rows = data.get("conflicts")
    if not isinstance(rows, list) or not rows:
        return [issue("polarity_conflicts", data.get("version"), "conflicts", "must be a non-empty list")]
    ids: set[str] = set()
    orders: set[int] = set()
    terms: set[str] = set()
    for index, row in enumerate(rows):
        ident = row.get("conflict_id") if isinstance(row, dict) else index
        if not isinstance(row, dict):
            issues.append(issue("polarity_conflict", index, "$", "entry must be an object"))
            continue
        conflict_id = row.get("conflict_id")
        if not isinstance(conflict_id, str) or not conflict_id:
            issues.append(issue("polarity_conflict", ident, "conflict_id", "must be a non-empty string"))
        elif conflict_id in ids:
            issues.append(issue("polarity_conflict", conflict_id, "conflict_id", "conflict_id must be unique"))
        else:
            ids.add(conflict_id)
        term = row.get("term")
        if not isinstance(term, str) or term not in vocab:
            issues.append(issue("polarity_conflict", ident, "term", "must name a controlled-vocabulary term"))
        elif vocab[term].get("kind") == "punctuation":
            issues.append(issue("polarity_conflict", ident, "term", "punctuation cannot be a conflict word"))
        elif term in terms:
            issues.append(issue("polarity_conflict", ident, "term", "conflict term must be unique"))
        else:
            terms.add(term)
        order = row.get("order")
        if not is_int(order) or order <= 0:
            issues.append(issue("polarity_conflict", ident, "order", "must be a positive integer"))
        elif order in orders:
            issues.append(issue("polarity_conflict", ident, "order", "conflict order must be unique"))
        else:
            orders.add(order)
    return issues


def validate_channels(data: Any) -> tuple[dict[str, int], list[dict[str, Any]]]:
    issues = _require_mapping(data, "channel_weights")
    if issues:
        return {}, issues
    rows = data.get("channels")
    if not isinstance(rows, list) or not rows:
        return {}, [issue("channel_weights", data.get("version"), "channels", "must be a non-empty list")]
    weights: dict[str, int] = {}
    for index, row in enumerate(rows):
        channel_id = row.get("channel_id") if isinstance(row, dict) else index
        if not isinstance(row, dict):
            issues.append(issue("channel", index, "$", "entry must be an object"))
            continue
        if not isinstance(channel_id, str) or not channel_id:
            issues.append(issue("channel", channel_id, "channel_id", "must be a non-empty string"))
        elif channel_id in weights:
            issues.append(issue("channel", channel_id, "channel_id", "channel_id must be unique"))
        weight = row.get("weight_permille")
        if not is_int(weight):
            issues.append(issue("channel", channel_id, "weight_permille", "must be an integer permille weight"))
        elif weight < 0 or weight > 1000:
            issues.append(issue("channel", channel_id, "weight_permille", "must be in [0, 1000] permille"))
        elif isinstance(channel_id, str) and channel_id:
            weights[channel_id] = weight
    return weights, issues


def validate_config(config: Any) -> list[dict[str, Any]]:
    issues = _require_mapping(config, "config")
    if issues:
        return issues

    def positive_int(field: str) -> None:
        value = config.get(field)
        if not is_int(value) or value <= 0:
            issues.append(issue("config", config.get("global_version"), field, "must be a positive integer"))

    positive_int("seed")
    positive_int("global_version")
    positive_int("scope_max_tokens")
    positive_int("max_workers")
    positive_int("batch_budget_points")

    for field in (
        "cancellation_strength_points",
        "score_lower_bound_points",
        "positive_threshold_points",
        "negative_threshold_points",
    ):
        if field not in config or not is_number(config[field]):
            issues.append(issue("config", config.get("global_version"), field, "must be a finite number"))

    penalty = config.get("length_penalty_bp_per_token")
    if not is_int(penalty) or penalty < 0 or penalty > 10000:
        issues.append(issue("config", None, "length_penalty_bp_per_token", "must be an integer in [0, 10000]"))
    retention = config.get("length_min_retention_bp")
    if not is_int(retention) or not 1 <= retention <= 10000:
        issues.append(issue("config", None, "length_min_retention_bp", "must be an integer in [1, 10000]"))

    thresholds = config.get("risk_thresholds_points")
    if not isinstance(thresholds, dict) or any(
        not is_number(thresholds.get(name)) for name in ("medium", "high", "critical")
    ):
        issues.append(issue("config", None, "risk_thresholds_points", "must contain numeric medium/high/critical"))
    elif not (float(thresholds["medium"]) < float(thresholds["high"]) < float(thresholds["critical"])):
        issues.append(issue("config", None, "risk_thresholds_points", "thresholds must increase"))

    fixed = config.get("fixed_point")
    if not isinstance(fixed, dict):
        issues.append(issue("config", None, "fixed_point", "must be an object"))
    else:
        if fixed.get("bit_width") not in ALLOWED_BIT_WIDTHS:
            issues.append(issue("config", None, "fixed_point.bit_width", f"must be one of {ALLOWED_BIT_WIDTHS}"))
        if fixed.get("decimal_width") not in ALLOWED_DECIMAL_WIDTHS:
            issues.append(
                issue("config", None, "fixed_point.decimal_width", f"must be one of {ALLOWED_DECIMAL_WIDTHS}")
            )
        if fixed.get("quantization_gear") not in ALLOWED_QUANT_GEARS:
            issues.append(
                issue("config", None, "fixed_point.quantization_gear", f"must be one of {ALLOWED_QUANT_GEARS}")
            )
        if fixed.get("rounding_mode") not in ALLOWED_ROUNDING:
            issues.append(
                issue("config", None, "fixed_point.rounding_mode", f"must be one of {ALLOWED_ROUNDING}")
            )

    tolerances = config.get("error_tolerances_points")
    if not isinstance(tolerances, dict):
        issues.append(issue("config", None, "error_tolerances_points", "must be an object"))
    else:
        for name in ("scope_assignment", "cancellation", "length_penalty", "composition"):
            if not is_number(tolerances.get(name)) or float(tolerances[name]) < 0:
                issues.append(issue("config", None, f"error_tolerances_points.{name}", "must be non-negative number"))
    return issues


def validate_tickets(
    data: Any, vocab: dict[str, dict[str, Any]], channels: dict[str, int]
) -> list[dict[str, Any]]:
    issues = _require_mapping(data, "tickets")
    if issues:
        return issues
    tickets = data.get("tickets")
    if not isinstance(tickets, list) or not tickets:
        return [issue("tickets", data.get("seed"), "tickets", "must be a non-empty list")]
    seen: set[str] = set()
    for index, ticket in enumerate(tickets):
        ticket_id = ticket.get("ticket_id") if isinstance(ticket, dict) else f"#{index}"
        if not isinstance(ticket, dict):
            issues.append(issue("ticket", index, "$", "ticket must be an object"))
            continue
        if not isinstance(ticket_id, str) or not ticket_id:
            issues.append(issue("ticket", ticket_id, "ticket_id", "must be a non-empty string"))
        elif ticket_id in seen:
            issues.append(issue("ticket", ticket_id, "ticket_id", "ticket_id must be unique"))
        else:
            seen.add(ticket_id)
        tokens = ticket.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            issues.append(issue("ticket", ticket_id, "tokens", "must be a non-empty term list"))
        else:
            for pos, token in enumerate(tokens):
                if not isinstance(token, str) or token not in vocab:
                    issues.append(issue("ticket", ticket_id, f"tokens[{pos}]", f"term {token!r} is outside controlled vocabulary"))
        channel_id = ticket.get("channel_id")
        if channel_id not in channels:
            issues.append(issue("ticket", ticket_id, "channel_id", f"unknown channel {channel_id!r}"))
        version = ticket.get("annotator_version")
        if not is_int(version) or version <= 0:
            issues.append(issue("ticket", ticket_id, "annotator_version", "must be a positive integer"))
        status = ticket.get("completion_status")
        if status not in ALLOWED_TICKET_STATUS:
            issues.append(issue("ticket", ticket_id, "completion_status", f"must be one of {ALLOWED_TICKET_STATUS}"))
    return issues


def validate_bundle(bundle: dict[str, Any]) -> None:
    """Validate a normalized in-memory bundle and raise one aggregated error."""
    issues: list[dict[str, Any]] = []
    vocab, vocab_issues = validate_vocabulary({"terms": bundle.get("vocab_list")}, ) if bundle.get("vocab_list") else ({}, [issue("controlled_vocabulary", None, "terms", "missing vocab_list")])
    issues.extend(vocab_issues)
    issues.extend(validate_config(bundle.get("config")))
    channels, channel_issues = validate_channels({"channels": [
        {"channel_id": key, "weight_permille": value} for key, value in (bundle.get("channels") or {}).items()
    ]})
    issues.extend(channel_issues)
    if vocab:
        issues.extend(validate_trigger_list({"triggers": list((bundle.get("triggers") or {}).values())}, vocab))
        issues.extend(validate_conflict_list({"conflicts": list((bundle.get("conflicts") or {}).values())}, vocab))
        issues.extend(validate_tickets({"tickets": bundle.get("tickets")}, vocab, channels or bundle.get("channels") or {}))
    if issues:
        raise SchemaViolation(issues)


def load_sample(sample_dir: str | Path) -> dict[str, Any]:
    root = Path(sample_dir)
    vocab_doc = load_json(root / "controlled_vocabulary.json")
    tickets_doc = load_json(root / "tickets.json")
    triggers_doc = load_json(root / "negation_triggers.json")
    conflicts_doc = load_json(root / "polarity_conflicts.json")
    channels_doc = load_json(root / "channel_weights.json")
    config = load_json(root / "config.json")

    vocab, issues = validate_vocabulary(vocab_doc)
    channel_weights, channel_issues = validate_channels(channels_doc)
    issues.extend(channel_issues)
    issues.extend(validate_config(config))
    if vocab:
        issues.extend(validate_trigger_list(triggers_doc, vocab))
        issues.extend(validate_conflict_list(conflicts_doc, vocab))
    if vocab and channel_weights:
        issues.extend(validate_tickets(tickets_doc, vocab, channel_weights))
    if issues:
        raise SchemaViolation(issues)

    triggers = {row["term"]: dict(row) for row in triggers_doc["triggers"]}
    conflicts = {row["term"]: dict(row) for row in conflicts_doc["conflicts"]}
    trigger_ids = {row["trigger_id"]: dict(row) for row in triggers_doc["triggers"]}
    conflict_ids = {row["conflict_id"]: dict(row) for row in conflicts_doc["conflicts"]}
    return {
        "sample_dir": str(root),
        "vocab": vocab,
        "vocab_list": vocab_doc["terms"],
        "tickets_doc": tickets_doc,
        "tickets": tickets_doc["tickets"],
        "triggers": triggers,
        "trigger_ids": trigger_ids,
        "triggers_doc": triggers_doc,
        "conflicts": conflicts,
        "conflict_ids": conflict_ids,
        "conflicts_doc": conflicts_doc,
        "channels": channel_weights,
        "channels_doc": channels_doc,
        "config": config,
    }
