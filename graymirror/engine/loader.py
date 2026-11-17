"""Strong validation loaders for rule JSON and NDJSON request stream.

Any single invalid record rejects the *whole* load with business code
``E_RULE`` / ``E_STREAM``; the raised :class:`GrayError` carries the physical
line number and a JSON pointer style field path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import E_RULE, E_STREAM, GrayError
from .models import Request, Rule

_RULE_FIELDS = {
    "rule_id", "prefix", "stable_target", "gray_target", "gray_weight",
    "mirror_rate", "gray_retries", "mirror_timeout_ms", "mirror_target",
}
_REQUIRED_RULE = _RULE_FIELDS - {"mirror_target"}
_REQUEST_FIELDS = {"request_id", "path", "client_key", "timestamp_ms"}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _line_of_index(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _rule_line(text: str, rule_id: str) -> int:
    marker = json.dumps(rule_id)
    idx = text.find(f'"rule_id": {marker}')
    if idx < 0:
        idx = text.find(f'"rule_id":{marker}')
    if idx < 0:
        idx = text.find(marker)
    return _line_of_index(text, max(idx, 0))


def load_rules(path: str | Path) -> list[Rule]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GrayError(E_RULE, f"cannot read rule file: {exc}", field_path=str(path))
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GrayError(E_RULE, f"invalid JSON: {exc.msg}",
                        field_path="$", line=exc.lineno,
                        details={"column": exc.colno})
    if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
        raise GrayError(E_RULE, "top level must be an object with a 'rules' array",
                        field_path="$.rules")

    seen_ids: set[str] = set()
    rules: list[Rule] = []
    for index, raw in enumerate(payload["rules"]):
        base = f"$.rules[{index}]"
        if not isinstance(raw, dict):
            raise GrayError(E_RULE, "rule entry must be an object", field_path=base)
        unknown = set(raw) - _RULE_FIELDS
        if unknown:
            raise GrayError(E_RULE, f"unknown field(s): {sorted(unknown)}",
                            field_path=f"{base}.{sorted(unknown)[0]}")
        missing = _REQUIRED_RULE - set(raw)
        if missing:
            raise GrayError(E_RULE, f"missing field(s): {sorted(missing)}",
                            field_path=f"{base}.{sorted(missing)[0]}")
        line = _rule_line(text, str(raw.get("rule_id")))
        for name in ("rule_id", "prefix", "stable_target", "gray_target"):
            if not isinstance(raw[name], str) or not raw[name].strip():
                raise GrayError(E_RULE, f"{name} must be a non empty string",
                                field_path=f"{base}.{name}", line=line)
        if not raw["prefix"].startswith("/"):
            raise GrayError(E_RULE, "prefix must start with '/'",
                            field_path=f"{base}.prefix", line=line,
                            details={"prefix": raw["prefix"]})
        for name in ("gray_weight", "mirror_rate"):
            if not _is_int(raw[name]) or not 0 <= raw[name] <= 10000:
                raise GrayError(E_RULE, f"{name} must be an integer in 0..10000",
                                field_path=f"{base}.{name}", line=line,
                                details={"got": raw[name]})
        for name in ("gray_retries", "mirror_timeout_ms"):
            if not _is_int(raw[name]) or raw[name] < 0:
                raise GrayError(E_RULE, f"{name} must be a non negative integer",
                                field_path=f"{base}.{name}", line=line,
                                details={"got": raw[name]})
        if raw["rule_id"] in seen_ids:
            raise GrayError(E_RULE, "duplicate rule_id",
                            field_path=f"{base}.rule_id", line=line,
                            details={"rule_id": raw["rule_id"]})
        seen_ids.add(raw["rule_id"])
        mirror_target = raw.get("mirror_target") or raw["gray_target"]
        if not isinstance(mirror_target, str) or not mirror_target.strip():
            raise GrayError(E_RULE, "mirror_target must be a non empty string",
                            field_path=f"{base}.mirror_target", line=line)
        rules.append(Rule(
            rule_id=raw["rule_id"], prefix=raw["prefix"],
            stable_target=raw["stable_target"], gray_target=raw["gray_target"],
            gray_weight=raw["gray_weight"], mirror_rate=raw["mirror_rate"],
            gray_retries=raw["gray_retries"], mirror_target=mirror_target,
            mirror_timeout_ms=raw["mirror_timeout_ms"], source_line=line,
        ))
    if not rules:
        raise GrayError(E_RULE, "at least one rule is required", field_path="$.rules")
    return rules


def load_stream(path: str | Path, rules: list[Rule]) -> list[Request]:
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GrayError(E_STREAM, f"cannot read stream file: {exc}",
                        field_path=str(path))

    seen_ids: set[str] = set()
    requests: list[Request] = []
    last_ts: int | None = None
    for physical, line in enumerate(lines, start=1):
        if not line.strip():
            raise GrayError(E_STREAM, "blank line is not allowed",
                            field_path="$", line=physical)
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GrayError(E_STREAM, f"invalid NDJSON: {exc.msg}",
                            field_path="$", line=physical,
                            details={"column": exc.colno})
        base = "$"
        if not isinstance(raw, dict):
            raise GrayError(E_STREAM, "stream entry must be an object",
                            field_path=base, line=physical)
        unknown = set(raw) - _REQUEST_FIELDS
        if unknown:
            raise GrayError(E_STREAM, f"unknown field(s): {sorted(unknown)}",
                            field_path=f"{base}.{sorted(unknown)[0]}", line=physical)
        missing = _REQUEST_FIELDS - set(raw)
        if missing:
            raise GrayError(E_STREAM, f"missing field(s): {sorted(missing)}",
                            field_path=f"{base}.{sorted(missing)[0]}", line=physical)
        for name in ("request_id", "path", "client_key"):
            if not isinstance(raw[name], str) or not raw[name].strip():
                raise GrayError(E_STREAM, f"{name} must be a non empty string",
                                field_path=f"{base}.{name}", line=physical)
        if not _is_int(raw["timestamp_ms"]):
            raise GrayError(E_STREAM, "timestamp_ms must be an integer",
                            field_path=f"{base}.timestamp_ms", line=physical,
                            details={"got": raw["timestamp_ms"]})
        if raw["request_id"] in seen_ids:
            raise GrayError(E_STREAM, "duplicate request_id",
                            field_path=f"{base}.request_id", line=physical,
                            details={"request_id": raw["request_id"]})
        seen_ids.add(raw["request_id"])
        if last_ts is not None and raw["timestamp_ms"] < last_ts:
            raise GrayError(E_STREAM, "timestamp_ms must be monotonic non decreasing",
                            field_path=f"{base}.timestamp_ms", line=physical,
                            details={"previous": last_ts, "got": raw["timestamp_ms"]})
        last_ts = raw["timestamp_ms"]
        if match_rule(rules, raw["path"]) is None:
            raise GrayError(E_STREAM, "path matches no rule prefix",
                            field_path=f"{base}.path", line=physical,
                            details={"path": raw["path"]})
        requests.append(Request(
            request_id=raw["request_id"], path=raw["path"],
            client_key=raw["client_key"], timestamp_ms=raw["timestamp_ms"],
            source_line=physical,
        ))
    if not requests:
        raise GrayError(E_STREAM, "request stream is empty", field_path="$")
    return requests


def match_rule(rules: list[Rule], path: str) -> Rule | None:
    """Longest prefix wins; equal length prefixes break by rule_id deterministically."""
    candidates = [rule for rule in rules if path.startswith(rule.prefix)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda rule: (-len(rule.prefix), rule.rule_id))[0]
