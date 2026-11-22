"""Strict local declaration loading for topics, groups, policies, and paths."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any

from .canonical import canonical_json, read_json, sha256_obj
from .errors import LabError


FORMAT_VERSIONS = ("v1", "v2", "v3")
VERDICTS = ("at_least_once", "at_most_once", "exactly_once")
DEFAULT_POLICY = {
    "exactly_once_requires_ack": True,
    "exactly_once_requires_idempotency": True,
    "exactly_once_requires_complete_range": True,
    "allow_ack_with_uncovered_range": False,
    "max_allowed_retries": 3,
    "semantic_marker_authoritative": False,
}
DEFAULT_PATHS = {
    "ledger": "state/ledger.sqlite3",
    "snapshots": "snapshots",
    "backup": "backup",
    "out": "out",
    "logs": "logs/structured.jsonl",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _config_error(field: str, reason: str, **details: Any) -> LabError:
    payload = {"field": field, "reason": reason}
    payload.update(details)
    return LabError("E_CONFIG", f"declaration field {field}: {reason}", payload)


@dataclass(frozen=True)
class SessionConfig:
    scale_power: int
    truncation: str

    @property
    def scale_factor(self) -> int:
        return 10 ** self.scale_power


@dataclass(frozen=True)
class TopicConfig:
    name: str
    start: int
    end: int
    step: int
    id_template: str
    delivery_time_ms: int
    policy: dict[str, Any]

    def message_id(self, seq: int) -> str:
        return self.id_template.format(topic=self.name, seq=seq)

    @property
    def sequences(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end + 1, self.step))

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(self.message_id(seq) for seq in self.sequences)

    def sequence_for_id(self, message_id: str) -> int | None:
        for seq in self.sequences:
            if self.message_id(seq) == message_id:
                return seq
        return None


@dataclass(frozen=True)
class GroupConfig:
    name: str
    format_version: str


@dataclass
class Declarations:
    raw: dict[str, Any]
    session: SessionConfig
    topics: dict[str, TopicConfig]
    groups: dict[str, GroupConfig]
    paths: dict[str, Path]
    source_path: str | None = None

    @property
    def topic_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.topics))

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.groups))

    @property
    def format_versions(self) -> tuple[str, ...]:
        return tuple(sorted({group.format_version for group in self.groups.values()}))

    def parameter_document(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "session": {
                "scale_power": self.session.scale_power,
                "truncation": self.session.truncation,
            },
            "topics": [
                {
                    "name": topic.name,
                    "message_seq_rule": {
                        "start": topic.start,
                        "end": topic.end,
                        "step": topic.step,
                        "id_template": topic.id_template,
                    },
                    "delivery_time_ms": topic.delivery_time_ms,
                    "confirmation_policy": topic.policy,
                }
                for topic in sorted(self.topics.values(), key=lambda item: item.name)
            ],
            "consumer_groups": [
                {"name": group.name, "format_version": group.format_version}
                for group in sorted(self.groups.values(), key=lambda item: item.name)
            ],
        }

    def parameter_fingerprint(self) -> str:
        return sha256_obj(self.parameter_document())

    def declaration_document(self) -> dict[str, Any]:
        return self.parameter_document()


def _require_dict(raw: dict[str, Any], field: str) -> dict[str, Any]:
    value = raw.get(field)
    if not isinstance(value, dict):
        raise _config_error(field, "must be an object")
    return value


def _require_int(raw: dict[str, Any], field: str, *, minimum: int | None = None) -> int:
    value = raw.get(field)
    if not _is_int(value):
        raise _config_error(field, "must be an integer")
    if minimum is not None and value < minimum:
        raise _config_error(field, f"must be >= {minimum}", value=value)
    return value


def _require_bool(raw: dict[str, Any], field: str) -> bool:
    value = raw.get(field)
    if not isinstance(value, bool):
        raise _config_error(field, "must be a boolean")
    return value


def load_declarations(path: str | Path) -> Declarations:
    source = Path(path)
    raw = read_json(source)
    return declarations_from_document(raw, source_path=str(source))


def declarations_from_document(raw_value: dict[str, Any], *, source_path: str | None = None) -> Declarations:
    if not isinstance(raw_value, dict):
        raise _config_error("$", "declaration root must be an object")
    raw = deepcopy(raw_value)
    raw.setdefault("schema_version", 1)
    raw.setdefault("paths", {})

    if raw.get("schema_version") != 1:
        raise _config_error("schema_version", "must equal 1", value=raw.get("schema_version"))

    session_raw = _require_dict(raw, "session")
    scale_power = _require_int(session_raw, "scale_power", minimum=0)
    if scale_power > 12:
        raise _config_error("session.scale_power", "must be <= 12", value=scale_power)
    truncation = session_raw.get("truncation", "floor_toward_negative_infinity")
    if truncation != "floor_toward_negative_infinity":
        raise _config_error(
            "session.truncation",
            "only floor_toward_negative_infinity is supported",
            value=truncation,
        )
    session = SessionConfig(scale_power=scale_power, truncation=truncation)

    topics_raw = raw.get("topics")
    if not isinstance(topics_raw, list) or not topics_raw:
        raise _config_error("topics", "must be a non-empty array")
    topics: dict[str, TopicConfig] = {}
    for index, item in enumerate(topics_raw):
        prefix = f"topics[{index}]"
        if not isinstance(item, dict):
            raise _config_error(prefix, "must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise _config_error(f"{prefix}.name", "must be a non-empty string")
        if name in topics:
            raise _config_error(f"{prefix}.name", "duplicate topic", value=name)
        rule = _require_dict(item, "message_seq_rule")
        start = _require_int(rule, "start", minimum=0)
        end = _require_int(rule, "end", minimum=0)
        step = _require_int(rule, "step", minimum=1)
        if end < start:
            raise _config_error(f"{prefix}.message_seq_rule.end", "must be >= start")
        template = rule.get("id_template")
        if not isinstance(template, str) or "{seq}" not in template or "{topic}" not in template:
            raise _config_error(
                f"{prefix}.message_seq_rule.id_template",
                "must contain {topic} and {seq}",
            )
        try:
            sample = template.format(topic=name, seq=start)
        except Exception as exc:
            raise _config_error(
                f"{prefix}.message_seq_rule.id_template",
                "invalid format template",
                reason=str(exc),
            )
        if not sample:
            raise _config_error(f"{prefix}.message_seq_rule.id_template", "must not produce empty id")
        delivery_time_ms = _require_int(item, "delivery_time_ms", minimum=0)
        policy_raw = _require_dict(item, "confirmation_policy")
        policy = dict(DEFAULT_POLICY)
        for key in DEFAULT_POLICY:
            if key in policy_raw:
                if isinstance(DEFAULT_POLICY[key], bool):
                    policy[key] = _require_bool(policy_raw, key)
                else:
                    policy[key] = _require_int(policy_raw, key, minimum=0)
        topics[name] = TopicConfig(
            name=name,
            start=start,
            end=end,
            step=step,
            id_template=template,
            delivery_time_ms=delivery_time_ms,
            policy=policy,
        )

    groups_raw = raw.get("consumer_groups")
    if not isinstance(groups_raw, list) or not groups_raw:
        raise _config_error("consumer_groups", "must be a non-empty array")
    groups: dict[str, GroupConfig] = {}
    for index, item in enumerate(groups_raw):
        prefix = f"consumer_groups[{index}]"
        if not isinstance(item, dict):
            raise _config_error(prefix, "must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise _config_error(f"{prefix}.name", "must be a non-empty string")
        if name in groups:
            raise _config_error(f"{prefix}.name", "duplicate consumer group", value=name)
        version = item.get("format_version")
        if version not in FORMAT_VERSIONS:
            raise _config_error(
                f"{prefix}.format_version",
                f"must be one of {list(FORMAT_VERSIONS)}",
                value=version,
            )
        groups[name] = GroupConfig(name=name, format_version=version)

    paths_raw = raw.get("paths", {})
    if not isinstance(paths_raw, dict):
        raise _config_error("paths", "must be an object")
    paths: dict[str, Path] = {}
    for key, default in DEFAULT_PATHS.items():
        value = paths_raw.get(key, default)
        if not isinstance(value, str) or not value:
            raise _config_error(f"paths.{key}", "must be a non-empty string")
        paths[key] = Path(value)

    return Declarations(
        raw=raw,
        session=session,
        topics=topics,
        groups=groups,
        paths=paths,
        source_path=source_path,
    )


def declarations_fingerprint(declarations: Declarations) -> str:
    return sha256_obj(declarations.parameter_document())


def declarations_json(declarations: Declarations) -> str:
    return canonical_json(declarations.parameter_document())
