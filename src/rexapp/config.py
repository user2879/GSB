"""Rule extraction configuration: structures, loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import ErrorCode

DEFAULT_BAND_LABELS = ("low", "medium", "high")


class ConfigError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class RelationTrigger:
    entry_id: str
    trigger: str
    relation: str
    head_type: str
    tail_type: str
    compatibility: float

    def as_fingerprint_obj(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "trigger": self.trigger,
            "relation": self.relation,
            "head_type": self.head_type,
            "tail_type": self.tail_type,
            "compatibility": self.compatibility,
        }


@dataclass(frozen=True)
class RoleSpec:
    role: str
    entity_type: str
    required: bool
    priority: int

    def as_fingerprint_obj(self) -> dict:
        return {
            "role": self.role,
            "entity_type": self.entity_type,
            "required": self.required,
            "priority": self.priority,
        }


@dataclass(frozen=True)
class EventTrigger:
    entry_id: str
    trigger: str
    event_type: str
    args: tuple[RoleSpec, ...]
    max_args: int

    def as_fingerprint_obj(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "trigger": self.trigger,
            "event_type": self.event_type,
            "args": [role.as_fingerprint_obj() for role in self.args],
            "max_args": self.max_args,
        }


@dataclass
class RuleConfig:
    entity_vocab: dict[str, tuple[str, ...]]
    relation_triggers: tuple[RelationTrigger, ...]
    event_triggers: tuple[EventTrigger, ...]
    relation_band_thresholds: tuple[float, ...]
    event_band_thresholds: tuple[float, ...]
    band_labels: tuple[str, ...]
    fixed_point_scale: int
    interaction_budget_ms: int
    thread_pool_workers: int
    snapshot_commit_interval: int
    relation_window: int
    event_window: int
    max_event_args: int
    scoring_weights: dict[str, float] = field(default_factory=dict)

    def word_type_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for entity_type, words in self.entity_vocab.items():
            for word in words:
                index[word] = entity_type
        return index

    def with_updates(self, **changes: Any) -> "RuleConfig":
        return replace(self, **changes)


def _require(mapping: dict, key: str) -> Any:
    if key not in mapping:
        raise ConfigError(ErrorCode.E_DOC_INVALID, f"missing config key: {key}")
    return mapping[key]


def _is_strictly_increasing(values: list[float]) -> bool:
    return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def load_config(path: str | Path) -> RuleConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(ErrorCode.E_JSON_PARSE, f"config parse error: {exc}") from exc
    return parse_config(raw)


def parse_config(raw: dict) -> RuleConfig:
    if not isinstance(raw, dict):
        raise ConfigError(ErrorCode.E_JSON_PARSE, "config root must be an object")

    vocab_raw = _require(raw, "entity_types")
    if not isinstance(vocab_raw, dict) or not vocab_raw:
        raise ConfigError(ErrorCode.E_DOC_INVALID, "entity_types must be a non-empty object")
    entity_vocab: dict[str, tuple[str, ...]] = {}
    seen_words: dict[str, str] = {}
    for entity_type, words in vocab_raw.items():
        if not isinstance(words, list) or not words:
            raise ConfigError(ErrorCode.E_DOC_INVALID, f"empty vocab for type {entity_type}")
        if len(set(words)) != len(words):
            raise ConfigError(ErrorCode.E_DOC_INVALID, f"duplicate vocab word in type {entity_type}")
        for word in words:
            if not isinstance(word, str) or not word.strip():
                raise ConfigError(ErrorCode.E_DOC_INVALID, f"empty vocab entry in type {entity_type}")
            if word in seen_words:
                raise ConfigError(
                    ErrorCode.E_DOC_INVALID,
                    f"vocab word {word!r} bound to multiple types",
                )
            seen_words[word] = entity_type
        entity_vocab[entity_type] = tuple(words)

    relation_triggers = _parse_relation_triggers(_require(raw, "relation_triggers"), entity_vocab)
    event_triggers = _parse_event_triggers(_require(raw, "event_triggers"), entity_vocab)

    relation_thresholds = list(_require(raw, "relation_band_thresholds"))
    event_thresholds = list(_require(raw, "event_band_thresholds"))
    if not _is_strictly_increasing(event_thresholds):
        raise ConfigError(ErrorCode.E_DOC_INVALID, "event band thresholds must be strictly increasing")
    if not _is_strictly_increasing(relation_thresholds):
        raise ConfigError(ErrorCode.E_DOC_INVALID, "relation band thresholds must be strictly increasing")

    labels = tuple(raw.get("band_labels", DEFAULT_BAND_LABELS))
    if len(labels) != len(event_thresholds) + 1 or len(labels) != len(relation_thresholds) + 1:
        raise ConfigError(ErrorCode.E_DOC_INVALID, "band labels count must be thresholds + 1")

    scale = int(_require(raw, "fixed_point_scale"))
    if scale < 0:
        raise ConfigError(ErrorCode.E_DOC_INVALID, "fixed_point_scale must be >= 0")
    budget = int(_require(raw, "interaction_budget_ms"))
    workers = int(_require(raw, "thread_pool_workers"))
    snapshot_interval = int(_require(raw, "snapshot_commit_interval"))
    relation_window = int(_require(raw, "relation_window"))
    if budget < 0 or workers < 1 or snapshot_interval < 1 or relation_window < 1:
        raise ConfigError(ErrorCode.E_DOC_INVALID, "budget/workers/interval/window have invalid ranges")

    event_window = int(raw.get("event_window", relation_window))
    max_event_args = int(raw.get("max_event_args", 3))
    weights = raw.get(
        "scoring_weights",
        {"trigger_decay": 0.5, "compatibility": 0.3, "distance": 0.2},
    )
    if set(weights) != {"trigger_decay", "compatibility", "distance"}:
        raise ConfigError(ErrorCode.E_DOC_INVALID, "scoring_weights needs trigger_decay/compatibility/distance")

    return RuleConfig(
        entity_vocab=entity_vocab,
        relation_triggers=tuple(relation_triggers),
        event_triggers=tuple(event_triggers),
        relation_band_thresholds=tuple(relation_thresholds),
        event_band_thresholds=tuple(event_thresholds),
        band_labels=labels,
        fixed_point_scale=scale,
        interaction_budget_ms=budget,
        thread_pool_workers=workers,
        snapshot_commit_interval=snapshot_interval,
        relation_window=relation_window,
        event_window=event_window,
        max_event_args=max_event_args,
        scoring_weights=dict(weights),
    )


def _parse_relation_triggers(raw: list, entity_vocab: dict[str, tuple[str, ...]]) -> list[RelationTrigger]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(ErrorCode.E_DOC_INVALID, "relation_triggers must be a non-empty list")
    triggers: list[RelationTrigger] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError(ErrorCode.E_DOC_INVALID, f"relation trigger {index} not an object")
        trigger = item.get("trigger")
        if not isinstance(trigger, str) or not trigger.strip():
            raise ConfigError(ErrorCode.E_DOC_INVALID, f"relation trigger {index} has empty trigger text")
        head_type = item.get("head_type")
        tail_type = item.get("tail_type")
        for entity_type in (head_type, tail_type):
            if entity_type not in entity_vocab:
                raise ConfigError(
                    ErrorCode.E_ENTITY_TYPE_UNKNOWN,
                    f"relation trigger {trigger!r} references unknown entity type {entity_type!r}",
                )
        entry_id = item.get("entry_id", f"rel-{index}")
        triggers.append(
            RelationTrigger(
                entry_id=str(entry_id),
                trigger=trigger,
                relation=str(_require(item, "relation")),
                head_type=head_type,
                tail_type=tail_type,
                compatibility=float(item.get("compatibility", 1.0)),
            )
        )
    return triggers


def _parse_event_triggers(raw: list, entity_vocab: dict[str, tuple[str, ...]]) -> list[EventTrigger]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(ErrorCode.E_DOC_INVALID, "event_triggers must be a non-empty list")
    triggers: list[EventTrigger] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError(ErrorCode.E_DOC_INVALID, f"event trigger {index} not an object")
        trigger = item.get("trigger")
        if not isinstance(trigger, str) or not trigger.strip():
            raise ConfigError(ErrorCode.E_DOC_INVALID, f"event trigger {index} has empty trigger text")
        args_raw = _require(item, "args")
        if not isinstance(args_raw, list) or not args_raw:
            raise ConfigError(ErrorCode.E_DOC_INVALID, f"event trigger {trigger!r} needs a non-empty args list")
        roles: list[RoleSpec] = []
        for arg in args_raw:
            entity_type = arg.get("entity_type")
            if entity_type not in entity_vocab:
                raise ConfigError(
                    ErrorCode.E_ENTITY_TYPE_UNKNOWN,
                    f"event trigger {trigger!r} references unknown entity type {entity_type!r}",
                )
            roles.append(
                RoleSpec(
                    role=str(_require(arg, "role")),
                    entity_type=entity_type,
                    required=bool(arg.get("required", False)),
                    priority=int(arg.get("priority", 0)),
                )
            )
        entry_id = item.get("entry_id", f"evt-{index}")
        triggers.append(
            EventTrigger(
                entry_id=str(entry_id),
                trigger=trigger,
                event_type=str(_require(item, "event_type")),
                args=tuple(roles),
                max_args=int(item.get("max_args", len(roles))),
            )
        )
    return triggers
