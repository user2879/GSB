"""Interactive command stream model and deterministic application."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import (
    RelationTrigger,
    RuleConfig,
    _parse_event_triggers,
)
from .documents import Document, parse_document
from .errors import Failure


@dataclass
class CommandEffect:
    command_id: str
    command_type: str
    affected_docs: list[str]
    invalidated_stages: dict[str, list[str]]
    replaced_doc: Document | None = None
    deleted_doc: str | None = None
    config_changed: bool = False
    failures: list[Failure] | None = None


def load_commands(path: str | Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("command stream must be a JSON array")
    return raw


def _replace_relation_trigger(config: RuleConfig, payload: dict) -> RuleConfig:
    updated = RelationTrigger(
        entry_id=str(payload["entry_id"]),
        trigger=str(payload["trigger"]),
        relation=str(payload.get("relation", payload["entry_id"])),
        head_type=str(payload["head_type"]),
        tail_type=str(payload["tail_type"]),
        compatibility=float(payload.get("compatibility", 1.0)),
    )
    entries = [entry for entry in config.relation_triggers if entry.entry_id != updated.entry_id]
    entries.append(updated)
    entries.sort(key=lambda entry: entry.entry_id)
    return config.with_updates(relation_triggers=tuple(entries))


def _replace_event_trigger(config: RuleConfig, payload: dict) -> RuleConfig:
    parsed = _parse_event_triggers([payload], config.entity_vocab)[0]
    entries = [entry for entry in config.event_triggers if entry.entry_id != parsed.entry_id]
    entries.append(parsed)
    entries.sort(key=lambda entry: entry.entry_id)
    return config.with_updates(event_triggers=tuple(entries))


def apply_command(
    command: dict,
    config: RuleConfig,
    documents: dict[str, Document],
) -> tuple[RuleConfig, dict[str, Document], CommandEffect]:
    command_id = str(command.get("command_id", "cmd"))
    command_type = command["type"]
    affected: list[str] = []
    invalidated: dict[str, list[str]] = {}
    replaced_doc: Document | None = None
    deleted_doc: str | None = None
    config_changed = False
    failures: list[Failure] = []
    new_config = config
    new_documents = dict(documents)

    if command_type == "update_relation_trigger":
        entry_id = str(command["entry_id"])
        if not any(entry.entry_id == entry_id for entry in config.relation_triggers):
            raise KeyError(f"unknown relation trigger entry {entry_id}")
        new_config = _replace_relation_trigger(config, command)
        config_changed = True
        affected = sorted(documents)
        for doc_id in affected:
            invalidated[doc_id] = ["relation", "confidence"]

    elif command_type == "update_event_trigger":
        entry_id = str(command["entry_id"])
        if not any(entry.entry_id == entry_id for entry in config.event_triggers):
            raise KeyError(f"unknown event trigger entry {entry_id}")
        new_config = _replace_event_trigger(config, command)
        config_changed = True
        affected = sorted(documents)
        for doc_id in affected:
            invalidated[doc_id] = ["event", "confidence"]

    elif command_type == "replace_document":
        raw_doc = command["document"]
        known_types = set(config.entity_vocab)
        document, failure = parse_document(raw_doc, known_types, 0)
        doc_id = str(raw_doc.get("doc_id"))
        if failure is not None:
            failures.append(failure)
            affected = [doc_id]
        else:
            assert document is not None
            new_documents[document.doc_id] = document
            replaced_doc = document
            affected = [document.doc_id]
            invalidated[document.doc_id] = ["entity", "relation", "event", "confidence"]

    elif command_type == "delete_document":
        doc_id = str(command["doc_id"])
        if doc_id in new_documents:
            del new_documents[doc_id]
            deleted_doc = doc_id
        affected = [doc_id]
        invalidated[doc_id] = ["entity", "relation", "event", "confidence"]

    elif command_type == "update_event_band_thresholds":
        thresholds = [float(value) for value in command["thresholds"]]
        if any(thresholds[i] >= thresholds[i + 1] for i in range(len(thresholds) - 1)):
            raise ValueError("event band thresholds must be strictly increasing")
        new_config = config.with_updates(event_band_thresholds=tuple(thresholds))
        config_changed = True
        affected = sorted(documents)
        for doc_id in affected:
            invalidated[doc_id] = ["confidence"]

    elif command_type == "update_relation_window":
        window = int(command["window"])
        if window < 1:
            raise ValueError("relation window must be >= 1")
        new_config = config.with_updates(relation_window=window)
        config_changed = True
        affected = sorted(documents)
        for doc_id in affected:
            invalidated[doc_id] = ["relation", "confidence"]

    else:
        raise ValueError(f"unknown command type {command_type!r}")

    effect = CommandEffect(
        command_id=command_id,
        command_type=command_type,
        affected_docs=affected,
        invalidated_stages=invalidated,
        replaced_doc=replaced_doc,
        deleted_doc=deleted_doc,
        config_changed=config_changed,
        failures=failures,
    )
    return new_config, new_documents, effect
