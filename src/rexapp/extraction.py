"""Deterministic NER, relation candidate and event argument extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EventTrigger, RelationTrigger, RoleSpec, RuleConfig


@dataclass(frozen=True)
class Entity:
    entity_id: str
    text: str
    entity_type: str
    start: int
    end: int

    def as_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "text": self.text,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class ExcludedPair:
    head_id: str
    tail_id: str
    reason: str
    trigger: str
    trigger_pos: int

    def as_dict(self) -> dict:
        return {
            "head_id": self.head_id,
            "tail_id": self.tail_id,
            "reason": self.reason,
            "trigger": self.trigger,
            "trigger_pos": self.trigger_pos,
        }


@dataclass
class RelationCandidate:
    triple_id: str
    head: Entity
    relation: str
    tail: Entity
    trigger: str
    trigger_pos: int
    entry_id: str
    raw_score: float = 0.0
    quantized_score: int = 0
    band: str = ""
    boundary: bool = False
    drift: float = 0.0

    def as_dict(self) -> dict:
        return {
            "triple_id": self.triple_id,
            "head": self.head.as_dict(),
            "relation": self.relation,
            "tail": self.tail.as_dict(),
            "trigger": self.trigger,
            "trigger_pos": self.trigger_pos,
            "entry_id": self.entry_id,
            "raw_score": self.raw_score,
            "quantized_score": self.quantized_score,
            "band": self.band,
            "boundary": self.boundary,
            "drift": self.drift,
        }


@dataclass(frozen=True)
class SuppressedAssignment:
    entity_id: str
    role: str
    winner_role: str

    def as_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "role": self.role,
            "winner_role": self.winner_role,
        }


@dataclass
class EventArgument:
    role: str
    entity: Entity
    raw_score: float = 0.0
    quantized_score: int = 0
    band: str = ""
    boundary: bool = False
    drift: float = 0.0

    def as_dict(self) -> dict:
        payload = {
            "role": self.role,
            "entity": self.entity.as_dict(),
            "raw_score": self.raw_score,
            "quantized_score": self.quantized_score,
            "band": self.band,
            "boundary": self.boundary,
            "drift": self.drift,
        }
        return payload


@dataclass
class EventMention:
    event_id: str
    event_type: str
    trigger: str
    trigger_pos: int
    entry_id: str
    args: list[EventArgument] = field(default_factory=list)
    suppressed: list[SuppressedAssignment] = field(default_factory=list)
    missing_roles: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "trigger": self.trigger,
            "trigger_pos": self.trigger_pos,
            "entry_id": self.entry_id,
            "args": [arg.as_dict() for arg in self.args],
            "suppressed": [item.as_dict() for item in self.suppressed],
            "missing_roles": list(self.missing_roles),
        }


def recognize_entities(words: tuple[str, ...], config: RuleConfig) -> list[Entity]:
    """Max-match NER with longest-first, then start-position tie breaking."""
    word_type_index = config.word_type_index()
    max_length = max((len(entries) for entries in config.entity_vocab.values()), default=1)
    candidates: list[Entity] = []
    counter = 0
    occupied: set[int] = set()
    for span_len in range(min(max_length, len(words)), 0, -1):
        for start in range(0, len(words) - span_len + 1):
            end = start + span_len
            if any(index in occupied for index in range(start, end)):
                continue
            phrase = " ".join(words[start:end])
            entity_type = word_type_index.get(phrase)
            if entity_type is None:
                continue
            entity = Entity(
                entity_id=f"e{counter:04d}",
                text=phrase,
                entity_type=entity_type,
                start=start,
                end=end,
            )
            candidates.append(entity)
            occupied.update(range(start, end))
            counter += 1
    candidates.sort(key=lambda entity: (entity.start, -(entity.end - entity.start), entity.entity_id))
    return candidates


def _relation_entries_by_trigger(config: RuleConfig) -> dict[str, list[RelationTrigger]]:
    index: dict[str, list[RelationTrigger]] = {}
    for entry in config.relation_triggers:
        index.setdefault(entry.trigger, []).append(entry)
    for entries in index.values():
        entries.sort(key=lambda entry: entry.entry_id)
    return index


def _trigger_decay(distance: int) -> float:
    return float(1.0 / (1.0 + np.float64(distance)))


def _distance_component(distance: int) -> float:
    return float(1.0 / (1.0 + np.float64(distance) * np.float64(0.1)))


def _relation_raw_score(
    trigger_distance: int,
    pair_distance: int,
    compatibility: float,
    weights: dict[str, float],
) -> float:
    decay = _trigger_decay(trigger_distance)
    distance = _distance_component(pair_distance)
    score = (
        np.float64(weights["trigger_decay"]) * np.float64(decay)
        + np.float64(weights["compatibility"]) * np.float64(compatibility)
        + np.float64(weights["distance"]) * np.float64(distance)
    )
    return float(score)


def generate_relation_candidates(
    doc_id: str,
    words: tuple[str, ...],
    entities: list[Entity],
    config: RuleConfig,
) -> tuple[list[RelationCandidate], list[ExcludedPair]]:
    entries_by_trigger = _relation_entries_by_trigger(config)
    candidates: list[RelationCandidate] = []
    excluded: list[ExcludedPair] = []
    window = config.relation_window
    counter = 0
    for trigger_pos, word in enumerate(words):
        entries = entries_by_trigger.get(word)
        if not entries:
            continue
        for head_index, head in enumerate(entities):
            for tail_index, tail in enumerate(entities):
                if head_index == tail_index:
                    continue
                for entry in entries:
                    if head.entity_type != entry.head_type or tail.entity_type != entry.tail_type:
                        continue
                    head_distance = abs(trigger_pos - _entity_anchor(head, trigger_pos))
                    tail_distance = abs(trigger_pos - _entity_anchor(tail, trigger_pos))
                    max_entity_distance = max(head_distance, tail_distance)
                    if max_entity_distance > window:
                        excluded.append(
                            ExcludedPair(
                                head_id=head.entity_id,
                                tail_id=tail.entity_id,
                                reason="outside_window",
                                trigger=entry.trigger,
                                trigger_pos=trigger_pos,
                            )
                        )
                        continue
                    pair_distance = abs(head.start - tail.start)
                    raw_score = _relation_raw_score(
                        max_entity_distance,
                        pair_distance,
                        entry.compatibility,
                        config.scoring_weights,
                    )
                    candidates.append(
                        RelationCandidate(
                            triple_id=f"{doc_id}:r{counter:04d}",
                            head=head,
                            relation=entry.relation,
                            tail=tail,
                            trigger=entry.trigger,
                            trigger_pos=trigger_pos,
                            entry_id=entry.entry_id,
                            raw_score=raw_score,
                        )
                    )
                    counter += 1
    candidates.sort(
        key=lambda item: (
            item.trigger_pos,
            item.head.start,
            item.tail.start,
            item.entry_id,
            item.triple_id,
        )
    )
    for index, candidate in enumerate(candidates):
        object.__setattr__(candidate, "triple_id", f"{doc_id}:r{index:04d}")
    excluded.sort(key=lambda item: (item.trigger_pos, item.head_id, item.tail_id, item.trigger))
    return candidates, excluded


def _entity_anchor(entity: Entity, trigger_pos: int) -> int:
    if trigger_pos < entity.start:
        return entity.start
    if trigger_pos >= entity.end:
        return entity.end - 1
    return trigger_pos


def _event_entries_by_trigger(config: RuleConfig) -> dict[str, list[EventTrigger]]:
    index: dict[str, list[EventTrigger]] = {}
    for entry in config.event_triggers:
        index.setdefault(entry.trigger, []).append(entry)
    for entries in index.values():
        entries.sort(key=lambda entry: entry.entry_id)
    return index


def _event_raw_score(
    trigger_distance: int,
    compatibility: float,
    weights: dict[str, float],
) -> float:
    decay = _trigger_decay(trigger_distance)
    distance = _distance_component(trigger_distance)
    score = (
        np.float64(weights["trigger_decay"]) * np.float64(decay)
        + np.float64(weights["compatibility"]) * np.float64(compatibility)
        + np.float64(weights["distance"]) * np.float64(distance)
    )
    return float(score)


def extract_events(
    doc_id: str,
    words: tuple[str, ...],
    entities: list[Entity],
    config: RuleConfig,
) -> list[EventMention]:
    entries_by_trigger = _event_entries_by_trigger(config)
    events: list[EventMention] = []
    window = config.event_window
    event_counter = 0
    for trigger_pos, word in enumerate(words):
        entries = entries_by_trigger.get(word)
        if not entries:
            continue
        for entry in entries:
            mention = EventMention(
                event_id=f"{doc_id}:e{event_counter:04d}",
                event_type=entry.event_type,
                trigger=entry.trigger,
                trigger_pos=trigger_pos,
                entry_id=entry.entry_id,
            )
            assigned_entities: dict[str, EventArgument] = {}
            suppressed: list[SuppressedAssignment] = []
            roles_in_order = sorted(entry.args, key=lambda spec: (-spec.priority, spec.role))
            arg_counter = 0
            for role_spec in roles_in_order:
                if arg_counter >= min(entry.max_args, config.max_event_args):
                    break
                best_entity = _best_entity_for_role(entities, role_spec, trigger_pos, window, assigned_entities)
                if best_entity is None:
                    continue
                anchor = _entity_anchor(best_entity, trigger_pos)
                raw_score = _event_raw_score(
                    abs(trigger_pos - anchor),
                    1.0 if best_entity.entity_type == role_spec.entity_type else 0.5,
                    config.scoring_weights,
                )
                if best_entity.entity_id in assigned_entities:
                    winner = assigned_entities[best_entity.entity_id]
                    suppressed.append(
                        SuppressedAssignment(
                            entity_id=best_entity.entity_id,
                            role=role_spec.role,
                            winner_role=winner.role,
                        )
                    )
                    continue
                assigned_entities[best_entity.entity_id] = EventArgument(
                    role=role_spec.role,
                    entity=best_entity,
                    raw_score=raw_score,
                )
                arg_counter += 1
            mention.args = list(assigned_entities.values())
            mention.args.sort(key=lambda arg: (_role_priority(entry, arg.role), arg.role))
            present_roles = {arg.role for arg in mention.args}
            mention.missing_roles = sorted(
                spec.role for spec in entry.args if spec.required and spec.role not in present_roles
            )
            suppressed.sort(key=lambda item: (item.entity_id, item.role))
            mention.suppressed = suppressed
            events.append(mention)
            event_counter += 1
    events.sort(key=lambda mention: (mention.trigger_pos, mention.event_type, mention.event_id))
    for index, mention in enumerate(events):
        object.__setattr__(mention, "event_id", f"{doc_id}:e{index:04d}")
    return events


def _role_priority(entry: EventTrigger, role: str) -> int:
    for spec in entry.args:
        if spec.role == role:
            return -spec.priority
    return 0


def _best_entity_for_role(
    entities: list[Entity],
    role_spec: RoleSpec,
    trigger_pos: int,
    window: int,
    assigned: dict[str, EventArgument],
) -> Entity | None:
    eligible: list[tuple[int, int, Entity]] = []
    for entity in entities:
        if entity.entity_type != role_spec.entity_type:
            continue
        anchor = _entity_anchor(entity, trigger_pos)
        distance = abs(trigger_pos - anchor)
        if distance > window:
            continue
        eligible.append((distance, entity.start, entity))
    if not eligible:
        return None
    eligible.sort(key=lambda item: (item[0], item[1], item[2].entity_id))
    return eligible[0][2]
