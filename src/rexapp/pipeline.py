"""Per-document staged pipeline and fixed-point confidence classification."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import RuleConfig
from .documents import Document
from .extraction import (
    Entity,
    EventMention,
    ExcludedPair,
    RelationCandidate,
    extract_events,
    generate_relation_candidates,
    recognize_entities,
)
from .fingerprint import fingerprint, fingerprint_text, stage_key
from .fixedpoint import classify_band, drift, is_boundary, quantize_score, quantize_threshold


@dataclass
class BoundaryRecord:
    doc_id: str
    kind: str
    identifier: str
    quantized_score: int
    raw_score: float
    drift: float
    band: str

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "kind": self.kind,
            "identifier": self.identifier,
            "quantized_score": self.quantized_score,
            "raw_score": self.raw_score,
            "drift": self.drift,
            "band": self.band,
        }


@dataclass
class DocumentResult:
    doc_id: str
    entities: list[Entity]
    relations: list[RelationCandidate]
    excluded_pairs: list[ExcludedPair]
    events: list[EventMention]
    boundaries: list[BoundaryRecord] = field(default_factory=list)
    stage_keys: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "entities": [entity.as_dict() for entity in self.entities],
            "relations": [relation.as_dict() for relation in self.relations],
            "excluded_pairs": [pair.as_dict() for pair in self.excluded_pairs],
            "events": [event.as_dict() for event in self.events],
            "boundaries": [item.as_dict() for item in self.boundaries],
            "stage_keys": dict(self.stage_keys),
        }


STAGES = ("entity", "relation", "event", "confidence")


def config_fingerprints(config: RuleConfig) -> dict[str, str]:
    return {
        "entity_vocab": fingerprint({k: list(v) for k, v in config.entity_vocab.items()}),
        "relation_entries": fingerprint([entry.as_fingerprint_obj() for entry in config.relation_triggers]),
        "event_entries": fingerprint([entry.as_fingerprint_obj() for entry in config.event_triggers]),
        "relation_thresholds": fingerprint(list(config.relation_band_thresholds)),
        "event_thresholds": fingerprint(list(config.event_band_thresholds)),
        "relation_window": fingerprint(config.relation_window),
        "event_window": fingerprint(config.event_window),
        "scale": fingerprint(config.fixed_point_scale),
        "weights": fingerprint(config.scoring_weights),
        "max_event_args": fingerprint(config.max_event_args),
    }


def compute_stage_keys(document: Document, config: RuleConfig) -> dict[str, str]:
    fps = config_fingerprints(config)
    words_fp = fingerprint_text(" ".join(document.words))
    entity_key = stage_key("entity", words_fp, fps["entity_vocab"])
    relation_key = stage_key(
        "relation",
        entity_key,
        fps["relation_entries"],
        fps["relation_window"],
        fps["weights"],
    )
    event_key = stage_key(
        "event",
        entity_key,
        fps["event_entries"],
        fps["event_window"],
        fps["weights"],
        fps["max_event_args"],
    )
    confidence_key = stage_key(
        "confidence",
        relation_key,
        event_key,
        fps["relation_thresholds"],
        fps["event_thresholds"],
        fps["scale"],
    )
    return {
        "entity": entity_key,
        "relation": relation_key,
        "event": event_key,
        "confidence": confidence_key,
    }


def run_entity_stage(document: Document, config: RuleConfig) -> list[Entity]:
    return recognize_entities(document.words, config)


def run_relation_stage(
    document: Document,
    config: RuleConfig,
    entities: list[Entity],
) -> tuple[list[RelationCandidate], list[ExcludedPair]]:
    return generate_relation_candidates(document.doc_id, document.words, entities, config)


def run_event_stage(
    document: Document,
    config: RuleConfig,
    entities: list[Entity],
) -> list[EventMention]:
    return extract_events(document.doc_id, document.words, entities, config)


def run_confidence_stage(
    document: Document,
    config: RuleConfig,
    relations: list[RelationCandidate],
    events: list[EventMention],
) -> list[BoundaryRecord]:
    scale = config.fixed_point_scale
    rel_thresholds = [quantize_threshold(value, scale) for value in config.relation_band_thresholds]
    evt_thresholds = [quantize_threshold(value, scale) for value in config.event_band_thresholds]
    boundaries: list[BoundaryRecord] = []

    for relation in relations:
        quantized = quantize_score(relation.raw_score, scale)
        relation.quantized_score = quantized
        relation.drift = drift(relation.raw_score, quantized, scale)
        relation.band = classify_band(quantized, rel_thresholds, list(config.band_labels))
        relation.boundary = is_boundary(quantized, rel_thresholds)
        if relation.boundary:
            boundaries.append(
                BoundaryRecord(
                    doc_id=document.doc_id,
                    kind="relation",
                    identifier=relation.triple_id,
                    quantized_score=quantized,
                    raw_score=relation.raw_score,
                    drift=relation.drift,
                    band=relation.band,
                )
            )

    for event in events:
        for arg in event.args:
            quantized = quantize_score(arg.raw_score, scale)
            arg.quantized_score = quantized
            arg.drift = drift(arg.raw_score, quantized, scale)
            arg.band = classify_band(quantized, evt_thresholds, list(config.band_labels))
            arg.boundary = is_boundary(quantized, evt_thresholds)
            if arg.boundary:
                boundaries.append(
                    BoundaryRecord(
                        doc_id=document.doc_id,
                        kind="event_arg",
                        identifier=f"{event.event_id}:{arg.role}",
                        quantized_score=quantized,
                        raw_score=arg.raw_score,
                        drift=arg.drift,
                        band=arg.band,
                    )
                )
    boundaries.sort(key=lambda item: (item.doc_id, item.kind, item.identifier))
    return boundaries


def run_full_pipeline(document: Document, config: RuleConfig) -> DocumentResult:
    keys = compute_stage_keys(document, config)
    entities = run_entity_stage(document, config)
    relations, excluded = run_relation_stage(document, config, entities)
    events = run_event_stage(document, config, entities)
    boundaries = run_confidence_stage(document, config, relations, events)
    return DocumentResult(
        doc_id=document.doc_id,
        entities=entities,
        relations=relations,
        excluded_pairs=excluded,
        events=events,
        boundaries=boundaries,
        stage_keys=keys,
    )
