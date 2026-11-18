"""Generate the deterministic local synthetic corpus and model package.

This script is a fixture builder, not part of runtime inference. It writes
small JSON/JSONL files with no random state. Runtime extraction never reads the
gold entity or relation annotations.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kg_workbench.entity import (  # noqa: E402
    character_feature_keys,
    character_slots,
    entity_span_feature_keys,
)
from kg_workbench.loader import EntityAnnotation, RelationTruth, Sentence  # noqa: E402
from kg_workbench.relation import relation_feature_keys  # noqa: E402
from kg_workbench.util import fixed_hash_bucket  # noqa: E402


FEATURE_DIM = 509
RELATION_TYPES = ["任职于", "位于", "创办"]
TRIGGER_WORDS = {
    "任职于": ["工作", "加入", "任教"],
    "位于": ["位于", "在"],
    "创办": ["创办"],
}
LEXICON = [
    "张三",
    "李四",
    "王五",
    "赵六",
    "陈七",
    "北京",
    "大学",
    "上海",
    "公司",
    "星河",
    "科技",
    "杭州",
    "南京",
]


RAW_SENTENCES = [
    {
        "id": "S001",
        "words": "张三 在 北京 大学 工作",
        "pos": "NR P NS NN VV",
        "entities": [
            {"start": 0, "end": 0, "type": "PER"},
            {"start": 2, "end": 3, "type": "ORG"},
        ],
        "relations": [{"subject_entity": 0, "object_entity": 1, "type": "任职于"}],
    },
    {
        "id": "S002",
        "words": "李四 加入 上海 公司",
        "pos": "NR VV NS NN",
        "entities": [
            {"start": 0, "end": 0, "type": "PER"},
            {"start": 2, "end": 3, "type": "ORG"},
        ],
        "relations": [{"subject_entity": 0, "object_entity": 1, "type": "任职于"}],
    },
    {
        "id": "S003",
        "words": "王五 创办 星河 科技",
        "pos": "NR VV NN NN",
        "entities": [
            {"start": 0, "end": 0, "type": "PER"},
            {"start": 2, "end": 3, "type": "ORG"},
        ],
        "relations": [{"subject_entity": 0, "object_entity": 1, "type": "创办"}],
    },
    {
        "id": "S004",
        "words": "赵六 位于 杭州",
        "pos": "NR VV NS",
        "entities": [
            {"start": 0, "end": 0, "type": "PER"},
            {"start": 2, "end": 2, "type": "LOC"},
        ],
        "relations": [{"subject_entity": 0, "object_entity": 1, "type": "位于"}],
    },
    {
        "id": "S005",
        "words": "陈七 在 南京 大学 任教",
        "pos": "NR P NS NN VV",
        "entities": [
            {"start": 0, "end": 0, "type": "PER"},
            {"start": 2, "end": 3, "type": "ORG"},
        ],
        "relations": [{"subject_entity": 0, "object_entity": 1, "type": "任职于"}],
    },
    {
        "id": "S006",
        "words": "孙八 加入 杭州 公司",
        "pos": "NR VV NS NN",
        "entities": [
            {"start": 0, "end": 0, "type": "PER"},
            {"start": 2, "end": 3, "type": "ORG"},
        ],
        "relations": [{"subject_entity": 0, "object_entity": 1, "type": "任职于"}],
    },
    {
        "id": "S007",
        "words": "周九 在 苏州",
        "pos": "NR P",
        "entities": [],
        "relations": [],
    },
    {
        "id": "S008",
        "words": "吴十 在 成都 工作",
        "pos": "NR P NS VV",
        "entities": [{"start": 2, "end": 5, "type": "LOC"}],
        "relations": [],
    },
]


EVIDENCE_PAIRS = [
    {
        "sentence_id": "S001",
        "subject_span": [0, 0],
        "object_span": [2, 3],
        "relations": ["任职于", "位于"],
    },
    {
        "sentence_id": "S002",
        "subject_span": [0, 0],
        "object_span": [2, 3],
        "relations": ["任职于", "位于"],
    },
    {
        "sentence_id": "S003",
        "subject_span": [0, 0],
        "object_span": [2, 3],
        "relations": ["创办", "任职于"],
    },
    {
        "sentence_id": "S004",
        "subject_span": [0, 0],
        "object_span": [2, 2],
        "relations": ["位于"],
    },
    {
        "sentence_id": "S005",
        "subject_span": [0, 0],
        "object_span": [2, 3],
        "relations": ["任职于", "位于"],
    },
    {
        "sentence_id": "S006",
        "subject_span": [0, 0],
        "object_span": [2, 3],
        "relations": ["任职于", "位于"],
    },
]


INJECTIONS = {
    "injections": [
        {
            "id": "INJ-001-POS",
            "sentence_id": "S002",
            "stage": "entity",
            "kind": "missing_pos",
        },
        {
            "id": "INJ-002-RANGE",
            "sentence_id": "S004",
            "stage": "load",
            "kind": "annotation_out_of_range",
        },
        {
            "id": "INJ-003-BUCKET",
            "sentence_id": "S003",
            "stage": "relation",
            "kind": "feature_hash_bucket_mismatch",
        },
        {
            "id": "INJ-004-THRESHOLD",
            "sentence_id": "S001",
            "stage": "fusion",
            "kind": "threshold_equal",
        },
    ]
}


CONFIG = {
    "paths": {
        "sentences": "samples/sentences.jsonl",
        "model": "samples/model.json",
        "evidence": "samples/evidence.json",
        "injections": "samples/injections.json",
        "snapshot_dir": "artifacts/snapshot",
        "report": "artifacts/report.json",
    },
    "entity": {"threshold": 0.5},
    "relation": {
        "relation_types": RELATION_TYPES,
        "trigger_words": TRIGGER_WORDS,
    },
    "fixed": {
        "decimals": 4,
        "quantization_decimals": 3,
        "temperature": 0.75,
        "confidence_threshold": 0.62,
        "entity_weight": 0.45,
        "relation_weight": 0.55,
        "bias": -0.2,
        "threshold_equal_policy": "reject",
    },
    "diagnostic": {"tolerance_units": 200},
    "runtime": {"workers": 4, "snapshot_commit_interval": 2},
}


def _sentences() -> list[Sentence]:
    result: list[Sentence] = []
    for raw in RAW_SENTENCES:
        words = tuple(raw["words"].split())
        pos = tuple(raw["pos"].split())
        if len(words) != len(pos):
            continue
        if any(entity["end"] >= len(words) for entity in raw["entities"]):
            continue
        entities = tuple(
            EntityAnnotation(item["start"], item["end"], item["type"])
            for item in raw["entities"]
        )
        relations = tuple(
            RelationTruth(item["subject_entity"], item["object_entity"], item["type"])
            for item in raw["relations"]
        )
        result.append(
            Sentence(raw["id"], words, pos, entities, relations, 0)
        )
    return result


def _gold_bieso(sentence: Sentence) -> tuple[str, ...]:
    slots = character_slots(sentence)
    labels = ["O"] * len(slots)
    for entity in sentence.entities:
        start = next(
            index
            for index, slot in enumerate(slots)
            if slot.word_index == entity.start and slot.char_index == 0
        )
        end = max(
            index
            for index, slot in enumerate(slots)
            if slot.word_index == entity.end
        )
        if start == end:
            labels[start] = "S"
        else:
            labels[start] = "B"
            labels[end] = "E"
            for index in range(start + 1, end):
                labels[index] = "I"
    return tuple(labels)


def _entity_for_span(sentence: Sentence, span: tuple[int, int]) -> dict[str, object]:
    for entity in sentence.entities:
        if (entity.start, entity.end) == span:
            return {"start": span[0], "end": span[1], "type": entity.entity_type}
    raise KeyError(span)


def _trigger_for(
    sentence: Sentence,
    relation_type: str,
    subject: dict[str, object],
    object_: dict[str, object],
) -> str:
    trigger_set = set(TRIGGER_WORDS[relation_type])
    excluded = set(range(int(subject["start"]), int(subject["end"]) + 1)).union(
        range(int(object_["start"]), int(object_["end"]) + 1)
    )
    for index, word in enumerate(sentence.words):
        if index not in excluded and word in trigger_set:
            return word
    return "<NONE>"


def _gold_relation(
    sentence: Sentence, subject_span: tuple[int, int], object_span: tuple[int, int]
) -> str | None:
    for relation in sentence.relations:
        subject = sentence.entities[relation.subject_entity]
        object_ = sentence.entities[relation.object_entity]
        if (subject.start, subject.end) == subject_span and (
            object_.start,
            object_.end,
        ) == object_span:
            return relation.relation_type
    return None


def _add(weights: dict[str, list[Decimal]], label: str, bucket: int, value: str) -> None:
    weights[label][bucket] += Decimal(value)


def build_model(sentences: list[Sentence]) -> dict[str, object]:
    feature_keys: set[str] = set()
    char_records: list[tuple[Sentence, tuple[str, ...], list[tuple[str, ...]]]] = []
    for sentence in sentences:
        slots = character_slots(sentence)
        gold = _gold_bieso(sentence)
        keys = [
            character_feature_keys(slots, index, tuple(LEXICON))
            for index in range(len(slots))
        ]
        for item in keys:
            feature_keys.update(item)
        char_records.append((sentence, gold, keys))

    type_records: list[tuple[tuple[str, ...], str]] = []
    for sentence in sentences:
        for entity in sentence.entities:
            keys = entity_span_feature_keys(
                sentence, entity.start, entity.end, tuple(LEXICON)
            )
            feature_keys.update(keys)
            type_records.append((keys, entity.entity_type))

    relation_records: list[tuple[tuple[str, ...], str, str | None]] = []
    for pair in EVIDENCE_PAIRS:
        sentence = next(
            (item for item in sentences if item.sentence_id == pair["sentence_id"]),
            None,
        )
        if sentence is None:
            continue
        subject_span = tuple(pair["subject_span"])
        object_span = tuple(pair["object_span"])
        subject = _entity_for_span(sentence, subject_span)
        object_ = _entity_for_span(sentence, object_span)
        gold = _gold_relation(sentence, subject_span, object_span)
        between_start = int(subject["end"]) + 1
        between_end = int(object_["start"]) - 1
        for relation_type in pair["relations"]:
            trigger = _trigger_for(sentence, relation_type, subject, object_)
            keys = relation_feature_keys(
                sentence,
                subject,
                object_,
                relation_type,
                trigger,
                between_start,
                between_end,
            )
            feature_keys.update(keys)
            relation_records.append((keys, relation_type, gold))

    buckets = {
        key: fixed_hash_bucket(key, FEATURE_DIM) for key in sorted(feature_keys)
    }
    entity_weights = {
        label: [Decimal(0)] * FEATURE_DIM for label in ("B", "I", "E", "S", "O")
    }
    for _sentence, gold, keys_by_position in char_records:
        for label, keys in zip(gold, keys_by_position):
            for key in keys:
                if key.startswith("char:"):
                    _add(entity_weights, label, buckets[key], "6.0")
                elif key.startswith("char_prev:") or key.startswith("char_next:"):
                    _add(entity_weights, label, buckets[key], "0.10")
                elif key.startswith("pos:"):
                    _add(entity_weights, label, buckets[key], "0.15")
                elif key.startswith("pos_prev:") or key.startswith("pos_next:"):
                    _add(entity_weights, label, buckets[key], "0.05")
                else:
                    _add(entity_weights, label, buckets[key], "0.25")

    type_weights = {
        entity_type: [Decimal(0)] * FEATURE_DIM
        for entity_type in ("PER", "ORG", "LOC", "MISC")
    }
    for keys, entity_type in type_records:
        for key in keys:
            if key.startswith("span_word:"):
                _add(type_weights, entity_type, buckets[key], "3.0")
            elif key.startswith("span_pos:"):
                _add(type_weights, entity_type, buckets[key], "0.20")
            else:
                _add(type_weights, entity_type, buckets[key], "0.15")

    relation_weights = {
        relation_type: [Decimal(0)] * FEATURE_DIM
        for relation_type in RELATION_TYPES
    }
    for keys, relation_type, gold in relation_records:
        for key in keys:
            if key.startswith("candidate:"):
                if relation_type == gold:
                    _add(relation_weights, relation_type, buckets[key], "8.0")
                else:
                    _add(relation_weights, relation_type, buckets[key], "-0.10")
            elif relation_type == gold:
                if key.startswith("trigger:"):
                    _add(relation_weights, relation_type, buckets[key], "2.0")
                else:
                    _add(relation_weights, relation_type, buckets[key], "0.20")

    transitions: dict[str, dict[str, float]] = {}
    valid = {"B": {"I", "E"}, "I": {"I", "E"}, "E": {"B", "S", "O"}, "S": {"B", "S", "O"}, "O": {"B", "S", "O"}}
    for source in ("B", "I", "E", "S", "O"):
        transitions[source] = {
            target: 0.75 if target in valid[source] else -100.0
            for target in ("B", "I", "E", "S", "O")
        }

    def vectors(weights: dict[str, list[Decimal]]) -> dict[str, list[float]]:
        return {key: [float(value) for value in values] for key, values in weights.items()}

    return {
        "format": "kg-model-v1",
        "version": "synthetic-1",
        "feature_dim": FEATURE_DIM,
        "hash_buckets": FEATURE_DIM,
        "labels": ["B", "I", "E", "S", "O"],
        "transitions": transitions,
        "entity_bias": {label: 0.1 if label == "O" else 0.0 for label in ("B", "I", "E", "S", "O")},
        "entity_weights": vectors(entity_weights),
        "entity_types": ["PER", "ORG", "LOC", "MISC"],
        "entity_type_bias": {entity_type: 0.0 for entity_type in ("PER", "ORG", "LOC", "MISC")},
        "entity_type_weights": vectors(type_weights),
        "relation_types": RELATION_TYPES,
        "relation_bias": {relation_type: 0.0 for relation_type in RELATION_TYPES},
        "relation_weights": vectors(relation_weights),
        "feature_hashes": buckets,
        "lexicon": LEXICON,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    (ROOT / "samples").mkdir(parents=True, exist_ok=True)
    sentences_path = ROOT / "samples" / "sentences.jsonl"
    with sentences_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sentence in RAW_SENTENCES:
            handle.write(json.dumps(sentence, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    write_json(
        ROOT / "samples" / "evidence.json",
        {
            "candidate_pairs": EVIDENCE_PAIRS,
            "exclusive_relations": [
                {"relation": "任职于", "excludes": ["位于"]},
                {"relation": "创办", "excludes": ["任职于"]},
            ],
        },
    )
    write_json(ROOT / "samples" / "injections.json", INJECTIONS)
    write_json(ROOT / "config.json", CONFIG)
    write_json(ROOT / "samples" / "model.json", build_model(_sentences()))
    print("generated samples, config.json and samples/model.json")


if __name__ == "__main__":
    main()
