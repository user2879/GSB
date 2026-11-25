"""Stable label and entity type encodings."""

from __future__ import annotations


ENTITY_TYPES = ("LOC", "ORG", "PER", "TIME")
TYPE_CODES = {name: index for index, name in enumerate(ENTITY_TYPES)}


def build_labels() -> tuple[str, ...]:
    labels = ["O"]
    for entity_type in ENTITY_TYPES:
        for prefix in "BIES":
            labels.append(f"{prefix}-{entity_type}")
    return tuple(sorted(labels))


LABELS = build_labels()
LABEL_CODES = {label: index for index, label in enumerate(LABELS)}
CODE_LABELS = {index: label for label, index in LABEL_CODES.items()}


def label_code(label: str) -> int:
    return LABEL_CODES[label]


def parse_label(label: str) -> tuple[str, str] | None:
    if label == "O":
        return ("O", "O")
    if "-" not in label:
        return None
    prefix, entity_type = label.split("-", 1)
    if prefix not in "BIES" or entity_type not in TYPE_CODES:
        return None
    return prefix, entity_type
