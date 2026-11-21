from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


ENTITY_TYPES: tuple[str, ...] = tuple(sorted(("人物", "地点", "机构", "时间")))
TYPE_ALIASES: dict[str, str] = {
    "PER": "人物",
    "PERSON": "人物",
    "LOC": "地点",
    "LOCATION": "地点",
    "ORG": "机构",
    "ORGANIZATION": "机构",
    "TIME": "时间",
    "DATE": "时间",
}
TAG_SYMBOLS: tuple[str, ...] = ("B", "I", "E", "S")


def canonical_type(value: str) -> str:
    text = str(value).strip()
    if text in ENTITY_TYPES:
        return text
    alias = TYPE_ALIASES.get(text.upper())
    if alias:
        return alias
    raise ValueError(f"unknown entity type: {value!r}")


def parse_tag(tag: str) -> tuple[str, str | None]:
    text = str(tag).strip()
    if text == "O":
        return "O", None
    if "-" not in text:
        raise ValueError(f"unknown BIESO tag: {tag!r}")
    symbol, raw_type = text.split("-", 1)
    symbol = symbol.upper()
    if symbol not in TAG_SYMBOLS:
        raise ValueError(f"unknown BIESO tag symbol: {tag!r}")
    return symbol, canonical_type(raw_type)


def canonical_tag(tag: str) -> str:
    symbol, entity_type = parse_tag(tag)
    return "O" if symbol == "O" else f"{symbol}-{entity_type}"


def required_tags() -> tuple[str, ...]:
    labels = ["O"]
    for entity_type in ENTITY_TYPES:
        for symbol in TAG_SYMBOLS:
            labels.append(f"{symbol}-{entity_type}")
    return tuple(labels)


def stable_type_order(types: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(types)))


@dataclass(frozen=True)
class CorpusSample:
    sample_id: str
    chars: tuple[str, ...]
    gold_labels: tuple[str, ...]
    entity_types: tuple[str, ...]
    gold_entities: tuple[dict[str, object], ...]
    source_line: int

    @property
    def text(self) -> str:
        return "".join(self.chars)
