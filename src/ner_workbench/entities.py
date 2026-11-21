from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .types import parse_tag, stable_type_order


@dataclass(frozen=True)
class RepairRecord:
    code: str
    positions: tuple[int, ...]
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {"code": self.code, "positions": list(self.positions), "reason": self.reason}


@dataclass(frozen=True)
class EntityFragment:
    sample_id: str
    start: int
    end: int
    entity_type: str
    candidate_types: tuple[str, ...]
    text: str

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "start": self.start,
            "end": self.end,
            "type": self.entity_type,
            "candidate_types": list(self.candidate_types),
            "text": self.text,
        }


@dataclass(frozen=True)
class MergeResult:
    entities: tuple[EntityFragment, ...]
    repairs: tuple[RepairRecord, ...]


def merge_entities(sample_id: str, chars: Sequence[str], tags: Sequence[str]) -> MergeResult:
    if len(chars) != len(tags):
        raise ValueError(f"chars length {len(chars)} != tags length {len(tags)}")

    entities: list[EntityFragment] = []
    repairs: list[RepairRecord] = []
    open_start: int | None = None
    open_types: set[str] = set()

    def close_segment(end: int, positions: tuple[int, ...], reason: str, code: str | None = None) -> None:
        nonlocal open_start, open_types
        if open_start is None:
            return
        start = open_start
        candidate_types = stable_type_order(open_types)
        selected = candidate_types[0]
        if len(candidate_types) > 1:
            repairs.append(
                RepairRecord(
                    code="TYPE_TIE_LEXICOGRAPHIC",
                    positions=positions,
                    reason=(
                        f"segment [{start},{end}] has candidate types {list(candidate_types)}; "
                        f"selected {selected!r} by dictionary order"
                    ),
                )
            )
        if code:
            repairs.append(RepairRecord(code=code, positions=positions, reason=reason))
        entities.append(
            EntityFragment(
                sample_id=sample_id,
                start=start,
                end=end,
                entity_type=selected,
                candidate_types=candidate_types,
                text="".join(chars[start : end + 1]),
            )
        )
        open_start = None
        open_types = set()

    for index, tag in enumerate(tags):
        symbol, entity_type = parse_tag(tag)
        if symbol == "O":
            if open_start is not None:
                close_segment(
                    index - 1,
                    (open_start, index - 1, index),
                    "open B/I segment interrupted by O; implicit E inserted at previous position",
                    code="B_INTERRUPTED_BY_O",
                )
            continue

        assert entity_type is not None
        if symbol == "S":
            if open_start is not None:
                close_segment(
                    index - 1,
                    (open_start, index - 1, index),
                    "open segment interrupted by S; previous segment closed before single entity",
                    code="B_INTERRUPTED_BY_S",
                )
            entities.append(
                EntityFragment(
                    sample_id=sample_id,
                    start=index,
                    end=index,
                    entity_type=entity_type,
                    candidate_types=(entity_type,),
                    text=chars[index],
                )
            )
            continue

        if symbol == "B":
            if open_start is not None:
                close_segment(
                    index - 1,
                    (open_start, index - 1, index),
                    "new B starts before previous segment closed; implicit E inserted",
                    code="B_INTERRUPTED_BY_B",
                )
            open_start = index
            open_types = {entity_type}
            continue

        if symbol == "I":
            if open_start is None:
                open_start = index
                open_types = {entity_type}
                repairs.append(
                    RepairRecord(
                        code="I_WITHOUT_B_PROMOTED_B",
                        positions=(index,),
                        reason="I encountered without an open B; promoted to B",
                    )
                )
            else:
                if entity_type not in open_types:
                    repairs.append(
                        RepairRecord(
                            code="TYPE_JUMP_ON_I",
                            positions=(open_start, index),
                            reason=(
                                f"I-{entity_type} continues a segment opened with "
                                f"{stable_type_order(open_types)}; type conflict retained for tie resolution"
                            ),
                        )
                    )
                open_types.add(entity_type)
            continue

        if symbol == "E":
            if open_start is None:
                entities.append(
                    EntityFragment(
                        sample_id=sample_id,
                        start=index,
                        end=index,
                        entity_type=entity_type,
                        candidate_types=(entity_type,),
                        text=chars[index],
                    )
                )
                repairs.append(
                    RepairRecord(
                        code="E_WITHOUT_B_PROMOTED_S",
                        positions=(index,),
                        reason="E encountered without an open B; promoted to single-character entity",
                    )
                )
            else:
                if entity_type not in open_types:
                    repairs.append(
                        RepairRecord(
                            code="TYPE_JUMP_ON_E",
                            positions=(open_start, index),
                            reason=(
                                f"E-{entity_type} closes a segment opened with "
                                f"{stable_type_order(open_types)}; type conflict retained for tie resolution"
                            ),
                        )
                    )
                open_types.add(entity_type)
                close_segment(
                    index,
                    (open_start, index),
                    "valid E closes the open segment",
                    code=None,
                )
            continue

    if open_start is not None:
        close_segment(
            len(chars) - 1,
            (open_start, len(chars) - 1),
            "unclosed B/I segment at end of sentence; implicit E inserted at last position",
            code="UNCLOSED_B_CLOSED_AT_END",
        )

    entities.sort(key=lambda entity: (entity.start, entity.end, entity.entity_type))
    repairs.sort(key=lambda repair: (min(repair.positions) if repair.positions else -1, repair.code))
    return MergeResult(entities=tuple(entities), repairs=tuple(repairs))
