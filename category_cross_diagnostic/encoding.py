"""Salted deterministic cross encoding and exact collision statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .enumeration import Combo
from .hashing import canonical_json, hash_cross_value, sha256_hex, stable_salt
from .sample_data import TableData


@dataclass(frozen=True)
class EncodingResult:
    encoding_width: int
    fixed_point_width: int
    quantization_levels: int
    code_space_size: int
    unique_code_count: int
    true_unique_count: int
    null_row_count: int
    null_code_count: int
    collision_count: int
    colliding_bucket_count: int
    colliding_pair_count: int
    dictionary_digest: str
    entries: tuple[dict[str, int], ...]
    first_row_codes: tuple[int, ...]

    def cardinality_diagnosis(self) -> dict[str, Any]:
        return {
            "true_unique_count": self.true_unique_count,
            "unique_code_count": self.unique_code_count,
            "null_row_count": self.null_row_count,
            "null_code_count": self.null_code_count,
            "collision_count": self.collision_count,
            "colliding_bucket_count": self.colliding_bucket_count,
            "colliding_pair_count": self.colliding_pair_count,
            "code_space_size": self.code_space_size,
            "space_utilization": round(
                self.unique_code_count / max(1, self.code_space_size), 12
            ),
        }

    def dictionary_summary(self, preview_limit: int = 20) -> dict[str, Any]:
        entries = list(self.entries)
        return {
            "algorithm": "blake2b-256:salted:canonical-tuple:bit-mask",
            "dictionary_size": self.unique_code_count,
            "entry_count": len(entries),
            "digest": self.dictionary_digest,
            "preview_limit": preview_limit,
            "preview_entries": entries[:preview_limit],
        }


def encode_combo(
    combo: Combo,
    table: TableData,
    *,
    encoding_width: int,
    fixed_point_width: int,
    quantization_levels: int,
    preview_limit: int = 20,
) -> EncodingResult:
    # Encoding width is intentionally not part of the salt.  Adjacent encoding
    # widths therefore refine the same hash equivalence classes, making the
    # compression attribution mathematically interpretable.  Fixed-point /
    # quantization gears do create a separate profile salt.
    salt_fingerprint = sha256_hex(
        "encoding-salt-v2",
        combo.base_fingerprint,
        fixed_point_width,
        quantization_levels,
    )
    salt = stable_salt(salt_fingerprint)

    code_row_counts: dict[int, int] = {}
    code_raw_tuples: dict[int, set[str]] = {}
    raw_tuple_codes: dict[str, int] = {}
    null_codes: set[int] = set()
    first_row_codes: list[int] = []
    null_row_count = 0

    # Explicitly ascending row order.  No set or dictionary iteration is used
    # for accumulation; final outputs are sorted before serialization.
    for row_index in range(table.row_count):
        parts: list[Any] = []
        is_null = False
        for column in combo.columns:
            value = table.values[column][row_index]
            missing = table.missing_bitmap[column][row_index]
            is_null = is_null or missing or value is None
            parts.append(None if missing or value is None else [type_tag(value), value])

        raw_key = canonical_json(parts)
        canonical_input = "\x1f".join(_flat_part(part) for part in parts)
        code = hash_cross_value(salt, canonical_input, encoding_width)
        first_row_codes.append(code)

        code_row_counts[code] = code_row_counts.get(code, 0) + 1
        code_raw_tuples.setdefault(code, set()).add(raw_key)
        raw_tuple_codes[raw_key] = code
        if is_null:
            null_row_count += 1
            null_codes.add(code)

    entries: list[dict[str, int]] = []
    collision_count = 0
    colliding_bucket_count = 0
    colliding_pair_count = 0
    for code in sorted(code_row_counts):
        raw_count = len(code_raw_tuples[code])
        collision_excess = max(0, raw_count - 1)
        collision_count += collision_excess
        if raw_count > 1:
            colliding_bucket_count += 1
            colliding_pair_count += raw_count * (raw_count - 1) // 2
        entries.append(
            {
                "code": code,
                "row_count": code_row_counts[code],
                "raw_tuple_count": raw_count,
                "collision_excess": collision_excess,
            }
        )

    digest = sha256_hex(
        "dictionary-v1",
        encoding_width,
        fixed_point_width,
        quantization_levels,
        [[entry["code"], entry["row_count"], entry["raw_tuple_count"]] for entry in entries],
    )
    code_space_size = 1 << encoding_width
    return EncodingResult(
        encoding_width=encoding_width,
        fixed_point_width=fixed_point_width,
        quantization_levels=quantization_levels,
        code_space_size=code_space_size,
        unique_code_count=len(code_row_counts),
        true_unique_count=len(raw_tuple_codes),
        null_row_count=null_row_count,
        null_code_count=len(null_codes),
        collision_count=collision_count,
        colliding_bucket_count=colliding_bucket_count,
        colliding_pair_count=colliding_pair_count,
        dictionary_digest=digest,
        entries=tuple(entries),
        first_row_codes=tuple(first_row_codes),
    )


def type_tag(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _flat_part(part: Any) -> str:
    if part is None:
        return "MISSING"
    return canonical_json(part)
