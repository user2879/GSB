"""v3 binary partitioned exchange format.

File layout (little endian)::

    magic            6 bytes  b"FEATV3"
    format_version   uint16   currently 3
    partition_count  uint32
    feature_count    uint32
    feature name table: feature_count records of
        name_length uint16 + utf-8 bytes
    for each partition (partition id is the array index):
        entity_count   uint32
        feature_count  uint32
        entity name table: entity_count records of
            name_length uint16 + utf-8 bytes
        offset table   entity_count * feature_count uint32 byte offsets,
                       offsets are relative to the start of the data block
        data block     entity_count * feature_count float32 little endian,
                       NaN marks a missing slot

Partition boundaries and feature name order are preserved on the entity
table so downgrade/upgrade round trips can keep the same physical layout.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

from ..errors import (
    E_NONFINITE_INPUT,
    E_OFFSET_OUT_OF_RANGE,
    NonFiniteInputError,
    OffsetOutOfRangeError,
)
from ..models import V3, Cell, EntityTable

MAGIC = b"FEATV3"
FORMAT_VERSION = 3


def write_file(path: str | Path, table: EntityTable) -> None:
    features = table.feature_names()
    feature_index = {name: pos for pos, name in enumerate(features)}

    if table.partition_boundaries:
        partitions = [
            table.partition_boundaries[pid]
            for pid in sorted(table.partition_boundaries)
        ]
    else:
        partitions = [table.entity_ids()]

    lookup = table.cell_map()
    payload = bytearray()
    payload.extend(MAGIC)
    payload.extend(struct.pack("<HII", FORMAT_VERSION, len(partitions), len(features)))
    for name in features:
        encoded = name.encode("utf-8")
        payload.extend(struct.pack("<H", len(encoded)))
        payload.extend(encoded)

    for entity_ids in partitions:
        rows = len(entity_ids)
        cols = len(features)
        payload.extend(struct.pack("<II", rows, cols))
        for entity_id in entity_ids:
            encoded = str(entity_id).encode("utf-8")
            payload.extend(struct.pack("<H", len(encoded)))
            payload.extend(encoded)
        offset_table_pos = len(payload)
        payload.extend(b"\x00" * (4 * rows * cols))
        block_start = len(payload)
        offsets: list[int] = []
        for entity_id in entity_ids:
            for feature in features:
                offsets.append(len(payload) - block_start)
                cell = lookup.get((entity_id, feature))
                if cell is None or cell.missing:
                    number = float("nan")
                else:
                    number = float(cell.value)
                payload.extend(struct.pack("<f", number))
        payload[offset_table_pos : offset_table_pos + 4 * len(offsets)] = b"".join(
            struct.pack("<I", offset) for offset in offsets
        )

    Path(path).write_bytes(bytes(payload))


def parse_file(path: str | Path) -> EntityTable:
    data = Path(path).read_bytes()
    return parse_bytes(data, source_path=str(path))


def parse_bytes(data: bytes, source_path: str = "<memory>") -> EntityTable:
    cursor = 0

    def unpack(fmt: str, size: int):
        nonlocal cursor
        if cursor + size > len(data):
            raise OffsetOutOfRangeError(
                f"truncated v3 header at byte {cursor} in {source_path}",
                stage="v3_parse",
            )
        values = struct.unpack_from(fmt, data, cursor)
        cursor += size
        return values

    magic = data[0:6]
    if magic != MAGIC:
        raise OffsetOutOfRangeError(
            f"bad v3 magic {magic!r} in {source_path}", stage="v3_parse"
        )
    cursor = 6
    format_version, partition_count, feature_count = unpack("<HII", 10)
    if format_version != FORMAT_VERSION:
        raise OffsetOutOfRangeError(
            f"unsupported v3 format_version {format_version}, expected "
            f"{FORMAT_VERSION} in {source_path}",
            stage="v3_parse",
        )

    features: list[str] = []
    for _ in range(feature_count):
        (name_length,) = unpack("<H", 2)
        if cursor + name_length > len(data):
            raise OffsetOutOfRangeError(
                f"feature name overruns file in {source_path}", stage="v3_parse"
            )
        features.append(data[cursor : cursor + name_length].decode("utf-8"))
        cursor += name_length

    table = EntityTable(source_version=V3, feature_order=list(features))

    for partition_id in range(partition_count):
        entity_count, cols = unpack("<II", 8)
        if cols != feature_count:
            raise OffsetOutOfRangeError(
                f"partition {partition_id} feature_count {cols} != header "
                f"{feature_count} in {source_path}",
                stage="v3_parse",
            )
        table_size = entity_count * cols
        # Entity name table precedes the offset table and the float32 block.
        partition_entities: list[str] = []
        for _ in range(entity_count):
            (name_length,) = unpack("<H", 2)
            if cursor + name_length > len(data):
                raise OffsetOutOfRangeError(
                    f"partition {partition_id} entity name overruns file in "
                    f"{source_path}",
                    stage="v3_parse",
                )
            partition_entities.append(
                data[cursor : cursor + name_length].decode("utf-8")
            )
            cursor += name_length

        table_end = cursor + 4 * table_size
        if table_end > len(data):
            raise OffsetOutOfRangeError(
                f"partition {partition_id} offset table overruns file in "
                f"{source_path}",
                stage="v3_parse",
            )
        offsets = list(struct.unpack_from(f"<{table_size}I", data, cursor))
        cursor = table_end

        block_start = cursor
        block_byte_size = table_size * 4
        block_end = block_start + block_byte_size
        if block_end > len(data):
            raise OffsetOutOfRangeError(
                f"partition {partition_id} float32 block overruns file in "
                f"{source_path}",
                stage="v3_parse",
            )

        for index, offset in enumerate(offsets):
            if offset >= block_byte_size or offset % 4 != 0:
                raise OffsetOutOfRangeError(
                    f"partition {partition_id} slot {index} offset {offset} outside "
                    f"data block [0,{block_byte_size}) in {source_path}",
                    stage="v3_parse",
                )

        for row in range(entity_count):
            entity_id = partition_entities[row]
            for col, feature in enumerate(features):
                slot = row * cols + col
                absolute = block_start + offsets[slot]
                (number,) = struct.unpack_from("<f", data, absolute)
                raw_bytes = data[absolute : absolute + 4]
                missing = math.isnan(number)
                if not missing and not math.isfinite(number):
                    raise NonFiniteInputError(
                        f"non-finite v3 slot for {entity_id}/{feature}",
                        entity_id=entity_id,
                        stage="v3_parse",
                    )
                table.cells.append(
                    Cell(
                        entity_id=entity_id,
                        feature=feature,
                        source_version=V3,
                        value=float("nan") if missing else float(number),
                        missing=missing,
                        precision_label="float32",
                        raw_bytes=raw_bytes,
                        partition_id=partition_id,
                    )
                )
        table.partition_boundaries[partition_id] = partition_entities
        cursor = block_end

    return table


BUSINESS_CODES = (E_OFFSET_OUT_OF_RANGE, E_NONFINITE_INPUT)
