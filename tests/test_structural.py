import struct

import pytest

from fwaudit.errors import E_OVERLAP, E_SELFREF
from fwaudit.generator import build_image
from fwaudit.image import parse_image, HEADER_SIZE, PART_FIXED_SIZE


def _rebuild_with_regions(specs, version=1, base_image_id=100, base_slot=0):
    """按指定 (name, slot, offset, length) 手工排布镜像。"""
    table_offset = HEADER_SIZE
    records = bytearray()
    payloads = bytearray()
    cursor = table_offset + sum(
        PART_FIXED_SIZE + len(s[0].encode()) for s in specs
    )
    for name, slot, payload_offset, payload_length in specs:
        name_bytes = name.encode()
        records += (
            struct.pack("<H", len(name_bytes))
            + struct.pack("<BB", 1, slot)
            + struct.pack("<QQ", payload_offset, payload_length)
            + name_bytes
        )
    payload_end = max(
        (off + length for _, _, off, length in specs), default=0
    )
    total = cursor if payload_end <= cursor else payload_end
    reserved = struct.pack("<IB", base_image_id, base_slot) + b"\x00"
    header = (
        b"FWIM"
        + struct.pack("<H", version)
        + struct.pack("<I", len(specs))
        + struct.pack("<Q", table_offset)
        + struct.pack("<Q", total)
        + reserved
    )
    blob = bytearray(total)
    blob[0:HEADER_SIZE] = header
    blob[table_offset:table_offset + len(records)] = records
    return bytes(blob)


def test_partial_overlap_pair_names():
    blob = _rebuild_with_regions([
        ("alpha", 0, 128, 128),
        ("beta", 0, 200, 128),
    ])
    info = parse_image(blob)
    codes = [f["code"] for f in info.findings]
    assert E_OVERLAP in codes
    overlap = next(f for f in info.findings if f["code"] == E_OVERLAP)
    assert overlap["partitions"] == ["alpha", "beta"]
    assert overlap["relation"] == "partial"
    assert overlap["offset"] is not None


def test_full_containment():
    blob = _rebuild_with_regions([
        ("outer", 0, 128, 256),
        ("inner", 0, 160, 64),
    ])
    info = parse_image(blob)
    overlap = next(f for f in info.findings if f["code"] == E_OVERLAP)
    assert overlap["relation"] in ("contains", "contained_by")
    assert overlap["partitions"] == ["inner", "outer"]


def test_order_independence():
    a = _rebuild_with_regions([
        ("alpha", 0, 128, 128), ("beta", 0, 200, 128),
    ])
    b = _rebuild_with_regions([
        ("beta", 0, 200, 128), ("alpha", 0, 128, 128),
    ])
    fa = parse_image(a).findings
    fb = parse_image(b).findings
    assert fa == fb


def test_selfref_partition_table():
    # 载荷指向分区表内部
    blob = _rebuild_with_regions([
        ("alpha", 0, HEADER_SIZE + 2, 32),
    ])
    info = parse_image(blob)
    codes = [f["code"] for f in info.findings]
    assert E_SELFREF in codes
    selfref = next(f for f in info.findings if f["code"] == E_SELFREF)
    assert selfref["partition"] == "alpha"


def test_selfref_header_rejected_at_parse():
    blob = _rebuild_with_regions([
        ("alpha", 0, 16, 64),
    ])
    from fwaudit.errors import AuditError, E_PART
    with pytest.raises(AuditError) as exc:
        parse_image(blob)
    assert exc.value.code == E_PART


def test_clean_image_no_conflicts(image_blob):
    info = parse_image(image_blob)
    assert info.findings == []
