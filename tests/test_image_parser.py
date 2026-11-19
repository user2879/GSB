import pytest

from fwaudit.errors import AuditError, E_HEADER, E_PART
from fwaudit.generator import build_image
from fwaudit.image import parse_image, HEADER_SIZE


def test_parse_valid_image(image_blob):
    info = parse_image(image_blob)
    assert info.version == 1
    assert info.part_count == 3
    assert info.base_image_id == 100
    assert [p.name for p in info.partitions] == [
        "bootloader", "kernel", "appfs"
    ]
    assert info.ok is True
    # 载荷区紧随分区表
    assert info.table_end == info.partitions[0].payload_offset


def test_bad_magic_reports_offset(partitions):
    blob = build_image(partitions, corrupt={"magic": True})
    with pytest.raises(AuditError) as exc:
        parse_image(blob)
    assert exc.value.code == E_HEADER
    assert exc.value.offset == 0


def test_unsupported_version(partitions):
    blob = build_image(partitions, corrupt={"version": 9})
    with pytest.raises(AuditError) as exc:
        parse_image(blob)
    assert exc.value.code == E_HEADER
    assert exc.value.offset == 4


def test_declared_length_mismatch(partitions):
    blob = build_image(partitions, corrupt={"declared_delta": 3})
    with pytest.raises(AuditError) as exc:
        parse_image(blob)
    assert exc.value.code == E_HEADER


def test_truncated_image(partitions):
    blob = build_image(partitions, corrupt={"truncate": -50})
    with pytest.raises(AuditError) as exc:
        parse_image(blob)
    assert exc.value.code in (E_HEADER, E_PART)
    assert exc.value.offset is not None


def test_bad_utf8_name(partitions):
    blob = build_image(partitions, corrupt={"bad_name": 1})
    with pytest.raises(AuditError) as exc:
        parse_image(blob)
    assert exc.value.code == E_PART


def test_bad_type_and_slot(partitions):
    blob = build_image(partitions, corrupt={"bad_type": 0})
    with pytest.raises(AuditError) as exc:
        parse_image(blob)
    assert exc.value.code == E_PART
    blob = build_image(partitions, corrupt={"bad_slot": 0})
    with pytest.raises(AuditError) as exc:
        parse_image(blob)
    assert exc.value.code == E_PART


def test_u64_overflow_rejected(partitions):
    blob = build_image(partitions, corrupt={"force_overflow": 0})
    with pytest.raises(AuditError) as exc:
        parse_image(blob)
    assert exc.value.code == E_PART


def test_reserved_tail_nonzero(partitions):
    blob = build_image(partitions, corrupt={"reserved_tail": True})
    with pytest.raises(AuditError) as exc:
        parse_image(blob)
    assert exc.value.code == E_HEADER


def test_short_header():
    with pytest.raises(AuditError) as exc:
        parse_image(b"FW" + b"\x00" * 10)
    assert exc.value.code == E_HEADER
    assert exc.value.offset == 12
