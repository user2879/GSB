from __future__ import annotations

from pathlib import Path

import pytest

from fcdiag.errors import InputInvalidError
from fcdiag.receipts import (
    IRRECOVERABLE,
    PRESENT_BY_VERSION,
    downgrade_v2_to_v1,
    downgrade_v3_to_v2,
    parse_receipts,
    render_receipts,
    upgrade_v1_to_v2,
    upgrade_v2_to_v3,
)


def test_three_versions_normalize_and_parse_idempotently(context):
    for version in (1, 2, 3):
        records = context["receipts"][version]
        assert all(record.source_version == version for record in records)
        assert all(record.wire_version == version for record in records)
        assert parse_receipts(version, render_receipts(version, records)) == records
        for record in records:
            assert record.present == PRESENT_BY_VERSION[version]
            if version == 1:
                assert record.param_version is None and record.digest is None
            if version == 2:
                assert record.digest is None and record.duplicate_refs == ()


def test_repeated_parsing_is_field_identical_and_missing_fields_stay_none():
    text = Path("samples/receipts/v1.txt").read_text(encoding="utf-8")
    left = parse_receipts(1, text)
    right = parse_receipts(1, text)
    assert left == right
    assert [item.to_dict() for item in left] == [item.to_dict() for item in right]
    assert all(item.param_version is None and item.elapsed_ms is None for item in left)


def test_v1_to_v2_exact_float_and_precision_tag(context):
    result = upgrade_v1_to_v2(context["receipts"][1], param_version=9)
    first = result.records[0]
    source = context["receipts"][1][0]
    assert first.source_version == 1 and first.wire_version == 2
    assert first.param_version == 9
    assert result.sidecar[first.artifact_id]["hit_rate_float"] == source.hit_permille / 1000
    assert "permille" in result.precision_tags[first.artifact_id]
    reparsed = parse_receipts(2, result.text, strict=False)
    assert [item.to_dict() | {"source_version": 1} for item in reparsed] == [
        item.to_dict() for item in result.records
    ]
    assert any(diff["field"] == "param_version" for diff in result.diffs)


def test_v2_to_v1_sidecar_keeps_truncation_metadata(context):
    result = downgrade_v2_to_v1(context["receipts"][2])
    record = result.records[0]
    sidecar = result.sidecar[record.artifact_id]
    assert record.source_version == 2 and record.wire_version == 1
    assert record.present["elapsed_ms"] is False
    assert "truncation_residual" in sidecar and "precision_tag" in sidecar
    assert sidecar["param_version"] == context["receipts"][2][0].param_version


def test_v3_v2_roundtrip_and_irrecoverable_marker(context):
    down = downgrade_v3_to_v2(context["receipts"][3])
    assert down.records[0].digest is None
    assert down.sidecar[down.records[0].artifact_id]["digest"]
    restored = upgrade_v2_to_v3(down.records, down.sidecar)
    assert restored.records == context["receipts"][3]
    lost = upgrade_v2_to_v3(context["receipts"][2])
    assert all(record.digest == IRRECOVERABLE for record in lost.records)
    fields = {diff["field"] for diff in down.diffs}
    assert {"wire_version", "digest", "duplicate_refs", "present"} <= fields


def test_width_and_duplicate_validation(context):
    text = render_receipts(3, context["receipts"][3])
    lines = text.splitlines()
    with pytest.raises(InputInvalidError) as exc:
        parse_receipts(3, "\n".join(lines[:-1] + [lines[1]]) + "\n")
    messages = " ".join(v.message for v in exc.value.violations)
    assert "duplicate artifact_id" in messages
    with pytest.raises(InputInvalidError) as exc:
        parse_receipts(3, "\n".join(lines[:2] + [lines[1][:-1]]) + "\n")
    assert "field width violation" in str(exc.value)
