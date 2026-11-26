from __future__ import annotations

import json

import pytest

from voice_diag.errors import E_FIELD_TRUNCATED, DiagnosticError, FieldTruncatedError
from voice_diag.receipts import (
    A1Layout,
    ReceiptDocument,
    ReceiptRecord,
    compare_normalized_receipts,
    load_receipt_a1,
    parse_a1,
    parse_a2,
    to_a1,
    to_a2,
)


VOCAB = {1: "开灯", 2: "升温", 3: "关灯", 4: "播放"}


def _document() -> ReceiptDocument:
    return ReceiptDocument(
        (
            ReceiptRecord(
                interval_id="I001",
                speaker_id="S01",
                endpoint_confidence=0.923456,
                command_name="开灯",
                command_code=1,
                quant_scheme="float32",
                source_version="a2",
            ),
            ReceiptRecord(
                interval_id="I002",
                speaker_id="S02",
                endpoint_confidence=0.876543,
                command_name="升温",
                command_code=2,
                quant_scheme="float32",
                source_version="a2",
            ),
        ),
        "a2",
    )


def test_a2_to_a1_to_a2_restores_overwritten_fields_with_sidecar() -> None:
    original = _document()
    a1_conversion = to_a1(original, vocabulary=VOCAB)
    assert a1_conversion.text.splitlines()[0] == "I001        |S01     |000923|00001"
    sidecar_record = a1_conversion.sidecar["records"]["I001"]
    assert sidecar_record["precision_label"] == "float_to_permille_round_half_away"
    assert sidecar_record["residual"] == pytest.approx(0.000456)
    assert "endpoint_confidence" in sidecar_record["overwritten_fields"]
    assert "command_name" in sidecar_record["overwritten_fields"]

    restored = parse_a1(
        a1_conversion.text,
        vocabulary=VOCAB,
        sidecar=a1_conversion.sidecar,
    )
    assert restored.by_interval["I001"].endpoint_confidence == pytest.approx(0.923456)
    assert restored.by_interval["I001"].command_name == "开灯"
    assert restored.by_interval["I001"].quant_scheme == "float32"
    assert set(restored.by_interval["I001"].preserved_fields) >= {
        "endpoint_confidence",
        "command_name",
        "quant_scheme",
    }

    upconverted = to_a2(restored, vocabulary=VOCAB)
    roundtripped = parse_a2(upconverted.text, vocabulary=VOCAB)
    for interval_id in ("I001", "I002"):
        left = roundtripped.by_interval[interval_id]
        right = original.by_interval[interval_id]
        assert left.endpoint_confidence == pytest.approx(right.endpoint_confidence, abs=1e-12)
        assert left.command_name == right.command_name
        assert left.quant_scheme == right.quant_scheme


def test_without_sidecar_a1_is_permille_precision() -> None:
    a1_conversion = to_a1(_document(), vocabulary=VOCAB)
    parsed = parse_a1(a1_conversion.text, vocabulary=VOCAB)
    assert parsed.by_interval["I001"].endpoint_confidence == pytest.approx(0.923)
    assert parsed.by_interval["I001"].storage["confidence_permille"] == 923
    assert "quant_scheme" in parsed.by_interval["I001"].missing_fields


def test_dual_version_normalization() -> None:
    a1_conversion = to_a1(_document(), vocabulary=VOCAB)
    a2_conversion = to_a2(_document(), vocabulary=VOCAB)
    a1_doc = parse_a1(a1_conversion.text, vocabulary=VOCAB)
    a2_doc = parse_a2(a2_conversion.text, vocabulary=VOCAB)
    comparison = compare_normalized_receipts(a1_doc, a2_doc)
    assert comparison["ok"] is True
    assert comparison["normalized_digest"]


def test_a1_to_a2_to_a1_roundtrip_preserves_storage_fields() -> None:
    source = to_a1(_document(), vocabulary=VOCAB)
    parsed_a1 = parse_a1(source.text, vocabulary=VOCAB)
    upward = to_a2(parsed_a1, vocabulary=VOCAB)
    parsed_a2 = parse_a2(upward.text, vocabulary=VOCAB)
    for interval_id in ("I001", "I002"):
        left = parsed_a1.by_interval[interval_id]
        right = parsed_a2.by_interval[interval_id]
        assert right.speaker_id == left.speaker_id
        assert right.endpoint_confidence == pytest.approx(left.endpoint_confidence)
        assert right.command_code == left.command_code
        assert right.quant_scheme == "permille"

    downward = to_a1(parsed_a2, vocabulary=VOCAB)
    assert downward.text == source.text
    restored = parse_a1(downward.text, vocabulary=VOCAB, sidecar=downward.sidecar)
    assert restored.by_interval["I001"].endpoint_confidence == pytest.approx(0.923)
    assert restored.by_interval["I001"].command_name == "开灯"


@pytest.mark.parametrize(
    "line",
    [
        "I001|S01|000923|00001\n",
        "I001        |S01     |00X923|00001\n",
        "I001        |S01     |000923|0000!\n",
    ],
)
def test_a1_field_truncation_and_illegal_chars(line: str) -> None:
    with pytest.raises(FieldTruncatedError) as caught:
        parse_a1(line, vocabulary=VOCAB)
    assert caught.value.code == E_FIELD_TRUNCATED


def test_a1_serializer_rejects_too_narrow_field() -> None:
    document = ReceiptDocument(
        (
            ReceiptRecord(
                interval_id="INTERVAL-TOO-LONG",
                speaker_id="S01",
                endpoint_confidence=0.5,
                command_name="开灯",
                command_code=1,
                quant_scheme="permille",
                source_version="a2",
            ),
        ),
        "a2",
    )
    with pytest.raises(FieldTruncatedError):
        to_a1(document, layout=A1Layout(), vocabulary=VOCAB)


def test_a2_rejects_unknown_named_quantization() -> None:
    payload = {
        "version": "a2",
        "records": [
            {
                "interval_id": "I001",
                "speaker_id": "S01",
                "endpoint_confidence": 0.5,
                "command": "开灯",
                "quant_scheme": "made-up",
            }
        ],
    }
    with pytest.raises(DiagnosticError) as caught:
        parse_a2(payload, vocabulary=VOCAB, allowed_quant_schemes={"permille", "float32"})
    assert caught.value.code == "E_SCHEMA"


def test_a1_rejects_non_ascii_decimal_digits() -> None:
    with pytest.raises(FieldTruncatedError) as caught:
        parse_a1("I001        |S01     |０００９２３|00001\n", vocabulary=VOCAB)
    assert caught.value.code == E_FIELD_TRUNCATED


def test_missing_receipt_file_is_schema_error(tmp_path) -> None:
    with pytest.raises(DiagnosticError) as caught:
        load_receipt_a1(tmp_path / "missing.a1", vocabulary=VOCAB)
    assert caught.value.code == "E_SCHEMA"


def test_receipt_export_rejects_non_finite_confidence_and_negative_code() -> None:
    document = ReceiptDocument(
        (
            ReceiptRecord(
                interval_id="I001",
                speaker_id="S01",
                endpoint_confidence=float("nan"),
                command_name="开灯",
                command_code=1,
                quant_scheme="permille",
                source_version="a2",
            ),
        ),
        "a2",
    )
    with pytest.raises(DiagnosticError) as caught:
        to_a1(document, vocabulary=VOCAB)
    assert caught.value.code == "E_SCHEMA"

    negative = ReceiptDocument(
        (
            ReceiptRecord(
                interval_id="I001",
                speaker_id="S01",
                endpoint_confidence=0.5,
                command_name="开灯",
                command_code=-1,
                quant_scheme="permille",
                source_version="a2",
            ),
        ),
        "a2",
    )
    with pytest.raises(DiagnosticError) as caught:
        to_a2(negative, vocabulary=VOCAB)
    assert caught.value.code == "E_SCHEMA"
