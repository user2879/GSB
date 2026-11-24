"""Three-format codec, strong validation and normalisation provenance."""
from __future__ import annotations

import json

import pytest

from obsflow.codec import load_stream, decode_stream, write_stream, encode_record
from obsflow.config import load_source_set
from obsflow.errors import FormatError, IntegrityError, OrderError
from obsflow.models import NormalizedRecord


def _declarations(sample_dir):
    return {s.source_id: s for s in
            load_source_set(sample_dir["declaration_path"]).sources}


def test_three_formats_normalise_with_provenance(sample_dir):
    declarations = _declarations(sample_dir)
    base = sample_dir["output_dir"]
    v1 = load_stream(f"{base}/src-v1.jsonl", declarations["src-v1"])
    v2 = load_stream(f"{base}/src-v2.jsonl", declarations["src-v2"])
    v3 = load_stream(f"{base}/src-v3.jsonl", declarations["src-v3"])

    assert len(v1) == len(v2) == len(v3) == 24
    assert set(v1[0].filled_fields) == {"source_id", "ts_ms", "metric_key",
                                        "scale", "source_format"}
    assert v1[0].source_id == "src-v1"
    assert v1[0].source_format == 1
    assert "declaration" in v1[0].fill_basis["metric_key"]
    assert set(v2[0].filled_fields) == {"metric_key", "scale",
                                        "source_format"}
    assert v3[0].filled_fields == ()
    # Original values must never be overwritten by fills.
    assert v3[0].record_scale == 10_000
    assert v2[0].ts_ms >= 0


def test_unknown_version_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"obsflow": "record-stream", "version": 9})
                    + "\n", encoding="utf-8")
    declaration = load_source_set.__self__ if False else None
    from obsflow.models import SourceDeclaration
    decl = SourceDeclaration("s", 1, "m", 100, {}, "bad.jsonl")
    with pytest.raises(FormatError) as info:
        load_stream(str(path), decl)
    assert info.value.code == "E_FORMAT"
    assert info.value.field == "version"
    assert info.value.line_no == 1


def test_missing_field_reports_line_and_field(tmp_path):
    from obsflow.models import SourceDeclaration
    decl = SourceDeclaration("s", 2, "m", 100, {}, "f.jsonl")
    path = tmp_path / "f.jsonl"
    lines = [
        json.dumps({"obsflow": "record-stream", "version": 2}),
        json.dumps({"seq": 1, "value": 2, "ts_ms": 3}),  # missing source_id
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(FormatError) as info:
        load_stream(str(path), decl)
    assert info.value.field == "source_id"
    assert info.value.line_no == 2


def test_sequence_regression_is_order_error(tmp_path):
    from obsflow.models import SourceDeclaration
    decl = SourceDeclaration("s", 1, "m", 100, {}, "f.jsonl")
    path = tmp_path / "f.jsonl"
    lines = [
        json.dumps({"obsflow": "record-stream", "version": 1}),
        json.dumps({"seq": 2, "value": 1}),
        json.dumps({"seq": 2, "value": 1}),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(OrderError) as info:
        load_stream(str(path), decl)
    assert info.value.code == "E_ORDER"
    assert isinstance(info.value, FormatError)  # load-time E_FORMAT family
    assert info.value.line_no == 3
    assert info.value.field == "seq"


def test_negative_timestamp_and_bad_scale(tmp_path):
    from obsflow.models import SourceDeclaration
    decl2 = SourceDeclaration("s", 2, "m", 100, {}, "f2.jsonl")
    p2 = tmp_path / "f2.jsonl"
    p2.write_text("\n".join([
        json.dumps({"obsflow": "record-stream", "version": 2}),
        json.dumps({"seq": 1, "value": 1, "source_id": "s", "ts_ms": -1}),
    ]), encoding="utf-8")
    with pytest.raises(FormatError) as info:
        load_stream(str(p2), decl2)
    assert info.value.field == "ts_ms"

    decl3 = SourceDeclaration("s", 3, "m", 100, {}, "f3.jsonl")
    p3 = tmp_path / "f3.jsonl"
    p3.write_text("\n".join([
        json.dumps({"obsflow": "record-stream", "version": 3}),
        json.dumps({"seq": 1, "value": 1, "source_id": "s", "ts_ms": 1,
                    "metric_key": "m", "scale": 7, "source_format": 3}),
    ]), encoding="utf-8")
    with pytest.raises(FormatError) as info:
        load_stream(str(p3), decl3)
    assert info.value.field == "scale"


def test_truncated_stream_is_integrity_error(tmp_path):
    from obsflow.models import SourceDeclaration
    decl = SourceDeclaration("s", 1, "m", 100, {}, "f.jsonl")
    path = tmp_path / "f.jsonl"
    path.write_text(json.dumps({"obsflow": "record-stream", "version": 1})
                    + '\n{"seq": 1, "val', encoding="utf-8")
    with pytest.raises(IntegrityError) as info:
        load_stream(str(path), decl)
    assert info.value.code == "E_INTEGRITY"


def test_roundtrip_encoding(sample_dir):
    declarations = _declarations(sample_dir)
    records = load_stream(f"{sample_dir['output_dir']}/src-v3.jsonl",
                          declarations["src-v3"])
    out = sample_dir["output_dir"] + "/roundtrip-v1.jsonl"
    write_stream(out, 1, records)
    reloaded = load_stream(out, declarations["src-v1"])
    assert [r.seq for r in reloaded] == [r.seq for r in records]
    assert all(r.source_format == 1 for r in reloaded)
