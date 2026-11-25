"""Requirement 2: v1/v2/v3 codecs, normalization provenance, E_FIELD/E_VERSION."""

import pytest

from obs_pipeline.errors import FieldError, VersionError
from obs_pipeline.records import decode_line, decode_stream, decimal_to_fixed, encode_line
from obs_pipeline.spec import validate_spec
from obs_pipeline import gen


def _source(fmt, count=3, scale=100):
    spec = validate_spec({
        "spec_version": 1,
        "task_interval": {"start": 0, "end": count},
        "stages": [{"id": "s", "kind": "std_diff", "params": {}, "scale": 100, "upstream": []}],
        "sources": [{
            "id": "src", "seed": 1, "record_count": count, "record_format": fmt,
            "metrics": ["m0"], "default_scale": scale, "null_bitmap": [0] * count,
        }],
        "windows": [{"id": "w0", "start": 0, "end": count}],
    })
    return spec.sources[0]


def test_v1_normalization_fills_fields():
    record = decode_line("v1|1|12.5", 1, _source(1), None)
    assert record.version == 1
    assert record.value_fixed == 1250
    assert set(record.filled_fields) == {"source_id", "timestamp", "metric_key", "scale"}
    rationale = dict(record.fill_rationale)
    assert "missing-in-v1" in rationale["source_id"]
    assert "missing-in-v1" in rationale["scale"]


def test_v2_normalization_fills_metric_and_scale():
    record = decode_line("v2|1|src|10|-3", 1, _source(2), None)
    assert record.source_id == "src"
    assert record.timestamp == 10
    assert set(record.filled_fields) == {"metric_key", "scale"}
    assert record.value_fixed == -300


def test_v3_fills_nothing():
    record = decode_line("v3|1|src|10|m0|100|1.23", 1, _source(3), None)
    assert record.filled_fields == ()
    assert record.value_fixed == 123


def test_monotonic_obs_seq_violation_keeps_line_number():
    source = _source(1, count=2)
    with pytest.raises(FieldError) as excinfo:
        decode_stream(["v1|1|5", "v1|1|6"], source)
    assert excinfo.value.code == "E_FIELD"
    assert excinfo.value.line_no == 2
    assert excinfo.value.field == "obs_seq"


def test_negative_timestamp():
    with pytest.raises(FieldError) as excinfo:
        decode_line("v2|1|src|-1|5", 7, _source(2), None)
    assert excinfo.value.line_no == 7
    assert excinfo.value.field == "timestamp"


def test_unsupported_scale_is_e_version():
    with pytest.raises(VersionError) as excinfo:
        decode_line("v3|1|src|0|m0|7|1.0", 4, _source(3), None)
    assert excinfo.value.code == "E_VERSION"
    assert excinfo.value.line_no == 4
    assert excinfo.value.field == "scale"


def test_roundtrip_encode_decode(sample_spec):
    for source in sample_spec.sources:
        records = gen.generate_source_records(source)
        lines = [encode_line(r) for r in records]
        decoded = decode_stream(lines, source)
        if source.record_format == 3:
            # v3 is lossless.
            assert [r.canonical() for r in decoded] == [r.canonical() for r in records]
        # All formats: encode->decode is idempotent from the normalized form.
        assert [encode_line(r) for r in decoded] == lines


def test_decimal_to_fixed_half_even():
    assert decimal_to_fixed("1.005", 100) == 100  # 100.5 -> 100 (half-even)
    assert decimal_to_fixed("1.015", 100) == 102  # 101.5 -> 102 (half-even)
    assert decimal_to_fixed("-2.5", 1) == -2
