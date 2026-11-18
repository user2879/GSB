from __future__ import annotations

import json

import pytest

from slowprof.errors import E_DEPTH, E_SAMPLE, SlowProfError
from slowprof.models import Interval
from slowprof.ndjson import load_samples, scan_ndjson


def test_small_metadata_and_validation(samples_dir, policy, small_interval):
    metadata = scan_ndjson(samples_dir / "small.ndjson", policy, small_interval)
    assert metadata.line_count == 12
    assert metadata.selected_row_count == 12
    assert metadata.selected_total_weight == 24
    assert metadata.unique_stack_count == 8
    assert metadata.selected_max_depth == 3
    assert len(metadata.frame_names) == 11
    assert len(metadata.file_sha256) == 64
    assert len(metadata.selected_samples_sha256) == 64


def test_zero_weight_rejected_with_line_and_field(samples_dir, policy):
    interval = Interval(1_720_000_000_000, 1_720_000_000_100, "obs-v1")
    with pytest.raises(SlowProfError) as raised:
        scan_ndjson(samples_dir / "anomalies" / "zero_weight.ndjson", policy, interval)
    error = raised.value
    assert error.code == E_SAMPLE
    assert error.family == E_SAMPLE
    assert error.line_no == 2
    assert error.field == "weight"


def test_depth_error_is_sample_family_with_specific_code(samples_dir, policy):
    interval = Interval(1_720_000_001_000, 1_720_000_001_100, "obs-v1")
    with pytest.raises(SlowProfError) as raised:
        load_samples(samples_dir / "anomalies" / "depth_exceeded.ndjson", policy, interval)
    error = raised.value
    assert error.code == E_DEPTH
    assert error.family == E_SAMPLE
    assert error.line_no == 2
    assert error.field == "frames"
    assert error.details == {"depth": 65, "max_depth": 64}


def test_timestamp_non_monotonic_rejected(tmp_path, policy, small_interval):
    path = tmp_path / "bad-time.ndjson"
    rows = [
        {
            "request_id": "r1",
            "timestamp_ms": 1_700_000_000_100,
            "duration_ms": 1,
            "service_id": "checkout",
            "frames": ["api.handle", "db.query"],
            "weight": 1,
        },
        {
            "request_id": "r2",
            "timestamp_ms": 1_700_000_000_099,
            "duration_ms": 1,
            "service_id": "checkout",
            "frames": ["api.handle", "db.query"],
            "weight": 1,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(SlowProfError) as raised:
        scan_ndjson(path, policy, small_interval)
    assert raised.value.code == E_SAMPLE
    assert raised.value.line_no == 2
    assert raised.value.field == "timestamp_ms"


def test_duplicate_json_key_rejected(tmp_path, policy, small_interval):
    path = tmp_path / "duplicate-key.ndjson"
    path.write_text(
        '{"request_id":"r1","timestamp_ms":1700000000100,"duration_ms":1,'
        '"service_id":"checkout","frames":["api.handle"],"weight":1,"weight":2}\n',
        encoding="utf-8",
    )
    with pytest.raises(SlowProfError) as raised:
        scan_ndjson(path, policy, small_interval)
    assert raised.value.code == E_SAMPLE
    assert raised.value.line_no == 1
    assert raised.value.field == "line"


@pytest.mark.parametrize(
    ("field", "value", "expected_field"),
    [
        ("duration_ms", 0, "duration_ms"),
        ("weight", True, "weight"),
        ("service_id", "", "service_id"),
        ("frames", ["not-a-frame"], "frames[0]"),
    ],
)
def test_strong_field_validation(tmp_path, policy, small_interval, field, value, expected_field):
    row = {
        "request_id": "r1",
        "timestamp_ms": 1_700_000_000_100,
        "duration_ms": 1,
        "service_id": "checkout",
        "frames": ["api.handle", "db.query"],
        "weight": 1,
    }
    row[field] = value
    path = tmp_path / "bad-field.ndjson"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(SlowProfError) as raised:
        scan_ndjson(path, policy, small_interval)
    assert raised.value.code == E_SAMPLE
    assert raised.value.line_no == 1
    assert raised.value.field == expected_field


def test_interval_filters_rows_but_validates_whole_stream(samples_dir, policy, changed_interval):
    samples, metadata = load_samples(samples_dir / "small.ndjson", policy, changed_interval)
    assert metadata.line_count == 12
    assert metadata.selected_row_count == 9
    assert metadata.selected_total_weight == 17
    assert len(samples) == 9
    assert samples[0].request_id == "req-small-0004"
