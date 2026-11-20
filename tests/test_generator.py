from __future__ import annotations

import pytest

from trace_diag.errors import RecordError
from trace_diag.generator import generate_records, summarize_records


def test_generation_is_byte_deterministic(service_doc):
    service = service_doc.services[0]
    first = generate_records(service)
    second = generate_records(service)
    assert first.stream_bytes == second.stream_bytes
    assert first.stream_digest == second.stream_digest
    assert first.records == second.records


def test_records_have_expected_types_and_monotonic_time(service_doc):
    service = service_doc.services[0]
    stream = generate_records(service)
    assert len(stream.records) == service.record_count
    previous = None
    for record in stream.records:
        assert type(record["timestamp_ms"]) is int
        assert type(record["magnitude"]) is int
        assert isinstance(record["client_key"], str)
        assert type(record["duration_us"]) is int
        assert isinstance(record["trace_id"], str)
        assert type(record["sampled"]) is bool
        if previous is not None:
            assert record["timestamp_ms"] >= previous
        previous = record["timestamp_ms"]
    summary = summarize_records(service, stream.records)
    assert summary["null_or_invalid_field_count"] == 0
    assert summary["duration_min_us"] <= summary["duration_max_us"]
    assert summary["stream_digest"] == stream.stream_digest


def test_record_duration_violation_points_to_first_record(service_doc):
    service = service_doc.services[0]
    records = list(generate_records(service).records)
    records[3]["duration_us"] = service.fields["duration_us"]["max_us"] + 1
    with pytest.raises(RecordError) as captured:
        summarize_records(service, records)
    assert captured.value.context["record_index"] == 3
    assert captured.value.path == "$.services[svc-a].records[3].duration_us"


def test_timestamp_regression_points_to_first_record(service_doc):
    service = service_doc.services[0]
    records = list(generate_records(service).records)
    records[7]["timestamp_ms"] = records[6]["timestamp_ms"] - 1
    with pytest.raises(RecordError) as captured:
        summarize_records(service, records)
    assert captured.value.context["record_index"] == 7
    assert captured.value.path == "$.services[svc-a].records[7].timestamp_ms"


def test_null_field_is_rejected(service_doc):
    service = service_doc.services[0]
    records = list(generate_records(service).records)
    records[1]["client_key"] = None
    with pytest.raises(RecordError) as captured:
        summarize_records(service, records)
    assert captured.value.path == "$.services[svc-a].records[1].client_key"
