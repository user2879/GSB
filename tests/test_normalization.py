from __future__ import annotations

import json

from audit_forensics.canonical import CORE_EVENT_FIELDS, normalize_timestamp
from audit_forensics.ingest import parse_file
from helpers import BASE_TS, codes, event, import_workspace, rules_config, write_v1, write_v2


def test_cross_format_core_and_iso_normalization(tmp_path):
    v1_event = event(
        "e-1",
        ts=BASE_TS + 123,
        severity="low",
        source="endpoint",
    )
    v2_event = dict(v1_event)
    v2_event["ts"] = "2026-11-19T08:00:00.123+08:00"
    v2_event["severity"] = "high"
    v2_event["source"] = "collector"

    v1_path = write_v1(tmp_path / "looks-like-json.json", [v1_event])
    v2_path = write_v2(tmp_path / "looks-like-jsonl.jsonl", [v2_event])

    parsed_v1 = parse_file(v1_path)
    parsed_v2 = parse_file(v2_path)
    assert parsed_v1.detected_format == "v1"
    assert parsed_v2.detected_format == "v2"
    assert parsed_v1.occurrences[0].core == parsed_v2.occurrences[0].core
    assert parsed_v1.occurrences[0].core["ts"] == BASE_TS + 123

    result = import_workspace(
        tmp_path / "workspace",
        [v1_path, v2_path],
        rules_config(),
    )
    report = result.payload
    assert result.exit_code == 0
    assert len(report["events"]) == 1
    state = report["events"][0]
    assert state["status"] == "duplicate"
    assert state["occurrence_count"] == 2
    assert state["event"] == parsed_v1.occurrences[0].core
    assert tuple(state["event"].keys()) == CORE_EVENT_FIELDS
    assert state["attributes"]["severity"] == "high"
    assert report["conflicts"] == []


def test_iso_formats_are_timezone_exact():
    expected = 1_795_046_400_123
    assert normalize_timestamp("2026-11-19T08:00:00.123+08:00") == expected
    assert normalize_timestamp("2026-11-19T00:00:00.123Z") == expected
    assert normalize_timestamp("2026-11-19T00:00:00.123") == expected


def test_content_detection_handles_single_object_and_multiline_jsonl(tmp_path):
    single = write_v1(tmp_path / "single.data", [event("single")])
    assert parse_file(single).detected_format == "v1"

    multiline = tmp_path / "multi.data"
    multiline.write_text(
        json.dumps(event("m-1")) + "\n" + json.dumps(event("m-2", seq=2)) + "\n",
        encoding="utf-8",
    )
    parsed = parse_file(multiline)
    assert parsed.detected_format == "v1"
    assert len(parsed.occurrences) == 2

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(event("pretty"), indent=2) + "\n", encoding="utf-8")
    pretty_parsed = parse_file(pretty)
    assert pretty_parsed.detected_format == "v1"
    assert [item.core["event_id"] for item in pretty_parsed.occurrences] == ["pretty"]


def test_malformed_lines_and_elements_continue(tmp_path):
    v1 = write_v1(
        tmp_path / "v1.jsonl",
        [event("good-v1")],
        malformed_lines=['{"event_id":"broken"'],
    )
    v2_events = [
        event("good-v2", ts="2026-11-19T00:00:01Z", seq=2),
        event("bad-ts", ts="not-a-time", seq=3),
        {"event_id": "missing-fields"},
        42,
    ]
    v2 = write_v2(tmp_path / "v2.json", v2_events)
    result = import_workspace(
        tmp_path / "workspace",
        [v1, v2],
        rules_config(),
    )
    report = result.payload
    assert result.exit_code == 1
    assert {item["event"]["event_id"] for item in report["events"]} == {
        "good-v1",
        "good-v2",
    }
    assert {"E_MALFORMED_JSON", "E_TIMESTAMP", "E_SCHEMA"} <= codes(report)
    error_sources = {
        (error.get("source_format"), error.get("position"))
        for error in report["errors"]
        if error.get("source")
    }
    assert ("v1", "line:2") in error_sources
    assert ("v2", "element:1") in error_sources


def test_empty_and_unknown_inputs_are_file_level_errors(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    unknown = tmp_path / "unknown.data"
    unknown.write_text("not json at all\n", encoding="utf-8")
    good = write_v1(tmp_path / "good.jsonl", [event("e-ok")])
    result = import_workspace(
        tmp_path / "workspace", [empty, unknown, good], rules_config()
    )
    assert result.exit_code == 1
    assert {"E_EMPTY_INPUT", "E_FORMAT_UNKNOWN"} <= codes(result.payload)
    assert len(result.payload["events"]) == 1
