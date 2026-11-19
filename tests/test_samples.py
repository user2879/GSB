from __future__ import annotations

import json

from audit_forensics.samples import generate_sample_dataset, verify_sync_manifest
from helpers import codes, parse_stdout, run_cli


def test_sample_generator_is_deterministic_and_sync_truth_checks(tmp_path):
    dataset = tmp_path / "dataset"
    first = generate_sample_dataset(dataset, force=True)
    first_bytes = {
        path.relative_to(dataset): path.read_bytes()
        for path in sorted(dataset.rglob("*"))
        if path.is_file()
    }
    second = generate_sample_dataset(dataset, force=True)
    second_bytes = {
        path.relative_to(dataset): path.read_bytes()
        for path in sorted(dataset.rglob("*"))
        if path.is_file()
    }
    assert first_bytes == second_bytes
    assert first["seed"] == 20261119
    assert len(first["occurrences"]) == len(second["occurrences"])

    workspace = dataset / "workspace"
    imported = run_cli(
        "import",
        "--workspace",
        str(workspace),
        "--rules",
        str(dataset / "rules.json"),
        "--input",
        str(dataset / "input" / "v1_primary.jsonl"),
        str(dataset / "input" / "v1_conflict.jsonl"),
        str(dataset / "input" / "v1_malformed.jsonl"),
        str(dataset / "input" / "v2_batch.json"),
        "--workers",
        "4",
    )
    assert imported.returncode == 1
    report = parse_stdout(imported)
    assert {
        "E_MALFORMED_JSON",
        "E_SCHEMA",
        "E_TIMESTAMP",
        "E_EVENT_CONFLICT",
        "E_ORPHAN",
        "E_CYCLE",
        "E_CROSS_TRACE",
        "E_SEQ_GAP",
        "E_SEQ_ROLLBACK",
    } <= codes(report)
    assert len(report["cases"]) == 4
    assert len(report["graph"]["nodes"]) == 17
    assert len(report["graph"]["edges"]) == 17
    assert {edge["kind"] for edge in report["graph"]["edges"]} == {
        "case_member",
        "causal_parent",
    }
    assert all(edge.get("basis") for edge in report["graph"]["edges"])
    assert report["checkpoint"]["watermark"]["unique_events"] == 13
    assert report["checkpoint"]["watermark"]["processed_occurrences"] == 15
    assert report["checkpoint"]["watermark"]["arrival_watermark"] == 14
    conflict = report["conflicts"][0]
    assert conflict["event_id"] == "evt-700-001"
    assert set(conflict["source_formats"]) == {"v1", "v2"}

    verification = verify_sync_manifest(first, report)
    assert verification["ok"] is True
    assert verification["checked"] == len(first["occurrences"])

    verified = run_cli("verify-checkpoint", "--workspace", str(workspace))
    assert verified.returncode == 0
    assert parse_stdout(verified)["valid"] is True
