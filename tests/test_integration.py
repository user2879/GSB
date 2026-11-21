from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mtcomp.runner import run_directory
from mtcomp.samplegen import SEED, write_sample


def _payload(result: dict) -> dict[str, dict]:
    fields = (
        "translation",
        "fragments",
        "confidence_permille",
        "confidence_fixed",
        "status",
        "boundary",
        "boundary_detail",
        "references",
        "cache",
        "error",
    )
    return {
        row["id"]: {field: row.get(field) for field in fields}
        for row in result["sentences"]
    }


def test_sample_incremental_full_parity_and_diagnostics(tmp_path: Path):
    sample = tmp_path / "sample"
    output = tmp_path / "out"
    write_sample(sample, seed=SEED)
    result = run_directory(sample, output, reset=True, compare_full=True)

    assert result["status"] == "OK"
    assert result["summary"]["sentence_count"] == 7
    assert result["summary"]["pending_count"] == 0
    assert set(result["summary"]["status_counts"]) == {"ACCEPT", "REJECT", "REVIEW"}
    assert result["parity"]["all_required_parity_checks_pass"] is True
    assert result["parity"]["sentence_payloads_equal"] is True
    assert result["parity"]["active_failures_equal"] is True
    assert result["parity"]["ledger_order_equal"] is True
    assert result["parity"]["ledger_static_fields_equal"] is True

    boundary_rows = [row for row in result["sentences"] if row["boundary"]]
    assert len(boundary_rows) == 1
    boundary = boundary_rows[0]
    assert boundary["status"] == "REVIEW"
    assert boundary["confidence_fixed"] == boundary["confidence"]["threshold_fixed"]
    detail = boundary["boundary_detail"]
    assert detail["reference_fixed_bits"] == boundary["confidence_fixed_bits"] + 1
    assert detail["reference_confidence_fixed"] == boundary["confidence_fixed"] * 2
    assert detail["reference_confidence_permille"] == boundary["confidence_permille"]

    for row in result["sentences"]:
        assert row["confidence_fixed_bits"] == 12
        assert row["confidence_fixed_scale"] == 4096
        if row["status"] != "ERROR":
            assert isinstance(row["confidence_permille"], int)
            assert isinstance(row["confidence_fixed"], int)

    injections = result["diagnostics"]["injections"]
    assert injections["all_isolated"] is True
    assert [item["item_id"] for item in injections["injections"]] == [
        "INJ-LEX-001",
        "INJ-WEIGHT-002",
        "INJ-SENT-003",
    ]
    assert [item["code"] for item in injections["injections"]] == [
        "E_LEX_INVALID",
        "E_CONFIG_INVALID",
        "E_SENT_INVALID",
    ]

    # The first edit deletes a lexeme used by no sentence: all seven records
    # must be reused and no decode point may be charged.
    run_two = [
        event
        for event in result["runtime_ledger"]["events"]
        if event["run_id"] == 2
    ]
    assert len(run_two) == 6  # S002 exists at this point and is reused too.
    assert all(event["action"] == "REUSED" for event in run_two)
    assert all(event["charged_points"] == 0 for event in run_two)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["completed_batch_cursor"] == 3
    assert manifest["command_cursor"] == len(json.loads((sample / "edits.json").read_text()))
    for artifact in manifest["artifacts"].values():
        path = output / "stages" / manifest["version_dir"] / artifact["file"]
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_resume_and_snapshot_stale_detection(tmp_path: Path):
    sample = tmp_path / "sample"
    output = tmp_path / "out"
    write_sample(sample, seed=SEED)
    first = run_directory(sample, output, reset=True, compare_full=True)
    first_payload = _payload(first)

    second = run_directory(sample, output, reset=False, compare_full=True)
    assert second["diagnostics"]["snapshot"]["resumed"] is True
    assert second["diagnostics"]["snapshot"]["stale"] is False
    assert _payload(second) == first_payload
    assert second["summary"]["total_charged_points"] == first["summary"]["total_charged_points"]
    assert second["parity"]["all_required_parity_checks_pass"] is True

    with (sample / "lexicon.tsv").open("a", encoding="utf-8") as handle:
        handle.write("extra 额外 1 0.5\n")
    third = run_directory(sample, output, reset=False, compare_full=True)
    assert third["diagnostics"]["snapshot"]["stale"] is True
    assert third["diagnostics"]["snapshot"]["resumed"] is False
    assert third["diagnostics"]["snapshot"]["stale_code"] == "E_SNAPSHOT_STALE"
    assert any(
        failure["code"] == "E_SNAPSHOT_STALE"
        for failure in third["failures"]
    )
    assert third["parity"]["all_required_parity_checks_pass"] is True


def test_failure_projection_ignores_historical_edit_errors(tmp_path: Path):
    sample = tmp_path / "sample"
    output = tmp_path / "out"
    write_sample(sample, seed=SEED)
    edits = json.loads((sample / "edits.json").read_text(encoding="utf-8"))
    edits.append(
        {
            "id": "E-INVALID",
            "op": "lex_frequency",
            "source": "does-not-exist",
            "target": "无",
            "value": 2,
        }
    )
    (sample / "edits.json").write_text(
        json.dumps(edits, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    result = run_directory(sample, output, reset=True, compare_full=True)
    assert any(
        failure["code"] == "E_EDIT_INVALID" for failure in result["failures"]
    )
    assert result["parity"]["all_required_parity_checks_pass"] is True
    assert not any(
        failure["code"] == "E_EDIT_INVALID"
        for failure in result["active_failures"]
        if failure["active"]
    )


def test_tampered_snapshot_artifact_is_stale(tmp_path: Path):
    sample = tmp_path / "sample"
    output = tmp_path / "out"
    write_sample(sample, seed=SEED)
    run_directory(sample, output, reset=True, compare_full=False)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    artifact = manifest["artifacts"]["decode"]["file"]
    path = output / "stages" / manifest["version_dir"] / artifact
    with path.open("ab") as handle:
        handle.write(b" ")
    result = run_directory(sample, output, reset=False, compare_full=True)
    assert result["diagnostics"]["snapshot"]["stale"] is True
    assert result["diagnostics"]["snapshot"]["stale_code"] == "E_SNAPSHOT_STALE"
    assert any(
        failure["code"] == "E_SNAPSHOT_STALE" for failure in result["failures"]
    )
    assert result["parity"]["all_required_parity_checks_pass"] is True


def test_loader_failures_are_identical_in_incremental_and_full_runs(tmp_path: Path):
    sample = tmp_path / "sample"
    output = tmp_path / "out"
    write_sample(sample, seed=SEED)
    with (sample / "sentences.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"id":"S-BAD","tokens":"outside vocabulary","batch":8}\n')
    with (sample / "lexicon.tsv").open("a", encoding="utf-8") as handle:
        handle.write("bad bad 0 1.4\n")
    result = run_directory(sample, output, reset=True, compare_full=True)
    codes = {failure["code"] for failure in result["active_failures"]}
    assert "E_SENT_INVALID" in codes
    assert "E_LEX_INVALID" in codes
    assert result["parity"]["active_failures_equal"] is True
    assert result["parity"]["all_required_parity_checks_pass"] is True
