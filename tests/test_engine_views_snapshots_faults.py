from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

from gsb_ledger.engine import LedgerEngine
from gsb_ledger.scenario import run_scenario
from gsb_ledger.views import compare_views, render_view


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample"


def _copy_sample(tmp_path: Path) -> Path:
    target = tmp_path / "project"
    shutil.copytree(
        SAMPLE,
        target,
        ignore=shutil.ignore_patterns(".state", "__pycache__", "*.pyc"),
    )
    return target


def test_ledger_authority_rules_and_reuse_boundary(tmp_path):
    engine = LedgerEngine(SAMPLE, state_dir=tmp_path / "state")
    result = engine.build(resume=False)
    assert result.ledger["summary"]["entries_authoritative"] == 6
    assert result.ledger["summary"]["batches_invalidated"] == 1
    assert result.ledger["invalidation"]["invalidated_batch_ids"] == [
        "b-alpha-legacy/w-alpha"
    ]
    assert set(result.ledger["invalidation"]["reused_request_ids"]) == {
        "r-001",
        "r-002",
        "r-003",
        "r-004",
        "r-005",
        "r-006",
    }
    assert result.ledger["failures"] == []


def test_incremental_scenario_matches_blank_full_rebuild(tmp_path):
    report = run_scenario(
        source_project=SAMPLE,
        work_dir=tmp_path / "work",
        commands_path=SAMPLE / "commands" / "incremental_sequence.json",
        out_dir=tmp_path / "out",
    )
    assert report["identical"], report["differences"]
    assert report["incremental_ledger_version"] == report["full_ledger_version"]
    assert report["summary"]["batches_total"] == 7
    assert report["summary"]["batches_invalidated"] == 3
    assert report["summary"]["batches_reused"] == 3
    assert report["summary"]["invalidated_requests"] == 2
    incremental = json.loads((tmp_path / "out" / "incremental-ledger.json").read_text())
    assert len(incremental["entries"]) == 6
    assert {entry["authority_rule"] for entry in incremental["entries"]} >= {
        "coverage_ref",
        "param_version_desc_batch_id_asc",
    }


def test_fast_partial_and_authoritative_common_range_are_field_identical(tmp_path):
    engine = LedgerEngine(SAMPLE, state_dir=tmp_path / "state")
    result = engine.build(resume=False)
    fast_complete = render_view(result.ledger, mode="fast", budget_ms=1000)
    authority = render_view(result.ledger, mode="authoritative", budget_ms=None)
    assert fast_complete["status"] == "complete"
    assert fast_complete["entries"] == authority["entries"]
    assert compare_views(fast_complete, authority) == []

    exhausted = render_view(result.ledger, mode="fast", budget_ms=0)
    assert exhausted["status"] == "partial"
    assert exhausted["error_codes"] == ["E_RESPONSE_BUDGET_EXCEEDED"]
    assert exhausted["progress"]["completed"] == 0
    assert compare_views(exhausted, authority) == []

    mutated = copy.deepcopy(fast_complete)
    mutated["entries"][0]["authority"]["output_digest"] = "sha256:" + "0" * 64
    issues = compare_views(mutated, authority)
    assert issues and issues[0].code == "E_LEDGER_DIVERGED"
    assert "authority.output_digest" in issues[0].context["fields"]


def test_cold_query_with_zero_budget_returns_before_full_build(tmp_path):
    engine = LedgerEngine(SAMPLE, state_dir=tmp_path / "cold-state")
    view = engine.query(mode="fast", budget_ms=0, resume=False)
    assert view["status"] == "partial"
    assert view["error_codes"] == ["E_RESPONSE_BUDGET_EXCEEDED"]
    assert view["progress"]["completed"] == 0
    assert view["progress"]["total"] == 6
    assert view["summary"]["cold_start"] is True
    assert not (tmp_path / "cold-state" / "CURRENT.json").exists()


def test_snapshot_stage_versions_resume_and_config_change_detection(tmp_path):
    project = _copy_sample(tmp_path)
    state = tmp_path / "state"
    first = LedgerEngine(project, state_dir=state, command_cursor=3).build(resume=False)
    assert first.build_info["stage_versions"].keys() >= {
        "parse",
        "attribution",
        "resources",
        "ledger",
    }
    current = json.loads((state / "CURRENT.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (state / current["run_dir"] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["command_cursor"] == 3
    assert manifest["config_hash"] == first.ledger["config_hash"]
    assert manifest["global_version"] == first.build_info["global_version"]
    assert not any((state / "staging").iterdir())

    resumed = LedgerEngine(project, state_dir=state, command_cursor=3).build(resume=True)
    assert resumed.build_info["resumed"] is True
    assert resumed.ledger == first.ledger

    cursor_changed = LedgerEngine(project, state_dir=state, command_cursor=4).build(
        resume=True
    )
    assert cursor_changed.build_info["resumed"] is False
    assert cursor_changed.build_info["resume_status"]["status"] == "cursor_changed"
    current = json.loads((state / "CURRENT.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (state / current["run_dir"] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["command_cursor"] == 4

    raw = json.loads((project / "config.json").read_text(encoding="utf-8"))
    raw["batch_size"] = 2
    (project / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    changed = LedgerEngine(project, state_dir=state, command_cursor=5).build(resume=True)
    assert changed.build_info["resumed"] is False
    assert changed.build_info["config_changed"] is True
    assert changed.build_info["resume_status"]["status"] == "config_changed"


def test_corrupt_snapshot_pointer_is_identified_and_recovered(tmp_path):
    engine = LedgerEngine(SAMPLE, state_dir=tmp_path / "state")
    engine.build(resume=False)
    (tmp_path / "state" / "CURRENT.json").write_text("{not-json", encoding="utf-8")
    status = LedgerEngine(SAMPLE, state_dir=tmp_path / "state").status()
    assert status["status"] == "corrupt"
    assert status["error_code"] == "E_SNAPSHOT_CORRUPT"
    rebuilt = LedgerEngine(SAMPLE, state_dir=tmp_path / "state").build(resume=True)
    assert rebuilt.build_info["resumed"] is False
    assert rebuilt.build_info["resume_status"]["status"] == "corrupt"


def test_corrupt_snapshot_stage_is_identified(tmp_path):
    state = tmp_path / "state"
    LedgerEngine(SAMPLE, state_dir=state).build(resume=False)
    pointer = json.loads((state / "CURRENT.json").read_text(encoding="utf-8"))
    ledger_path = state / pointer["run_dir"] / "ledger.json"
    ledger_path.write_text("{}", encoding="utf-8")
    status = LedgerEngine(SAMPLE, state_dir=state).status()
    assert status["status"] == "corrupt"
    assert status["error_code"] == "E_SNAPSHOT_CORRUPT"


def test_render_downgrade_is_recorded_and_restored_without_loss(tmp_path):
    normal = LedgerEngine(SAMPLE, state_dir=tmp_path / "normal").build(resume=False)
    degraded = LedgerEngine(SAMPLE, state_dir=tmp_path / "degraded").build(
        resume=False,
        render_version="r1",
    )
    assert degraded.ledger["summary"]["degradations"] == 8
    assert not any(
        issue["code"] == "E_CONVERSION_LOSS" for issue in degraded.ledger["failures"]
    )
    assert all(
        not item["lost_fields"] for item in degraded.ledger["degradations"]
    )
    normal_authority = {
        entry["request_id"]: entry["authority"] for entry in normal.ledger["entries"]
    }
    degraded_authority = {
        entry["request_id"]: entry["authority"] for entry in degraded.ledger["entries"]
    }
    for request_id, authority in degraded_authority.items():
        for field in (
            "batch_id",
            "worker_id",
            "param_version",
            "output_digest",
            "duration_ms",
            "stable_rank",
            "param_digest",
        ):
            assert authority[field] == normal_authority[request_id][field]


def test_faults_hit_exact_ids_codes_and_source_files_are_unchanged(tmp_path):
    engine = LedgerEngine(
        SAMPLE,
        state_dir=tmp_path / "state",
        fault_config="faults/isolated_faults.json",
    )
    receipt_paths = sorted((SAMPLE / "receipts").glob("*.ndjson"))
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in receipt_paths}
    result = engine.build(resume=False)
    diagnostics = result.ledger["diagnostics"]
    assert {item["injection_id"] for item in diagnostics} == {
        "INJ-TRUNCATE-001",
        "INJ-UNKNOWN-VERSION-001",
        "INJ-NONFINITE-DURATION-001",
        "INJ-DUPLICATE-RANK-001",
    }
    assert all(item["hit"] for item in diagnostics)
    observed = {
        item["injection_id"]: set(item["observed_codes"]) for item in diagnostics
    }
    assert observed["INJ-TRUNCATE-001"] == {"E_RECEIPT_MALFORMED"}
    assert observed["INJ-UNKNOWN-VERSION-001"] == {"E_UNKNOWN_PARAM_VERSION"}
    assert observed["INJ-NONFINITE-DURATION-001"] == {"E_RECEIPT_MALFORMED"}
    assert observed["INJ-DUPLICATE-RANK-001"] == {"E_DUPLICATE_RANK"}
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in receipt_paths}
    assert before == after

    second = engine.build(resume=False)
    assert second.build_info["parse_cache"] == "hit"
    assert second.ledger == result.ledger
