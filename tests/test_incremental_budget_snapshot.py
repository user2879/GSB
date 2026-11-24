from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from routing_workbench.engine import Workbench
from routing_workbench.errors import SimulatedCrash
from routing_workbench.injection import run_injections
from routing_workbench.verification import compare_reports, materialize_final_inputs


def _reference(sample_dir: Path, root: Path):
    inputs = materialize_final_inputs(
        sample_dir / "utterances.jsonl",
        sample_dir / "routing.yaml",
        sample_dir / "commands.json",
        root / "final-inputs",
    )
    return Workbench(
        utterances_path=inputs["utterances"],
        config_path=inputs["config"],
        commands_path=inputs["commands"],
        vocabulary_path=sample_dir / "vocabulary.txt",
        state_dir=root / "reference-state",
        full=True,
        batch_budget=10**9,
    ).run()


def test_incremental_equals_full_and_noop_reuses_all(sample_dir: Path, tmp_path: Path) -> None:
    incremental = Workbench(
        samples_dir=sample_dir,
        state_dir=tmp_path / "incremental",
        full=False,
        batch_budget=10**9,
    ).run()
    reference = _reference(sample_dir, tmp_path)
    assert compare_reports(incremental, reference)["equal"]
    assert incremental["business_digest"] == reference["business_digest"]

    noop = Workbench(
        samples_dir=sample_dir,
        state_dir=tmp_path / "incremental",
        full=False,
        batch_budget=10**9,
    ).run()
    assert noop["business_digest"] == incremental["business_digest"]
    for stage, counts in noop["cache"]["stages"].items():
        assert counts["recomputed"] == 0
        assert counts["reused"] == 13


def test_cache_invalidation_is_scoped_to_changed_trigger_domain(sample_dir: Path, tmp_path: Path) -> None:
    state = tmp_path / "state"
    empty_commands = tmp_path / "empty-commands.json"
    empty_commands.write_text('{"commands": []}', encoding="utf-8")
    first = Workbench(
        samples_dir=sample_dir,
        commands_path=empty_commands,
        state_dir=state,
        full=False,
        batch_budget=10**9,
    ).run()
    assert first["summary"]["result_count"] == 13
    command_path = tmp_path / "one-command.json"
    command_path.write_text(
        json.dumps(
            {
                "commands": [
                    {
                        "id": "C-ADD",
                        "type": "add_trigger",
                        "scope": "intent",
                        "domain": "account",
                        "intent": "password_reset",
                        "trigger": "锁定",
                        "weight": 2,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    second = Workbench(
        samples_dir=sample_dir,
        commands_path=command_path,
        state_dir=state,
        full=False,
        batch_budget=10**9,
    ).run()
    assert second["snapshot"]["resumed"] is True
    domain_counts = second["cache"]["stages"]["domain"]
    assert domain_counts["recomputed"] == 0
    assert domain_counts["reused"] == 13
    intent_counts = second["cache"]["stages"]["intent"]
    assert 0 < intent_counts["recomputed"] < 13
    # No billing utterance should have its slot stage recomputed for an account-only trigger edit.
    rows = second["cache"]["reuse_recompute"]
    billing_ids = {"S002-U1", "S002-U2", "S002-U3", "S004-U1", "S004-U3"}
    assert not any(row["utterance_id"] in billing_ids and row["stage"] == "slots" and row["status"] == "recompute" for row in rows)


def test_budget_pending_priority_and_drain(sample_dir: Path, tmp_path: Path) -> None:
    state = tmp_path / "budget-state"
    constrained = Workbench(samples_dir=sample_dir, state_dir=state, full=False, batch_budget=24).run()
    pending_before = [item["utterance_id"] for item in constrained["pending_queue"]]
    assert pending_before
    assert any(failure["business_code"] == "E_ROUTE_BUDGET_PENDING" for failure in constrained["failures"])
    prior_ledger_length = len(constrained["budget"]["sequence"])
    drained = Workbench(samples_dir=sample_dir, state_dir=state, full=False, batch_budget=10**9).run()
    assert drained["pending_queue"] == []
    new_ledger = drained["budget"]["sequence"][prior_ledger_length:]
    assert new_ledger
    assert new_ledger[0]["utterance_id"] == pending_before[0]
    assert drained["business_digest"] == _reference(sample_dir, tmp_path)["business_digest"]


def test_crash_resume_and_appended_command_prefix(sample_dir: Path, tmp_path: Path) -> None:
    state = tmp_path / "resume-state"
    crashed = False
    try:
        Workbench(
            samples_dir=sample_dir,
            state_dir=state,
            full=False,
            crash_after_commands=4,
            batch_budget=10**9,
        ).run()
    except SimulatedCrash:
        crashed = True
    assert crashed
    resumed = Workbench(samples_dir=sample_dir, state_dir=state, full=False, batch_budget=10**9).run()
    assert resumed["snapshot"]["resumed"] is True
    assert compare_reports(resumed, _reference(sample_dir, tmp_path))["equal"]

    all_commands = json.loads((sample_dir / "commands.json").read_text(encoding="utf-8"))["commands"]
    prefix_dir = tmp_path / "prefix"
    prefix_dir.mkdir()
    (prefix_dir / "c6.json").write_text(json.dumps({"commands": all_commands[:6]}, ensure_ascii=False), encoding="utf-8")
    (prefix_dir / "c12.json").write_text(json.dumps({"commands": all_commands}, ensure_ascii=False), encoding="utf-8")
    append_state = tmp_path / "append-state"
    Workbench(samples_dir=sample_dir, commands_path=prefix_dir / "c6.json", state_dir=append_state, full=False, batch_budget=10**9).run()
    appended = Workbench(samples_dir=sample_dir, commands_path=prefix_dir / "c12.json", state_dir=append_state, full=False, batch_budget=10**9).run()
    assert appended["snapshot"]["resumed"] is True
    assert appended["business_digest"] == _reference(sample_dir, tmp_path)["business_digest"]


def test_stale_version_forces_full_rebuild(sample_dir: Path, tmp_path: Path) -> None:
    state = tmp_path / "state"
    initial = Workbench(samples_dir=sample_dir, state_dir=state, full=False, batch_budget=10**9).run()
    expected_version = initial["snapshot"]["global_version"]
    stale = Workbench(
        samples_dir=sample_dir,
        state_dir=state,
        full=False,
        expected_global_version=expected_version + 100,
        batch_budget=10**9,
    ).run()
    assert stale["snapshot"]["resumed"] is False
    assert any(item["business_code"] == "E_SNAPSHOT_STALE" for item in stale["diagnostics"])
    assert stale["business_digest"] == _reference(sample_dir, tmp_path)["business_digest"]


def test_serial_and_parallel_outputs_are_identical(sample_dir: Path, tmp_path: Path) -> None:
    inputs = materialize_final_inputs(
        sample_dir / "utterances.jsonl",
        sample_dir / "routing.yaml",
        sample_dir / "commands.json",
        tmp_path / "inputs",
    )
    common = {
        "utterances_path": inputs["utterances"],
        "config_path": inputs["config"],
        "commands_path": inputs["commands"],
        "vocabulary_path": sample_dir / "vocabulary.txt",
        "full": True,
        "batch_budget": 10**9,
    }
    serial = Workbench(state_dir=tmp_path / "serial", workers=1, **common).run()
    parallel = Workbench(state_dir=tmp_path / "parallel", workers=4, **common).run()
    assert compare_reports(serial, parallel)["equal"]
    assert serial["cache"]["reuse_recompute"] == parallel["cache"]["reuse_recompute"]
    assert serial["budget"] == parallel["budget"]
    assert parallel["cache"]["workers"] == 4


def test_real_fault_injections_hit_and_runtime_is_isolated(sample_dir: Path) -> None:
    report = run_injections(
        utterances_path=sample_dir / "utterances.jsonl",
        config_path=sample_dir / "routing.yaml",
        injections_path=sample_dir / "injections.yaml",
        vocabulary_path=sample_dir / "vocabulary.txt",
    )
    assert report["all_hit"] is True
    by_id = {item["id"]: item for item in report["injections"]}
    assert by_id["INJ-001"]["failure_codes"] == ["E_ROUTE_INVALID"]
    assert by_id["INJ-002"]["failure_codes"] == ["E_ROUTE_INVALID"]
    assert by_id["INJ-003"]["failure_codes"] == ["E_ROUTE_INVALID"]
    assert by_id["INJ-004"]["failure_codes"] == ["E_ROUTE_INVALID"]
    assert by_id["INJ-005"]["failure_codes"] == ["E_ROUTE_INJECTED_RUNTIME"]
    assert by_id["INJ-005"]["result_count"] == 12
