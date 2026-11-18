"""Requirement 8: versioned snapshots, resume gating and JSON reports."""

from __future__ import annotations

import json

from rag_app.engine import Engine
from rag_app.errors import E_CONFIG_MISMATCH
from rag_app.replay import load_inputs, run_replay
from rag_app.snapshot import ConfigMismatchError, SnapshotStore


def _engine(sample_dir, snapshot_dir):
    config, rules, faults, docs, turns, _cmds, failures = load_inputs(
        sample_dir
    )
    return Engine(
        docs=docs,
        turns=turns,
        config=config,
        rules=rules,
        store=SnapshotStore(snapshot_dir),
        base_failures=failures,
        faults=faults,
    )


def test_snapshot_manifest_tracks_versions_and_cursor(
    sample_dir, snapshot_dir
):
    engine = _engine(sample_dir, snapshot_dir)
    engine.run_full()
    manifest = engine.manifest()
    assert manifest["global_version"] == 1
    assert manifest["completed_cursor"] == len(engine.records)
    assert manifest["config_hash"]
    assert manifest["rules_fingerprint"]
    assert set(manifest["stage_versions"]) == {
        "retrieval",
        "extraction",
        "citation",
    }
    on_disk = SnapshotStore(snapshot_dir).read_manifest()
    assert on_disk == manifest
    engine.shutdown()


def test_resume_reuses_every_stage_and_skips_retrieval(
    sample_dir, snapshot_dir
):
    first = _engine(sample_dir, snapshot_dir)
    first.run_full()
    first.shutdown()

    second = _engine(sample_dir, snapshot_dir)
    prior = second.resume_from_manifest()
    assert prior is not None
    second.run_full()
    assert second.stats.retrievals_executed == 0
    pipeline_turns = len(second.records) - 1  # one forced-missing turn
    assert second.stats.cache_hits["retrieval"] == pipeline_turns
    assert second.stats.cache_hits["citation"] == pipeline_turns
    second.shutdown()


def test_config_mismatch_blocks_resume(sample_dir, snapshot_dir):
    engine = _engine(sample_dir, snapshot_dir)
    engine.run_full()
    engine.shutdown()

    config, rules, *_ = load_inputs(sample_dir)
    tampered = config.replace(recall_top_k=config.recall_top_k + 1)
    store = SnapshotStore(snapshot_dir)
    try:
        store.check_resume(tampered.fingerprint(), rules.fingerprint())
    except ConfigMismatchError as exc:
        assert exc.code == E_CONFIG_MISMATCH
        assert exc.detail["current_config_hash"] != (
            exc.detail["stored_config_hash"]
        )
    else:  # pragma: no cover
        raise AssertionError("expected E_CONFIG_MISMATCH")


def test_atomic_stage_files_have_no_temp_leftovers(
    sample_dir, snapshot_dir
):
    engine = _engine(sample_dir, snapshot_dir)
    engine.run_full()
    engine.shutdown()
    leftovers = list(snapshot_dir.rglob("*.tmp"))
    assert leftovers == []


def test_replay_report_contains_required_sections(
    sample_dir, snapshot_dir, tmp_path
):
    report_path = tmp_path / "replay.json"
    result = run_replay(sample_dir, snapshot_dir)
    from rag_app.report import write_report

    write_report(report_path, result)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    for section in (
        "incremental_turns",
        "full_turns",
        "incremental_failures",
        "incremental_stats",
        "command_log",
        "final_manifest",
        "diffs",
    ):
        assert section in payload
    for entry in payload["command_log"]:
        assert "elapsed_ms" in entry
        assert "over_budget" in entry
        assert "budget_ms" in entry
    assert payload["consistent"] is True


def test_global_version_increments_after_accepted_commands(
    sample_dir, snapshot_dir
):
    from rag_app.models import Command

    engine = _engine(sample_dir, snapshot_dir)
    engine.run_full()
    start_version = engine.global_version
    engine.handle_command(
        Command(seq=1, op="set_template", args={"template": "concise"})
    )
    assert engine.global_version == start_version + 1
    engine.shutdown()
