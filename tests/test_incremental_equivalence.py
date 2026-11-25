from __future__ import annotations

import shutil

from gsb_index.canonical import write_json
from gsb_index.edits import apply_edit_sequence, project_to_config
from gsb_index.engine import equivalence_projection, run_build
from gsb_index.snapshot import SnapshotStore


def test_edit_sequence_incremental_matches_blank_full_rebuild(sample_project, sample_config, sample_edits, tmp_path):
    current = sample_project
    incremental_work = tmp_path / "incremental"
    blank_full_work = tmp_path / "full-baseline"
    baseline_incremental = run_build(sample_config, incremental_work, mode="incremental", workers=3, global_version=1)
    baseline_full = run_build(sample_config, blank_full_work, mode="full", workers=3, global_version=1)
    assert equivalence_projection(baseline_incremental) == equivalence_projection(baseline_full)

    replacement_bucket_reuse = 0
    for edit_index, edit in enumerate(sample_edits, start=1):
        current, edit_issues = apply_edit_sequence(current, [edit])
        assert edit_issues == []
        config_path = tmp_path / f"edit-{edit_index:02d}.json"
        write_json(config_path, project_to_config(current))
        version = edit_index + 1
        incremental = run_build(
            config_path,
            incremental_work,
            mode="incremental",
            workers=3,
            global_version=version,
        )
        full = run_build(
            config_path,
            tmp_path / f"full-{edit_index:02d}",
            mode="full",
            workers=3,
            global_version=version,
        )
        assert equivalence_projection(incremental) == equivalence_projection(full), edit["edit_id"]
        assert incremental.failure_list == full.failure_list
        assert incremental.report["pending_shards"] == []
        if edit["edit_id"] == "e02-replace":
            replacement_bucket_reuse = incremental.report["reuse"]["counts"]["bucket_reuse"]
            assert incremental.report["reuse"]["counts"]["shard_reuse"] >= 1
    assert replacement_bucket_reuse > 0

    # Exactly unaffected shard snapshots are byte-for-byte reusable.
    current_loaded = SnapshotStore(incremental_work).load_current()
    assert current_loaded is not None
    assert current_loaded.manifest["completed_doc_cursor"] == len(current.documents)
    assert current_loaded.manifest["consumed_budget"] == current_loaded.ledger["entries_count"]


def test_serial_and_parallel_execution_match_field_by_field(sample_project, sample_edits, tmp_path):
    baseline_work = tmp_path / "baseline"
    baseline_config = tmp_path / "baseline.json"
    write_json(baseline_config, project_to_config(sample_project))
    baseline = run_build(
        baseline_config,
        baseline_work,
        mode="full",
        workers=3,
        global_version=1,
    )
    assert baseline.committed
    current, issues = apply_edit_sequence(sample_project, sample_edits[:1])
    assert issues == []
    config_path = tmp_path / "append.json"
    write_json(config_path, project_to_config(current))
    serial_work = tmp_path / "serial"
    parallel_work = tmp_path / "parallel"
    shutil.copytree(baseline_work, serial_work)
    shutil.copytree(baseline_work, parallel_work)
    serial = run_build(config_path, serial_work, mode="incremental", workers=1, global_version=2)
    parallel = run_build(config_path, parallel_work, mode="incremental", workers=8, global_version=2)
    assert equivalence_projection(serial) == equivalence_projection(parallel)
    assert serial.reuse_ledger == parallel.reuse_ledger
    assert serial.failure_list == parallel.failure_list
