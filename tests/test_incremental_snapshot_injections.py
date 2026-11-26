from __future__ import annotations

import copy
from collections import Counter

from gsb.commands import apply_commands, load_command_stream, run_command_stream
from gsb.injections import run_injection_suite
from gsb.models import EditCommand
from gsb.pipeline import run_pipeline
from gsb.snapshot import SnapshotStore
from gsb.verification import compare_core_results


def _status_counts(invalidation: list[dict], artifact_type: str) -> dict[str, int]:
    return dict(
        Counter(
            item["status"]
            for item in invalidation
            if item["artifact_type"] == artifact_type
        )
    )


def test_annotation_replacement_invalidates_exactly_affected_artifacts(project, baseline):
    command = EditCommand(
        seq=1,
        op="replace_annotation_version",
        payload={
            "sample_id": "page_a",
            "annotation": "annotations/page_a.ann-v2.json",
        },
    )
    dataset, config, log = apply_commands(project.dataset, project.config, [command])
    result = run_pipeline(
        dataset,
        config=config,
        templates=project.require_templates(),
        cache=copy.deepcopy(baseline.cache),
        prior_recognition_results=baseline.recognition_results,
    )

    assert log[0]["status"] == "applied"
    assert _status_counts(result.report["invalidation"], "keypoint") == {
        "invalidated": 2,
        "reused": 34,
    }
    assert _status_counts(result.report["invalidation"], "recognition") == {
        "invalidated": 2,
        "reused": 34,
    }
    assert _status_counts(result.report["invalidation"], "reading") == {
        "invalidated": 2,
        "reused": 34,
    }
    assert result.report["incremental"]["recognition_reuses"] == 34
    assert all(
        item["status"] == "reused"
        for item in result.report["invalidation"]
        if item["artifact_type"] == "recognition"
        and item["artifact_id"].startswith("char:page_b-")
    )


def test_full_command_stream_incremental_equals_fresh_rebuild(project, baseline):
    commands, issues = load_command_stream(project.root / "configs" / "commands.jsonl")
    assert not issues
    incremental, log, _ = run_command_stream(
        dataset=project.dataset,
        config=project.config,
        templates=project.require_templates(),
        commands=commands,
        cache=copy.deepcopy(baseline.cache),
        prior_result=baseline,
    )
    fresh_dataset, fresh_config, fresh_log = apply_commands(
        project.dataset,
        project.config,
        commands,
    )
    fresh = run_pipeline(
        fresh_dataset,
        config=fresh_config,
        templates=project.require_templates(),
    )

    assert len(commands) == 12
    assert all(item["status"] == "applied" for item in log)
    assert all(item["status"] == "applied" for item in fresh_log)
    assert compare_core_results(incremental.report, fresh.report)["equal"]


def test_snapshot_commit_versioning_and_corruption_recovery(tmp_path, baseline):
    store = SnapshotStore(tmp_path / "snapshots")
    first = store.commit(
        baseline.report,
        completed_command_cursor=0,
        consumed_budget=baseline.report["budget"]["consumed"],
    )
    first_load = store.load_current()
    assert first_load.current_valid
    assert first_load.report == baseline.report

    second_report = copy.deepcopy(baseline.report)
    second_report["snapshot_marker"] = "second"
    second = store.commit(
        second_report,
        completed_command_cursor=12,
        consumed_budget=12345,
    )
    second_load = store.load_current()
    assert second_load.current_valid
    assert second.global_version != first.global_version
    assert second.manifest["completed_command_cursor"] == 12
    assert second.manifest["consumed_budget"] == 12345

    (second.snapshot_dir / "stages" / "keypoints.json").write_text(
        "{\"corrupt\":true}\n",
        encoding="utf-8",
    )
    recovery = store.load_current()
    assert recovery.recovered
    assert not recovery.current_valid
    assert recovery.report == baseline.report
    assert recovery.manifest["global_version"] == first.global_version
    assert any("hash mismatch" in issue for issue in recovery.issues)


def test_all_numbered_injections_hit_without_stopping_other_sample(project):
    result = run_injection_suite(
        root=project.root,
        config_path=project.config_path,
        dataset_path=project.dataset_path,
        templates_path=project.templates_path,
        injections_path=project.root / "injections" / "injections.yaml",
    )
    by_id = {item["id"]: item for item in result["injections"]}

    assert result["all_hit"]
    assert list(by_id) == ["INJ-01", "INJ-02", "INJ-03", "INJ-04"]
    assert by_id["INJ-01"]["code"] == "E_SAMPLE_FORMAT"
    assert by_id["INJ-01"]["surviving_samples"] == ["page_b"]
    assert by_id["INJ-02"]["code"] == "E_SAMPLE_FORMAT"
    assert by_id["INJ-02"]["surviving_samples"] == ["page_b"]
    assert by_id["INJ-03"]["code"] == "E_TEMPLATE_VARIANT_DIMENSION"
    assert by_id["INJ-04"]["code"] == "E_CONFIG_INVALID"
