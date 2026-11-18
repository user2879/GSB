from __future__ import annotations

import copy
import json
from pathlib import Path

from remix_workbench.config import load_config, load_json
from remix_workbench.engine import RemixEngine, compare_signatures, render_signature
from remix_workbench.sequence import run_edit_sequence


def _run(config: dict, samples_root: Path, workdir: Path, workers: int = 4):
    return RemixEngine(
        samples_root=samples_root,
        config=config,
        workdir=workdir,
        force_workers=workers,
    ).execute()


def test_unchanged_rerun_reuses_every_artifact(
    config: dict, samples_root: Path, tmp_path: Path
) -> None:
    first = _run(config, samples_root, tmp_path / "run")
    assert first.report is not None
    second = _run(config, samples_root, tmp_path / "run")
    assert second.report is not None
    assert second.report["cache_summary"]["recompute_count"] == 0
    assert second.report["cache_summary"]["reuse_count"] > 0
    assert (
        render_signature(first.report, first.workdir)
        == render_signature(second.report, second.workdir)
    )


def test_single_gain_edit_only_invalidates_its_track_and_fragments(
    config: dict, samples_root: Path, tmp_path: Path
) -> None:
    workdir = tmp_path / "incremental"
    first = _run(config, samples_root, workdir)
    assert first.report is not None
    edited = copy.deepcopy(config)
    clip = next(item for item in edited["clips"] if item["clip_id"] == "clip_000")
    clip["tracks"]["drums"]["gain"] = 0.41
    second = _run(edited, samples_root, workdir)
    assert second.report is not None
    assert second.report["cache_summary"]["reuse_count"] > 0
    assert (
        second.report["cache_summary"]["recompute_count"]
        < second.report["cache_summary"]["events"]
    )
    changed_gain_events = [
        event
        for event in second.report["cache_recompute"]
        if event["stage"] == "gain" and event["clip_id"] == "clip_000" and event["track"] == "drums"
    ]
    assert changed_gain_events
    unaffected_gain_events = [
        event
        for event in second.report["cache_reuse"]
        if event["stage"] == "gain" and event["clip_id"] == "clip_001"
    ]
    assert unaffected_gain_events
    full = _run(edited, samples_root, tmp_path / "full", workers=1)
    assert full.report is not None
    comparison = compare_signatures(
        render_signature(second.report, second.workdir),
        render_signature(full.report, full.workdir),
    )
    assert comparison["matches"] is True


def test_edit_sequence_matches_full_render_after_every_edit(
    workspace: Path, samples_root: Path, tmp_path: Path
) -> None:
    config = load_config(workspace / "configs" / "remix.json")
    edits = load_json(workspace / "configs" / "edit_sequence.json")["edits"]
    manifest = load_json(samples_root / "manifest.json")
    result = run_edit_sequence(
        samples_root=samples_root,
        base_config=config,
        edits=edits,
        workdir=tmp_path / "sequence",
        sample_manifest=manifest,
    )
    assert result["stopped"] is False
    report = result["report"]
    assert report["edit_sequence"]["edit_count"] == len(edits)
    assert report["edit_sequence"]["all_edits_match_full"] is True
    assert report["edit_sequence"]["all_event_orders_match_serial"] is True
    assert report["edit_sequence"]["all_failure_orders_match_serial"] is True
    assert any(
        item["cache_summary"]["recompute_count"] > 0
        for item in report["edit_sequence"]["edits"]
    )
    assert any(
        item["cache_summary"]["reuse_count"] > 0
        for item in report["edit_sequence"]["edits"]
    )

