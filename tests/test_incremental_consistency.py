from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from rsseg.config import load_config
from rsseg.edits import apply_edit, copy_sample_set
from rsseg.runner import build_inputs, run_full_rebuild, run_incremental
from rsseg.snapshot import SnapshotManager
from rsseg.tiles import load_index, load_tile_bands
from rsseg.features import classify_tile


def _artifact_bundle(sample_dir: Path, work_dir: Path):
    inputs = build_inputs(sample_dir, work_dir,
                          patches_dir=sample_dir / "patches" if (sample_dir / "patches").exists() else None)
    manager = SnapshotManager(work_dir)
    info = manager.latest()
    assert info is not None
    grid = np.load(info.path / "global_grid.npy")
    postprocess_records = json.loads((info.path / "postprocess.json").read_text())
    score_maps = {}
    cfg = inputs.config
    for tile in inputs.index.tiles:
        bands = load_tile_bands(tile, sample_dir, inputs.index,
                                patches_dir=inputs.patches_dir)
        _class_map, _confidence, score_map = classify_tile(bands, cfg)
        score_maps[tile.key] = score_map
    return grid, postprocess_records, score_maps, info.manifest


def test_every_edit_step_matches_clean_full_rebuild(sample_set, tmp_path):
    sequence = json.loads((sample_set / "edit_sequence.json").read_text(encoding="utf-8"))
    incremental_sources = tmp_path / "inc_samples"
    incremental_work = tmp_path / "inc_work"
    copy_sample_set(sample_set, incremental_sources)

    # Baseline full run seeds the incremental cache.
    run_incremental(incremental_sources, incremental_work,
                    patches_dir=incremental_sources / "patches"
                    if (incremental_sources / "patches").exists() else None)

    for step in sequence["steps"]:
        apply_edit(incremental_sources, step)
        patches = incremental_sources / "patches"
        kwargs = {"patches_dir": patches if patches.exists() else None}
        inc_report = run_incremental(incremental_sources, incremental_work, **kwargs)[1]
        assert inc_report.snapshot_version is not None

        full_sources = tmp_path / f"full_sources_{step['id']}"
        full_work = tmp_path / f"full_work_{step['id']}"
        copy_sample_set(incremental_sources, full_sources)
        full_kwargs = {"patches_dir": full_sources / "patches"
                       if (full_sources / "patches").exists() else None}
        full_report = run_full_rebuild(full_sources, full_work, **full_kwargs)[1]
        assert full_report.snapshot_version is not None

        inc_bundle = _artifact_bundle(incremental_sources, incremental_work)
        full_bundle = _artifact_bundle(full_sources, full_work)
        assert np.array_equal(inc_bundle[0], full_bundle[0]), f"grid mismatch at {step['id']}"
        assert inc_bundle[1] == full_bundle[1], f"postprocess mismatch at {step['id']}"
        for key in inc_bundle[2]:
            assert np.array_equal(inc_bundle[2][key], full_bundle[2][key]), \
                f"score map mismatch at {step['id']} {key}"
        assert inc_bundle[3]["config_hash"] == full_bundle[3]["config_hash"]


def test_unchanged_tiles_are_reused_not_recomputed(sample_set, tmp_path):
    work = tmp_path / "work"
    sources = tmp_path / "sources"
    copy_sample_set(sample_set, sources)
    first = run_incremental(sources, work)[1]
    assert len(first.recomputed_tiles) == 9
    second = run_incremental(sources, work)[1]
    assert second.recomputed_tiles == []
    assert len(second.reused_tiles) == 9
    assert second.snapshot_version == 1


def test_band_patch_invalidates_only_target_tile_and_related_fusions(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    run_incremental(sources, work)
    sequence = json.loads((sources / "edit_sequence.json").read_text(encoding="utf-8"))
    patch_step = next(step for step in sequence["steps"] if step["type"] == "band_patch")
    apply_edit(sources, patch_step)
    report = run_incremental(sources, work, patches_dir=sources / "patches")[1]
    assert report.recomputed_tiles == [(1, 1)]
    affected_fusions = {name for name in report.recomputed_fusions if "1_1" in name}
    assert affected_fusions
    assert len(report.reused_tiles) == 8


def test_config_edit_invalidates_all_tiles_but_keeps_deterministic_output(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work_a = tmp_path / "work_a"
    work_b = tmp_path / "work_b"
    copy_sample_set(sample_set, sources)
    run_incremental(sources, work_a)
    edit = {"id": "threshold", "type": "config",
            "patch": {"feature_thresholds": {"ndvi": 0.33}}}
    apply_edit(sources, edit)
    report = run_incremental(sources, work_a)[1]
    assert len(report.recomputed_tiles) == 9
    full = run_full_rebuild(sources, work_b)[1]
    inc_grid = SnapshotManager(work_a).latest()
    full_grid = SnapshotManager(work_b).latest()
    assert np.array_equal(np.load(inc_grid.path / "global_grid.npy"),
                          np.load(full_grid.path / "global_grid.npy"))
