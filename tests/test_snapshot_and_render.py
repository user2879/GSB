from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from spatial_diag.canonical import sha256_file
from spatial_diag.render import render_snapshot_to_directory
from spatial_diag.service import run_workbench
from spatial_diag.snapshot import SnapshotError, SnapshotStore


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "qt-offscreen" not in path.name
    }


def test_two_identical_runs_are_byte_identical(committed_workspace):
    root, _ = committed_workspace
    before = _tree_hashes(root)
    run_workbench(root, render_view=True)
    after = _tree_hashes(root)
    assert before == after


def test_snapshot_verify_and_tamper_detection(committed_workspace, tmp_path: Path):
    root, _ = committed_workspace
    copied = tmp_path / "copied"
    shutil.copytree(root, copied)
    store = SnapshotStore(copied)
    assert store.verify_current()["ok"] is True
    snapshot = store.load_current()
    wav_record = next(record for record in snapshot["artifacts"] if record["path"].endswith(".wav"))
    wav_path = store.run_dir(snapshot) / wav_record["path"]
    with wav_path.open("ab") as handle:
        handle.write(b"tamper")
    result = store.verify_current()
    assert result["ok"] is False
    assert result["mismatches"]


def test_recovery_recycles_uncommitted_orphans(committed_workspace):
    root, _ = committed_workspace
    store = SnapshotStore(root)
    staging = store.staging / "partial-run"
    staging.mkdir()
    (staging / "partial.bin").write_bytes(b"incomplete")
    stale_run = store.runs / "sd1-stale"
    stale_run.mkdir()
    (stale_run / "snapshot.json").write_text("{}", encoding="utf-8")
    stale_view = store.views / "sd1-stale"
    stale_view.mkdir()
    (stale_view / "manifest.json").write_text("{}", encoding="utf-8")
    committed = store.load_current()["version"]
    recovered = store.recover()
    assert not staging.exists()
    assert not stale_run.exists()
    assert not stale_view.exists()
    assert store.load_current()["version"] == committed
    assert any(path.startswith("runs/sd1-stale") for path in recovered["recovered_orphans"])


def test_renderer_rejects_modified_snapshot(committed_workspace, tmp_path: Path):
    root, _ = committed_workspace
    snapshot = SnapshotStore(root).load_current()
    snapshot["tiers"]["small"]["decision"]["selected_order"] = 99
    with pytest.raises(SnapshotError):
        render_snapshot_to_directory(snapshot, tmp_path / "render")


def test_rendered_views_are_versioned_and_complete(committed_workspace):
    root, _ = committed_workspace
    store = SnapshotStore(root)
    snapshot = store.load_current()
    view_dir = store.views / snapshot["version"]
    for tier in snapshot["tiers"]:
        assert (view_dir / f"{tier}.png").is_file()
    manifest = json.loads((view_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["snapshot_version"] == snapshot["version"]
    assert manifest["snapshot_content_sha256"] == snapshot["integrity"]["content_sha256"]
    assert len(manifest["files"]) == len(snapshot["tiers"])


def test_qt_offscreen_export(committed_workspace):
    pytest.importorskip("PyQt5")
    root, _ = committed_workspace
    from spatial_diag.qt_view import launch_viewer

    assert launch_viewer(root, offscreen=True) == 0
    version = SnapshotStore(root).load_current()["version"]
    assert (root / "views" / version / "qt-offscreen-window.png").is_file()

