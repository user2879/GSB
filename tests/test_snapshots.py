from __future__ import annotations

import json
from pathlib import Path

import pytest

from npr_editor.editing import Editor
from npr_editor.errors import NprError
from npr_editor.render import RenderConfig, RenderEngine
from npr_editor.snapshots import COMMITTED_NAME, SnapshotStore


def _render(project, config):
    engine = RenderEngine(project, config)
    return engine, engine.render("lineart")


def test_snapshot_commit_and_cache_restore(sample_project, tmp_path):
    config = RenderConfig(width=220, height=150, supersample=1)
    engine, result = _render(sample_project, config)
    store = SnapshotStore(tmp_path)
    snapshot = store.commit(sample_project, mode="lineart", result=result, engine=engine)
    report = store.recover()
    assert report.latest is not None
    assert report.latest.version_stamp == sample_project.version_stamp
    assert report.latest.snapshot_hash == snapshot.snapshot_hash
    assert report.reclaimed_orphans == ()

    restored_project = store.load_project(snapshot)
    assert restored_project.content_hash() == sample_project.content_hash()
    restored_engine = RenderEngine(restored_project, config)
    restored = store.restore_render_cache(snapshot, restored_engine, "lineart")
    assert set(restored) == {shape.id for shape in sample_project.shapes}
    restored_result = restored_engine.render("lineart")
    assert restored_result.pixel_sha256 == result.pixel_sha256
    assert restored_result.render_builds == 0
    assert restored_result.render_cache_hits == len(restored)


def test_snapshot_stale_version_and_conflicting_hash(sample_project, tmp_path):
    config = RenderConfig(width=180, height=120, supersample=1)
    store = SnapshotStore(tmp_path)
    engine, result = _render(sample_project, config)
    store.commit(sample_project, mode="lineart", result=result, engine=engine)

    older_doc = sample_project.to_dict()
    older_doc["version_stamp"] = "000000000000"
    from npr_editor.model import project_from_dict

    older = project_from_dict(older_doc)
    older_engine, older_result = _render(older, config)
    with pytest.raises(NprError) as caught:
        store.commit(older, mode="lineart", result=older_result, engine=older_engine)
    assert caught.value.code == "E_SNAPSHOT_STALE"

    conflicting_doc = sample_project.to_dict()
    conflicting_doc["style"]["line_color"] = [1, 2, 3]
    conflicting = project_from_dict(conflicting_doc)
    conflict_engine, conflict_result = _render(conflicting, config)
    with pytest.raises(NprError) as caught:
        store.commit(conflicting, mode="lineart", result=conflict_result, engine=conflict_engine)
    assert caught.value.code == "E_SNAPSHOT_STALE"


def test_interrupted_commit_is_orphan_and_recovered(sample_project, tmp_path):
    config = RenderConfig(width=180, height=120, supersample=1)
    store = SnapshotStore(tmp_path)
    engine, result = _render(sample_project, config)
    first = store.commit(sample_project, mode="lineart", result=result, engine=engine)

    editor = Editor(sample_project, config=config)
    editor.edit_control_point("tree_crown", 0, 0, 1, 150.0, 80.0)
    second_project = editor.project
    second_engine, second_result = _render(second_project, config)

    def crash(stage):
        if stage == "after_payload":
            raise RuntimeError("simulated forced termination")

    with pytest.raises(RuntimeError):
        store.commit(
            second_project,
            mode="lineart",
            result=second_result,
            engine=second_engine,
            fault_hook=crash,
        )
    assert any(path.name.startswith(".staging-") for path in store.root.iterdir())
    report = store.recover()
    assert report.latest is not None
    assert report.latest.snapshot_hash == first.snapshot_hash
    assert report.reclaimed_orphans
    assert not any(path.name.startswith(".staging-") for path in store.root.iterdir())


def test_uncommitted_directory_is_reclaimed(sample_project, tmp_path):
    config = RenderConfig(width=160, height=100, supersample=1)
    store = SnapshotStore(tmp_path)
    engine, result = _render(sample_project, config)
    first = store.commit(sample_project, mode="lineart", result=result, engine=engine)
    final_without_marker = store.root / "000000000099-deadbeefdeadbeef"
    final_without_marker.mkdir()
    (final_without_marker / "manifest.json").write_text("{}", encoding="utf-8")
    report = store.recover()
    assert report.latest is not None
    assert report.latest.snapshot_hash == first.snapshot_hash
    assert final_without_marker.name in report.reclaimed_orphans
    assert not final_without_marker.exists()


def test_snapshot_cache_is_not_reused_for_different_project(sample_project, tmp_path):
    config = RenderConfig(width=180, height=120, supersample=1)
    store = SnapshotStore(tmp_path)
    engine, result = _render(sample_project, config)
    snapshot = store.commit(sample_project, mode="lineart", result=result, engine=engine)
    editor = Editor(sample_project, config=config)
    editor.edit_control_point("tree_crown", 0, 0, 1, 150.0, 82.0)
    changed_engine = RenderEngine(editor.project, config)
    with pytest.raises(NprError) as caught:
        store.restore_render_cache(snapshot, changed_engine, "lineart")
    assert caught.value.code == "E_SNAPSHOT_STALE"


def test_corrupt_committed_snapshot_is_not_silently_used(sample_project, tmp_path):
    config = RenderConfig(width=160, height=100, supersample=1)
    store = SnapshotStore(tmp_path)
    engine, result = _render(sample_project, config)
    snapshot = store.commit(sample_project, mode="lineart", result=result, engine=engine)
    sidecar = snapshot.path / "project.json"
    sidecar.write_text("{}\n", encoding="utf-8")
    with pytest.raises(NprError) as caught:
        store.recover()
    assert caught.value.code == "E_SNAPSHOT_STALE"
