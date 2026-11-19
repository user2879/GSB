from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import pytest

from acoustic_workbench.engine import IncrementalEngine
from acoustic_workbench.errors import (
    E_DEPENDENCY_STALE,
    E_GRID_BOUNDS,
    E_PARAM_RANGE,
    E_SNAPSHOT_CONFLICT,
    E_SPH_ORDER,
    E_TILE_HASH,
    WorkbenchError,
)
from acoustic_workbench.store import WorkspaceStore


def _copy_workspace(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    return target


def _expect_code(code: str, action) -> WorkbenchError:
    with pytest.raises(WorkbenchError) as caught:
        action()
    assert caught.value.code == code
    return caught.value


def test_parameter_errors_leave_workspace_usable(workspace: Path) -> None:
    engine = IncrementalEngine.open(workspace)
    revision = engine.scene["revision"]
    state_before = (workspace / "state.json").read_bytes()
    _expect_code(E_SPH_ORDER, lambda: engine.submit({"reconstruction.order": 99}, expected_revision=revision))
    _expect_code(
        E_SPH_ORDER,
        lambda: engine.submit({"receiver.sample_density": 32, "reconstruction.order": 5}, expected_revision=revision),
    )
    _expect_code(E_GRID_BOUNDS, lambda: engine.submit({"grid.width": 7}, expected_revision=revision))
    _expect_code(E_PARAM_RANGE, lambda: engine.submit({"wall.xmax.alpha": 1.5}, expected_revision=revision))
    assert (workspace / "state.json").read_bytes() == state_before
    reopened = IncrementalEngine.open(workspace)
    assert reopened.scene["revision"] == revision
    report = reopened.submit({"wall.xmax.alpha": 0.31}, expected_revision=revision)
    assert report["revision"] == revision + 1


def test_concurrent_submissions_produce_snapshot_conflict(workspace: Path) -> None:
    first = IncrementalEngine.open(workspace)
    second = IncrementalEngine.open(workspace)
    barrier = threading.Barrier(2)
    results: list[object] = []
    lock = threading.Lock()

    def submit(engine: IncrementalEngine, patch: dict[str, object]) -> None:
        barrier.wait(timeout=10)
        try:
            outcome: object = engine.submit(patch, expected_revision=1)
        except Exception as exc:
            outcome = exc
        with lock:
            results.append(outcome)

    threads = [
        threading.Thread(target=submit, args=(first, {"wall.xmax.alpha": 0.30})),
        threading.Thread(target=submit, args=(second, {"wall.ymin.alpha": 0.31})),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    errors = [result for result in results if isinstance(result, WorkbenchError)]
    successes = [result for result in results if isinstance(result, dict)]
    assert len(successes) == 1, results
    assert len(errors) == 1, results
    assert errors[0].code == E_SNAPSHOT_CONFLICT
    assert IncrementalEngine.open(workspace).scene["revision"] == 2


def test_tile_hash_mismatch_raises_e_tile_hash_on_isolated_copy(workspace: Path, tmp_path: Path) -> None:
    copy = _copy_workspace(workspace, tmp_path / "bad-tile")
    engine = IncrementalEngine.open(copy)
    victim = next(node for node in engine.state["nodes"].values() if node["type"] == "tile")
    engine.store.blob_path(victim["artifact_hash"]).write_bytes(b"tampered tile bytes")
    _expect_code(E_TILE_HASH, lambda: IncrementalEngine.open(copy))
    assert IncrementalEngine.open(workspace).scene["revision"] == 1


def test_artifact_hash_mismatch_raises_dependency_stale(workspace: Path, tmp_path: Path) -> None:
    copy = _copy_workspace(workspace, tmp_path / "bad-sh")
    engine = IncrementalEngine.open(copy)
    victim = next(node for node in engine.state["nodes"].values() if node["type"] == "sph")
    engine.store.blob_path(victim["artifact_hash"]).write_bytes(b"tampered SH coefficients")
    _expect_code(E_DEPENDENCY_STALE, lambda: IncrementalEngine.open(copy))


def test_input_hash_mismatch_raises_dependency_stale(workspace: Path, tmp_path: Path) -> None:
    copy = _copy_workspace(workspace, tmp_path / "bad-input")
    store = WorkspaceStore(copy)
    state = store.load_state()
    energy_id = next(node_id for node_id in state["nodes"] if node_id.startswith("energy:"))
    dependency_id = state["nodes"][energy_id]["dependencies"][0]
    state["nodes"][energy_id]["input_hashes"][dependency_id] = "0" * 64
    (copy / "state.json").write_text(json.dumps(state, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    _expect_code(E_DEPENDENCY_STALE, lambda: IncrementalEngine.open(copy))


def test_stale_expected_revision_produces_snapshot_conflict(workspace: Path) -> None:
    engine = IncrementalEngine.open(workspace)
    engine.submit({"wall.xmax.alpha": 0.29}, expected_revision=1)
    _expect_code(E_SNAPSHOT_CONFLICT, lambda: engine.submit({"wall.ymin.alpha": 0.30}, expected_revision=1))
    assert IncrementalEngine.open(workspace).scene["revision"] == 2
