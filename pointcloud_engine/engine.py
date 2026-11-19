"""Top-level render pipeline orchestrating scheduling, recovery and output."""

from __future__ import annotations

import os

from .errors import EngineError
from .image import assemble_grids, verify_pair_same_version, write_outputs
from .io_utils import ensure_dir, hash_array
from .render import (
    PreparedPoints,
    downsample,
    prepare_points,
    render_serial,
)
from .scheduler import INTERRUPT_EXIT_CODE, InterruptFlag, RunOutcome, run_scheduler
from .scene import Scene, load_scene
from .state import TileStore, compute_input_digest


def render_scene(scene_path: str, *, install_signals: bool = True) -> dict:
    scene = load_scene(scene_path)
    assert scene.cloud is not None

    ensure_dir(scene.state_dir)
    from .io_utils import cleanup_temp_files

    cleanup_temp_files(scene.state_dir)

    version_id = scene.version_id(scene.cloud)
    store = TileStore(scene, version_id)
    prepared = prepare_points(scene)
    interrupt = InterruptFlag(scene.state_dir)
    if install_signals:
        interrupt.install()

    outcome = run_scheduler(scene, prepared, store, interrupt)
    if install_signals:
        interrupt.restore()

    if outcome.failure is not None:
        report = _failure_report(scene, prepared, outcome)
        return {"status": "failed", "exit_code": 1, "report": report, "error": outcome.failure.to_dict()}

    if outcome.interrupted:
        report = _progress_report(scene, prepared, outcome, version_id)
        report["interrupted"] = True
        report["exit_code"] = INTERRUPT_EXIT_CODE
        return {"status": "interrupted", "exit_code": INTERRUPT_EXIT_CODE, "report": report}

    # Final verification of every tile; repair any digest/pixel mismatch.
    verify_and_repair(scene, prepared, store, outcome)

    records = {idx: store.load_tile(idx) for idx in range(scene.tile_count)}
    if any(record is None for record in records.values()):
        missing = [idx for idx, record in records.items() if record is None]
        raise EngineError("E_DIGEST", "missing tile state after scheduling", tiles=missing)

    grid_gray, grid_depth = assemble_grids(scene, records)  # type: ignore[arg-type]
    gray, depth = downsample(scene, grid_gray, grid_depth)
    write_info = write_outputs(scene, gray, depth, version_id)
    verify_pair_same_version(scene.output_dir)

    report = _progress_report(scene, prepared, outcome, version_id)
    report.update(
        {
            "interrupted": False,
            "output": {
                "dir": scene.output_dir,
                "gray_pgm": os.path.join(scene.output_dir, "gray.pgm"),
                "depth_pfm": os.path.join(scene.output_dir, "depth.pfm"),
                "gray_npy": os.path.join(scene.output_dir, "gray.npy"),
                "depth_npy": os.path.join(scene.output_dir, "depth.npy"),
                "gray_shape": list(gray.shape),
                "depth_shape": list(depth.shape),
                "gray_digest": hash_array(gray),
                "depth_digest": hash_array(depth),
                "write": write_info,
            },
        }
    )
    return {"status": "completed", "exit_code": 0, "report": report}


def verify_and_repair(scene: Scene, prepared: PreparedPoints, store: TileStore,
                      outcome: RunOutcome) -> None:
    """Validate every finished tile once more and re-render stale ones.

    A tile that still fails verification after a fresh deterministic
    re-render indicates persistent corruption and raises ``E_DIGEST`` rather
    than silently using invalid data.
    """
    bad: list[int] = []
    for tile_index in range(scene.tile_count):
        record = store.load_tile(tile_index)
        if record is None:
            bad.append(tile_index)
            continue
        expected = compute_input_digest(scene, prepared, tile_index)
        ok, reason = store.verify_record(record, expected)
        if not ok:
            bad.append(tile_index)

    if not bad:
        return

    outcome.invalidated.extend(idx for idx in bad if idx not in outcome.invalidated)
    for tile_index in bad:
        store.archive_stale_files([store.path_for(tile_index)])
    # Deterministic re-render without signals/faults for the repair pass.
    from .state import render_and_persist

    for tile_index in bad:
        result = render_and_persist(scene, prepared, store, tile_index)
        outcome.rerendered.append(tile_index)
        record = store.load_tile(tile_index)
        expected = compute_input_digest(scene, prepared, tile_index)
        ok, reason = store.verify_record(record, expected)  # type: ignore[arg-type]
        if not ok:
            raise EngineError(
                "E_DIGEST",
                f"tile {tile_index} still fails verification after re-render: {reason}",
                tile_index=tile_index,
            )
        if result["hit_count"] != record.declared_hits:  # type: ignore[union-attr]
            raise EngineError("E_DIGEST", "declared pixel count mismatch after re-render",
                              tile_index=tile_index)


def render_serial_reference(scene_path: str):
    """Expose the serial ground-truth buffers for equivalence testing."""
    scene = load_scene(scene_path)
    prepared = prepare_points(scene)
    grid_gray, grid_depth, _ = render_serial(scene, prepared)
    gray, depth = downsample(scene, grid_gray, grid_depth)
    return scene, prepared, grid_gray, grid_depth, gray, depth


def _progress_report(scene: Scene, prepared: PreparedPoints, outcome: RunOutcome,
                     version_id: str) -> dict:
    return {
        "version_id": version_id,
        "point_count": prepared.total_points,
        "points_behind_near": prepared.behind_near,
        "points_culled_frustum": prepared.culled_frustum,
        "points_rendered": int(prepared.index.size),
        "grid": [scene.grid_width, scene.grid_height],
        "image": [scene.width, scene.height],
        "tile_count": scene.tile_count,
        "tiles_reused": sorted(outcome.reused),
        "tiles_rendered": sorted(outcome.rendered),
        "tiles_invalidated": sorted(outcome.invalidated),
        "tiles_rerendered": sorted(outcome.rerendered),
        "archived_files": outcome.archived,
        "concurrency": scene.concurrency,
        "memory_budget_bytes": scene.memory_budget_bytes,
        "max_inflight_tiles": min(
            scene.concurrency,
            max(1, scene.memory_budget_bytes // scene.tile_bytes()),
        ),
    }


def _failure_report(scene: Scene, prepared: PreparedPoints, outcome: RunOutcome) -> dict:
    report = _progress_report(scene, prepared, outcome, version_id="")
    report.update(
        {
            "failed_tile": outcome.failed_tile,
            "success_count_at_failure": outcome.success_count_at_failure,
        }
    )
    return report
