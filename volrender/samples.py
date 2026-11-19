from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import run_task
from .spec import load_task


def _transfer_a() -> list[dict[str, float]]:
    return [
        {"scalar": 0.0, "opacity": 0.0, "gray": 0.0},
        {"scalar": 0.20, "opacity": 0.015, "gray": 0.10},
        {"scalar": 0.55, "opacity": 0.060, "gray": 0.62},
        {"scalar": 1.00, "opacity": 0.160, "gray": 1.00},
    ]


def _transfer_b() -> list[dict[str, float]]:
    return [
        {"scalar": 0.0, "opacity": 0.0, "gray": 0.0},
        {"scalar": 0.20, "opacity": 0.025, "gray": 0.15},
        {"scalar": 0.55, "opacity": 0.105, "gray": 0.78},
        {"scalar": 1.00, "opacity": 0.190, "gray": 1.00},
    ]


def _transfer_c() -> list[dict[str, float]]:
    return [
        {"scalar": 0.0, "opacity": 0.0, "gray": 0.0},
        {"scalar": 0.15, "opacity": 0.020, "gray": 0.08},
        {"scalar": 0.50, "opacity": 0.080, "gray": 0.55},
        {"scalar": 1.00, "opacity": 0.210, "gray": 0.96},
    ]


def _camera_a() -> dict[str, Any]:
    return {
        "position": [55.0, 47.0, 53.0],
        "target": [18.0, 16.0, 15.5],
        "up": [0.0, 1.0, 0.0],
        "fov_y_deg": 58.0,
    }


def _camera_b() -> dict[str, Any]:
    return {
        "position": [50.0, 52.0, 60.0],
        "target": [20.0, 16.0, 14.0],
        "up": [0.0, 1.0, 0.0],
        "fov_y_deg": 44.0,
    }


def _camera_c() -> dict[str, Any]:
    return {
        "position": [44.0, 55.0, 58.0],
        "target": [19.0, 17.0, 14.0],
        "up": [0.0, 1.0, 0.0],
        "fov_y_deg": 41.0,
    }


def _task(
    *,
    tf: list[dict[str, float]],
    camera: dict[str, Any],
    edits: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "volume": {
            "path": "../volume.npy",
            "dims": [40, 36, 32],
            "spacing_mm": [0.9, 0.9, 1.1],
            "scalar_domain": {"min": 0.0, "max": 1.0},
        },
        "transfer_function": {"points": tf},
        "camera": camera,
        "output": {"width": 48, "height": 48},
        "blocks": {"width": 8, "height": 8},
        "concurrency": 4,
        "memory_budget_bytes": 16 * 1024 * 1024,
        "sampling_step_mm": 0.45,
        "background_gray": 0.0,
        "edits": edits,
        "state_dir": "../state",
        "outputs": {
            "gray": "../out/gray.npy",
            "depth": "../out/depth.npy",
            "report": "../out/report.json",
        },
    }


def generate_volume(seed: int = 20261119) -> np.ndarray:
    nx, ny, nz = 40, 36, 32
    x = np.arange(nx, dtype=np.float64)[None, None, :]
    y = np.arange(ny, dtype=np.float64)[None, :, None]
    z = np.arange(nz, dtype=np.float64)[:, None, None]
    rng = np.random.default_rng(seed)

    main = 0.82 * np.exp(
        -(
            ((x - 25.0) ** 2) / 120.0
            + ((y - 18.0) ** 2) / 95.0
            + ((z - 17.0) ** 2) / 110.0
        )
    )
    side = 0.58 * np.exp(
        -(
            ((x - 13.0) ** 2) / 45.0
            + ((y - 24.0) ** 2) / 60.0
            + ((z - 12.0) ** 2) / 38.0
        )
    )
    shells = 0.20 * np.exp(
        -(
            ((x - 28.0) ** 2 + (y - 12.0) ** 2 + (z - 22.0) ** 2 - 28.0) ** 2
        )
        / 260.0
    )
    noise = 0.045 * rng.standard_normal((nz, ny, nx))
    data = main + side + shells + noise
    data[data < 0.035] = 0.0
    data[:, :, :5] = 0.0
    data[:5, :, :] = 0.0
    data[29:, 0:10, 0:12] = 0.0
    return np.clip(data, 0.0, 1.0).astype(np.float64)


def generate_samples(samples_dir: str | Path, seed: int = 20261119) -> dict[str, Any]:
    root = Path(samples_dir).resolve()
    tasks_dir = root / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    volume = generate_volume(seed)
    volume_path = root / "volume.npy"
    np.save(volume_path, volume, allow_pickle=False)

    region_edit = {
        "center": [24, 18, 18],
        "size": [7, 5, 7],
        "value": 0.95,
    }
    second_edit = {
        "center": [15, 21, 14],
        "size": [5, 5, 5],
        "value": 0.08,
    }
    tasks: list[tuple[str, dict[str, Any]]] = [
        ("00_initial.json", _task(tf=_transfer_a(), camera=_camera_a(), edits=[])),
        (
            "01_region_edit.json",
            _task(tf=_transfer_a(), camera=_camera_a(), edits=[region_edit]),
        ),
        (
            "02_transfer_edit.json",
            _task(tf=_transfer_b(), camera=_camera_a(), edits=[region_edit]),
        ),
        (
            "03_camera_edit.json",
            _task(tf=_transfer_b(), camera=_camera_b(), edits=[region_edit]),
        ),
        (
            "04_combined.json",
            _task(tf=_transfer_c(), camera=_camera_c(), edits=[region_edit, second_edit]),
        ),
    ]
    for name, payload in tasks:
        (tasks_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "seed": seed,
        "volume": "volume.npy",
        "volume_shape_nz_ny_nx": list(volume.shape),
        "tasks": [f"tasks/{name}" for name, _ in tasks],
        "pipeline_order": [f"tasks/{name}" for name, _ in tasks],
        "shared_state_dir": "state",
        "shared_output_dir": "out",
    }
    (root / "samples.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _new_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def verify_samples(samples_dir: str | Path) -> dict[str, Any]:
    root = Path(samples_dir).resolve()
    task_paths = sorted((root / "tasks").glob("*.json"))
    if not task_paths:
        raise FileNotFoundError(f"no sample tasks under {root / 'tasks'}")

    verification_root = root / "verification"
    _new_directory(verification_root)
    incremental_state = verification_root / "incremental_state"
    incremental_output = verification_root / "incremental_out"
    incremental_state.mkdir(parents=True, exist_ok=True)
    incremental_output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    all_equal = True
    for index, task_path in enumerate(task_paths):
        incremental = load_task(
            task_path,
            state_dir_override=incremental_state,
            output_dir_override=incremental_output,
        )
        incremental_result = run_task(incremental)

        full_state = verification_root / "full" / f"{index:02d}" / "state"
        full_output = verification_root / "full" / f"{index:02d}" / "out"
        full_task = load_task(
            task_path,
            state_dir_override=full_state,
            output_dir_override=full_output,
        )
        full_result = run_task(full_task, force_full=True)

        gray_equal = bool(np.array_equal(incremental_result.gray, full_result.gray))
        depth_equal = bool(np.array_equal(incremental_result.depth, full_result.depth))
        shapes_equal = (
            incremental_result.gray.shape
            == incremental_result.depth.shape
            == full_result.gray.shape
            == full_result.depth.shape
            == (incremental.height, incremental.width)
        )
        summary = incremental_result.report["summary"]
        blocks = incremental_result.report["blocks"]
        category_counts: dict[str, int] = {}
        for block in blocks:
            for category in block["invalid_categories"]:
                category_counts[category] = category_counts.get(category, 0) + 1
        bbox_only = [
            block
            for block in blocks
            if block["volume_edit"]["changed_voxels_in_bounds"] > 0
            and block["volume_edit"]["rays_hitting_changed_voxels"] == 0
        ]
        volume_affected = [block for block in blocks if block["volume_edit"]["affected"]]
        volume_unaffected = [block for block in blocks if not block["volume_edit"]["affected"]]
        nonempty = [block for block in blocks if block["has_nonzero_samples"]]
        zero = [block for block in blocks if not block["has_nonzero_samples"]]
        record = {
            "task": task_path.name,
            "incremental_vs_full_gray_equal": gray_equal,
            "incremental_vs_full_depth_equal": depth_equal,
            "strict_shapes_equal": shapes_equal,
            "reused_blocks": summary["reused_blocks"],
            "rerendered_blocks": summary["rerendered_blocks"],
            "invalid_blocks_by_change_criteria": summary[
                "invalid_blocks_by_change_criteria"
            ],
            "missing_before_render": summary["missing_before_render"],
            "persisted_recovery_blocks": summary["persisted_recovery_blocks"],
            "cache_missing_without_change_rule": summary[
                "cache_missing_without_change_rule"
            ],
            "category_counts": dict(sorted(category_counts.items())),
            "bbox_overlap_without_ray_blocks": len(bbox_only),
            "bbox_overlap_without_ray_all_reused": all(
                block["cache"]["reused"] for block in bbox_only
            ),
            "volume_affected_blocks": len(volume_affected),
            "volume_affected_all_rerendered": all(
                block["cache"]["rerendered"] and "volume_edit" in block["invalid_categories"]
                for block in volume_affected
            ),
            "volume_unaffected_all_reused": all(
                block["cache"]["reused"] and "volume_edit" not in block["invalid_categories"]
                for block in volume_unaffected
            ),
            "nonempty_blocks": len(nonempty),
            "zero_blocks": len(zero),
            "transfer_nonempty_all_rerendered": all(
                block["cache"]["rerendered"]
                and "transfer_function" in block["invalid_categories"]
                for block in nonempty
            ),
            "transfer_zero_all_reused": all(
                block["cache"]["reused"] and "transfer_function" not in block["invalid_categories"]
                for block in zero
            ),
            "gray_digest": incremental_result.report["outputs"]["gray"]["digest"],
            "depth_digest": incremental_result.report["outputs"]["depth"]["digest"],
        }
        records.append(record)
        all_equal = all_equal and gray_equal and depth_equal and shapes_equal

    region = next((item for item in records if item["task"] == "01_region_edit.json"), None)
    transfer = next(
        (item for item in records if item["task"] == "02_transfer_edit.json"), None
    )
    camera = next((item for item in records if item["task"] == "03_camera_edit.json"), None)
    total_blocks = (
        records[0]["reused_blocks"] + records[0]["rerendered_blocks"] if records else 0
    )

    checks = {
        "every_incremental_image_matches_full_render": all_equal,
        "region_edit_rerendered_fewer_than_all_blocks": bool(
            region is not None
            and total_blocks is not None
            and region["rerendered_blocks"] < total_blocks
            and region["reused_blocks"] > 0
            and "volume_edit" in region["category_counts"]
            and region["volume_affected_all_rerendered"]
            and region["volume_unaffected_all_reused"]
            and region["bbox_overlap_without_ray_blocks"] > 0
            and region["bbox_overlap_without_ray_all_reused"]
        ),
        "transfer_edit_uses_nonzero_block_rule": bool(
            transfer is not None
            and "transfer_function" in transfer["category_counts"]
            and transfer["nonempty_blocks"] > 0
            and transfer["zero_blocks"] > 0
            and transfer["transfer_nonempty_all_rerendered"]
            and transfer["transfer_zero_all_reused"]
        ),
        "camera_edit_invalidates_all_blocks": bool(
            camera is not None
            and total_blocks is not None
            and camera["rerendered_blocks"] == total_blocks
            and "camera" in camera["category_counts"]
        ),
        "recovery_accounting_is_consistent": all(
            item["missing_before_render"]
            == item["invalid_blocks_by_change_criteria"]
            - item["persisted_recovery_blocks"]
            + item["cache_missing_without_change_rule"]
            for item in records
        ),
    }
    result = {
        "ok": all(checks.values()),
        "seed": json.loads((root / "samples.json").read_text(encoding="utf-8"))["seed"],
        "tasks": records,
        "checks": checks,
    }
    (root / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
