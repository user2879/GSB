from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .canonical import canonical_line, write_json
from .diagnostics import compare_or_raise
from .errors import ErrorCode, WorkbenchError
from .heightfield import DEFAULT_SEED, TerrainConfig, load_terrain, validate_height_range
from .hiz import build_hiz, validate_depth_bias
from .quadtree import build_quadtree, load_quadtree_manifest
from .streaming import StreamingBudgets, StreamingManager


def _capture(expected: ErrorCode, function: Callable[[], Any]) -> dict[str, Any]:
    try:
        function()
    except WorkbenchError as exc:
        return {
            "expected_code": expected.value,
            "observed_code": exc.code.value,
            "message": exc.message,
            "context": exc.context,
            "passed": exc.code == expected,
            "raised": True,
        }
    except Exception as exc:
        return {
            "expected_code": expected.value,
            "observed_code": type(exc).__name__,
            "message": str(exc),
            "context": {},
            "passed": False,
            "raised": True,
        }
    return {
        "expected_code": expected.value,
        "observed_code": None,
        "message": "no error was raised",
        "context": {},
        "passed": False,
        "raised": False,
    }


def _resolve_source_seed(root: Path, seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    candidates = sorted((root / "manifests").glob("terrain_257_*.terrain.json"))
    if len(candidates) == 1:
        metadata = json.loads(candidates[0].read_text(encoding="utf-8"))
        return int(metadata["seed"])
    return int(DEFAULT_SEED)


def run_error_code_injections(
    source_workspace: str | Path,
    output_path: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    root = Path(source_workspace)
    resolved_seed = _resolve_source_seed(root, seed)
    config = TerrainConfig(257, resolved_seed)
    heights, terrain_metadata = load_terrain(root, config)
    manifest_data = load_quadtree_manifest(root, config.size, config.seed)
    manifest = build_quadtree(heights, config.seed, terrain_metadata["content_sha256"])
    if manifest_data.get("content_sha256") != manifest.content_sha256:
        raise WorkbenchError(
            ErrorCode.E_BLOCK_MISSING,
            "anomaly source quadtree does not match regenerated manifest",
            {"source_workspace": str(root)},
        )
    first_block = sorted(node.block_id for node in manifest.leaves if node.block_id is not None)[0]
    pyramid = build_hiz(np.zeros((8, 8), dtype=np.float32))

    e_height = _capture(
        ErrorCode.E_HEIGHT_RANGE,
        lambda: validate_height_range(np.array([0, 70000], dtype=np.uint32)),
    )
    e_block = _capture(
        ErrorCode.E_BLOCK_MISSING,
        lambda: manifest.get_block("B999_99999_99999"),
    )
    e_hiz = _capture(
        ErrorCode.E_HIZ_LEVEL,
        lambda: pyramid.get_level(99),
    )
    e_bias = _capture(
        ErrorCode.E_DEPTH_BIAS,
        lambda: validate_depth_bias(-0.01),
    )

    tiny_manager = StreamingManager(
        heights,
        manifest,
        StreamingBudgets(memory_bytes=1, vram_bytes=1, prefetch_radius=0),
    )
    e_budget = _capture(
        ErrorCode.E_BUDGET_DENIED,
        lambda: tiny_manager.step(0, (first_block,), (), "anomaly-budget"),
    )

    leak_manager = StreamingManager(
        heights,
        manifest,
        StreamingBudgets(memory_bytes=16 * 1024, vram_bytes=16 * 1024, prefetch_radius=0),
    )
    leak_manager.step(0, (first_block,), (), "anomaly-leak")
    leak_manager.inject_resident_leak()

    def leak_check() -> None:
        leak_manager.audit()

    e_leak = _capture(ErrorCode.E_RESIDENT_LEAK, leak_check)
    leak_manager.close(1, "anomaly-leak-cleanup")

    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        left = temporary_path / "left.jsonl"
        right = temporary_path / "right.jsonl"
        decision_left = {
            "block_id": first_block,
            "status": "VISIBLE",
            "errors": {"total": 0.001},
            "parameter_stamp": "track-a",
            "input": {"depth_bias": 0.00001},
        }
        decision_right = {
            "block_id": first_block,
            "status": "OCCLUDED",
            "errors": {"total": 0.002},
            "parameter_stamp": "track-b",
            "input": {"depth_bias": 0.00002},
        }
        base_frame = {
            "frame": 0,
            "camera": {"eye": [0, 0, 0], "target": [0, 0, 1]},
            "parameter_stamp": "frame-a",
        }
        left.write_bytes(
            canonical_line({**base_frame, "decisions": [decision_left]}).encode("utf-8")
        )
        right.write_bytes(
            canonical_line(
                {
                    **base_frame,
                    "parameter_stamp": "frame-b",
                    "decisions": [decision_right],
                }
            ).encode("utf-8")
        )
        e_track = _capture(
            ErrorCode.E_TRACK_DIVERGE,
            lambda: compare_or_raise(left, right),
        )

    checks = {
        ErrorCode.E_HEIGHT_RANGE.value: e_height,
        ErrorCode.E_BLOCK_MISSING.value: e_block,
        ErrorCode.E_HIZ_LEVEL.value: e_hiz,
        ErrorCode.E_DEPTH_BIAS.value: e_bias,
        ErrorCode.E_BUDGET_DENIED.value: e_budget,
        ErrorCode.E_RESIDENT_LEAK.value: e_leak,
        ErrorCode.E_TRACK_DIVERGE.value: e_track,
    }
    report = {
        "schema_version": 1,
        "source_size": config.size,
        "source_seed": int(config.seed),
        "checks": checks,
        "all_passed": all(check["passed"] for check in checks.values()),
    }
    if output_path is not None:
        write_json(output_path, report)
    return report
