"""Repeatable fixed-sequence incremental diagnostics demonstration."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .diagnostics import manifest_from_state
from .engine import IncrementalEngine
from .errors import E_DEPENDENCY_STALE, fail
from .hashing import atomic_write_bytes, canonical_json_bytes, sha256_bytes
from .rir import read_pcm16_wav_bytes
from .store import WorkspaceStore


FIXED_SEQUENCE: tuple[dict[str, Any], ...] = (
    {"source.s0.position": [1.15, 0.95, 1.28]},
    {"wall.xmax.alpha": 0.37},
    {"receiver.sample_density": 192, "reconstruction.order": 4},
    {"wall.ymin.alpha": 0.34},
    {"source.s1.position": [4.70, 3.65, 1.33]},
)


def build_step_manifest(engine: IncrementalEngine, steps: list[dict[str, Any]]) -> bytes:
    payload = {
        "schema": 1,
        "sequence": [copy.deepcopy(step["patch"]) for step in steps],
        "steps": steps,
        "final_scene_hash": engine.state["scene_hash"],
        "final_revision": engine.scene["revision"],
    }
    return canonical_json_bytes(payload)


def run_demo(
    base_dir: Path | str = "artifacts/demo",
    *,
    print_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    base = Path(base_dir).resolve()
    run_a = base / "run_a"
    run_b = base / "run_b"
    full_final = base / "full_final"
    for path in (run_a, run_b, full_final):
        if path.exists():
            shutil.rmtree(path)
    base.mkdir(parents=True, exist_ok=True)

    manifest_a, steps_a, timings_a, engine_a = _run_sequence(run_a, label="run-a", print_fn=print_fn)
    manifest_b, steps_b, timings_b, _engine_b = _run_sequence(run_b, label="run-b", print_fn=print_fn)
    identical = manifest_a == manifest_b
    if not identical:
        fail(E_DEPENDENCY_STALE, "two identical parameter sequences produced different manifests")

    final_scene = copy.deepcopy(engine_a.scene)
    final_versions = dict(engine_a.versions)
    full_store = WorkspaceStore(full_final)
    full_store.ensure()
    full_store.save_scene_document(final_scene, final_versions)
    full_engine = IncrementalEngine.open(full_final)
    manifest_full = full_engine.manifest_bytes()
    manifest_incremental_final = engine_a.manifest_bytes()
    incremental_matches_full = manifest_incremental_final == manifest_full
    if not incremental_matches_full:
        fail(E_DEPENDENCY_STALE, "final incremental manifest differs from a fresh full rebuild")

    node_hash_mismatches = sorted(
        node_id
        for node_id in engine_a.state["nodes"]
        if engine_a.state["nodes"][node_id]["content_hash"] != full_engine.state["nodes"][node_id]["content_hash"]
    )
    tile_hash_mismatches = sorted(
        node_id
        for node_id in engine_a.state["nodes"]
        if node_id.startswith("tile:") and engine_a.state["nodes"][node_id]["content_hash"] != full_engine.state["nodes"][node_id]["content_hash"]
    )
    rir_exact = _rir_wav_equal(run_a, full_final)
    numeric_differences = _numeric_differences(engine_a, full_engine)
    tolerance = {
        "rir_pcm16_samples": 0,
        "spherical_coefficients_float64": 0.0,
        "energy_float64": 0.0,
        "tile_bytes": 0,
    }
    acceptance = all(
        [
            identical,
            incremental_matches_full,
            not node_hash_mismatches,
            not tile_hash_mismatches,
            rir_exact,
            all(value == 0.0 for value in numeric_differences.values()),
        ]
    )

    atomic_write_bytes(base / "manifest_run_a.json", manifest_a)
    atomic_write_bytes(base / "manifest_run_b.json", manifest_b)
    atomic_write_bytes(base / "manifest_incremental_final.json", manifest_incremental_final)
    atomic_write_bytes(base / "manifest_full.json", manifest_full)
    verification = {
        "schema": 1,
        "accepted": acceptance,
        "identical_sequential_manifests": identical,
        "incremental_manifest_equals_full_rebuild": incremental_matches_full,
        "node_hash_mismatches": node_hash_mismatches,
        "tile_hash_mismatches": tile_hash_mismatches,
        "four_channel_rir_exact_against_full": rir_exact,
        "numeric_max_abs_differences": numeric_differences,
        "declared_tolerance": tolerance,
        "sequence_manifest_sha256": sha256_bytes(manifest_a),
        "incremental_final_manifest_sha256": sha256_bytes(manifest_incremental_final),
        "full_manifest_sha256": sha256_bytes(manifest_full),
        "final_revision": int(engine_a.scene["revision"]),
        "final_node_count": len(engine_a.state["nodes"]),
        "run_a_step_timings_ms": timings_a,
        "run_b_step_timings_ms": timings_b,
        "outputs": {
            "run_a_workspace": str(run_a),
            "run_b_workspace": str(run_b),
            "full_final_workspace": str(full_final),
            "four_channel_wav_run_a": str(run_a / "outputs" / "room_rir_48k_16bit_4ch.wav"),
            "four_channel_wav_full": str(full_final / "outputs" / "room_rir_48k_16bit_4ch.wav"),
            "projection_png_run_a": str(run_a / "outputs" / "equirectangular_projection.png"),
        },
    }
    atomic_write_bytes(base / "verification.json", canonical_json_bytes(verification))

    print_fn("")
    print_fn(f"sequential manifests identical: {identical}")
    print_fn(f"incremental equals full rebuild: {incremental_matches_full}")
    print_fn(f"sequence manifest sha256: {verification['sequence_manifest_sha256']}")
    print_fn(f"incremental final manifest sha256: {verification['incremental_final_manifest_sha256']}")
    print_fn(f"full manifest sha256: {verification['full_manifest_sha256']}")
    print_fn(f"tile hash mismatches: {len(tile_hash_mismatches)}; node hash mismatches: {len(node_hash_mismatches)}")
    print_fn(f"numeric max abs differences: {numeric_differences}")
    print_fn(f"acceptance: {'PASS' if acceptance else 'FAIL'}")
    if not acceptance:
        fail(E_DEPENDENCY_STALE, "demo acceptance checks failed", verification=verification)
    return verification


def _run_sequence(
    workspace: Path,
    *,
    label: str,
    print_fn: Callable[[str], None],
) -> tuple[bytes, list[dict[str, Any]], list[dict[str, Any]], IncrementalEngine]:
    engine = IncrementalEngine.open(workspace)
    print_fn(f"[{label}] initial revision {engine.scene['revision']} nodes {len(engine.state['nodes'])}")
    steps: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for step_number, patch in enumerate(FIXED_SEQUENCE, start=1):
        report = engine.submit(patch, expected_revision=engine.scene["revision"])
        tile_reused = sum(1 for node_id in report["reused_nodes"] if node_id.startswith("tile:"))
        tile_recomputed = sum(1 for node_id in report["recomputed_nodes"] if node_id.startswith("tile:"))
        timings.append({"step": step_number, **report["timings_ms"]})
        print_fn(
            f"[{label}] step {step_number} rev {report['revision']} patch={patch} "
            f"closure={report['closure_size']} reused={report['reused_count']} recomputed={report['recomputed_count']} "
            f"tiles_reused={tile_reused} tiles_recomputed={tile_recomputed} time_ms={report['timings_ms']['total']:.3f}"
        )
        step_payload = {
            "step": step_number,
            "revision": int(report["revision"]),
            "patch": copy.deepcopy(patch),
            "closure_size": int(report["closure_size"]),
            "reused_count": int(report["reused_count"]),
            "recomputed_count": int(report["recomputed_count"]),
            "tiles_reused": tile_reused,
            "tiles_recomputed": tile_recomputed,
            "closure": sorted(report["closure"]),
            "reused_nodes": sorted(report["reused_nodes"]),
            "node_hashes": {
                node_id: engine.state["nodes"][node_id]["content_hash"]
                for node_id in sorted(engine.state["nodes"])
            },
            "tile_hashes": {
                node_id: engine.state["nodes"][node_id]["content_hash"]
                for node_id in sorted(engine.state["nodes"])
                if node_id.startswith("tile:")
            },
        }
        steps.append(step_payload)
    manifest = build_step_manifest(engine, steps)
    return manifest, steps, timings, engine


def _rir_wav_equal(left_root: Path, right_root: Path) -> bool:
    left = (left_root / "outputs" / "room_rir_48k_16bit_4ch.wav").read_bytes()
    right = (right_root / "outputs" / "room_rir_48k_16bit_4ch.wav").read_bytes()
    left_samples, left_rate, left_channels = read_pcm16_wav_bytes(left)
    right_samples, right_rate, right_channels = read_pcm16_wav_bytes(right)
    return (
        left_rate == right_rate == 48000
        and left_channels == right_channels == 4
        and np.array_equal(left_samples, right_samples)
    )


def _numeric_differences(incremental: IncrementalEngine, full: IncrementalEngine) -> dict[str, float]:
    differences = {"sph_float64": 0.0, "energy_float64": 0.0, "tile_bytes": 0.0}
    for node_id in sorted(incremental.state["nodes"]):
        left_node = incremental.state["nodes"][node_id]
        right_node = full.state["nodes"][node_id]
        left_blob = incremental.store.read_blob(left_node["artifact_hash"], node_type=left_node["type"])
        right_blob = full.store.read_blob(right_node["artifact_hash"], node_type=right_node["type"])
        if left_node["type"] == "sph":
            left_values = np.frombuffer(left_blob, dtype="<f8")
            right_values = np.frombuffer(right_blob, dtype="<f8")
            differences["sph_float64"] = max(
                differences["sph_float64"],
                float(np.max(np.abs(left_values - right_values))) if left_values.size else 0.0,
            )
        elif left_node["type"] == "energy":
            left_values = np.frombuffer(left_blob, dtype="<f8")
            right_values = np.frombuffer(right_blob, dtype="<f8")
            differences["energy_float64"] = max(
                differences["energy_float64"],
                float(np.max(np.abs(left_values - right_values))) if left_values.size else 0.0,
            )
        elif left_node["type"] == "tile":
            differences["tile_bytes"] = max(
                differences["tile_bytes"],
                float(abs(len(left_blob) - len(right_blob)) or (0.0 if left_blob == right_blob else 1.0)),
            )
    return differences
