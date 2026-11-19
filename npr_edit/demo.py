"""Repeatable end-to-end demonstration and two-run determinism verifier."""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .engine import EditRecord, NprEngine
from .errors import (
    E_CLOSURE_MISMATCH,
    E_FORMAT_ENDIAN,
    E_FORMAT_INDEX,
    E_GL_CONTEXT,
    E_OUTLINE_DEGENERATE,
    E_SCHEMA_VERSION,
    E_SNAPSHOT_STALE,
    E_STROKE_SAMPLE,
    NPRError,
)
from .formats import (
    MIGRATION_DEFAULTS,
    migrate_project,
    roundtrip_report,
    scene_to_skeleton_binary,
    skeleton_binary_to_project,
)
from .generator import DEFAULT_SEED, build_element_hash_manifest, generate_scene
from .geometry import GeometryRecord, flatten_element
from .gl_backend import open_offscreen_gl
from .model import Point, Segment, sampling_from_dict
from .persistence import PersistenceStore
from .raster import render_snapshot, write_bgra, write_bmp
from .snapshot import RenderSnapshot
from .strategy import StrategySelector, scene_render_stats
from .util import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    sorted_hash_pairs,
)

TIERS = (200, 2000, 20000)
APP_DIRS = ("generated", "rendered", "report", "state")


def _reset_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for name in APP_DIRS:
        target = workspace / name
        if target.exists():
            shutil.rmtree(target)
    for name in ("generated", "rendered", "report"):
        (workspace / name).mkdir(parents=True, exist_ok=True)


def _map_digest(mapping: Any) -> str:
    return sha256_json(dict(sorted(mapping.items())))


def _error_probe(code: str, stage: str, function: Callable[[], Any]) -> dict[str, Any]:
    try:
        function()
    except NPRError as exc:
        return {
            "stage": stage,
            "expected": code,
            "actual": exc.code,
            "triggered": exc.code == code,
        }
    return {"stage": stage, "expected": code, "actual": None, "triggered": False}


def _write_ledger(path: Path, snapshot: RenderSnapshot, scene_hash: str) -> dict[str, Any]:
    payload = {
        "format": "npr-render-ledger-v1",
        "scene_hash": scene_hash,
        "version": snapshot.version,
        "mode": snapshot.mode,
        "snapshot_hash": snapshot.snapshot_hash,
        "image_hash": snapshot.image_hash,
        "layer_hashes": dict(sorted(snapshot.layer_hashes.items())),
        "skeleton_hashes": dict(sorted(snapshot.skeleton_hashes.items())),
        "render_hashes": dict(sorted(snapshot.element_hashes.items())),
        "geometry_hashes": dict(sorted(snapshot.geometry_hashes.items())),
    }
    atomic_write_json(path, payload)
    return {
        "path": path.name,
        "file_hash": sha256_bytes(canonical_json_bytes(payload) + b"\n"),
        "layer_hash_digest": _map_digest(snapshot.layer_hashes),
        "skeleton_hash_digest": _map_digest(snapshot.skeleton_hashes),
        "render_hash_digest": _map_digest(snapshot.element_hashes),
        "geometry_hash_digest": _map_digest(snapshot.geometry_hashes),
    }


def _legacy_project(scene: Any) -> dict[str, Any]:
    legacy = copy.deepcopy(scene.as_project_dict())
    legacy["schema_version"] = 1
    legacy.pop("sampling", None)
    for layer in legacy.get("layers", []):
        layer.pop("order", None)
    for element in legacy.get("elements", []):
        element.pop("style", None)
        element.pop("sampling", None)
        element.pop("layer_order", None)
    return legacy


def _strategy_cache_probe(engines: dict[int, NprEngine]) -> dict[str, Any]:
    selector = StrategySelector()
    decisions = {}
    for tier in TIERS:
        stats = scene_render_stats(
            engines[tier].scene,
            {element_id: record.flat for element_id, record in engines[tier].geometry.records.items()},
            engines[tier].viewport,
        )
        decisions[tier] = selector.select(tier, stats)
    calls_after_first_pass = selector.selection_calls
    hits_after_first_pass = selector.cache_hits
    for tier in (200, 20000):
        stats = decisions[tier].stats
        selector.select(tier, stats)
    calls_after_reuse = selector.selection_calls
    hits_after_reuse = selector.cache_hits
    before = {decision.fingerprint for decision in selector.cache_entries.values()}
    budget = dict(engines[2000].budget or {})
    budget["frame_ms"] = float(budget.get("frame_ms", 40.0)) + 5.0
    stats_2000 = decisions[2000].stats
    selector.select(2000, stats_2000, budget)
    after = {decision.fingerprint for decision in selector.cache_entries.values()}
    changed = sorted({
        decision.tier
        for decision in selector.cache_entries.values()
        if decision.fingerprint not in before
    })
    return {
        "first_pass_selection_calls": calls_after_first_pass,
        "first_pass_cache_hits": hits_after_first_pass,
        "selective_reuse_calls": calls_after_reuse,
        "selective_reuse_cache_hits": hits_after_reuse,
        "budget_change_selection_calls": selector.selection_calls,
        "budget_change_affected_tiers": changed,
        "only_requested_tier_reselected": changed == [2000],
        "chosen": {str(tier): decisions[tier].chosen for tier in TIERS},
    }


def _run_gl_probe(require_gl: bool) -> tuple[Any | None, dict[str, Any] | None, list[dict[str, Any]]]:
    degradations: list[dict[str, Any]] = []
    try:
        backend = open_offscreen_gl()
        return backend, {"available": True, "backend": type(backend.platform_backend).__name__}, degradations
    except NPRError as exc:
        if exc.code != E_GL_CONTEXT:
            raise
        if require_gl:
            raise
        degradations.append(
            {
                "code": E_GL_CONTEXT,
                "stage": "offscreen_gl_probe",
                "action": "software_bgra_fallback",
                "message": "offscreen OpenGL context unavailable; software BGRA path used",
            }
        )
        return None, {"available": False, "code": exc.code}, degradations


def _render_tier(
    scene: Any,
    tier: int,
    viewport: tuple[int, int],
    gl_backend: Any | None,
    geometry_records: dict[str, GeometryRecord],
) -> tuple[NprEngine, dict[str, Any]]:
    engine = NprEngine(
        scene,
        viewport=viewport,
        tier=tier,
        gl_backend=gl_backend,
        geometry_records=geometry_records,
    )
    full = engine.full_render(version=1)
    first_snapshot = full.snapshot
    strategy = engine.strategy
    assert strategy is not None
    target_index = min(5, len(scene.elements) - 1)
    target = scene.elements[target_index]
    segment = target.segments[0]
    original = segment.points[0]
    edit = engine.edit_control_point(
        target.id,
        0,
        0,
        original.x + 0.75,
        original.y + 0.25,
        expected_version=first_snapshot.version,
        expected_hash=first_snapshot.snapshot_hash,
    )
    post_edit_snapshot = engine.current_snapshot
    unchanged_outside = all(
        post_edit_snapshot.element_hashes[element_id] == first_snapshot.element_hashes[element_id]
        for element_id in post_edit_snapshot.element_hashes
        if element_id != target.id
    )
    mode_records: list[dict[str, Any]] = []
    if tier == 200:
        for mode in ("sketch", "cartoon"):
            geometry_hashes_before = dict(engine.geometry.geometry_hashes())
            skeleton_hashes_before = dict(engine.current_snapshot.skeleton_hashes)
            switch = engine.set_style_mode(
                mode,
                expected_version=engine.current_snapshot.version,
                expected_hash=engine.current_snapshot.snapshot_hash,
            )
            mode_records.append(
                {
                    "record": switch.as_dict(),
                    "geometry_hashes_shared": geometry_hashes_before == dict(engine.geometry.geometry_hashes()),
                    "skeleton_hashes_shared": skeleton_hashes_before == dict(engine.current_snapshot.skeleton_hashes),
                }
            )
    final_snapshot = engine.current_snapshot
    payload = {
        "tier": tier,
        "full_render": {
            "version": first_snapshot.version,
            "snapshot_hash": first_snapshot.snapshot_hash,
            "image_hash": first_snapshot.image_hash,
            "layer_hash_digest": _map_digest(first_snapshot.layer_hashes),
            "strategy": strategy.as_dict(),
            "gl_output": gl_backend is not None,
        },
        "incremental_edit": edit.as_dict(),
        "incremental_closure_exact": (
            set(edit.recomputed_elements) == set(edit.element_ids)
            and set(edit.recomputed_layers) == set(edit.layer_ids)
            and edit.geometry_rebuilt == edit.element_ids
            and edit.skeleton_hashes_unchanged_outside_closure
            and edit.render_hashes_unchanged_outside_closure
        ),
        "unchanged_render_hashes_outside_closure": unchanged_outside,
        "mode_switches": mode_records,
        "all_modes_share_geometry": all(
            item["geometry_hashes_shared"] and item["skeleton_hashes_shared"] for item in mode_records
        ) if mode_records else True,
        "final": {
            "version": final_snapshot.version,
            "mode": final_snapshot.mode,
            "snapshot_hash": final_snapshot.snapshot_hash,
            "image_hash": final_snapshot.image_hash,
        },
    }
    return engine, payload


def run_crash_worker(workspace: Path | str) -> None:
    """Child entrypoint: stage an uncommitted edit, then die without cleanup."""

    store = PersistenceStore(workspace)
    recovered = store.recover()
    if recovered is None:
        os._exit(41)
    engine = NprEngine.from_snapshot(recovered.scene, recovered.snapshot)
    element = recovered.scene.elements[0]
    point = element.segments[0].points[0]
    engine.edit_control_point(
        element.id,
        0,
        0,
        point.x + 0.5,
        point.y - 0.25,
        expected_version=recovered.version,
        expected_hash=recovered.snapshot.snapshot_hash,
    )
    store.write_uncommitted_generation(engine.scene, engine.current_snapshot)
    os._exit(137)


def _force_termination(workspace: Path, repo_root: Path) -> dict[str, Any]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + existing if existing else "")
    command = [sys.executable, "-m", "npr_edit", "_crash-worker", str(workspace)]
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return {
        "command_kind": "subprocess_os_exit",
        "exit_code": int(completed.returncode),
        "force_terminated": completed.returncode != 0,
        "stdout_empty": completed.stdout == "",
        "stderr_empty": completed.stderr == "",
    }


def _persistence_sequence(workspace: Path, engine: NprEngine, repo_root: Path) -> dict[str, Any]:
    store = PersistenceStore(workspace)
    baseline_scene = engine.scene
    baseline_snapshot = engine.current_snapshot
    manifest = store.commit(baseline_scene, baseline_snapshot)
    crash = _force_termination(workspace, repo_root)
    recovered = store.recover()
    if recovered is None:
        raise NPRError(E_SNAPSHOT_STALE, "recovery returned no committed state")
    recovery_consistent = (
        recovered.recovered_scene_hash == baseline_scene.content_hash
        and recovered.recovered_snapshot_hash == baseline_snapshot.snapshot_hash
        and recovered.version == baseline_snapshot.version
    )
    process_engine = NprEngine.from_snapshot(recovered.scene, recovered.snapshot)
    target = recovered.scene.elements[0]
    continued = process_engine.edit_stroke_width(
        target.id,
        target.stroke_width + 0.125,
        expected_version=recovered.version,
        expected_hash=recovered.snapshot.snapshot_hash,
    )
    if continued.old_version != recovered.version:
        raise NPRError(E_SNAPSHOT_STALE, "continued edit closure did not use the recovered version stamp")
    close_marker = store.close(process_engine.scene, process_engine.current_snapshot)
    final_recovery = store.recover()
    if final_recovery is None:
        raise NPRError(E_SNAPSHOT_STALE, "final close could not be recovered")
    return {
        "baseline": {
            "generation": manifest["generation"],
            "version": baseline_snapshot.version,
            "scene_hash": baseline_scene.content_hash,
            "snapshot_hash": baseline_snapshot.snapshot_hash,
        },
        "force_termination": crash,
        "recovered": {
            **recovered.as_dict(),
            "hashes_match_baseline": recovery_consistent,
        },
        "continued_incremental_after_recovery": continued.as_dict(),
        "continued_version_uses_recovered_stamp": continued.old_version == recovered.version,
        "close_marker": close_marker,
        "recovery_after_normal_close": final_recovery.as_dict(),
        "normal_close_verified": (
            final_recovery.clean_shutdown
            and final_recovery.version == process_engine.current_snapshot.version
            and final_recovery.recovered_snapshot_hash == process_engine.current_snapshot.snapshot_hash
        ),
    }


def run_demo(
    workspace: Path | str,
    *,
    seed: int = DEFAULT_SEED,
    viewport: tuple[int, int] = (256, 192),
    require_gl: bool = False,
    reset: bool = True,
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    repo_root = Path(__file__).resolve().parent.parent
    if reset:
        _reset_workspace(root)
    else:
        root.mkdir(parents=True, exist_ok=True)
        for name in ("generated", "rendered", "report"):
            (root / name).mkdir(parents=True, exist_ok=True)
    store = PersistenceStore(root)
    store.check_writable()

    gl_backend, gl_probe, degradations = _run_gl_probe(require_gl)
    scenes: dict[int, Any] = {}
    generate_report: list[dict[str, Any]] = []
    format_report: list[dict[str, Any]] = []
    geometry_records_by_tier: dict[int, dict[str, GeometryRecord]] = {}
    try:
        # 1. Deterministic generation and per-element hash manifests.
        for tier in TIERS:
            scene = generate_scene(tier, seed=seed)
            second = generate_scene(tier, seed=seed)
            deterministic = scene.project_bytes == second.project_bytes and scene.content_hash == second.content_hash
            scene_path = root / "generated" / f"scene-{tier}.json"
            manifest = build_element_hash_manifest(scene, tier)
            manifest_path = root / "generated" / f"element-hashes-{tier}.json"
            atomic_write_bytes(scene_path, scene.project_bytes + b"\n")
            atomic_write_json(manifest_path, manifest)
            scenes[tier] = scene
            generate_report.append(
                {
                    "tier": tier,
                    "element_count": len(scene.elements),
                    "layer_count": len(scene.layers),
                    "scene_hash": scene.content_hash,
                    "element_hash_manifest_hash": sha256_bytes(canonical_json_bytes(manifest) + b"\n"),
                    "scene_file_hash": sha256_bytes(scene.project_bytes + b"\n"),
                    "byte_identical_on_regeneration": deterministic,
                }
            )

        # 2. Binary skeleton round trips plus endian/index/width variants.
        binary_by_tier: dict[int, bytes] = {}
        for tier in TIERS:
            scene = scenes[tier]
            flat_cache = {}
            geometry_records: dict[str, GeometryRecord] = {}
            for element in scene.elements:
                flat, skeleton = flatten_element(
                    element,
                    target_viewport=viewport,
                    canvas_size=(scene.canvas_width, scene.canvas_height),
                )
                flat_cache[element.id] = (flat, skeleton)
                geometry_records[element.id] = GeometryRecord(
                    element_hash=flat.element_hash,
                    tolerance=flat.tolerance,
                    flat=flat,
                    skeleton=skeleton,
                )
            geometry_records_by_tier[tier] = geometry_records
            binary = scene_to_skeleton_binary(
                scene,
                endian="little",
                index_width=4,
                width_encoding="f32",
                viewport=viewport,
                flat_geometry=flat_cache,
            )
            binary_by_tier[tier] = binary
            binary_path = root / "generated" / f"scene-{tier}.nprb"
            atomic_write_bytes(binary_path, binary)
            report = roundtrip_report(scene, binary)
            variant = None
            if tier == 200:
                variant_binary = scene_to_skeleton_binary(
                    scene,
                    endian="big",
                    index_width=2,
                    width_encoding="u16_fixed",
                    viewport=viewport,
                    flat_geometry=flat_cache,
                )
                variant_roundtrip = roundtrip_report(scene, variant_binary)
                variant = {
                    "endian_marker": variant_binary[8:10].decode("ascii"),
                    "index_width": variant_binary[10],
                    "width_encoding_code": variant_binary[13],
                    "binary_hash": variant_roundtrip["binary_hash"],
                    "within_tolerance": variant_roundtrip["within_tolerance"],
                }
            format_report.append(
                {
                    "tier": tier,
                    "binary_file": f"scene-{tier}.nprb",
                    "binary_bytes": len(binary),
                    "binary_hash": report["binary_hash"],
                    "roundtrip_scene_hash": report["roundtrip_scene_hash"],
                    "within_tolerance": report["within_tolerance"],
                    "roundtrip_exact_scene": report["roundtrip_exact_scene"],
                    "max_control_point_error": report["max_control_point_error"],
                    "max_stroke_width_error": report["max_stroke_width_error"],
                    "variant": variant,
                }
            )

        first_binary = binary_by_tier[200]
        bad_endian = bytearray(first_binary)
        bad_endian[8:10] = b"XX"
        bad_index = bytearray(first_binary)
        bad_index[10] = 3
        error_chain = [
            _error_probe(
                E_FORMAT_ENDIAN,
                "format_endian_marker",
                lambda: skeleton_binary_to_project(bytes(bad_endian)),
            ),
            _error_probe(
                E_FORMAT_INDEX,
                "format_index_width",
                lambda: skeleton_binary_to_project(bytes(bad_index)),
            ),
        ]

        # 3. Migration and idempotence.
        legacy = _legacy_project(scenes[200])
        migrated = migrate_project(legacy)
        migrated_scene_bytes = canonical_json_bytes(migrated)
        migrated_twice = migrate_project(migrated)
        migrated_twice_bytes = canonical_json_bytes(migrated_twice)
        from .model import scene_from_project_dict

        migrated_scene = scene_from_project_dict(migrated)
        migration_report = {
            "legacy_schema_version": 1,
            "migrated_schema_version": migrated["schema_version"],
            "migration_defaults_hash": sha256_json(MIGRATION_DEFAULTS),
            "migrated_hash": sha256_bytes(migrated_scene_bytes),
            "migrated_twice_hash": sha256_bytes(migrated_twice_bytes),
            "idempotent": migrated_scene_bytes == migrated_twice_bytes,
            "migrated_scene_hash": migrated_scene.content_hash,
            "style_filled": migrated["elements"][0].get("style") == MIGRATION_DEFAULTS["style"],
            "sampling_filled": migrated["elements"][0].get("sampling") == MIGRATION_DEFAULTS["sampling"],
            "layer_order_filled": "layer_order" in migrated["elements"][0],
        }
        error_chain.append(
            _error_probe(
                E_SCHEMA_VERSION,
                "future_schema_version",
                lambda: migrate_project({"schema_version": 999}),
            )
        )
        error_chain.append(
            _error_probe(
                E_STROKE_SAMPLE,
                "invalid_stroke_sampling",
                lambda: sampling_from_dict({"tolerance_px": 0.0}),
            )
        )
        zero_line_element = replace(
            scenes[200].elements[0],
            segments=(Segment("line", (Point(1.0, 1.0), Point(1.0, 1.0))),),
        )
        error_chain.append(
            _error_probe(
                E_OUTLINE_DEGENERATE,
                "degenerate_outline",
                lambda: flatten_element(zero_line_element),
            )
        )

        # 4. Full rendering, per-tier strategy, edits and shared-mode checks.
        engines: dict[int, NprEngine] = {}
        render_report: list[dict[str, Any]] = []
        for tier in TIERS:
            engine, payload = _render_tier(
                scenes[tier],
                tier,
                viewport,
                gl_backend,
                geometry_records_by_tier[tier],
            )
            engines[tier] = engine
            ledger = _write_ledger(root / "rendered" / f"ledger-{tier}.json", engine.current_snapshot, engine.scene.content_hash)
            payload["ledger"] = ledger
            image_path = root / "rendered" / f"tier-{tier}.bgra"
            bmp_path = root / "rendered" / f"tier-{tier}.bmp"
            write_bgra(image_path, engine.current_snapshot.image)
            write_bmp(bmp_path, engine.current_snapshot.image, engine.current_snapshot.viewport)
            payload["artifacts"] = {
                "bgra": image_path.name,
                "bgra_hash": sha256_bytes(engine.current_snapshot.image),
                "bmp": bmp_path.name,
            }
            render_report.append(payload)

        strategy_report = _strategy_cache_probe(engines)

        closure_engine = engines[200]
        current = closure_engine.current_snapshot
        wrong_layer = next(layer.id for layer in closure_engine.scene.layers if layer.id != closure_engine.scene.elements[0].layer_id)
        error_chain.append(
            _error_probe(
                E_CLOSURE_MISMATCH,
                "closure_layer_mismatch",
                lambda: render_snapshot(
                    closure_engine.scene,
                    closure_engine.geometry,
                    closure_engine.mode,
                    closure_engine.viewport,
                    current.version + 100,
                    prior=current,
                    dirty_elements={closure_engine.scene.elements[0].id},
                    dirty_layers={wrong_layer},
                    strategy=closure_engine.strategy,
                ),
            )
        )
        error_chain.append(
            _error_probe(
                E_SNAPSHOT_STALE,
                "stale_snapshot_read",
                lambda: closure_engine.read_snapshot(0),
            )
        )
        previous_disable = os.environ.get("NPR_DISABLE_GL")
        os.environ["NPR_DISABLE_GL"] = "1"
        try:
            error_chain.append(_error_probe(E_GL_CONTEXT, "forced_gl_unavailable", open_offscreen_gl))
        finally:
            if previous_disable is None:
                os.environ.pop("NPR_DISABLE_GL", None)
            else:
                os.environ["NPR_DISABLE_GL"] = previous_disable

        # 5. Atomic persistence, real child-process termination, recovery, and
        #    a continued edit based on the recovered version stamp.
        persistence_report = _persistence_sequence(root, engines[2000], repo_root)

        # 6. A final read verifies that only the normal-close generation is used.
        final_store = PersistenceStore(root)
        final_read = final_store.recover()
        final_close_visible = final_read is not None and final_read.clean_shutdown

        verifications = {
            "deterministic_generation": all(item["byte_identical_on_regeneration"] for item in generate_report),
            "format_roundtrip_all_tiers": all(item["within_tolerance"] for item in format_report),
            "migration_idempotent": migration_report["idempotent"],
            "strategy_choices_recorded": all(item["full_render"]["strategy"]["chosen"] for item in render_report),
            "strategy_cache_scope": strategy_report["only_requested_tier_reselected"],
            "incremental_closures_exact": all(item["incremental_closure_exact"] for item in render_report),
            "mode_switches_share_geometry": all(item["all_modes_share_geometry"] for item in render_report),
            "force_recovery_consistent": persistence_report["recovered"]["hashes_match_baseline"],
            "continued_edit_uses_recovered_version": persistence_report["continued_version_uses_recovered_stamp"],
            "normal_close_verified": persistence_report["normal_close_verified"] and final_close_visible,
            "error_chain_complete": all(item["triggered"] for item in error_chain),
        }
        verifications["all_steps_passed"] = all(verifications.values())
        report_body = {
            "format": "npr-demo-report-v1",
            "config": {
                "seed": int(seed),
                "tiers": list(TIERS),
                "viewport": [int(viewport[0]), int(viewport[1])],
                "software_renderer": "deterministic_python_bgra",
                "style_modes": ["outline", "sketch", "cartoon"],
            },
            "steps": {
                "generate": generate_report,
                "format": format_report,
                "migration": migration_report,
                "strategy": strategy_report,
                "render": render_report,
                "persistence": persistence_report,
            },
            "degradation_ledger": degradations,
            "error_chain": error_chain,
            "verification": verifications,
        }
        report = dict(report_body)
        report["content_digest"] = sha256_json(report_body)
        atomic_write_json(root / "report" / "summary.json", report)
        return report
    finally:
        if gl_backend is not None:
            gl_backend.close()


def _deterministic_tree(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative_root in ("generated", "rendered", "state"):
        directory = root / relative_root
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            result[path.relative_to(root).as_posix()] = sha256_bytes(path.read_bytes())
    summary = root / "report" / "summary.json"
    if summary.exists():
        result["report/summary.json"] = sha256_bytes(summary.read_bytes())
    return result


def run_verification(
    workspace: Path | str,
    *,
    seed: int = DEFAULT_SEED,
    viewport: tuple[int, int] = (256, 192),
) -> dict[str, Any]:
    root = Path(workspace).resolve()
    run_a = root / "run-a"
    run_b = root / "run-b"
    if run_a.exists():
        shutil.rmtree(run_a)
    if run_b.exists():
        shutil.rmtree(run_b)
    report_a = run_demo(run_a, seed=seed, viewport=viewport)
    report_b = run_demo(run_b, seed=seed, viewport=viewport)
    bytes_a = canonical_json_bytes(report_a)
    bytes_b = canonical_json_bytes(report_b)
    tree_a = _deterministic_tree(run_a)
    tree_b = _deterministic_tree(run_b)
    differing_files = sorted(
        key for key in set(tree_a) | set(tree_b) if tree_a.get(key) != tree_b.get(key)
    )
    result = {
        "format": "npr-two-run-verification-v1",
        "summary_byte_identical": bytes_a == bytes_b,
        "summary_hash_a": sha256_bytes(bytes_a),
        "summary_hash_b": sha256_bytes(bytes_b),
        "deterministic_file_count": len(tree_a),
        "tree_files_identical": not differing_files,
        "differing_files": differing_files,
        "demo_content_digest_a": report_a["content_digest"],
        "demo_content_digest_b": report_b["content_digest"],
        "all_steps_passed_a": report_a["verification"]["all_steps_passed"],
        "all_steps_passed_b": report_b["verification"]["all_steps_passed"],
    }
    result["verified"] = (
        result["summary_byte_identical"]
        and result["tree_files_identical"]
        and result["all_steps_passed_a"]
        and result["all_steps_passed_b"]
    )
    (root / "report").mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "report" / "verification.json", result)
    return result
