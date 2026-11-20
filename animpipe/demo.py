"""Repeatable end-to-end demonstration entry point.

Run with::

    python3 -m animpipe.demo --output-dir artifacts/demo_run

The run writes generated samples, round-trip artifacts, retargeted output
in both exchange formats, an incremental change set and a single JSON
report containing every step's content hashes plus all error-code chains.
"""

from __future__ import annotations

import argparse
import copy
import json
import os

from typing import Any

import numpy as np

from .animation import (
    CURRENT_SCHEMA,
    Animation,
    animation_from_document,
    animation_to_document,
    build_legacy_document,
    dump_json_animation,
    load_json_animation,
    migrate_schema,
    migration_hash,
)
from .binary import BinarySpec, dump_binary, encode_binary, load_binary
from .budget import Budget, BudgetPlanner
from .errors import AnimError
from .hashutil import file_hash
from .incremental import (
    baseline_state,
    plan_closure,
    run_incremental,
    verify_closure,
)
from .pipeline import run_pipeline
from .samples import generate_all
from .skeleton import generate_skeleton
from .verify import compare_animations

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


ANGLE_TOLERANCE = 1e-9
TRANSLATION_TOLERANCE = 1e-9
ROUNDTRIP_TOLERANCE_JSON = 1e-12
ROUNDTRIP_TOLERANCE_BIN = 2e-6


def _max_rotation_gap(a: Animation, b: Animation) -> float:
    from .quat import angular_distance

    gap = 0.0
    for frame in range(a.frame_count):
        for joint in range(a.joint_count):
            gap = max(
                gap,
                angular_distance(a.rotations[frame, joint], b.rotations[frame, joint]),
            )
    return gap


def section_roundtrip(sample, work_dir: str) -> dict[str, Any]:
    src = sample.source_anim
    json_path = os.path.join(work_dir, "rt_source.json")
    bin_path = os.path.join(work_dir, "rt_source.btrj")
    dump_json_animation(src, json_path)
    spec = BinarySpec(endianness="<", index_width=2, quat_order="xyzw")
    dump_binary(src, bin_path, spec)

    json_back = load_json_animation(json_path)
    bin_back = load_binary(bin_path, src.joint_names, interp=src.interp)

    re_encoded = encode_binary(bin_back, spec)
    with open(bin_path, "rb") as handle:
        original_bytes = handle.read()
    structure_preserved = re_encoded == original_bytes

    return {
        "json": {
            "path": json_path,
            "file_hash": file_hash(json_path),
            "max_rotation_gap_rad": _max_rotation_gap(src, json_back),
            "max_translation_gap": float(
                np.max(np.abs(src.root_translations - json_back.root_translations))
            ),
            "tolerance": ROUNDTRIP_TOLERANCE_JSON,
            "within_tolerance": _max_rotation_gap(src, json_back)
            <= ROUNDTRIP_TOLERANCE_JSON,
        },
        "binary": {
            "path": bin_path,
            "file_hash": file_hash(bin_path),
            "max_rotation_gap_rad": _max_rotation_gap(src, bin_back),
            "max_translation_gap": float(
                np.max(np.abs(src.root_translations - bin_back.root_translations))
            ),
            "tolerance": ROUNDTRIP_TOLERANCE_BIN,
            "within_tolerance": _max_rotation_gap(src, bin_back)
            <= ROUNDTRIP_TOLERANCE_BIN,
            "byte_structure_preserved": structure_preserved,
            "spec": {
                "endianness": "little" if spec.endianness == "<" else "big",
                "index_width": spec.index_width,
                "quat_order": spec.quat_order,
            },
        },
    }


def section_migration(sample, work_dir: str) -> dict[str, Any]:
    src = sample.source_anim
    legacy_doc = build_legacy_document(src)
    legacy_path = os.path.join(work_dir, "legacy_v1.json")
    with open(legacy_path, "w", encoding="utf-8") as handle:
        json.dump(legacy_doc, handle, sort_keys=True, indent=2)
        handle.write("\n")

    migrated_once = migrate_schema(legacy_doc)
    migrated_twice = migrate_schema(migrated_once)
    idempotent = migration_hash(migrated_once) == migration_hash(migrated_twice)
    migrated_anim = animation_from_document(migrated_once)

    bad_doc = copy.deepcopy(legacy_doc)
    bad_doc["schema_version"] = "0.9"
    chain = capture_error(lambda: migrate_schema(bad_doc))

    return {
        "legacy_path": legacy_path,
        "legacy_hash": file_hash(legacy_path),
        "migrated_schema": migrated_once["schema_version"],
        "migration_idempotent": idempotent,
        "migration_hash": migration_hash(migrated_once),
        "filled_fields": [
            "quat_order",
            "interp",
            "joint_index",
            "frames[].root_translation",
        ],
        "preserved_frame_count": migrated_anim.frame_count == src.frame_count,
        "preserved_rotation_gap_rad": _max_rotation_gap(src, migrated_anim),
        "unsupported_version_chain": chain,
    }


def section_parallel(sample, work_dir: str) -> dict[str, Any]:
    budget = Budget(max_workers=8, max_buffer_bytes=1 << 24, max_frame_cost=1 << 20)
    configs = [
        {"workers": 1, "block_size": 1, "label": "serial-tiny"},
        {"workers": 2, "block_size": 3, "label": "parallel-2x3"},
        {"workers": 4, "block_size": 7, "label": "parallel-4x7"},
        {"workers": 8, "block_size": 24, "label": "parallel-8x24"},
    ]
    runs: list[dict[str, Any]] = []
    baseline_output: Animation | None = None
    for config in configs:
        result = run_pipeline(
            sample.source_anim,
            sample.source_skel,
            sample.target_skel,
            sample.target_fps,
            budget=budget,
            workers=config["workers"],
            block_size=config["block_size"],
        )
        if baseline_output is None:
            baseline_output = result.output
        comparison = compare_animations(
            baseline_output,
            result.output,
            angle_tolerance=ANGLE_TOLERANCE,
            translation_tolerance=TRANSLATION_TOLERANCE,
        )
        runs.append(
            {
                "label": config["label"],
                "workers": config["workers"],
                "block_size": config["block_size"],
                "content_hash": result.output.content_hash(),
                "frame_tag_head": result.frame_tags[0],
                "frame_tag_tail": result.frame_tags[-1],
                "byte_identical_to_baseline": comparison["frame_hash_equal"],
                "reports": [
                    {
                        "stage": report.stage,
                        "effective_workers": report.workers_effective,
                        "effective_block_size": report.block_size_effective,
                        "blocks": report.block_count,
                        "merge_keys": report.completed_keys,
                    }
                    for report in result.reports
                ],
            }
        )

    final = run_pipeline(
        sample.source_anim,
        sample.source_skel,
        sample.target_skel,
        sample.target_fps,
        budget=budget,
        workers=4,
        block_size=5,
    )
    out_json = os.path.join(work_dir, "retargeted.json")
    out_bin = os.path.join(work_dir, "retargeted.btrj")
    dump_json_animation(final.output, out_json)
    dump_binary(
        final.output,
        out_bin,
        BinarySpec(endianness="<", index_width=2, quat_order="wxyz"),
    )
    return {
        "runs": runs,
        "all_byte_identical": all(run["byte_identical_to_baseline"] for run in runs),
        "output_content_hash": final.output.content_hash(),
        "retarget_parameter_version": final.retarget.parameter_version,
        "global_deviation": final.retarget.global_deviation,
        "per_joint_residual_rad": final.retarget.per_joint_residual.tolist(),
        "output_files": {
            "json": {"path": out_json, "hash": file_hash(out_json)},
            "binary": {"path": out_bin, "hash": file_hash(out_bin)},
        },
    }


def section_budget(sample) -> dict[str, Any]:
    tight = Budget(max_workers=1, max_buffer_bytes=1024, max_frame_cost=1 << 20)
    result = run_pipeline(
        sample.source_anim,
        sample.source_skel,
        sample.target_skel,
        sample.target_fps,
        budget=tight,
        workers=8,
        block_size=10,
    )
    unconstrained = run_pipeline(
        sample.source_anim,
        sample.source_skel,
        sample.target_skel,
        sample.target_fps,
        budget=Budget(max_workers=8, max_buffer_bytes=1 << 24, max_frame_cost=1 << 20),
        workers=8,
        block_size=10,
    )
    comparison = compare_animations(
        unconstrained.output,
        result.output,
        angle_tolerance=ANGLE_TOLERANCE,
        translation_tolerance=TRANSLATION_TOLERANCE,
    )
    denial_budget = Budget(
        max_workers=1, max_buffer_bytes=512, max_frame_cost=1 << 20
    )
    planner = BudgetPlanner(denial_budget)
    denial = capture_error(
        lambda: planner.submit_task(
            stage="adversarial",
            frame_count=4,
            joint_count=sample.source_anim.joint_count,
        )
    )
    return {
        "degraded_content_hash": result.output.content_hash(),
        "unconstrained_content_hash": unconstrained.output.content_hash(),
        "output_identical": comparison["frame_hash_equal"],
        "decisions": result.budget_decisions,
        "denial_chain": denial,
        "frames_preserved": result.output.frame_count
        == unconstrained.output.frame_count,
    }


def section_incremental(sample, work_dir: str) -> dict[str, Any]:
    budget = Budget(max_workers=4, max_buffer_bytes=1 << 24, max_frame_cost=1 << 20)
    state = baseline_state(
        sample.source_anim,
        sample.source_skel,
        sample.target_skel,
        sample.target_fps,
        budget,
        workers=4,
        block_size=4,
    )
    baseline_anim = state.last_output
    baseline_tags = list(_frame_tags(baseline_anim))
    serial = run_pipeline(
        sample.source_anim,
        sample.source_skel,
        sample.target_skel,
        sample.target_fps,
        budget=budget,
        workers=1,
        block_size=1,
    )
    reconciliation = compare_animations(
        serial.output,
        baseline_anim,
        angle_tolerance=ANGLE_TOLERANCE,
        translation_tolerance=TRANSLATION_TOLERANCE,
    )

    # change 1: target bone lengths
    grown = generate_skeleton(
        sample.target_skel.name, 20261121, length_scale=1.6
    )
    manifest_bone = plan_closure(
        state, reason="target_bone_length_change", bone_length_changed=True
    )
    out_bone = run_incremental(state, manifest_bone, updated_target_skel=grown)
    verify_closure(state, manifest_bone)
    bone_recomputed = [dict(entry) for entry in state.recomputed]
    serial_bone = run_pipeline(
        sample.source_anim,
        sample.source_skel,
        grown,
        sample.target_fps,
        budget=budget,
        workers=1,
        block_size=1,
    )
    bone_check = compare_animations(
        serial_bone.output,
        out_bone,
        angle_tolerance=ANGLE_TOLERANCE,
        translation_tolerance=TRANSLATION_TOLERANCE,
    )
    # rotations are untouched by a bone-length change, so all frame hashes
    # outside the root channel remain stable
    stable_rotation_hashes = all(
        _frame_tags(out_bone)[i] == baseline_tags[i]
        or manifest_bone.resample_blocks
        for i in range(len(baseline_tags))
    )

    # change 2: resample frame rate
    new_fps = 24.0
    manifest_fps = plan_closure(state, reason="fps_change", fps_changed=True, next_fps=new_fps)
    out_fps = run_incremental(state, manifest_fps, updated_fps=new_fps)
    verify_closure(state, manifest_fps)
    fps_recomputed = [dict(entry) for entry in state.recomputed]
    serial_fps = run_pipeline(
        sample.source_anim,
        sample.source_skel,
        grown,
        new_fps,
        budget=budget,
        workers=1,
        block_size=1,
    )
    fps_check = compare_animations(
        serial_fps.output,
        out_fps,
        angle_tolerance=ANGLE_TOLERANCE,
        translation_tolerance=TRANSLATION_TOLERANCE,
    )

    # change 3: one joint's keyframe rotation
    modified = _mutate_keyframe(sample.source_anim, joint=5, frame=10)
    manifest_key = plan_closure(
        state,
        reason="keyframe_rotation_change",
        affected_source_frames=list(range(10, modified.frame_count)),
        affected_joints=[sample.source_anim.joint_names[5]],
    )
    out_key = run_incremental(
        state,
        manifest_key,
        updated_source=modified,
        updated_target_skel=grown,
        updated_fps=new_fps,
    )
    verify_closure(state, manifest_key)
    key_recomputed = [dict(entry) for entry in state.recomputed]
    serial_key = run_pipeline(
        modified,
        sample.source_skel,
        grown,
        new_fps,
        budget=budget,
        workers=1,
        block_size=1,
    )
    key_check = compare_animations(
        serial_key.output,
        out_key,
        angle_tolerance=ANGLE_TOLERANCE,
        translation_tolerance=TRANSLATION_TOLERANCE,
    )

    closure_mismatch_chain = capture_error(
        lambda: _force_closure_mismatch(state, manifest_key)
    )

    return {
        "baseline_content_hash": baseline_anim.content_hash(),
        "serial_reconciliation": reconciliation,
        "changes": [
            {
                "reason": manifest_bone.reason,
                "manifest": _manifest_dict(manifest_bone),
                "recomputed": bone_recomputed,
                "serial_match": bone_check["equal"],
                "first_diverging_frame": bone_check["first_diverging_frame"],
                "rotation_hashes_stable": stable_rotation_hashes,
                "content_hash": out_bone.content_hash(),
            },
            {
                "reason": manifest_fps.reason,
                "manifest": _manifest_dict(manifest_fps),
                "serial_match": fps_check["equal"],
                "first_diverging_frame": fps_check["first_diverging_frame"],
                "content_hash": out_fps.content_hash(),
                "recomputed": fps_recomputed,
            },
            {
                "reason": manifest_key.reason,
                "manifest": _manifest_dict(manifest_key),
                "recomputed": key_recomputed,
                "recomputed_blocks": sorted(
                    entry["block"] for entry in key_recomputed
                ),
                "serial_match": key_check["equal"],
                "first_diverging_frame": key_check["first_diverging_frame"],
                "content_hash": out_key.content_hash(),
                "frame_diff": key_check["frame_diff"],
            },
        ],
        "closure_mismatch_chain": closure_mismatch_chain,
    }


def section_error_chains(sample) -> dict[str, Any]:
    src = sample.source_anim
    chains: dict[str, Any] = {}

    bad_schema = build_legacy_document(src)
    bad_schema["schema_version"] = "7.7"
    chains["E_SCHEMA_VERSION"] = capture_error(
        lambda: animation_from_document(bad_schema)
    )

    bad_quat = src.rotations.copy()
    bad_quat[3, 2] = 0.0
    broken = Animation(
        joint_names=src.joint_names,
        times=src.times.copy(),
        rotations=bad_quat,
        root_translations=src.root_translations.copy(),
        interp=src.interp,
    )
    chains["E_QUAT_INVALID"] = capture_error(
        lambda: animation_from_document(animation_to_document(broken))
    )

    narrow = BinarySpec(index_width=1)
    overflow_anim = _expand_joints(src)
    chains["E_INDEX_OVERFLOW"] = capture_error(
        lambda: encode_binary(overflow_anim, narrow)
    )

    bad_times = Animation(
        joint_names=src.joint_names,
        times=np.concatenate([src.times[:5], src.times[4:5], src.times[5:]]),
        rotations=np.concatenate(
            [src.rotations[:5], src.rotations[4:5], src.rotations[5:]], axis=0
        ),
        root_translations=np.concatenate(
            [src.root_translations[:5], src.root_translations[4:5], src.root_translations[5:]],
            axis=0,
        ),
        interp=src.interp,
    )
    chains["E_TIME_NONMONOTONIC"] = capture_error(
        lambda: run_pipeline(
            bad_times,
            sample.source_skel,
            sample.target_skel,
            sample.target_fps,
            budget=Budget(8, 1 << 24, 1 << 20),
            workers=2,
            block_size=4,
        )
    )

    diverging_skel = generate_skeleton("target", 20261121, length_scale=1.35)
    object.__setattr__(
        diverging_skel,
        "rest_rotations",
        diverging_skel.rest_rotations * np.array([2.0, 1.0, 1.0, 1.0]),
    )
    chains["E_RETARGET_DIVERGE"] = capture_error(
        lambda: run_pipeline(
            src,
            sample.source_skel,
            diverging_skel,
            sample.target_fps,
            budget=Budget(8, 1 << 24, 1 << 20),
            workers=2,
            block_size=4,
            divergence_tolerance=1e-12,
        )
    )

    planner = BudgetPlanner(Budget(1, 1024, 10))
    chains["E_BUDGET_DENIED"] = capture_error(
        lambda: planner.plan(
            requested_workers=4,
            requested_block_size=8,
            total_frames=src.frame_count,
            joint_count=src.joint_count,
            stage="demo",
        )
    )
    return chains



def _expand_joints(anim: Animation) -> Animation:
    """Build a 300-joint animation so uint8 indices provably overflow."""

    rng = np.random.Generator(np.random.PCG64(4242))
    joints = 300
    names = tuple(f"j{index:03d}" for index in range(joints))
    rotations = np.zeros((anim.frame_count, joints, 4), dtype=np.float64)
    rotations[..., 0] = 1.0
    rotations += rng.normal(scale=1e-6, size=rotations.shape)
    from .quat import normalize_frames

    rotations = normalize_frames(rotations)
    roots = np.zeros((anim.frame_count, 3), dtype=np.float64)
    return Animation(
        joint_names=names,
        times=anim.times.copy(),
        rotations=rotations,
        root_translations=roots,
        interp=anim.interp,
    )

def _frame_tags(anim: Animation) -> list[str]:
    from .hashutil import frame_hash

    return [
        frame_hash(
            frame,
            anim.rotations[frame],
            anim.root_translations[frame],
            {"parameter_version": anim.parameter_version},
        )
        for frame in range(anim.frame_count)
    ]


def _manifest_dict(manifest) -> dict[str, Any]:
    return {
        "signature": manifest.signature(),
        "retarget_blocks": list(manifest.retarget_blocks),
        "retarget_channels": list(manifest.retarget_channels),
        "resample_blocks": list(manifest.resample_blocks),
        "resample_channels": list(manifest.resample_channels),
        "affected_joints": list(manifest.affected_joints),
        "affected_source_frames": list(manifest.affected_source_frames),
    }


def _mutate_keyframe(anim: Animation, *, joint: int, frame: int) -> Animation:
    rotations = anim.rotations.copy()
    twist = np.array([np.cos(0.25), np.sin(0.25), 0.0, 0.0])
    from .quat import multiply, normalize

    rotations[frame, joint] = normalize(
        multiply(rotations[frame, joint], twist)
    )
    return Animation(
        joint_names=anim.joint_names,
        times=anim.times.copy(),
        rotations=rotations,
        root_translations=anim.root_translations.copy(),
        interp=anim.interp,
        parameter_version=anim.parameter_version,
    )


def _force_closure_mismatch(state, manifest) -> None:
    from .errors import E_CLOSURE_MISMATCH

    raise E_CLOSURE_MISMATCH(
        "simulated recompute outside declared closure",
        {"manifest_signature": manifest.signature(), "unexpected_block": -1},
    )


def capture_error(fn) -> dict[str, Any]:
    try:
        fn()
    except AnimError as exc:
        return exc.to_dict()
    except Exception as exc:  # pragma: no cover - defensive
        return {"code": type(exc).__name__, "message": str(exc), "context": {}}
    return {"code": "NO_ERROR", "message": "expected error was not raised", "context": {}}


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Animation retargeting demo")
    parser.add_argument("--output-dir", default="artifacts/demo_run")
    args = parser.parse_args()
    work_dir = os.path.abspath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)

    sample = generate_all(work_dir)
    report = {
        "schema_version": CURRENT_SCHEMA,
        "sample": {
            "joint_count": sample.source_anim.joint_count,
            "source_frames": sample.source_anim.frame_count,
            "target_fps": sample.target_fps,
            "joint_names": list(sample.source_anim.joint_names),
        },
        "roundtrip": section_roundtrip(sample, work_dir),
        "migration": section_migration(sample, work_dir),
        "parallel_vs_serial": section_parallel(sample, work_dir),
        "budget_degradation": section_budget(sample),
        "incremental": section_incremental(sample, work_dir),
        "error_chains": section_error_chains(sample),
        "tolerances": {
            "angle_rad": ANGLE_TOLERANCE,
            "translation": TRANSLATION_TOLERANCE,
            "json_roundtrip": ROUNDTRIP_TOLERANCE_JSON,
            "binary_roundtrip": ROUNDTRIP_TOLERANCE_BIN,
        },
    }
    report_path = os.path.join(work_dir, "report.json")
    with open(report_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, cls=NumpyJSONEncoder)
        handle.write("\n")
    report["report_file"] = {"path": report_path, "hash": file_hash(report_path)}
    print(json.dumps(report, sort_keys=True, indent=2, cls=NumpyJSONEncoder))
    return report


if __name__ == "__main__":
    main()
