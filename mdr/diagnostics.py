from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from .pipeline import Pipeline
from .utils import read_json, round_div, write_json
from .validation import validate_dataset
from .voxels import build_spatial_refs, run_voxel_stage


def _norm_ref(value: int) -> int:
    result, _ = round_div(int(value), 2)
    return result


def reference_config(base: dict[str, Any]) -> dict[str, Any]:
    ref = deepcopy(base)
    ref["frac_bits"] = int(base["frac_bits"]) + 1
    ref["quant_level"] = "Q0"
    ref.setdefault("quant_shifts", {})["Q0"] = 0
    allowed = set(ref.get("allowed_frac_bits", []))
    allowed.add(ref["frac_bits"])
    ref["allowed_frac_bits"] = sorted(allowed)
    ref["budgets"]["max_workers"] = 1
    ref["budgets"]["global_point_budget"] = 10_000
    ref["budgets"]["view_point_quota"] = 10_000
    ref["budgets"]["max_subdivisions_per_batch"] = 10_000
    return ref


def coarse_config(base: dict[str, Any]) -> dict[str, Any]:
    coarse = deepcopy(base)
    coarse["quant_level"] = "Q2"
    coarse["budgets"]["max_workers"] = 1
    coarse["budgets"]["global_point_budget"] = 10_000
    coarse["budgets"]["view_point_quota"] = 10_000
    coarse["budgets"]["max_subdivisions_per_batch"] = 10_000
    return coarse


def _run_variant(source_root: Path, work_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    shutil.copytree(source_root / "samples", work_root / "samples")
    for directory in ("configs", "ledger/artifacts", "ledger/snapshots", "reports"):
        (work_root / directory).mkdir(parents=True, exist_ok=True)
    write_json(work_root / "configs" / "reconstruction.json", config)
    return Pipeline(work_root, config=config, parallel=False, use_snapshot=False).run()


def _feature_maps(state: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    result: dict[tuple[str, int, int], dict[str, Any]] = {}
    for view_id, artifact in state["features"].items():
        for feature in artifact["features"]:
            result[(view_id, int(feature["x"]), int(feature["y"]))] = feature
    return result


def _point_map(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {point["match_id"]: point for point in state["triangulation"]["points"]}


def _depth_map(state: dict[str, Any]) -> dict[tuple[str, int, int], list[Any]]:
    result: dict[tuple[str, int, int], list[Any]] = {}
    for view_id, depth in state["depth_maps"].items():
        for row in depth["pixels"]:
            result[(view_id, int(row[0]), int(row[1]))] = row
    return result


def compare_voxel_diagnostics(
    normal_state: dict[str, Any],
    ref_state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    normal_features = _feature_maps(normal_state)
    ref_features = _feature_maps(ref_state)
    normal_points = _point_map(normal_state)
    ref_points = _point_map(ref_state)
    normal_depth = _depth_map(normal_state)
    ref_depth = _depth_map(ref_state)
    tolerance = int(config["diagnostic_tolerance_q"])

    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    flip_examples: list[dict[str, Any]] = []

    normal_nodes = normal_state["voxels"]["nodes"]
    ref_nodes = ref_state["voxels"]["nodes"]
    for voxel_id in sorted(normal_nodes, key=lambda vid: tuple(normal_nodes[vid]["sort_path"])):
        node = normal_nodes[voxel_id]
        ref_node = ref_nodes.get(voxel_id)
        feature_delta = 0
        tri_delta = 0
        depth_delta = 0
        voxel_delta = 10**15 if ref_node is None else abs(int(node["combined_score_q"]) - _norm_ref(ref_node["combined_score_q"]))

        for link in node["tri_point_refs"]:
            point = next((p for p in normal_state["triangulation"]["points"] if p["point_id"] == link["point_id"]), None)
            if point is None:
                continue
            ref_point = ref_points.get(point["match_id"])
            for view_id, xy in point["observations"].items():
                normal_feature = normal_features.get((view_id, int(xy[0]), int(xy[1])))
                ref_feature = ref_features.get((view_id, int(xy[0]), int(xy[1])))
                if normal_feature and ref_feature:
                    score_delta = abs(int(normal_feature["score_q"]) - _norm_ref(ref_feature["score_q"]))
                    descriptor_q_delta = 0
                    for normal_value, ref_value in zip(
                        normal_feature.get("descriptor_q", []),
                        ref_feature.get("descriptor_q", []),
                    ):
                        descriptor_q_delta = max(
                            descriptor_q_delta,
                            abs(int(normal_value) - _norm_ref(int(ref_value))),
                        )
                    descriptor_delta = sum(
                        abs(int(a) - int(b))
                        for a, b in zip(normal_feature["descriptor"], ref_feature["descriptor"])
                    )
                    feature_delta = max(
                        feature_delta,
                        score_delta,
                        descriptor_q_delta,
                        descriptor_delta * (1 << max(0, int(config["frac_bits"]) - 5)),
                    )
            if ref_point:
                residual_delta = abs(int(point["residual_q"]) - _norm_ref(ref_point["residual_q"]))
                coordinate_delta = max(
                    abs(int(a) - _norm_ref(int(b)))
                    for a, b in zip(point["point_q"], ref_point["point_q"])
                )
                tri_delta = max(tri_delta, residual_delta, coordinate_delta)

        for link in node.get("depth_pixel_links", []):
            parts = link["pixel_ref"].split(":")
            view_id, x, y = parts[0], int(parts[1]), int(parts[2])
            normal_row = normal_depth.get((view_id, x, y))
            ref_row = ref_depth.get((view_id, x, y))
            if normal_row and ref_row:
                depth_delta = max(depth_delta, abs(int(normal_row[2]) - _norm_ref(int(ref_row[2]))))
        if ref_node is not None:
            depth_delta = max(
                depth_delta,
                abs(int(node["depth_inconsistency_q"]) - _norm_ref(ref_node["depth_inconsistency_q"])),
                abs(int(node["depth_mad_q"]) - _norm_ref(ref_node["depth_mad_q"])),
            )

        stages = [
            ("feature_quantization", feature_delta),
            ("triangulation_residual", tri_delta),
            ("depth_consistency", depth_delta),
        ]
        exceeded = next((name for name, delta in stages if delta > tolerance), None)
        decision_flip = ref_node is None or node["status"] != ref_node["status"] or set(node["child_ids"]) != set(ref_node.get("child_ids", []))
        if exceeded is None and (voxel_delta > tolerance or decision_flip):
            exceeded = "voxel_subdivision"
        attribution = exceeded or "within_tolerance"
        counts[attribution] = counts.get(attribution, 0) + 1
        for stage_name, delta in (
            ("feature_quantization", feature_delta),
            ("triangulation_residual", tri_delta),
            ("depth_consistency", depth_delta),
            ("voxel_subdivision", voxel_delta),
        ):
            if delta > tolerance:
                key = f"isolated_exceeded:{stage_name}"
                counts[key] = counts.get(key, 0) + 1
        flip_source = attribution if decision_flip else None
        if decision_flip:
            counts["decision_flip"] = counts.get("decision_flip", 0) + 1
            if flip_source:
                counts[f"flip_source:{flip_source}"] = counts.get(f"flip_source:{flip_source}", 0) + 1

        record = {
            "voxel_id": voxel_id,
            "sort_path": node["sort_path"],
            "level": node["depth_level"],
            "normal_status": node["status"],
            "reference_status": None if ref_node is None else ref_node["status"],
            "normal_score_q": node["combined_score_q"],
            "reference_score_normalized_q": None if ref_node is None else _norm_ref(ref_node["combined_score_q"]),
            "feature_delta_q": feature_delta,
            "triangulation_delta_q": tri_delta,
            "depth_delta_q": depth_delta,
            "voxel_delta_q": voxel_delta,
            "tolerance_q": tolerance,
            "first_exceeded_stage": exceeded,
            "attribution": attribution,
            "decision_flip": decision_flip,
            "flip_source": flip_source,
        }
        records.append(record)
        if decision_flip and len(flip_examples) < 10:
            flip_examples.append(record)

    return {
        "tolerance_q": tolerance,
        "voxel_count": len(records),
        "counts": counts,
        "flip_examples": flip_examples,
        "records": records,
    }


def run_boundary_probe(
    source_root: Path,
    base_config: dict[str, Any],
    coarse_state: dict[str, Any],
    coarse_ref_state: dict[str, Any],
) -> dict[str, Any]:
    coarse = coarse_config(base_config)
    coarse_ref = reference_config(coarse)
    candidates = compare_voxel_diagnostics(coarse_state, coarse_ref_state, coarse)["records"]
    target = next(
        (
            record
            for record in candidates
            if record["level"] < int(coarse["max_subdivision_depth"])
            and record["normal_score_q"] != record["reference_score_normalized_q"]
            and record["normal_score_q"] > 0
        ),
        None,
    )
    if target is None:
        return {"triggered": False, "reason": "no score boundary candidate"}

    threshold = max(1, min(int(target["normal_score_q"]), int(target["reference_score_normalized_q"] or 0)))
    probe_coarse = deepcopy(coarse)
    probe_ref = deepcopy(coarse_ref)
    probe_coarse["residual_threshold"] = threshold
    probe_ref["residual_threshold"] = threshold

    raw_dataset = read_json(source_root / "samples" / "dataset.json")
    validated = validate_dataset(raw_dataset, source_root / "samples", probe_coarse)
    coarse_spatial = build_spatial_refs(coarse_state["triangulation"], coarse_state["depth_maps"], validated.views, validated.cameras, probe_coarse)
    coarse_voxels = run_voxel_stage(coarse_spatial, probe_coarse, max_workers=1, parallel=False)

    validated_ref = validate_dataset(raw_dataset, source_root / "samples", probe_ref)
    ref_spatial = build_spatial_refs(coarse_ref_state["triangulation"], coarse_ref_state["depth_maps"], validated_ref.views, validated_ref.cameras, probe_ref)
    ref_voxels = run_voxel_stage(ref_spatial, probe_ref, max_workers=1, parallel=False)

    coarse_probe_state = {"features": coarse_state["features"], "triangulation": coarse_state["triangulation"], "depth_maps": coarse_state["depth_maps"], "voxels": coarse_voxels}
    ref_probe_state = {"features": coarse_ref_state["features"], "triangulation": coarse_ref_state["triangulation"], "depth_maps": coarse_ref_state["depth_maps"], "voxels": ref_voxels}
    comparison = compare_voxel_diagnostics(coarse_probe_state, ref_probe_state, probe_coarse)
    return {
        "triggered": True,
        "target_voxel_id": target["voxel_id"],
        "probe_threshold_q": threshold,
        "coarse_quant_level": coarse["quant_level"],
        "reference_frac_bits": probe_ref["frac_bits"],
        "coarse_counts": coarse_voxels["counts"],
        "reference_counts": ref_voxels["counts"],
        "attribution_summary": comparison["counts"],
        "flip_examples": comparison["flip_examples"],
        "records": comparison["records"],
    }


def run_diagnostics(root: str | Path) -> dict[str, Any]:
    source_root = Path(root)
    base_config = read_json(source_root / "configs" / "reconstruction.json")
    work_base = source_root / "reports" / "diagnostic-work"
    normal = _run_variant(source_root, work_base / "normal", base_config)
    ref_cfg = reference_config(base_config)
    reference = _run_variant(source_root, work_base / "reference", ref_cfg)
    default_comparison = compare_voxel_diagnostics(normal, reference, base_config)
    coarse_cfg = coarse_config(base_config)
    coarse = _run_variant(source_root, work_base / "coarse", coarse_cfg)
    coarse_ref_cfg = reference_config(coarse_cfg)
    coarse_reference = _run_variant(source_root, work_base / "coarse-reference", coarse_ref_cfg)
    boundary = run_boundary_probe(source_root, base_config, coarse, coarse_reference)
    report = {
        "default_comparison": {k: v for k, v in default_comparison.items() if k != "records"},
        "default_records": default_comparison["records"],
        "boundary_probe": boundary,
        "recovery_note": "Recovery is decided only from ledger/snapshots/current.json committed state.",
    }
    write_json(source_root / "reports" / "diagnostic-report.json", report)
    shutil.rmtree(work_base, ignore_errors=True)
    return report
