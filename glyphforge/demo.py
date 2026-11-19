"""Deterministic end-to-end demonstration and hash report."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .boolean import BooleanOp, ResolutionOrder, boolean_operation
from .diagnostics import region_metrics
from .errors import ErrorCode, GlyphError
from .flatten import FlattenOptions, flatten_contours
from .incremental import IncrementalEngine
from .model import Contour, GlyphSuite, OutlineNode, Point, SegmentType, canonical_json_bytes
from .raster import check_bitmap_consistency, rasterize_boolean, rasterize_stroke
from .samples import generate_suite
from .stroke import StrokeOptions, stroke_polyline


def run_demo(seed: int = 20261119, *, sensitivity_sizes: tuple[int, ...] = (12, 24, 48)) -> dict[str, Any]:
    suite = generate_suite(seed)
    suite.validate()
    report: dict[str, Any] = {
        "schema": "glyphforge.demo.v1",
        "seed": seed,
        "sample_hash": suite.content_hash(),
        "suggested_sizes": [12, 16, 24, 32, 48],
        "error_codes": [code.value for code in ErrorCode],
        "glyphs": _glyph_reports(suite),
        "sensitivity": _sensitivity_report(suite, sensitivity_sizes),
        "incremental": _incremental_report(seed, suite),
        "error_checks": _error_checks(),
    }
    return report


def canonical_report(report: dict[str, Any]) -> bytes:
    return canonical_json_bytes(report)


def run_demo_text(seed: int = 20261119) -> str:
    return canonical_report(run_demo(seed)).decode("utf-8")


def write_report(report: dict[str, Any], path: str | Path) -> str:
    data = canonical_report(report)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _glyph_reports(suite: GlyphSuite) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for case in suite.cases:
        sizes: dict[str, Any] = {}
        for size in case.suggested_sizes:
            flat_a = flatten_contours(case.source_a, FlattenOptions(size), source="a")
            flat_b = flatten_contours(case.source_b, FlattenOptions(size), source="b")
            flat_open = flatten_contours((case.open_path,), FlattenOptions(size), source="open")
            operations: dict[str, Any] = {}
            for op in BooleanOp:
                result = boolean_operation(flat_a.loops, flat_b.loops, op)
                boundary_bitmap, direct_bitmap = rasterize_boolean(result, size)
                consistency = check_bitmap_consistency(
                    boundary_bitmap,
                    direct_bitmap,
                    boundary_loops=result.loops,
                    direct_loops=result.direct_loops,
                    declared_tolerance=flat_a.tolerance,
                )
                if not consistency.consistent:
                    raise GlyphError(ErrorCode.BITMAP_HASH, "boolean bitmap mismatch", consistency.as_dict())
                operations[op.value] = {
                    "boundary_hash": result.boundary_hash(),
                    "bitmap_hash": boundary_bitmap.bitmap_hash,
                    "direct_bitmap_hash": direct_bitmap.bitmap_hash,
                    "consistent": consistency.consistent,
                    "differing_pixels": consistency.differing_pixels,
                    "loops": len(result.loops),
                    "boundary_points": result.boundary_point_count,
                    "area": _round(result.area),
                }
            stroke = stroke_polyline(
                flat_open.open_points,
                StrokeOptions(
                    case.stroke_width,
                    case.stroke_cap,
                    case.stroke_join,
                    flat_open.tolerance,
                ),
            )
            stroke_bitmap, stroke_direct = rasterize_stroke(stroke, size)
            stroke_consistency = check_bitmap_consistency(
                stroke_bitmap,
                stroke_direct,
                boundary_loops=stroke.loops,
                direct_loops=stroke.direct_loops,
                declared_tolerance=flat_open.tolerance,
            )
            if not stroke_consistency.consistent:
                raise GlyphError(ErrorCode.BITMAP_HASH, "stroke bitmap mismatch", stroke_consistency.as_dict())
            sizes[str(size)] = {
                "flatten_tolerance": _round(flat_a.tolerance),
                "flatten": {
                    "a_vertices": flat_a.vertex_count,
                    "b_vertices": flat_b.vertex_count,
                    "open_vertices": flat_open.vertex_count,
                    "a_degenerate_segments": flat_a.degenerate_segments,
                    "b_degenerate_segments": flat_b.degenerate_segments,
                    "open_degenerate_segments": flat_open.degenerate_segments,
                    "a_hash": flat_a.content_hash(),
                    "b_hash": flat_b.content_hash(),
                    "open_hash": flat_open.content_hash(),
                },
                "boolean": operations,
                "stroke": {
                    "boundary_hash": stroke.boundary_hash(),
                    "bitmap_hash": stroke_bitmap.bitmap_hash,
                    "direct_bitmap_hash": stroke_direct.bitmap_hash,
                    "consistent": stroke_consistency.consistent,
                    "differing_pixels": stroke_consistency.differing_pixels,
                    "loops": len(stroke.loops),
                    "boundary_points": stroke.boundary_point_count,
                    "arc_approximation_error": _round(stroke.arc_approximation_error),
                    "primitive_count": stroke.primitive_count,
                },
            }
        output[case.glyph_id] = {
            "category": case.category,
            "stroke_width": case.stroke_width,
            "stroke_cap": case.stroke_cap,
            "stroke_join": case.stroke_join,
            "sizes": sizes,
        }
    return output


def _sensitivity_report(suite: GlyphSuite, sizes: tuple[int, ...]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for case in suite.cases:
        per_size: dict[str, Any] = {}
        for size in sizes:
            if size not in case.suggested_sizes:
                continue
            base_a = flatten_contours(case.source_a, FlattenOptions(size), source="a")
            base_b = flatten_contours(case.source_b, FlattenOptions(size), source="b")
            base_open = flatten_contours((case.open_path,), FlattenOptions(size), source="open")
            base_union = boolean_operation(base_a.loops, base_b.loops, BooleanOp.UNION)
            base_stroke = stroke_polyline(
                base_open.open_points,
                StrokeOptions(case.stroke_width, case.stroke_cap, case.stroke_join, base_open.tolerance),
            )

            variants: list[dict[str, Any]] = []
            for label, scale in (("tight", 0.5), ("loose", 2.0)):
                variant_a = flatten_contours(case.source_a, FlattenOptions(size, tolerance_scale=scale), source="a")
                variant_b = flatten_contours(case.source_b, FlattenOptions(size, tolerance_scale=scale), source="b")
                variant_open = flatten_contours(
                    (case.open_path,),
                    FlattenOptions(size, tolerance_scale=scale),
                    source="open",
                )
                variant_union = boolean_operation(variant_a.loops, variant_b.loops, BooleanOp.UNION)
                variant_stroke = stroke_polyline(
                    variant_open.open_points,
                    StrokeOptions(
                        case.stroke_width,
                        case.stroke_cap,
                        case.stroke_join,
                        variant_open.tolerance,
                    ),
                )
                variants.append(
                    {
                        "label": label,
                        "tolerance_scale": scale,
                        "union": _safe_metrics(
                            base_union.loops,
                            variant_union.loops,
                            base_a.tolerance,
                            variant_a.tolerance,
                            size,
                        ),
                        "stroke": _safe_metrics(
                            base_stroke.loops,
                            variant_stroke.loops,
                            base_open.tolerance,
                            variant_open.tolerance,
                            size,
                        ),
                    }
                )

            order_union = boolean_operation(
                base_a.loops,
                base_b.loops,
                BooleanOp.UNION,
                resolution_order=ResolutionOrder.SOURCE_THEN_LEX,
            )
            order_stroke = stroke_polyline(
                base_open.open_points,
                StrokeOptions(case.stroke_width, case.stroke_cap, case.stroke_join, base_open.tolerance),
                resolution_order=ResolutionOrder.SOURCE_THEN_LEX,
            )
            per_size[str(size)] = {
                "base": {
                    "union_hash": base_union.boundary_hash(),
                    "stroke_hash": base_stroke.boundary_hash(),
                },
                "tolerance_variants": variants,
                "resolution_order_variant": {
                    "name": ResolutionOrder.SOURCE_THEN_LEX.value,
                    "union": _safe_metrics(
                        base_union.loops,
                        order_union.loops,
                        base_a.tolerance,
                        base_a.tolerance,
                        size,
                    ),
                    "stroke": _safe_metrics(
                        base_stroke.loops,
                        order_stroke.loops,
                        base_open.tolerance,
                        base_open.tolerance,
                        size,
                    ),
                },
            }
        output[case.glyph_id] = per_size
    return output


def _safe_metrics(
    loops_a: Any,
    loops_b: Any,
    tolerance_a: float,
    tolerance_b: float,
    size: int,
) -> dict[str, Any]:
    try:
        report = region_metrics(
            loops_a,
            loops_b,
            tolerance_a=tolerance_a,
            tolerance_b=tolerance_b,
            bitmap_size=size,
        )
        return _metric_payload(report)
    except GlyphError as exc:
        if exc.code is not ErrorCode.CONSISTENCY_DRIFT:
            raise
        payload = _metric_payload(exc.details)
        payload["raised_code"] = exc.code.value
        return payload


def _metric_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "symmetric_difference_area": _round(float(report["symmetric_difference_area"])),
        "symmetric_difference_area_method": report.get("symmetric_difference_area_method"),
        "hausdorff_distance": _round(float(report["hausdorff_distance"])),
        "hausdorff_direction": report["hausdorff_direction"],
        "bitmap_differing_pixels": report["bitmap"]["differing_pixels"],
        "bitmap_mean_abs_gray": _round(float(report["bitmap"]["mean_abs_gray"])),
        "error_interval": {
            key: _round(float(value)) if isinstance(value, (int, float)) else value
            for key, value in report["error_interval"].items()
        },
        "classification": report["classification"],
        "drift": report["drift"],
        "divergent_edges": report.get("divergent_edges", []),
    }


def _incremental_report(seed: int, suite: GlyphSuite) -> dict[str, Any]:
    engine = IncrementalEngine(suite)
    initial_tasks = engine.build_all()
    baseline = engine.all_glyph_hashes()
    steps: list[dict[str, Any]] = []

    case = engine.cases["G01_SIMPLE"]
    control = case.source_a[0].nodes[2].controls[0]
    roots = [
        f"flat/G01_SIMPLE/{size}/a" for size in case.suggested_sizes
    ]
    closure = engine.edit_control_point(
        "G01_SIMPLE",
        "a",
        0,
        2,
        0,
        Point(float(control.x) + 0.17, float(control.y) + 0.11),
    )
    steps.append(_incremental_step(engine, "edit_control_point", "G01_SIMPLE", roots, closure, baseline))

    before = engine.all_glyph_hashes()
    roots = [f"stroke/G04_OPEN_PATH/{size}" for size in engine.cases["G04_OPEN_PATH"].suggested_sizes]
    closure = engine.set_stroke_parameters("G04_OPEN_PATH", cap="square", join="miter")
    steps.append(_incremental_step(engine, "change_stroke_parameters", "G04_OPEN_PATH", roots, closure, before))

    before = engine.all_glyph_hashes()
    case3 = engine.cases["G03_OVERLAP_COINCIDENT"]
    roots = [
        f"flat/G03_OVERLAP_COINCIDENT/{size}/{source}"
        for size in case3.suggested_sizes
        for source in ("a", "b", "open")
    ]
    closure = engine.set_tolerance_scale("G03_OVERLAP_COINCIDENT", 0.75)
    steps.append(_incremental_step(engine, "tighten_tolerance", "G03_OVERLAP_COINCIDENT", roots, closure, before))

    full_checks = {}
    for gid in sorted(engine.cases):
        incremental = engine.glyph_summary(gid)
        full = engine.recompute_glyph_full(gid)
        full_checks[gid] = {
            "incremental_hash": incremental["content_hash"],
            "full_rebuild_hash": full["content_hash"],
            "equal": incremental["content_hash"] == full["content_hash"],
        }
        if not full_checks[gid]["equal"]:
            raise GlyphError(ErrorCode.CLOSURE_MISMATCH, "incremental and full rebuild differ", glyph_id=gid)

    return {
        "initial_task_count": len(initial_tasks),
        "baseline_glyph_hashes": baseline,
        "steps": steps,
        "final_glyph_hashes": engine.all_glyph_hashes(),
        "full_rebuild_checks": full_checks,
    }


def _incremental_step(
    engine: IncrementalEngine,
    name: str,
    glyph_id: str,
    roots: list[str],
    closure: list[str],
    before_hashes: dict[str, str],
) -> dict[str, Any]:
    actual = engine.recompute(closure)
    after_hashes = engine.all_glyph_hashes()
    if actual != closure:
        raise GlyphError(
            ErrorCode.CLOSURE_MISMATCH,
            "actual recompute list differs from closure",
            step=name,
            closure=closure,
            actual=actual,
        )
    unaffected = {
        gid: before_hashes[gid] == after_hashes[gid]
        for gid in sorted(before_hashes)
        if gid != glyph_id
    }
    if not all(unaffected.values()):
        raise GlyphError(
            ErrorCode.CLOSURE_MISMATCH,
            "a glyph outside the invalidation closure was rebuilt",
            step=name,
            hashes=unaffected,
        )
    full = engine.recompute_glyph_full(glyph_id)
    return {
        "step": name,
        "glyph_id": glyph_id,
        "roots": roots,
        "closure": closure,
        "closure_count": len(closure),
        "actual_recomputed": actual,
        "actual_matches_closure": actual == closure,
        "outside_glyph_hashes_unchanged": unaffected,
        "changed_glyph": before_hashes[glyph_id] != after_hashes[glyph_id],
        "incremental_summary_hash": after_hashes[glyph_id],
        "full_rebuild_summary_hash": full["content_hash"],
        "incremental_matches_full": after_hashes[glyph_id] == full["content_hash"],
    }


def _error_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        bad_node = OutlineNode(Point(0.0, 0.0), SegmentType.LINE, (Point(1.0, 1.0),))
        Contour((OutlineNode(Point(0.0, 0.0), SegmentType.MOVE), bad_node)).validate()
    except GlyphError as exc:
        checks["E_OUTLINE_INVALID"] = exc.code.value
    try:
        stroke_polyline([(0.0, 0.0), (1.0, 0.0)], StrokeOptions(0.0))
    except GlyphError as exc:
        checks["E_STROKE_WIDTH"] = exc.code.value
    try:
        stroke_polyline([(0.0, 0.0), (1.0, 0.0)], StrokeOptions(1.0, cap="bad"))
    except GlyphError as exc:
        checks["E_CAP_JOIN"] = exc.code.value
    checks.update(
        {
            "E_BOOLEAN_DEGENERATE": "enforced by boundary reconstruction and invalid geometry checks",
            "E_CONSISTENCY_DRIFT": "enforced by sensitivity classification outside error interval",
            "E_CLOSURE_MISMATCH": "enforced by incremental closure/topological audit",
            "E_BITMAP_HASH": "enforced by boundary/direct bitmap equality checks",
        }
    )
    return checks


def _round(value: float) -> float:
    return float(f"{value:.12f}")
