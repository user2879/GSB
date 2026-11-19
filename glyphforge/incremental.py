"""Task-granular incremental recomputation with explicit invalidation closure."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .boolean import BooleanOp, ResolutionOrder, boolean_operation
from .errors import ErrorCode, GlyphError
from .flatten import FlattenOptions, FlattenedOutline, flatten_contours
from .model import Contour, GlyphCase, GlyphSuite, Point, canonical_json_bytes
from .raster import (
    check_bitmap_consistency,
    rasterize_boolean,
    rasterize_stroke,
)
from .stroke import StrokeOptions, StrokeResult, stroke_polyline


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    glyph_id: str
    kind: str
    dependencies: tuple[str, ...]
    meta: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CacheValue:
    task_id: str
    kind: str
    content_hash: str
    value: Any


class IncrementalEngine:
    """A dependency graph whose cache nodes are recomputed only inside a closure."""

    def __init__(
        self,
        suite: GlyphSuite,
        *,
        resolution_order: ResolutionOrder | str = ResolutionOrder.LEXICOGRAPHIC,
    ):
        suite.validate()
        self.cases: dict[str, GlyphCase] = {case.glyph_id: case for case in suite.cases}
        self.seed = suite.seed
        self.resolution_order = ResolutionOrder(resolution_order)
        self.resolution_orders: dict[str, ResolutionOrder] = {
            case.glyph_id: self.resolution_order for case in suite.cases
        }
        self.tolerance_scales: dict[str, float] = {case.glyph_id: 1.0 for case in suite.cases}
        self.specs, self.spec_by_id, self.children = _build_specs(self.cases)
        self.cache: dict[str, CacheValue] = {}
        self.last_recomputed: list[str] = []
        self.last_closure: list[str] = []

    @property
    def suite(self) -> GlyphSuite:
        return GlyphSuite(self.seed, tuple(self.cases[gid] for gid in sorted(self.cases)))

    def build_all(self) -> list[str]:
        executed: list[str] = []
        for spec in self.specs:
            self.cache[spec.task_id] = self._compute(spec)
            executed.append(spec.task_id)
        self.last_recomputed = executed
        self.last_closure = executed.copy()
        return executed

    def closure(self, roots: Iterable[str]) -> list[str]:
        root_set = set(roots)
        for root in root_set:
            if root not in self.spec_by_id:
                raise GlyphError(ErrorCode.CLOSURE_MISMATCH, "unknown invalidation root", task_id=root)
        closure = set(root_set)
        queue = list(sorted(root_set))
        while queue:
            current = queue.pop(0)
            for child in self.children.get(current, ()):  # deterministic insertion order
                if child not in closure:
                    closure.add(child)
                    queue.append(child)
        return [spec.task_id for spec in self.specs if spec.task_id in closure]

    def recompute(self, roots: Iterable[str]) -> list[str]:
        self.last_closure = self.closure(roots)
        closure_set = set(self.last_closure)
        recovered: list[str] = []
        for spec in self.specs:
            if spec.task_id not in closure_set:
                continue
            missing = [dep for dep in spec.dependencies if dep not in self.cache]
            if missing:
                raise GlyphError(
                    ErrorCode.CLOSURE_MISMATCH,
                    "closure omitted a required dependency",
                    task_id=spec.task_id,
                    missing=missing,
                )
            self.cache[spec.task_id] = self._compute(spec)
            recovered.append(spec.task_id)
        if set(recovered) != closure_set or recovered != self.last_closure:
            raise GlyphError(
                ErrorCode.CLOSURE_MISMATCH,
                "actual recomputation set differs from invalidation closure",
                closure=self.last_closure,
                actual=recovered,
            )
        self.last_recomputed = recovered
        return recovered

    def task(self, task_id: str) -> CacheValue:
        if task_id not in self.cache:
            raise GlyphError(ErrorCode.CLOSURE_MISMATCH, "task not computed", task_id=task_id)
        return self.cache[task_id]

    def task_hash(self, task_id: str) -> str:
        return self.task(task_id).content_hash

    def glyph_summary(self, glyph_id: str) -> dict[str, Any]:
        summary = self.task(f"summary/{glyph_id}")
        return summary.value

    def all_glyph_hashes(self) -> dict[str, str]:
        return {gid: self.glyph_summary(gid)["content_hash"] for gid in sorted(self.cases)}

    def edit_control_point(
        self,
        glyph_id: str,
        source: str,
        contour_index: int,
        node_index: int,
        control_index: int,
        point: Point,
    ) -> list[str]:
        case = self._case(glyph_id)
        contour = _select_contour(case, source, contour_index)
        if node_index < 0 or node_index >= len(contour.nodes):
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "node index out of range")
        node = contour.nodes[node_index]
        if control_index < 0 or control_index >= len(node.controls):
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "control index out of range")
        controls = list(node.controls)
        controls[control_index] = point
        nodes = list(contour.nodes)
        nodes[node_index] = replace(node, controls=tuple(controls))
        new_contour = Contour(tuple(nodes))
        new_contour.validate(require_closed=(source != "open"), context=f"{glyph_id}.{source}")
        self._replace_contour(case, source, contour_index, new_contour)
        roots = [f"flat/{glyph_id}/{size}/{source}" for size in case.suggested_sizes]
        return self.closure(roots)

    def edit_node_point(
        self,
        glyph_id: str,
        source: str,
        contour_index: int,
        node_index: int,
        point: Point,
    ) -> list[str]:
        case = self._case(glyph_id)
        contour = _select_contour(case, source, contour_index)
        if node_index < 0 or node_index >= len(contour.nodes):
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "node index out of range")
        nodes = list(contour.nodes)
        nodes[node_index] = replace(nodes[node_index], point=point)
        if contour.closed:
            if node_index == 0:
                nodes[-1] = replace(nodes[-1], point=point)
            elif node_index == len(nodes) - 1:
                nodes[0] = replace(nodes[0], point=point)
        new_contour = Contour(tuple(nodes))
        new_contour.validate(require_closed=(source != "open"), context=f"{glyph_id}.{source}")
        self._replace_contour(case, source, contour_index, new_contour)
        roots = [f"flat/{glyph_id}/{size}/{source}" for size in case.suggested_sizes]
        return self.closure(roots)

    def set_stroke_parameters(
        self,
        glyph_id: str,
        *,
        width: float | None = None,
        cap: str | None = None,
        join: str | None = None,
    ) -> list[str]:
        case = self._case(glyph_id)
        new_case = replace(
            case,
            stroke_width=case.stroke_width if width is None else float(width),
            stroke_cap=case.stroke_cap if cap is None else cap,
            stroke_join=case.stroke_join if join is None else join,
        )
        new_case.validate()
        self.cases[glyph_id] = new_case
        roots = [f"stroke/{glyph_id}/{size}" for size in new_case.suggested_sizes]
        return self.closure(roots)

    def set_tolerance_scale(self, glyph_id: str, scale: float) -> list[str]:
        if scale <= 0.0:
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "tolerance scale must be positive")
        self._case(glyph_id)
        self.tolerance_scales[glyph_id] = float(scale)
        case = self.cases[glyph_id]
        roots: list[str] = []
        for size in case.suggested_sizes:
            for source in ("a", "b", "open"):
                roots.append(f"flat/{glyph_id}/{size}/{source}")
        return self.closure(roots)

    def set_resolution_order(self, glyph_id: str, order: ResolutionOrder | str) -> list[str]:
        self._case(glyph_id)
        self.resolution_orders[glyph_id] = ResolutionOrder(order)
        case = self.cases[glyph_id]
        roots = [f"bool/{glyph_id}/{size}/{op.value}" for size in case.suggested_sizes for op in BooleanOp]
        roots.extend(f"stroke/{glyph_id}/{size}" for size in case.suggested_sizes)
        return self.closure(roots)

    def recompute_glyph_full(self, glyph_id: str) -> dict[str, Any]:
        """Fresh engine with current case/config, used as an incremental oracle."""

        case = self._case(glyph_id)
        temporary = GlyphSuite(self.seed, (case,))
        engine = IncrementalEngine(temporary, resolution_order=self.resolution_orders[glyph_id])
        engine.tolerance_scales[glyph_id] = self.tolerance_scales[glyph_id]
        engine.build_all()
        return engine.glyph_summary(glyph_id)

    def _case(self, glyph_id: str) -> GlyphCase:
        if glyph_id not in self.cases:
            raise GlyphError(ErrorCode.CLOSURE_MISMATCH, "unknown glyph", glyph_id=glyph_id)
        return self.cases[glyph_id]

    def _replace_contour(self, case: GlyphCase, source: str, contour_index: int, contour: Contour) -> None:
        if source == "a":
            contours = list(case.source_a)
            contours[contour_index] = contour
            new_case = replace(case, source_a=tuple(contours))
        elif source == "b":
            contours = list(case.source_b)
            contours[contour_index] = contour
            new_case = replace(case, source_b=tuple(contours))
        elif source == "open":
            if contour_index != 0:
                raise GlyphError(ErrorCode.OUTLINE_INVALID, "open path contour index must be 0")
            new_case = replace(case, open_path=contour)
        else:
            raise GlyphError(ErrorCode.OUTLINE_INVALID, "source must be a, b, or open")
        new_case.validate()
        self.cases[case.glyph_id] = new_case

    def _compute(self, spec: TaskSpec) -> CacheValue:
        meta = spec.meta
        gid = spec.glyph_id
        case = self.cases[gid]
        if spec.kind == "flat":
            size = int(meta["size"])
            source = str(meta["source"])
            contours = case.source_a if source == "a" else case.source_b if source == "b" else (case.open_path,)
            options = FlattenOptions(size_px=size, tolerance_scale=self.tolerance_scales[gid])
            value = flatten_contours(contours, options, source=source)
            return CacheValue(spec.task_id, spec.kind, value.content_hash(), value)
        if spec.kind == "bool":
            size = int(meta["size"])
            op = BooleanOp(meta["operation"])
            flat_a: FlattenedOutline = self.task(f"flat/{gid}/{size}/a").value
            flat_b: FlattenedOutline = self.task(f"flat/{gid}/{size}/b").value
            value = boolean_operation(
                flat_a.loops,
                flat_b.loops,
                op,
                resolution_order=self.resolution_orders[gid],
            )
            return CacheValue(spec.task_id, spec.kind, value.boundary_hash(), value)
        if spec.kind == "stroke":
            size = int(meta["size"])
            flat_open: FlattenedOutline = self.task(f"flat/{gid}/{size}/open").value
            options = StrokeOptions(
                width=case.stroke_width,
                cap=case.stroke_cap,
                join=case.stroke_join,
                tolerance=flat_open.tolerance,
            )
            value = stroke_polyline(
                flat_open.open_points,
                options,
                resolution_order=self.resolution_orders[gid],
            )
            return CacheValue(spec.task_id, spec.kind, value.boundary_hash(), value)
        if spec.kind == "bitmap_bool":
            size = int(meta["size"])
            op = BooleanOp(meta["operation"])
            boolean_result = self.task(f"bool/{gid}/{size}/{op.value}").value
            boundary, direct = rasterize_boolean(boolean_result, size)
            consistency = check_bitmap_consistency(
                boundary,
                direct,
                boundary_loops=boolean_result.loops,
                direct_loops=boolean_result.direct_loops,
                declared_tolerance=_flatten_tolerance(self.task(f"flat/{gid}/{size}/a").value),
            )
            if not consistency.consistent:
                raise GlyphError(ErrorCode.BITMAP_HASH, "boolean bitmap hash mismatch", consistency.as_dict())
            payload = {
                "boundary": boundary.as_payload(),
                "direct": direct.as_payload(),
                "consistency": consistency.as_dict(),
            }
            return CacheValue(spec.task_id, spec.kind, _hash_payload(payload), payload)
        if spec.kind == "bitmap_stroke":
            size = int(meta["size"])
            stroke_result: StrokeResult = self.task(f"stroke/{gid}/{size}").value
            boundary, direct = rasterize_stroke(stroke_result, size)
            consistency = check_bitmap_consistency(
                boundary,
                direct,
                boundary_loops=stroke_result.loops,
                direct_loops=stroke_result.direct_loops,
                declared_tolerance=stroke_result.tolerance,
            )
            if not consistency.consistent:
                raise GlyphError(ErrorCode.BITMAP_HASH, "stroke bitmap hash mismatch", consistency.as_dict())
            payload = {
                "boundary": boundary.as_payload(),
                "direct": direct.as_payload(),
                "consistency": consistency.as_dict(),
            }
            return CacheValue(spec.task_id, spec.kind, _hash_payload(payload), payload)
        if spec.kind == "summary":
            child_hashes = [
                {"task": dep, "hash": self.task(dep).content_hash}
                for dep in spec.dependencies
            ]
            payload = {
                "glyph_id": gid,
                "content_hash": _hash_payload(child_hashes),
                "tasks": child_hashes,
            }
            return CacheValue(spec.task_id, spec.kind, payload["content_hash"], payload)
        raise GlyphError(ErrorCode.CLOSURE_MISMATCH, f"unknown task kind: {spec.kind}")


def _build_specs(cases: dict[str, GlyphCase]) -> tuple[list[TaskSpec], dict[str, TaskSpec], dict[str, list[str]]]:
    specs: list[TaskSpec] = []
    for gid in sorted(cases):
        case = cases[gid]
        for size in case.suggested_sizes:
            for source in ("a", "b", "open"):
                specs.append(
                    TaskSpec(
                        f"flat/{gid}/{size}/{source}",
                        gid,
                        "flat",
                        (),
                        {"size": size, "source": source},
                    )
                )
        for size in case.suggested_sizes:
            deps = (f"flat/{gid}/{size}/a", f"flat/{gid}/{size}/b")
            for op in BooleanOp:
                specs.append(
                    TaskSpec(
                        f"bool/{gid}/{size}/{op.value}",
                        gid,
                        "bool",
                        deps,
                        {"size": size, "operation": op.value},
                    )
                )
        for size in case.suggested_sizes:
            specs.append(
                TaskSpec(
                    f"stroke/{gid}/{size}",
                    gid,
                    "stroke",
                    (f"flat/{gid}/{size}/open",),
                    {"size": size},
                )
            )
        for size in case.suggested_sizes:
            for op in BooleanOp:
                specs.append(
                    TaskSpec(
                        f"bitmap/bool/{gid}/{size}/{op.value}",
                        gid,
                        "bitmap_bool",
                        (f"bool/{gid}/{size}/{op.value}",),
                        {"size": size, "operation": op.value},
                    )
                )
            specs.append(
                TaskSpec(
                    f"bitmap/stroke/{gid}/{size}",
                    gid,
                    "bitmap_stroke",
                    (f"stroke/{gid}/{size}",),
                    {"size": size},
                )
            )
        child_ids = [spec.task_id for spec in specs if spec.glyph_id == gid]
        specs.append(TaskSpec(f"summary/{gid}", gid, "summary", tuple(child_ids), {}))

    spec_by_id = {spec.task_id: spec for spec in specs}
    children: dict[str, list[str]] = {spec.task_id: [] for spec in specs}
    for spec in specs:
        for dependency in spec.dependencies:
            children.setdefault(dependency, []).append(spec.task_id)
    return specs, spec_by_id, children


def _select_contour(case: GlyphCase, source: str, contour_index: int) -> Contour:
    if source == "a":
        contours = case.source_a
    elif source == "b":
        contours = case.source_b
    elif source == "open":
        contours = (case.open_path,)
    else:
        raise GlyphError(ErrorCode.OUTLINE_INVALID, "source must be a, b, or open")
    if contour_index < 0 or contour_index >= len(contours):
        raise GlyphError(ErrorCode.OUTLINE_INVALID, "contour index out of range")
    return contours[contour_index]


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _flatten_tolerance(flattened: FlattenedOutline) -> float:
    return flattened.tolerance
