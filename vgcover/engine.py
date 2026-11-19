"""Scene rendering, ledger verification and incremental invalidation closure."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .composite import (
    COMPOSITE_POLICY,
    CompositeResult,
    SourceLayer,
    build_source_layer,
    compose_layers,
)
from .errors import CoverageError, ErrorCode
from .geometry import discretize_region
from .hashing import hash_float32, hash_int32, hash_uint8, sha256_json
from .ledger import assemble_ledger, ledger_hash
from .quantize import QUANTIZATION_POLICY, QuantizationResult, quantize_coverage
from .raster import DualCoverage, rasterize_geometry_dual
from .scene import clone_scene, scene_hash, scene_by_id, validate_scene


LEDGER_LAYER_DOMAIN = "vgcover.region_layer_ledger.v1"


@dataclass
class RegionState:
    region: dict[str, Any]
    position: int
    input_hash: str
    geometry: dict[str, Any]
    geometry_hash: str
    dual: DualCoverage
    selected_coverage: np.ndarray
    coverage_hashes: dict[str, str]
    quantization: QuantizationResult
    layer: SourceLayer
    layer_hash: str
    ledger_record: dict[str, Any]


@dataclass
class SceneRender:
    scene: dict[str, Any]
    scene_hash: str
    width: int
    height: int
    regions: dict[int, RegionState]
    composite: CompositeResult
    ledger: dict[str, Any]

    @property
    def buffer(self) -> np.ndarray:
        return self.composite.buffer

    @property
    def region_ids(self) -> list[int]:
        return sorted(self.regions)

    def ledger_inputs(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        records = [copy.deepcopy(self.regions[region_id].ledger_record) for region_id in self.region_ids]
        return records, _image_record(self.composite)

    def summary(self) -> dict[str, Any]:
        region_summaries = []
        for region_id in self.region_ids:
            state = self.regions[region_id]
            ledger_entry = next(
                entry for entry in self.ledger["regions"] if entry["region_id"] == region_id
            )
            region_summaries.append(
                {
                    "region_id": region_id,
                    "category": state.region.get("category"),
                    "fill_rule": state.region["fill_rule"],
                    "tolerance": state.region["tolerance"],
                    "alpha": state.region["alpha"],
                    "geometry_hash": state.geometry_hash,
                    "coverage_hashes": state.coverage_hashes,
                    "gray_hash": state.quantization.gray_hash,
                    "layer_content_hash": state.layer.layer_content_hash,
                    "layer_hash": state.layer_hash,
                    "coverage_nonzero_pixels": int(np.count_nonzero(state.dual.nonzero.coverage)),
                    "coverage_evenodd_pixels": int(np.count_nonzero(state.dual.evenodd.coverage)),
                    "rule_difference_pixels": state.dual.difference_pixel_count,
                    "edge_count": state.dual.edge_count,
                    "geometry": dict(state.geometry["summary"]),
                    "quantization": copy.deepcopy(state.quantization.distribution),
                    "ledger_total_error_bound": ledger_entry["total_error_bound"],
                    "ledger_complete": ledger_entry["complete"],
                }
            )
        return {
            "scene_hash": self.scene_hash,
            "canvas": {"width": self.width, "height": self.height},
            "regions": region_summaries,
            "composite": {
                "buffer_hash": self.composite.buffer_hash,
                "layer_sequence": list(self.composite.layer_sequence),
                "layer_content_hashes": {
                    str(region_id): value
                    for region_id, value in sorted(self.composite.layer_hashes.items())
                },
                "steps": [_step_record(step) for step in self.composite.steps],
            },
            "ledger": {
                "hash": ledger_hash(self.ledger),
                "complete": self.ledger["complete"],
                "image_total_error_bound": self.ledger["image"]["total_error_bound"],
            },
        }


def _coverage_hashes(dual: DualCoverage, selected_rule: str) -> dict[str, str]:
    selected = dual.nonzero if selected_rule == "nonzero" else dual.evenodd
    return {
        "nonzero": hash_float32(dual.nonzero.coverage),
        "evenodd": hash_float32(dual.evenodd.coverage),
        "selected": hash_float32(selected.coverage),
        "difference": hash_float32(dual.difference),
        "dependency_edges": hash_int32(dual.nonzero.dependency_edge_counts),
    }


def _layer_hash_payload(
    *,
    region: dict[str, Any],
    position: int,
    input_hash: str,
    geometry_hash: str,
    coverage_hashes: dict[str, str],
    quantization: QuantizationResult,
    layer: SourceLayer,
) -> dict[str, Any]:
    return {
        "domain": LEDGER_LAYER_DOMAIN,
        "region_id": int(region["region_id"]),
        "position": int(position),
        "input_hash": input_hash,
        "geometry_hash": geometry_hash,
        "coverage_hashes": dict(coverage_hashes),
        "selected_fill_rule": region["fill_rule"],
        "quantization": {
            "policy_version": quantization.policy["version"],
            "gray_hash": quantization.gray_hash,
            "quant_error_hash": quantization.quant_error_hash,
        },
        "fill_color": list(region["fill_color"]),
        "alpha": float(region["alpha"]),
        "layer_content_hash": layer.layer_content_hash,
        "layer_pixels_hash": hash_uint8(layer.bgra),
    }


def _make_layer_hash(**kwargs: Any) -> str:
    return sha256_json(_layer_hash_payload(**kwargs))


def _region_ledger_record(
    *,
    region: dict[str, Any],
    position: int,
    width: int,
    height: int,
    input_hash: str,
    geometry: dict[str, Any],
    geometry_hash: str,
    coverage_hashes: dict[str, str],
    quantization: QuantizationResult,
    layer: SourceLayer,
    layer_hash: str,
) -> dict[str, Any]:
    delta_px = float(geometry["summary"]["max_analytic_error_bound_px"])
    storage_rounding_px = float(
        geometry["policy"]["coordinate_rounding_error_bound_px"]
    )
    effective_delta_px = delta_px + storage_rounding_px
    measured_px = float(geometry["summary"]["max_measured_chord_height_px"])
    perimeter_px = float(geometry["summary"]["perimeter_px"])
    band_area_px2 = (
        2.0 * effective_delta_px * perimeter_px
        + math.pi * (2.0 * effective_delta_px) ** 2
    )
    geometry_coverage_error = min(1.0, band_area_px2 / float(width * height))
    quant_coverage_error = min(1.0, float(quantization.max_abs_error))
    composite_bound_u8 = float(COMPOSITE_POLICY["per_step_channel_rounding_error_bound_u8"])
    composite_coverage_error = composite_bound_u8 / 255.0

    return {
        "region_id": int(region["region_id"]),
        "category": region.get("category"),
        "position": int(position),
        "alpha": float(region["alpha"]),
        "fill_color": list(region["fill_color"]),
        "input_hash": input_hash,
        "geometry_hash": geometry_hash,
        "coverage_hashes": dict(coverage_hashes),
        "selected_fill_rule": region["fill_rule"],
        "gray_hash": quantization.gray_hash,
        "quant_error_hash": quantization.quant_error_hash,
        "layer_hash": layer_hash,
        "layer_content_hash": layer.layer_content_hash,
        "layer_pixels_hash": hash_uint8(layer.bgra),
        "components": {
            "geometry": {
                "present": True,
                "max_analytic_error_bound_px": delta_px,
                "coordinate_rounding_error_bound_px": storage_rounding_px,
                "effective_error_bound_px": effective_delta_px,
                "max_measured_chord_height_px": measured_px,
                "polyline_perimeter_px": perimeter_px,
                "boundary_band_area_px2": band_area_px2,
                "coverage_equivalent_error": geometry_coverage_error,
                "formula": "min(1,(2*delta_effective*perimeter+pi*(2*delta_effective)^2)/(width*height))",
            },
            "quantization": {
                "present": True,
                "policy_version": QUANTIZATION_POLICY["version"],
                "step": QUANTIZATION_POLICY["step"],
                "rounding": QUANTIZATION_POLICY["rounding"],
                "max_abs_error": float(quantization.max_abs_error),
                "reconstruction_max_abs_error": float(
                    quantization.max_reconstruction_abs_error
                ),
                "max_abs_error_bound": QUANTIZATION_POLICY["max_abs_error_bound"],
                "over_tolerance_pixels": int(quantization.over_tolerance_pixels),
                "total_pixels": int(quantization.total_pixels),
                "coverage_equivalent_error": quant_coverage_error,
            },
            "composite": {
                "present": True,
                "policy_version": COMPOSITE_POLICY["version"],
                "order_position": int(position),
                "rounding_error_bound_u8": composite_bound_u8,
                "coverage_equivalent_error": composite_coverage_error,
            },
        },
        "diagnostics": {
            "geometry": copy.deepcopy(geometry["summary"]),
            "quantization": copy.deepcopy(quantization.distribution),
            "rule_difference_pixels": int(
                0 if quantization.gray.size == 0 else 0
            ),
        },
    }


def _step_record(step: Any) -> dict[str, Any]:
    return {
        "position": int(step.position),
        "region_id": int(step.region_id),
        "layer_content_hash": step.layer_content_hash,
        "nonzero_source_pixels": int(step.nonzero_source_pixels),
        "step_rounding_error_bound_u8": float(step.step_rounding_error_bound_u8),
        "cumulative_rounding_error_bound_u8": float(step.cumulative_rounding_error_bound_u8),
        "step_rounding_error_bound_coverage": float(step.step_rounding_error_bound_coverage),
    }


def _image_record(composite: CompositeResult) -> dict[str, Any]:
    return {
        "buffer_hash": composite.buffer_hash,
        "layer_sequence": list(composite.layer_sequence),
        "layer_hashes": {
            int(region_id): value for region_id, value in composite.layer_hashes.items()
        },
        "steps": [_step_record(step) for step in composite.steps],
    }


def _compute_region(
    region: dict[str, Any],
    *,
    position: int,
    width: int,
    height: int,
) -> RegionState:
    geometry = discretize_region(region)
    geometry_hash = str(geometry["geometry_hash"])
    dual = rasterize_geometry_dual(geometry, width, height)
    selected = dual.selected(region["fill_rule"])
    quantization = quantize_coverage(selected.coverage)
    coverage_hashes = _coverage_hashes(dual, region["fill_rule"])
    layer = build_source_layer(
        region_id=int(region["region_id"]),
        gray=quantization.gray,
        fill_color=region["fill_color"],
        alpha=float(region["alpha"]),
    )
    input_hash = sha256_json(region)
    layer_hash = _make_layer_hash(
        region=region,
        position=position,
        input_hash=input_hash,
        geometry_hash=geometry_hash,
        coverage_hashes=coverage_hashes,
        quantization=quantization,
        layer=layer,
    )
    record = _region_ledger_record(
        region=region,
        position=position,
        width=width,
        height=height,
        input_hash=input_hash,
        geometry=geometry,
        geometry_hash=geometry_hash,
        coverage_hashes=coverage_hashes,
        quantization=quantization,
        layer=layer,
        layer_hash=layer_hash,
    )
    record["diagnostics"]["rule_difference_pixels"] = int(dual.difference_pixel_count)
    return RegionState(
        region=region,
        position=position,
        input_hash=input_hash,
        geometry=geometry,
        geometry_hash=geometry_hash,
        dual=dual,
        selected_coverage=selected.coverage,
        coverage_hashes=coverage_hashes,
        quantization=quantization,
        layer=layer,
        layer_hash=layer_hash,
        ledger_record=record,
    )


def render_scene(scene: dict[str, Any]) -> SceneRender:
    """Render a complete scene and build its error ledger."""

    validate_scene(scene)
    width = int(scene["width"])
    height = int(scene["height"])
    sorted_regions = sorted(scene["regions"], key=lambda region: int(region["region_id"]))
    states: dict[int, RegionState] = {}
    layers: list[SourceLayer] = []
    for position, region in enumerate(sorted_regions):
        state = _compute_region(region, position=position, width=width, height=height)
        states[int(region["region_id"])] = state
        layers.append(state.layer)
    composite = compose_layers(layers)
    records = [copy.deepcopy(states[region_id].ledger_record) for region_id in sorted(states)]
    ledger = assemble_ledger(
        scene,
        records,
        _image_record(composite),
        scene_hash=scene_hash(scene),
    )
    return SceneRender(
        scene=scene,
        scene_hash=scene_hash(scene),
        width=width,
        height=height,
        regions=states,
        composite=composite,
        ledger=ledger,
    )


def verify_ledger(scene: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    """Recompute a scene from scratch and verify every ledger link.

    The only inputs are the scene document and the ledger policy that is
    embedded in the ledger. A mismatch raises ``E_LEDGER_MISMATCH`` and names a
    region when the discrepancy is region-local.
    """

    if not isinstance(ledger, dict):
        raise CoverageError(ErrorCode.LEDGER_MISMATCH, "ledger must be an object")
    expected_render = render_scene(scene)
    expected = expected_render.ledger
    if ledger.get("schema") != expected["schema"]:
        raise CoverageError(ErrorCode.LEDGER_MISMATCH, "ledger schema mismatch")
    if ledger.get("scene_hash") != expected["scene_hash"]:
        raise CoverageError(
            ErrorCode.LEDGER_MISMATCH,
            "ledger scene hash does not match the supplied scene",
            details={"ledger": ledger.get("scene_hash"), "recomputed": expected["scene_hash"]},
        )

    actual_regions = ledger.get("regions")
    if not isinstance(actual_regions, list):
        raise CoverageError(ErrorCode.LEDGER_MISMATCH, "ledger.regions must be a list")
    expected_by_id = {int(entry["region_id"]): entry for entry in expected["regions"]}
    actual_by_id: dict[int, dict[str, Any]] = {}
    for entry in actual_regions:
        if not isinstance(entry, dict) or "region_id" not in entry:
            raise CoverageError(ErrorCode.LEDGER_MISMATCH, "malformed region ledger entry")
        region_id = int(entry["region_id"])
        if region_id in actual_by_id:
            raise CoverageError(
                ErrorCode.LEDGER_MISMATCH,
                "duplicate region ledger entry",
                region_id=region_id,
            )
        actual_by_id[region_id] = entry
    if set(actual_by_id) != set(expected_by_id):
        raise CoverageError(
            ErrorCode.LEDGER_MISMATCH,
            "ledger region IDs do not match the recomputed scene",
            details={
                "ledger_region_ids": sorted(actual_by_id),
                "recomputed_region_ids": sorted(expected_by_id),
            },
        )
    for region_id in sorted(expected_by_id):
        if actual_by_id[region_id] != expected_by_id[region_id]:
            raise CoverageError(
                ErrorCode.LEDGER_MISMATCH,
                "region ledger entry does not match independent recomputation",
                region_id=region_id,
            )

    if ledger.get("image") != expected["image"]:
        raise CoverageError(
            ErrorCode.LEDGER_MISMATCH,
            "image ledger entry does not match independent recomputation",
        )
    if ledger.get("ledger_policy") != expected["ledger_policy"]:
        raise CoverageError(
            ErrorCode.LEDGER_MISMATCH,
            "ledger policy does not match the declared recomputation policy",
        )
    if ledger.get("complete") != expected["complete"]:
        raise CoverageError(ErrorCode.LEDGER_MISMATCH, "ledger completeness flag mismatch")
    return {
        "ok": True,
        "scene_hash": expected["scene_hash"],
        "buffer_hash": expected_render.composite.buffer_hash,
        "checked_region_ids": sorted(expected_by_id),
        "complete": expected["complete"],
        "ledger_hash": ledger_hash(expected),
    }


@dataclass
class ChangePlan:
    changes: list[dict[str, Any]]
    scene: dict[str, Any]
    scene_hash: str
    affected_region_ids: list[int]
    changed_fields_by_region: dict[int, list[str]]
    closure: list[str]


@dataclass
class UpdateReceipt:
    previous_scene_hash: str
    new_scene_hash: str
    affected_region_ids: list[int]
    changed_fields_by_region: dict[int, list[str]]
    closure: list[str]
    recomputed: list[str]
    unchanged_region_ids: list[int]
    outside_layer_hashes_preserved: bool
    outside_ledger_entries_preserved: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_scene_hash": self.previous_scene_hash,
            "new_scene_hash": self.new_scene_hash,
            "affected_region_ids": list(self.affected_region_ids),
            "changed_fields_by_region": {
                str(region_id): list(fields)
                for region_id, fields in sorted(self.changed_fields_by_region.items())
            },
            "closure": list(self.closure),
            "recomputed": list(self.recomputed),
            "actual_matches_closure": sorted(set(self.recomputed)) == sorted(set(self.closure)),
            "closure_size": len(self.closure),
            "recomputed_size": len(set(self.recomputed)),
            "unchanged_region_ids": list(self.unchanged_region_ids),
            "outside_layer_hashes_preserved": bool(self.outside_layer_hashes_preserved),
            "outside_ledger_entries_preserved": bool(self.outside_ledger_entries_preserved),
        }


GEOMETRY_FIELDS = {"control_points", "segment_control_points", "tolerance"}
RULE_FIELDS = {"fill_rule"}
LAYER_FIELDS = {"alpha", "fill_color"}
ALLOWED_FIELDS = GEOMETRY_FIELDS | RULE_FIELDS | LAYER_FIELDS


def _closure_for_field(region_id: int, field_name: str) -> list[str]:
    prefix = f"r{region_id}"
    if field_name in GEOMETRY_FIELDS:
        nodes = [
            f"geometry:{prefix}",
            f"coverage_nonzero:{prefix}",
            f"coverage_evenodd:{prefix}",
            f"coverage_selected:{prefix}",
            f"quantize:{prefix}",
            f"layer:{prefix}",
            f"ledger_region:{prefix}",
            "composite",
            "ledger_image",
        ]
    elif field_name in RULE_FIELDS:
        nodes = [
            f"coverage_selected:{prefix}",
            f"quantize:{prefix}",
            f"layer:{prefix}",
            f"ledger_region:{prefix}",
            "composite",
            "ledger_image",
        ]
    else:
        nodes = [
            f"layer:{prefix}",
            f"ledger_region:{prefix}",
            "composite",
            "ledger_image",
        ]
    return nodes


def _apply_change(scene: dict[str, Any], change: dict[str, Any]) -> tuple[int, str]:
    if not isinstance(change, dict):
        raise CoverageError(ErrorCode.SCENE_INVALID, "change must be an object")
    region_id = change.get("region_id")
    if not isinstance(region_id, int) or isinstance(region_id, bool):
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "change.region_id must be an integer",
            details={"received": region_id},
        )
    field_name = change.get("field")
    if field_name not in ALLOWED_FIELDS:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "unsupported incremental change field",
            region_id=region_id,
            details={"field": field_name, "allowed": sorted(ALLOWED_FIELDS)},
        )
    regions = scene_by_id(scene)
    if region_id not in regions:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "incremental change targets an unknown region_id",
            region_id=region_id,
        )
    region = regions[region_id]
    if "value" not in change:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "incremental change requires a value",
            region_id=region_id,
            details={"field": field_name},
        )
    value = copy.deepcopy(change["value"])

    if field_name in {"control_points", "segment_control_points"}:
        contour_index = change.get("contour_index")
        segment_index = change.get("segment_index")
        if not isinstance(contour_index, int) or isinstance(contour_index, bool):
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "control point change requires integer contour_index",
                region_id=region_id,
                details={"contour_index": contour_index},
            )
        if not isinstance(segment_index, int) or isinstance(segment_index, bool):
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "control point change requires integer segment_index",
                region_id=region_id,
                contour_index=contour_index,
                details={"segment_index": segment_index},
            )
        if not 0 <= contour_index < len(region["contours"]):
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "contour_index is outside the region",
                region_id=region_id,
                contour_index=contour_index,
            )
        contour = region["contours"][contour_index]
        if not 0 <= segment_index < len(contour["segments"]):
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "segment_index is outside the contour",
                region_id=region_id,
                contour_index=contour_index,
                segment_index=segment_index,
            )
        contour["segments"][segment_index]["control_points"] = value
    elif field_name in {"fill_rule", "tolerance", "alpha", "fill_color"}:
        region[field_name] = value
    else:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "unsupported incremental change field",
            region_id=region_id,
            details={"field": field_name},
        )
    return region_id, field_name


class IncrementalRenderer:
    """Incremental scene renderer with an exact node invalidation closure."""

    def __init__(self, scene: dict[str, Any]) -> None:
        self._scene = clone_scene(scene)
        rendered = render_scene(self._scene)
        self._scene = rendered.scene
        self._scene_hash = rendered.scene_hash
        self._width = rendered.width
        self._height = rendered.height
        self._states = rendered.regions
        self._composite = rendered.composite
        self._ledger = rendered.ledger

    @property
    def scene(self) -> dict[str, Any]:
        return copy.deepcopy(self._scene)

    @property
    def scene_hash(self) -> str:
        return self._scene_hash

    @property
    def ledger(self) -> dict[str, Any]:
        return copy.deepcopy(self._ledger)

    @property
    def composite(self) -> CompositeResult:
        return self._composite

    @property
    def states(self) -> dict[int, RegionState]:
        return self._states

    def current_render(self) -> SceneRender:
        return SceneRender(
            scene=copy.deepcopy(self._scene),
            scene_hash=self._scene_hash,
            width=self._width,
            height=self._height,
            regions=self._states,
            composite=self._composite,
            ledger=self._ledger,
        )

    def plan_changes(self, changes: Iterable[dict[str, Any]]) -> ChangePlan:
        change_list = [copy.deepcopy(change) for change in changes]
        if not change_list:
            raise CoverageError(ErrorCode.SCENE_INVALID, "at least one change is required")
        next_scene = clone_scene(self._scene)
        fields_by_region: dict[int, set[str]] = {}
        affected: set[int] = set()
        for change in change_list:
            region_id, field_name = _apply_change(next_scene, change)
            affected.add(region_id)
            fields_by_region.setdefault(region_id, set()).add(field_name)
        validate_scene(next_scene)

        closure: set[str] = set()
        for region_id in affected:
            for field_name in fields_by_region[region_id]:
                closure.update(_closure_for_field(region_id, field_name))
        normalized_fields = {
            region_id: sorted(fields) for region_id, fields in sorted(fields_by_region.items())
        }
        return ChangePlan(
            changes=change_list,
            scene=next_scene,
            scene_hash=scene_hash(next_scene),
            affected_region_ids=sorted(affected),
            changed_fields_by_region=normalized_fields,
            closure=sorted(closure),
        )

    def _update_region_state(
        self,
        old_state: RegionState,
        new_region: dict[str, Any],
        *,
        closure: set[str],
        recomputed: list[str],
    ) -> RegionState:
        region_id = int(new_region["region_id"])
        prefix = f"r{region_id}"
        geometry = old_state.geometry
        geometry_hash = old_state.geometry_hash
        dual = old_state.dual
        coverage_hashes = dict(old_state.coverage_hashes)
        selected_coverage = old_state.selected_coverage
        quantization = old_state.quantization
        layer = old_state.layer
        layer_hash = old_state.layer_hash

        geometry_node = f"geometry:{prefix}"
        nonzero_node = f"coverage_nonzero:{prefix}"
        evenodd_node = f"coverage_evenodd:{prefix}"
        selected_node = f"coverage_selected:{prefix}"
        quantize_node = f"quantize:{prefix}"
        layer_node = f"layer:{prefix}"
        ledger_node = f"ledger_region:{prefix}"

        if geometry_node in closure:
            geometry = discretize_region(new_region)
            geometry_hash = str(geometry["geometry_hash"])
            recomputed.append(geometry_node)
        if nonzero_node in closure or evenodd_node in closure:
            dual = rasterize_geometry_dual(geometry, self._width, self._height)
            if nonzero_node in closure:
                recomputed.append(nonzero_node)
            if evenodd_node in closure:
                recomputed.append(evenodd_node)
        if selected_node in closure:
            selected = dual.selected(new_region["fill_rule"])
            selected_coverage = selected.coverage
            coverage_hashes = _coverage_hashes(dual, new_region["fill_rule"])
            recomputed.append(selected_node)
        if quantize_node in closure:
            quantization = quantize_coverage(selected_coverage)
            recomputed.append(quantize_node)
        if layer_node in closure:
            layer = build_source_layer(
                region_id=region_id,
                gray=quantization.gray,
                fill_color=new_region["fill_color"],
                alpha=float(new_region["alpha"]),
            )
            input_hash = sha256_json(new_region)
            layer_hash = _make_layer_hash(
                region=new_region,
                position=old_state.position,
                input_hash=input_hash,
                geometry_hash=geometry_hash,
                coverage_hashes=coverage_hashes,
                quantization=quantization,
                layer=layer,
            )
            recomputed.append(layer_node)
        else:
            input_hash = sha256_json(new_region)

        if ledger_node in closure:
            record = _region_ledger_record(
                region=new_region,
                position=old_state.position,
                width=self._width,
                height=self._height,
                input_hash=input_hash,
                geometry=geometry,
                geometry_hash=geometry_hash,
                coverage_hashes=coverage_hashes,
                quantization=quantization,
                layer=layer,
                layer_hash=layer_hash,
            )
            record["diagnostics"]["rule_difference_pixels"] = int(dual.difference_pixel_count)
            recomputed.append(ledger_node)
        else:
            record = old_state.ledger_record

        return RegionState(
            region=new_region,
            position=old_state.position,
            input_hash=input_hash,
            geometry=geometry,
            geometry_hash=geometry_hash,
            dual=dual,
            selected_coverage=selected_coverage,
            coverage_hashes=coverage_hashes,
            quantization=quantization,
            layer=layer,
            layer_hash=layer_hash,
            ledger_record=record,
        )

    def apply_changes(
        self,
        changes: Iterable[dict[str, Any]],
        *,
        expected_closure: Iterable[str] | None = None,
    ) -> UpdateReceipt:
        plan = self.plan_changes(changes)
        expected = sorted(set(plan.closure))
        if expected_closure is not None:
            supplied = [expected_closure] if isinstance(expected_closure, str) else list(expected_closure)
            provided = sorted(set(supplied))
            if provided != expected:
                locator = plan.affected_region_ids[0] if len(plan.affected_region_ids) == 1 else None
                raise CoverageError(
                    ErrorCode.CLOSURE_MISMATCH,
                    "provided closure does not equal the exact invalidation closure",
                    region_id=locator,
                    details={
                        "provided_closure": provided,
                        "required_closure": expected,
                        "missing": sorted(set(expected) - set(provided)),
                        "extra": sorted(set(provided) - set(expected)),
                    },
                )

        previous_scene_hash = self._scene_hash
        previous_states = self._states
        unchanged_region_ids = sorted(set(previous_states) - set(plan.affected_region_ids))
        before_layer_hashes = {
            region_id: previous_states[region_id].layer_hash for region_id in unchanged_region_ids
        }
        before_ledger_entries = {
            region_id: copy.deepcopy(previous_states[region_id].ledger_record)
            for region_id in unchanged_region_ids
        }

        closure_set = set(plan.closure)
        recomputed: list[str] = []
        next_states = dict(previous_states)
        next_regions = scene_by_id(plan.scene)
        for region_id in plan.affected_region_ids:
            next_states[region_id] = self._update_region_state(
                previous_states[region_id],
                copy.deepcopy(next_regions[region_id]),
                closure=closure_set,
                recomputed=recomputed,
            )

        if "composite" in closure_set:
            ordered_layers = [next_states[region_id].layer for region_id in sorted(next_states)]
            next_composite = compose_layers(ordered_layers)
            recomputed.append("composite")
        else:
            next_composite = self._composite

        if "ledger_image" in closure_set:
            records = [
                copy.deepcopy(next_states[region_id].ledger_record)
                for region_id in sorted(next_states)
            ]
            next_ledger = assemble_ledger(
                plan.scene,
                records,
                _image_record(next_composite),
                scene_hash=plan.scene_hash,
            )
            recomputed.append("ledger_image")
        else:
            next_ledger = self._ledger

        actual_set = set(recomputed)
        if actual_set != closure_set:
            raise CoverageError(
                ErrorCode.CLOSURE_MISMATCH,
                "actual recomputed node set differs from the closure manifest",
                details={
                    "closure": sorted(closure_set),
                    "actual": sorted(actual_set),
                    "missing": sorted(closure_set - actual_set),
                    "extra": sorted(actual_set - closure_set),
                },
            )

        after_layer_hashes = {
            region_id: next_states[region_id].layer_hash for region_id in unchanged_region_ids
        }
        after_ledger_entries = {
            region_id: next_states[region_id].ledger_record for region_id in unchanged_region_ids
        }
        layers_preserved = before_layer_hashes == after_layer_hashes
        entries_preserved = before_ledger_entries == after_ledger_entries
        if not layers_preserved or not entries_preserved:
            raise CoverageError(
                ErrorCode.CLOSURE_MISMATCH,
                "an unchanged region's layer or ledger entry changed outside the closure",
                details={
                    "outside_layer_hashes_preserved": layers_preserved,
                    "outside_ledger_entries_preserved": entries_preserved,
                },
            )

        self._scene = plan.scene
        self._scene_hash = plan.scene_hash
        self._states = next_states
        self._composite = next_composite
        self._ledger = next_ledger
        return UpdateReceipt(
            previous_scene_hash=previous_scene_hash,
            new_scene_hash=plan.scene_hash,
            affected_region_ids=list(plan.affected_region_ids),
            changed_fields_by_region=plan.changed_fields_by_region,
            closure=list(plan.closure),
            recomputed=sorted(actual_set),
            unchanged_region_ids=unchanged_region_ids,
            outside_layer_hashes_preserved=layers_preserved,
            outside_ledger_entries_preserved=entries_preserved,
        )
