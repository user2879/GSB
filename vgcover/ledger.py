"""Error propagation ledger assembly with explicit completeness semantics."""

from __future__ import annotations

import copy
import math
from typing import Any, Iterable

from .composite import COMPOSITE_POLICY
from .errors import CoverageError, ErrorCode
from .hashing import sha256_json
from .quantize import QUANTIZATION_POLICY


LEDGER_SCHEMA = "vgcover.error_ledger.v1"
REQUIRED_COMPONENTS = ("geometry", "quantization", "composite")


def build_ledger_policy(width: int, height: int) -> dict[str, Any]:
    return {
        "version": "vgcover.error_ledger_policy.v1",
        "error_unit": "full-frame coverage rate in [0, 1] (mean covered/alpha fraction)",
        "canvas_pixels": int(width * height),
        "pixel_size": 1.0,
        "component_order": list(REQUIRED_COMPONENTS),
        "layer": {
            "hash_domain": "vgcover.region_layer_ledger.v1",
            "hash_fields": [
                "region_id",
                "position",
                "input_hash",
                "geometry_hash",
                "coverage_hashes",
                "selected_fill_rule",
                "quantization.policy_version",
                "quantization.gray_hash",
                "quantization.quant_error_hash",
                "fill_color",
                "alpha",
                "layer_content_hash",
                "layer_pixels_hash",
            ],
            "layer_content_hash_domain": "vgcover.layer_content.v1",
        },
        "geometry": {
            "source": "maximum analytic chord-error bound across the region's flattened segments",
            "band_area_px2": "2 * delta_px * polyline_perimeter_px + pi * (2 * delta_px)^2",
            "coverage_rate": "min(1, band_area_px2 / (width * height))",
            "delta_px": "components.geometry.effective_error_bound_px (analytic bound + 1e-12 coordinate storage rounding)",
            "perimeter_px": "components.geometry.polyline_perimeter_px",
        },
        "quantization": {
            "source": "maximum absolute per-pixel quantization error",
            "coverage_rate": "components.quantization.max_abs_error",
            "policy": copy.deepcopy(QUANTIZATION_POLICY),
        },
        "composite": {
            "source": "declared per-layer BGR(A) rounding bound in u8 code values",
            "coverage_rate": "components.composite.rounding_error_bound_u8 / 255",
            "policy": copy.deepcopy(COMPOSITE_POLICY),
        },
        "region_total": "min(1, geometry + quantization + composite)",
        "image_total": "min(1, sum(region.alpha * region.total_error_bound))",
        "missing_component_policy": "total_error_bound=null; complete=false; never substitute zero",
    }


def _is_finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _component_missing(reason: str) -> dict[str, Any]:
    return {"reason": reason}


def assemble_ledger(
    scene: dict[str, Any],
    region_records: Iterable[dict[str, Any]],
    image_record: dict[str, Any],
    *,
    scene_hash: str,
) -> dict[str, Any]:
    """Assemble a ledger from region/image records.

    Component absence is represented structurally. A missing component makes
    the region and image incomplete and leaves their total bound as ``None``.
    """

    width = int(scene["width"])
    height = int(scene["height"])
    records = sorted((copy.deepcopy(record) for record in region_records), key=lambda r: r["region_id"])
    seen_ids: set[int] = set()
    entries: list[dict[str, Any]] = []
    missing_components: list[str] = []

    for record in records:
        region_id = int(record["region_id"])
        if region_id in seen_ids:
            raise CoverageError(
                ErrorCode.LEDGER_MISMATCH,
                "duplicate region ledger record",
                region_id=region_id,
            )
        seen_ids.add(region_id)
        components = record.get("components")
        if not isinstance(components, dict):
            components = {}
        missing: list[str] = []
        values: dict[str, float] = {}
        for component_name in REQUIRED_COMPONENTS:
            component = components.get(component_name)
            if not isinstance(component, dict) or component.get("present") is False:
                missing.append(component_name)
                continue
            value = component.get("coverage_equivalent_error")
            if not _is_finite_nonnegative(value):
                missing.append(f"{component_name}.coverage_equivalent_error")
                continue
            values[component_name] = float(value)

        complete = not missing
        total = None
        if complete:
            total = min(1.0, sum(values.values()))
        else:
            missing_components.extend(f"region:{region_id}:{item}" for item in missing)

        entries.append(
            {
                "region_id": region_id,
                "category": record.get("category"),
                "position": record.get("position"),
                "alpha": float(record["alpha"]),
                "fill_color": copy.deepcopy(record.get("fill_color")),
                "input_hash": record["input_hash"],
                "geometry_hash": record["geometry_hash"],
                "coverage_hashes": copy.deepcopy(record.get("coverage_hashes", {})),
                "selected_fill_rule": record.get("selected_fill_rule"),
                "gray_hash": record.get("gray_hash"),
                "quant_error_hash": record.get("quant_error_hash"),
                "layer_hash": record["layer_hash"],
                "layer_content_hash": record.get("layer_content_hash"),
                "layer_pixels_hash": record.get("layer_pixels_hash"),
                "components": components,
                "propagation_chain": {
                    "geometry_px_error": (
                        components.get("geometry", {}).get("effective_error_bound_px")
                        if complete
                        else None
                    ),
                    "geometry_to_coverage": values.get("geometry"),
                    "coverage_to_gray_u8": values.get("quantization"),
                    "gray_to_premultiplied_bgra": values.get("composite"),
                    "accumulated_coverage_rate": total,
                },
                "total_error_bound": total,
                "complete": complete,
                "missing_components": missing,
            }
        )

    entry_by_id = {entry["region_id"]: entry for entry in entries}
    layer_sequence = [int(region_id) for region_id in image_record.get("layer_sequence", [])]
    if layer_sequence != sorted(entry_by_id):
        raise CoverageError(
            ErrorCode.LEDGER_MISMATCH,
            "image layer sequence must equal region ledger IDs in ascending order",
            details={"layer_sequence": layer_sequence, "ledger_region_ids": sorted(entry_by_id)},
        )
    image_layer_hashes = {
        int(region_id): value
        for region_id, value in image_record.get("layer_hashes", {}).items()
    }
    for region_id in layer_sequence:
        entry = entry_by_id[region_id]
        if image_layer_hashes.get(region_id) != entry.get("layer_content_hash"):
            raise CoverageError(
                ErrorCode.LEDGER_MISMATCH,
                "image layer content hash does not match its region ledger entry",
                region_id=region_id,
                details={
                    "image_hash": image_layer_hashes.get(region_id),
                    "ledger_hash": entry.get("layer_content_hash"),
                },
            )

    image_complete = not missing_components and all(entry["complete"] for entry in entries)
    image_components: dict[str, Any] = {}
    if image_complete:
        geometry_total = sum(
            entry["alpha"] * entry["components"]["geometry"]["coverage_equivalent_error"]
            for entry in entries
        )
        quant_total = sum(
            entry["alpha"] * entry["components"]["quantization"]["coverage_equivalent_error"]
            for entry in entries
        )
        composite_total = sum(
            entry["alpha"] * entry["components"]["composite"]["coverage_equivalent_error"]
            for entry in entries
        )
        image_components = {
            "geometry": {
                "present": True,
                "coverage_equivalent_error_sum": geometry_total,
            },
            "quantization": {
                "present": True,
                "coverage_equivalent_error_sum": quant_total,
            },
            "composite": {
                "present": True,
                "coverage_equivalent_error_sum": composite_total,
            },
        }
        image_total = min(
            1.0,
            sum(entry["alpha"] * float(entry["total_error_bound"]) for entry in entries),
        )
    else:
        image_total = None

    step_payload = []
    cumulative = 0.0
    for step in image_record.get("steps", []):
        step_copy = copy.deepcopy(step)
        cumulative += float(step_copy.get("step_rounding_error_bound_u8", 0.0))
        step_copy["cumulative_rounding_error_bound_u8"] = cumulative
        step_payload.append(step_copy)

    image_entry = {
        "width": width,
        "height": height,
        "buffer_hash": image_record.get("buffer_hash"),
        "layer_sequence": layer_sequence,
        "layer_hashes": [
            {
                "region_id": region_id,
                "ledger_layer_hash": entry_by_id[region_id]["layer_hash"],
                "layer_content_hash": entry_by_id[region_id]["layer_content_hash"],
            }
            for region_id in layer_sequence
        ],
        "steps": step_payload,
        "components": image_components,
        "propagation_chain": {
            "geometry_to_coverage": image_components.get("geometry", {}).get(
                "coverage_equivalent_error_sum"
            ),
            "coverage_to_gray_u8": image_components.get("quantization", {}).get(
                "coverage_equivalent_error_sum"
            ),
            "gray_to_premultiplied_bgra": image_components.get("composite", {}).get(
                "coverage_equivalent_error_sum"
            ),
            "accumulated_coverage_rate": image_total,
        },
        "aggregation": "min(1, sum(alpha_i * region_total_i))",
        "total_error_bound": image_total,
        "complete": image_complete,
        "missing_components": sorted(set(missing_components)),
    }

    ledger = {
        "schema": LEDGER_SCHEMA,
        "scene_hash": scene_hash,
        "canvas": {"width": width, "height": height, "pixel_size": 1.0},
        "ledger_policy": build_ledger_policy(width, height),
        "regions": entries,
        "image": image_entry,
        "complete": bool(image_complete and all(entry["complete"] for entry in entries)),
    }
    return ledger


def ledger_hash(ledger: dict[str, Any]) -> str:
    return sha256_json(ledger)


def missing_component_marker(component_name: str) -> dict[str, Any]:
    """Helper for callers/tests that need to represent an absent component."""

    return _component_missing(f"missing:{component_name}")
