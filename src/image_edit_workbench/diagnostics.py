from __future__ import annotations

import platform
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import PIL
from PIL import Image

from .codec_v1 import parse_v1
from .convert import downgrade_v2, restore_v1_with_sidecar
from .errors import E_BOUNDARY_SENSITIVE
from .exchange import MaterializedExchange, parse_exchange_bytes
from .models import ImageInfo, RegionEdit
from .runner import ScenarioState, cold_render_exchange
from .util import dec_str, read_json, round_half_up, sha256_bytes


COORD_BINS = ("0", "(0,0.25]", "(0.25,0.5]", "(0.5,1]", ">1")


def validate_injection_cases(
    root: Path,
    *,
    images: Mapping[str, ImageInfo],
    manifest_keys: set[str],
) -> list[dict[str, Any]]:
    cases_path = root / "injections" / "cases.json"
    document = read_json(cases_path)
    cases = document.get("cases", []) if isinstance(document, dict) else []
    results: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        path = root / str(case["path"])
        parsed = parse_exchange_bytes(
            path.read_bytes(),
            version=int(case["version"]),
            images=images,
            manifest_keys=manifest_keys,
            source=str(path),
        )
        observed_codes = sorted({failure.code for failure in parsed.parsed.failures})
        expected = str(case["expected_code"])
        results.append(
            {
                "injection_id": str(case["id"]),
                "path": str(path),
                "version": int(case["version"]),
                "expected_code": expected,
                "observed_codes": observed_codes,
                "failure_count": len(parsed.parsed.failures),
                "valid_region_count": len(parsed.parsed.regions),
                "hit": expected in observed_codes,
                "failures": [failure.to_dict() for failure in parsed.parsed.failures],
            }
        )
    return results


def _coord_bin(value: Decimal) -> str:
    if value == 0:
        return "0"
    if value <= Decimal("0.25"):
        return "(0,0.25]"
    if value <= Decimal("0.5"):
        return "(0.25,0.5]"
    if value <= 1:
        return "(0.5,1]"
    return ">1"


def _region_local_diff(
    baseline: np.ndarray,
    variant: np.ndarray,
    region: RegionEdit,
) -> tuple[int, int]:
    pad = int(region.feather_radius) + 3
    x0 = max(0, round_half_up(region.rect.x) - pad)
    y0 = max(0, round_half_up(region.rect.y) - pad)
    x1 = min(baseline.shape[1], round_half_up(region.rect.x + region.rect.w) + pad)
    y1 = min(baseline.shape[0], round_half_up(region.rect.y + region.rect.h) + pad)
    patch_a = baseline[y0:y1, x0:x1, :]
    patch_b = variant[y0:y1, x0:x1, :]
    delta = np.abs(patch_a.astype(np.int16) - patch_b.astype(np.int16))
    return int(delta.max()) if delta.size else 0, int(np.count_nonzero(np.any(delta > 0, axis=2)))


def cross_version_diagnostics(
    *,
    regions: list[RegionEdit],
    images: Mapping[str, ImageInfo],
    config: Any,
    work_root: Path,
) -> dict[str, Any]:
    manifest_keys = {region.region_key for region in regions}
    from .exchange import materialize_logical_state

    v2_exchange = materialize_logical_state(
        regions,
        version=2,
        images=images,
        manifest_keys=manifest_keys,
    )
    downgraded = downgrade_v2(v2_exchange.regions)
    v1_parsed = parse_v1(
        downgraded.text,
        images=images,
        manifest_keys=manifest_keys,
        source="cross-v1-downgrade",
    )
    v1_exchange = MaterializedExchange(
        version=1,
        text=downgraded.text,
        parsed=v1_parsed,
        exchange_fingerprint=sha256_bytes(downgraded.text.encode("utf-8")),
        sidecar_hash=downgraded.sidecar_hash,
        sidecar=downgraded.sidecar,
    )
    restored = restore_v1_with_sidecar(v1_parsed.regions, downgraded.sidecar or {})

    base_state = ScenarioState(regions=list(v2_exchange.regions), config=config, exchange_version=2)
    v2_result, _, _ = cold_render_exchange(
        exchange=v2_exchange,
        state=base_state,
        images=images,
        config=config,
        state_dir=work_root / "v2" / "state",
        output_dir=work_root / "v2" / "outputs",
        drain=True,
    )
    v1_state = ScenarioState(regions=list(v1_exchange.regions), config=config, exchange_version=1)
    v1_result, _, _ = cold_render_exchange(
        exchange=v1_exchange,
        state=v1_state,
        images=images,
        config=config,
        state_dir=work_root / "v1" / "state",
        output_dir=work_root / "v1" / "outputs",
        drain=True,
    )

    v2_by_key = {region.region_key: region for region in v2_exchange.parsed.regions}
    v1_by_key = {region.region_key: region for region in v1_parsed.regions}
    restored_by_key = {region.region_key: region for region in restored.regions}
    region_rows: list[dict[str, Any]] = []
    boundary_samples: list[dict[str, Any]] = []
    style_deviations: list[Decimal] = []
    mapping_matches = 0
    mapping_total = 0
    restored_equal = 0
    lost_info_count = 0
    aggregate_hist: Counter[str] = Counter()

    baseline_arrays: dict[str, np.ndarray] = {}
    variant_arrays: dict[str, np.ndarray] = {}
    for image_name in sorted(images):
        with Image.open(work_root / "v2" / "outputs" / image_name) as image:
            baseline_arrays[image_name] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        with Image.open(work_root / "v1" / "outputs" / image_name) as image:
            variant_arrays[image_name] = np.asarray(image.convert("RGB"), dtype=np.uint8)

    for key in sorted(v2_by_key):
        high = v2_by_key[key]
        low = v1_by_key.get(key)
        if low is None:
            continue
        coordinates = [
            abs(high.rect.x - low.rect.x),
            abs(high.rect.y - low.rect.y),
            abs(high.rect.w - low.rect.w),
            abs(high.rect.h - low.rect.h),
        ]
        max_coord = max(coordinates)
        histogram = Counter(_coord_bin(value) for value in coordinates)
        for name in COORD_BINS:
            aggregate_hist[name] += histogram.get(name, 0)
        style_delta = abs(high.style_strength - low.style_strength)
        style_deviations.append(style_delta)
        repair_match = high.repair_method == low.repair_method
        mapping_total += 1
        mapping_matches += int(repair_match)
        restored_region = restored_by_key.get(key)
        field_equal = restored_region is not None and _logical_equal(high, restored_region)
        restored_equal += int(field_equal)
        rotation_lost = high.rotation_deg != 0 and low.rotation_deg == 0
        feather_lost = high.feather_radius != 0 and low.feather_radius == 0
        lost = rotation_lost or feather_lost
        lost_info_count += int(lost)
        local_max, local_changed = _region_local_diff(
            baseline_arrays[high.image_name],
            variant_arrays[high.image_name],
            high,
        )
        causes: list[str] = []
        if max_coord > config.coordinate_tolerance:
            causes.append("coordinate_integerization")
        if style_delta > 0:
            causes.append("strength_permille_truncation")
        if lost:
            causes.append("dropped_rotation_or_feather")
        sensitive = bool(causes) and (local_max > config.pixel_difference_tolerance or max_coord > config.coordinate_tolerance)
        if not causes and local_max > config.pixel_difference_tolerance:
            causes.append("unclassified_pixel_difference")
            sensitive = True
        row = {
            "region_key": key,
            "coordinate_abs_delta": {
                "x": dec_str(coordinates[0]),
                "y": dec_str(coordinates[1]),
                "w": dec_str(coordinates[2]),
                "h": dec_str(coordinates[3]),
            },
            "coordinate_max_abs_delta": dec_str(max_coord),
            "coordinate_histogram": {name: histogram.get(name, 0) for name in COORD_BINS},
            "style_abs_delta": dec_str(style_delta),
            "repair_v1": low.repair_method,
            "repair_v2": high.repair_method,
            "repair_mapping_restored": repair_match,
            "sidecar_restore_equal": field_equal,
            "rotation_v1": dec_str(low.rotation_deg),
            "rotation_v2": dec_str(high.rotation_deg),
            "feather_v1": dec_str(low.feather_radius),
            "feather_v2": dec_str(high.feather_radius),
            "local_pixel_max_delta": local_max,
            "local_changed_pixels": local_changed,
            "boundary_sensitive": sensitive,
            "causes": causes,
        }
        region_rows.append(row)
        if sensitive:
            boundary_samples.append(
                {
                    "region_key": key,
                    "code": E_BOUNDARY_SENSITIVE,
                    "causes": causes,
                    "coordinate_max_abs_delta": dec_str(max_coord),
                    "style_abs_delta": dec_str(style_delta),
                    "local_pixel_max_delta": local_max,
                }
            )

    global_pixel_stats: dict[str, Any] = {}
    for image_name in sorted(images):
        delta = np.abs(baseline_arrays[image_name].astype(np.int16) - variant_arrays[image_name].astype(np.int16))
        global_pixel_stats[image_name] = {
            "max_channel_delta": int(delta.max()) if delta.size else 0,
            "mean_channel_delta": round(float(delta.mean()), 6) if delta.size else 0.0,
            "changed_pixels": int(np.count_nonzero(np.any(delta > 0, axis=2))),
            "within_tolerance": bool(
                (int(delta.max()) if delta.size else 0) <= config.pixel_difference_tolerance
            ),
        }

    style_max = max(style_deviations) if style_deviations else Decimal(0)
    return {
        "schema": "image-edit-cross-version-diagnostics/v1",
        "v1_exchange_fingerprint": v1_exchange.exchange_fingerprint,
        "v2_exchange_fingerprint": v2_exchange.exchange_fingerprint,
        "sidecar_hash": downgraded.sidecar_hash,
        "sidecar_record_count": len((downgraded.sidecar or {}).get("records", [])),
        "sidecar_restore_hits": restored.sidecar_hits,
        "sidecar_restore_misses": restored.sidecar_misses,
        "restored_records_field_equal": restored_equal,
        "restored_records_total": len(v2_by_key),
        "style_strength_max_abs_delta": dec_str(style_max),
        "coordinate_deviation_histogram_aggregate": {name: aggregate_hist.get(name, 0) for name in COORD_BINS},
        "regions": region_rows,
        "repair_mapping": {
            "matches": mapping_matches,
            "total": mapping_total,
            "all_restored": mapping_matches == mapping_total,
        },
        "downgrade_info": {
            "rotation_feather_loss_regions": lost_info_count,
            "hit_regions": restored.sidecar_hits,
        },
        "pixel_diff": global_pixel_stats,
        "boundary_samples": boundary_samples,
        "boundary_business_codes": [E_BOUNDARY_SENSITIVE] if boundary_samples else [],
        "rendered_stage_summaries": {
            "v1": v1_result.stage_summaries,
            "v2": v2_result.stage_summaries,
        },
        "rendered_content_fingerprints": {
            "v1": v1_result.content_fingerprint,
            "v2": v2_result.content_fingerprint,
        },
        "output_directories": {
            "v1": str(work_root / "v1" / "outputs"),
            "v2": str(work_root / "v2" / "outputs"),
        },
    }


def _logical_equal(left: RegionEdit, right: RegionEdit) -> bool:
    return (
        left.exchange_version == right.exchange_version
        and left.image_name == right.image_name
        and left.region_index == right.region_index
        and left.rect == right.rect
        and left.style_strength == right.style_strength
        and left.target_attribute == right.target_attribute
        and left.repair_method == right.repair_method
        and left.precision_label == right.precision_label
        and left.rotation_deg == right.rotation_deg
        and left.feather_radius == right.feather_radius
    )


def environment_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "offline": True,
        "gpu": False,
        "container": False,
    }
