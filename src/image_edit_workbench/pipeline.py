from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .config import WorkbenchConfig
from .errors import (
    BusinessFailure,
    E_RECOMPUTE_BUDGET_PENDING,
    E_STRUCTURE_INVALID,
    WorkbenchError,
)
from .image_registry import image_collection_fingerprint, load_rgb
from .models import ImageInfo, RegionEdit
from .renderer import (
    attribute_replace,
    compose_tile,
    generate_mask,
    repair_fill,
    style_map,
)
from .snapshot import SnapshotStore
from .util import atomic_write_json, dec_str, fingerprint, round_half_up, sha256_json


STAGES = ("parse", "mask", "repair", "attribute", "style", "compose")


@dataclass
class StageTrace:
    stage: str
    cache_key: str
    status: str
    summary: str
    upstream: list[str]
    image_block: list[int] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "cache_key": self.cache_key,
            "status": self.status,
            "summary": self.summary,
            "upstream": self.upstream,
            "image_block": self.image_block,
        }


@dataclass
class RegionTrace:
    region_key: str
    image_name: str
    region_index: int
    sort_key: str
    status: str
    budget_cost: int
    stages: list[StageTrace] = field(default_factory=list)
    failure: BusinessFailure | None = None
    content_fingerprint: str = ""
    affected_pixels: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = {
            "region_key": self.region_key,
            "image": self.image_name,
            "region_index": self.region_index,
            "sort_key": self.sort_key,
            "status": self.status,
            "budget_cost": self.budget_cost,
            "stages": [stage.to_dict() for stage in self.stages],
            "content_fingerprint": self.content_fingerprint,
            "affected_pixels": self.affected_pixels,
        }
        if self.failure is not None:
            data["failure"] = self.failure.to_dict()
        return data


@dataclass
class PipelineResult:
    region_records: dict[str, dict[str, Any]]
    pending_queue: list[str]
    budget_log: list[dict[str, Any]]
    failures: list[BusinessFailure]
    traces: list[RegionTrace]
    reused_regions: list[str]
    recomputed_regions: list[str]
    output_images: dict[str, str]
    image_fingerprints: dict[str, str]
    stage_summaries: dict[str, dict[str, str]]
    content_fingerprint: str
    max_active_tasks: int
    artifact_records: dict[str, dict[str, Any]]
    affected_pixels: int
    metadata: dict[str, Any]

    def to_diagnostic_dict(self) -> dict[str, Any]:
        return {
            "reused_regions": self.reused_regions,
            "recomputed_regions": self.recomputed_regions,
            "pending_queue": self.pending_queue,
            "budget_log": self.budget_log,
            "region_trace": [trace.to_dict() for trace in self.traces],
            "image_fingerprints": self.image_fingerprints,
            "stage_summaries": self.stage_summaries,
            "content_fingerprint": self.content_fingerprint,
            "max_active_tasks": self.max_active_tasks,
            "affected_pixels": self.affected_pixels,
            "metadata": self.metadata,
        }


def estimate_region_pixels(region: RegionEdit, image_width: int, image_height: int) -> int:
    """Cheap mask-bounds estimate used before rendering for budget allocation."""
    pad = int(region.feather_radius) + 2
    x0 = max(0, round_half_up(region.rect.x) - pad)
    y0 = max(0, round_half_up(region.rect.y) - pad)
    x1 = min(image_width, round_half_up(region.rect.x + region.rect.w) + pad)
    y1 = min(image_height, round_half_up(region.rect.y + region.rect.h) + pad)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    if region.rotation_deg != 0:
        diagonal = int((width * width + height * height) ** 0.5 + 0.999999)
        width = height = diagonal
    return width * height


def compute_stage_keys(
    region: RegionEdit,
    image: ImageInfo,
    config: WorkbenchConfig,
) -> dict[str, str]:
    canonical = region.canonical_dict()
    parse_key = fingerprint(
        "parse",
        "exchange-version",
        region.exchange_version,
        "source-format",
        region.source_format,
        canonical,
    )
    mask_key = fingerprint(
        "mask",
        parse_key,
        image.source_fingerprint,
        image.width,
        image.height,
        "coord-bits-8",
        "trig-bits-20",
    )
    repair_key = fingerprint(
        "repair",
        mask_key,
        image.source_fingerprint,
        config.repair_neighborhood_radius,
        config.fixed_fraction_bits,
        config.rounding,
    )
    attribute_key = fingerprint(
        "attribute",
        repair_key,
        region.target_attribute,
        config.fixed_fraction_bits,
        config.rounding,
    )
    style_key = fingerprint(
        "style",
        attribute_key,
        dec_str(region.style_strength),
        config.style_quantization_levels,
        config.fixed_fraction_bits,
        config.rounding,
    )
    compose_key = fingerprint(
        "compose",
        style_key,
        mask_key,
        region.exchange_version,
        "alpha-8bit",
        config.rounding,
    )
    return {
        "parse": parse_key,
        "mask": mask_key,
        "repair": repair_key,
        "attribute": attribute_key,
        "style": style_key,
        "compose": compose_key,
    }


def _record_is_usable(record: dict[str, Any] | None, store: SnapshotStore) -> bool:
    if record is None:
        return False
    if record.get("artifact") is None:
        return record.get("stage") == "parse"
    return store.artifact_path(record) is not None


def _crop_array(crop: tuple[int, int, int, int]) -> np.ndarray:
    return np.array(crop, dtype=np.int32)


def _crop_from(array: np.ndarray) -> tuple[int, int, int, int]:
    return tuple(int(value) for value in array.tolist())  # type: ignore[return-value]


def _array_summary(record: dict[str, Any]) -> str:
    return str(record.get("sha256", ""))


class _RegionProcessor:
    def __init__(
        self,
        *,
        image: ImageInfo,
        base_rgb: np.ndarray,
        config: WorkbenchConfig,
        store: SnapshotStore,
    ) -> None:
        self.image = image
        self.base_rgb = base_rgb
        self.config = config
        self.store = store

    def _stage_record(
        self,
        *,
        stage: str,
        cache_key: str,
        region: RegionEdit,
        upstream: list[str],
        image_block: list[int] | None,
        arrays_factory: Any,
        metadata_factory: Any,
        summary_only_metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray] | None, str]:
        existing = self.store.get_record(cache_key)
        if _record_is_usable(existing, self.store):
            arrays = None
            if existing is not None and existing.get("artifact"):
                loaded = self.store.load_arrays(existing)
                if loaded is None:
                    existing = None
                else:
                    arrays, _ = loaded
            if existing is not None:
                return existing, arrays, "reused"
        if stage == "parse":
            metadata = summary_only_metadata or {}
            record = self.store.save_artifact(
                cache_key=cache_key,
                stage=stage,
                region_key=region.region_key,
                arrays=None,
                metadata=metadata,
                upstream=upstream,
                image_block=image_block,
            )
            return record, None, "computed"
        arrays, metadata = arrays_factory()
        record = self.store.save_artifact(
            cache_key=cache_key,
            stage=stage,
            region_key=region.region_key,
            arrays=arrays,
            metadata=metadata,
            upstream=upstream,
            image_block=image_block,
        )
        return record, arrays, "computed"

    def process(self, region: RegionEdit, keys: Mapping[str, str], budget_cost: int) -> tuple[dict[str, Any], RegionTrace]:
        trace = RegionTrace(
            region_key=region.region_key,
            image_name=region.image_name,
            region_index=region.region_index,
            sort_key=f"{region.image_name}|{region.region_index:06d}|mask+repair+attribute+style+compose",
            status="computed",
            budget_cost=budget_cost,
        )
        stage_records: dict[str, dict[str, Any]] = {}

        parse_record, _, parse_status = self._stage_record(
            stage="parse",
            cache_key=keys["parse"],
            region=region,
            upstream=[],
            image_block=None,
            arrays_factory=lambda: ({}, {}),
            metadata_factory=lambda: {},
            summary_only_metadata={"canonical_region": region.canonical_dict()},
        )
        stage_records["parse"] = parse_record
        trace.stages.append(
            StageTrace("parse", keys["parse"], parse_status, _array_summary(parse_record), [], None)
        )

        mask_record, mask_arrays, mask_status = self._stage_record(
            stage="mask",
            cache_key=keys["mask"],
            region=region,
            upstream=[keys["parse"]],
            image_block=None,
            arrays_factory=lambda: self._make_mask(region),
            metadata_factory=lambda: {},
        )
        stage_records["mask"] = mask_record
        if mask_arrays is None:
            loaded = self.store.load_arrays(mask_record)
            if loaded is None:
                raise WorkbenchError(E_STRUCTURE_INVALID, "mask artifact disappeared", stage="mask")
            mask_arrays, _ = loaded
        core = mask_arrays["core"]
        alpha = mask_arrays["alpha"]
        mask_crop = _crop_from(mask_arrays["crop"])
        trace.stages.append(
            StageTrace(
                "mask",
                keys["mask"],
                mask_status,
                _array_summary(mask_record),
                [keys["parse"]],
                list(mask_crop),
            )
        )

        repair_record, repair_arrays, repair_status = self._stage_record(
            stage="repair",
            cache_key=keys["repair"],
            region=region,
            upstream=[keys["mask"]],
            image_block=None,
            arrays_factory=lambda: self._make_repair(region, core, mask_crop),
            metadata_factory=lambda: {},
        )
        stage_records["repair"] = repair_record
        if repair_arrays is None:
            loaded = self.store.load_arrays(repair_record)
            if loaded is None:
                raise WorkbenchError(E_STRUCTURE_INVALID, "repair artifact disappeared", stage="repair")
            repair_arrays, _ = loaded
        filled = repair_arrays["filled"]
        repair_crop = _crop_from(repair_arrays["crop"])
        trace.stages.append(
            StageTrace(
                "repair",
                keys["repair"],
                repair_status,
                _array_summary(repair_record),
                [keys["mask"]],
                list(repair_crop),
            )
        )

        # The mask crop and repair crop differ by the repair halo. Expand alpha/core consistently.
        ex0, ey0, ex1, ey1 = repair_crop
        mx0, my0, mx1, my1 = mask_crop
        expanded_core = np.zeros((ey1 - ey0, ex1 - ex0), dtype=np.uint8)
        expanded_alpha = np.zeros_like(expanded_core)
        expanded_core[my0 - ey0 : my0 - ey0 + core.shape[0], mx0 - ex0 : mx0 - ex0 + core.shape[1]] = core
        expanded_alpha[my0 - ey0 : my0 - ey0 + alpha.shape[0], mx0 - ex0 : mx0 - ex0 + alpha.shape[1]] = alpha
        active = expanded_alpha > 0

        attribute_record, attribute_arrays, attribute_status = self._stage_record(
            stage="attribute",
            cache_key=keys["attribute"],
            region=region,
            upstream=[keys["repair"]],
            image_block=None,
            arrays_factory=lambda: self._make_attribute(region, filled, active, repair_crop),
            metadata_factory=lambda: {},
        )
        stage_records["attribute"] = attribute_record
        if attribute_arrays is None:
            loaded = self.store.load_arrays(attribute_record)
            if loaded is None:
                raise WorkbenchError(E_STRUCTURE_INVALID, "attribute artifact disappeared", stage="attribute")
            attribute_arrays, _ = loaded
        attributed = attribute_arrays["attributed"]
        trace.stages.append(
            StageTrace(
                "attribute",
                keys["attribute"],
                attribute_status,
                _array_summary(attribute_record),
                [keys["repair"]],
                list(repair_crop),
            )
        )

        style_record, style_arrays, style_status = self._stage_record(
            stage="style",
            cache_key=keys["style"],
            region=region,
            upstream=[keys["attribute"]],
            image_block=None,
            arrays_factory=lambda: self._make_style(region, filled, attributed, active, repair_crop),
            metadata_factory=lambda: {},
        )
        stage_records["style"] = style_record
        if style_arrays is None:
            loaded = self.store.load_arrays(style_record)
            if loaded is None:
                raise WorkbenchError(E_STRUCTURE_INVALID, "style artifact disappeared", stage="style")
            style_arrays, _ = loaded
        styled = style_arrays["styled"]
        trace.stages.append(
            StageTrace(
                "style",
                keys["style"],
                style_status,
                _array_summary(style_record),
                [keys["attribute"]],
                list(repair_crop),
            )
        )

        compose_record, compose_arrays, compose_status = self._stage_record(
            stage="compose",
            cache_key=keys["compose"],
            region=region,
            upstream=[keys["style"], keys["mask"]],
            image_block=None,
            arrays_factory=lambda: self._make_compose(filled, styled, expanded_alpha, repair_crop),
            metadata_factory=lambda: {},
        )
        stage_records["compose"] = compose_record
        if compose_arrays is None:
            loaded = self.store.load_arrays(compose_record)
            if loaded is None:
                raise WorkbenchError(E_STRUCTURE_INVALID, "compose artifact disappeared", stage="compose")
            compose_arrays, _ = loaded
        compose_alpha = compose_arrays["alpha"]
        trace.affected_pixels = int(np.count_nonzero(compose_alpha))
        trace.stages.append(
            StageTrace(
                "compose",
                keys["compose"],
                compose_status,
                _array_summary(compose_record),
                [keys["style"], keys["mask"]],
                list(repair_crop),
            )
        )

        content_fp = fingerprint("region-content", keys["compose"], compose_record.get("sha256", ""))
        trace.content_fingerprint = content_fp
        trace.status = "computed" if any(stage.status == "computed" for stage in trace.stages) else "reused"
        region_record = {
            "region_key": region.region_key,
            "image": region.image_name,
            "region_index": region.region_index,
            "exchange_version": region.exchange_version,
            "semantic": region.canonical_dict(),
            "pixel_count": trace.affected_pixels,
            "budget_cost": budget_cost,
            "stages": stage_records,
            "layer_key": keys["compose"],
            "layer_artifact": compose_record.get("artifact"),
            "content_fingerprint": content_fp,
        }
        return region_record, trace

    def _make_mask(self, region: RegionEdit) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        result = generate_mask(region, self.image.width, self.image.height, self.config)
        return (
            {
                "core": result.core,
                "alpha": result.alpha,
                "crop": _crop_array(result.crop),
            },
            result.metadata,
        )

    def _make_repair(
        self,
        region: RegionEdit,
        core: np.ndarray,
        mask_crop: tuple[int, int, int, int],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        result = repair_fill(
            self.base_rgb,
            core,
            mask_crop,
            repair_method=region.repair_method,
            radius=self.config.repair_neighborhood_radius,
        )
        return {"filled": result.filled, "crop": _crop_array(result.crop)}, result.metadata

    def _make_attribute(
        self,
        region: RegionEdit,
        filled: np.ndarray,
        active: np.ndarray,
        crop: tuple[int, int, int, int],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        result = attribute_replace(filled, active, region.target_attribute)
        metadata = dict(result.metadata)
        metadata["crop"] = list(crop)
        return {"attributed": result.attributed, "crop": _crop_array(crop)}, metadata

    def _make_style(
        self,
        region: RegionEdit,
        filled: np.ndarray,
        attributed: np.ndarray,
        active: np.ndarray,
        crop: tuple[int, int, int, int],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        result = style_map(
            filled,
            attributed,
            active,
            style_strength=region.style_strength,
            levels=self.config.style_quantization_levels,
            fraction_bits=self.config.fixed_fraction_bits,
        )
        metadata = dict(result.metadata)
        metadata["crop"] = list(crop)
        return {"styled": result.styled, "crop": _crop_array(crop)}, metadata

    def _make_compose(
        self,
        filled: np.ndarray,
        styled: np.ndarray,
        alpha: np.ndarray,
        crop: tuple[int, int, int, int],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        result = compose_tile(filled, styled, alpha)
        metadata = dict(result.metadata)
        metadata["crop"] = list(crop)
        return {
            "tile": result.tile,
            "alpha": result.alpha,
            "crop": _crop_array(crop),
        }, metadata


def _save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.png")
    image = Image.fromarray(rgb, mode="RGB")
    image.save(tmp, format="PNG", compress_level=9, optimize=False)
    tmp.replace(path)


def _compose_images(
    *,
    images: Mapping[str, ImageInfo],
    region_records: Mapping[str, dict[str, Any]],
    store: SnapshotStore,
    output_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    output_paths: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    for image_name in sorted(images):
        info = images[image_name]
        canvas = load_rgb(info.path).astype(np.int64)
        records = sorted(
            (record for record in region_records.values() if record["image"] == image_name),
            key=lambda record: int(record["region_index"]),
        )
        for record in records:
            layer = store.get_record(str(record["layer_key"]))
            if layer is None:
                continue
            loaded = store.load_arrays(layer)
            if loaded is None:
                continue
            arrays, _ = loaded
            tile = arrays["tile"].astype(np.int64)
            alpha = arrays["alpha"].astype(np.int64)
            x0, y0, x1, y1 = _crop_from(arrays["crop"])
            crop_alpha = alpha[:, :, None]
            region_canvas = canvas[y0:y1, x0:x1, :]
            canvas[y0:y1, x0:x1, :] = (
                region_canvas * (255 - crop_alpha) + tile * crop_alpha + 127
            ) // 255
        final_rgb = np.clip(canvas, 0, 255).astype(np.uint8)
        output_path = output_dir / image_name
        _save_png(output_path, final_rgb)
        output_paths[image_name] = str(output_path)
        fingerprints[image_name] = fingerprint(
            "output-image-rgb",
            image_name,
            info.width,
            info.height,
            final_rgb.tobytes(),
        )
    return output_paths, fingerprints


def run_pipeline(
    *,
    regions: list[RegionEdit],
    images: Mapping[str, ImageInfo],
    config: WorkbenchConfig,
    store: SnapshotStore,
    prior_manifest: dict[str, Any],
    output_dir: Path,
    failures: list[BusinessFailure] | None = None,
    sidecar_hash: str = "",
    exchange_fingerprint: str = "",
) -> PipelineResult:
    store.begin_work()
    prior_pending = [str(key) for key in prior_manifest.get("pending_queue", [])]
    failures_out = list(failures or [])
    region_records: dict[str, dict[str, Any]] = {}
    traces: list[RegionTrace] = []
    active: list[tuple[RegionEdit, dict[str, str], int]] = []
    reusable_keys: set[str] = set()
    reused_regions: list[str] = []

    for region in sorted(regions, key=lambda item: (item.image_name, item.region_index)):
        if region.image_name not in images:
            failures_out.append(
                BusinessFailure(
                    "E_REGION_OUT_OF_RANGE",
                    "schedule",
                    "pipeline",
                    f"unknown image {region.image_name}",
                    region.region_key,
                )
            )
            continue
        image = images[region.image_name]
        keys = compute_stage_keys(region, image, config)
        cached_stages: dict[str, dict[str, Any]] = {}
        all_reusable = True
        for stage in STAGES:
            cached = store.get_record(keys[stage])
            if not _record_is_usable(cached, store):
                all_reusable = False
                break
            assert cached is not None
            cached_stages[stage] = cached
        if all_reusable:
            layer_record = cached_stages["compose"]
            pixel_count = int(layer_record.get("metadata", {}).get("alpha_nonzero", 0))
            content_fp = fingerprint("region-content", keys["compose"], layer_record.get("sha256", ""))
            record = {
                "region_key": region.region_key,
                "image": region.image_name,
                "region_index": region.region_index,
                "exchange_version": region.exchange_version,
                "semantic": region.canonical_dict(),
                "pixel_count": pixel_count,
                "budget_cost": estimate_region_pixels(region, image.width, image.height),
                "stages": {stage: dict(cached_stages[stage]) for stage in STAGES},
                "layer_key": keys["compose"],
                "layer_artifact": layer_record.get("artifact"),
                "content_fingerprint": content_fp,
            }
            region_records[record["region_key"]] = record
            reused_regions.append(record["region_key"])
            reusable_keys.update(record["stages"][stage]["cache_key"] for stage in STAGES)
            traces.append(
                RegionTrace(
                    region_key=record["region_key"],
                    image_name=record["image"],
                    region_index=int(record["region_index"]),
                    sort_key=f"{record['image']}|{int(record['region_index']):06d}|pipeline",
                    status="reused",
                    budget_cost=int(record.get("budget_cost", 0)),
                    stages=[
                        StageTrace(
                            stage,
                            record["stages"][stage]["cache_key"],
                            "reused",
                            _array_summary(record["stages"][stage]),
                            list(record["stages"][stage].get("upstream", [])),
                            record["stages"][stage].get("image_block"),
                        )
                        for stage in STAGES
                    ],
                    content_fingerprint=record.get("content_fingerprint", ""),
                    affected_pixels=int(record.get("pixel_count", 0)),
                )
            )
            continue
        active.append((region, keys, estimate_region_pixels(region, image.width, image.height)))

    active_by_key = {region.region_key: (region, keys, cost) for region, keys, cost in active}
    ordered_keys = [key for key in prior_pending if key in active_by_key]
    ordered_keys += [
        region.region_key
        for region, _, _ in sorted(active, key=lambda item: (item[0].image_name, item[0].region_index))
        if region.region_key not in ordered_keys
    ]
    # The final queue is always presented in stable image/region order.
    ordered_keys = sorted(
        ordered_keys,
        key=lambda key: (active_by_key[key][0].image_name, active_by_key[key][0].region_index),
    )
    pending_priority = {key for key in ordered_keys if key in prior_pending}
    # Pending items are considered first while retaining deterministic image/region ordering.
    selection_order = sorted(
        ordered_keys,
        key=lambda key: (
            0 if key in pending_priority else 1,
            active_by_key[key][0].image_name,
            active_by_key[key][0].region_index,
        ),
    )
    remaining_budget = config.recompute_region_budget
    selected: list[tuple[RegionEdit, dict[str, str], int]] = []
    pending_queue: list[str] = []
    budget_log: list[dict[str, Any]] = []
    budget_exhausted = False
    for seq, key in enumerate(selection_order, start=1):
        region, keys, cost = active_by_key[key]
        before = remaining_budget
        if not budget_exhausted and cost <= remaining_budget:
            remaining_budget -= cost
            action = "accepted"
            selected.append((region, keys, cost))
        else:
            budget_exhausted = True
            action = "pending"
            pending_queue.append(key)
        budget_log.append(
            {
                "seq": seq,
                "region_key": key,
                "sort_key": f"{region.image_name}|{region.region_index:06d}|mask+repair+attribute+style+compose",
                "priority": "pending" if key in pending_priority else "new",
                "required_pixels": cost,
                "remaining_before": before,
                "remaining_after": remaining_budget,
                "action": action,
                "business_code": E_RECOMPUTE_BUDGET_PENDING if action == "pending" else None,
            }
        )

    pending_queue.sort(key=lambda key: (active_by_key[key][0].image_name, active_by_key[key][0].region_index))
    for key in pending_queue:
        failures_out.append(
            BusinessFailure(
                E_RECOMPUTE_BUDGET_PENDING,
                "schedule",
                "run_pipeline",
                f"region deferred by pixel budget; required={active_by_key[key][2]} remaining={remaining_budget}",
                key,
            )
        )

    base_cache: dict[str, np.ndarray] = {}
    for image_name in sorted({region.image_name for region, _, _ in selected}):
        base_cache[image_name] = load_rgb(images[image_name].path)

    active_count = 0
    active_lock = threading.Lock()
    max_active = 0
    artifact_records: dict[str, dict[str, Any]] = {}

    def run_region(item: tuple[RegionEdit, dict[str, str], int]) -> tuple[dict[str, Any] | None, RegionTrace]:
        nonlocal active_count, max_active
        region, keys, cost = item
        with active_lock:
            active_count += 1
            max_active = max(max_active, active_count)
        try:
            processor = _RegionProcessor(
                image=images[region.image_name],
                base_rgb=base_cache[region.image_name],
                config=config,
                store=store,
            )
            try:
                record, trace = processor.process(region, keys, cost)
                return record, trace
            except WorkbenchError as exc:
                failure = BusinessFailure(
                    exc.code,
                    exc.stage,
                    region.region_key,
                    exc.message,
                    region.region_key,
                )
                return None, RegionTrace(
                    region_key=region.region_key,
                    image_name=region.image_name,
                    region_index=region.region_index,
                    sort_key=f"{region.image_name}|{region.region_index:06d}|pipeline",
                    status="failed",
                    budget_cost=cost,
                    failure=failure,
                )
        finally:
            with active_lock:
                active_count -= 1

    if selected:
        with ThreadPoolExecutor(max_workers=min(config.thread_pool_workers, len(selected))) as executor:
            processed = list(executor.map(run_region, selected))
    else:
        processed = []

    recomputed_regions: list[str] = []
    for record, trace in processed:
        traces.append(trace)
        if record is None:
            if trace.failure is not None:
                failures_out.append(trace.failure)
            continue
        region_records[record["region_key"]] = record
        recomputed_regions.append(record["region_key"])
        for stage in STAGES:
            stage_record = record["stages"].get(stage)
            if stage_record:
                artifact_records[stage_record["cache_key"]] = stage_record

    # Snapshot the full stage closure for every completed/reused region, including
    # stage-level reuse inside a region that otherwise needed recomputation.
    for record in region_records.values():
        for stage in STAGES:
            stage_record = record.get("stages", {}).get(stage)
            if isinstance(stage_record, dict):
                artifact_records[stage_record["cache_key"]] = stage_record

    # Ensure the compose artifacts required to assemble current outputs are present in the index.
    for record in region_records.values():
        layer = store.get_record(str(record["layer_key"]))
        if layer is not None:
            artifact_records[layer["cache_key"]] = layer

    recomputed_regions.sort(key=lambda key: (region_records[key]["image"], int(region_records[key]["region_index"])))
    reused_regions.sort(key=lambda key: (region_records[key]["image"], int(region_records[key]["region_index"])))
    traces.sort(key=lambda trace: (trace.image_name, trace.region_index, trace.status))

    output_paths, image_fingerprints = _compose_images(
        images=images,
        region_records=region_records,
        store=store,
        output_dir=output_dir,
    )
    stage_summaries: dict[str, dict[str, str]] = {stage: {} for stage in STAGES}
    for key in sorted(region_records):
        record = region_records[key]
        for stage in STAGES:
            stage_record = record.get("stages", {}).get(stage)
            if isinstance(stage_record, dict):
                stage_summaries[stage][key] = _array_summary(stage_record)
    merged_failures = sorted(
        failures_out,
        key=lambda failure: (failure.code, failure.region_key or "", failure.message),
    )
    affected = sum(int(record.get("pixel_count", 0)) for record in region_records.values())
    exchange_versions = sorted({region.exchange_version for region in regions})
    exchange_version = exchange_versions[0] if len(exchange_versions) == 1 else exchange_versions
    content_fp = sha256_json(
        {
            "config_hash": config.render_config_hash,
            "exchange_version": exchange_version,
            "exchange_fingerprint": exchange_fingerprint,
            "sidecar_hash": sidecar_hash,
            "images": image_fingerprints,
            "regions": [
                {
                    "region_key": key,
                    "content_fingerprint": region_records[key].get("content_fingerprint", ""),
                    "exchange_version": region_records[key].get("exchange_version"),
                }
                for key in sorted(region_records)
            ],
        }
    )
    metadata = {
        "style_quantization_levels": config.style_quantization_levels,
        "fixed_fraction_bits": config.fixed_fraction_bits,
        "rounding": config.rounding,
        "repair_neighborhood_radius": config.repair_neighborhood_radius,
        "feather_radius_max": str(config.feather_radius_max),
        "deterministic": True,
    }
    atomic_write_json(
        output_dir / "metadata.json",
        {
            **metadata,
            "content_fingerprint": content_fp,
            "image_fingerprints": image_fingerprints,
            "region_count": len(region_records),
        },
    )
    return PipelineResult(
        region_records=region_records,
        pending_queue=pending_queue,
        budget_log=budget_log,
        failures=merged_failures,
        traces=traces,
        reused_regions=reused_regions,
        recomputed_regions=recomputed_regions,
        output_images=output_paths,
        image_fingerprints=image_fingerprints,
        stage_summaries=stage_summaries,
        content_fingerprint=content_fp,
        max_active_tasks=max_active,
        artifact_records=artifact_records,
        affected_pixels=affected,
        metadata=metadata,
    )
