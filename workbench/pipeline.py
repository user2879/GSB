"""Six-stage editing pipeline with content-fingerprinted incremental reuse."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import algorithms as algo
from .config import Config, STAGE_NAMES, Tier
from .errors import WorkbenchError
from .sample_gen import validate_sample
from .store import SnapshotStore
from .utils import (
    ALGORITHM_VERSION,
    digest_array,
    digest_bytes,
    digest_obj,
    png_bytes,
    stable_sort_key,
)


class InterruptFlag:
    """Thread-safe cooperative shutdown flag for exit-signal handling."""

    def __init__(self):
        self._event = threading.Event()
        self.raised_signal = None

    def trigger(self, signal_name: str = "SIGTERM") -> None:
        self.raised_signal = signal_name
        self._event.set()

    @property
    def set(self) -> bool:
        return self._event.is_set()

    def wait_stage_boundary(self) -> bool:
        return self._event.is_set()


def stage_chain_keys(
    *,
    algo_version: str,
    image_hash: str,
    instruction_hash: str,
    resolution: tuple,
    tier_profile: dict,
    mask_path: str,
    mask_params: dict,
    inpaint_path: str,
    config_fingerprint: str,
    style_path: str,
    quant_levels: int,
) -> dict:
    """Derive every stage cache key from upstream content + parameter fingerprints."""
    parse_key = digest_obj(
        [
            "parse",
            algo_version,
            image_hash,
            instruction_hash,
            list(resolution),
            tier_profile,
        ]
    )
    mask_key = digest_obj(
        ["mask", algo_version, parse_key, mask_path, mask_params]
    )
    inpaint_key = digest_obj(
        ["inpaint", algo_version, mask_key, inpaint_path]
    )
    recolor_key = digest_obj(
        ["recolor", algo_version, inpaint_key, config_fingerprint]
    )
    style_key = digest_obj(
        ["style", algo_version, recolor_key, style_path, quant_levels]
    )
    composite_key = digest_obj(
        ["composite", algo_version, style_key, mask_key]
    )
    return {
        "parse": parse_key,
        "mask": mask_key,
        "inpaint": inpaint_key,
        "recolor": recolor_key,
        "style": style_key,
        "composite": composite_key,
    }


def effective_paths(global_selection: dict, instruction: dict) -> dict:
    """Apply per-sample deterministic hints on top of the global decision."""
    hints = {op.get("mask_hint") for op in instruction["ops"] if op.get("mask_hint")}
    repairs = {
        op.get("repair")
        for op in instruction["ops"]
        if op.get("repair") and op.get("repair") != "auto"
    }
    mask_path = next(iter(hints)) if len(hints) == 1 else global_selection["mask_path"]
    inpaint_path = (
        next(iter(repairs)) if len(repairs) == 1 else global_selection["inpaint_path"]
    )
    return {
        "mask_path": mask_path,
        "inpaint_path": inpaint_path,
        "style_path": global_selection["style_path"],
        "quant_levels": global_selection["quant_levels"],
        "source": "sample_hint" if (len(hints) == 1 or len(repairs) == 1) else "global",
    }


def _union_mask(masks: list) -> np.ndarray:
    union = np.zeros_like(masks[0])
    for mask in masks:
        union = np.maximum(union, mask)
    return union


def process_sample(
    config: Config,
    tier: Tier,
    name: str,
    store: SnapshotStore,
    global_decision: dict,
    manifest: dict | None = None,
    interrupt: InterruptFlag | None = None,
) -> dict:
    """Run (or reuse) the full stage chain for one sample."""
    persistent_fp = config.persistent_fingerprint()
    result = {
        "sample": name,
        "sort_key": stable_sort_key(name, "pipeline"),
        "status": "ok",
        "reused": [],
        "recomputed": [],
        "stage_summaries": {},
        "failure": None,
    }
    try:
        loaded = validate_sample(config, name, tier)
    except WorkbenchError as exc:
        result["status"] = "failed"
        result["failure"] = exc.to_record()
        result["failure"]["sample"] = name
        return result

    image = loaded["image"]
    instruction = loaded["instruction"]
    image_hash = loaded["image_hash"]
    instruction_hash = digest_obj(instruction)
    max_tier = max(config.tiers, key=lambda item: item.sample_count)
    paths = effective_paths(global_decision["selected"], instruction)
    mask_params = {
        "mask_threshold": config.mask_threshold,
        "edge_threshold": config.edge_threshold,
    }
    keys = stage_chain_keys(
        algo_version=ALGORITHM_VERSION,
        image_hash=image_hash,
        instruction_hash=instruction_hash,
        resolution=config.resolution,
        tier_profile={
            "regions": max_tier.regions,
            "instruction_length": max_tier.instruction_length,
            "primitives": max_tier.primitives,
        },
        mask_path=paths["mask_path"],
        mask_params=mask_params,
        inpaint_path=paths["inpaint_path"],
        config_fingerprint=persistent_fp,
        style_path=paths["style_path"],
        quant_levels=paths["quant_levels"],
    )

    if manifest is None:
        manifest = store.load_manifest()
    entry = manifest["samples"].get(name, {})
    prior_versions = entry.get("stage_versions", {})
    reuse_plan = {}
    for stage in STAGE_NAMES:
        prior = prior_versions.get(stage)
        reuse_plan[stage] = bool(prior and prior.get("key") == keys[stage])

    result["decision"] = {
        "global_decision_name": global_decision["selected"]["name"],
        "effective_paths": paths,
        "keys": keys,
    }
    result["image_hash"] = image_hash
    result["instruction_hash"] = instruction_hash

    # Stage 1: parse summary (pure fingerprint bookkeeping).
    if reuse_plan["parse"]:
        result["reused"].append("parse")
        result["stage_summaries"]["parse"] = store.read_json_object(keys["parse"])
    else:
        parse_summary = {
            "regions": len(instruction["regions"]),
            "ops": len(instruction["ops"]),
            "image_hash": image_hash,
            "instruction_hash": instruction_hash,
        }
        store.write_json_object(keys["parse"], parse_summary)
        result["recomputed"].append("parse")
        result["stage_summaries"]["parse"] = parse_summary

    # Stage 2: mask generation (one mask per region, integer bitmaps).
    if reuse_plan["mask"]:
        result["reused"].append("mask")
        mask_summary = store.read_json_object(keys["mask"])
        result["stage_summaries"]["mask"] = mask_summary
        union = store.read_array(mask_summary["union_mask_key"])
    else:
        masks = []
        region_areas = []
        for region in instruction["regions"]:
            mask = algo.generate_mask(paths["mask_path"], image, region, mask_params)
            masks.append(mask)
            region_areas.append(int(mask.sum()))
        union = _union_mask(masks)
        union_key = digest_array(union)
        store.write_array(union_key, union)
        mask_summary = {
            "path": paths["mask_path"],
            "params": mask_params,
            "region_areas": region_areas,
            "union_area": int(union.sum()),
            "union_mask_key": union_key,
        }
        store.write_json_object(keys["mask"], mask_summary)
        result["recomputed"].append("mask")
        result["stage_summaries"]["mask"] = mask_summary
    if interrupt is not None and interrupt.wait_stage_boundary():
        return _interrupted(result, interrupt)

    # Stage 3: inpainting (fixed-point integer kernels).
    if reuse_plan["inpaint"]:
        result["reused"].append("inpaint")
        inpaint_summary = store.read_json_object(keys["inpaint"])
        result["stage_summaries"]["inpaint"] = inpaint_summary
        inpainted = store.read_array(inpaint_summary["array_key"])
    else:
        inpainted = algo.inpaint(paths["inpaint_path"], image, union)
        array_key = digest_array(inpainted)
        store.write_array(array_key, inpainted)
        inpaint_summary = {
            "path": paths["inpaint_path"],
            "filled_pixels": int(union.sum()),
            "array_key": array_key,
        }
        store.write_json_object(keys["inpaint"], inpaint_summary)
        result["recomputed"].append("inpaint")
        result["stage_summaries"]["inpaint"] = inpaint_summary

    # Stage 4: deterministic recolouring per edit op.
    if reuse_plan["recolor"]:
        result["reused"].append("recolor")
        recolor_summary = store.read_json_object(keys["recolor"])
        result["stage_summaries"]["recolor"] = recolor_summary
        recolored = store.read_array(recolor_summary["array_key"])
    else:
        recolored = inpainted.copy()
        op_records = []
        total_changed = 0
        for op in instruction["ops"]:
            region = instruction["regions"][op["region_index"]]
            op_mask = algo.generate_mask(paths["mask_path"], image, region, mask_params)
            recolored, op_summary = algo.apply_recolor(
                recolored,
                op_mask,
                op["attribute_value"],
                op["style_strength"],
                persistent_fp,
            )
            op_records.append({"op_id": op["op_id"], **op_summary})
            total_changed += op_summary["changed_pixels"]
        array_key = digest_array(recolored)
        store.write_array(array_key, recolored)
        recolor_summary = {
            "changed_pixels_total": total_changed,
            "ops": op_records,
            "array_key": array_key,
        }
        store.write_json_object(keys["recolor"], recolor_summary)
        result["recomputed"].append("recolor")
        result["stage_summaries"]["recolor"] = recolor_summary
    if interrupt is not None and interrupt.wait_stage_boundary():
        return _interrupted(result, interrupt)

    # Stage 5: style mapping (LUT or per-channel fixed-point linear).
    style_seed = int(persistent_fp[:8], 16)
    if reuse_plan["style"]:
        result["reused"].append("style")
        style_summary = store.read_json_object(keys["style"])
        result["stage_summaries"]["style"] = style_summary
        styled = store.read_array(style_summary["array_key"])
    else:
        styled, style_meta = algo.apply_style(
            paths["style_path"], recolored, paths["quant_levels"], style_seed
        )
        array_key = digest_array(styled)
        store.write_array(array_key, styled)
        style_summary = {**style_meta, "array_key": array_key}
        store.write_json_object(keys["style"], style_summary)
        result["recomputed"].append("style")
        result["stage_summaries"]["style"] = style_summary

    # Stage 6: composition + deterministic PNG export.
    if reuse_plan["composite"]:
        result["reused"].append("composite")
        composite_summary = store.read_json_object(keys["composite"])
        result["stage_summaries"]["composite"] = composite_summary
        final_array = store.read_array(composite_summary["array_key"])
        png_data = store.object_path(composite_summary["png_key"], "png").read_bytes()
    else:
        final_array = algo.composite(image, styled, union)
        png_data = png_bytes(final_array)
        array_key = digest_array(final_array)
        png_key = digest_bytes(png_data)
        store.write_array(array_key, final_array)
        store.write_png(png_key, png_data)
        composite_summary = {
            "mask_retention": "edited-inside-mask/source-outside-mask",
            "edited_pixels": int(union.sum()),
            "array_key": array_key,
            "png_key": png_key,
            "pixel_hash": array_key,
        }
        store.write_json_object(keys["composite"], composite_summary)
        result["recomputed"].append("composite")
        result["stage_summaries"]["composite"] = composite_summary

    result["final_pixel_hash"] = composite_summary["array_key"]
    result["png_key"] = composite_summary["png_key"]
    result["png_bytes"] = png_data
    result["keys"] = keys
    return result


def _interrupted(result: dict, interrupt: InterruptFlag) -> dict:
    result["status"] = "interrupted"
    result["signal"] = interrupt.raised_signal
    return result


def deterministic_stage_cost(
    config: Config, tier: Tier, decision: dict, mask_path: str, inpaint_path: str
) -> dict:
    """Deterministic per-stage cost units (identical serial vs parallel)."""
    from .algorithms import estimate_cost_units

    pixels = config.width * config.height
    mask_cost = pixels * tier.regions * algo.MASK_COST[mask_path]
    inpaint_cost = pixels * tier.regions * algo.INPAINT_COST[inpaint_path]
    recolor_cost = pixels * tier.instruction_length
    style_unit = estimate_cost_units(
        pixels=pixels,
        regions=tier.regions,
        instruction_length=tier.instruction_length,
        mask_path=mask_path,
        inpaint_path=inpaint_path,
        style_path=decision["style_path"],
        quant_levels=decision["quant_levels"],
    )
    style_cost = (
        pixels
        * algo.STYLE_COST[decision["style_path"]]
        * (256.0 / decision["quant_levels"])
    )
    return {
        "parse": round(tier.regions + tier.instruction_length, 3),
        "mask": round(mask_cost, 3),
        "inpaint": round(inpaint_cost, 3),
        "recolor": round(recolor_cost, 3),
        "style": round(style_cost, 3),
        "composite": round(float(pixels), 3),
        "total_estimate_units": round(style_unit, 3),
    }


def _commit_sample(store: SnapshotStore, manifest: dict, result: dict) -> None:
    """Atomically update the manifest for one finished sample."""
    name = result["sample"]
    if result["status"] == "failed":
        manifest["global_version"] += 1
        failure = result["failure"]
        failure["version"] = manifest["global_version"]
        manifest["failures"] = [
            item for item in manifest["failures"] if item["sample"] != name
        ]
        manifest["failures"].append(failure)
        manifest["samples"].pop(name, None)
        if name in manifest["completed_samples"]:
            manifest["completed_samples"].remove(name)
        return

    if result["status"] == "interrupted":
        return

    prior_entry = manifest["samples"].get(name)
    if (
        prior_entry is not None
        and prior_entry.get("final_pixel_hash") == result["final_pixel_hash"]
        and all(
            prior_entry.get("stage_versions", {}).get(stage, {}).get("key")
            == result["keys"][stage]
            for stage in STAGE_NAMES
        )
    ):
        # Nothing changed: reuse the prior immutable versions, no version bump.
        return

    keys = result["keys"]
    manifest["global_version"] += 1
    stage_versions = {
        stage: {"key": keys[stage], "version": manifest["global_version"]}
        for stage in STAGE_NAMES
    }
    manifest["samples"][name] = {
        "version": manifest["global_version"],
        "image_hash": result["image_hash"],
        "instruction_hash": result["instruction_hash"],
        "decision": result["decision"],
        "stage_versions": stage_versions,
        "final_pixel_hash": result["final_pixel_hash"],
        "png_key": result["png_key"],
        "reused": result["reused"],
        "recomputed": result["recomputed"],
    }
    manifest["failures"] = [
        item for item in manifest["failures"] if item["sample"] != name
    ]
    if name not in manifest["completed_samples"]:
        manifest["completed_samples"].append(name)


def run_pipeline(
    config: Config,
    *,
    run_mode: str = "parallel",
    reset: bool = False,
    interrupt: InterruptFlag | None = None,
) -> dict:
    """Execute one incremental run and return a fully deterministic report."""
    from .sample_gen import generate_tier
    from .strategy import select_strategy

    tier = config.tier()
    store = SnapshotStore(config.snapshots_dir)
    names = generate_tier(config, tier)

    if reset:
        manifest = store.fresh_manifest(config.persistent_fingerprint())
    else:
        manifest = store.resume_manifest(config.persistent_fingerprint())
    pruned = store.prune_manifest(manifest, set(names)) if not reset else 0
    store.commit_manifest(manifest)

    # Global strategy decision from aggregate scale metrics.
    constraints = {"mask": None, "inpaint": None}
    total_regions = tier.sample_count * tier.regions
    global_decision = select_strategy(
        sample_count=tier.sample_count,
        primitives=tier.primitives,
        total_regions=total_regions,
        pixels=config.width * config.height,
        regions_per_sample=tier.regions,
        instruction_length=tier.instruction_length,
        quant_levels=tuple(
            sorted({config.effective_quant(), *config.quant_levels})
        ),
        latency_budget_ms=config.latency_budget_ms,
        quality_weight=config.quality_weight,
        constraints=constraints,
    )
    # quant_override forces the selected quantization level.
    if config.quant_override is not None:
        global_decision["selected"]["quant_levels"] = config.quant_override
        global_decision["selected"]["name"] = (
            f"{global_decision['selected']['mask_path']}|"
            f"{global_decision['selected']['inpaint_path']}|"
            f"{global_decision['selected']['style_path']}|"
            f"q{config.quant_override}"
        )
        global_decision["reason"] += "; quant_override applied"

    manifest = store.load_manifest()
    started_version = manifest["global_version"]

    # Stable per-sample ordering derived from sample names; tasks also carry a
    # sample::stage key so scheduling order is auditable.
    ordered_names = sorted(names)
    task_specs = [
        {
            "sample": name,
            "sort_key": stable_sort_key(name, "pipeline"),
            "stage_keys": [stable_sort_key(name, stage) for stage in STAGE_NAMES],
        }
        for name in ordered_names
    ]

    results = []
    wall_times = {}
    import time

    def _work(name):
        start = time.perf_counter()
        outcome = process_sample(
            config, tier, name, store, global_decision, manifest, interrupt
        )
        wall_times[name] = round((time.perf_counter() - start) * 1000.0, 3)
        return outcome

    workers = config.effective_workers()
    if run_mode == "serial" or workers == 1:
        for spec in task_specs:
            if interrupt is not None and interrupt.set:
                break
            results.append(_work(spec["sample"]))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(_work, spec["sample"]): spec["sample"]
                for spec in task_specs
            }
            for future in future_map:
                future.add_done_callback(lambda _: None)
            gathered = {}
            for future in future_map:
                gathered[future_map[future]] = future.result()
            results = [gathered[spec["sample"]] for spec in task_specs]

    # Commit in canonical sample-name order on the same in-memory manifest the
    # workers consulted, so unchanged samples short-circuit without a bump.
    for result in results:
        _commit_sample(store, manifest, result)
        store.commit_manifest(manifest)

    interrupted = any(result["status"] == "interrupted" for result in results)
    completed = sorted(
        name for name in manifest["completed_samples"] if name in set(names)
    )
    failures = sorted(
        [failure for failure in manifest["failures"] if failure["sample"] in set(names)],
        key=lambda item: item["sample"],
    )

    reused_counts = {stage: 0 for stage in STAGE_NAMES}
    recomputed_counts = {stage: 0 for stage in STAGE_NAMES}
    for result in results:
        for stage in result.get("reused", []):
            reused_counts[stage] += 1
        for stage in result.get("recomputed", []):
            recomputed_counts[stage] += 1

    stage_cost = deterministic_stage_cost(
        config, tier, global_decision["selected"],
        global_decision["selected"]["mask_path"],
        global_decision["selected"]["inpaint_path"],
    )
    run_id = digest_obj(
        [
            config.full_fingerprint(),
            tier.name,
            run_mode,
            started_version,
            manifest["global_version"],
        ]
    )[:12]
    # Run bookkeeping does not change stage artifacts, so it never bumps the
    # global version; version numbers track content changes only.
    run_record = {
        "run_id": run_id,
        "tier": tier.name,
        "run_mode": run_mode,
        "workers": workers if run_mode == "parallel" else 1,
        "interrupted": interrupted,
        "signal": interrupt.raised_signal if interrupt else None,
        "pruned_samples": pruned,
        "version_range": [started_version, manifest["global_version"]],
    }
    manifest["runs"].append(run_record)
    store.commit_manifest(manifest)

    return {
        "run_id": run_id,
        "tier": tier.name,
        "run_mode": run_mode,
        "workers": workers,
        "global_decision": global_decision,
        "samples": [
            {
                key: value
                for key, value in result.items()
                if key not in ("png_bytes",)
            }
            for result in results
        ],
        "completed_samples": completed,
        "failures": failures,
        "reuse_counts": reused_counts,
        "recompute_counts": recomputed_counts,
        "stage_cost_units": stage_cost,
        "wall_time_ms": dict(sorted(wall_times.items())),
        "task_specs": task_specs,
        "interrupted": interrupted,
        "pruned_samples": pruned,
        "final_version": manifest["global_version"],
        "config_fingerprint": config.persistent_fingerprint(),
    }
