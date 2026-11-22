from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from .config import WorkbenchConfig, config_from_dict
from .errors import BusinessFailure, E_STRUCTURE_INVALID, WorkbenchError
from .exchange import MaterializedExchange, load_exchange_file, materialize_logical_state
from .image_registry import image_collection_fingerprint, load_images
from .manifest import load_edit_manifest
from .models import ImageInfo, ParseOutcome, Rect, RegionEdit
from .pipeline import PipelineResult, run_pipeline
from .snapshot import SnapshotStore, resume_decision
from .util import dec, dec_str, read_json, sha256_json


@dataclass
class ScenarioState:
    regions: list[RegionEdit]
    config: WorkbenchConfig
    exchange_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_version": self.exchange_version,
            "config": self.config.to_dict(),
            "regions": [
                {
                    "image": region.image_name,
                    "region_index": region.region_index,
                    "rect": region.rect.to_dict(),
                    "style_strength": dec_str(region.style_strength),
                    "target_attribute": region.target_attribute,
                    "repair_method": region.repair_method,
                    "precision_label": region.precision_label,
                    "rotation_deg": dec_str(region.rotation_deg),
                    "feather_radius": dec_str(region.feather_radius),
                }
                for region in sorted(self.regions, key=lambda item: (item.image_name, item.region_index))
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScenarioState":
        config = config_from_dict(dict(payload["config"]))
        regions: list[RegionEdit] = []
        for record in payload["regions"]:
            regions.append(
                RegionEdit(
                    exchange_version=2,
                    image_name=str(record["image"]),
                    region_index=int(record["region_index"]),
                    rect=Rect.from_values(
                        record["rect"]["x"],
                        record["rect"]["y"],
                        record["rect"]["w"],
                        record["rect"]["h"],
                    ),
                    style_strength=dec(record["style_strength"]),
                    target_attribute=str(record["target_attribute"]),
                    repair_method=str(record["repair_method"]),
                    precision_label=str(record.get("precision_label", "manifest:logical")),
                    rotation_deg=dec(record.get("rotation_deg", 0)),
                    feather_radius=dec(record.get("feather_radius", 0)),
                    source_format="manifest",
                )
            )
        return cls(
            regions=sorted(regions, key=lambda item: (item.image_name, item.region_index)),
            config=config,
            exchange_version=int(payload.get("exchange_version", 2)),
        )


@dataclass
class BatchSummary:
    budget_log: list[dict[str, Any]]
    pending_queue: list[str]
    recomputed_regions: list[str]
    reused_regions: list[str]
    max_active_tasks: int
    content_fingerprint: str
    affected_pixels: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_log": self.budget_log,
            "pending_queue": self.pending_queue,
            "recomputed_regions": self.recomputed_regions,
            "reused_regions": self.reused_regions,
            "max_active_tasks": self.max_active_tasks,
            "content_fingerprint": self.content_fingerprint,
            "affected_pixels": self.affected_pixels,
        }


@dataclass
class CommandRun:
    command_id: str
    op: str
    state: ScenarioState
    exchange: MaterializedExchange
    result: PipelineResult
    batches: list[BatchSummary]
    snapshot: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "op": self.op,
            "exchange_version": self.exchange.version,
            "exchange_fingerprint": self.exchange.exchange_fingerprint,
            "sidecar_hash": self.exchange.sidecar_hash,
            "batches": [batch.to_dict() for batch in self.batches],
            "pipeline": self.result.to_diagnostic_dict(),
            "snapshot": self.snapshot,
            "verification": self.verification,
        }


@dataclass
class ScenarioRun:
    commands: list[CommandRun]
    final_state: ScenarioState
    final_result: PipelineResult
    final_exchange: MaterializedExchange
    snapshot_history: list[dict[str, Any]] = field(default_factory=list)
    resume: dict[str, Any] = field(default_factory=dict)


def load_commands(path: Path) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                command = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorkbenchError(
                    E_STRUCTURE_INVALID,
                    f"invalid command JSON at line {line_number}: {exc}",
                    stage="commands",
                ) from exc
            if not isinstance(command, dict) or "op" not in command:
                raise WorkbenchError(E_STRUCTURE_INVALID, f"command at line {line_number} needs op", stage="commands")
            command.setdefault("id", f"CMD-{len(commands) + 1:03d}")
            commands.append(command)
    return commands


def _find_region(state: ScenarioState, image: str, region_index: int) -> RegionEdit:
    for region in state.regions:
        if region.image_name == image and region.region_index == region_index:
            return region
    raise WorkbenchError(E_STRUCTURE_INVALID, f"region not found: {image}#{region_index:03d}", stage="commands")


def _replace_region(state: ScenarioState, replacement: RegionEdit) -> None:
    for index, region in enumerate(state.regions):
        if region.image_name == replacement.image_name and region.region_index == replacement.region_index:
            state.regions[index] = replacement
            state.regions.sort(key=lambda item: (item.image_name, item.region_index))
            return
    raise WorkbenchError(E_STRUCTURE_INVALID, f"region not found: {replacement.region_key}", stage="commands")


CONFIG_PATHS = {
    "coordinate_tolerance": "coordinate_tolerance",
    "style_quantization_levels": "style_quantization_levels",
    "fixed_fraction_bits": "fixed_fraction_bits",
    "repair_neighborhood_radius": "repair_neighborhood_radius",
    "feather_radius_max": "feather_radius_max",
    "recompute_region_budget": "recompute_region_budget",
    "thread_pool_workers": "thread_pool_workers",
    "snapshot_commit_interval_commands": "snapshot_commit_interval_commands",
    "pixel_difference_tolerance": "pixel_difference_tolerance",
}


def apply_command(state: ScenarioState, command: Mapping[str, Any]) -> dict[str, Any]:
    op = str(command["op"])
    if op == "set_exchange_version":
        version = int(command["version"])
        if version not in (1, 2):
            raise WorkbenchError(E_STRUCTURE_INVALID, "exchange version must be 1 or 2", stage="commands")
        state.exchange_version = version
        return {}
    if op == "set_config":
        key = str(command.get("key", ""))
        if key not in CONFIG_PATHS:
            raise WorkbenchError(E_STRUCTURE_INVALID, f"unsupported config key {key!r}", stage="commands")
        value = command.get("value")
        if key in ("coordinate_tolerance", "feather_radius_max"):
            value = dec(value)
        state.config = state.config.updated(**{CONFIG_PATHS[key]: value})
        return {}
    if op == "update_region":
        image = str(command["image"])
        region_index = int(command["region_index"])
        region = _find_region(state, image, region_index)
        changes = command.get("changes", {})
        if not isinstance(changes, dict):
            raise WorkbenchError(E_STRUCTURE_INVALID, "changes must be an object", stage="commands")
        updates: dict[str, Any] = {}
        if "rect" in changes:
            rect = changes["rect"]
            updates["rect"] = Rect.from_values(rect["x"], rect["y"], rect["w"], rect["h"])
        if "style_strength" in changes:
            updates["style_strength"] = dec(changes["style_strength"])
        if "target_attribute" in changes:
            updates["target_attribute"] = str(changes["target_attribute"])
        if "repair_method" in changes:
            updates["repair_method"] = str(changes["repair_method"])
        if "rotation_deg" in changes:
            updates["rotation_deg"] = dec(changes["rotation_deg"])
        if "feather_radius" in changes:
            updates["feather_radius"] = dec(changes["feather_radius"])
        replacement = region.with_fields(**updates)
        replacement.validate_semantics()
        _replace_region(state, replacement)
        return {"changed_region": replacement.region_key, "changes": sorted(updates)}
    if op in ("run", "drain", "noop", "load_exchange_file"):
        return {}
    raise WorkbenchError(E_STRUCTURE_INVALID, f"unsupported command op {op!r}", stage="commands")


def build_snapshot_manifest(
    *,
    result: PipelineResult,
    state: ScenarioState,
    exchange: MaterializedExchange,
    images: Mapping[str, ImageInfo],
    config: WorkbenchConfig,
    command_id: str | None = None,
    scenario_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_index: dict[str, dict[str, Any]] = {}
    for record in result.region_records.values():
        for stage in ("parse", "mask", "repair", "attribute", "style", "compose"):
            stage_record = record.get("stages", {}).get(stage)
            if isinstance(stage_record, dict):
                artifact_index[stage_record["cache_key"]] = stage_record
    region_records = sorted(
        result.region_records.values(),
        key=lambda record: (record["image"], int(record["region_index"])),
    )
    return {
        "schema_version": 1,
        "config_hash": config.config_hash,
        "config": config.to_dict(),
        "exchange_version": exchange.version,
        "exchange_fingerprint": exchange.exchange_fingerprint,
        "sidecar_hash": exchange.sidecar_hash,
        "image_fingerprints": {name: images[name].source_fingerprint for name in sorted(images)},
        "image_collection_fingerprint": image_collection_fingerprint(images),
        "completed_regions": sorted(result.region_records),
        "pending_queue": list(result.pending_queue),
        "failures": [failure.to_dict() for failure in result.failures],
        "region_records": region_records,
        "artifact_index": artifact_index,
        "budget_log": result.budget_log,
        "max_active_tasks": result.max_active_tasks,
        "content_fingerprint": result.content_fingerprint,
        "stage_summaries": result.stage_summaries,
        "reused_regions": result.reused_regions,
        "recomputed_regions": result.recomputed_regions,
        "last_command_id": command_id,
        "scenario_state": scenario_state,
        "metadata": result.metadata,
    }


def _cold_render(
    *,
    exchange: MaterializedExchange,
    state: ScenarioState,
    images: Mapping[str, ImageInfo],
    config: WorkbenchConfig,
    state_dir: Path,
    output_dir: Path,
    drain: bool,
) -> tuple[PipelineResult, list[BatchSummary], list[dict[str, Any]]]:
    if state_dir.exists():
        shutil.rmtree(state_dir)
    store = SnapshotStore(state_dir)
    store.load()
    prior_manifest: dict[str, Any] = {}
    result: PipelineResult | None = None
    batches: list[BatchSummary] = []
    snapshots: list[dict[str, Any]] = []
    for _ in range(100):
        result = run_pipeline(
            regions=exchange.parsed.regions,
            images=images,
            config=config,
            store=store,
            prior_manifest=prior_manifest,
            output_dir=output_dir,
            failures=list(exchange.parsed.failures),
            sidecar_hash=exchange.sidecar_hash,
            exchange_fingerprint=exchange.exchange_fingerprint,
        )
        batches.append(
            BatchSummary(
                budget_log=result.budget_log,
                pending_queue=result.pending_queue,
                recomputed_regions=result.recomputed_regions,
                reused_regions=result.reused_regions,
                max_active_tasks=result.max_active_tasks,
                content_fingerprint=result.content_fingerprint,
                affected_pixels=result.affected_pixels,
            )
        )
        manifest = build_snapshot_manifest(
            result=result,
            state=state,
            exchange=exchange,
            images=images,
            config=config,
        )
        snapshot = store.commit(manifest)
        snapshots.append(snapshot)
        prior_manifest = manifest
        if not drain or not result.pending_queue:
            break
        if not result.recomputed_regions and not result.reused_regions:
            break
    assert result is not None
    return result, batches, snapshots


def cold_render_exchange(
    *,
    exchange: MaterializedExchange,
    state: ScenarioState,
    images: Mapping[str, ImageInfo],
    config: WorkbenchConfig,
    state_dir: Path,
    output_dir: Path,
    drain: bool = True,
) -> tuple[PipelineResult, list[BatchSummary], list[dict[str, Any]]]:
    return _cold_render(
        exchange=exchange,
        state=state,
        images=images,
        config=config,
        state_dir=state_dir,
        output_dir=output_dir,
        drain=drain,
    )


def _compare_results(
    incremental: PipelineResult,
    reference: PipelineResult,
    *,
    incremental_output: Path,
    reference_output: Path,
    images: Mapping[str, ImageInfo],
    tolerance: int,
) -> dict[str, Any]:
    pixel_stats: dict[str, Any] = {}
    all_equal = True
    for image_name in sorted(images):
        left_path = incremental_output / image_name
        right_path = reference_output / image_name
        if not left_path.exists() or not right_path.exists():
            all_equal = False
            pixel_stats[image_name] = {"available": False}
            continue
        with Image.open(left_path) as left_image, Image.open(right_path) as right_image:
            left = np.asarray(left_image.convert("RGB"), dtype=np.int16)
            right = np.asarray(right_image.convert("RGB"), dtype=np.int16)
        delta = np.abs(left - right)
        changed = int(np.count_nonzero(np.any(delta > 0, axis=2)))
        max_delta = int(delta.max()) if delta.size else 0
        mean_delta = float(delta.mean()) if delta.size else 0.0
        image_equal = max_delta <= tolerance
        all_equal = all_equal and image_equal
        pixel_stats[image_name] = {
            "available": True,
            "max_channel_delta": max_delta,
            "mean_channel_delta": round(mean_delta, 6),
            "changed_pixels": changed,
            "within_tolerance": image_equal,
            "fingerprint_equal": incremental.image_fingerprints.get(image_name)
            == reference.image_fingerprints.get(image_name),
        }
    stage_equal = incremental.stage_summaries == reference.stage_summaries
    content_equal = incremental.content_fingerprint == reference.content_fingerprint
    pending_equal = incremental.pending_queue == reference.pending_queue
    return {
        "pixels_equal": all_equal,
        "pixel_stats": pixel_stats,
        "stage_summaries_equal": stage_equal,
        "content_fingerprint_equal": content_equal,
        "pending_queue_equal": pending_equal,
        "incremental_reused": incremental.reused_regions,
        "incremental_recomputed": incremental.recomputed_regions,
        "reference_recomputed": reference.recomputed_regions,
        "raw_reuse_lists_equal": incremental.reused_regions == reference.reused_regions,
        "semantic_equivalence": all_equal and stage_equal and content_equal and pending_equal,
    }


class ScenarioRunner:
    def __init__(
        self,
        *,
        sample_dir: Path,
        state_dir: Path,
        output_dir: Path,
        reference_root: Path,
    ) -> None:
        self.sample_dir = sample_dir
        self.state_dir = state_dir
        self.output_dir = output_dir
        self.reference_root = reference_root

    def run(
        self,
        *,
        commands: list[dict[str, Any]],
        config: WorkbenchConfig,
        exchange_version: int,
        drain_after: bool = True,
        verify_each_command: bool = True,
        keep_reference: bool = True,
        initial_state: ScenarioState | None = None,
    ) -> ScenarioRun:
        image_dir = self.sample_dir / "images" if (self.sample_dir / "images").is_dir() else self.sample_dir
        images = load_images(image_dir, expected_size=config.image_size)
        manifest = load_edit_manifest(self.sample_dir / "edits.json", images=images)
        if initial_state is None:
            state = ScenarioState(regions=list(manifest.regions), config=config, exchange_version=exchange_version)
        else:
            state = initial_state
        if manifest.failures:
            raise WorkbenchError(E_STRUCTURE_INVALID, f"edit manifest has {len(manifest.failures)} failures", stage="manifest")

        store = SnapshotStore(self.state_dir)
        store.load()
        store.begin_work()
        prior_manifest = store.current_manifest
        initial_exchange = materialize_logical_state(
            state.regions,
            version=state.exchange_version,
            images=images,
            manifest_keys={region.region_key for region in state.regions},
        )
        resume = resume_decision(
            prior_manifest,
            config_hash=config.config_hash,
            exchange_version=state.exchange_version,
            exchange_fingerprint=initial_exchange.exchange_fingerprint,
            sidecar_hash=initial_exchange.sidecar_hash,
            image_fingerprints={name: images[name].source_fingerprint for name in sorted(images)},
        )
        command_runs: list[CommandRun] = []
        snapshot_history: list[dict[str, Any]] = []
        final_result: PipelineResult | None = None
        final_exchange = initial_exchange

        for index, command in enumerate(commands, start=1):
            command_id = str(command["id"])
            op = str(command["op"])
            apply_command(state, command)
            if op == "load_exchange_file":
                path = Path(str(command["path"]))
                if not path.is_absolute():
                    path = self.sample_dir.parent / path
                exchange = load_exchange_file(
                    path,
                    version=int(command.get("version", state.exchange_version)),
                    images=images,
                    manifest_keys={region.region_key for region in state.regions},
                )
            else:
                exchange = materialize_logical_state(
                    state.regions,
                    version=state.exchange_version,
                    images=images,
                    manifest_keys={region.region_key for region in state.regions},
                )
            final_exchange = exchange
            batches: list[BatchSummary] = []
            result: PipelineResult | None = None
            max_batches = 100 if (drain_after or op == "drain") else 1
            for _ in range(max_batches):
                result = run_pipeline(
                    regions=exchange.parsed.regions,
                    images=images,
                    config=state.config,
                    store=store,
                    prior_manifest=prior_manifest,
                    output_dir=self.output_dir,
                    failures=list(exchange.parsed.failures),
                    sidecar_hash=exchange.sidecar_hash,
                    exchange_fingerprint=exchange.exchange_fingerprint,
                )
                batches.append(
                    BatchSummary(
                        budget_log=result.budget_log,
                        pending_queue=result.pending_queue,
                        recomputed_regions=result.recomputed_regions,
                        reused_regions=result.reused_regions,
                        max_active_tasks=result.max_active_tasks,
                        content_fingerprint=result.content_fingerprint,
                        affected_pixels=result.affected_pixels,
                    )
                )
                prior_manifest = build_snapshot_manifest(
                    result=result,
                    state=state,
                    exchange=exchange,
                    images=images,
                    config=state.config,
                    command_id=command_id,
                    scenario_state=state.to_dict(),
                )
                if not result.pending_queue:
                    break
                if op != "drain" and not drain_after:
                    break
                if not result.recomputed_regions:
                    break
            assert result is not None
            final_result = result

            verification = None
            if verify_each_command:
                ref_dir = self.reference_root / command_id
                ref_state_dir = ref_dir / "state"
                ref_output = ref_dir / "outputs"
                reference, _, _ = _cold_render(
                    exchange=exchange,
                    state=state,
                    images=images,
                    config=state.config,
                    state_dir=ref_state_dir,
                    output_dir=ref_output,
                    drain=drain_after or op == "drain",
                )
                verification = _compare_results(
                    result,
                    reference,
                    incremental_output=self.output_dir,
                    reference_output=ref_output,
                    images=images,
                    tolerance=state.config.pixel_difference_tolerance,
                )
                if op == "load_exchange_file" and command.get("injection_id"):
                    expected = str(command.get("expected_code", ""))
                    hits = [failure for failure in result.failures if failure.code == expected]
                    verification["injection_id"] = str(command["injection_id"])
                    verification["expected_code"] = expected
                    verification["injection_hit"] = bool(hits)
                if not keep_reference and command_id != commands[-1]["id"]:
                    shutil.rmtree(ref_dir, ignore_errors=True)

            snapshot = None
            interval = state.config.snapshot_commit_interval_commands
            if index % interval == 0 or index == len(commands):
                snapshot = store.commit(
                    build_snapshot_manifest(
                        result=result,
                        state=state,
                        exchange=exchange,
                        images=images,
                        config=state.config,
                        command_id=command_id,
                        scenario_state=state.to_dict(),
                    )
                )
                snapshot_history.append(snapshot)
            command_runs.append(
                CommandRun(
                    command_id=command_id,
                    op=op,
                    state=ScenarioState(
                        regions=list(state.regions),
                        config=state.config,
                        exchange_version=state.exchange_version,
                    ),
                    exchange=exchange,
                    result=result,
                    batches=batches,
                    snapshot=snapshot,
                    verification=verification,
                )
            )

        if final_result is None:
            raise WorkbenchError(E_STRUCTURE_INVALID, "no commands were executed", stage="commands")
        return ScenarioRun(
            commands=command_runs,
            final_state=state,
            final_result=final_result,
            final_exchange=final_exchange,
            snapshot_history=snapshot_history,
            resume=resume,
        )


def load_resume_state(state_dir: Path) -> ScenarioState:
    store = SnapshotStore(state_dir)
    store.load()
    scenario_state = store.current_manifest.get("scenario_state")
    if not isinstance(scenario_state, dict):
        raise WorkbenchError(
            E_STRUCTURE_INVALID,
            "latest snapshot does not contain a resumable scenario_state",
            stage="resume",
        )
    return ScenarioState.from_dict(scenario_state)
