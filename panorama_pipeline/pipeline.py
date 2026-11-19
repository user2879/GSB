from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .codec import (
    load_metadata_document,
    parse_v1_package,
    parse_v2_package,
)
from .config import PipelineConfig
from .errors import E_MASK_FORMAT_INVALID, Failure, PipelineError
from .fusion import FusionResult, fuse_instances, fusion_dependency_fingerprint
from .models import (
    InstanceMask,
    ParseBundle,
    instance_key_numeric_sort_key,
    instance_numeric_sort_key,
    parse_instance_key,
)
from .rle import mask_to_runs, runs_to_mask


@dataclass(frozen=True)
class SampleLayout:
    root: Path
    config_path: Path
    metadata_path: Path
    v1_path: Path
    v2_path: Path
    images_dir: Path
    snapshot_dir: Path

    @classmethod
    def discover(cls, root: str | Path) -> "SampleLayout":
        sample_root = Path(root).resolve()
        return cls(
            root=sample_root,
            config_path=sample_root / "config.json",
            metadata_path=sample_root / "instances.json",
            v1_path=sample_root / "masks_v1.bin",
            v2_path=sample_root / "masks_v2.json",
            images_dir=sample_root / "images",
            snapshot_dir=sample_root.parent / "snapshots",
        )


class PipelineSession:
    def __init__(
        self,
        layout: SampleLayout,
        config: PipelineConfig,
        *,
        formats: tuple[str, ...] = ("v1", "v2"),
    ):
        self.layout = layout
        self.config = config
        self.formats = tuple(sorted(formats))
        self.metadata_document = load_metadata_document(layout.metadata_path)
        self.instances: dict[str, InstanceMask] = {}
        self._format_bundles: dict[str, ParseBundle] = {}
        self._format_failures: dict[str, list[Failure]] = {}
        self.decode_counts: dict[str, int] = {}
        self.events: list[dict[str, Any]] = []
        self.fusion_result: FusionResult | None = None
        self.latest_preview: dict[str, Any] | None = None
        self._progress_lock = threading.Lock()
        self._progress: dict[str, Any] = {
            "command_seq": 0,
            "fraction": 0.0,
            "completed_instances": 0,
            "total_instances": 0,
            "phase": "initializing",
        }
        self._lock = threading.RLock()
        self._load_selected_formats(initial=True)
        self.fusion_result = self._compute_fusion(command_seq=0)
        self.latest_preview = self._preview(command_seq=0, event=None, fusion_reused=False)

    @property
    def width(self) -> int:
        bundles = [bundle for bundle in self._format_bundles.values() if bundle.width > 0]
        return bundles[0].width if bundles else 0

    @property
    def height(self) -> int:
        bundles = [bundle for bundle in self._format_bundles.values() if bundle.height > 0]
        return bundles[0].height if bundles else 0

    @property
    def sorted_instances(self) -> list[InstanceMask]:
        return sorted(self.instances.values(), key=instance_numeric_sort_key)

    @property
    def failures(self) -> list[Failure]:
        failures = list(self.metadata_document.failures)
        for source_format in self.formats:
            failures.extend(self._format_failures.get(source_format, []))
        bundles = list(self._format_bundles.values())
        if len({(bundle.width, bundle.height) for bundle in bundles}) > 1:
            failures.append(
                Failure(
                    "fusion:package",
                    E_MASK_FORMAT_INVALID,
                    "fusion",
                    "v1 and v2 image dimensions disagree",
                )
            )
        return sorted(set(failures), key=lambda item: item.stable_sort_key)

    def _load_selected_formats(self, *, initial: bool) -> None:
        parser = {
            "v1": lambda: parse_v1_package(
                self.layout.v1_path,
                config=self.config,
                metadata=self.metadata_document,
            ),
            "v2": lambda: parse_v2_package(
                self.layout.v2_path,
                config=self.config,
            ),
        }
        for source_format in self.formats:
            bundle = parser[source_format]()
            self._format_bundles[source_format] = bundle
            self._format_failures[source_format] = bundle.sorted_failures()
            for item in bundle.instances:
                self.instances[item.key] = item
                self.decode_counts[item.key] = 1
        self._progress["total_instances"] = len(self.instances)

    def _compute_fusion(self, *, command_seq: int) -> FusionResult:
        return fuse_instances(
            self.sorted_instances,
            config=self.config,
            max_workers=self.config.max_workers,
            progress_callback=lambda fraction, completed: self._update_progress(
                command_seq,
                fraction,
                completed,
                "fusion",
            ),
        )

    def _update_progress(
        self,
        command_seq: int,
        fraction: float,
        completed_instances: int,
        phase: str,
    ) -> None:
        with self._progress_lock:
            self._progress = {
                "command_seq": command_seq,
                "fraction": float(fraction),
                "completed_instances": int(completed_instances),
                "total_instances": len(self.instances),
                "phase": phase,
            }

    def progress(self) -> dict[str, Any]:
        with self._progress_lock:
            return dict(self._progress)

    def _preview(
        self,
        *,
        command_seq: int,
        event: dict[str, Any] | None,
        fusion_reused: bool,
    ) -> dict[str, Any]:
        assert self.fusion_result is not None
        invalidated = [] if event is None else list(event["invalidated_instances"])
        codes = list(self.fusion_result.failure_codes)
        return {
            "command_seq": int(command_seq),
            "completed_instances": len(self.instances),
            "total_instances": len(self.instances),
            "affected_instances": invalidated,
            "class_distribution": dict(self.fusion_result.class_distribution),
            "progress": 1.0,
            "params_snapshot_version": self.config.params_snapshot_version(),
            "fusion_result_version": self.fusion_result.fusion_result_version,
            "fusion_reused": bool(fusion_reused),
            "business_codes": codes,
            "is_latest": True,
            "instance_result_versions": {
                item.key: item.cache_key for item in self.sorted_instances
            },
        }

    def _event(
        self,
        *,
        command_seq: int,
        trigger: str,
        invalidated: list[str],
        reused: list[str],
        fusion_invalidated: bool,
        changed_fields: list[str],
        decoded_instances: list[str],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "command_seq": int(command_seq),
            "trigger": trigger,
            "changed_fields": sorted(changed_fields),
            "invalidated_instances": sorted(
                invalidated,
                key=instance_key_numeric_sort_key,
            ),
            "reused_instances": sorted(
                reused,
                key=instance_key_numeric_sort_key,
            ),
            "decoded_instances": sorted(
                decoded_instances,
                key=instance_key_numeric_sort_key,
            ),
            "fusion_invalidated": bool(fusion_invalidated),
            "fusion_result_version": (
                None
                if self.fusion_result is None
                else self.fusion_result.fusion_result_version
            ),
            "instance_result_versions": {
                item.key: item.cache_key for item in self.sorted_instances
            },
            "decode_counts": dict(
                sorted(
                    self.decode_counts.items(),
                    key=lambda item: instance_key_numeric_sort_key(item[0]),
                )
            ),
        }
        if extra:
            event.update(extra)
        self.events.append(event)
        return event

    def _fuse_or_reuse(self, *, command_seq: int) -> tuple[FusionResult, bool]:
        dependency = fusion_dependency_fingerprint(
            self.sorted_instances,
            iou_threshold=self.config.iou_match_threshold,
            retention_mode=self.config.pixel_retention.mode,
            min_coverage=self.config.pixel_retention.min_coverage,
        )
        if (
            self.fusion_result is not None
            and self.fusion_result.fusion_result_version == dependency
        ):
            self._update_progress(command_seq, 1.0, len(self.instances), "fusion_reused")
            return self.fusion_result, True
        self.fusion_result = self._compute_fusion(command_seq=command_seq)
        return self.fusion_result, False

    def execute(
        self,
        command: dict[str, Any],
        *,
        command_seq: int,
        progress_callback: Callable[[float, int], None] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            command_type = command.get("type")
            if command_type == "set_params":
                return self._set_params(command, command_seq, progress_callback)
            if command_type == "replace_metadata":
                return self._replace_metadata(command, command_seq, progress_callback)
            if command_type == "set_formats":
                return self._set_formats(command, command_seq, progress_callback)
            if command_type == "fuse":
                return self._refresh_fusion(command_seq, progress_callback)
            raise PipelineError(
                "E_PARAM_INVALID",
                f"unsupported interaction command: {command_type}",
            )

    def _start_progress(self, command_seq: int, phase: str) -> None:
        self._update_progress(command_seq, 0.0, len(self.instances), phase)

    def _set_params(
        self,
        command: dict[str, Any],
        command_seq: int,
        progress_callback: Callable[[float, int], None] | None,
    ) -> dict[str, Any]:
        self._start_progress(command_seq, "set_params")
        before = self.config
        new_config = self.config.with_params(
            iou_match_threshold=command.get("iou_match_threshold"),
            retention_mode=command.get("retention_mode"),
            min_coverage=command.get("min_coverage"),
        )
        changed_fields: list[str] = []
        if new_config.iou_match_threshold != before.iou_match_threshold:
            changed_fields.append("iou_match_threshold")
        if new_config.pixel_retention.mode != before.pixel_retention.mode:
            changed_fields.append("pixel_retention.mode")
        if new_config.pixel_retention.min_coverage != before.pixel_retention.min_coverage:
            changed_fields.append("pixel_retention.min_coverage")
        self.config = new_config
        reused = [item.key for item in self.sorted_instances]
        trigger = (
            "retention_policy_change"
            if any(field.startswith("pixel_retention") for field in changed_fields)
            else "fusion_threshold_change"
        )
        _, fusion_reused = self._fuse_or_reuse(command_seq=command_seq)
        event = self._event(
            command_seq=command_seq,
            trigger=trigger,
            invalidated=[],
            reused=reused,
            fusion_invalidated=not fusion_reused,
            changed_fields=changed_fields,
            decoded_instances=[],
        )
        self.latest_preview = self._preview(
            command_seq=command_seq,
            event=event,
            fusion_reused=fusion_reused,
        )
        if progress_callback:
            progress_callback(1.0, len(self.instances))
        return self.latest_preview

    def _replace_metadata(
        self,
        command: dict[str, Any],
        command_seq: int,
        progress_callback: Callable[[float, int], None] | None,
    ) -> dict[str, Any]:
        self._start_progress(command_seq, "replace_metadata")
        raw_key = command.get("instance_key", command.get("instance_id"))
        if raw_key is None:
            raise PipelineError("E_PARAM_INVALID", "metadata replacement needs instance_key")
        key = str(raw_key)
        if ":" not in key:
            matches = [
                item.key
                for item in self.sorted_instances
                if item.instance_id == int(key)
            ]
            if len(matches) != 1:
                raise PipelineError(
                    "E_PARAM_INVALID",
                    f"instance_id {key} is ambiguous; use source:instance_id",
                )
            key = matches[0]
        parse_instance_key(key)
        if key not in self.instances:
            raise PipelineError("E_PARAM_INVALID", f"unknown instance key: {key}")
        previous = self.instances[key]
        flat_mask = runs_to_mask(
            mask_to_runs(previous.mask),
            previous.mask.size,
        ).reshape(previous.mask.shape)
        confidence = float(command.get("confidence", previous.confidence))
        crowd_value = command.get("crowd", previous.crowd)
        if type(crowd_value) is not bool:
            raise PipelineError("E_PARAM_INVALID", "replacement crowd must be boolean")
        palette = tuple(command.get("palette", previous.palette))
        if len(palette) != 4:
            raise PipelineError("E_PARAM_INVALID", "replacement palette must be RGBA")
        updated = previous.with_metadata(
            confidence=confidence,
            crowd=crowd_value,
            palette=palette,
            defaulted_fields=(),
        )
        updated.mask = flat_mask
        self.instances[key] = updated
        self.decode_counts[key] = self.decode_counts.get(key, 0) + 1
        self.metadata_document.by_key[key] = {
            "instance_id": previous.instance_id,
            "source_format": previous.source_format,
            "class_id": previous.class_id,
            "confidence": confidence,
            "crowd": crowd_value,
            "palette": list(palette),
        }
        reused = [item.key for item in self.sorted_instances if item.key != key]
        _, fusion_reused = self._fuse_or_reuse(command_seq=command_seq)
        event = self._event(
            command_seq=command_seq,
            trigger="metadata_replacement",
            invalidated=[key],
            reused=reused,
            fusion_invalidated=not fusion_reused,
            changed_fields=["metadata.confidence", "metadata.crowd", "metadata.palette"],
            decoded_instances=[key],
            extra={
                "previous_cache_key": previous.cache_key,
                "new_cache_key": updated.cache_key,
            },
        )
        self.latest_preview = self._preview(
            command_seq=command_seq,
            event=event,
            fusion_reused=fusion_reused,
        )
        if progress_callback:
            progress_callback(1.0, len(self.instances))
        return self.latest_preview

    def _set_formats(
        self,
        command: dict[str, Any],
        command_seq: int,
        progress_callback: Callable[[float, int], None] | None,
    ) -> dict[str, Any]:
        self._start_progress(command_seq, "set_formats")
        raw_formats = command.get("formats", command.get("format"))
        if isinstance(raw_formats, str):
            raw_formats = [raw_formats]
        if not isinstance(raw_formats, list):
            raise PipelineError("E_PARAM_INVALID", "formats must be a list")
        requested = tuple(sorted(set(str(item) for item in raw_formats)))
        if not requested or any(item not in {"v1", "v2"} for item in requested):
            raise PipelineError("E_PARAM_INVALID", "formats must contain v1 and/or v2")
        previous_formats = self.formats
        added = tuple(item for item in requested if item not in previous_formats)
        removed = tuple(item for item in previous_formats if item not in requested)
        invalidated: list[str] = []
        decoded_instances: list[str] = []
        for source_format in removed:
            keys = [
                key
                for key, item in self.instances.items()
                if item.source_format == source_format
            ]
            invalidated.extend(keys)
            for key in keys:
                del self.instances[key]
            self._format_bundles.pop(source_format, None)
            self._format_failures.pop(source_format, None)
        self.formats = requested
        parser = {
            "v1": lambda: parse_v1_package(
                self.layout.v1_path,
                config=self.config,
                metadata=self.metadata_document,
            ),
            "v2": lambda: parse_v2_package(
                self.layout.v2_path,
                config=self.config,
            ),
        }
        for source_format in added:
            bundle = parser[source_format]()
            self._format_bundles[source_format] = bundle
            self._format_failures[source_format] = bundle.sorted_failures()
            for item in bundle.instances:
                if item.key in self.instances:
                    continue
                self.instances[item.key] = item
                self.decode_counts[item.key] = self.decode_counts.get(item.key, 0) + 1
                invalidated.append(item.key)
                decoded_instances.append(item.key)
        reused = [
            item.key
            for item in self.sorted_instances
            if item.key not in set(invalidated)
        ]
        _, fusion_reused = self._fuse_or_reuse(command_seq=command_seq)
        event = self._event(
            command_seq=command_seq,
            trigger="format_version_switch",
            invalidated=invalidated,
            reused=reused,
            fusion_invalidated=not fusion_reused,
            changed_fields=["formats"],
            decoded_instances=decoded_instances,
            extra={
                "previous_formats": list(previous_formats),
                "current_formats": list(self.formats),
                "added_formats": list(added),
                "removed_formats": list(removed),
            },
        )
        self.latest_preview = self._preview(
            command_seq=command_seq,
            event=event,
            fusion_reused=fusion_reused,
        )
        if progress_callback:
            progress_callback(1.0, len(self.instances))
        return self.latest_preview

    def _refresh_fusion(
        self,
        command_seq: int,
        progress_callback: Callable[[float, int], None] | None,
    ) -> dict[str, Any]:
        self._start_progress(command_seq, "fuse")
        reused = [item.key for item in self.sorted_instances]
        _, fusion_reused = self._fuse_or_reuse(command_seq=command_seq)
        event = self._event(
            command_seq=command_seq,
            trigger="explicit_fuse",
            invalidated=[],
            reused=reused,
            fusion_invalidated=not fusion_reused,
            changed_fields=[],
            decoded_instances=[],
        )
        self.latest_preview = self._preview(
            command_seq=command_seq,
            event=event,
            fusion_reused=fusion_reused,
        )
        if progress_callback:
            progress_callback(1.0, len(self.instances))
        return self.latest_preview

    def report_state(self) -> dict[str, Any]:
        assert self.fusion_result is not None
        return {
            "width": self.width,
            "height": self.height,
            "config": self.config.to_dict(),
            "config_hash": self.config.config_hash,
            "formats": list(self.formats),
            "instance_summaries": [item.summary() for item in self.sorted_instances],
            "instance_result_versions": {
                item.key: item.cache_key for item in self.sorted_instances
            },
            "decode_counts": dict(
                sorted(
                    self.decode_counts.items(),
                    key=lambda item: instance_key_numeric_sort_key(item[0]),
                )
            ),
            "fusion": self.fusion_result.summary(),
            "class_distribution": self.fusion_result.class_distribution,
            "matches": self.fusion_result.matches,
            "boundary_diagnostics": self.fusion_result.boundary_diagnostics,
            "failure_codes": sorted({failure.code for failure in self.failures})
            + self.fusion_result.failure_codes,
            "failure_list": [failure.to_dict() for failure in self.failures],
            "latest_preview": self.latest_preview,
            "events": self.events,
        }


def build_session(
    sample_dir: str | Path,
    *,
    config: PipelineConfig | None = None,
    formats: tuple[str, ...] = ("v1", "v2"),
) -> PipelineSession:
    layout = SampleLayout.discover(sample_dir)
    active_config = config or PipelineConfig.load(layout.config_path)
    return PipelineSession(layout, active_config, formats=formats)


def measure_build_and_fuse(
    sample_dir: str | Path,
    *,
    config: PipelineConfig,
    max_workers: int,
    formats: tuple[str, ...] = ("v1", "v2"),
) -> tuple[PipelineSession, float]:
    started = time.perf_counter()
    session = build_session(sample_dir, config=config, formats=formats)
    with_workers = config
    if max_workers != config.max_workers:
        from dataclasses import replace

        with_workers = replace(config, max_workers=max_workers)
    if max_workers == 1:
        session.config = with_workers
        session.fusion_result = fuse_instances(
            session.sorted_instances,
            config=with_workers,
            max_workers=1,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return session, elapsed_ms
