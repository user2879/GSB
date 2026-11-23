"""Command-driven workbench state machine with incremental computation."""

from __future__ import annotations

import copy
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    ENGINE_SCHEMA_VERSION,
    config_hash,
    load_labels,
    semantic_config,
    validate_config,
)
from .errors import (
    E_COMMAND_INVALID,
    E_FIRST_RESPONSE_PENDING,
    E_SNAPSHOT_STALE,
    WorkbenchError,
)
from .pipeline import ProgressTracker, recompute_pipeline
from .segment import build_segment_defs, detect_speech_segments, segment_id_for
from .snapshot import SnapshotManager, command_prefix_hash, snapshot_state_payload
from .util import (
    canonical_json,
    normalize_relpath,
    sha256_bytes,
    sha256_obj,
    stable_float,
)
from .wavio import AudioData, read_wav, scan_wav_directory


def _segment_flat(segments_by_file: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_name in sorted(segments_by_file, key=lambda value: (value.casefold(), value)):
        for segment in sorted(
            segments_by_file[file_name], key=lambda row: (int(row["start_sample"]), int(row["end_sample"]))
        ):
            rows.append({**segment, "file": file_name})
    return rows


def _all_segment_ids(segments_by_file: dict[str, list[dict[str, Any]]]) -> list[str]:
    return sorted(
        segment_id_for(row["file"], int(row["start_sample"]), int(row["end_sample"]))
        for row in _segment_flat(segments_by_file)
    )


def _sorted_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        failures,
        key=lambda row: (
            str(row.get("sort_key", "")),
            str(row.get("error_code", "")),
            str(row.get("message", "")),
        ),
    )


def semantic_projection(report: dict[str, Any]) -> dict[str, Any]:
    """Fields that must be byte-for-byte identical for incremental and full runs.

    Execution-only cache events are deliberately kept in diagnostics. The canonical
    dependency manifest and current cache manifest remain part of this projection.
    """
    summary = report.get("summary", {})
    canonical_summary = {
        key: summary.get(key)
        for key in (
            "completed_segments",
            "speaker_count",
            "vector_dimension_count",
            "affected_segment_ids",
            "valid_audio_file_count",
            "failed_audio_file_count",
            "conflict_count",
            "failure_count",
        )
    }
    return {
        "schema_version": report.get("schema_version"),
        "base_config_hash": report.get("base_config_hash"),
        "effective_config_hash": report.get("effective_config_hash"),
        "labels_hash": report.get("labels_hash"),
        "command_cursor": report.get("command_cursor"),
        "quantization": report.get("quantization"),
        "audio_files": report.get("audio_files"),
        "segments": report.get("segments"),
        "speakers": report.get("speakers"),
        "conflicts": report.get("conflicts"),
        "failures": report.get("failures"),
        "dependencies": report.get("dependencies"),
        "cache_manifest": report.get("cache_manifest"),
        "invalidation_reuse_manifest": report.get("invalidation_reuse_manifest"),
        "summary": canonical_summary,
    }


def canonical_digest(report: dict[str, Any]) -> str:
    return sha256_obj(semantic_projection(report))


class Workbench:
    """A deterministic command-stream engine over local audio assets."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        labels: dict[str, Any],
        assets: dict[str, AudioData],
        audio_failures: list[dict[str, Any]] | None = None,
        segments_by_file: dict[str, list[dict[str, Any]]] | None = None,
        caches: dict[str, dict[str, Any]] | None = None,
        tier_overrides: dict[str, int | None] | None = None,
        global_version: int = 0,
        command_cursor: int = 0,
        failures: list[dict[str, Any]] | None = None,
        last_result: dict[str, Any] | None = None,
        snapshot_manager: SnapshotManager | None = None,
        snapshot_warnings: list[dict[str, Any]] | None = None,
        command_base_dir: Path | None = None,
    ) -> None:
        self.base_config = copy.deepcopy(config)
        self.config = copy.deepcopy(config)
        self.labels = copy.deepcopy(labels)
        self.assets = assets
        self.audio_failures = list(audio_failures or [])
        self.failures = list(failures if failures is not None else self.audio_failures)
        self.tier_overrides = {
            str(key): (None if value is None else int(value)) for key, value in (tier_overrides or {}).items()
        }
        self.snapshot_manager = snapshot_manager
        self.snapshot_warnings = list(snapshot_warnings or [])
        self.command_base_dir = command_base_dir or Path.cwd()
        self.commands: list[dict[str, Any]] = []
        self.global_version = int(global_version)
        self.command_cursor = int(command_cursor)
        self.caches = caches or {"feature": {}, "description": {}, "assignment": {}, "aggregation": {}}
        self.last_result = last_result
        self.last_snapshot: dict[str, Any] | None = None
        self.progress = ProgressTracker()
        if segments_by_file is None:
            self.segments_by_file = self._segment_all_assets()
        else:
            self.segments_by_file = self._normalize_segments(segments_by_file)

    @property
    def base_config_hash(self) -> str:
        return config_hash(self.base_config)

    @property
    def labels_hash(self) -> str:
        return sha256_obj(self.labels)

    @property
    def effective_config_hash(self) -> str:
        return sha256_obj(
            {
                "config": semantic_config(self.config),
                "tier_overrides": dict(sorted(self.tier_overrides.items())),
            }
        )

    @classmethod
    def from_directory(
        cls,
        input_dir: Path,
        *,
        config: dict[str, Any],
        labels: dict[str, Any],
        snapshot_manager: SnapshotManager | None = None,
        command_base_dir: Path | None = None,
    ) -> "Workbench":
        required_sample_rate = int(config["audio"]["sample_rate"])
        assets_list, audio_failures = scan_wav_directory(input_dir, required_sample_rate=required_sample_rate)
        assets = {asset.logical_name: asset for asset in assets_list}
        return cls(
            config=config,
            labels=labels,
            assets=assets,
            audio_failures=audio_failures,
            snapshot_manager=snapshot_manager,
            command_base_dir=command_base_dir or input_dir,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, Any],
        *,
        config: dict[str, Any],
        labels: dict[str, Any],
        snapshot_manager: SnapshotManager | None,
        command_base_dir: Path | None,
        snapshot_warnings: list[dict[str, Any]] | None = None,
    ) -> "Workbench":
        effective = copy.deepcopy(snapshot["effective_config"])
        effective["runtime"] = copy.deepcopy(config.get("runtime", {}))
        effective["snapshot"] = copy.deepcopy(config.get("snapshot", {}))
        validate_config(effective)
        instance = cls(
            config=effective,
            labels=labels,
            assets=snapshot["assets"],
            audio_failures=[row for row in snapshot.get("failures", []) if row.get("stage") == "audio_read"],
            segments_by_file=snapshot["segments_by_file"],
            caches=snapshot.get("caches", {}),
            tier_overrides=snapshot.get("tier_overrides", {}),
            global_version=snapshot.get("global_version", 0),
            command_cursor=snapshot.get("command_cursor", 0),
            failures=snapshot.get("failures", []),
            last_result=snapshot.get("last_result"),
            snapshot_manager=snapshot_manager,
            snapshot_warnings=snapshot_warnings,
            command_base_dir=command_base_dir,
        )
        snapshot_dir = Path(snapshot.get("_snapshot_dir", ""))
        manifest_path = snapshot_dir / "snapshot.json"
        snapshot_info = {
            "snapshot_id": snapshot_dir.name,
            "path": str(snapshot_dir),
            "global_version": int(snapshot.get("global_version", 0)),
            "command_cursor": int(snapshot.get("command_cursor", 0)),
            "manifest_hash": sha256_bytes(manifest_path.read_bytes()) if manifest_path.exists() else "",
            "atomic_commit": True,
            "restored": True,
        }
        instance.last_snapshot = snapshot_info
        if instance.last_result is not None:
            instance.last_result.setdefault("diagnostics", {})["snapshot"] = snapshot_info
        return instance

    def _segment_all_assets(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for file_name in sorted(self.assets, key=lambda value: (value.casefold(), value)):
            asset = self.assets[file_name]
            intervals = detect_speech_segments(
                asset.samples, asset.sample_rate, self.config["segmentation"]
            )
            result[file_name] = build_segment_defs(file_name, intervals)
        return result

    def _normalize_segments(self, segments_by_file: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
        normalized: dict[str, list[dict[str, Any]]] = {}
        for file_name in sorted(segments_by_file, key=lambda value: (value.casefold(), value)):
            rows = sorted(
                segments_by_file[file_name],
                key=lambda row: (int(row["start_sample"]), int(row["end_sample"])),
            )
            normalized[file_name] = [
                {
                    "file": file_name,
                    "start_sample": int(row["start_sample"]),
                    "end_sample": int(row["end_sample"]),
                    "ordinal_in_file": index,
                    "segment_id": segment_id_for(file_name, int(row["start_sample"]), int(row["end_sample"])),
                }
                for index, row in enumerate(rows)
            ]
        return normalized

    def _set_file_segments(self, file_name: str, intervals: list[tuple[int, int]]) -> None:
        self.segments_by_file[file_name] = build_segment_defs(file_name, intervals)

    def _segments_for_file(self, file_name: str) -> list[dict[str, Any]]:
        file_name = normalize_relpath(file_name)
        if file_name not in self.assets:
            raise WorkbenchError(E_COMMAND_INVALID, f"unknown audio file: {file_name}", {"file": file_name})
        if file_name not in self.segments_by_file:
            self.segments_by_file[file_name] = build_segment_defs(file_name, [])
        return self.segments_by_file[file_name]

    def _resolve_segment(self, command: dict[str, Any]) -> dict[str, Any]:
        file_name = normalize_relpath(str(command.get("file", "")))
        rows = self._segments_for_file(file_name)
        if command.get("segment_id") is not None:
            segment_id = str(command["segment_id"])
            for row in rows:
                if row["segment_id"] == segment_id:
                    return row
            raise WorkbenchError(E_COMMAND_INVALID, f"unknown segment_id: {segment_id}", {"file": file_name})
        ordinal = command.get("ordinal", command.get("segment_index"))
        if ordinal is None:
            raise WorkbenchError(E_COMMAND_INVALID, "segment_id or ordinal is required", {"file": file_name})
        ordinal = int(ordinal)
        if ordinal < 0 or ordinal >= len(rows):
            raise WorkbenchError(
                E_COMMAND_INVALID,
                f"segment ordinal out of range: {ordinal}",
                {"file": file_name, "segment_count": len(rows)},
            )
        return rows[ordinal]

    def _resolve_source(self, source: str) -> Path:
        path = Path(source)
        if not path.is_absolute():
            path = self.command_base_dir / path
        return path

    def _compute_asset_sha(self, samples: np.ndarray) -> str:
        return sha256_bytes(np.asarray(samples, dtype="<i2").tobytes(order="C"))

    def apply_command(self, command: dict[str, Any], *, command_index: int) -> dict[str, Any]:
        op = str(command.get("op", ""))
        effects: dict[str, Any] = {
            "op": op,
            "reason": op or "invalid_command",
            "affected_segment_ids": [],
            "error_codes": [],
            "scope": "none",
        }
        try:
            if op == "split_segment":
                row = self._resolve_segment(command)
                file_name = row["file"]
                at_sample = int(command["at_sample"])
                if not int(row["start_sample"]) < at_sample < int(row["end_sample"]):
                    raise WorkbenchError(E_COMMAND_INVALID, "split point must lie inside the segment")
                old_id = row["segment_id"]
                intervals = [
                    (int(item["start_sample"]), int(item["end_sample"]))
                    for item in self._segments_for_file(file_name)
                ]
                ordinal = int(row["ordinal_in_file"])
                intervals[ordinal : ordinal + 1] = [
                    (int(row["start_sample"]), at_sample),
                    (at_sample, int(row["end_sample"])),
                ]
                self._set_file_segments(file_name, intervals)
                new_ids = [
                    segment_id_for(file_name, int(row["start_sample"]), at_sample),
                    segment_id_for(file_name, at_sample, int(row["end_sample"])),
                ]
                effects.update(
                    {
                        "affected_segment_ids": sorted({old_id, *new_ids}),
                        "scope": "segment_split",
                        "split_at_sample": at_sample,
                    }
                )
            elif op == "merge_segments":
                file_name = normalize_relpath(str(command.get("file", "")))
                rows = self._segments_for_file(file_name)
                specifications = command.get("segment_ids")
                if specifications is None:
                    ordinals = command.get("ordinals")
                    if ordinals is None:
                        ordinals = command.get("segment_indices", [])
                    indices = [int(value) for value in ordinals]
                else:
                    ids = {str(value) for value in specifications}
                    indices = [index for index, row in enumerate(rows) if row["segment_id"] in ids]
                    if len(indices) != len(ids):
                        raise WorkbenchError(E_COMMAND_INVALID, "one or more merge segment_ids are missing")
                indices = sorted(indices)
                if len(indices) < 2 or indices != list(range(indices[0], indices[0] + len(indices))):
                    raise WorkbenchError(E_COMMAND_INVALID, "merge requires at least two adjacent segments")
                old_ids = [rows[index]["segment_id"] for index in indices]
                merged_start = int(rows[indices[0]]["start_sample"])
                merged_end = int(rows[indices[-1]]["end_sample"])
                intervals = [
                    (int(item["start_sample"]), int(item["end_sample"])) for item in rows
                ]
                intervals[indices[0] : indices[-1] + 1] = [(merged_start, merged_end)]
                self._set_file_segments(file_name, intervals)
                new_id = segment_id_for(file_name, merged_start, merged_end)
                effects.update(
                    {
                        "affected_segment_ids": sorted({*old_ids, new_id}),
                        "scope": "segment_merge",
                    }
                )
            elif op == "set_segment_boundary":
                row = self._resolve_segment(command)
                file_name = row["file"]
                asset = self.assets[file_name]
                edge = str(command.get("edge", ""))
                sample = int(command["sample"])
                rows = self._segments_for_file(file_name)
                index = int(row["ordinal_in_file"])
                intervals = [(int(item["start_sample"]), int(item["end_sample"])) for item in rows]
                old_ids = sorted(
                    {
                        rows[index]["segment_id"],
                        rows[index - 1]["segment_id"] if index > 0 else rows[index]["segment_id"],
                        rows[index + 1]["segment_id"] if index + 1 < len(rows) else rows[index]["segment_id"],
                    }
                )
                if edge == "end":
                    if not int(row["start_sample"]) < sample <= asset.sample_count:
                        raise WorkbenchError(E_COMMAND_INVALID, "invalid end boundary sample")
                    if index + 1 < len(rows) and sample >= int(rows[index + 1]["end_sample"]):
                        raise WorkbenchError(E_COMMAND_INVALID, "end boundary would cross the next segment")
                    intervals[index] = (int(row["start_sample"]), sample)
                    if index + 1 < len(rows):
                        intervals[index + 1] = (sample, int(rows[index + 1]["end_sample"]))
                elif edge == "start":
                    if not 0 <= sample < int(row["end_sample"]):
                        raise WorkbenchError(E_COMMAND_INVALID, "invalid start boundary sample")
                    if index > 0 and sample <= int(rows[index - 1]["start_sample"]):
                        raise WorkbenchError(E_COMMAND_INVALID, "start boundary would cross the previous segment")
                    intervals[index] = (sample, int(row["end_sample"]))
                    if index > 0:
                        intervals[index - 1] = (int(rows[index - 1]["start_sample"]), sample)
                else:
                    raise WorkbenchError(E_COMMAND_INVALID, "edge must be 'start' or 'end'")
                self._set_file_segments(file_name, intervals)
                new_ids = _all_segment_ids({file_name: self.segments_by_file[file_name]})
                effects.update(
                    {
                        "affected_segment_ids": sorted(set(old_ids) | set(new_ids)),
                        "scope": "boundary_edit",
                    }
                )
            elif op == "replace_segment_audio":
                row = self._resolve_segment(command)
                file_name = row["file"]
                source = self._resolve_source(str(command["source"]))
                source_asset = read_wav(source, logical_name="replacement.wav", required_sample_rate=16000)
                expected = int(row["end_sample"]) - int(row["start_sample"])
                if source_asset.sample_count != expected:
                    raise WorkbenchError(
                        E_COMMAND_INVALID,
                        "replacement audio length must equal the target segment length",
                        {"expected_samples": expected, "actual_samples": source_asset.sample_count},
                    )
                asset = self.assets[file_name]
                samples = asset.samples.copy()
                samples[int(row["start_sample"]) : int(row["end_sample"])] = source_asset.samples
                asset.samples = samples
                asset.sha256 = self._compute_asset_sha(samples)
                effects.update(
                    {
                        "affected_segment_ids": [row["segment_id"]],
                        "scope": "segment_audio_replace",
                    }
                )
            elif op == "replace_audio":
                file_name = normalize_relpath(str(command.get("file", "")))
                self._segments_for_file(file_name)
                old_ids = [row["segment_id"] for row in self.segments_by_file[file_name]]
                source = self._resolve_source(str(command["source"]))
                replacement = read_wav(source, logical_name=file_name, required_sample_rate=16000)
                self.assets[file_name] = replacement
                intervals = detect_speech_segments(
                    replacement.samples, replacement.sample_rate, self.config["segmentation"]
                )
                self._set_file_segments(file_name, [(row["start_sample"], row["end_sample"]) for row in intervals])
                new_ids = [row["segment_id"] for row in self.segments_by_file[file_name]]
                effects.update(
                    {
                        "affected_segment_ids": sorted(set(old_ids) | set(new_ids)),
                        "scope": "audio_replace",
                    }
                )
            elif op == "set_quantization_tier":
                dimension = str(command.get("dimension", ""))
                tier_value = command.get("tier", "auto")
                tier = None if tier_value in (None, "auto") else int(tier_value)
                dimensions = self.config["quantization"]["dimensions"]
                targets = dimensions if dimension == "*" else [row for row in dimensions if row["name"] == dimension]
                if not targets:
                    raise WorkbenchError(E_COMMAND_INVALID, f"unknown quantization dimension: {dimension}")
                for target in targets:
                    if tier is not None and tier not in [int(value) for value in target["candidate_tiers"]]:
                        raise WorkbenchError(
                            E_COMMAND_INVALID,
                            "tier is not a configured candidate for the dimension",
                            {"dimension": target["name"], "tier": tier},
                        )
                for target in targets:
                    self.tier_overrides[str(target["name"])] = tier
                effects.update(
                    {
                        "affected_segment_ids": _all_segment_ids(self.segments_by_file),
                        "scope": "quantization_global",
                        "dimension": dimension,
                        "tier": tier,
                    }
                )
            elif op in {"set_similarity_threshold", "set_threshold"}:
                threshold = float(command["value"])
                if not 0.0 <= threshold <= 1.0:
                    raise WorkbenchError(E_COMMAND_INVALID, "similarity threshold must be in [0, 1]")
                self.config["assignment"]["similarity_threshold"] = stable_float(threshold)
                effects.update(
                    {
                        "affected_segment_ids": _all_segment_ids(self.segments_by_file),
                        "scope": "assignment_global",
                        "threshold": stable_float(threshold),
                    }
                )
            elif op == "set_assignment_metric":
                metric = str(command["metric"])
                if metric not in {"rbf", "cosine", "normalized_l1"}:
                    raise WorkbenchError(E_COMMAND_INVALID, f"unsupported metric: {metric}")
                gamma = None
                if "rbf_gamma" in command:
                    gamma = float(command["rbf_gamma"])
                    if gamma <= 0:
                        raise WorkbenchError(E_COMMAND_INVALID, "rbf_gamma must be positive")
                self.config["assignment"]["metric"] = metric
                if gamma is not None:
                    self.config["assignment"]["rbf_gamma"] = gamma
                effects.update(
                    {
                        "affected_segment_ids": _all_segment_ids(self.segments_by_file),
                        "scope": "assignment_global",
                        "metric": metric,
                    }
                )
            elif op == "noop":
                effects.update({"scope": "noop", "reason": "noop"})
            else:
                raise WorkbenchError(E_COMMAND_INVALID, f"unsupported command op: {op}")
        except WorkbenchError as exc:
            failure = exc.as_failure(
                file=str(command.get("file", "")),
                stage=f"command:{op}",
                command_index=command_index,
                sort_key=f"__commands__|{command_index:08d}|{E_COMMAND_INVALID}",
            )
            self.failures.append(failure)
            effects["error_codes"].append(exc.code)
            effects["error_message"] = exc.message
        except (KeyError, TypeError, ValueError) as exc:
            wrapped = WorkbenchError(E_COMMAND_INVALID, str(exc))
            failure = wrapped.as_failure(
                file=str(command.get("file", "")),
                stage=f"command:{op}",
                command_index=command_index,
                sort_key=f"__commands__|{command_index:08d}|{E_COMMAND_INVALID}",
            )
            self.failures.append(failure)
            effects["error_codes"].append(E_COMMAND_INVALID)
            effects["error_message"] = str(exc)
        return effects

    def _make_pending_summary(self, effects: dict[str, Any]) -> dict[str, Any]:
        latest = (self.last_result or {}).get("summary", {})
        pending_progress = self.progress.snapshot(complete=False)
        return {
            "completed_segments": latest.get("completed_segments", 0),
            "speaker_count": latest.get("speaker_count", 0),
            "vector_dimension_count": latest.get("vector_dimension_count", 0),
            "affected_segment_ids": sorted(set(effects.get("affected_segment_ids", []))),
            "affected_segment_count": len(set(effects.get("affected_segment_ids", []))),
            "progress": pending_progress,
            "latest_completed_global_version": self.global_version - 1 if self.global_version else 0,
            "latest_completed_segment_count": latest.get("completed_segments", 0),
        }

    def _stable_state_snapshot(self, report: dict[str, Any]) -> dict[str, Any]:
        return {
            "canonical_digest": report.get("canonical_digest"),
            "speaker_vectors": [
                {
                    "speaker_id": speaker["speaker_id"],
                    "vector": speaker["vector"],
                    "segment_count": speaker["segment_count"],
                    "aggregation_degenerate": speaker["aggregation_degenerate"],
                }
                for speaker in report.get("speakers", [])
            ],
            "quantization_tiers": [
                {
                    "dimension": dimension["name"],
                    "bits": dimension["bits"],
                    "levels": dimension["levels"],
                    "step": dimension["step"],
                    "mode": dimension["tier_mode"],
                }
                for dimension in report.get("quantization", {}).get("dimensions", [])
            ],
            "tier_overrides": dict(sorted(self.tier_overrides.items())),
            "similarity_threshold": self.config["assignment"]["similarity_threshold"],
            "completed_segments": report.get("summary", {}).get("completed_segments"),
            "speaker_count": report.get("summary", {}).get("speaker_count"),
        }

    def _assemble_report(self, outcome: dict[str, Any], first_response: dict[str, Any] | None = None) -> dict[str, Any]:
        report = copy.deepcopy(outcome)
        report.pop("_caches", None)
        audio_failure_count = sum(1 for failure in self.failures if failure.get("stage") == "audio_read")
        report["summary"]["failed_audio_file_count"] = audio_failure_count
        report["summary"]["failure_count"] = len(self.failures)
        report.update(
            {
                "schema_version": ENGINE_SCHEMA_VERSION,
                "global_version": self.global_version,
                "command_cursor": self.command_cursor,
                "base_config_hash": self.base_config_hash,
                "effective_config_hash": self.effective_config_hash,
                "labels_hash": self.labels_hash,
                "failures": _sorted_failures(self.failures),
            }
        )
        report["diagnostics"] = {
            "cache_events": report.pop("cache_events"),
            "task_orders": report.pop("task_orders"),
            "threading": report.pop("threading"),
            "first_response": first_response,
            "snapshot": self.last_snapshot,
            "snapshot_warnings": self.snapshot_warnings,
        }
        report["canonical_digest"] = canonical_digest(report)
        return report

    def _commit_snapshot(self, report: dict[str, Any]) -> dict[str, Any] | None:
        if self.snapshot_manager is None:
            return None
        state = snapshot_state_payload(
            global_version=self.global_version,
            command_cursor=self.command_cursor,
            command_prefix=command_prefix_hash(self.commands, self.command_cursor),
            effective_config=self.config,
            effective_config_hash=self.effective_config_hash,
            tier_overrides=self.tier_overrides,
            segments_by_file={
                file_name: [
                    {
                        "file": file_name,
                        "start_sample": int(row["start_sample"]),
                        "end_sample": int(row["end_sample"]),
                    }
                    for row in rows
                ]
                for file_name, rows in self.segments_by_file.items()
            },
            caches=self.caches,
            failures=self.failures,
            last_result=report,
            labels=self.labels,
            product_versions=report.get("product_versions", {}),
            quantization=report.get("quantization", {}),
        )
        keep = int(self.config.get("runtime", {}).get("snapshot_keep", 5))
        return self.snapshot_manager.commit(assets=self.assets, state=state, keep=keep)

    def _finalize_outcome(self, outcome: dict[str, Any], *, commit: bool = True) -> dict[str, Any]:
        self.caches = outcome.get("_caches", self.caches)
        report = self._assemble_report(outcome)
        if commit:
            snapshot_info = self._commit_snapshot(report)
            self.last_snapshot = snapshot_info
            report["diagnostics"]["snapshot"] = snapshot_info
            report["canonical_digest"] = canonical_digest(report)
        self.last_result = report
        return report

    def _start_compute(self, *, force_full: bool, effects: dict[str, Any], reason: str) -> tuple[Future[dict[str, Any]], float]:
        self.progress = ProgressTracker()
        self.progress.begin_phase("scheduled", 0)
        affected = sorted(set(effects.get("affected_segment_ids", [])))
        started = time.perf_counter()

        def job() -> dict[str, Any]:
            return recompute_pipeline(
                assets=self.assets,
                segments_by_file=self.segments_by_file,
                config=self.config,
                labels=self.labels,
                tier_overrides=self.tier_overrides,
                caches=self.caches,
                progress=self.progress,
                force_recompute=force_full,
                affected_segment_ids=affected,
                reason=reason,
            )

        coordinator = ThreadPoolExecutor(max_workers=1, thread_name_prefix="timbre-coordinator")
        future = coordinator.submit(job)
        future.add_done_callback(lambda _future: coordinator.shutdown(wait=False))
        return future, started

    def execute_command(
        self,
        command_index: int,
        *,
        budget_ms: int | None = None,
        wait_stable: bool = True,
        force_full: bool = False,
    ) -> dict[str, Any]:
        if command_index != self.command_cursor:
            raise WorkbenchError(
                E_COMMAND_INVALID,
                "command cursor mismatch",
                {"expected": self.command_cursor, "actual": command_index},
            )
        if command_index >= len(self.commands):
            raise WorkbenchError(E_COMMAND_INVALID, "command index outside command stream")
        command = self.commands[command_index]
        budget = int(self.config["runtime"]["first_response_budget_ms"] if budget_ms is None else budget_ms)
        command_started = time.perf_counter()
        effects = self.apply_command(command, command_index=command_index)
        self.global_version += 1
        self.command_cursor += 1
        future, _compute_started = self._start_compute(
            force_full=force_full,
            effects=effects,
            reason=effects.get("reason", str(command.get("op", ""))),
        )

        budget_seconds = max(0.0, budget / 1000.0)
        elapsed = time.perf_counter() - command_started
        remaining = max(0.0, budget_seconds - elapsed)
        outcome: dict[str, Any] | None = None
        timed_out = budget == 0
        if not timed_out:
            try:
                outcome = future.result(timeout=remaining)
            except TimeoutError:
                timed_out = True

        first_response_latency_ms = round((time.perf_counter() - command_started) * 1000.0, 6)
        scheduling_tolerance = float(
            self.config.get("runtime", {}).get("first_response_scheduling_tolerance_ms", 20.0)
        )
        within_budget = first_response_latency_ms <= budget + scheduling_tolerance
        if timed_out:
            pending_summary = self._make_pending_summary(effects)
            first_response = {
                "error_code": E_FIRST_RESPONSE_PENDING,
                "status": "pending",
                "budget_ms": budget,
                "latency_ms": first_response_latency_ms,
                "within_budget": within_budget,
                "summary": pending_summary,
                "affected_segment_ids": pending_summary["affected_segment_ids"],
                "progress": pending_summary["progress"],
            }
            if wait_stable:
                outcome = future.result()
            else:
                return {
                    "command_index": command_index,
                    "op": command.get("op"),
                    "first_response": first_response,
                    "status": "pending",
                    "stable_after_wait": False,
                    "effect": effects,
                }
            report = self._finalize_outcome(outcome)
            first_response["stable_after_wait"] = True
            first_response["stable_canonical_digest"] = report["canonical_digest"]
            report["diagnostics"]["first_response"] = first_response
            self.last_result = report
            return {
                "command_index": command_index,
                "op": command.get("op"),
                "status": "pending",
                "first_response": first_response,
                "error_codes": [E_FIRST_RESPONSE_PENDING],
                "latency_ms": first_response_latency_ms,
                "within_budget": first_response["within_budget"],
                "stable_after_wait": True,
                "effect": effects,
                "stable_summary": report["summary"],
                "stable_state": self._stable_state_snapshot(report),
                "canonical_digest": report["canonical_digest"],
            }

        assert outcome is not None
        report = self._finalize_outcome(outcome)
        first_response = {
            "status": "complete",
            "budget_ms": budget,
            "latency_ms": first_response_latency_ms,
            "within_budget": within_budget,
            "summary": report["summary"],
            "affected_segment_ids": report["summary"]["affected_segment_ids"],
            "progress": report["summary"]["progress"],
        }
        report["diagnostics"]["first_response"] = first_response
        self.last_result = report
        return {
            "command_index": command_index,
            "op": command.get("op"),
            "status": "complete",
            "first_response": first_response,
            "error_codes": [],
            "latency_ms": first_response_latency_ms,
            "within_budget": first_response["within_budget"],
            "stable_after_wait": True,
            "effect": effects,
            "stable_summary": report["summary"],
            "stable_state": self._stable_state_snapshot(report),
            "canonical_digest": report["canonical_digest"],
        }

    def run_commands(
        self,
        commands: list[dict[str, Any]],
        *,
        command_base_dir: Path | None = None,
        wait_stable: bool = True,
        force_full: bool = False,
    ) -> list[dict[str, Any]]:
        self.commands = copy.deepcopy(commands)
        if command_base_dir is not None:
            self.command_base_dir = command_base_dir
        responses: list[dict[str, Any]] = []
        while self.command_cursor < len(self.commands):
            responses.append(
                self.execute_command(
                    self.command_cursor,
                    wait_stable=wait_stable,
                    force_full=force_full,
                )
            )
        if self.last_result is None:
            self.rebuild_full(commit=self.config.get("runtime", {}).get("snapshot_after_each_command", True))
        return responses

    def rebuild_full(self, *, commit: bool = True) -> dict[str, Any]:
        effects = {
            "affected_segment_ids": _all_segment_ids(self.segments_by_file),
            "reason": "full_rebuild",
        }
        future, _started = self._start_compute(force_full=True, effects=effects, reason="full_rebuild")
        outcome = future.result()
        return self._finalize_outcome(outcome, commit=commit)

    def refresh_report(self) -> dict[str, Any]:
        if self.last_result is None:
            return self.rebuild_full(commit=False)
        return copy.deepcopy(self.last_result)

def build_snapshot_manager(
    work_dir: Path,
    *,
    config: dict[str, Any],
    labels: dict[str, Any],
) -> SnapshotManager:
    return SnapshotManager(
        work_dir,
        base_config_hash=config_hash(config),
        labels_hash=sha256_obj(labels),
        engine_schema_version=ENGINE_SCHEMA_VERSION,
    )


def load_or_initialize_workbench(
    *,
    input_dir: Path,
    config: dict[str, Any],
    labels: dict[str, Any],
    commands: list[dict[str, Any]],
    work_dir: Path,
    resume: bool,
    expected_global_version: int | None = None,
    command_base_dir: Path | None = None,
) -> tuple[Workbench, list[dict[str, Any]]]:
    manager = build_snapshot_manager(work_dir, config=config, labels=labels)
    warnings: list[dict[str, Any]] = []
    if resume:
        snapshot, stale = manager.load(expected_global_version=expected_global_version, commands=commands)
        if snapshot is not None:
            workbench = Workbench.from_snapshot(
                snapshot,
                config=config,
                labels=labels,
                snapshot_manager=manager,
                command_base_dir=command_base_dir or input_dir,
            )
            return workbench, warnings
        if stale is not None:
            warnings.append(stale)
    workbench = Workbench.from_directory(
        input_dir,
        config=config,
        labels=labels,
        snapshot_manager=manager,
        command_base_dir=command_base_dir or input_dir,
    )
    workbench.snapshot_warnings = warnings
    return workbench, warnings


def write_report_files(report: dict[str, Any], json_path: Path, html_path: Path | None = None) -> None:
    from .render import render_html
    from .util import atomic_write_json, atomic_write_text

    atomic_write_json(json_path, report)
    if html_path is not None:
        atomic_write_text(html_path, render_html(report))
