"""End-to-end deterministic engine with stage caches, budgets, and audit output."""

from __future__ import annotations

import copy
import json
import math
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .attribution import attribute_turn, attribution_cache_key
from .calibration import calibrate_turn, calibration_cache_key, effective_gain
from .config import load_config
from .constants import (
    E_AUDIO_FORMAT,
    E_AUDIO_MISSING,
    E_CALIB_BUDGET_PENDING,
    E_CONFIG_MISMATCH,
    E_INJECTED_FAULT,
    E_INTERNAL,
    E_SEGMENT_TOO_SHORT,
    LEDGER_SCHEMA_VERSION,
    PRODUCT_VERSION,
    REPORT_SCHEMA_VERSION,
    STAGE_ARTIFACT_VERSION,
    STAGES,
)
from .dialogue import load_dialogue
from .edits import apply_batch, initial_state, load_edit_stream
from .errors import VoicecalError
from .evidence import build_initial_evidence, initial_cache_key
from .features import ALGORITHM_METADATA, WorkStats, aggregate_audio, extract_audio_frames
from .injections import empty_injections, injections_by_turn_stage, load_injections
from .snapshot import SnapshotStore
from .util import bytes_hash, canonical_json, file_hash, json_hash, read_json, safe_relative_path, write_json
from .wav_io import WavPayload, parse_wav_bytes, read_wav, scan_wav_files

STAGE_ORDER = {stage: index for index, stage in enumerate(STAGES)}


@dataclass
class EnginePaths:
    sample_dir: Path
    dialogue_path: Path
    config_path: Path
    edits_path: Path | None
    injections_path: Path | None
    snapshot_dir: Path
    output_dir: Path


def _failure(
    *,
    file: str,
    turn_id: str | None,
    turn_index: int | None,
    stage: str,
    code: str,
    exception_type: str,
    message: str,
    injection_id: str | None = None,
    details: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "file": file,
        "turn_id": turn_id,
        "turn_index": turn_index,
        "stage": stage,
        "code": code,
        "exception_type": exception_type,
        "message": message,
    }
    if injection_id is not None:
        payload["injection_id"] = injection_id
    if details is not None:
        payload["details"] = details
    return payload


def _injection_fingerprint(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"injection_id": item["injection_id"], "kind": item["kind"], "stage": item["stage"]}
        for item in sorted(items, key=lambda value: value["injection_id"])
    ]


def extract_cache_key(
    relative_path: str,
    audio_hash: str,
    config: dict[str, Any],
    injection_items: list[dict[str, Any]],
) -> str:
    return json_hash(
        {
            "stage": "extract",
            "artifact_version": STAGE_ARTIFACT_VERSION,
            "file": relative_path,
            "audio_sha256": audio_hash,
            "frame": config["frame"],
            "algorithms": ALGORITHM_METADATA,
            "decimal_places": config["fixed_point"]["decimal_places"],
            "injections": _injection_fingerprint(injection_items),
        }
    )


def _stage_failure(
    turn: dict[str, Any],
    stage: str,
    exc: BaseException,
    injection_id: str | None = None,
    code_override: str | None = None,
) -> dict[str, Any]:
    if isinstance(exc, VoicecalError):
        code = code_override or exc.code
        details: Any = exc.details or None
    else:
        code = code_override or E_INTERNAL
        details = {"traceback_tail": traceback.format_exception_only(type(exc), exc)[-1].strip()}
    return _failure(
        file=turn["audio"],
        turn_id=turn["turn_id"],
        turn_index=turn["turn_index"],
        stage=stage,
        code=code,
        exception_type=type(exc).__name__,
        message=str(exc),
        injection_id=injection_id,
        details=details,
    )


def _injection_hit(injection: dict[str, Any], turn: dict[str, Any], batch_id: str, code: str) -> dict[str, Any]:
    return {
        "injection_id": injection["injection_id"],
        "turn_id": turn["turn_id"],
        "turn_index": turn["turn_index"],
        "stage": injection["stage"],
        "kind": injection["kind"],
        "batch_id": batch_id,
        "outcome": "failed",
        "business_code": code,
    }


class Engine:
    def __init__(self, paths: EnginePaths) -> None:
        self.paths = paths
        self.config = load_config(paths.config_path)
        self.base_config_hash = self.config["config_hash"]
        self.dialogue = load_dialogue(paths.dialogue_path)
        self.edit_stream = load_edit_stream(paths.edits_path)
        self.injections = load_injections(paths.injections_path)
        self.snapshot = SnapshotStore(paths.snapshot_dir)
        self.work = WorkStats()
        self.audit_events: list[dict[str, Any]] = []
        self.injection_hits: list[dict[str, Any]] = []
        self._state: dict[str, Any] | None = None
        self._target_batch_id = "base"
        self._asset_cache: dict[str, dict[str, Any]] = {}
        self._scan_failures: list[dict[str, Any]] = []
        self._turn_keys: dict[str, dict[str, str]] = {}
        self._prefetched_extract_keys: set[str] = set()
        self._prefetched_initial_keys: set[str] = set()

    def effective_config_hash(self, state: dict[str, Any]) -> str:
        return json_hash(
            {
                "base_config_hash": self.base_config_hash,
                "gain_override": state.get("config_overrides", {}).get("gain"),
                "turn_gain_overrides": state.get("config_overrides", {}).get("turn_gain", {}),
            }
        )

    def _empty_state(self) -> dict[str, Any]:
        return {
            "base_config_hash": self.base_config_hash,
            "effective_state": initial_state(self.dialogue),
            "next_batch_index": 0,
            "target_state": None,
            "target_batch_index": None,
            "stage_pools": {stage: {} for stage in STAGES},
            "turn_keys": {},
            "injection_hits": [],
            "scan_failures": [],
            "completed": False,
        }

    def _load_or_new_state(self, fresh: bool) -> tuple[dict[str, Any], bool]:
        if not fresh and self.snapshot.has_snapshot():
            manifest = self.snapshot.validate_manifest(self.base_config_hash)
            state = self.snapshot.load_state(manifest)
            if not isinstance(state.get("stage_pools"), dict):
                raise VoicecalError(E_INTERNAL, "snapshot state has invalid stage pools")
            return state, True
        return self._empty_state(), False

    def _scan_audio_assets(self) -> None:
        self._asset_cache = {}
        self._scan_failures = []
        all_wavs = scan_wav_files(self.paths.sample_dir)
        for path in all_wavs:
            relative = path.relative_to(self.paths.sample_dir).as_posix()
            try:
                raw = path.read_bytes()
                audio_hash = bytes_hash(raw)
                try:
                    payload = parse_wav_bytes(raw, path)
                    self._asset_cache[relative] = {"raw": raw, "sha256": audio_hash, "payload": payload, "error": None}
                except VoicecalError as exc:
                    self._asset_cache[relative] = {"raw": raw, "sha256": audio_hash, "payload": None, "error": exc}
                    self._scan_failures.append(
                        _failure(
                            file=relative,
                            turn_id=None,
                            turn_index=None,
                            stage="extract",
                            code=exc.code,
                            exception_type=type(exc).__name__,
                            message=str(exc),
                            details=exc.details or None,
                        )
                    )
            except OSError as exc:
                self._asset_cache[relative] = {"raw": None, "sha256": None, "payload": None, "error": exc}

    def _asset_for_turn(self, turn: dict[str, Any]) -> tuple[Path, str, bytes | None, str | None, Exception | None]:
        try:
            path = safe_relative_path(self.paths.sample_dir, turn["audio"])
            relative = path.relative_to(self.paths.sample_dir.resolve()).as_posix()
        except (ValueError, OSError) as exc:
            return self.paths.sample_dir / turn["audio"], turn["audio"], None, None, exc
        asset = self._asset_cache.get(relative)
        if asset is None:
            if not path.exists():
                return path, relative, None, None, VoicecalError(E_AUDIO_MISSING, "audio file is missing", file=relative)
            try:
                raw = path.read_bytes()
            except OSError as exc:
                return path, relative, None, None, exc
            return path, relative, raw, bytes_hash(raw), None
        return path, relative, asset["raw"], asset["sha256"], asset["error"]

    def _injections_for(self, turn_id: str, stage: str) -> list[dict[str, Any]]:
        return injections_by_turn_stage(self.injections, turn_id, stage)

    def _record_audit(self, turn: dict[str, Any], stage: str, status: str, cache_key: str | None, detail: str | None = None) -> None:
        event = {
            "turn_index": turn["turn_index"],
            "turn_id": turn["turn_id"],
            "stage": stage,
            "status": status,
            "cache_key": cache_key,
        }
        if detail is not None:
            event["detail"] = detail
        self.audit_events.append(event)

    def _compute_extract(
        self,
        turn: dict[str, Any],
        key: str,
        relative_path: str,
        audio_hash: str,
        asset_error: Exception | None,
        raw: bytes | None,
        injection_items: list[dict[str, Any]],
        frame_workers: int | None = None,
    ) -> dict[str, Any]:
        if injection_items:
            injection = injection_items[0]
            if injection["kind"] == "corrupt_format":
                assert raw is not None
                corrupted = raw[:-1]
                try:
                    parse_wav_bytes(corrupted, self.paths.sample_dir / relative_path)
                    parse_exc: Exception = VoicecalError(E_AUDIO_FORMAT, "injected corruption was not detected")
                except VoicecalError as exc:
                    parse_exc = exc
                hit = _injection_hit(injection, turn, self._target_batch_id, E_AUDIO_FORMAT)
                self.injection_hits.append(hit)
                failure = _stage_failure(turn, "extract", parse_exc, injection_id=injection["injection_id"], code_override=E_AUDIO_FORMAT)
                return {
                    "cache_key": key,
                    "status": "failed",
                    "file": relative_path,
                    "audio_sha256": audio_hash,
                    "failure": failure,
                    "injection_hits": [hit],
                }
            hit = _injection_hit(injection, turn, self._target_batch_id, E_INJECTED_FAULT)
            self.injection_hits.append(hit)
            exc = VoicecalError(E_INJECTED_FAULT, injection["message"], injection_id=injection["injection_id"], stage="extract")
            failure = _stage_failure(turn, "extract", exc, injection_id=injection["injection_id"])
            return {"cache_key": key, "status": "failed", "file": relative_path, "audio_sha256": audio_hash, "failure": failure, "injection_hits": [hit]}
        parse_error: Exception | None = asset_error
        payload: WavPayload | None = None
        if parse_error is None and raw is not None:
            try:
                payload = parse_wav_bytes(raw, self.paths.sample_dir / relative_path)
            except VoicecalError as exc:
                parse_error = exc
        if parse_error is None and payload is None:
            parse_error = VoicecalError(E_AUDIO_MISSING, "audio bytes are unavailable", file=relative_path)
        if parse_error is not None:
            failure = _stage_failure(turn, "extract", parse_error)
            return {"cache_key": key, "status": "failed", "file": relative_path, "audio_sha256": audio_hash, "failure": failure, "injection_hits": []}
        assert payload is not None
        try:
            frames = extract_audio_frames(
                payload.samples,
                relative_path,
                self.config["frame"],
                int(self.config["concurrency"]) if frame_workers is None else frame_workers,
                self.work,
            )
            aggregate = aggregate_audio(frames, int(payload.samples.size), relative_path, self.config["frame"], self.work)
        except Exception as exc:
            return {
                "cache_key": key,
                "status": "failed",
                "file": relative_path,
                "audio_sha256": audio_hash,
                "failure": _stage_failure(turn, "extract", exc),
                "injection_hits": [],
            }
        return {
            "cache_key": key,
            "status": "ok",
            "file": relative_path,
            "audio_sha256": audio_hash,
            "sample_count": int(payload.samples.size),
            "duration_ms": round(payload.samples.size / 16.0, 6),
            "frame_count": len(frames),
            "padded_frame_count": aggregate["padded_frame_count"],
            "algorithm_metadata": dict(ALGORITHM_METADATA),
            "frame_parameters": self.config["frame"],
            "aggregate_features": aggregate,
            "initial_task_sort_key": f"{relative_path}|segment|initial",
            "frames": frames,
            "injection_hits": [],
        }

    def _compute_initial(self, turn: dict[str, Any], extract_record: dict[str, Any]) -> dict[str, Any]:
        injection_items = self._injections_for(turn["turn_id"], "initial")
        key = initial_cache_key(extract_record["cache_key"], self.config, _injection_fingerprint(injection_items))
        if injection_items:
            injection = injection_items[0]
            hit = _injection_hit(injection, turn, self._target_batch_id, E_INJECTED_FAULT)
            self.injection_hits.append(hit)
            exc = VoicecalError(E_INJECTED_FAULT, injection["message"], injection_id=injection["injection_id"], stage="initial")
            return {
                "cache_key": key,
                "status": "failed",
                "extract_key": extract_record["cache_key"],
                "failure": _stage_failure(turn, "initial", exc, injection_id=injection["injection_id"]),
                "injection_hits": [hit],
            }
        return self._build_initial_safe(turn, extract_record, key)

    def _build_initial_safe(self, turn: dict[str, Any], extract_record: dict[str, Any], key: str) -> dict[str, Any]:
        try:
            _, record = build_initial_evidence(extract_record["aggregate_features"], extract_record["cache_key"], self.config)
        except Exception as exc:
            return {
                "cache_key": key,
                "status": "failed",
                "extract_key": extract_record["cache_key"],
                "failure": _stage_failure(turn, "initial", exc),
                "injection_hits": [],
            }
        record["cache_key"] = key
        record["task_sort_key"] = extract_record.get("initial_task_sort_key", f"{extract_record['file']}|segment|initial")
        record["injection_hits"] = []
        return record

    def _compute_calibration(
        self,
        turn: dict[str, Any],
        initial_record: dict[str, Any],
        previous_turn: dict[str, Any] | None,
        previous_calibration: dict[str, Any] | None,
        chain_sum_before: dict[str, int],
        gain: float,
    ) -> dict[str, Any]:
        injection_items = self._injections_for(turn["turn_id"], "calibration")
        previous_key = previous_calibration["cache_key"] if previous_calibration is not None else None
        key = calibration_cache_key(
            initial_record["cache_key"], previous_key, turn, previous_turn, gain, self.config, _injection_fingerprint(injection_items)
        )
        if injection_items:
            injection = injection_items[0]
            hit = _injection_hit(injection, turn, self._target_batch_id, E_INJECTED_FAULT)
            self.injection_hits.append(hit)
            exc = VoicecalError(E_INJECTED_FAULT, injection["message"], injection_id=injection["injection_id"], stage="calibration")
            return {
                "cache_key": key,
                "status": "failed",
                "initial_key": initial_record["cache_key"],
                "previous_calibration_key": previous_key,
                "failure": _stage_failure(turn, "calibration", exc, injection_id=injection["injection_id"]),
                "injection_hits": [hit],
            }
        try:
            _, record = calibrate_turn(initial_record, turn, previous_turn, previous_calibration, chain_sum_before, gain, self.config)
        except Exception as exc:
            return {
                "cache_key": key,
                "status": "failed",
                "initial_key": initial_record["cache_key"],
                "previous_calibration_key": previous_key,
                "failure": _stage_failure(turn, "calibration", exc),
                "injection_hits": [],
            }
        record["injection_hits"] = []
        return record

    def _compute_attribution(
        self,
        turn: dict[str, Any],
        calibration_record: dict[str, Any],
        extract_record: dict[str, Any],
        previous_attribution: dict[str, Any] | None,
        trajectory_context: dict[str, Any] | None,
        all_turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        injection_items = self._injections_for(turn["turn_id"], "attribution")
        previous_key = previous_attribution["cache_key"] if previous_attribution is not None else None
        key = attribution_cache_key(
            calibration_record["cache_key"],
            extract_record["cache_key"],
            turn,
            previous_key,
            self.config,
            _injection_fingerprint(injection_items),
        )
        if injection_items:
            injection = injection_items[0]
            hit = _injection_hit(injection, turn, self._target_batch_id, E_INJECTED_FAULT)
            self.injection_hits.append(hit)
            exc = VoicecalError(E_INJECTED_FAULT, injection["message"], injection_id=injection["injection_id"], stage="attribution")
            return {
                "cache_key": key,
                "status": "failed",
                "calibration_key": calibration_record["cache_key"],
                "failure": _stage_failure(turn, "attribution", exc, injection_id=injection["injection_id"]),
                "injection_hits": [hit],
            }
        try:
            _, record = attribute_turn(
                turn,
                calibration_record,
                extract_record,
                previous_attribution,
                trajectory_context,
                self.config,
                all_turns,
            )
        except Exception as exc:
            return {
                "cache_key": key,
                "status": "failed",
                "calibration_key": calibration_record["cache_key"],
                "failure": _stage_failure(turn, "attribution", exc),
                "injection_hits": [],
            }
        record["injection_hits"] = []
        return record

    def _pool(self, state: dict[str, Any], stage: str) -> dict[str, Any]:
        return state["stage_pools"].setdefault(stage, {})

    def _key_for_stage(
        self,
        stage: str,
        turn: dict[str, Any],
        extract_key: str | None,
        initial_key: str | None,
        calibration_key: str | None,
        previous_calibration: dict[str, Any] | None,
        previous_turn: dict[str, Any] | None,
        previous_attribution: dict[str, Any] | None,
        gain: float,
    ) -> str | None:
        if stage == "extract":
            return extract_key
        if stage == "initial":
            return initial_key
        if stage == "calibration":
            return calibration_key
        if stage == "attribution":
            assert calibration_key is not None and extract_key is not None
            return attribution_cache_key(
                calibration_key,
                extract_key,
                turn,
                previous_attribution["cache_key"] if previous_attribution is not None else None,
                self.config,
                _injection_fingerprint(self._injections_for(turn["turn_id"], "attribution")),
            )
        raise AssertionError(stage)

    def _needs_recompute(
        self,
        state: dict[str, Any],
        turn: dict[str, Any],
        extract_key: str,
        initial_key: str | None,
        calibration_key: str | None,
        attribution_key: str | None,
        previous_calibration: dict[str, Any] | None,
        previous_attribution: dict[str, Any] | None,
    ) -> bool:
        extract_pool = self._pool(state, "extract")
        extract_record = extract_pool.get(extract_key)
        if extract_record is None:
            return True
        if extract_record.get("status") != "ok":
            return False
        assert initial_key is not None
        initial_record = self._pool(state, "initial").get(initial_key)
        if initial_record is None:
            return True
        if initial_record.get("status") != "ok":
            return False
        assert calibration_key is not None
        calibration_record = self._pool(state, "calibration").get(calibration_key)
        if calibration_record is None:
            return True
        if calibration_record.get("status") != "ok":
            return False
        assert attribution_key is not None
        return self._pool(state, "attribution").get(attribution_key) is None

    def _process_target(
        self,
        state: dict[str, Any],
        target_state: dict[str, Any],
        budget: int,
        commit_interval: int,
        on_commit: Callable[[int, int], None],
    ) -> tuple[dict[str, Any], bool, int, list[str], int]:
        turns = target_state["dialogue"]["turns"]
        previous_turn_by_conv: dict[str, dict[str, Any]] = {}
        previous_calibration_by_conv: dict[str, dict[str, Any]] = {}
        previous_attribution_by_conv: dict[str, dict[str, Any]] = {}
        trajectory_context_by_conv: dict[str, dict[str, Any]] = {}
        chain_sum_by_conv: dict[str, dict[str, int]] = {}
        pending = False
        budget_used = 0
        processed_count = 0
        turn_keys = state.setdefault("turn_keys", {})
        processed_ids: list[str] = []
        last_cursor = 0
        self._prefetch_parallel(state, target_state, budget)
        for turn in turns:
            turn_id = turn["turn_id"]
            conversation = turn["conversation_id"]
            previous_turn = previous_turn_by_conv.get(conversation)
            previous_calibration = previous_calibration_by_conv.get(conversation)
            previous_attribution = previous_attribution_by_conv.get(conversation)
            trajectory_context = trajectory_context_by_conv.get(conversation)
            chain_sum_before = chain_sum_by_conv.get(conversation, {emotion: 0 for emotion in self.config["evidence"]["emotions"]})
            gain = effective_gain(self.config, target_state["config_overrides"], turn_id)

            path, relative_path, raw, audio_hash, asset_error = self._asset_for_turn(turn)
            extract_injections = self._injections_for(turn_id, "extract")
            extract_key = extract_cache_key(relative_path, audio_hash or f"missing:{relative_path}", self.config, extract_injections)
            initial_injections = self._injections_for(turn_id, "initial")
            initial_key = initial_cache_key(extract_key, self.config, _injection_fingerprint(initial_injections))
            calib_injections = self._injections_for(turn_id, "calibration")
            calibration_key = calibration_cache_key(
                initial_key,
                previous_calibration["cache_key"] if previous_calibration is not None else None,
                turn,
                previous_turn,
                gain,
                self.config,
                _injection_fingerprint(calib_injections),
            )
            attribution_key = attribution_cache_key(
                calibration_key,
                extract_key,
                turn,
                previous_attribution["cache_key"] if previous_attribution is not None else None,
                self.config,
                _injection_fingerprint(self._injections_for(turn_id, "attribution")),
            )
            needs = self._needs_recompute(
                state,
                turn,
                extract_key,
                initial_key,
                calibration_key,
                attribution_key,
                previous_calibration,
                previous_attribution,
            )
            if needs and budget_used >= budget:
                pending = True
                break

            turn_events_start = len(self.audit_events)
            if needs:
                budget_used += 1

            extract_record = self._pool(state, "extract").get(extract_key)
            if extract_record is None:
                extract_record = self._compute_extract(
                    turn,
                    extract_key,
                    relative_path,
                    audio_hash or f"missing:{relative_path}",
                    asset_error,
                    raw,
                    extract_injections,
                )
                self._pool(state, "extract")[extract_key] = extract_record
                self._record_audit(turn, "extract", "recomputed", extract_key)
            else:
                status = "recomputed" if extract_key in self._prefetched_extract_keys else "reused"
                self._record_audit(turn, "extract", status, extract_key)
            self.injection_hits.extend(extract_record.get("injection_hits", []))

            key_map = {"extract": extract_key}
            if extract_record.get("status") != "ok":
                turn_keys[turn_id] = key_map
                processed_count += 1
                processed_ids.append(turn_id)
                last_cursor = turn["turn_index"]
                self._record_audit(turn, "initial", "skipped_upstream_failure", None)
                self._record_audit(turn, "calibration", "skipped_upstream_failure", None)
                self._record_audit(turn, "attribution", "skipped_upstream_failure", None)
                if commit_interval > 0 and processed_count % commit_interval == 0:
                    on_commit(turn["turn_index"], processed_count)
                continue

            initial_record = self._pool(state, "initial").get(initial_key)
            if initial_record is None:
                initial_record = self._compute_initial(turn, extract_record)
                self._pool(state, "initial")[initial_key] = initial_record
                self._record_audit(turn, "initial", "recomputed", initial_key)
            else:
                status = "recomputed" if initial_key in self._prefetched_initial_keys else "reused"
                self._record_audit(turn, "initial", status, initial_key)
            self.injection_hits.extend(initial_record.get("injection_hits", []))
            key_map["initial"] = initial_key
            if initial_record.get("status") != "ok":
                turn_keys[turn_id] = key_map
                processed_count += 1
                processed_ids.append(turn_id)
                last_cursor = turn["turn_index"]
                self._record_audit(turn, "calibration", "skipped_upstream_failure", None)
                self._record_audit(turn, "attribution", "skipped_upstream_failure", None)
                if commit_interval > 0 and processed_count % commit_interval == 0:
                    on_commit(turn["turn_index"], processed_count)
                continue

            calibration_record = self._pool(state, "calibration").get(calibration_key)
            if calibration_record is None:
                calibration_record = self._compute_calibration(
                    turn,
                    initial_record,
                    previous_turn,
                    previous_calibration,
                    chain_sum_before,
                    gain,
                )
                self._pool(state, "calibration")[calibration_key] = calibration_record
                self._record_audit(turn, "calibration", "recomputed", calibration_key)
            else:
                self._record_audit(turn, "calibration", "reused", calibration_key)
            self.injection_hits.extend(calibration_record.get("injection_hits", []))
            key_map["calibration"] = calibration_key
            if calibration_record.get("status") != "ok":
                turn_keys[turn_id] = key_map
                processed_count += 1
                processed_ids.append(turn_id)
                last_cursor = turn["turn_index"]
                self._record_audit(turn, "attribution", "skipped_upstream_failure", None)
                if commit_interval > 0 and processed_count % commit_interval == 0:
                    on_commit(turn["turn_index"], processed_count)
                continue

            attribution_record = self._pool(state, "attribution").get(attribution_key)
            if attribution_record is None:
                attribution_record = self._compute_attribution(
                    turn,
                    calibration_record,
                    extract_record,
                    previous_attribution,
                    trajectory_context,
                    turns,
                )
                self._pool(state, "attribution")[attribution_key] = attribution_record
                self._record_audit(turn, "attribution", "recomputed", attribution_key)
            else:
                self._record_audit(turn, "attribution", "reused", attribution_key)
            self.injection_hits.extend(attribution_record.get("injection_hits", []))
            key_map["attribution"] = attribution_key
            turn_keys[turn_id] = key_map
            processed_count += 1
            processed_ids.append(turn_id)
            last_cursor = turn["turn_index"]

            previous_turn_by_conv[conversation] = turn
            previous_calibration_by_conv[conversation] = calibration_record
            chain_sum_by_conv[conversation] = calibration_record["chain_sum_fixed"]
            if attribution_record.get("status") == "ok":
                previous_attribution_by_conv[conversation] = attribution_record
                trajectory_context_by_conv[conversation] = {
                    "trajectory_id": attribution_record["trajectory_id"],
                    "turn_ids": attribution_record["trajectory_turn_ids"],
                    "turn_indices": attribution_record["trajectory_turn_indices"],
                    "speaker_id": attribution_record["speaker_id"],
                    "turn": turn,
                }
            if commit_interval > 0 and processed_count % commit_interval == 0:
                on_commit(turn["turn_index"], processed_count)

        return state, pending, budget_used, processed_ids, last_cursor

    def _prefetch_parallel(self, state: dict[str, Any], target_state: dict[str, Any], budget: int) -> None:
        """Parallelise full-batch extraction and initial evidence without changing budget semantics."""
        turns = target_state["dialogue"]["turns"]
        if budget < len(turns) or self.injections["injections"]:
            return
        workers = int(self.config["concurrency"])
        if workers <= 1 or len(turns) <= 1:
            return
        jobs: list[tuple[int, dict[str, Any], str, str, str, bytes | None, Exception | None]] = []
        for turn in turns:
            path, relative_path, raw, audio_hash, asset_error = self._asset_for_turn(turn)
            extract_key = extract_cache_key(relative_path, audio_hash or f"missing:{relative_path}", self.config, [])
            if extract_key in self._pool(state, "extract"):
                continue
            jobs.append((turn["turn_index"], turn, extract_key, relative_path, audio_hash or f"missing:{relative_path}", raw, asset_error))
        if len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(jobs)), thread_name_prefix="voicecal-extract-batch") as pool:
                futures = [
                    (
                        turn,
                        extract_key,
                        pool.submit(
                            self._compute_extract,
                            turn,
                            extract_key,
                            relative_path,
                            audio_hash,
                            asset_error,
                            raw,
                            [],
                            1,
                        ),
                    )
                    for _, turn, extract_key, relative_path, audio_hash, raw, asset_error in jobs
                ]
                for turn, extract_key, future in futures:
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = {
                            "cache_key": extract_key,
                            "status": "failed",
                            "file": turn["audio"],
                            "audio_sha256": None,
                            "failure": _stage_failure(turn, "extract", exc),
                            "injection_hits": [],
                        }
                    self._pool(state, "extract")[extract_key] = record
                    self._prefetched_extract_keys.add(extract_key)

        initial_jobs: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
        for turn in turns:
            _, relative_path, _, audio_hash, _ = self._asset_for_turn(turn)
            extract_key = extract_cache_key(relative_path, audio_hash or f"missing:{relative_path}", self.config, [])
            extract_record = self._pool(state, "extract").get(extract_key)
            if extract_record is None or extract_record.get("status") != "ok":
                continue
            initial_key = initial_cache_key(extract_key, self.config, [])
            if initial_key in self._pool(state, "initial"):
                continue
            initial_jobs.append((turn, initial_key, extract_record))
        if len(initial_jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(initial_jobs)), thread_name_prefix="voicecal-initial-batch") as pool:
                futures = []
                for turn, initial_key, extract_record in initial_jobs:
                    def build(current_turn: dict[str, Any] = turn, record: dict[str, Any] = extract_record, key: str = initial_key) -> dict[str, Any]:
                        self.work.mark_thread()
                        self.work.enter_initial()
                        try:
                            return self._build_initial_safe(current_turn, record, key)
                        finally:
                            self.work.leave_initial()

                    futures.append((turn, initial_key, pool.submit(build)))
                for turn, initial_key, future in futures:
                    self._pool(state, "initial")[initial_key] = future.result()
                    self._prefetched_initial_keys.add(initial_key)

    def _build_outputs(self, state: dict[str, Any], target_state: dict[str, Any], complete: bool, pending: list[dict[str, Any]]) -> dict[str, Any]:
        turns = target_state["dialogue"]["turns"]
        pools = state["stage_pools"]
        turn_keys = state.get("turn_keys", {})
        extraction_turns: list[dict[str, Any]] = []
        initial_turns: list[dict[str, Any]] = []
        calibration_turns: list[dict[str, Any]] = []
        attribution_turns: list[dict[str, Any]] = []
        ledger_entries: list[dict[str, Any]] = []
        referenced_files = {turn["audio"] for turn in turns}
        failures: list[dict[str, Any]] = [failure for failure in self._scan_failures if failure["file"] not in referenced_files]
        trajectory_groups: dict[str, dict[str, Any]] = {}

        for turn in turns:
            keys = turn_keys.get(turn["turn_id"], {})
            extract_record = pools["extract"].get(keys.get("extract", "")) if keys.get("extract") else None
            initial_record = pools["initial"].get(keys.get("initial", "")) if keys.get("initial") else None
            calibration_record = pools["calibration"].get(keys.get("calibration", "")) if keys.get("calibration") else None
            attribution_record = pools["attribution"].get(keys.get("attribution", "")) if keys.get("attribution") else None
            failure: dict[str, Any] | None = None
            for stage, record in (
                ("extract", extract_record),
                ("initial", initial_record),
                ("calibration", calibration_record),
                ("attribution", attribution_record),
            ):
                if record is not None and record.get("status") == "failed":
                    failure = copy.deepcopy(record["failure"])
                    failure.setdefault("stage", stage)
                    break
            if failure is not None:
                failures.append(copy.deepcopy(failure))

            extraction_turns.append(
                {
                    "turn_index": turn["turn_index"],
                    "turn_id": turn["turn_id"],
                    "file": turn["audio"],
                    "status": "ok" if extract_record and extract_record.get("status") == "ok" else "failed" if extract_record else "pending",
                    "cache_key": keys.get("extract"),
                    "record": extract_record,
                }
            )
            initial_turns.append(
                {
                    "turn_index": turn["turn_index"],
                    "turn_id": turn["turn_id"],
                    "status": "ok" if initial_record and initial_record.get("status") == "ok" else "failed" if initial_record else "pending",
                    "cache_key": keys.get("initial"),
                    "record": initial_record,
                }
            )
            calibration_turns.append(
                {
                    "turn_index": turn["turn_index"],
                    "turn_id": turn["turn_id"],
                    "status": "ok" if calibration_record and calibration_record.get("status") == "ok" else "failed" if calibration_record else "pending",
                    "cache_key": keys.get("calibration"),
                    "record": calibration_record,
                }
            )
            attribution_turns.append(
                {
                    "turn_index": turn["turn_index"],
                    "turn_id": turn["turn_id"],
                    "status": "ok" if attribution_record and attribution_record.get("status") == "ok" else "failed" if attribution_record else "pending",
                    "cache_key": keys.get("attribution"),
                    "record": attribution_record,
                }
            )

            if attribution_record is not None and attribution_record.get("status") == "ok":
                trajectory_id = attribution_record["trajectory_id"]
                group = trajectory_groups.setdefault(
                    trajectory_id,
                    {
                        "trajectory_id": trajectory_id,
                        "conversation_id": turn["conversation_id"],
                        "speaker_id": attribution_record["speaker_id"],
                        "turn_ids": [],
                        "turn_indices": [],
                        "breaks": [],
                    },
                )
                group["turn_ids"].append(turn["turn_id"])
                group["turn_indices"].append(turn["turn_index"])
                if attribution_record.get("trajectory_break_reason") is not None:
                    group["breaks"].append(
                        {
                            "turn_id": turn["turn_id"],
                            "turn_index": turn["turn_index"],
                            "reason": attribution_record["trajectory_break_reason"],
                            "gap_turns": attribution_record["gap_turns"],
                            "gap_ms": attribution_record["gap_ms"],
                        }
                    )

            entry = {
                "turn_index": turn["turn_index"],
                "turn_id": turn["turn_id"],
                "conversation_id": turn["conversation_id"],
                "conversation_position": turn["conversation_position"],
                "audio": turn["audio"],
                "script_speaker_id": turn["speaker_id"],
                "speaker_candidates": turn["speaker_candidates"],
                "previous_turn_id": turn["previous_turn_id"],
                "link_type": turn["link_type"],
                "cache_keys": {
                    "extract": keys.get("extract"),
                    "initial": keys.get("initial"),
                    "calibration": keys.get("calibration"),
                    "attribution": keys.get("attribution"),
                },
                "references": {
                    "audio_sha256": extract_record.get("audio_sha256") if extract_record else None,
                    "initial_cache_key": initial_record.get("cache_key") if initial_record else None,
                    "previous_calibration_cache_key": calibration_record.get("previous_calibration_key") if calibration_record else None,
                    "previous_turn_id": calibration_record.get("previous_turn_id") if calibration_record else None,
                },
                "initial_evidence_fixed": initial_record.get("intensities_fixed") if initial_record and initial_record.get("status") == "ok" else None,
                "calibrated_evidence_fixed": calibration_record.get("calibrated_evidence_fixed") if calibration_record and calibration_record.get("status") == "ok" else None,
                "emotion_decision": calibration_record.get("emotion_decision") if calibration_record and calibration_record.get("status") == "ok" else None,
                "speaker_attributed": attribution_record.get("speaker_id") if attribution_record and attribution_record.get("status") == "ok" else None,
                "trajectory_id": attribution_record.get("trajectory_id") if attribution_record and attribution_record.get("status") == "ok" else None,
                "attribution_basis": attribution_record.get("speaker_scores_fixed") if attribution_record and attribution_record.get("status") == "ok" else None,
                "calibration_coefficients": calibration_record.get("coefficients") if calibration_record and calibration_record.get("status") == "ok" else None,
                "failure": failure,
            }
            ledger_entries.append(entry)

        failures.sort(key=lambda item: (item.get("turn_index") or 0, STAGE_ORDER.get(item.get("stage", ""), 99), item.get("file", ""), item.get("code", "")))
        trajectories = sorted(
            trajectory_groups.values(),
            key=lambda trajectory: (trajectory["turn_indices"][0], trajectory["trajectory_id"]),
        )
        for trajectory in trajectories:
            trajectory["turn_ids"] = list(trajectory["turn_ids"])
            trajectory["turn_indices"] = list(trajectory["turn_indices"])

        fixed = self.config["fixed_point"]
        stage_outputs = {
            "extraction": {
                "stage": "extract",
                "artifact_version": STAGE_ARTIFACT_VERSION,
                "frame_algorithm_metadata": ALGORITHM_METADATA,
                "fixed_point": fixed,
                "turns": extraction_turns,
            },
            "initial_evidence": {
                "stage": "initial",
                "artifact_version": STAGE_ARTIFACT_VERSION,
                "algorithm": "weighted_features_stable_softmax",
                "fixed_point": fixed,
                "turns": initial_turns,
            },
            "calibration": {
                "stage": "calibration",
                "artifact_version": STAGE_ARTIFACT_VERSION,
                "algorithm": "explicit_ordered_fixed_point_chain",
                "fixed_point": fixed,
                "turns": calibration_turns,
            },
            "attribution": {
                "stage": "attribution",
                "artifact_version": STAGE_ARTIFACT_VERSION,
                "algorithm": "calibrated_emotion_affinity_plus_pitch_and_script_prior",
                "fixed_point": fixed,
                "turns": attribution_turns,
                "trajectories": trajectories,
            },
        }
        stage_digests = {
            "initial": json_hash(stage_outputs["initial_evidence"]["turns"]),
            "calibration": json_hash(stage_outputs["calibration"]["turns"]),
            "attribution": json_hash(stage_outputs["attribution"]),
        }
        chain_summary: dict[str, Any] = {}
        chain_records: dict[str, list[dict[str, Any]]] = {}
        for turn in turns:
            keys = turn_keys.get(turn["turn_id"], {})
            calibration_record = pools["calibration"].get(keys.get("calibration", "")) if keys.get("calibration") else None
            if calibration_record is not None and calibration_record.get("status") == "ok":
                digest_records = chain_records.setdefault(turn["conversation_id"], [])
                digest_records.append(
                    {
                        "turn_id": turn["turn_id"],
                        "turn_index": turn["turn_index"],
                        "calibrated_evidence_fixed": calibration_record["calibrated_evidence_fixed"],
                    }
                )
                chain_summary[turn["conversation_id"]] = {
                    "last_turn_id": turn["turn_id"],
                    "chain_sum_fixed": calibration_record["chain_sum_fixed"],
                    "chain_digest": json_hash(digest_records),
                }
        ledger = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "dialogue_id": target_state["dialogue"]["dialogue_id"],
            "fixed_point": fixed,
            "entries": ledger_entries,
            "evidence_chain_summary": chain_summary,
            "stage_digests": stage_digests,
        }
        failures_payload = {"schema_version": 1, "failures": failures}
        pending_payload = [
            {"turn_index": turn["turn_index"], "turn_id": turn["turn_id"], "code": E_CALIB_BUDGET_PENDING}
            for turn in pending
        ]
        return {
            "stage_outputs": stage_outputs,
            "ledger": ledger,
            "failures": failures_payload,
            "pending": pending_payload,
            "trajectories": trajectories,
        }

    def _report(
        self,
        outputs: dict[str, Any],
        target_state: dict[str, Any],
        complete: bool,
        pending_turns: list[dict[str, Any]],
        reused: bool,
        budget_used: int,
        next_batch_index: int,
    ) -> dict[str, Any]:
        counts: dict[str, dict[str, int]] = {}
        for event in self.audit_events:
            stage_counts = counts.setdefault(event["stage"], {"reused": 0, "recomputed": 0, "invalidated": 0, "pending": 0})
            if event["status"] == "reused":
                stage_counts["reused"] += 1
            elif event["status"] == "recomputed":
                stage_counts["recomputed"] += 1
            elif event["status"] == "invalidated":
                stage_counts["invalidated"] += 1
            elif event["status"] == "pending":
                stage_counts["pending"] += 1
        business_codes: list[str] = []
        if not complete:
            business_codes.append(E_CALIB_BUDGET_PENDING)
        for failure in outputs["failures"]["failures"]:
            business_codes.append(failure["code"])
        business_codes = sorted(set(business_codes))
        unique_hits: dict[tuple[str, str, str], dict[str, Any]] = {}
        for hit in list(target_state.get("injection_hits", [])) + self.injection_hits:
            key = (hit["injection_id"], hit["turn_id"], hit["stage"])
            unique_hits[key] = hit
        hits = [unique_hits[key] for key in sorted(unique_hits)]
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "product_version": PRODUCT_VERSION,
            "dialogue_id": target_state["dialogue"]["dialogue_id"],
            "status": "complete" if complete else "pending",
            "business_codes": business_codes,
            "snapshot_resumed": reused,
            "batch": {
                "next_batch_index": next_batch_index,
                "total_batches": len(self.edit_stream["batches"]),
                "target_batch_id": self._target_batch_id,
                "budget_per_batch": int(self.config["batch_budget"]),
                "budget_used": budget_used,
            },
            "calibration": [
                {
                    "turn_index": item["turn_index"],
                    "turn_id": item["turn_id"],
                    "emotion": item["record"].get("emotion_decision") if item["record"] else None,
                    "calibrated_evidence_fixed": item["record"].get("calibrated_evidence_fixed") if item["record"] else None,
                    "cache_key": item["cache_key"],
                }
                for item in outputs["stage_outputs"]["calibration"]["turns"]
            ],
            "attribution": {
                "turns": [
                    {
                        "turn_index": item["turn_index"],
                        "turn_id": item["turn_id"],
                        "speaker": item["record"].get("speaker_id") if item["record"] else None,
                        "trajectory_id": item["record"].get("trajectory_id") if item["record"] else None,
                    }
                    for item in outputs["stage_outputs"]["attribution"]["turns"]
                ],
                "trajectories": outputs["trajectories"],
            },
            "ledger": {
                "hit_counts": {stage: counts.get(stage, {}).get("reused", 0) for stage in STAGES},
                "recompute_counts": {stage: counts.get(stage, {}).get("recomputed", 0) for stage in STAGES},
                "invalidation_counts": {
                    stage: counts.get(stage, {}).get("recomputed", 0) + counts.get(stage, {}).get("invalidated", 0)
                    for stage in STAGES
                },
                "events": sorted(self.audit_events, key=lambda item: (item["turn_index"], STAGE_ORDER.get(item["stage"], 99), item["turn_id"])),
            },
            "injection": {
                "injection_hash": self.injections["injection_hash"],
                "configured_count": len(self.injections["injections"]),
                "hits": hits,
            },
            "failures": outputs["failures"]["failures"],
            "pending": outputs["pending"],
            "concurrency": {
                "configured_max_workers": int(self.config["concurrency"]),
                "max_active_frame_workers_observed": self.work.max_active_workers,
                "max_active_workers_limit_respected": self.work.max_active_workers <= int(self.config["concurrency"]),
            },
            "work": {
                "frame_tasks": self.work.frame_tasks,
                "aggregation_tasks": self.work.aggregation_tasks,
                "initial_evidence_tasks": self.work.initial_evidence_tasks,
                "max_active_initial_workers_observed": self.work.max_active_initial_workers,
            },
        }

    def _write_outputs(self, outputs: dict[str, Any], report: dict[str, Any], state: dict[str, Any], target_state: dict[str, Any]) -> None:
        out = self.paths.output_dir
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "extraction.json", outputs["stage_outputs"]["extraction"])
        write_json(out / "initial_evidence.json", outputs["stage_outputs"]["initial_evidence"])
        write_json(out / "calibration.json", outputs["stage_outputs"]["calibration"])
        write_json(out / "attribution.json", outputs["stage_outputs"]["attribution"])
        write_json(out / "calibration_ledger.json", outputs["ledger"])
        write_json(out / "failures.json", outputs["failures"])
        write_json(out / "run_report.json", report)
        write_json(out / "effective_dialogue.json", target_state["dialogue"])
        write_json(
            out / "effective_state.json",
            {
                "dialogue": target_state["dialogue"],
                "config_overrides": target_state["config_overrides"],
                "applied_batch_count": target_state["applied_batch_count"],
                "applied_batches": target_state.get("applied_batches", []),
            },
        )

    def _commit(self, state: dict[str, Any], turn_cursor: int, processed_count: int) -> None:
        effective_state = state.get("target_state") or state["effective_state"]
        self.snapshot.commit(
            state=state,
            stage_pools=state["stage_pools"],
            base_config_hash=self.base_config_hash,
            effective_config_hash=self.effective_config_hash(effective_state),
            injection_hash=self.injections["injection_hash"],
            fixed_point_bits=int(self.config["fixed_point"]["bit_width"]),
            turn_cursor=turn_cursor,
            processed_turn_count=processed_count,
        )

    def run(self, fresh: bool = False, process_all_batches: bool = True) -> dict[str, Any]:
        state, resumed = self._load_or_new_state(fresh)
        self._scan_audio_assets()
        batches = self.edit_stream["batches"]
        self.injection_hits = []
        self.audit_events = []
        final_outputs: dict[str, Any] | None = None
        final_report: dict[str, Any] | None = None
        final_target: dict[str, Any] | None = None

        while True:
            if state.get("target_state") is not None:
                target_state = state["target_state"]
                target_batch_index = int(state["target_batch_index"])
                self._target_batch_id = f"batch-{target_batch_index}"
            elif int(state["next_batch_index"]) < len(batches):
                target_batch_index = int(state["next_batch_index"]) + 1
                batch = batches[target_batch_index - 1]
                target_state = apply_batch(state["effective_state"], batch, target_batch_index)
                state["target_state"] = target_state
                state["target_batch_index"] = target_batch_index
                state["turn_keys"] = {}
                self._target_batch_id = batch["batch_id"]
            else:
                target_state = state["effective_state"]
                target_batch_index = int(state["next_batch_index"])
                self._target_batch_id = "no-pending-batch"

            old_turn_ids = {turn["turn_id"]: turn for turn in state["effective_state"]["dialogue"]["turns"]}
            new_turns = target_state["dialogue"]["turns"]
            new_ids = {turn["turn_id"] for turn in new_turns}
            for old_turn in old_turn_ids.values():
                if old_turn["turn_id"] not in new_ids and target_state is not state["effective_state"]:
                    self._record_audit(old_turn, "delete", "invalidated", None, "turn removed by edit command")

            budget = len(new_turns) + len(batches) + 1 if fresh else int(self.config["batch_budget"])
            commit_interval = int(self.config["snapshot_commit_interval"])
            last_cursor = {"value": 0}

            def on_commit(cursor: int, processed: int) -> None:
                last_cursor["value"] = cursor
                self._commit(state, cursor, processed)

            state, pending, budget_used, processed_ids, last_processed_cursor = self._process_target(
                state,
                target_state,
                budget,
                commit_interval,
                on_commit,
            )
            complete = not pending
            pending_turns = []
            if pending:
                processed_id_set = set(processed_ids)
                pending_turns = [turn for turn in new_turns if turn["turn_id"] not in processed_id_set]
                for turn in pending_turns:
                    for stage in STAGES:
                        self._record_audit(turn, stage, "pending", None)
            else:
                state["effective_state"] = target_state
                state["next_batch_index"] = target_batch_index
                state["target_state"] = None
                state["target_batch_index"] = None
                target_ids = {turn["turn_id"] for turn in new_turns}
                state["turn_keys"] = {
                    turn_id: key_map for turn_id, key_map in state.get("turn_keys", {}).items() if turn_id in target_ids
                }
                used_keys = {key for key_map in state.get("turn_keys", {}).values() for key in key_map.values()}
                for stage in STAGES:
                    state["stage_pools"][stage] = {
                        key: record for key, record in state["stage_pools"].get(stage, {}).items() if key in used_keys
                    }

            outputs = self._build_outputs(state, target_state, complete, pending_turns)
            report = self._report(outputs, target_state, complete, pending_turns, resumed, budget_used, target_batch_index)
            self._write_outputs(outputs, report, state, target_state)
            self._commit(
                state,
                last_processed_cursor if pending else new_turns[-1]["turn_index"] if new_turns else 0,
                len(state.get("turn_keys", {})) if pending else len(new_turns),
            )
            final_outputs = outputs
            final_report = report
            final_target = target_state
            if pending or not process_all_batches or target_batch_index >= len(batches):
                break
            resumed = True

        assert final_outputs is not None and final_report is not None and final_target is not None
        final_report["state"] = {
            "next_batch_index": state["next_batch_index"],
            "complete": state["next_batch_index"] >= len(batches) and state.get("target_state") is None,
        }
        write_json(self.paths.output_dir / "run_report.json", final_report)
        return final_report


def run_engine(paths: EnginePaths, fresh: bool = False, process_all_batches: bool = True) -> dict[str, Any]:
    return Engine(paths).run(fresh=fresh, process_all_batches=process_all_batches)
