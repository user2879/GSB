"""End-to-end deterministic VAD pipeline with caching and snapshots."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .audio import AudioFormatError, read_wav
from .cache import StageCache, stage_cache_key
from .config import ConfigError, SampleManifest, VadConfig, load_config, load_manifest
from .endpoints import build_endpoints
from .features import extract_features, feature_statistics
from .policy import STAGE_ORDER, estimate_scale_metrics, select_policy
from .snapshot import SnapshotStore
from .state_machine import estimate_noise_floor, run_state_machine
from .util import (
    PRODUCT_VERSIONS,
    atomic_write_json,
    round_float,
    sha256_json,
)


STAGE_SORT_ORDER = {
    "policy": 0,
    "features": 1,
    "state_machine": 2,
    "endpoints": 3,
}


class ConcurrencyCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.maximum = 0

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.maximum = max(self.maximum, self.current)

    def leave(self) -> None:
        with self._lock:
            self.current -= 1


def run_vad(
    input_dir: str | Path,
    config: str | Path | VadConfig,
    manifest: str | Path | SampleManifest,
    state_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    concurrency_override: int | None = None,
    stop_event: threading.Event | None = None,
    block_size: int = 1024,
) -> dict[str, Any]:
    config_obj = config if isinstance(config, VadConfig) else load_config(config)
    manifest_obj = manifest if isinstance(manifest, SampleManifest) else load_manifest(manifest)
    input_root = Path(input_dir).resolve()
    if not input_root.is_dir():
        raise ConfigError(f"input directory not found: {input_root}")

    wav_files = _scan_wav_files(input_root)
    stable_files = sorted(wav_files, key=lambda item: item[0])
    total_samples = len(stable_files)
    concurrency = int(config_obj.raw["concurrency"])
    if concurrency_override is not None:
        if int(concurrency_override) < 1:
            raise ConfigError("concurrency override must be positive")
        concurrency = min(concurrency, int(concurrency_override))
    snapshot_interval = int(config_obj.raw["snapshot_interval"])
    stop_event = stop_event or threading.Event()
    counter = ConcurrencyCounter()

    store = SnapshotStore(state_dir)
    existing = store.read_manifest()
    same_input = bool(
        existing
        and existing.get("config_hash") == config_obj.fingerprint
        and existing.get("injection_manifest_hash") == manifest_obj.fingerprint
    )
    prior_entries_by_file: dict[str, dict[str, Any]] = {}
    if existing:
        for entry in existing.get("sample_entries", []):
            if isinstance(entry, dict) and isinstance(entry.get("file"), str):
                prior_entries_by_file[entry["file"]] = entry

    global_version: int
    run_id: str
    entries: list[dict[str, Any]]
    outcomes: dict[int, dict[str, Any]]
    cursor: int
    manifest_base: dict[str, Any]
    if (
        same_input
        and existing.get("status") in {"running", "interrupted"}
        and isinstance(existing.get("global_version"), int)
    ):
        stored_cursor = existing.get("cursor", 0)
        prior_cursor = min(
            int(stored_cursor) if isinstance(stored_cursor, int) else 0,
            total_samples,
        )
        expected_prefix_files = [name for name, _path in stable_files[:prior_cursor]]
        prefix_hashes: list[str] = []
        for _name, path in stable_files[:prior_cursor]:
            try:
                prefix_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            except OSError:
                prefix_hashes.append("")
        records, cursor, recovery_problems = store.recover_prefix(
            existing,
            config_hash=config_obj.fingerprint,
            manifest_hash=manifest_obj.fingerprint,
            expected_files=expected_prefix_files,
            current_audio_hashes=prefix_hashes,
        )
        entries = list(existing.get("sample_entries", []))[:cursor]
        outcomes = {}
        for index, record in enumerate(records):
            outcomes[index] = _outcome_from_snapshot(record, entries[index])
        global_version = int(existing["global_version"])
        run_id = str(existing.get("run_id") or "")
        recovery = {
            "mode": "snapshot_resume",
            "manifest_cursor_before_recovery": int(existing.get("cursor", 0)),
            "validated_cursor": cursor,
            "recovered_sample_count": cursor,
            "recovery_problems": recovery_problems,
        }
        manifest_base = store.commit_manifest(
            existing,
            entries=entries,
            cursor=cursor,
            status="running",
            recovery=recovery,
        )
    else:
        prior_global = 0
        if existing and isinstance(existing.get("global_version"), int):
            prior_global = int(existing["global_version"])
        global_version = prior_global + 1
        if existing and not same_input:
            store.archive_prior_manifest(existing)
        run_id = sha256_json(
            {
                "config_hash": config_obj.fingerprint,
                "injection_manifest_hash": manifest_obj.fingerprint,
                "global_version": global_version,
            }
        )
        entries = []
        outcomes = {}
        cursor = 0
        recovery = {
            "mode": "fresh",
            "manifest_cursor_before_recovery": (
                int(existing.get("cursor", 0)) if existing else 0
            ),
            "validated_cursor": 0,
            "recovered_sample_count": 0,
            "recovery_problems": [],
        }
        manifest_base = store.start_manifest(
            config_hash=config_obj.fingerprint,
            manifest_hash=manifest_obj.fingerprint,
            total_samples=total_samples,
            global_version=global_version,
            run_id=run_id,
        )
    manifest_base["total_samples"] = total_samples

    diagnostics: list[dict[str, Any]] = []
    scanned_names = {name for name, _path in stable_files}
    for manifest_name in sorted(manifest_obj.samples_by_file):
        if manifest_name not in scanned_names:
            diagnostics.append(
                {
                    "code": "E_MANIFEST_SAMPLE_MISSING",
                    "severity": "warning",
                    "file": manifest_name,
                    "detail": "sample is listed in the injection manifest but no WAV was found",
                }
            )

    pending_entries: dict[int, dict[str, Any]] = {}
    pending_outcomes: dict[int, dict[str, Any]] = {}
    processed_since_commit = 0
    next_submit = cursor
    final_status = "running"

    def flush_commit(status: str) -> None:
        nonlocal processed_since_commit
        final_recovery = dict(recovery)
        final_recovery.update(
            {
                "resume_cursor": cursor,
                "resumable": status in {"running", "interrupted"},
            }
        )
        nonlocal manifest_base
        manifest_base = store.commit_manifest(
            manifest_base,
            entries=entries,
            cursor=cursor,
            status=status,
            recovery=final_recovery,
        )
        processed_since_commit = 0

    def finalize_ready() -> None:
        nonlocal cursor, processed_since_commit
        while cursor in pending_entries:
            entry = pending_entries.pop(cursor)
            outcome = pending_outcomes.pop(cursor)
            entries.append(entry)
            outcomes[cursor] = outcome
            cursor += 1
            processed_since_commit += 1

    if total_samples > 0:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            in_flight: dict[Future[dict[str, Any]], int] = {}
            while next_submit < total_samples or in_flight:
                while (
                    not stop_event.is_set()
                    and next_submit < total_samples
                    and len(in_flight) < concurrency
                ):
                    relative_name, path = stable_files[next_submit]
                    future = executor.submit(
                        _guarded_process_sample,
                        next_submit,
                        relative_name,
                        path,
                        config_obj,
                        manifest_obj,
                        prior_entries_by_file.get(relative_name),
                        store.cache,
                        counter,
                        block_size,
                        total_samples,
                    )
                    in_flight[future] = next_submit
                    next_submit += 1
                if not in_flight:
                    if stop_event.is_set():
                        break
                    continue
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in sorted(completed, key=lambda item: in_flight[item]):
                    index = in_flight.pop(future)
                    outcome = future.result()
                    snapshot_name = _write_outcome_snapshot(
                        store,
                        outcome,
                        config_hash=config_obj.fingerprint,
                        manifest_hash=manifest_obj.fingerprint,
                        global_version=global_version,
                        run_id=run_id,
                    )
                    outcome["snapshot"] = snapshot_name
                    pending_outcomes[index] = outcome
                    pending_entries[index] = _entry_from_outcome(outcome)
                finalize_ready()
                if processed_since_commit >= snapshot_interval:
                    flush_commit("running")
            finalize_ready()

    if cursor >= total_samples and not stop_event.is_set():
        final_status = "complete"
    else:
        final_status = "interrupted"
    flush_commit(final_status)

    ordered_outcomes = [outcomes[index] for index in sorted(outcomes)]
    sample_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    cache_events: list[dict[str, Any]] = []
    for outcome in ordered_outcomes:
        cache_events.extend(outcome.get("cache_events", []))
        if outcome["kind"] == "failure":
            failures.append(dict(outcome["failure"]))
        else:
            sample_report = dict(outcome["result"])
            sample_report["cache_events"] = list(outcome.get("cache_events", []))
            sample_reports.append(sample_report)
            diagnostics.extend(sample_report.pop("_diagnostics", []))
    sample_reports.sort(key=lambda item: item["stable_sort_key"])
    failures.sort(key=lambda item: (item.get("stable_sort_key", item.get("file", "")), item["code"]))
    diagnostics.sort(
        key=lambda item: (
            item.get("file", ""),
            item.get("code", ""),
            item.get("detail", ""),
        )
    )
    cache_events.sort(
        key=lambda item: (
            item["index"],
            STAGE_SORT_ORDER.get(item["stage"], 99),
            item["stage"],
        )
    )
    reuse_events = [event for event in cache_events if event["action"] == "reuse"]
    recompute_events = [event for event in cache_events if event["action"] == "recompute"]
    reuse_by_stage = {
        stage: sum(1 for event in reuse_events if event["stage"] == stage)
        for stage in STAGE_ORDER
    }
    recompute_by_stage = {
        stage: sum(1 for event in recompute_events if event["stage"] == stage)
        for stage in STAGE_ORDER
    }

    recovery_out = dict(recovery)
    recovery_out.update(
        {
            "resume_cursor": cursor,
            "processed_in_this_run": cursor - int(recovery.get("validated_cursor", 0)),
            "resumable": final_status in {"running", "interrupted"},
        }
    )
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "report_kind": "vad_endpoint_diagnostic",
        "status": final_status,
        "global_version": global_version,
        "run_id": run_id,
        "config_hash": config_obj.fingerprint,
        "injection_manifest_hash": manifest_obj.fingerprint,
        "product_versions": PRODUCT_VERSIONS,
        "algorithm_decisions": [
            {
                "topic": "framing",
                "choice": "20 ms frame / 10 ms hop by config; final short frame is zero padded",
                "rationale": "preserves overlap and makes every frame timestamp exact",
            },
            {
                "topic": "features",
                "choice": "float64 log RMS, strict-sign ZCR, Hann-window magnitude spectral flatness",
                "rationale": "separates high-energy harmonic speech from broadband stationary noise",
            },
            {
                "topic": "thresholds",
                "choice": "adaptive low threshold plus high threshold and hysteresis; strict comparisons",
                "rationale": "equal-boundary cases deterministically prefer silence",
            },
            {
                "topic": "state_machine",
                "choice": "SILENCE -> SUSPECT -> SPEECH -> TRAIL with consecutive-frame confirmation",
                "rationale": "prevents isolated amplitude spikes from entering SPEECH",
            },
            {
                "topic": "endpoint_merge",
                "choice": "merge gaps shorter than min_silence, then drop intervals shorter than min_speech",
                "rationale": "removes fragmented decisions while preserving short-gap speech continuity",
            },
            {
                "topic": "scale_policy",
                "choice": "weighted frame/density/load distance selects precision, noise window and threshold estimator",
                "rationale": "changes the actual numeric path across tens to tens-of-thousands of frames",
            },
            {
                "topic": "incremental_execution",
                "choice": "stage-specific content keys and atomic cache objects",
                "rationale": "invalidates only affected samples and stages while preserving exact reuse",
            },
            {
                "topic": "determinism",
                "choice": "bounded thread-pool tasks plus filename-indexed final ordering",
                "rationale": "completion order cannot change reports, cache audits or failure ordering",
            },
        ],
        "inputs": {
            "wav_file_count": total_samples,
            "manifest_sample_count": len(manifest_obj.samples_by_file),
            "input_directory_files": [name for name, _path in stable_files],
        },
        "execution": {
            "mode": "serial" if concurrency == 1 else "thread_pool",
            "configured_concurrency": int(config_obj.raw["concurrency"]),
            "effective_concurrency_limit": concurrency,
            "observed_max_concurrency": counter.maximum,
            "snapshot_interval": snapshot_interval,
        },
        "recovery": recovery_out,
        "samples": sample_reports,
        "failures": failures,
        "diagnostics": diagnostics,
        "cache_audit": {
            "reuse": reuse_events,
            "recompute": recompute_events,
            "reuse_count": len(reuse_events),
            "recompute_count": len(recompute_events),
            "reuse_by_stage": reuse_by_stage,
            "recompute_by_stage": recompute_by_stage,
            "snapshot_reuse_count": sum(
                1
                for event in reuse_events
                if event.get("reason") == "validated_atomic_snapshot_prefix"
            ),
        },
        "summary": {
            "samples_scanned": total_samples,
            "samples_processed": len(ordered_outcomes),
            "successful_samples": len(sample_reports),
            "failed_samples": len(failures),
            "endpoint_segments": sum(
                int(sample["segment_count"]) for sample in sample_reports
            ),
            "segment_count_mismatches": sum(
                1 for sample in sample_reports if not sample["segment_count_ok"]
            ),
            "noise_tolerance_failures": sum(
                1
                for sample in sample_reports
                if sample.get("noise_tolerance", {}).get("applicable")
                and not sample.get("noise_tolerance", {}).get("ok", True)
            ),
        },
    }
    result_samples = []
    for sample in report["samples"]:
        without_cache_events = dict(sample)
        without_cache_events.pop("cache_events", None)
        result_samples.append(without_cache_events)
    result_material = {
        "samples": result_samples,
        "failures": report["failures"],
        "diagnostics": report["diagnostics"],
    }
    report["result_digest"] = sha256_json(result_material)
    deterministic_material = dict(result_material)
    deterministic_material["cache_audit"] = report["cache_audit"]
    report["deterministic_digest"] = sha256_json(deterministic_material)

    if output_path is not None:
        atomic_write_json(Path(output_path), report)
    return report


def _scan_wav_files(root: Path) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".wav":
            relative = path.relative_to(root).as_posix()
            result.append((relative, path))
    return result


def _guarded_process_sample(
    index: int,
    relative_name: str,
    path: Path,
    config: VadConfig,
    manifest: SampleManifest,
    prior_entry: dict[str, Any] | None,
    cache: StageCache,
    counter: ConcurrencyCounter,
    block_size: int,
    total_sample_count: int,
) -> dict[str, Any]:
    counter.enter()
    try:
        return _process_sample(
            index,
            relative_name,
            path,
            config,
            manifest,
            prior_entry,
            cache,
            block_size,
            total_sample_count,
        )
    except AudioFormatError as exc:
        return _failure_outcome(
            index,
            relative_name,
            code=exc.code,
            message=exc.message,
            path=path,
        )
    except Exception as exc:  # Defensive isolation keeps one bad sample from aborting all.
        return _failure_outcome(
            index,
            relative_name,
            code="E_INTERNAL",
            message=f"{type(exc).__name__}: {exc}",
            path=path,
        )
    finally:
        counter.leave()


def _failure_outcome(
    index: int,
    file: str,
    *,
    code: str,
    message: str,
    path: Path | None = None,
) -> dict[str, Any]:
    audio_hash = _file_sha256(path) if path is not None else ""
    return {
        "index": index,
        "file": file,
        "stable_sort_key": file,
        "kind": "failure",
        "failure": {
            "file": file,
            "stable_sort_key": file,
            "code": code,
            "message": message,
        },
        "cache_events": [],
        "stage_keys": {},
        "fingerprints": {"audio_sha256": audio_hash},
        "profiles": {},
    }


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _process_sample(
    index: int,
    relative_name: str,
    path: Path,
    config: VadConfig,
    manifest: SampleManifest,
    prior_entry: dict[str, Any] | None,
    cache: StageCache,
    block_size: int,
    total_sample_count: int,
) -> dict[str, Any]:
    audio = read_wav(path, config.audio, relative_name)
    manifest_sample = manifest.samples_by_file.get(relative_name)
    if manifest_sample is None:
        expected_range = (0, 2_147_483_647)
        expected_profile = None
        shard_count = 1
        manifest_entry_present = False
    else:
        expected_range = tuple(int(item) for item in manifest_sample["expected_segments"])
        expected_profile = str(manifest_sample["scale_profile"])
        shard_count = int(manifest_sample.get("shard_count", 1))
        manifest_entry_present = True

    metrics = estimate_scale_metrics(
        audio.samples,
        audio.sample_rate,
        config.raw,
        sample_count=total_sample_count,
    )
    fingerprints = {
        "audio_sha256": audio.content_sha256,
        "policy_config": config.stage_fingerprint("policy"),
        "features_config": config.stage_fingerprint("features"),
        "state_config": config.stage_fingerprint("state"),
        "endpoints_config": config.stage_fingerprint("endpoints"),
    }
    profiles = {
        "policy": None,
        "features": None,
        "state_machine": None,
        "endpoints": None,
    }
    events: list[dict[str, Any]] = []

    policy_key = stage_cache_key(
        "policy",
        sample_key=relative_name,
        audio_fingerprint=audio.content_sha256,
        config_fingerprint=fingerprints["policy_config"],
        profile=None,
        extras={"metrics": metrics},
    )
    policy_result, policy_event = _obtain_stage(
        cache,
        "policy",
        policy_key,
        prior_entry,
        fingerprints,
        profiles,
        lambda: select_policy(metrics, config.raw),
        relative_name,
        index,
    )
    events.append(policy_event)
    selected_profile = str(policy_result["selected_profile"])
    effective = policy_result["effective_strategy"]
    profiles = {
        "policy": selected_profile,
        "features": selected_profile,
        "state_machine": selected_profile,
        "endpoints": selected_profile,
    }

    feature_key = stage_cache_key(
        "features",
        sample_key=relative_name,
        audio_fingerprint=audio.content_sha256,
        config_fingerprint=fingerprints["features_config"],
        profile=selected_profile,
        extras={
            "feature_precision": int(effective["feature_precision"]),
            "sample_rate": audio.sample_rate,
        },
    )
    features_result, feature_event = _obtain_stage(
        cache,
        "features",
        feature_key,
        prior_entry,
        fingerprints,
        profiles,
        lambda: extract_features(
            audio.samples,
            audio.sample_rate,
            config.raw,
            int(effective["feature_precision"]),
            block_size=block_size,
        ),
        relative_name,
        index,
    )
    events.append(feature_event)

    state_key = stage_cache_key(
        "state_machine",
        sample_key=relative_name,
        audio_fingerprint=audio.content_sha256,
        config_fingerprint=fingerprints["state_config"],
        profile=selected_profile,
        upstream_key=feature_key,
        extras={
            "noise_window_frames": int(effective["noise_window_frames"]),
            "threshold_estimation": effective["threshold_estimation"],
            "low_quantile": float(effective.get("low_quantile", 0.0)),
        },
    )

    def compute_state() -> dict[str, Any]:
        noise = estimate_noise_floor(features_result, policy_result, config.raw)
        return run_state_machine(features_result, noise, config.raw)

    state_result, state_event = _obtain_stage(
        cache,
        "state_machine",
        state_key,
        prior_entry,
        fingerprints,
        profiles,
        compute_state,
        relative_name,
        index,
    )
    events.append(state_event)

    endpoint_key = stage_cache_key(
        "endpoints",
        sample_key=relative_name,
        audio_fingerprint=audio.content_sha256,
        config_fingerprint=fingerprints["endpoints_config"],
        profile=selected_profile,
        upstream_key=state_key,
        extras={"expected_segment_range": list(expected_range)},
    )
    endpoints_result, endpoint_event = _obtain_stage(
        cache,
        "endpoints",
        endpoint_key,
        prior_entry,
        fingerprints,
        profiles,
        lambda: build_endpoints(
            features_result, state_result, config.raw, expected_range
        ),
        relative_name,
        index,
    )
    events.append(endpoint_event)

    precision = int(features_result["significant_digits"])
    statistics = feature_statistics(features_result, precision)
    tolerance = _noise_tolerance_check(
        relative_name, statistics, config.raw["noise_feature_tolerance"], manifest
    )
    frames: list[dict[str, Any]] = []
    for feature_frame, state_frame in zip(
        features_result["frames"], state_result["frame_states"]
    ):
        merged = dict(feature_frame)
        merged.update(state_frame)
        frames.append(merged)
    sample_profiles_match = (
        expected_profile is None or expected_profile == selected_profile
    )
    sample_result: dict[str, Any] = {
        "index": index,
        "file": relative_name,
        "stable_sort_key": relative_name,
        "content_sha256": audio.content_sha256,
        "sample_rate": audio.sample_rate,
        "duration_ms": round(float(audio.samples.size) * 1000.0 / audio.sample_rate, 9),
        "frame_count": int(features_result["frame_count"]),
        "shard_count": shard_count,
        "manifest_entry_present": manifest_entry_present,
        "expected_segment_count": [int(expected_range[0]), int(expected_range[1])],
        "segment_count": int(endpoints_result["segment_count"]),
        "segment_count_ok": bool(endpoints_result["segment_count_ok"]),
        "segments": endpoints_result["segments"],
        "noise_floor": state_result["noise_floor"],
        "scale": {
            "expected_profile": expected_profile,
            "selected_profile": selected_profile,
            "match": sample_profiles_match,
            "decision": policy_result,
        },
        "state_machine": {
            "confirm_frames": state_result["confirm_frames"],
            "release_frames": state_result["release_frames"],
            "state_counts": state_result["state_counts"],
            "transition_count": state_result["transition_count"],
            "endpoint_states": ["SUSPECT", "SPEECH", "TRAIL"],
        },
        "endpoint_postprocess": {
            key: value
            for key, value in endpoints_result.items()
            if key
            in {
                "raw_interval_count",
                "merged_interval_count",
                "discarded_short_interval_count",
                "min_speech_ms",
                "min_silence_ms",
                "frame_index_semantics",
            }
        },
        "frames": frames,
        "feature_statistics": statistics,
        "noise_tolerance": tolerance,
        "_diagnostics": [],
    }
    if not manifest_entry_present:
        sample_result["_diagnostics"].append(
            {
                "code": "E_MANIFEST_ENTRY_MISSING",
                "severity": "warning",
                "file": relative_name,
                "detail": "WAV was discovered on disk but has no manifest entry",
            }
        )
    if not sample_profiles_match:
        sample_result["_diagnostics"].append(
            {
                "code": "E_SCALE_PROFILE_HINT_MISMATCH",
                "severity": "warning",
                "file": relative_name,
                "detail": (
                    f"manifest expected {expected_profile}, adaptive selector chose "
                    f"{selected_profile}"
                ),
            }
        )
    if not sample_result["segment_count_ok"]:
        sample_result["_diagnostics"].append(
            {
                "code": "E_SEGMENT_COUNT_MISMATCH",
                "severity": "warning",
                "file": relative_name,
                "detail": (
                    f"detected {sample_result['segment_count']} segments; expected "
                    f"{expected_range[0]}..{expected_range[1]}"
                ),
            }
        )
    if tolerance["applicable"] and not tolerance["ok"]:
        sample_result["_diagnostics"].append(
            {
                "code": "E_NOISE_FEATURE_TOLERANCE",
                "severity": "error",
                "file": relative_name,
                "detail": "stable-noise feature statistics are outside configured tolerance",
            }
        )
    return {
        "index": index,
        "file": relative_name,
        "stable_sort_key": relative_name,
        "kind": "success",
        "result": sample_result,
        "cache_events": events,
        "stage_keys": {
            "policy": policy_key,
            "features": feature_key,
            "state_machine": state_key,
            "endpoints": endpoint_key,
        },
        "fingerprints": fingerprints,
        "profiles": profiles,
    }


def _obtain_stage(
    cache: StageCache,
    stage: str,
    key: str,
    prior_entry: dict[str, Any] | None,
    fingerprints: dict[str, Any],
    profiles: dict[str, Any],
    compute: Any,
    file: str,
    index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cached, cache_reason = cache.read(stage, key)
    if cached is not None:
        return cached, {
            "file": file,
            "index": index,
            "stage": stage,
            "action": "reuse",
            "cache_key": key,
            "reason": "cache_object_match",
        }
    result = compute()
    cache.write(stage, key, result)
    reason = _invalidation_reason(
        stage,
        key,
        prior_entry,
        fingerprints,
        profiles,
        cache_reason,
    )
    return result, {
        "file": file,
        "index": index,
        "stage": stage,
        "action": "recompute",
        "cache_key": key,
        "reason": reason,
    }


def _invalidation_reason(
    stage: str,
    current_key: str,
    prior_entry: dict[str, Any] | None,
    fingerprints: dict[str, Any],
    profiles: dict[str, Any],
    cache_reason: str,
) -> str:
    if prior_entry is None:
        return "new_sample_or_no_prior_snapshot"
    if prior_entry.get("kind") != "success":
        return "prior_sample_had_no_reusable_stage_outputs"
    prior_fingerprints = prior_entry.get("fingerprints") or {}
    if prior_fingerprints.get("audio_sha256") != fingerprints["audio_sha256"]:
        return "audio_content_changed"
    prior_profiles = prior_entry.get("profiles") or {}
    if stage != "policy" and prior_profiles.get(stage) != profiles.get(stage):
        return "scale_profile_changed"
    config_field = {
        "policy": "policy_config",
        "features": "features_config",
        "state_machine": "state_config",
        "endpoints": "endpoints_config",
    }[stage]
    if prior_fingerprints.get(config_field) != fingerprints[config_field]:
        return f"{stage}_configuration_changed"
    prior_key = (prior_entry.get("stage_keys") or {}).get(stage)
    if prior_key != current_key:
        if stage == "features":
            return "feature_precision_or_scale_profile_changed"
        if stage == "state_machine":
            return "upstream_features_or_noise_strategy_changed"
        if stage == "endpoints":
            return "upstream_state_or_endpoint_merge_inputs_changed"
        return "policy_scale_metrics_or_profile_changed"
    return cache_reason


def _noise_tolerance_check(
    file: str,
    statistics: dict[str, Any],
    tolerance: dict[str, Any],
    manifest: SampleManifest,
) -> dict[str, Any]:
    applicable = file in set(manifest.noise_reference_samples)
    checks: list[dict[str, Any]] = []
    if not applicable:
        return {"applicable": False, "ok": True, "checks": checks}
    for name, specification in tolerance.items():
        if not isinstance(specification, dict):
            continue
        prefix, separator, statistic_name = name.rpartition("_")
        if not separator or prefix not in statistics or statistic_name not in statistics[prefix]:
            checks.append(
                {
                    "name": name,
                    "ok": False,
                    "detail": "tolerance refers to an unknown feature statistic",
                }
            )
            continue
        value = float(statistics[prefix][statistic_name])
        lower = specification.get("min")
        upper = specification.get("max")
        passed = True
        if isinstance(lower, (int, float)) and value < float(lower):
            passed = False
        if isinstance(upper, (int, float)) and value > float(upper):
            passed = False
        checks.append(
            {
                "name": name,
                "value": round_float(value, 10),
                "min": lower,
                "max": upper,
                "ok": passed,
            }
        )
    return {
        "applicable": True,
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
    }


def _write_outcome_snapshot(
    store: SnapshotStore,
    outcome: dict[str, Any],
    *,
    config_hash: str,
    manifest_hash: str,
    global_version: int,
    run_id: str,
) -> str:
    if outcome["kind"] == "success":
        content = {
            "sample": outcome["result"],
            "stage_keys": outcome["stage_keys"],
            "fingerprints": outcome["fingerprints"],
            "profiles": outcome["profiles"],
        }
    else:
        content = {"failure": outcome["failure"]}
    return store.write_snapshot(
        index=int(outcome["index"]),
        file=outcome["file"],
        stable_sort_key=outcome["stable_sort_key"],
        kind=outcome["kind"],
        content=content,
        config_hash=config_hash,
        manifest_hash=manifest_hash,
        global_version=global_version,
        run_id=run_id,
    )


def _entry_from_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": int(outcome["index"]),
        "file": outcome["file"],
        "stable_sort_key": outcome["stable_sort_key"],
        "kind": outcome["kind"],
        "snapshot": outcome["snapshot"],
        "stage_keys": outcome.get("stage_keys", {}),
        "fingerprints": outcome.get("fingerprints", {}),
        "profiles": outcome.get("profiles", {}),
    }


def _outcome_from_snapshot(
    record: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any]:
    content = record["content"]
    kind = str(record["kind"])
    outcome: dict[str, Any] = {
        "index": int(record["index"]),
        "file": record["file"],
        "stable_sort_key": record.get("stable_sort_key", record["file"]),
        "kind": kind,
        "snapshot": entry.get("snapshot"),
        "stage_keys": entry.get("stage_keys", {}),
        "fingerprints": entry.get("fingerprints", {}),
        "profiles": entry.get("profiles", {}),
        "cache_events": [],
    }
    if kind == "success":
        outcome["result"] = content["sample"]
        for stage in STAGE_ORDER:
            key = outcome["stage_keys"].get(stage)
            if key:
                outcome["cache_events"].append(
                    {
                        "file": outcome["file"],
                        "index": outcome["index"],
                        "stage": stage,
                        "action": "reuse",
                        "cache_key": key,
                        "reason": "validated_atomic_snapshot_prefix",
                    }
                )
    else:
        outcome["failure"] = content["failure"]
    return outcome
