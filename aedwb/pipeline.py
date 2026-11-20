"""End-to-end AED shard calibration, decision, drift, and cache pipeline."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .alignment import align_packet_frames
from .calibration import calibrate_packet, derive_calibration
from .cache import ObjectCache
from .config import (
    codebook_hash,
    config_hash,
    load_audio_metadata,
    load_baseline,
    load_codebooks,
    load_config,
    load_injections,
    load_truth,
)
from .constants import STAGES, STAGE_ORDER
from .decision import decide_packet, fuse_and_decide_global_frames
from .drift import build_reference, diagnose_packet
from .loader import (
    PacketSource,
    discover_packet_sources,
    packet_content_hash,
    parse_packet,
)
from .mapping import map_packet_classes
from .snapshot import SnapshotStore
from .util import sha256_json, write_json_atomic

PIPELINE_SCHEMA = 1


def _key(payload: dict[str, Any]) -> str:
    return sha256_json({"kind": "packet-stage-key", "schema": PIPELINE_SCHEMA, **payload})


def _failure(
    code: str,
    packet_id: str | None,
    source_file: str | None,
    packet_index: int | None,
    stage: str,
    message: str,
    details: dict[str, Any] | None = None,
    injection_ids: list[str] | None = None,
) -> dict[str, Any]:
    index = -1 if packet_index is None else int(packet_index)
    return {
        "code": code,
        "packet_id": packet_id,
        "source_file": source_file,
        "packet_index": packet_index,
        "stage": stage,
        "message": message,
        "details": details or {},
        "injection_ids": sorted(injection_ids or []),
        "sort_key": [index, STAGE_ORDER.get(stage, len(STAGES)), code, message],
    }


def _header_versions(parse_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not parse_payload or not parse_payload.get("valid"):
        return {}
    return {
        "score_type": parse_payload.get("raw_score_type"),
        "class_codebook_version": parse_payload.get("class_codebook_version"),
        "frame_grid_version": parse_payload.get("frame_grid_version"),
        "detector_version": parse_payload.get("detector_version"),
    }


def _evaluate_truth(
    global_frames: list[dict[str, Any]],
    truth: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    clips = truth.get("clips", {}) if isinstance(truth, dict) else {}
    shift_ms = int(config["frame_shift_ms"])
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    event_results: list[dict[str, Any]] = []
    correct = 0
    considered = 0
    frames_by_audio: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in global_frames:
        frames_by_audio[frame["audio_file"]].append(frame)

    for audio_file in sorted(clips):
        clip = clips[audio_file]
        events = clip.get("events", []) if isinstance(clip, dict) else []
        for event in events:
            start_ms = int(event["start_ms"])
            end_ms = int(event["end_ms"])
            event_frames = [
                frame
                for frame in frames_by_audio.get(audio_file, [])
                if start_ms <= frame["global_time_ms"] + shift_ms / 2.0 < end_ms
            ]
            matching = sum(1 for frame in event_frames if frame["decision"] == event["class"])
            event_results.append(
                {
                    "audio_file": audio_file,
                    "event_id": event["event_id"],
                    "class": event["class"],
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "frame_count": len(event_frames),
                    "matching_frame_count": matching,
                    "frame_recall": (matching / len(event_frames)) if event_frames else 0.0,
                    "event_detected": matching > 0,
                }
            )

    for frame in global_frames:
        midpoint = frame["global_time_ms"] + shift_ms / 2.0
        truth_label = "none"
        for event in clips.get(frame["audio_file"], {}).get("events", []):
            if int(event["start_ms"]) <= midpoint < int(event["end_ms"]):
                truth_label = event["class"]
                break
        confusion[truth_label][frame["decision"]] += 1
        considered += 1
        if frame["decision"] == truth_label:
            correct += 1

    return {
        "frame_accuracy": (correct / considered) if considered else 0.0,
        "frame_count": considered,
        "confusion": {
            truth_label: {decision: int(count) for decision, count in sorted(destinations.items())}
            for truth_label, destinations in sorted(confusion.items())
        },
        "events": sorted(
            event_results,
            key=lambda item: (item["audio_file"], item["start_ms"], item["event_id"]),
        ),
    }


def run_workbench(
    sample_dir: str | Path,
    state_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    """Run the complete workbench against a local sample directory."""

    started = time.perf_counter()
    sample_dir = Path(sample_dir)
    state_dir = Path(state_dir) if state_dir is not None else sample_dir / ".workbench"
    report_path = Path(report_path) if report_path is not None else state_dir / "report.json"
    state_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(sample_dir / "config.json")
    config_fingerprint = config_hash(config)
    codebooks = load_codebooks(sample_dir / "codebooks.json")
    codebook_fingerprint = codebook_hash(codebooks)
    baseline_ids = sorted(load_baseline(sample_dir / "baseline_packets.json"))
    injections = load_injections(sample_dir / "injections.json")
    truth = load_truth(sample_dir / "truth.json")
    audio_metadata = load_audio_metadata(sample_dir / "audio")
    if not audio_metadata:
        raise ValueError("sample directory contains no WAV audio files")

    cache = ObjectCache(state_dir / "objects")
    snapshots = SnapshotStore(state_dir, int(config["snapshot_commit_interval"]))
    recovery = snapshots.recovery_status(config["global_version"], config_fingerprint)
    sources = discover_packet_sources(sample_dir, injections)
    if not sources:
        raise ValueError("sample directory contains no packet JSON files")

    packet_id_counts: Counter[str] = Counter()
    for source in sources:
        if source.parsed and isinstance(source.parsed.get("packet_id"), str):
            packet_id_counts[source.parsed["packet_id"]] += 1

    workers = max(1, int(config["thread_pool_workers"]))
    workers_used = min(workers, max(1, len(sources)))
    failures: list[dict[str, Any]] = []
    traces: list[dict[str, dict[str, Any]]] = [dict() for _ in sources]
    ledger_bases: list[dict[str, Any]] = []
    for packet_index, source in enumerate(sources):
        ledger_bases.append(
            {
                "packet_index": packet_index,
                "stable_sort_key": [packet_index],
                "packet_id": source.fallback_packet_id,
                "source_file": source.relative_path,
                "injection_ids": sorted(
                    item["injection_id"] for item in source.applied_injections
                ),
            }
        )

    stage_manifests: dict[str, Any] = {}

    def mark(
        packet_index: int,
        stage: str,
        cache_key: str,
        status: str,
        error: str | None = None,
    ) -> None:
        traces[packet_index][stage] = {
            "cache_key": cache_key,
            "status": status,
            **({"error": error} if error else {}),
        }

    def run_tasks(
        stage: str,
        tasks: list[dict[str, Any]],
        compute: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ordered = sorted(tasks, key=lambda task: tuple(task["stable_sort_key"]))
        results: list[dict[str, Any] | None] = [None] * len(ordered)

        def execute(position: int, task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            key = task["cache_key"]
            cached = cache.load(key)
            if cached is not None:
                return position, {"payload": cached, "cache_status": "reused", "cache_key": key}
            try:
                payload = compute(task)
                cache.store(key, stage, payload)
                return position, {"payload": payload, "cache_status": "computed", "cache_key": key}
            except Exception as error:  # isolate one packet/stage failure
                return position, {
                    "error": f"{type(error).__name__}: {error}",
                    "cache_status": "failed",
                    "cache_key": key,
                }

        if not ordered:
            return []
        with ThreadPoolExecutor(max_workers=workers_used) as executor:
            futures = {
                executor.submit(execute, position, task): position
                for position, task in enumerate(ordered)
            }
            for future in as_completed(futures):
                position, result = future.result()
                results[position] = result
        final: list[dict[str, Any]] = []
        for task, result in zip(ordered, results):
            assert result is not None
            final.append({**task, **result})
        return final

    def snapshot_metadata(
        stage: str,
        payloads_by_index: dict[int, dict[str, Any]],
        calibration_fingerprint: str | None,
    ) -> dict[str, Any]:
        completed_ids = [
            ledger_bases[index]["packet_id"]
            for index in sorted(payloads_by_index)
            if payloads_by_index[index].get("valid", True)
        ]
        headers = {
            payloads_by_index[index].get("packet_id", ledger_bases[index]["packet_id"]): _header_versions(
                payloads_by_index[index]
            )
            for index in sorted(payloads_by_index)
            if _header_versions(payloads_by_index[index])
        }
        return {
            "global_version": config["global_version"],
            "config_hash": config_fingerprint,
            "calibration_fingerprint": calibration_fingerprint,
            "completed_packet_ids": completed_ids,
            "packet_header_versions": headers,
        }

    def commit(
        stage: str,
        keys_by_index: dict[int, str],
        payloads_by_index: dict[int, dict[str, Any]],
        calibration_fingerprint: str | None,
    ) -> None:
        records = [
            {
                "stable_sort_key": [index],
                "packet_id": payloads_by_index[index].get(
                    "packet_id", ledger_bases[index]["packet_id"]
                ),
                "cache_key": keys_by_index[index],
                "payload": payloads_by_index[index],
            }
            for index in sorted(payloads_by_index)
        ]
        stage_manifests[stage] = snapshots.commit_stage(
            stage,
            records,
            snapshot_metadata(stage, payloads_by_index, calibration_fingerprint),
        )

    # ---- Parse -----------------------------------------------------------------
    parse_tasks: list[dict[str, Any]] = []
    for packet_index, source in enumerate(sources):
        audio = audio_metadata.get(
            source.parsed.get("audio_file") if source.parsed else None
        )
        cache_key = _key(
            {
                "stage": "parse",
                "global_version": config["global_version"],
                "parser_version": 1,
                "content_sha256": packet_content_hash(source),
                "duplicate_packet_id_count": packet_id_counts.get(source.fallback_packet_id, 0),
                "audio": audio,
                "class_count": len(config["classes"]),
                "injection_ids": sorted(
                    item["injection_id"] for item in source.applied_injections
                ),
            }
        )
        parse_tasks.append(
            {
                "packet_index": packet_index,
                "stable_sort_key": [packet_index],
                "source": source,
                "cache_key": cache_key,
            }
        )

    parse_results = run_tasks(
        "parse",
        parse_tasks,
        lambda task: parse_packet(
            task["source"], audio_metadata, dict(packet_id_counts), len(config["classes"])
        ),
    )
    parse_by_index: dict[int, dict[str, Any]] = {}
    parse_keys: dict[int, str] = {}
    packet_indexes: dict[str, int] = {}
    header_version_map: dict[str, Any] = {}
    for result in parse_results:
        packet_index = int(result["packet_index"])
        source: PacketSource = result["source"]
        if "error" in result:
            mark(packet_index, "parse", result["cache_key"], "failed", result["error"])
            failures.append(
                _failure(
                    "E_PACKET_INVALID",
                    source.fallback_packet_id,
                    source.relative_path,
                    packet_index,
                    "parse",
                    result["error"],
                    injection_ids=[item["injection_id"] for item in source.applied_injections],
                )
            )
            continue
        payload = result["payload"]
        parse_by_index[packet_index] = payload
        parse_keys[packet_index] = result["cache_key"]
        mark(packet_index, "parse", result["cache_key"], result["cache_status"])
        ledger_bases[packet_index]["packet_id"] = payload.get(
            "packet_id", ledger_bases[packet_index]["packet_id"]
        )
        if payload.get("valid"):
            packet_indexes[payload["packet_id"]] = packet_index
            header_version_map[payload["packet_id"]] = _header_versions(payload)
        else:
            messages = [item["message"] for item in payload.get("errors", [])]
            failures.append(
                _failure(
                    "E_PACKET_INVALID",
                    payload.get("packet_id"),
                    payload.get("source_file"),
                    packet_index,
                    "parse",
                    "; ".join(messages) if messages else "packet validation failed",
                    details={
                        "errors": payload.get("errors", []),
                        "frame_count": payload.get("frame_count", 0),
                    },
                    injection_ids=payload.get("injection_ids", []),
                )
            )
    commit("parse", parse_keys, parse_by_index, None)

    # ---- Semantic codebook mapping ---------------------------------------------
    valid_parse_indices = sorted(
        index for index, payload in parse_by_index.items() if payload.get("valid")
    )
    map_tasks: list[dict[str, Any]] = []
    map_keys: dict[int, str] = {}
    for packet_index in valid_parse_indices:
        cache_key = _key(
            {
                "stage": "map",
                "global_version": config["global_version"],
                "mapper_version": 3,
                "upstream_parse_key": parse_keys[packet_index],
                "codebook_hash": codebook_fingerprint,
                "global_codebook_version": config["class_codebook_version"],
                "global_codebook": config["class_codebook"],
                "classes": config["classes"],
            }
        )
        map_keys[packet_index] = cache_key
        map_tasks.append(
            {
                "packet_index": packet_index,
                "stable_sort_key": [packet_index],
                "cache_key": cache_key,
            }
        )
    map_results = run_tasks(
        "map",
        map_tasks,
        lambda task: map_packet_classes(
            parse_by_index[task["packet_index"]], config, codebooks
        ),
    )
    map_by_index: dict[int, dict[str, Any]] = {}
    for result in map_results:
        packet_index = int(result["packet_index"])
        if "error" in result:
            mark(packet_index, "map", result["cache_key"], "failed", result["error"])
            failures.append(
                _failure(
                    "E_CLASS_UNMAPPED",
                    parse_by_index[packet_index]["packet_id"],
                    parse_by_index[packet_index]["source_file"],
                    packet_index,
                    "map",
                    result["error"],
                    injection_ids=parse_by_index[packet_index].get("injection_ids", []),
                )
            )
            continue
        payload = result["payload"]
        map_by_index[packet_index] = payload
        mark(packet_index, "map", result["cache_key"], result["cache_status"])
        if payload["unmapped_frame_count"]:
            failures.append(
                _failure(
                    "E_CLASS_UNMAPPED",
                    payload["packet_id"],
                    payload["source_file"],
                    packet_index,
                    "map",
                    f"{payload['unmapped_frame_count']} frame(s) had no name-based class mapping and were zeroed",
                    details={
                        "codebook_version": payload["class_codebook_version"],
                        "codebook_found": payload["codebook_found"],
                        "mapping_entries": payload["mapping_entries"],
                        "unmapped_frame_count": payload["unmapped_frame_count"],
                        "first_unmapped_frames": payload["unmapped_frames"][:10],
                    },
                    injection_ids=payload.get("injection_ids", []),
                )
            )
    commit("map", {i: map_keys[i] for i in map_by_index}, map_by_index, None)

    # ---- Global frame-grid realignment -----------------------------------------
    align_tasks: list[dict[str, Any]] = []
    align_keys: dict[int, str] = {}
    for packet_index in sorted(map_by_index):
        cache_key = _key(
            {
                "stage": "align",
                "global_version": config["global_version"],
                "aligner_version": 2,
                "upstream_map_key": map_keys[packet_index],
                "global_frame_shift_ms": config["frame_shift_ms"],
                "alignment_method": "linear_interp_edge_hold_v1",
            }
        )
        align_keys[packet_index] = cache_key
        align_tasks.append(
            {
                "packet_index": packet_index,
                "stable_sort_key": [packet_index],
                "cache_key": cache_key,
            }
        )
    align_results = run_tasks(
        "align",
        align_tasks,
        lambda task: align_packet_frames(
            map_by_index[task["packet_index"]], int(config["frame_shift_ms"])
        ),
    )
    align_by_index: dict[int, dict[str, Any]] = {}
    for result in align_results:
        packet_index = int(result["packet_index"])
        if "error" in result:
            mark(packet_index, "align", result["cache_key"], "failed", result["error"])
            failures.append(
                _failure(
                    "E_PACKET_INVALID",
                    map_by_index[packet_index]["packet_id"],
                    map_by_index[packet_index]["source_file"],
                    packet_index,
                    "align",
                    result["error"],
                    injection_ids=map_by_index[packet_index].get("injection_ids", []),
                )
            )
            continue
        align_by_index[packet_index] = result["payload"]
        mark(packet_index, "align", result["cache_key"], result["cache_status"])
    commit("align", {i: align_keys[i] for i in align_by_index}, align_by_index, None)

    # ---- Shared baseline calibration parameters --------------------------------
    baseline_alignments: list[dict[str, Any]] = []
    baseline_align_keys: dict[str, str] = {}
    for baseline_id in baseline_ids:
        packet_index = packet_indexes.get(baseline_id)
        if packet_index is None or packet_index not in align_by_index:
            failures.append(
                _failure(
                    "E_CALIB_BASELINE",
                    baseline_id,
                    None,
                    None,
                    "calibrate",
                    "baseline packet is missing or invalid",
                )
            )
            continue
        baseline_alignments.append(align_by_index[packet_index])
        baseline_align_keys[baseline_id] = align_keys[packet_index]
    calibration_params, calibration_fingerprint = derive_calibration(
        baseline_alignments, config, baseline_align_keys
    )

    calibrate_tasks: list[dict[str, Any]] = []
    calibrate_keys: dict[int, str] = {}
    for packet_index in sorted(align_by_index):
        cache_key = _key(
            {
                "stage": "calibrate",
                "global_version": config["global_version"],
                "calibrator_version": 2,
                "upstream_align_key": align_keys[packet_index],
                "calibration_fingerprint": calibration_fingerprint,
                "calibration_method": calibration_params["method"],
                "calibration_tolerance": config["calibration_tolerance"],
            }
        )
        calibrate_keys[packet_index] = cache_key
        calibrate_tasks.append(
            {
                "packet_index": packet_index,
                "stable_sort_key": [packet_index],
                "cache_key": cache_key,
            }
        )
    calibrate_results = run_tasks(
        "calibrate",
        calibrate_tasks,
        lambda task: calibrate_packet(
            align_by_index[task["packet_index"]],
            calibration_params,
            calibration_fingerprint,
        ),
    )
    calibration_by_index: dict[int, dict[str, Any]] = {}
    for result in calibrate_results:
        packet_index = int(result["packet_index"])
        if "error" in result:
            mark(packet_index, "calibrate", result["cache_key"], "failed", result["error"])
            failures.append(
                _failure(
                    "E_CALIB_RESIDUAL",
                    align_by_index[packet_index]["packet_id"],
                    align_by_index[packet_index]["source_file"],
                    packet_index,
                    "calibrate",
                    result["error"],
                    injection_ids=align_by_index[packet_index].get("injection_ids", []),
                )
            )
            continue
        payload = result["payload"]
        calibration_by_index[packet_index] = payload
        mark(packet_index, "calibrate", result["cache_key"], result["cache_status"])
        if payload["residual_violation_count"]:
            failures.append(
                _failure(
                    "E_CALIB_RESIDUAL",
                    payload["packet_id"],
                    payload["source_file"],
                    packet_index,
                    "calibrate",
                    f"{payload['residual_violation_count']} frame(s) exceeded the probability-sum tolerance",
                    details={
                        "max_sum_deviation": payload["max_sum_deviation"],
                        "tolerance": calibration_params["tolerance"],
                        "violation_frame_count": payload["residual_violation_count"],
                        "first_violation_frames": payload["residual_violation_frames"][:20],
                    },
                    injection_ids=payload.get("injection_ids", []),
                )
            )
    commit(
        "calibrate",
        {i: calibrate_keys[i] for i in calibration_by_index},
        calibration_by_index,
        calibration_fingerprint,
    )

    baseline_calibrations = [
        calibration_by_index[packet_indexes[baseline_id]]
        for baseline_id in baseline_ids
        if baseline_id in packet_indexes and packet_indexes[baseline_id] in calibration_by_index
    ]
    reference_metadata, reference_lookup = build_reference(baseline_calibrations)
    reference_fingerprint = reference_metadata["fingerprint"]

    # ---- Threshold decisions ---------------------------------------------------
    decision_tasks: list[dict[str, Any]] = []
    decision_keys: dict[int, str] = {}
    decision_signature = {
        "decision_threshold": config["decision_threshold"],
        "probability_quantum": config["probability_quantum"],
        "class_priority": config["class_priority"],
        "classes": config["classes"],
    }
    for packet_index in sorted(calibration_by_index):
        cache_key = _key(
            {
                "stage": "decide",
                "global_version": config["global_version"],
                "decision_version": 1,
                "upstream_calibration_key": calibrate_keys[packet_index],
                "decision_signature": decision_signature,
            }
        )
        decision_keys[packet_index] = cache_key
        decision_tasks.append(
            {
                "packet_index": packet_index,
                "stable_sort_key": [packet_index],
                "cache_key": cache_key,
            }
        )
    decision_results = run_tasks(
        "decide",
        decision_tasks,
        lambda task: decide_packet(calibration_by_index[task["packet_index"]], config),
    )
    decision_by_index: dict[int, dict[str, Any]] = {}
    for result in decision_results:
        packet_index = int(result["packet_index"])
        if "error" in result:
            mark(packet_index, "decide", result["cache_key"], "failed", result["error"])
            failures.append(
                _failure(
                    "E_PACKET_INVALID",
                    calibration_by_index[packet_index]["packet_id"],
                    calibration_by_index[packet_index]["source_file"],
                    packet_index,
                    "decide",
                    result["error"],
                    injection_ids=calibration_by_index[packet_index].get("injection_ids", []),
                )
            )
            continue
        decision_by_index[packet_index] = result["payload"]
        mark(packet_index, "decide", result["cache_key"], result["cache_status"])
    commit(
        "decide",
        {i: decision_keys[i] for i in decision_by_index},
        decision_by_index,
        calibration_fingerprint,
    )

    # ---- Drift diagnosis -------------------------------------------------------
    drift_tasks: list[dict[str, Any]] = []
    drift_keys: dict[int, str] = {}
    for packet_index in sorted(calibration_by_index):
        cache_key = _key(
            {
                "stage": "drift",
                "global_version": config["global_version"],
                "drift_version": 4,
                "upstream_calibration_key": calibrate_keys[packet_index],
                "reference_fingerprint": reference_fingerprint,
                "drift_alert_threshold": config["drift_alert_threshold"],
            }
        )
        drift_keys[packet_index] = cache_key
        drift_tasks.append(
            {
                "packet_index": packet_index,
                "stable_sort_key": [packet_index],
                "cache_key": cache_key,
            }
        )
    drift_results = run_tasks(
        "drift",
        drift_tasks,
        lambda task: diagnose_packet(
            calibration_by_index[task["packet_index"]], reference_lookup, config
        ),
    )
    drift_by_index: dict[int, dict[str, Any]] = {}
    for result in drift_results:
        packet_index = int(result["packet_index"])
        if "error" in result:
            mark(packet_index, "drift", result["cache_key"], "failed", result["error"])
            failures.append(
                _failure(
                    "E_PACKET_INVALID",
                    calibration_by_index[packet_index]["packet_id"],
                    calibration_by_index[packet_index]["source_file"],
                    packet_index,
                    "drift",
                    result["error"],
                    injection_ids=calibration_by_index[packet_index].get("injection_ids", []),
                )
            )
            continue
        drift_by_index[packet_index] = result["payload"]
        mark(packet_index, "drift", result["cache_key"], result["cache_status"])
    commit(
        "drift",
        {i: drift_keys[i] for i in drift_by_index},
        drift_by_index,
        calibration_fingerprint,
    )

    # ---- Global fusion and report assembly -------------------------------------
    calibration_results = [calibration_by_index[index] for index in sorted(calibration_by_index)]
    global_frames, boundary_samples, decision_counts = fuse_and_decide_global_frames(
        calibration_results, packet_indexes, config
    )
    for frame in global_frames:
        frame["stable_sort_key"] = [frame["audio_file"], frame["global_frame_index"]]
    truth_evaluation = _evaluate_truth(global_frames, truth, config)

    packets_report: list[dict[str, Any]] = []
    drift_alerts: list[dict[str, Any]] = []
    for packet_index, base in enumerate(ledger_bases):
        parse = parse_by_index.get(packet_index)
        mapping = map_by_index.get(packet_index)
        alignment = align_by_index.get(packet_index)
        calibration = calibration_by_index.get(packet_index)
        decision = decision_by_index.get(packet_index)
        drift = drift_by_index.get(packet_index)
        if parse and not parse.get("valid"):
            status = "failed"
        elif parse and (mapping and mapping["unmapped_frame_count"] or calibration and calibration["residual_violation_count"]):
            status = "partial"
        elif parse and decision:
            status = "ok"
        else:
            status = "failed"

        packet_report: dict[str, Any] = {
            "packet_index": packet_index,
            "stable_sort_key": [packet_index],
            "packet_id": parse.get("packet_id") if parse else base["packet_id"],
            "source_file": base["source_file"],
            "status": status,
            "valid": bool(parse and parse.get("valid")),
            "header": parse.get("header", {}) if parse else {},
            "raw_score_type": parse.get("raw_score_type") if parse else None,
            "raw_frame_count": parse.get("frame_count", 0) if parse else 0,
            "raw_scores": parse.get("frames", []) if parse and parse.get("valid") else [],
            "raw_score_provenance": {
                "type": parse.get("raw_score_type") if parse else None,
                "decoded_with": (
                    "natural_exponential_softmax"
                    if parse and parse.get("raw_score_type") == "logit"
                    else "row_l1_normalization"
                    if parse and parse.get("raw_score_type") == "probability"
                    else None
                ),
                "source_file": base["source_file"],
            },
            "parse_errors": parse.get("errors", []) if parse else [],
            "injection_ids": base["injection_ids"],
            "mapping": None,
            "alignment": None,
            "calibration": None,
            "decisions": None,
            "drift": None,
        }
        if mapping:
            packet_report["mapping"] = {
                "codebook_version": mapping["class_codebook_version"],
                "codebook_found": mapping["codebook_found"],
                "mapping_by_name": mapping["mapping_by_name"],
                "entries": mapping["mapping_entries"],
                "unmapped_entries": mapping["unmapped_entries"],
                "unmapped_frame_count": mapping["unmapped_frame_count"],
                "unmapped_frames": mapping["unmapped_frames"],
            }
        if alignment:
            packet_report["alignment"] = {
                "method": alignment["alignment_method"],
                "local_frame_shift_ms": alignment["frame_shift_ms"],
                "global_frame_shift_ms": config["frame_shift_ms"],
                "original_frame_count": alignment["original_frame_count"],
                "target_frame_count": alignment["target_frame_count"],
                "resampled": alignment["resampled"],
                "uncovered_intervals_ms": alignment["uncovered_intervals_ms"],
                "edge_hold_frame_count": alignment["edge_hold_frame_count"],
                "mean_residual": alignment["mean_alignment_residual"],
                "max_residual": alignment["max_alignment_residual"],
            }
        if calibration:
            frames = []
            for frame in calibration["frames"]:
                frames.append(
                    {
                        "global_frame_index": frame["global_frame_index"],
                        "global_time_ms": frame["global_time_ms"],
                        "frame_sort_key": [packet_index, frame["global_frame_index"]],
                        "original_frame_indices": frame["original_frame_indices"],
                        "original_frame_index_anchor": frame["original_frame_index_anchor"],
                        "interpolation_fraction": frame["interpolation_fraction"],
                        "resampled": frame["resampled"],
                        "edge_hold": frame["edge_hold"],
                        "unmapped": frame["unmapped"],
                        "source_score_type": frame["source_score_type"],
                        "pre_calibration_probability": frame["pre_calibration_probability"],
                        "calibrated_probability": frame["calibrated_probability"],
                        "probability_sum": frame["probability_sum"],
                        "sum_deviation": frame["sum_deviation"],
                        "alignment_residual": frame["alignment_residual"],
                        "mapping_residual": frame["mapping_residual"],
                        "conversion_residual": frame["conversion_residual"],
                    }
                )
            packet_report["calibration"] = {
                "fingerprint": calibration["calibration_fingerprint"],
                "method": calibration["calibration_method"],
                "frame_count": calibration["frame_count"],
                "max_sum_deviation": calibration["max_sum_deviation"],
                "mean_conversion_residual": calibration["mean_conversion_residual"],
                "violation_frame_count": calibration["residual_violation_count"],
                "frames": frames,
            }
        if decision:
            packet_report["decisions"] = {
                "counts": decision["decision_counts"],
                "boundary_sample_count": decision["boundary_sample_count"],
                "frames": decision["frames"],
            }
        if drift:
            packet_report["drift"] = drift
            for alert in drift["alerts"]:
                drift_alerts.append(
                    {
                        **alert,
                        "packet_index": packet_index,
                        "sort_key": [packet_index, alert["audio_file"], alert["class"], alert["metric"]],
                    }
                )
        packets_report.append(packet_report)

    failures.sort(key=lambda item: (tuple(item["sort_key"]), item["code"], item["message"]))
    drift_alerts.sort(
        key=lambda item: (
            item["packet_index"],
            item["audio_file"],
            item["class"],
            item["metric"],
            item["value"],
        )
    )

    applied_by_id: dict[str, dict[str, Any]] = {}
    for source in sources:
        for injection in source.applied_injections:
            applied_by_id[injection["injection_id"]] = {
                **injection,
                "applied": True,
                "matched_source_file": source.relative_path,
            }
    injection_hits: list[dict[str, Any]] = []
    for sequence, injection in enumerate(injections, start=1):
        injection_id = injection["injection_id"]
        matching_failures = [
            failure for failure in failures if injection_id in failure.get("injection_ids", [])
        ]
        expected_code = injection.get("expected_code")
        codes = sorted({failure["code"] for failure in matching_failures})
        business_code = expected_code if expected_code in codes else (codes[0] if codes else None)
        injection_hits.append(
            {
                "sequence": sequence,
                "injection_id": injection_id,
                "kind": injection["kind"],
                "packet_id": injection.get("packet_id"),
                "packet_file": injection.get("packet_file"),
                "applied": injection_id in applied_by_id,
                "matched_source_file": applied_by_id.get(injection_id, {}).get(
                    "matched_source_file"
                ),
                "hit": bool(matching_failures),
                "business_code": business_code,
                "business_codes": codes,
                "expected_code": expected_code,
                "expected_code_observed": expected_code in codes,
            }
        )
    injection_hits.sort(key=lambda item: item["injection_id"])

    ledger_entries: list[dict[str, Any]] = []
    cache_counts: Counter[str] = Counter()
    for packet_index, base in enumerate(ledger_bases):
        parse = parse_by_index.get(packet_index)
        mapping = map_by_index.get(packet_index)
        calibration = calibration_by_index.get(packet_index)
        drift = drift_by_index.get(packet_index)
        stage_records: list[dict[str, Any]] = []
        for stage in STAGES:
            if stage in traces[packet_index]:
                record = traces[packet_index][stage]
            else:
                record = {"cache_key": None, "status": "skipped"}
            cache_counts[record["status"]] += 1
            stage_records.append({"stage": stage, **record})
        status = "failed"
        if parse and parse.get("valid") and packet_index in decision_by_index:
            if (
                mapping
                and mapping["unmapped_frame_count"]
                or calibration
                and calibration["residual_violation_count"]
            ):
                status = "partial"
            else:
                status = "ok"
        ledger_entries.append(
            {
                **base,
                "packet_id": parse.get("packet_id") if parse else base["packet_id"],
                "header_versions": _header_versions(parse),
                "calibration_fingerprint": calibration["calibration_fingerprint"]
                if calibration
                else None,
                "calibration_baseline_packets": baseline_ids,
                "referenced_baseline_packets": drift["reference_packet_ids"] if drift else [],
                "codebook_entries": mapping["mapping_entries"] if mapping else [],
                "stage_ledger": stage_records,
                "status": status,
            }
        )

    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    cache_summary = {
        "reused_stage_count": int(cache_counts.get("reused", 0)),
        "computed_stage_count": int(cache_counts.get("computed", 0)),
        "failed_stage_count": int(cache_counts.get("failed", 0)),
        "skipped_stage_count": int(cache_counts.get("skipped", 0)),
    }
    failed_packet_ids = [
        entry["packet_id"] for entry in ledger_entries if entry["status"] == "failed"
    ]
    completed_packet_ids = [entry["packet_id"] for entry in ledger_entries]
    packet_header_versions = [
        {
            "packet_id": entry["packet_id"],
            "source_file": entry["source_file"],
            "versions": entry["header_versions"],
        }
        for entry in ledger_entries
    ]

    report: dict[str, Any] = {
        "schema_version": "aed-workbench/1.0",
        "run": {
            "global_version": config["global_version"],
            "config_hash": config_fingerprint,
            "calibration_fingerprint": calibration_fingerprint,
            "snapshot_status": recovery,
            "max_workers": workers_used,
            "interaction_budget_ms": int(config["interaction_budget_ms"]),
            "elapsed_ms": elapsed_ms,
            "within_interaction_budget": elapsed_ms <= int(config["interaction_budget_ms"]),
            "thread_completion_order_policy": "collect_then_sort_by_stable_packet_and_frame_keys",
        },
        "config": config,
        "audio": [audio_metadata[name] for name in sorted(audio_metadata)],
        "calibration": {
            "fingerprint": calibration_fingerprint,
            "parameters": calibration_params,
            "baseline_packet_ids": baseline_ids,
            "reference": reference_metadata,
        },
        "packets": packets_report,
        "global_frames": global_frames,
        "decision_counts": decision_counts,
        "boundary_samples": boundary_samples,
        "drift_alerts": drift_alerts,
        "truth_evaluation": truth_evaluation,
        "failures": failures,
        "injection_hits": injection_hits,
        "ledger": {
            "entries": ledger_entries,
            "cache_summary": cache_summary,
            "ordering": "packet source path, then stage order, then global frame index",
        },
        "snapshots": {
            "commit_interval": int(config["snapshot_commit_interval"]),
            "stage_manifests": {
                stage: {
                    key: stage_manifests[stage].get(key)
                    for key in (
                        "global_version",
                        "config_hash",
                        "calibration_fingerprint",
                        "completed_packet_ids",
                        "record_count",
                        "cache_keys",
                        "chunk_files",
                    )
                }
                for stage in STAGES
                if stage in stage_manifests
            },
        },
    }

    consistency_payload = {
        "calibration": report["calibration"],
        "audio": report["audio"],
        "packets": report["packets"],
        "global_frames": report["global_frames"],
        "decision_counts": report["decision_counts"],
        "boundary_samples": report["boundary_samples"],
        "drift_alerts": report["drift_alerts"],
        "truth_evaluation": report["truth_evaluation"],
        "failures": report["failures"],
        "injection_hits": report["injection_hits"],
    }
    report["consistency_signature"] = sha256_json(consistency_payload)

    manifest = {
        "global_version": config["global_version"],
        "config_hash": config_fingerprint,
        "calibration_fingerprint": calibration_fingerprint,
        "packet_header_versions": packet_header_versions,
        "completed_packet_ids": completed_packet_ids,
        "failed_packet_ids": failed_packet_ids,
        "stage_snapshots": {
            stage: {
                "record_count": stage_manifests[stage]["record_count"],
                "completed_packet_ids": stage_manifests[stage]["completed_packet_ids"],
                "cache_keys": stage_manifests[stage]["cache_keys"],
            }
            for stage in STAGES
            if stage in stage_manifests
        },
        "consistency_signature": report["consistency_signature"],
    }
    snapshots.write_final_manifest(manifest)
    if write_report:
        write_json_atomic(report_path, report)
    return report
