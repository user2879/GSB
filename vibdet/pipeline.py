"""End-to-end deterministic anomaly-detection pipeline orchestration."""

import json
from pathlib import Path

import numpy as np

from .cache import ContentCache
from .detectors import DETECTOR_ORDER, run_detectors
from .errors import (ALL_ERROR_CODES, E_WINDOW_INVALID,
                     PipelineError, failure_entry)
from .hashing import stable_hash
from .quantization import fuse_and_judge
from .scheduler import Scheduler, ShutdownRequested
from .segments import map_verdicts_to_segments
from .store import SnapshotStore
from .spectrum import build_spectrum
from .synth import generate_device, load_descriptions
from .edits import apply_window_overrides
from .windows import build_windows, feature_names, window_param_hash


def stable_task_key(device_id, index, stage):
    return f"{stage}|{device_id}|{int(index):08d}"


def run_pipeline(workspace, cfg, descriptions=None, shutdown_event=None,
                 point_cost=1):
    workspace = Path(workspace)
    cache = ContentCache(workspace / "cache")
    store = SnapshotStore(workspace)
    if descriptions is None:
        descriptions = load_descriptions(workspace / "descriptions.yaml")
    # normalize to the full document shape for membership and overrides,
    # preserving any top-level window overrides
    if "devices" not in descriptions:
        descriptions = {"devices": descriptions,
                        "window_overrides": descriptions.get(
                            "window_overrides", {})}
    store.reset_for_new_config(cfg["config_hash"],
                               _descriptions_fingerprint(descriptions))
    scheduler = Scheduler(
        cfg,
        budget_remaining=int(cfg["scheduler"]["point_budget"]),
        shutdown_event=shutdown_event)

    failures = []
    devices_output = []
    # global, de-duped pending map seeded from the last committed snapshot
    pending_map = {
        (item["device"], item["position"]): dict(item)
        for item in store.manifest.get("pending_queue", [])}

    for device_id in cfg["devices"]:
        if shutdown_event is not None and shutdown_event.is_set():
            break
        try:
            device_desc = descriptions["devices"].get(device_id)
            wave = generate_device(device_id, device_desc, cfg)
            replaced = apply_window_overrides(
                wave.values, descriptions, device_id,
                cfg["window"]["length"], cfg["window"]["hop"])
            if not np.array_equal(replaced, wave.values):
                wave.values = replaced
                from .hashing import array_fingerprint, stable_hash as _sh
                wave.data_fingerprint = _sh([
                    "wave", device_id, array_fingerprint(wave.values),
                    wave.missing_mask.tolist(), wave.resample_mask.tolist(),
                    [seg.seg_id for seg in wave.segments]])
            windows = build_windows(wave, cfg)
            if not windows:
                raise PipelineError(
                    E_WINDOW_INVALID,
                    f"device {device_id} waveform length {wave.values.shape[0]} "
                    f"is shorter than window length {cfg['window']['length']}",
                    device=device_id)
            spectrum = build_spectrum(device_id, windows, cfg,
                                      shutdown_event=shutdown_event)
        except PipelineError as exc:
            failures.append(failure_entry(exc.code, exc.message,
                                          device=device_id, stage="load"))
            continue
        except ShutdownRequested:
            break

        device_record = _score_device(device_id, wave, windows, spectrum, cfg,
                                      cache, scheduler, point_cost,
                                      pending_map, failures)
        if device_record is not None:
            devices_output.append(device_record)

    consumed_this_run = (int(cfg["scheduler"]["point_budget"])
                         - scheduler.budget_remaining)
    store.manifest["budget_consumed"] = int(consumed_this_run)
    global_pending = sorted(pending_map.values(),
                            key=lambda item: (item["device"], item["position"]))
    store.manifest["pending_queue"] = global_pending

    spectrum_payload, detector_payload, verdict_payload = _stage_payloads(
        devices_output)
    committed = {}
    try:
        for stage, payload in (("spectrum", spectrum_payload),
                               ("detector", detector_payload),
                               ("verdict", verdict_payload)):
            version, changed = store.commit_stage(stage, payload)
            committed[stage] = {"version": version, "changed": changed}
    except ShutdownRequested:
        pass

    failures.sort(key=lambda item: (item["device"] or "",
                                    item["stage"] or "", item["code"],
                                    item["position"]
                                    if item["position"] is not None else -1))
    result = {
        "config_hash": cfg["config_hash"],
        "window_param_hash": window_param_hash(cfg),
        "feature_names": list(feature_names(cfg)),
        "detector_order": list(DETECTOR_ORDER),
        "devices": devices_output,
        "failures": failures,
        "failure_code_order": list(ALL_ERROR_CODES),
        "pending_queue": global_pending,
        "snapshots": {
            "global_version": store.manifest["global_version"],
            "stages": committed,
            "budget_consumed": store.manifest["budget_consumed"],
            "budget_remaining": scheduler.budget_remaining,
        },
        "cache_stats": cache.stats(),
    }
    output_path = workspace / "output" / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, sort_keys=True, indent=2)
    tmp.replace(output_path)
    return result


def _score_device(device_id, wave, windows, spectrum, cfg, cache, scheduler,
                  point_cost, pending_map, failures):
    all_positions = sorted(pos.position for pos in spectrum.positions)
    # persisted pending items for this device are consumed first, then the
    # remaining positions in stable ascending order
    preferred = [p for p in all_positions
                 if (device_id, p) in pending_map]
    ordered_positions = preferred + [p for p in all_positions if p not in preferred]
    allowed_positions = []
    for order_index, position in enumerate(ordered_positions):
        scheduler.check_shutdown()
        token = (device_id, position)
        if scheduler.consume(point_cost):
            allowed_positions.append(position)
            pending_map.pop(token, None)
        else:
            # budget exhausted: every remaining position stays pending,
            # including positions already queued from a previous snapshot
            for leftover in ordered_positions[order_index:]:
                left_token = (device_id, leftover)
                pending_map[left_token] = {
                    "device": device_id, "position": leftover,
                    "cost": point_cost}
                failures.append(scheduler.budget_pending(
                    device_id, leftover, point_cost))
            break
    if not allowed_positions:
        return None

    sub_positions = [spectrum.positions[p] for p in allowed_positions]

    def _cache_spectrum(pos):
        payload, _ = cache.compute("spectrum", pos.cache_key, pos.to_dict)
        return payload

    spectrum_output = scheduler.map_ordered([
        (stable_task_key(device_id, pos.position, "spectrum"),
         lambda pos=pos: _cache_spectrum(pos))
        for pos in sub_positions])
    cached_spectrum = [payload for _, payload in spectrum_output]

    sub_matrix = np.vstack([pos.vector for pos in sub_positions])
    detector_results = run_detectors(sub_matrix, sub_positions, cfg)

    detector_tasks = []
    for name in DETECTOR_ORDER:
        result = detector_results[name]
        for local_index, position in enumerate(allowed_positions):
            def _produce(result=result, local_index=local_index, name=name,
                         position=position):
                payload, _ = cache.compute(
                    "detector", result.cache_keys[local_index],
                    lambda: {
                        "detector": name,
                        "position": position,
                        "score": float(result.scores[local_index]),
                        "param_hash": result.param_hash,
                        "cache_key": result.cache_keys[local_index],
                    })
                return payload
            detector_tasks.append(
                (stable_task_key(device_id, position, f"detector:{name}"),
                 _produce))
    detector_executed = scheduler.map_ordered(detector_tasks)

    detector_records = {name: [] for name in DETECTOR_ORDER}
    for _, payload in detector_executed:
        detector_records[payload["detector"]].append(payload)
    for name in DETECTOR_ORDER:
        detector_records[name].sort(key=lambda item: item["position"])

    verdicts = fuse_and_judge(detector_results, cfg)
    verdict_output = scheduler.map_ordered([
        (stable_task_key(device_id, allowed_positions[i], "verdict"),
         lambda verdict=verdicts[i]: cache.compute(
             "verdict", verdict.verdict_key, verdict.to_dict)[0])
        for i in range(len(verdicts))])
    verdict_records = []
    for i, (_, payload) in enumerate(verdict_output):
        payload = dict(payload)
        payload["position"] = allowed_positions[i]
        verdict_records.append(payload)
    verdict_records.sort(key=lambda item: item["position"])

    # back-map using every spectrum position's registered segments so
    # resumed runs (which score a partial subset) keep a stable segment map
    segment_map = map_verdicts_to_segments(spectrum.positions, verdicts,
                                           allowed_positions)
    return {
        "device_id": device_id,
        "wave": wave.to_manifest(),
        "window_count": len(windows),
        "windows": [
            {
                "index": win.index,
                "start": win.start,
                "end": win.end,
                "segments": list(win.segments),
                "features": {k: float(v) for k, v in win.features.items()},
                "content_hash": win.content_hash,
                "param_hash": win.param_hash,
                "fingerprint": win.fingerprint,
            }
            for win in windows
        ],
        "spectrum": spectrum.to_manifest(),
        "spectrum_positions": cached_spectrum,
        "detectors": {
            name: [dict(item) for item in detector_records[name]]
            for name in DETECTOR_ORDER
        },
        "verdicts": verdict_records,
        "segment_verdicts": segment_map,
        "allowed_positions": list(allowed_positions),
    }


def _stage_payloads(devices_output):
    spectrum_payload, detector_payload, verdict_payload = {}, {}, {}
    for record in devices_output:
        device_id = record["device_id"]
        spectrum_payload[f"{device_id}.json"] = {
            "spectrum": record["spectrum"],
            "positions": record["spectrum_positions"],
        }
        detector_payload[f"{device_id}.json"] = record["detectors"]
        verdict_payload[f"{device_id}.json"] = {
            "verdicts": record["verdicts"],
            "segment_verdicts": record["segment_verdicts"],
        }
    return spectrum_payload, detector_payload, verdict_payload


def _descriptions_fingerprint(descriptions):
    return stable_hash(["descriptions", json.dumps(
        descriptions, sort_keys=True, default=str)])
