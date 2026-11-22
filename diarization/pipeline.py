"""Pipeline orchestration: staged processing, concurrency, snapshots, report."""
import hashlib
import json
import logging
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import audio_io, clustering, embedding, injection, segmentation, tracking
from .snapshot import SnapshotManager

LOG = logging.getLogger("diarization.pipeline")

ERROR_INTERNAL = "E_INTERNAL"


class StageFailure(Exception):
    """Business failure bound to one audio; processing continues elsewhere."""

    def __init__(self, code, stage, window_index=None, message=""):
        super().__init__(message or code)
        self.code = code
        self.stage = stage
        self.window_index = window_index
        self.message = message or code


def stable_sort_key(filename, window_index, stage):
    """Deterministic ordering key derived from file, window and stage."""
    return "%s#%06d#%s" % (filename, window_index, stage)


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _hash(obj):
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _map(pool, fn, items):
    """Ordered map: results always follow input order, serial or pooled."""
    if pool is None:
        return [fn(x) for x in items]
    return list(pool.map(fn, items))


def process_audio(path, config, inj_entries, pool, snap):
    name = os.path.basename(path)
    rec = {"file": name, "status": "ok"}
    try:
        samples, sr = audio_io.load_wav(path)
    except audio_io.AudioError as exc:
        raise StageFailure(exc.code, "load", None, exc.message)

    samples, hits = injection.apply_injections(samples, sr, name, inj_entries, config)
    rec["injection_hits"] = hits

    # ---- stage: windowing ----
    finite = np.isfinite(samples)
    if not bool(finite.all()):
        first_bad = int(np.argmin(finite.astype(np.int8)))
        win_guess = first_bad // max(1, sr * int(config["hop_ms"]) // 1000)
        raise StageFailure("E_NONFINITE_INPUT", "windowing", win_guess,
                           "non-finite sample at offset %d" % first_bad)
    windows = audio_io.make_windows(len(samples), sr, config)
    rec["windows"] = windows
    snap.commit("windowing", {"file": name, "windows": windows}, name)

    frame_energy = embedding.frame_energies(samples, sr, config)
    noise_floor = embedding.estimate_noise_floor(frame_energy,
                                                 config["noise_floor_frames"])
    gate = noise_floor * float(config.get("vad_energy_ratio", 8.0)) + 1e-12

    # ---- stage: embedding (parallel over windows, chunked commits) ----
    interval = max(1, int(config.get("snapshot_commit_interval_windows", 8)))

    def embed_one(window):
        try:
            emb, _, _ = embedding.window_embedding(
                audio_io.extract_window(samples, window), sr, config,
                noise_floor=noise_floor)
        except StageFailure:
            raise
        except Exception as exc:
            raise StageFailure("E_EMBEDDING", "embedding",
                               window["window_index"], str(exc))
        return window["window_index"], emb

    embs = [None] * len(windows)
    done = 0
    for start in range(0, len(windows), interval):
        chunk = windows[start:start + interval]
        for idx, emb in _map(pool, embed_one, chunk):
            embs[idx] = emb
        done += len(chunk)
        snap.commit("embedding", {"file": name, "completed_windows": done,
                                  "total_windows": len(windows)}, name)

    flen = sr * int(config["frame_ms"]) // 1000
    fhop = sr * int(config["frame_hop_ms"]) // 1000
    speech_windows = []
    non_speech = []
    voiced_gate = 4.0 * noise_floor + 1e-9
    for w in windows:
        f0 = w["start_sample"] // fhop
        f1 = min(len(frame_energy), max(f0, (w["end_sample"] - flen) // fhop + 1))
        if f1 > f0:
            voiced_fraction = float(np.mean(frame_energy[f0:f1] > voiced_gate))
        else:
            voiced_fraction = 0.0
        (speech_windows if voiced_fraction >= 0.5 else non_speech).append(
            w["window_index"])
    rec["speech_windows"] = speech_windows
    rec["non_speech_windows"] = non_speech

    widths = [int(config["frac_bits"]), int(config["frac_bits_high"])]
    per_width = {}
    for width in widths:
        quant = [embedding.quantize(e, width, int(config["quant_levels"]))
                 for e in embs]
        deq = {i: quant[i]["dequantized"] for i in range(len(quant))}
        quant_records = [{
            "window_index": i,
            "step": quant[i]["step"],
            "quantized": [int(v) for v in quant[i]["quantized"]],
            "residuals": [float(v) for v in quant[i]["residual"]],
        } for i in range(len(quant))]

        # ---- stage: similarity (parallel over adjacent window pairs) ----
        def sim_one(i):
            return i, clustering.similarity(deq[i], deq[i + 1])

        sims = dict(_map(pool, sim_one, range(max(0, len(windows) - 1))))

        # ---- stage: clustering ----
        try:
            speech_embs = {i: deq[i] for i in speech_windows}
            clus = clustering.cluster_windows(
                speech_embs,
                float(config["similarity_threshold"]),
                float(config["merge_hysteresis"]))
        except StageFailure:
            raise
        except Exception as exc:
            raise StageFailure("E_CLUSTERING", "clustering", None, str(exc))
        snap.commit("clustering", {
            "file": name, "frac_bits": width,
            "assignments": {str(k): v for k, v in sorted(clus["assignments"].items())},
            "unassigned": clus["unassigned"],
        }, name)

        # ---- stage: segmentation ----
        try:
            segs = segmentation.build_segments(
                clus["assignments"], windows, frame_energy, len(samples), sr, config)
        except StageFailure:
            raise
        except Exception as exc:
            raise StageFailure("E_SEGMENTATION", "segmentation", None, str(exc))
        snap.commit("segmentation", {"file": name, "frac_bits": width,
                                     "segments": segs}, name)

        per_width[width] = {
            "dequantized": deq,
            "similarities": sims,
            "assignments": clus["assignments"],
            "segments": segs,
            "unassigned": clus["unassigned"],
            "clusters": clus["clusters"],
            "merge_events": clus["merge_events"],
            "merge_boundary_events": clus["merge_boundary_events"],
            "hysteresis_events": clus["hysteresis_events"],
            "decision_margins": clus["decision_margins"],
            "quantization": quant_records,
        }

    low = per_width[widths[0]]
    high = per_width[widths[1]]
    trk = tracking.compare_widths(low, high, config)
    rec["frac_bits_low"] = widths[0]
    rec["frac_bits_high"] = widths[1]
    rec["quantization_low"] = low["quantization"]
    rec["similarities_low"] = [
        {"pair": [i, i + 1], "similarity": low["similarities"][i]}
        for i in sorted(low["similarities"])
    ]
    rec["unassigned_windows_low"] = low["unassigned"]
    rec["unassigned_windows_high"] = high["unassigned"]
    rec["hysteresis_events_low"] = low["hysteresis_events"]
    rec["merge_events_low"] = low["merge_events"]
    rec["segments_low"] = low["segments"]
    rec["segments_high"] = high["segments"]
    rec["speaker_count_low"] = len({v for v in low["assignments"].values()})
    rec["speaker_count_high"] = len({v for v in high["assignments"].values()})
    rec["error_tracking"] = {
        "stages": trk["stages"],
        "first_exceed_stage": trk["first_exceed_stage"],
        "tolerance": trk["tolerance"],
    }
    rec["flip_attribution"] = trk["flip"]
    return rec


def run_pipeline(samples_dir, config, inj_manifest, out_dir, serial=False):
    os.makedirs(out_dir, exist_ok=True)
    config_hash = _hash(config)
    injection_hash = _hash(inj_manifest)
    snap = SnapshotManager(out_dir, config_hash, injection_hash,
                           int(config["frac_bits"]))
    if snap.recovered is not None:
        LOG.info("recovered snapshot state: %s",
                 json.dumps(snap.recovered)[:200])
    inj_entries = inj_manifest.get("injections", [])
    files = sorted(f for f in os.listdir(samples_dir) if f.lower().endswith(".wav"))
    max_workers = max(1, int(config.get("max_workers", 1)))
    pool = None if (serial or max_workers <= 1) else \
        ThreadPoolExecutor(max_workers=max_workers)

    stop_event = threading.Event()
    old_handlers = {}

    def _handler(signum, frame):
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            old_handlers[sig] = signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass

    records = []
    failures = []
    hits_all = []
    graceful = False
    try:
        for name in files:
            path = os.path.join(samples_dir, name)
            try:
                rec = process_audio(path, config, inj_entries, pool, snap)
                records.append(rec)
                hits_all.extend(rec.get("injection_hits", []))
            except StageFailure as exc:
                failures.append({
                    "file": name,
                    "stage": exc.stage,
                    "window_index": exc.window_index,
                    "code": exc.code,
                    "message": exc.message,
                })
                LOG.error("%s: %s at stage %s", name, exc.code, exc.stage)
            except Exception as exc:  # keep processing the remaining audios
                failures.append({
                    "file": name,
                    "stage": "unknown",
                    "window_index": None,
                    "code": ERROR_INTERNAL,
                    "message": str(exc),
                })
                LOG.exception("%s: unexpected error", name)
            if stop_event.is_set():
                graceful = True
                LOG.info("stop requested; exiting after completing %s", name)
                break
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    report = {
        "config_hash": config_hash,
        "injection_hash": injection_hash,
        "frac_bits_low": int(config["frac_bits"]),
        "frac_bits_high": int(config["frac_bits_high"]),
        "max_workers": max_workers,
        "serial": bool(serial),
        "graceful_stop": graceful,
        "snapshot": snap.final_manifest(),
        "recovered_snapshot": snap.recovered,
        "audios": records,
        "injection_hits": hits_all,
        "failures": failures,
        "exit_code": 0,
    }
    tmp = os.path.join(out_dir, "report.json.tmp")
    final = os.path.join(out_dir, "report.json")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, final)
    return 0
