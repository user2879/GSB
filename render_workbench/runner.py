"""Batch orchestration, signal handling and JSON diagnostic reports."""

import json
import os
import signal
import threading
import time

import numpy as np

from .config import load_prompts, load_yaml, validate_config
from .edits import EditPlanner, load_plan
from .errors import E_CONFIG_INVALID
from .hashing import bytes_fingerprint, fingerprint
from .config import global_semantic_payload, tier_payload
from .pipeline import Pipeline, StopRequested
from .png_codec import decode_png


class GracefulStop:
    """Thread-safe stop flag wired to SIGINT/SIGTERM (POSIX)."""

    def __init__(self):
        self.event = threading.Event()
        self._signal_name = None
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._handler)
            signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, _frame):
        self._signal_name = signal.Signals(signum).name
        self.event.set()

    @property
    def signal_name(self):
        return self._signal_name


def run_workflow(config_path, prompts_path, plan_path, workdir,
                 mode="incremental", baseline_concurrency=1, stop_gate=None):
    """Execute the full batch workflow.

    ``mode`` is ``incremental`` (default), ``full`` (empty fresh workdir) or
    ``baseline`` (forced single-thread serial run). ``stop_gate`` is an
    optional callback invoked after each wave; raising SystemExit simulates a
    hard termination before the next snapshot.
    """
    raw_cfg = load_yaml(config_path)
    cfg, config_errors = validate_config(raw_cfg)
    prompts, prompt_errors = load_prompts(prompts_path, cfg)
    plan = load_plan(plan_path) if plan_path else {"batches": []}

    if mode == "full":
        _reset_workdir(workdir)
    if mode == "baseline":
        cfg = dict(cfg)
        cfg["concurrency"] = baseline_concurrency

    stop = GracefulStop()
    pipeline = Pipeline(cfg, workdir, stop_event=stop.event)
    dropped_on_recovery = pipeline.store.recover()

    report = _new_report(mode, pipeline, config_errors + prompt_errors,
                         dropped_on_recovery)
    planner = EditPlanner(cfg, prompts, os.path.dirname(
        os.path.abspath(prompts_path)))

    failures = list(config_errors) + list(prompt_errors)

    # Baseline render of the pristine inputs before any edit, so an empty
    # plan still produces a complete, auditable set of artifacts.
    baseline_record = {
        "batch_index": -1,
        "batch_id": "baseline",
        "failures": [],
        "recompute": [],
        "reuse": [],
        "recomputed_nodes": 0,
        "elapsed_seconds": None,
        "status": "applied",
    }
    baseline_start = time.perf_counter()
    try:
        _refresh_fingerprints(pipeline, planner.cfg)
        pipeline.execute(planner.prompt_list())
    except StopRequested:
        baseline_record["status"] = "stopped"
        baseline_record["recompute"] = pipeline.recompute_records
        baseline_record["reuse"] = pipeline.reuse_records
        baseline_record["recomputed_nodes"] = pipeline.computed_count
        baseline_record["elapsed_seconds"] = round(
            time.perf_counter() - baseline_start, 6)
        report["batches"].append(baseline_record)
        report["status"] = "stopped"
        report["stop_signal"] = stop.signal_name
        report["resume_version"] = pipeline.store.manifest["version"]
        _write_report(workdir, report)
        return report, 0
    baseline_record["recompute"] = pipeline.recompute_records
    baseline_record["reuse"] = pipeline.reuse_records
    baseline_record["recomputed_nodes"] = pipeline.computed_count
    baseline_record["elapsed_seconds"] = round(
        time.perf_counter() - baseline_start, 6)
    report["batches"].append(baseline_record)

    for batch_index, batch in enumerate(plan.get("batches", [])):
        batch_start = time.perf_counter()
        batch_errors = planner.apply_batch(batch)
        batch_record = {
            "batch_index": batch_index,
            "batch_id": batch.get("id"),
            "failures": batch_errors,
            "recompute": [],
            "reuse": [],
            "recomputed_nodes": 0,
            "elapsed_seconds": None,
            "status": "failed" if batch_errors else "applied",
        }
        if not batch_errors:
            try:
                _refresh_fingerprints(pipeline, planner.cfg)
                pipeline.computed_count = 0
                pipeline.reuse_records = []
                pipeline.recompute_records = []
                pipeline.path_segments = []
                pipeline.execute(planner.prompt_list())
            except StopRequested:
                batch_record["status"] = "stopped"
                batch_record["recompute"] = pipeline.recompute_records
                batch_record["reuse"] = pipeline.reuse_records
                batch_record["recomputed_nodes"] = pipeline.computed_count
                batch_record["elapsed_seconds"] = round(
                    time.perf_counter() - batch_start, 6)
                report["batches"].append(batch_record)
                report["status"] = "stopped"
                report["stop_signal"] = stop.signal_name
                report["resume_version"] = pipeline.store.manifest["version"]
                _write_report(workdir, report)
                return report, 0
            batch_record["recompute"] = pipeline.recompute_records
            batch_record["reuse"] = pipeline.reuse_records
            batch_record["recomputed_nodes"] = pipeline.computed_count
        else:
            failures.extend(batch_errors)
        batch_record["elapsed_seconds"] = round(
            time.perf_counter() - batch_start, 6)
        report["batches"].append(batch_record)
        if stop_gate is not None:
            stop_gate(batch_index, pipeline)

    report["status"] = "completed"
    report["failures"] = failures
    report["total_recomputed_nodes"] = sum(
        batch["recomputed_nodes"] for batch in report["batches"])
    report["resume_version"] = pipeline.store.manifest["version"]
    report["cursors"] = pipeline.store.manifest.get("cursors", {})
    report["path_segments"] = pipeline.path_segments
    report["shared_segment_verification"] = \
        pipeline.verify_shared_segments(planner.prompt_list())
    report["final_prompts"] = [
        {"id": prompt["id"],
         "tokens": prompt["tokens"],
         "init_image": prompt.get("init_image"),
         "patches": prompt.get("patches", [])}
        for prompt in planner.prompt_list()
    ]
    _write_report(workdir, report)
    return report, 0


def _config_payload(cfg):
    from .config import config_fingerprint_payload
    return config_fingerprint_payload(cfg)


def _refresh_fingerprints(pipeline, cfg):
    """Recompute full, global and per-tier fingerprints after an edit."""
    pipeline.cfg = cfg
    pipeline.config_fp = fingerprint(_config_payload(cfg))
    pipeline.global_fp = fingerprint(global_semantic_payload(cfg))
    pipeline.tier_fps = {
        tier["id"]: fingerprint(tier_payload(tier))
        for tier in cfg["tiers"]
    }
    pipeline.store.manifest["config_fp"] = pipeline.config_fp


def _reset_workdir(workdir):
    if os.path.isdir(workdir):
        for name in os.listdir(workdir):
            path = os.path.join(workdir, name)
            if os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
            else:
                os.remove(path)
    os.makedirs(workdir, exist_ok=True)


def _new_report(mode, pipeline, startup_failures, recovered):
    return {
        "schema_version": 1,
        "mode": mode,
        "status": "started",
        "config_fp": pipeline.config_fp,
        "startup_failures": startup_failures,
        "recovery": {
            "recovered_dropped_nodes": recovered,
            "resume_version": pipeline.store.manifest.get("version", 0),
        },
        "batches": [],
        "failures": [],
        "total_recomputed_nodes": 0,
        "resume_version": 0,
    }


def _write_report(workdir, report):
    path = os.path.join(workdir, "report.json")
    tmp = os.path.join(workdir, "report.json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, ensure_ascii=False,
                  indent=2, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError("not JSON serializable: %r" % type(value))
