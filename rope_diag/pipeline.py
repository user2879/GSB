"""Staged diagnostic pipeline with atomic snapshot commits.

Stages: load_validate -> position_encoding -> scoring -> error_trace.

Each stage's artifacts are written into a temporary directory first and then
committed into the snapshot directory with ``os.replace`` (both directories
live on the same local disk, so the replace is atomic). The manifest records
the global version, config hash, injection-manifest hash and the completed
scheme/sample lists.

Concurrency: tasks carry a stable sort key (scheme, sample_id, stage). A
thread pool may execute them in any order, but results are always assembled
in sort-key order, so concurrent output is identical to serial execution.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import scheduler as sched
from .error_trace import trace_sample
from .fingerprints import (
    cache_key,
    pruning_fingerprint,
    sample_fingerprint,
    stable_hash,
)
from .schemes import SchemeFault, SchemeParams, apply_scheme
from .scoring import compute_scores, merge_stratified, stratified_stats

STAGES = ("load_validate", "position_encoding", "scoring", "error_trace")
STAGE_ORDER = {name: i for i, name in enumerate(STAGES)}


def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


class Pipeline:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.tmp_dir = self.output_dir / "stage_tmp"
        self.snap_dir = self.output_dir / "snapshots"
        for stage in STAGES:
            (self.tmp_dir / stage).mkdir(parents=True, exist_ok=True)
            (self.snap_dir / stage).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # manifest handling
    # ------------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.snap_dir / "manifest.json"

    def load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"global_version": 0, "entries": []}

    # ------------------------------------------------------------------
    # stage computations
    # ------------------------------------------------------------------
    def _run_unit_stage(self, task: dict) -> dict:
        """Compute one (scheme, sample, stage) task; returns the artifact."""
        stage = task["stage"]
        sample = task["sample"]
        config = task["config"]
        injections = task["injections"]
        policy = task["policy"]

        if stage == "load_validate":
            mutated = sched.apply_load_injections(sample, injections)
            sched.validate_sample(mutated, int(config["target_len"]), injections)
            return {"sample": mutated, "sample_fp": sample_fingerprint(sample)}

        params_dict = {
            "scheme": task["scheme"],
            "base": float(config["base"]),
            "scale": float(config["scale"]),
            "dim": int(config["dim"]),
            "train_len": int(config["train_len"]),
            "target_len": int(config["target_len"]),
        }
        params_dict = sched.apply_param_injections(params_dict, injections)
        params = SchemeParams(**params_dict)
        fault_by_type = {inj["fault"]: inj for inj in injections}
        try:
            if stage == "position_encoding":
                out = apply_scheme(
                    params,
                    np.asarray(task["validated"]["query"], dtype=np.float64),
                    np.asarray(task["validated"]["keys"], dtype=np.float64),
                    int(task["validated"]["query_pos"]),
                    np.asarray(task["validated"]["key_positions"], dtype=np.float64),
                )
                return {
                    "scheme": out.scheme,
                    "param_fingerprint": out.param_fingerprint,
                    "rotated_query": out.rotated_query,
                    "rotated_keys": out.rotated_keys,
                    "query_embedding": out.query_embedding,
                    "key_embeddings": out.key_embeddings,
                    "boundary_flags": out.boundary_flags,
                }
            if stage == "scoring":
                enc = task["encoding"]
                dim_range = tuple(policy["dim_range"])
                scores = compute_scores(
                    np.asarray(enc["rotated_query"], dtype=np.float64),
                    np.asarray(enc["rotated_keys"], dtype=np.float64),
                    dim_range,
                    int(policy["score_chunk"]),
                )
                return {
                    "scores": scores,
                    "stratified": stratified_stats(
                        scores,
                        int(task["validated"]["query_pos"]),
                        [int(p) for p in task["validated"]["key_positions"]],
                        int(config["train_len"]),
                    ),
                }
            # error_trace
            enc = task["encoding"]
            validated = task["validated"]
            return trace_sample(
                params,
                validated["query"],
                validated["keys"],
                int(validated["query_pos"]),
                [int(p) for p in validated["key_positions"]],
                np.asarray(enc["rotated_query"], dtype=np.float64),
                np.asarray(enc["rotated_keys"], dtype=np.float64),
                np.asarray(task["scoring"]["scores"], dtype=np.float64),
                float(config["error_tolerance"]),
                tuple(policy["dim_range"]),
            )
        except SchemeFault as exc:
            inj = fault_by_type.get("base_zero" if exc.code == "E_BASE_ZERO" else "scale_nonpositive")
            raise sched.InjectedFault(exc.code, inj["injection_id"] if inj else "NONE", str(exc))

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def run(
        self,
        config: dict,
        samples: list[dict],
        injection_manifest: dict,
        policy: dict,
        decision_record: dict,
        mode: str = "full",
    ) -> dict:
        prev_manifest = self.load_manifest()
        prev_keys = {e["key"] for e in prev_manifest.get("entries", [])}
        config_hash = stable_hash(config)
        injection_hash = stable_hash(injection_manifest)
        pruning_fp = pruning_fingerprint(policy)
        precision = str(config["precision"])
        concurrency = max(1, int(config.get("concurrency", 1)))
        commit_interval = max(1, int(config.get("snapshot_commit_interval", 1)))

        units = [
            sched.ScheduledUnit(scheme=s, sample_id=sm["sample_id"], points=len(sm["key_positions"]))
            for s in config["schemes"]
            for sm in samples
        ]
        plan = sched.schedule(units, int(config["point_budget"]))
        admitted = {(u.scheme, u.sample_id) for u in plan["admitted"]}
        sample_by_id = {sm["sample_id"]: sm for sm in samples}

        reused: list[dict] = []
        recomputed: list[dict] = []
        failures: list[dict] = []
        artifacts: dict[tuple, dict] = {}
        new_entries: list[dict] = []
        failed_units: set[tuple] = set()

        def unit_sort_key(scheme: str, sample_id: str, stage: str):
            return (scheme, sample_id, STAGE_ORDER[stage])

        for stage in STAGES:
            tasks = []
            for scheme, sample_id in sorted(admitted):
                if (scheme, sample_id) in failed_units:
                    continue
                sample = sample_by_id[sample_id]
                stage_injections = sched.injections_for(injection_manifest, sample_id, stage)
                params_dict = {
                    "scheme": scheme,
                    "base": float(config["base"]),
                    "scale": float(config["scale"]),
                    "dim": int(config["dim"]),
                    "train_len": int(config["train_len"]),
                    "target_len": int(config["target_len"]),
                }
                sample_fp = sample_fingerprint(sample)
                if stage == "load_validate":
                    # load/validation depends only on the sample content, the
                    # validation bound and this stage's injections -- not on
                    # scheme params, precision or pruning.
                    key = stable_hash(
                        {
                            "sample": sample_fp,
                            "target_len": int(config["target_len"]),
                            "injections": stage_injections,
                            "stage": stage,
                        }
                    )
                else:
                    key = cache_key(params_dict, sample_fp, precision, pruning_fp, stage)
                    key = stable_hash({"key": key, "injections": stage_injections})
                    if stage == "error_trace":
                        key = stable_hash(
                            {
                                "key": key,
                                "error_tolerance": repr(float(config["error_tolerance"])),
                            }
                        )
                entry = {
                    "key": key,
                    "scheme": scheme,
                    "sample_id": sample_id,
                    "stage": stage,
                }
                artifact_path = self.snap_dir / stage / f"{key}.json"
                # Content-addressed reuse: any persisted artifact with the
                # same derived key is valid, regardless of which global
                # version wrote it.
                if mode == "incremental" and artifact_path.exists():
                    artifacts[(scheme, sample_id, stage)] = json.loads(
                        artifact_path.read_text(encoding="utf-8")
                    )
                    new_entries.append(entry)
                    reused.append(entry)
                    continue
                tasks.append(
                    {
                        "scheme": scheme,
                        "sample_id": sample_id,
                        "stage": stage,
                        "key": key,
                        "entry": entry,
                        "sample": sample,
                        "config": config,
                        "policy": policy,
                        "injections": stage_injections,
                        "validated": artifacts.get((scheme, sample_id, "load_validate"), {}).get("sample"),
                        "encoding": artifacts.get((scheme, sample_id, "position_encoding")),
                        "scoring": artifacts.get((scheme, sample_id, "scoring")),
                    }
                )

            def execute(task):
                try:
                    artifact = self._run_unit_stage(task)
                    return {"task": task, "artifact": artifact, "failure": None}
                except sched.InjectedFault as exc:
                    return {
                        "task": task,
                        "artifact": None,
                        "failure": {
                            "scheme": task["scheme"],
                            "sample_id": task["sample_id"],
                            "stage": task["stage"],
                            "code": exc.code,
                            "injection_id": exc.injection_id,
                            "message": str(exc),
                        },
                    }

            if concurrency > 1 and len(tasks) > 1:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    results = list(pool.map(execute, tasks))
            else:
                results = [execute(t) for t in tasks]
            results.sort(
                key=lambda r: unit_sort_key(
                    r["task"]["scheme"], r["task"]["sample_id"], r["task"]["stage"]
                )
            )

            # stage outputs go to the tmp dir first, then commit atomically
            stage_tmp = self.tmp_dir / stage
            staged_files: list[tuple[Path, Path]] = []
            staged_seen: set[Path] = set()
            for res in results:
                task = res["task"]
                ident = (task["scheme"], task["sample_id"])
                if res["failure"] is not None:
                    failures.append(res["failure"])
                    failed_units.add(ident)
                    continue
                artifact = res["artifact"]
                artifacts[(task["scheme"], task["sample_id"], stage)] = artifact
                new_entries.append(task["entry"])
                recomputed.append(task["entry"])
                tmp_path = stage_tmp / f"{task['key']}.json"
                if tmp_path not in staged_seen:
                    staged_seen.add(tmp_path)
                    _write_json_atomic(tmp_path, _to_jsonable(artifact))
                    staged_files.append(
                        (tmp_path, self.snap_dir / stage / f"{task['key']}.json")
                    )

            if STAGE_ORDER[stage] % commit_interval == 0 or stage == STAGES[-1]:
                for tmp_path, final_path in staged_files:
                    os.replace(tmp_path, final_path)
                staged_files.clear()

        # propagate per-unit artifacts across stages for dependent stages
        # (handled above through the shared ``artifacts`` dict)

        invalidated = sorted(prev_keys - {e["key"] for e in new_entries})
        reused.sort(key=lambda e: unit_sort_key(e["scheme"], e["sample_id"], e["stage"]))
        recomputed.sort(key=lambda e: unit_sort_key(e["scheme"], e["sample_id"], e["stage"]))
        failures.sort(key=lambda f: unit_sort_key(f["scheme"], f["sample_id"], f["stage"]))

        manifest = {
            "global_version": int(prev_manifest.get("global_version", 0)) + 1,
            "config_hash": config_hash,
            "injection_hash": injection_hash,
            "pruning_fingerprint": pruning_fp,
            "precision": precision,
            "completed": {
                "schemes": sorted({e["scheme"] for e in new_entries}),
                "samples": sorted({e["sample_id"] for e in new_entries}),
            },
            "entries": sorted(
                new_entries,
                key=lambda e: unit_sort_key(e["scheme"], e["sample_id"], e["stage"]),
            ),
        }
        _write_json_atomic(self.manifest_path, manifest)

        # scheme-level aggregates
        stratified_by_scheme: dict[str, dict] = {}
        traces_by_scheme: dict[str, dict] = {}
        for scheme in config["schemes"]:
            parts = []
            traces = {}
            for sample_id in sorted(sample_by_id):
                art = artifacts.get((scheme, sample_id, "scoring"))
                if art:
                    parts.append(art["stratified"])
                tr = artifacts.get((scheme, sample_id, "error_trace"))
                if tr:
                    traces[sample_id] = tr
            stratified_by_scheme[scheme] = merge_stratified(parts)
            traces_by_scheme[scheme] = traces

        return {
            "mode": mode,
            "manifest": manifest,
            "stratified": stratified_by_scheme,
            "traces": traces_by_scheme,
            "artifacts": artifacts,
            "budget": {
                "point_budget": plan["point_budget"],
                "spent_points": plan["spent_points"],
                "pending": plan["pending"],
            },
            "reused": reused,
            "recomputed": recomputed,
            "invalidated_keys": invalidated,
            "failures": failures,
            "decision_record": decision_record,
        }
