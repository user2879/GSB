"""Pipeline orchestration: load -> generate -> preprocess -> schedule ->
evaluate -> diagnostics, with staged snapshot commits and exact caching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import cache as cache_mod
from .cache import ArtifactCache
from . import config_loader, datagen, preprocess as prep_mod
from .clustering import kmeans_train, dbscan_train, gmm_train
from .diagnostics import build_diagnostics
from .errors import (FailureLog, E_SNAPSHOT_MISMATCH, E_CANDIDATE_FAULT,
                     E_INJECTED_FAULT, E_CONFIG_INVALID)
from .fingerprint import fingerprint
from .metrics import evaluate_candidate
from .scheduler import build_candidates, run_schedule, Candidate
from .snapshot import SnapshotManager


class InjectedFault(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fault_injections(injections: list[dict], stage: str,
                      cand: Candidate) -> dict | None:
    for inj in injections:
        if inj.get("type") != "candidate_fault":
            continue
        if inj.get("stage") != stage:
            continue
        if inj.get("method") not in (None, cand.method):
            continue
        want = inj.get("params")
        if want and any(cand.params.get(k) != v for k, v in want.items()):
            continue
        return inj
    return None


def _run_candidate(cand: Candidate, X: np.ndarray, cfg) -> tuple[dict, int]:
    p = cand.params
    if cand.method == "kmeans":
        out = kmeans_train(X, p["n_clusters"], max_iter=cfg.kmeans_max_iter,
                           quota=cfg.candidate_quota)
    elif cand.method == "dbscan":
        out = dbscan_train(X, p["eps"], p["min_samples"],
                           quota=cfg.candidate_quota)
    elif cand.method == "gmm":
        out = gmm_train(X, p["n_components"], p["covariance_type"],
                        reg_covar=cfg.gmm_reg_covar, tol=cfg.gmm_tol,
                        max_iter=cfg.gmm_max_iter, quota=cfg.candidate_quota)
    else:  # pragma: no cover
        raise ValueError(f"unknown method {cand.method}")
    return out, int(out["iterations"])


def run(config_path: str, data_path: str, injections_path: str | None,
        workdir: str) -> dict[str, Any]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    failures = FailureLog()

    # --- load ---------------------------------------------------------------
    injections, injection_hash = config_loader.load_injections(
        injections_path, failures)
    desc = config_loader.load_data_description(data_path, failures)
    cfg = config_loader.load_experiment_config(config_path, failures)
    if desc is None or cfg is None:
        failures.add(E_CONFIG_INVALID, "cannot proceed without valid config "
                                       "and data description", stage="load")
        return {"failures": failures.to_list(), "status": "aborted"}

    # --- snapshot resume check ----------------------------------------------
    snaps = SnapshotManager(workdir / "snapshots")
    reuse_ok, old_manifest = snaps.check_resume(cfg.config_hash, injection_hash)
    if not reuse_ok:
        failures.add(E_SNAPSHOT_MISMATCH,
                     "snapshot config/injection hash mismatch; old products "
                     "are rejected and the run restarts fresh",
                     stage="snapshot",
                     old_version=(old_manifest or {}).get("version"))
        manifest = snaps.new_manifest(cfg.config_hash, injection_hash)
    else:
        manifest = old_manifest or snaps.new_manifest(cfg.config_hash,
                                                      injection_hash)

    cache = ArtifactCache(workdir / "cache")

    # --- data generation (+ sample replacement injections) ------------------
    data = datagen.generate(desc, cfg.sample_scale)
    applied_replacements = datagen.apply_sample_replacements(data, injections)
    data_fp = datagen.data_fingerprint(desc, data)

    # --- preprocess stage ----------------------------------------------------
    pre_key = cache_mod.preprocess_key(data_fp, fingerprint(
        "preprocess-params", cfg.preprocess_method, cfg.preprocess_params))
    pre_artifact = cache.get(pre_key)
    pre_cache_hit = pre_artifact is not None
    if pre_artifact is None:
        pre = prep_mod.preprocess(desc, cfg, data, data_fp)
        pre_artifact = {
            "matrix": pre["matrix"].tolist(),
            "feature_names": pre["feature_names"],
            "stats": pre["stats"],
            "n_train": pre["n_train"],
            "method": pre["method"],
            "tie_rule": pre["tie_rule"],
            "params_fingerprint": pre["params_fingerprint"],
            "content_fingerprint": pre["content_fingerprint"],
        }
        cache.put(pre_key, pre_artifact)
    X = np.array(pre_artifact["matrix"], dtype=np.float64)
    snaps.commit_stage("preprocess", {"preprocess": pre_artifact}, manifest)

    # --- cluster stage (parallel compute + deterministic ledger) -------------
    candidates = build_candidates(cfg.methods)
    cluster_keys = {c.stable_key: cache_mod.cluster_key(pre_key, c.method,
                                                        c.params)
                    for c in candidates}
    cache_hits: dict[str, bool] = {}

    def compute_fn(cand: Candidate) -> tuple[dict, int]:
        inj = _fault_injections(injections, "preprocess", cand)
        if inj is not None:
            raise InjectedFault(inj.get("code", E_INJECTED_FAULT),
                                f"injected fault at preprocess: {inj.get('id')}")
        ck = cluster_keys[cand.stable_key]
        cached = cache.get(ck)
        if cached is not None:
            cache_hits[cand.stable_key] = True
            return cached["output"], cached["iterations"]
        inj = _fault_injections(injections, "iterate", cand)
        if inj is not None:
            raise InjectedFault(inj.get("code", E_INJECTED_FAULT),
                                f"injected fault at iterate: {inj.get('id')}")
        cache_hits[cand.stable_key] = False
        out, iters = _run_candidate(cand, X, cfg)
        cache.put(ck, {"output": out, "iterations": iters,
                       "meta": {"preprocess_key": pre_key,
                                "method": cand.method,
                                "params": cand.params}})
        return out, iters

    completed: list[str] = list(manifest.get("completed_candidates", []))

    def on_committed(idx: int, res, consumed: int) -> None:
        completed.append(res.candidate.short_key)
        if (idx + 1) % cfg.snapshot_interval == 0:
            manifest["completed_candidates"] = list(completed)
            manifest["consumed_budget"] = consumed
            snaps.commit_manifest(manifest)

    report = run_schedule(candidates, compute_fn, cfg.global_budget,
                          cfg.candidate_quota, cfg.concurrency, failures,
                          on_committed=on_committed)
    manifest["completed_candidates"] = list(completed)
    manifest["consumed_budget"] = report.consumed

    cluster_artifacts: dict[str, Any] = {}
    for res in report.results:
        ck = cluster_keys[res.candidate.stable_key]
        res.cache_hit = cache_hits.get(res.candidate.stable_key, True)
        cluster_artifacts[res.candidate.short_key] = {
            "cache_key": ck,
            "preprocess_key": pre_key,
            "method": res.candidate.method,
            "params": res.candidate.params,
            "status": res.status,
            "iterations": res.iterations,
            "output": res.output,
            "error_code": res.error_code,
        }
    snaps.commit_stage("cluster", cluster_artifacts, manifest)

    # --- evaluate stage -------------------------------------------------------
    eval_entries: dict[str, Any] = {}
    eval_keys: list[str] = []
    for res in report.results:
        if res.status != "ok" or res.output is None:
            continue
        ck = cluster_keys[res.candidate.stable_key]
        ek = cache_mod.evaluate_key(ck, cfg.metrics, cfg.gmm_reg_covar)
        eval_keys.append(ek)
        entry = cache.get(ek)
        if entry is None:
            try:
                inj = _fault_injections(injections, "evaluate", res.candidate)
                if inj is not None:
                    raise InjectedFault(inj.get("code", E_INJECTED_FAULT),
                                        f"injected fault at evaluate: "
                                        f"{inj.get('id')}")
                labels = np.array(res.output["labels"], dtype=np.int64)
                model_ll = res.output.get("log_likelihood") \
                    if res.candidate.method == "gmm" else None
                entry = evaluate_candidate(X, labels, cfg.metrics,
                                           cfg.gmm_reg_covar,
                                           model_log_likelihood=model_ll)
                cache.put(ek, entry)
            except BaseException as exc:
                code = getattr(exc, "code", E_CANDIDATE_FAULT)
                res.status = "fault"
                res.error_code = code
                res.error_stage = "evaluate"
                failures.add(code, f"{type(exc).__name__}: {exc}",
                             stage="evaluate",
                             candidate_key=res.candidate.short_key)
                continue
        eval_entries[res.candidate.short_key] = entry
    snaps.commit_stage("evaluate", eval_entries, manifest)

    # --- diagnostics stage ----------------------------------------------------
    candidate_entries = []
    for res in report.results:  # stable-key order: schedule-independent
        metrics = (eval_entries.get(res.candidate.short_key) or {}).get(
            "metrics", {})
        from .fingerprint import _normalize
        metrics = _normalize(metrics)
        candidate_entries.append({
            "short_key": res.candidate.short_key,
            "method": res.candidate.method,
            "params": res.candidate.params,
            "status": res.status,
            "iterations": res.iterations,
            "cache_hit": res.cache_hit,
            "metrics": metrics,
            "events": (res.output or {}).get("events", []),
            "error_code": res.error_code,
            "error_stage": res.error_stage,
        })
    budget = {
        "global_iterations": cfg.global_budget,
        "candidate_quota": cfg.candidate_quota,
        "consumed": report.consumed,
        "remaining": report.remaining,
        "ledger": report.ledger,
        "todo": report.todo,
    }
    diag_key = cache_mod.diagnostics_key(eval_keys, {
        "config_hash": cfg.config_hash, "injection_hash": injection_hash})
    diagnostics = build_diagnostics(
        experiment=cfg.name, seed=cfg.seed, sample_scale=cfg.sample_scale,
        n_rows=int(X.shape[0]), n_features=int(X.shape[1]),
        config_hash=cfg.config_hash, injection_hash=injection_hash,
        data_fingerprint=data_fp,
        preprocess_fingerprint=pre_artifact["content_fingerprint"],
        preprocess_meta={"method": pre_artifact["method"],
                         "tie_rule": pre_artifact["tie_rule"],
                         "n_train": pre_artifact["n_train"],
                         "cache_hit": pre_cache_hit},
        candidate_entries=candidate_entries,
        primary_metric=cfg.metrics[0],
        budget=budget,
        failures=failures.to_list(),
        cache_stats=cache.stats(),
        manifest_version=int(manifest.get("version", 0)),
        diagnostics_fingerprint=diag_key,
    )
    diagnostics["applied_injections"] = applied_replacements
    snaps.commit_stage("diagnostics", {"summary": diagnostics}, manifest)

    out_dir = workdir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    from .fingerprint import _normalize
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(_normalize(diagnostics), fh, sort_keys=True,
                  ensure_ascii=True, indent=1)
    return diagnostics
