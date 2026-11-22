"""End-to-end pipeline: validate -> train -> merge -> diagnose -> export.

Determinism rules:
  * tasks are processed in task-name order everywhere;
  * concurrent training only parallelizes pure per-task work and results are
    re-sorted by the stable (task, layer) key, so output equals serial runs;
  * every stage commits via whole-directory atomic replacement;
  * resume decisions use the manifest's version, config hash and base
    fingerprint plus per-adapter keys.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import errors
from .basepack import load_base_pack, replace_layer
from .config import load_config
from .diagnose import diagnose_task
from .fixedpoint import quantize
from .merge import check_reversible, merge_layer, schedule_merges
from .store import Store, canonical_json
from .tasks import (
    load_samples,
    load_tasks,
    load_teacher,
    sample_input,
    teacher_delta_matrix,
)
from .train import TrainedAdapter, adapter_key, train_adapter_fixed

STAGES = ("validate", "train", "merge", "export")


def _adapter_meta(adapter: TrainedAdapter):
    return {
        "task_name": adapter.task_name,
        "target_layer": adapter.target_layer,
        "rank": adapter.rank,
        "alpha": adapter.alpha,
        "frac_bits": adapter.frac_bits,
        "losses_fixed": adapter.losses_fixed,
        "losses": adapter.losses,
        "ledger": adapter.ledger,
        "base_fingerprint": adapter.base_fingerprint,
        "layer_fingerprint": adapter.layer_fingerprint,
        "key": adapter.key,
    }


def _export_adapter_files(adapter: TrainedAdapter):
    return {
        f"{adapter.key}.A.npy": {"npy": adapter.A},
        f"{adapter.key}.B.npy": {"npy": adapter.B},
        f"{adapter.key}.json": {"json": _adapter_meta(adapter)},
    }


def _load_adapter(store: Store, key: str):
    meta = store.read_stage_file("train", f"{key}.json")
    if meta is None:
        return None
    A = store.read_stage_file("train", f"{key}.A.npy")
    B = store.read_stage_file("train", f"{key}.B.npy")
    if A is None or B is None:
        return None
    return TrainedAdapter(
        task_name=meta["task_name"],
        target_layer=meta["target_layer"],
        rank=meta["rank"],
        alpha=meta["alpha"],
        frac_bits=meta["frac_bits"],
        A=A,
        B=B,
        losses_fixed=list(meta["losses_fixed"]),
        ledger=list(meta["ledger"]),
        base_fingerprint=meta["base_fingerprint"],
        layer_fingerprint=meta["layer_fingerprint"],
        key=meta["key"],
    )


def _train_one(task, layer, xs_fixed, ys_fixed, cfg, base_fp, layer_fp):
    W_q = quantize(layer.weight, cfg.frac_bits)
    b_q = quantize(layer.bias, cfg.frac_bits)
    adapter = train_adapter_fixed(
        task, W_q, b_q, xs_fixed, ys_fixed, cfg.learning_rate, cfg.epochs, cfg.frac_bits
    )
    adapter.base_fingerprint = base_fp
    adapter.layer_fingerprint = layer_fp
    adapter.key = adapter_key(
        task.name, task.target_layer, task.rank, task.alpha, layer_fp, cfg.train_hash()
    )
    return adapter


def run_pipeline(sample_dir: str, store_dir: str, report_path: str | None = None):
    """Run one full pass. Returns the report dict."""
    store = Store(store_dir)
    manifest = store.load_manifest() or {
        "global_version": 0,
        "config_hash": None,
        "base_fingerprint": None,
        "stages": {},
    }
    failures = []

    # ---- stage: validate ----------------------------------------------
    cfg, cfg_failures = load_config(os.path.join(sample_dir, "peft_config.yaml"))
    pack, base_failures = load_base_pack(os.path.join(sample_dir, "base_pack"))
    failures.extend(cfg_failures)
    failures.extend(base_failures)

    tasks, samples, teacher = [], [], {}
    if cfg is not None:
        tasks, task_failures = load_tasks(
            os.path.join(sample_dir, "tasks.json"), cfg.target_layers
        )
        failures.extend(task_failures)
    if pack is not None:
        samples, sample_failures = load_samples(
            os.path.join(sample_dir, "samples.jsonl"), set(pack.layers)
        )
        failures.extend(sample_failures)
        teacher = load_teacher(os.path.join(sample_dir, "teacher.json"))

    store.commit_stage("validate", {"failures.json": {"json": failures}})

    base_fp = pack.fingerprint() if pack is not None else None
    config_hash = cfg.config_hash() if cfg is not None else None

    report = {
        "global_version": manifest["global_version"] + 1,
        "config_hash": config_hash,
        "base_fingerprint": base_fp,
        "resume": {
            "previous_version": manifest["global_version"],
            "previous_base_fingerprint": manifest["base_fingerprint"],
            "resumed": bool(
                manifest["base_fingerprint"] is not None
                and manifest["base_fingerprint"] == base_fp
                and manifest["config_hash"] == config_hash
            ),
        },
        "tasks": [],
        "merge": None,
        "diagnostics": [],
        "failures": failures,
    }

    if cfg is None or pack is None:
        _finalize(store, manifest, report, report_path, cfg, base_fp, config_hash)
        return report

    # ---- stage: train (thread pool, serial-equivalent) -----------------
    layer_fps = {name: pack.layer_fingerprint(name) for name in pack.layers}
    sample_inputs = {
        s.sample_id: sample_input(s, pack.layers[s.target_layer].shape[1]) for s in samples
    }
    teacher_deltas = {
        name: teacher_delta_matrix(teacher.get(name), pack.layers[name].shape)
        for name in pack.layers
    }

    def task_xy(task):
        layer = pack.layers[task.target_layer]
        delta = teacher_deltas[task.target_layer]
        w_ref = layer.weight if delta is None else layer.weight + delta
        xs, ys = [], []
        for s in samples:
            if s.target_layer != task.target_layer:
                continue
            x = sample_inputs[s.sample_id]
            xs.append(quantize(x, cfg.frac_bits))
            ys.append(quantize(w_ref @ x + layer.bias, cfg.frac_bits))
        return layer, xs, ys

    adapters = {}
    dispositions = {}
    to_train = []
    for task in tasks:
        key = adapter_key(
            task.name, task.target_layer, task.rank, task.alpha,
            layer_fps[task.target_layer], cfg.train_hash(),
        )
        cached = _load_adapter(store, key)
        if cached is not None:
            # Compatibility is judged per target layer: the adapter key and
            # the stored layer fingerprint both bind the adapter to the exact
            # layer content it was trained on, so a single-layer replacement
            # only invalidates adapters referencing that layer.  The global
            # base fingerprint is still carried for provenance and resume.
            if cached.layer_fingerprint != layer_fps[task.target_layer]:
                failures.append(
                    errors.failure(
                        errors.E_BASE_VERSION_MISMATCH,
                        "adapter layer fingerprint mismatch; refusing to load",
                        task=task.name,
                        layer=task.target_layer,
                    )
                )
                dispositions[task.name] = "retrained"
                to_train.append(task)
            else:
                adapters[task.name] = cached
                dispositions[task.name] = "reused"
        else:
            stale = _find_stale_adapter(store, task)
            if stale is not None and stale.layer_fingerprint != layer_fps[task.target_layer]:
                failures.append(
                    errors.failure(
                        errors.E_BASE_VERSION_MISMATCH,
                        "layer fingerprint changed; adapter invalidated and retrained",
                        task=task.name,
                        layer=task.target_layer,
                    )
                )
                dispositions[task.name] = "retrained"
            else:
                dispositions[task.name] = "trained"
            to_train.append(task)

    # Concurrent training of independent tasks; results re-sorted by the
    # stable (task name, target layer) key -> identical to serial execution.
    trained_results = {}
    if to_train:
        max_workers = min(cfg.max_workers, len(to_train))
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        _train_one, task, *task_xy(task), cfg, base_fp,
                        layer_fps[task.target_layer],
                    ): task
                    for task in to_train
                }
                for fut, task in futures.items():
                    trained_results[task.sort_key] = fut.result()
        else:
            for task in to_train:
                trained_results[task.sort_key] = _train_one(
                    task, *task_xy(task), cfg, base_fp, layer_fps[task.target_layer]
                )
    for sort_key in sorted(trained_results):
        adapter = trained_results[sort_key]
        adapters[adapter.task_name] = adapter

    train_files = {}
    for name in sorted(adapters):
        train_files.update(_export_adapter_files(adapters[name]))
    store.commit_stage("train", train_files)

    # ---- stage: merge (budgeted, reversible) ---------------------------
    layer_numel = lambda name: int(np.prod(pack.layers[name].shape))
    approved, merge_failures, pending, consumption = schedule_merges(
        tasks, layer_numel, cfg.merge_budget
    )
    failures.extend(merge_failures)

    merged_report = {}
    merge_files = {}
    adapters_by_layer = {}
    for task in approved:
        adapters_by_layer.setdefault(task.target_layer, []).append(adapters[task.name])
    for layer_name in sorted(adapters_by_layer):
        layer_adapters = sorted(adapters_by_layer[layer_name], key=lambda a: a.task_name)
        W_q = quantize(pack.layers[layer_name].weight, cfg.frac_bits)
        merged, steps, saturated = merge_layer(W_q, layer_adapters, cfg.frac_bits)
        rev_failures = check_reversible(W_q, merged, steps, layer_adapters, cfg.frac_bits)
        failures.extend(rev_failures)
        merged_report[layer_name] = {
            "adapters": [a.task_name for a in layer_adapters],
            "saturated": saturated,
            "reversible": not rev_failures,
            "steps": [
                {"task": s["task"], "saturated": s["saturated"]} for s in steps
            ],
        }
        merge_files[f"{layer_name}.npy"] = {"npy": merged}
        merge_files[f"{layer_name}.json"] = {"json": merged_report[layer_name]}
    report["merge"] = {
        "budget": cfg.merge_budget,
        "consumed": sum(c["cost"] for c in consumption),
        "per_task_quota": cfg.merge_budget // len(tasks) if tasks else 0,
        "consumption": consumption,
        "pending": [t.name for t in pending],
        "layers": merged_report,
    }
    store.commit_stage("merge", merge_files)

    # ---- stage: diagnostics + export -----------------------------------
    for task in tasks:
        layer = pack.layers[task.target_layer]
        layer_samples = [s for s in samples if s.target_layer == task.target_layer]
        xs = [sample_inputs[s.sample_id] for s in layer_samples]
        layer_adapters = adapters_by_layer.get(task.target_layer, [])
        diag = diagnose_task(
            task, layer, teacher_deltas[task.target_layer], layer_samples, xs,
            cfg.learning_rate, cfg.epochs, cfg.frac_bits, cfg.atol, cfg.rtol,
            layer_adapters=layer_adapters,
        )
        report["diagnostics"].append(diag)

    for task in tasks:
        adapter = adapters[task.name]
        report["tasks"].append(
            {
                "name": task.name,
                "target_layer": task.target_layer,
                "rank": task.rank,
                "alpha": task.alpha,
                "sort_key": list(task.sort_key),
                "disposition": dispositions[task.name],
                "adapter_key": adapter.key,
                "losses": adapter.losses,
                "merge_status": (
                    "merged"
                    if task in approved
                    else ("pending" if task in pending else "failed")
                ),
            }
        )

    export_files = {"report.json": {"json": report}}
    store.commit_stage("export", export_files)
    _finalize(store, manifest, report, report_path, cfg, base_fp, config_hash)
    return report


def _find_stale_adapter(store: Store, task):
    """Find an exported adapter for the same task identity but any layer
    fingerprint (used to detect fingerprint invalidation)."""
    for rel in store.list_stage_files("train"):
        if not rel.endswith(".json"):
            continue
        meta = store.read_stage_file("train", rel)
        if (
            meta.get("task_name") == task.name
            and meta.get("target_layer") == task.target_layer
            and meta.get("rank") == task.rank
            and meta.get("alpha") == task.alpha
        ):
            return _load_adapter(store, meta["key"])
    return None


def _finalize(store, manifest, report, report_path, cfg, base_fp, config_hash):
    version = report["global_version"]
    stage_versions = {stage: version for stage in STAGES}
    new_manifest = {
        "global_version": version,
        "config_hash": config_hash,
        "base_fingerprint": base_fp,
        "stages": stage_versions,
    }
    store.write_manifest(new_manifest)
    interval = cfg.snapshot_interval if cfg is not None else 1
    if interval >= 1 and version % interval == 0:
        store.snapshot(version)
    if report_path:
        tmp = report_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(canonical_json(report))
        os.replace(tmp, report_path)


def apply_upgrade_step(sample_dir: str, step_index: int):
    """Apply one step of the base upgrade plan: regenerate listed layers with
    the plan's deterministic seeds and atomically replace their binaries."""
    plan_path = os.path.join(sample_dir, "upgrade_plan.json")
    with open(plan_path, "r", encoding="utf-8") as fh:
        plan = json.load(fh)
    steps = plan.get("steps", [])
    step = steps[step_index]
    pack_dir = os.path.join(sample_dir, "base_pack")
    applied = []
    for repl in step.get("replacements", []):
        layer_name = repl["layer"]
        seed = int(repl["seed"])
        rng = np.random.default_rng(seed)
        shape = tuple(repl["shape"])
        weight = rng.normal(0.0, 0.5, size=shape)
        bias = rng.normal(0.0, 0.1, size=(shape[0],))
        replace_layer(pack_dir, layer_name, weight, bias)
        applied.append(layer_name)
    return applied
