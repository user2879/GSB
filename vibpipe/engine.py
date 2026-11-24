from __future__ import annotations

from pathlib import Path
from typing import Any

from .cache import make_ledger_record, stage_cache_key
from .concurrency import ConcurrencyMonitor
from .config import LoadedConfig, topological_stages
from .errors import (
    BATCH_COMPUTE,
    BUDGET_PENDING,
    CONFIG_INVALID,
    SNAPSHOT_WRITE,
    Failure,
    sorted_failures,
)
from .faults import FaultInjector, InjectedFault
from .hashing import fingerprint
from .samples import Sample
from .snapshot import SnapshotStore, atomic_write_json
from .stages import make_source_product, stage_parameter_fingerprint, compute_stage
from .stats import compute_stage_window_stats, decide_all_drift


def run_pipeline(
    config: LoadedConfig,
    samples: list[Sample],
    output_dir: str | Path | None = None,
    *,
    max_workers: int | None = None,
    budget: int | None = None,
    gear_override: str | None = None,
    decimal_width_override: int | None = None,
    include_attribution: bool = True,
    use_cache: bool = True,
    fault_injector: FaultInjector | None = None,
) -> dict[str, Any]:
    fault_injector = fault_injector or FaultInjector()
    requested_workers = int(max_workers if max_workers is not None else config.max_workers)
    workers = max(1, min(requested_workers, config.max_workers))
    point_budget = int(budget if budget is not None else config.point_budget)
    gear = gear_override or config.gear
    decimal_width = int(decimal_width_override or config.decimal_width)
    store = SnapshotStore(output_dir) if output_dir is not None else None
    sample_fingerprint = fingerprint([sample.to_dict() for sample in samples])
    recovery = (
        store.recovery_decision(config.fingerprint, sample_fingerprint)
        if store
        else {"decision": "ephemeral", "reason": "no_output_dir"}
    )
    monitor = ConcurrencyMonitor(limit=workers)
    failures: list[Failure] = []

    stages = topological_stages(config.stages)
    stage_rank = {"source": -1, **{stage["name"]: i for i, stage in enumerate(stages)}}
    batches = sorted({sample.batch for sample in samples})

    total_per_batch = _points_per_batch(config, stages)
    if point_budget < total_per_batch:
        return _run_budget_pending(
            config=config,
            samples=samples,
            stages=stages,
            stage_rank=stage_rank,
            batches=batches,
            budget=point_budget,
            gear=gear,
            decimal_width=decimal_width,
            workers=workers,
            monitor=monitor,
            store=store,
            recovery=recovery,
        )

    products: dict[str, dict[str, Any]] = {}
    stats: dict[str, dict[str, list[dict[str, Any]]]] = {}
    ledger: list[dict[str, Any]] = []
    cache_actions: list[dict[str, Any]] = []
    invalidation_list: list[str] = []
    reuse_list: list[str] = []
    failed_batches: set[int] = set()
    disk_failed_batches: set[int] = set()

    source_parameter_fp = fingerprint(
        {
            "input_scale": int(config.raw.get("input_scale", 10 ** config.decimal_width)),
            "windows": config.raw["windows"],
            "imputation": config.raw.get("imputation", {"strategy": "window_observed_mean"}),
        }
    )
    source_upstream_fp = fingerprint(
        {"sample_fingerprint": fingerprint([s.to_dict() for s in samples])}
    )
    source_key = stage_cache_key(source_upstream_fp, source_parameter_fp, decimal_width, gear)
    source_product = None
    if use_cache and store and not fault_injector.injection_ids:
        source_product = store.load_stage_cache(source_key)
    if source_product is not None:
        action = "REUSED"
        reuse_list.append("source")
    else:
        source_product = make_source_product(config, samples, gear, workers, monitor)
        action = "RECOMPUTED"
        invalidation_list.append("source")
        if use_cache and store and not fault_injector.injection_ids:
            store.save_stage_cache(source_key, source_product)
    source_product["cache_key"] = source_key
    source_product["decimal_width"] = decimal_width
    products["source"] = source_product
    cache_actions.append({"stage": "source", "status": action, "cache_key": source_key})
    source_record = make_ledger_record(
        "source", source_key, source_upstream_fp, source_parameter_fp, source_product, action
    )
    source_record["stage_rank"] = -1
    ledger.append(source_record)
    stats["source"] = compute_stage_window_stats(source_product, config)

    upstream_name_by_stage: dict[str, str] = {}
    for stage in stages:
        name = stage["name"]
        upstream_ref = stage.get("upstream", stage.get("upstream_stage", "source"))
        if isinstance(upstream_ref, list):
            upstream_name = upstream_ref[0] if upstream_ref else "source"
        else:
            upstream_name = upstream_ref or "source"
        upstream_name_by_stage[name] = upstream_name
        upstream = products[upstream_name]
        parameter_fp = stage_parameter_fingerprint(stage)
        key = stage_cache_key(
            upstream["product_fingerprint"], parameter_fp, decimal_width, gear
        )

        excluded_batches: set[int] = set()
        for spec in fault_injector.compute_faults:
            if spec["stage"] == name:
                batch = int(spec["batch"])
                try:
                    fault_injector.check_compute(batch, name)
                except InjectedFault as exc:
                    failures.append(
                        Failure(
                            BATCH_COMPUTE,
                            f"injected compute failure: {exc}",
                            batch=batch,
                            stage=name,
                            injection_id=exc.injection_id,
                        )
                    )
                    failed_batches.add(batch)
                    excluded_batches.add(batch)

        compute_upstream = upstream
        if excluded_batches:
            compute_upstream = dict(upstream)
            compute_upstream["rows"] = [
                row for row in upstream["rows"] if int(row["batch"]) not in excluded_batches
            ]

        product = None
        if use_cache and store and not fault_injector.injection_ids and not excluded_batches:
            product = store.load_stage_cache(key)
        if product is not None:
            action = "REUSED"
            reuse_list.append(name)
        else:
            product = compute_stage(
                stage, compute_upstream, config, gear, workers, monitor
            )
            action = "RECOMPUTED"
            invalidation_list.append(name)
            if use_cache and store and not fault_injector.injection_ids and not excluded_batches:
                store.save_stage_cache(key, product)

        product["cache_key"] = key
        product["decimal_width"] = decimal_width
        products[name] = product
        cache_actions.append({"stage": name, "status": action, "cache_key": key})
        record = make_ledger_record(
            name, key, upstream["product_fingerprint"], parameter_fp, product, action
        )
        record["stage_rank"] = stage_rank[name]
        ledger.append(record)
        stats[name] = compute_stage_window_stats(product, config)

        for spec in fault_injector.disk_faults:
            if spec["stage"] != name:
                continue
            batch = int(spec["batch"])
            try:
                fault_injector.check_disk(batch, name)
                if store is not None:
                    atomic_write_json(
                        store.root / "batch_journals" / f"batch-{batch:06d}" / f"{name}.json",
                        {"batch": batch, "stage": name, "product_fingerprint": product.get("product_fingerprint")},
                    )
            except InjectedFault as exc:
                disk_failed_batches.add(batch)
                failures.append(
                    Failure(
                        SNAPSHOT_WRITE,
                        f"injected snapshot write failure: {exc}",
                        batch=batch,
                        stage=name,
                        injection_id=exc.injection_id,
                    )
                )

    final_stage_name = str(config.raw.get("feature_stage", stages[-1]["name"]))
    final_product = products[final_stage_name]
    try:
        drift = decide_all_drift(final_product, stats[final_stage_name], config)
    except KeyError as exc:
        failures.append(Failure(CONFIG_INVALID, f"missing threshold configuration: {exc}"))
        drift = []

    boundary_features = [row["feature_id"] for row in drift if row["boundary"]]
    failures = sorted_failures(failures)

    attribution = None
    if include_attribution and not fault_injector.injection_ids and drift:
        from .precision import build_attribution

        attribution = build_attribution(config, samples, products, stats, drift, output_dir)

    completed_batches = [
        batch
        for batch in batches
        if batch not in failed_batches and batch not in disk_failed_batches
    ]
    cursor = _contiguous_cursor(batches, failed_batches | disk_failed_batches)
    snapshot_interval = int(config.raw["snapshot_interval_batches"])
    checkpoint_cursors = [
        batch_end
        for batch_end in range(snapshot_interval - 1, cursor + 1, snapshot_interval)
    ]
    manifest = {
        "global_version": _next_global_version(recovery),
        "config_hash": config.fingerprint,
        "sample_fingerprint": sample_fingerprint,
        "gear": gear,
        "decimal_width": decimal_width,
        "completed_batch_cursor": cursor,
        "completed_batches": completed_batches,
        "failed_batches": sorted(failed_batches | disk_failed_batches),
        "snapshot_interval_batches": snapshot_interval,
        "checkpoint_cursors": checkpoint_cursors,
        "point_budget_per_batch": point_budget,
        "stage_versions": {
            name: {
                "version": 1,
                "cache_key": products[name].get("cache_key"),
                "product_fingerprint": products[name].get("product_fingerprint"),
                "row_count": len(products[name].get("rows", [])),
                "complete": True,
            }
            for name in products
        },
    }
    commit_pointer = None
    if store is not None:
        atomic_write_json(
            store.root / "checkpoints" / "checkpoint_history.json",
            {
                "config_hash": config.fingerprint,
                "global_version": _next_global_version(recovery),
                "interval_batches": snapshot_interval,
                "checkpoint_cursors": checkpoint_cursors,
            },
        )
        commit_pointer = store.commit(manifest, products, [])

    status = "ok"
    if failures:
        status = "completed_with_failures"
    report = {
        "schema": "vibpipe-report/v1",
        "status": status,
        "config_fingerprint": config.fingerprint,
        "sample_fingerprint": manifest["sample_fingerprint"],
        "gears": {
            "configured": config.gear,
            "executed": gear,
            "high_precision": None if gear_override else _safe_next_gear(config.gear),
        },
        "concurrency": {
            "configured_max_workers": config.max_workers,
            "requested_workers": workers,
            "monitor_limit": monitor.limit,
            "max_observed_workers": monitor.max_observed,
        },
        "point_budget_per_batch": point_budget,
        "points_per_full_batch": total_per_batch,
        "checkpoint_cursors": checkpoint_cursors,
        "windows": config.raw["windows"],
        "stats": stats,
        "drift": drift,
        "boundary_features": boundary_features,
        "attribution": attribution,
        "ledger": sorted(ledger, key=lambda row: row["stage_rank"]),
        "cache_actions": cache_actions,
        "invalidation_list": invalidation_list,
        "reuse_list": reuse_list,
        "pending": [],
        "failures": failures,
        "injection_hits": sorted(fault_injector.hit_ids(failures)),
        "snapshot": {
            "recovery": recovery,
            "commit": commit_pointer,
            "manifest": manifest,
        },
        "summary": {
            "sample_count": len(samples),
            "batch_count": len(batches),
            "feature_count": len(final_product["features"]),
            "drift_count": sum(1 for row in drift if row["drift"]),
            "boundary_count": len(boundary_features),
            "failure_count": len(failures),
            "pending_count": 0,
            "stage_count": len(stages),
        },
    }
    return {"report": report, "products": products}


def _run_budget_pending(**kwargs: Any) -> dict[str, Any]:
    config: LoadedConfig = kwargs["config"]
    samples: list[Sample] = kwargs["samples"]
    stages = kwargs["stages"]
    stage_rank = kwargs["stage_rank"]
    batches = kwargs["batches"]
    budget: int = kwargs["budget"]
    gear: str = kwargs["gear"]
    decimal_width: int = kwargs["decimal_width"]
    workers: int = kwargs["workers"]
    monitor: ConcurrencyMonitor = kwargs["monitor"]
    store: SnapshotStore | None = kwargs["store"]
    recovery: dict[str, Any] = kwargs["recovery"]

    failures: list[Failure] = []
    pending: list[dict[str, Any]] = []
    partial_products: dict[str, dict[str, Any]] = {"source": make_source_product(config, samples, gear, workers, monitor)}
    partial_rows: dict[str, list[dict[str, Any]]] = {stage["name"]: [] for stage in stages}
    feature_counts = _stage_feature_counts(config, stages)

    for batch in batches:
        batch_samples = sorted((s for s in samples if s.batch == batch), key=lambda s: s.seq)
        remaining = budget
        batch_pending = 0
        for sample in batch_samples:
            one_upstream = partial_products["source"]
            one_upstream = {**one_upstream, "rows": [r for r in one_upstream["rows"] if r["seq"] == sample.seq]}
            blocked = False
            for stage in stages:
                name = stage["name"]
                cost = feature_counts[name]
                task = {
                    "code": BUDGET_PENDING,
                    "batch": batch,
                    "seq": sample.seq,
                    "stage": name,
                    "stage_rank": stage_rank[name],
                    "feature_points": cost,
                    "sort_key": [batch, sample.seq, stage_rank[name], name],
                }
                if blocked:
                    task["reason"] = "UPSTREAM_PENDING"
                    pending.append(task)
                    batch_pending += cost
                    continue
                if cost > remaining:
                    task["reason"] = "BUDGET_EXHAUSTED"
                    pending.append(task)
                    batch_pending += cost
                    blocked = True
                    continue
                product = compute_stage(stage, one_upstream, config, gear, 1, monitor)
                row = product["rows"][0]
                partial_rows[name].append(row)
                skeleton = dict(product)
                skeleton["rows"] = [row]
                one_upstream = skeleton
                remaining -= cost
            # Merge after the sample chain completes/admits.
        for stage in stages:
            name = stage["name"]
            # Rows were appended while walking the chain; use the last skeleton features by recomputing merge below.
        if batch_pending:
            failures.append(
                Failure(
                    BUDGET_PENDING,
                    f"batch {batch} feature-point budget {budget} exhausted; {batch_pending} points pending",
                    batch=batch,
                )
            )

    products: dict[str, dict[str, Any]] = {"source": partial_products["source"]}
    for stage in stages:
        name = stage["name"]
        rows = sorted(partial_rows[name], key=lambda r: r["seq"])
        products[name] = {
            "stage": name,
            "type": stage["type"],
            "scale": int(stage["scale"]),
            "gear": gear,
            "features": _skeleton_features(stage, config),
            "rows": rows,
            "partial": True,
            "product_fingerprint": fingerprint({"stage": name, "rows": rows}),
        }

    pending.sort(key=lambda item: tuple(item["sort_key"]))
    failures = sorted_failures(failures)
    manifest = {
        "global_version": _next_global_version(recovery),
        "config_hash": config.fingerprint,
        "sample_fingerprint": fingerprint([s.to_dict() for s in samples]),
        "gear": gear,
        "decimal_width": decimal_width,
        "completed_batch_cursor": -1,
        "completed_batches": [],
        "failed_batches": [],
        "snapshot_interval_batches": int(config.raw["snapshot_interval_batches"]),
        "point_budget_per_batch": budget,
        "budget_status": "pending",
        "stage_versions": {
            name: {
                "cache_key": None,
                "product_fingerprint": product.get("product_fingerprint"),
                "row_count": len(product.get("rows", [])),
                "complete": False,
            }
            for name, product in products.items()
        },
    }
    commit_pointer = None
    if store is not None:
        commit_pointer = store.commit(manifest, products, pending)
    report = {
        "schema": "vibpipe-report/v1",
        "status": "budget_pending",
        "config_fingerprint": config.fingerprint,
        "sample_fingerprint": manifest["sample_fingerprint"],
        "gears": {"configured": config.gear, "executed": gear, "high_precision": _safe_next_gear(config.gear)},
        "concurrency": {
            "configured_max_workers": config.max_workers,
            "requested_workers": workers,
            "monitor_limit": monitor.limit,
            "max_observed_workers": monitor.max_observed,
        },
        "point_budget_per_batch": budget,
        "points_per_full_batch": total_per_batch_placeholder(config, stages),
        "pending": pending,
        "failures": failures,
        "injection_hits": [],
        "ledger": [],
        "cache_actions": [],
        "invalidation_list": [stage["name"] for stage in stages],
        "reuse_list": [],
        "snapshot": {"recovery": recovery, "commit": commit_pointer, "manifest": manifest},
        "summary": {
            "sample_count": len(samples),
            "batch_count": len(batches),
            "pending_count": len(pending),
            "failure_count": len(failures),
            "drift_count": 0,
            "boundary_count": 0,
        },
    }
    return {"report": report, "products": products}


def _points_per_batch(config: LoadedConfig, stages: list[dict[str, Any]]) -> int:
    count = int(config.raw["data_source"]["samples_per_batch"])
    return count * sum(_stage_feature_counts(config, stages).values())


def total_per_batch_placeholder(config: LoadedConfig, stages: list[dict[str, Any]]) -> int:
    return _points_per_batch(config, stages)


def _stage_feature_counts(config: LoadedConfig, stages: list[dict[str, Any]]) -> dict[str, int]:
    channels = len(config.channels)
    counts: dict[str, int] = {}
    for stage in stages:
        stype = stage["type"]
        if stype in {"standardize", "bucket", "log"}:
            counts[stage["name"]] = channels
        elif stype == "interaction":
            counts[stage["name"]] = len(stage.get("params", {}).get("pairs", []))
        elif stype == "clip":
            upstream = stage.get("upstream", "interaction")
            interaction = next((s for s in stages if s["name"] == upstream), None)
            if interaction is not None:
                counts[stage["name"]] = len(interaction.get("params", {}).get("pairs", []))
            else:
                counts[stage["name"]] = channels
        else:
            counts[stage["name"]] = channels
    return counts


def _skeleton_features(stage: dict[str, Any], config: LoadedConfig) -> list[dict[str, Any]]:
    count = _stage_feature_counts(config, [stage])[stage["name"]]
    if stage["type"] == "interaction":
        pairs = stage.get("params", {}).get("pairs", [])
        names = stage.get("params", {}).get("output_names") or [
            f"pair_{a}_{b}" for a, b in pairs
        ]
        return [{"id": f"{stage['name']}:{name}", "name": name} for name in names]
    return [
        {"id": f"{stage['name']}:feature_{i}", "name": f"feature_{i}"}
        for i in range(count)
    ]


def _safe_next_gear(gear: str) -> str | None:
    try:
        from .fixedpoint import next_gear

        return next_gear(gear)
    except Exception:
        return None


def _next_global_version(recovery: dict[str, Any]) -> int:
    manifest = recovery.get("manifest") if recovery else None
    if not manifest:
        return 1
    return int(manifest.get("global_version", 0)) + 1


def _contiguous_cursor(batches: list[int], failed: set[int]) -> int:
    if not batches:
        return -1
    expected = min(batches)
    cursor = expected - 1
    for batch in batches:
        if batch != expected:
            break
        if batch in failed:
            break
        cursor = batch
        expected += 1
    return cursor
