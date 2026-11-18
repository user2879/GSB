"""End-to-end pipeline orchestration: forward, train, validate and snapshot."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import AdapterConfig, parse_config
from .deps import (
    build_task_identity,
    plan_invalidation,
    stable_sort_key,
    task_cache_key,
)
from .errors import (
    E_TRAIN_TASK_FAULT,
    E_VALIDATE_TASK_FAULT,
    WorkbenchError,
    failure_record,
)
from .forward import build_forward_stage
from .loader import load_base_pack, load_samples, load_task_manifest
from .lora import LoRAAdapter, train_task
from .snapshot import SnapshotStore
from .utils import copy_tree_contents, read_json, write_json_atomic


def _serialise_result(result: dict) -> dict:
    arrays = result.pop("arrays")
    payload = dict(result)
    payload["array_keys"] = sorted(arrays.keys())
    return payload, arrays


def _build_validation_summary(task_results: list[dict]) -> dict:
    entries = []
    for result in task_results:
        entries.append(
            {
                "task": result["task"],
                "sort_key": stable_sort_key(
                    result["task"], tuple(result["target_layers"])
                ),
                "metric": result["metric"],
                "score": result["best"]["score"],
                "threshold": result["best"]["threshold"],
                "passed": result["best"]["passed"],
                "best_epoch": result["best"]["best_epoch"],
                "mse": result["best"]["mse"],
                "final_fingerprint": result["final_fingerprint"],
            }
        )
    entries.sort(key=lambda entry: entry["sort_key"])
    summary = {
        "stage": "validate",
        "task_count": len(entries),
        "passed_tasks": [entry["task"] for entry in entries if entry["passed"]],
        "failed_threshold_tasks": [
            entry["task"] for entry in entries if not entry["passed"]
        ],
        "entries": entries,
    }
    return summary


def run_pipeline(
    input_dir: str,
    workspace: str,
    config: dict | AdapterConfig,
    base_subdir: str = "base_v1",
    strict_resume: bool = False,
    fault_hooks: dict | None = None,
) -> dict:
    """Execute one incremental pipeline run.

    ``fault_hooks`` maps ``(task, stage)`` tuples to exception instances. When
    the hook fires the exception is recorded in the failure list and the
    remaining tasks continue (business rule 8).
    """
    fault_hooks = fault_hooks or {}
    task_manifest = load_task_manifest(os.path.join(input_dir, "tasks.json"))
    base = load_base_pack(os.path.join(input_dir, base_subdir))
    allowed_tasks = {entry["task"] for entry in task_manifest["tasks"]}
    if isinstance(config, AdapterConfig):
        adapter_config = config
    else:
        adapter_config = parse_config(config, base.layer_count)
    samples, sample_failures = load_samples(
        os.path.join(input_dir, "samples.jsonl"),
        allowed_tasks,
        base.descriptor["vocab_size"],
    )
    sample_by_id = {sample["sample_id"]: sample for sample in samples}

    # Reject manifest ids that do not resolve to a loaded valid sample.
    failures: list[dict] = list(sample_failures)
    valid_entries: list[dict] = []
    for entry in task_manifest["tasks"]:
        missing = [
            sample_id
            for sample_id in (
                entry["train_sample_ids"] + entry["val_sample_ids"]
            )
            if sample_id not in sample_by_id
        ]
        if missing:
            failures.append(
                failure_record(
                    "E_SAMPLE_INVALID",
                    "load",
                    f"task {entry['task']}: unknown sample ids {sorted(missing)}",
                    task=entry["task"],
                )
            )
            continue
        valid_entries.append(entry)

    store = SnapshotStore(workspace)
    reconciliation = store.reconcile(
        adapter_config.config_hash(),
        base.base_fingerprint,
        strict=strict_resume,
    )
    previous_manifest = (
        reconciliation["manifest"] if reconciliation else None
    )
    old_identities = (
        previous_manifest.get("task_identities", {})
        if previous_manifest
        else {}
    )

    new_identities = {
        entry["task"]: build_task_identity(
            entry, sample_by_id, adapter_config, base.base_fingerprint
        )
        for entry in valid_entries
    }
    invalidation = plan_invalidation(old_identities, new_identities)
    reused_keys = {
        entry["task"]: entry["cache_key"] for entry in invalidation["reused"]
    }

    # Stage 1 materialises forward representations inside the shared cache
    # area; every committed snapshot below copies them into its own tree so
    # that versions remain self-contained and atomically replaceable.
    forward_cache_dir = os.path.join(store.root, "cache", "forward")
    os.makedirs(forward_cache_dir, exist_ok=True)
    forward_manifest = build_forward_stage(
        samples,
        base,
        forward_cache_dir,
        reuse_dir=store.previous_stage_dir("forward"),
    )
    forward_stage_meta = {
        "stage": "forward",
        "base_version": base.base_version,
        "base_fingerprint": base.base_fingerprint,
        "sample_count": len(samples),
    }

    ordered_entries = sorted(
        valid_entries,
        key=lambda entry: stable_sort_key(
            entry["task"], adapter_config.target_layers
        ),
    )

    def run_one(entry: dict) -> dict:
        task = entry["task"]
        identity = new_identities[task]
        cache_key = task_cache_key(identity)
        # Unchanged task: reuse artifact directly, never retrain (rule 4).
        if task in reused_keys:
            cache_path = store.cached_task_path(cache_key)
            if cache_path is not None:
                cached_arrays = store.load_task_cache(cache_key)
                cached_summary = store.load_task_summary(task)
            else:
                # Fresh cache directory but a compatible previous snapshot:
                # import the immutable artifact into the content store.
                previous_npz = os.path.join(
                    store.current_dir, "train", f"{task}.npz"
                )
                with np.load(previous_npz, allow_pickle=False) as data:
                    cached_arrays = {key: data[key] for key in data.files}
                cached_summary = read_json(
                    os.path.join(store.current_dir, "train", f"{task}.json")
                )
                store.put_task_cache(cache_key, cached_arrays)
                store.put_task_summary(task, cached_summary)
            rank = identity["rank"]
            adapter = LoRAAdapter.from_arrays(
                task,
                rank,
                identity["alpha"],
                tuple(identity["target_layers"]),
                base.hidden_dim,
                cached_arrays,
            )
            result = {
                key: value
                for key, value in cached_summary.items()
                if key != "cache_key"
            }
            result["arrays"] = adapter.to_arrays()
            return {
                "status": "ok",
                "result": result,
                "cache_key": cache_key,
                "reused": True,
            }

        hook = fault_hooks.get((task, "train"))
        if hook is not None:
            raise hook
        try:
            result = train_task(
                entry,
                samples,
                base,
                adapter_config,
                forward_cache_dir,
                forward_manifest,
            )
            hook = fault_hooks.get((task, "validate"))
            if hook is not None:
                raise hook
        except WorkbenchError:
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise WorkbenchError(
                E_TRAIN_TASK_FAULT,
                f"task {task}: unexpected {type(exc).__name__}: {exc}",
                stage="train",
            ) from exc
        return {
            "status": "ok",
            "result": result,
            "cache_key": cache_key,
            "reused": False,
        }

    task_results: list[dict] = []
    reused_flags: list[bool] = []
    reused_count = 0
    trained_count = 0
    failed_tasks: list[str] = []
    # Version numbers are drawn from the frozen history, never from a possibly
    # incompatible previous manifest, preventing cross-config overwrites.
    version_counter = store.next_version_number()

    def publish_snapshot(completed_results, completed_flags, final_commit):
        """Rebuild a self-contained staged snapshot and atomically commit it."""
        nonlocal version_counter
        staging_now = store.new_staging()
        fwd_dir = store.stage_subdir(staging_now, "forward")
        tr_dir = store.stage_subdir(staging_now, "train")
        val_dir = store.stage_subdir(staging_now, "validate")
        copy_tree_contents(forward_cache_dir, fwd_dir)
        write_json_atomic(
            os.path.join(fwd_dir, "stage.json"), forward_stage_meta
        )
        ordered_pairs = sorted(
            zip(completed_results, completed_flags),
            key=lambda item: stable_sort_key(
                item[0]["task"], tuple(item[0]["target_layers"])
            ),
        )
        ordered_results = [item[0] for item in ordered_pairs]
        for result in ordered_results:
            task = result["task"]
            summary = store.load_task_summary(task)
            write_json_atomic(os.path.join(tr_dir, f"{task}.json"), summary)
            arrays = store.load_task_cache(summary["cache_key"])
            tmp_npz = os.path.join(tr_dir, f".{task}.tmp.npz")
            np.savez(tmp_npz, **arrays)
            os.replace(tmp_npz, os.path.join(tr_dir, f"{task}.npz"))
        partial_summary = _build_validation_summary(ordered_results)
        write_json_atomic(
            os.path.join(val_dir, "summary.json"), partial_summary
        )
        current_failures = sorted(
            failures,
            key=lambda record: (
                record.get("task", ""),
                record["stage"],
                record["code"],
                record["message"],
            ),
        )
        write_json_atomic(
            os.path.join(staging_now, "failures.json"),
            {"failures": current_failures},
        )
        version_counter += 1
        partial_manifest = {
            "version": version_counter,
            "config_hash": adapter_config.config_hash(),
            "base_version": base.base_version,
            "base_fingerprint": base.base_fingerprint,
            "completed_tasks": [item["task"] for item in ordered_results],
            "failed_tasks": sorted(failed_tasks),
            "final": final_commit,
            "stages": {
                "forward": {
                    "product": "forward/index.json",
                    "base_fingerprint": base.base_fingerprint,
                    "entry_count": forward_manifest["entry_count"],
                },
                "train": {
                    "product": "train/",
                    "task_count": len(ordered_results),
                    "trained_count": sum(
                        1 for flag in completed_flags if not flag
                    ),
                    "reused_count": sum(1 for flag in completed_flags if flag),
                },
                "validate": {
                    "product": "validate/summary.json",
                    "task_count": partial_summary["task_count"],
                },
            },
            "task_identities": new_identities,
            "invalidation": invalidation,
            "config_payload": adapter_config.training_payload(),
        }
        store.commit(staging_now, partial_manifest)
        return partial_manifest

    # Thread pool: tasks are independent; all outputs are sorted afterwards so
    # scheduling order can never affect result order (rule 7).
    with ThreadPoolExecutor(max_workers=adapter_config.concurrency) as pool:
        futures = [pool.submit(run_one, entry) for entry in ordered_entries]
        for index, (future, entry) in enumerate(
            zip(futures, ordered_entries), start=1
        ):
            task = entry["task"]
            try:
                outcome = future.result()
            except WorkbenchError as exc:
                stage = exc.stage if exc.code in (
                    E_TRAIN_TASK_FAULT,
                    E_VALIDATE_TASK_FAULT,
                ) else ("validate" if exc.stage == "validate" else "train")
                failures.append(
                    failure_record(
                        exc.code,
                        stage,
                        exc.message,
                        task=task,
                        exception_type=type(exc).__name__,
                    )
                )
                failed_tasks.append(task)
                continue
            result = outcome["result"]
            cache_key = outcome["cache_key"]
            is_reused = outcome["reused"]
            if is_reused:
                reused_count += 1
            else:
                trained_count += 1
                payload, arrays = _serialise_result(dict(result))
                store.put_task_cache(cache_key, arrays)
                payload["cache_key"] = cache_key
                store.put_task_summary(task, payload)
                result["arrays"] = arrays
                result["cache_key"] = cache_key
            if is_reused:
                summary = store.load_task_summary(task)
                result["cache_key"] = summary["cache_key"]
            task_results.append(result)
            reused_flags.append(is_reused)
            if index % adapter_config.commit_interval == 0 and index < len(
                ordered_entries
            ):
                publish_snapshot(task_results, reused_flags, final_commit=False)

    paired = sorted(
        zip(task_results, reused_flags),
        key=lambda item: stable_sort_key(
            item[0]["task"], tuple(item[0]["target_layers"])
        ),
    )
    task_results = [item[0] for item in paired]
    reused_flags = [item[1] for item in paired]
    validation_summary = _build_validation_summary(task_results)
    manifest = publish_snapshot(task_results, reused_flags, final_commit=True)

    return {
        "manifest": manifest,
        "failures": failures,
        "invalidation": invalidation,
        "trained_tasks": [
            result["task"]
            for result, flag in paired
            if not flag
        ],
        "reused_tasks": [
            result["task"]
            for result, flag in paired
            if flag
        ],
        "task_results": task_results,
        "validation_summary": validation_summary,
        "reconciliation": reconciliation,
        "workspace": store.root,
    }
