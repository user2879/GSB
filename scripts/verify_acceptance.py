#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from fixed_lab.config import load_config
from fixed_lab.dataset import apply_commands, dataset_from_records, load_command_stream, load_samples_jsonl
from fixed_lab.hashing import canonical_sha256
from fixed_lab.pipeline import run_pipeline
from fixed_lab.snapshot import commit_snapshot


def main() -> int:
    config = load_config(ROOT / "configs/lab.json")
    records, errors = load_samples_jsonl(ROOT / "data/samples.jsonl", config["columns"])
    if errors:
        raise SystemExit(json.dumps({"status": "E_INPUT_INVALID", "errors": errors}, indent=2))
    dataset = dataset_from_records(config["columns"], records)
    commands, parse_errors = load_command_stream(ROOT / "edits/demo_edits.jsonl")
    if parse_errors:
        raise SystemExit(json.dumps({"status": "E_INPUT_INVALID", "errors": parse_errors}, indent=2))

    cache_dir = tempfile.mkdtemp(prefix="fixed-lab-verify-inc-")
    initial = run_pipeline(dataset, copy.deepcopy(config), cache_dir=cache_dir, worker_count=4)
    previous = {entry["artifact_id"]: entry for entry in initial["ledger"]["entries"]}
    inc_dataset, inc_config, command_results, command_errors = apply_commands(
        dataset, copy.deepcopy(config), commands
    )
    if command_errors:
        raise SystemExit(json.dumps({"status": "E_INPUT_INVALID", "errors": command_errors}, indent=2))
    incremental = run_pipeline(
        inc_dataset,
        inc_config,
        cache_dir=cache_dir,
        previous_lineage=previous,
        worker_count=4,
        command_results=command_results,
    )

    full_dataset, full_config, full_results, full_errors = apply_commands(
        dataset, copy.deepcopy(config), commands
    )
    full = run_pipeline(
        full_dataset,
        full_config,
        cache_dir=tempfile.mkdtemp(prefix="fixed-lab-verify-full-"),
        worker_count=4,
        command_results=full_results,
    )
    serial = run_pipeline(
        full_dataset, copy.deepcopy(full_config), cache_dir=None, worker_count=1, command_results=full_results
    )
    parallel = run_pipeline(
        full_dataset, copy.deepcopy(full_config), cache_dir=None, worker_count=8, command_results=full_results
    )

    inc_hash = canonical_sha256(incremental["report"])
    full_hash = canonical_sha256(full["report"])
    serial_hash = canonical_sha256(serial["report"])
    parallel_hash = canonical_sha256(parallel["report"])

    replacement_cache = tempfile.mkdtemp(prefix="fixed-lab-verify-reuse-")
    reuse_initial = run_pipeline(dataset, copy.deepcopy(config), cache_dir=replacement_cache, worker_count=1)
    reuse_previous = {entry["artifact_id"]: entry for entry in reuse_initial["ledger"]["entries"]}
    replacement = [{
        "op": "replace_sample",
        "id": "S0003",
        "sample": {"id": "S0003", "seed": 1, "features": [0.01] * 6, "target": 0.2, "weight": 1.0},
    }]
    replaced_dataset, replaced_config, _, replacement_errors = apply_commands(
        dataset, copy.deepcopy(config), replacement
    )
    reuse_result = run_pipeline(
        replaced_dataset,
        replaced_config,
        cache_dir=replacement_cache,
        previous_lineage=reuse_previous,
        worker_count=1,
    )
    mac_entries = [entry for entry in reuse_result["ledger"]["entries"] if entry["stage"] == "accumulation_mac"]
    reuse_counts = {
        "mac_reused": sum(entry["status"] == "reused" for entry in mac_entries),
        "mac_rebuilt": sum(entry["status"] == "rebuilt" for entry in mac_entries),
    }

    snapshot_root = tempfile.mkdtemp(prefix="fixed-lab-snapshot-")
    receipt = commit_snapshot(Path(snapshot_root) / "snapshots", {
        "report.json": incremental["report"],
        "ledger.json": incremental["ledger"],
    }, version_hash=inc_hash)

    result = {
        "status": "OK" if inc_hash == full_hash == serial_hash == parallel_hash else "MISMATCH",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "report_sha256": inc_hash,
        "incremental_equals_blank_rebuild": inc_hash == full_hash,
        "serial_equals_thread_pool": serial_hash == parallel_hash == full_hash,
        "lineage_entry_count": len(incremental["report"]["lineage_checklist"]),
        "replacement_precise_cache": reuse_counts,
        "path_event_counts": {
            name: len(path["events"]) for name, path in incremental["report"]["paths"].items()
        },
        "strong_pairs": [
            pair["columns"] for pair in incremental["report"]["correlation"]["pairs"] if pair["strong_match"]
        ],
        "budget_deferred_entries": sum(
            len(batch["deferred"]) for batch in incremental["report"]["accumulation"]["budget_plan"]
        ),
        "snapshot_receipt": receipt,
    }
    out_path = ROOT / "artifacts" / "verification_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
