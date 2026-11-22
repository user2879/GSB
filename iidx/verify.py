from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any

from .config import load_config
from .demo import generate_sample
from .editing import append_document, delete_document, patch_config, update_document
from .engine import Engine
from .indexer import canonical_projection
from .loader import discover_inputs
from .util import atomic_write_json, first_difference


SOURCE_FILES = ["documents.jsonl", "queries.jsonl", "vocabulary.txt", "query_versions.json", "config.json"]


def _copy_inputs(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy2(source / name, destination / name)


def _run(data_dir: Path, state_dir: Path, stop_event: threading.Event | None = None):
    paths = discover_inputs(data_dir)
    config_result = load_config(paths.config)
    engine = Engine(
        data_dir=data_dir,
        state_dir=state_dir,
        config=config_result.config,
        config_failures=config_result.failures,
        stop_event=stop_event,
    )
    return engine.run()


def compare_reports(incremental: dict[str, Any], full: dict[str, Any]) -> dict[str, Any]:
    incremental_projection = canonical_projection(incremental)
    full_projection = canonical_projection(full)
    difference = first_difference(incremental_projection, full_projection)
    return {
        "equivalent": difference is None,
        "incremental_digest": incremental.get("canonical_digest"),
        "full_digest": full.get("canonical_digest"),
        "first_difference": None
        if difference is None
        else {"path": difference[0], "incremental": difference[1], "full": difference[2]},
        "provenance_excluded": ["reuse ledger", "stage reuse", "run metadata", "timings", "checkpoint events"],
    }


def run_verification(work_root: Path, *, seed: int = 20261123) -> dict[str, Any]:
    work_root = Path(work_root).resolve()
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    incremental_data = work_root / "incremental-data"
    full_data = work_root / "full-data"
    generate_sample(incremental_data, seed=seed, force=True)
    _copy_inputs(incremental_data, full_data)
    incremental_state = work_root / "incremental-state"
    steps: list[dict[str, Any]] = []

    def record_step(step: str, details: dict[str, Any], full_state_name: str) -> None:
        _copy_inputs(incremental_data, full_data)
        incremental_result = _run(incremental_data, incremental_state)
        full_result = _run(full_data, work_root / full_state_name)
        comparison = compare_reports(incremental_result.report, full_result.report)
        steps.append(
            {
                "step": step,
                "details": details,
                "comparison": comparison,
                "summary": incremental_result.report.get("summary"),
                "reuse": incremental_result.report.get("reuse", {}).get("counts"),
                "failures": incremental_result.report.get("failures", []),
                "incremental_snapshot": str(incremental_result.snapshot_path)
                if incremental_result.snapshot_path
                else None,
                "full_snapshot": str(full_result.snapshot_path) if full_result.snapshot_path else None,
            }
        )

    record_step("initial", {"action": "none"}, "full-state-initial")
    update_document(
        incremental_data / "documents.jsonl",
        "doc-0002",
        {"title_terms": ["alpha", "beta", "gamma"], "body_terms": ["alpha", "index", "query"], "shard": "shard-b"},
    )
    record_step("single-document-correction", {"doc_id": "doc-0002"}, "full-state-update")
    append_document(
        incremental_data / "documents.jsonl",
        {
            "doc_id": "doc-9001",
            "title_terms": ["Search", "ＶＥＣＴＯＲ"],
            "body_terms": ["CACHE", "index", "alpha", "manifest"],
            "shard": "shard-c",
        },
    )
    record_step("shard-append", {"doc_id": "doc-9001"}, "full-state-append")
    delete_document(incremental_data / "documents.jsonl", "doc-0007")
    record_step("document-delete", {"doc_id": "doc-0007"}, "full-state-delete")
    patch_config(incremental_data / "config.json", {"compression_tier": 2})
    record_step("compression-tier-switch", {"compression_tier": 2}, "full-state-tier")
    patch_config(incremental_data / "config.json", {"chunk_size": 4})
    record_step("chunk-size-change", {"chunk_size": 4}, "full-state-chunk")
    patch_config(
        incremental_data / "config.json",
        {"global_token_budget": 1800, "per_shard_token_quota": 650, "compression_tier": "auto"},
    )
    record_step(
        "budget-and-tier-policy-change",
        {"global_token_budget": 1800, "per_shard_token_quota": 650, "compression_tier": "auto"},
        "full-state-budget",
    )

    final_incremental = _run(incremental_data, incremental_state)
    final_report = final_incremental.report
    lengths_differ = all(
        block["compression"]["lengths_differ"] for block in final_report.get("blocks", [])
    )
    decode_ok = all(
        block["compression"]["decode_ok"] for block in final_report.get("blocks", [])
    )
    selected_tiers = sorted(
        {block["compression"]["selected_tier"] for block in final_report.get("blocks", [])}
    )
    result = {
        "schema": 1,
        "verification": "incremental-vs-clean-full-rebuild",
        "seed": seed,
        "ok": all(step["comparison"]["equivalent"] for step in steps),
        "steps": steps,
        "final": {
            "block_count": len(final_report.get("blocks", [])),
            "query_count": len(final_report.get("queries", [])),
            "selected_tiers_present": selected_tiers,
            "all_two_tier_bytes_differ": lengths_differ,
            "all_decode_checksums_ok": decode_ok,
            "all_queries_have_checksums": all(
                query.get("result_checksum") for query in final_report.get("queries", [])
            ),
            "completed_shards": final_report.get("summary", {}).get("completed_shards"),
            "budget_ledger_entries": len(final_report.get("budget_ledger", [])),
            "concurrency": final_report.get("diagnostics", {}).get("concurrency"),
            "canonical_digest": final_report.get("canonical_digest"),
            "report_path": str(final_incremental.report_path),
        },
        "notes": [
            "Canonical comparison covers blocks, block fingerprints, both encoded byte lengths, decoded checksums, query result order, referenced blocks, budget ledger, compression decisions and stable failures.",
            "Reuse provenance is intentionally excluded because a clean full rebuild has an empty cache while an incremental run reuses persisted artifacts.",
        ],
    }
    result["resume_verification"] = run_resume_verification(work_root / "resume", seed=seed)
    result["budget_verification"] = run_budget_verification(work_root / "budget", seed=seed)
    result["validation_verification"] = run_validation_verification(work_root / "validation", seed=seed)
    result["ok"] = bool(
        result["ok"]
        and result["resume_verification"]["ok"]
        and result["budget_verification"]["ok"]
        and result["validation_verification"]["ok"]
        and lengths_differ
        and decode_ok
    )
    atomic_write_json(work_root / "verification-report.json", result)
    return result


def run_resume_verification(work_root: Path, *, seed: int) -> dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=True)
    data = work_root / "data"
    full_data = work_root / "full-data"
    generate_sample(data, seed=seed, force=True)
    _copy_inputs(data, full_data)
    stop_event = threading.Event()
    stop_event.set()
    interrupted = _run(data, work_root / "resume-state", stop_event)
    resumed = _run(data, work_root / "resume-state")
    full = _run(full_data, work_root / "full-state")
    comparison = compare_reports(resumed.report, full.report)
    return {
        "ok": interrupted.interrupted
        and interrupted.exit_code == 0
        and resumed.report["diagnostics"].get("checkpoint", {}).get("resumed") is True
        and comparison["equivalent"],
        "interrupted_exit_code": interrupted.exit_code,
        "interrupted_status": interrupted.report.get("run", {}).get("status"),
        "ledger_entries_at_stop": len(interrupted.report.get("budget_ledger", [])),
        "resumed": resumed.report["diagnostics"].get("checkpoint", {}),
        "comparison": comparison,
    }


def run_budget_verification(work_root: Path, *, seed: int) -> dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=True)
    data = work_root / "data"
    generate_sample(data, seed=seed, force=True)
    config_path = data / "config.json"
    patch_config(config_path, {"global_token_budget": 135, "per_shard_token_quota": 500})
    result = _run(data, work_root / "state")
    report = result.report
    codes = [failure["code"] for failure in report.get("failures", [])]
    ledger = report.get("budget_ledger", [])
    pending_queue = report.get("pending_queue", [])
    monotonic = all(
        left["deduction_index"] == index + 1 and left["remaining"] <= left["remaining_before"]
        for index, left in enumerate(ledger)
    )
    return {
        "ok": "E_BUDGET_PENDING" in codes
        and report.get("summary", {}).get("pending_shards")
        and any(item.get("pending_documents", 0) > 0 for item in pending_queue)
        and bool(ledger)
        and monotonic,
        "codes": codes,
        "pending_shards": report.get("summary", {}).get("pending_shards"),
        "pending_queue": pending_queue,
        "partial_shards": report.get("summary", {}).get("partial_shards"),
        "ledger_entries": len(ledger),
        "remaining_token_budget": report.get("summary", {}).get("remaining_token_budget"),
        "deductions_monotonic": monotonic,
    }


def run_validation_verification(work_root: Path, *, seed: int) -> dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=True)
    data = work_root / "data"
    generate_sample(data, seed=seed, force=True, include_invalid=True)
    patch_config(
        data / "config.json",
        {"thread_pool_size": 6, "concurrency_quota": 4, "compression_switch_points": [5, 5]},
    )
    result = _run(data, work_root / "state")
    codes = [failure["code"] for failure in result.report.get("failures", [])]
    required = {
        "E_CONFIG_CONCURRENCY_EXCEEDS_QUOTA",
        "E_CONFIG_SWITCH_POINTS_NOT_INCREASING",
        "E_DOCUMENT_TERM_OUT_OF_VOCAB",
        "E_DOCUMENT_ID_DUPLICATE",
        "E_QUERY_VERSION_UNSUPPORTED",
        "E_QUERY_ID_DUPLICATE",
    }
    return {
        "ok": required <= set(codes) and result.report.get("summary", {}).get("accepted_documents") == 36,
        "codes": codes,
        "accepted_documents": result.report.get("summary", {}).get("accepted_documents"),
        "accepted_queries": result.report.get("summary", {}).get("accepted_queries"),
        "failure_count": len(codes),
    }
