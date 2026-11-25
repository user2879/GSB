"""End-to-end orchestration of the four pipeline stages with incremental
replay, snapshot recovery and a canonical, arrival-order-independent result.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import authority as authority_stage
from . import loader as loader_stage
from . import scheduler as scheduler_stage
from . import view as view_stage
from .cache import CacheBook
from .errors import E_CONFIG_MISMATCH
from .fingerprint import canonical_json, content_fingerprint
from .report import build_report
from .snapshot import (STAGE_AUTHORITY, STAGE_LOAD, STAGE_SCORE, STAGE_VIEW,
                       SnapshotStore)


def preflight(workspace_root: str | Path) -> list[dict]:
    failures = []
    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".write_probe"
    try:
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        probe.unlink()
    except OSError as exc:
        failures.append({"code": E_CONFIG_MISMATCH,
                         "message": f"workspace not writable: {exc}"})
    try:
        str(root).encode("ascii")
    except UnicodeEncodeError:
        failures.append({"code": E_CONFIG_MISMATCH,
                         "message": "workspace path contains non-ASCII chars"})
    return failures


def _record_views(book: CacheBook, view_cache: dict) -> CacheBook:
    for key, entry in sorted(view_cache.items()):
        book.record_view(key, entry)
    return book


CACHE_BOOK_NAME = "cache_book.json"


def _load_previous_book(state_dir, recovery: dict):
    if recovery.get("mode") != "incremental":
        return None
    path = Path(state_dir) / CACHE_BOOK_NAME
    if not path.exists():
        return None
    book = CacheBook()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        book.view_cache = payload.get("view_cache", {})
        book.score_cache = payload.get("score_cache", {})
        book.ranking_cache = payload.get("ranking_cache", {})
        book.ranking_references = payload.get("ranking_references", {})
        return book
    except (json.JSONDecodeError, OSError):
        return None


def run_pipeline(config_path, batch_dir, requests_path, state_dir,
                 mode="full", concurrency_override=None,
                 previous_book: CacheBook | None = None):
    preflight_failures = preflight(Path(state_dir).parent)
    store = SnapshotStore(state_dir)
    store.cleanup_temp()

    load_result, requests = loader_stage.load_workspace(
        config_path, batch_dir, requests_path)
    config = load_result.config
    recovery = store.recovery_decision(config)
    if mode == "auto":
        mode = recovery["mode"]
    if previous_book is None:
        previous_book = _load_previous_book(state_dir, recovery)

    stage_paths = {}
    stage_paths[STAGE_LOAD] = store.write_stage(STAGE_LOAD, {
        "config_hash": store.config_hash(config),
        "slot_order": config.get("_slot_order", []),
        "batch_seqs": sorted(load_result.batches),
        "receipt_count": len(load_result.receipts),
        "failures": load_result.failures})

    authorities, rejections, backfill_ledger, duplicate_delivery = \
        authority_stage.adjudicate(load_result.receipts, load_result.slots)

    covered_groups = sorted({receipt.group_key
                             for receipt in load_result.receipts})
    stage_paths[STAGE_AUTHORITY] = store.write_stage(STAGE_AUTHORITY, {
        "authorities": {
            key: {
                "group_key": record.group_key,
                "slot_key": record.slot_key,
                "item_id": record.item_id,
                "winner": record.receipt.to_dict(),
                "authority_fingerprint": record.authority_fp,
                "tie_resolved": record.tie_resolved,
                "history": record.history,
                "pruned": record.pruned}
            for key, record in sorted(authorities.items())},
        "rejections": [item.__dict__ for item in rejections],
        "duplicate_deliveries": duplicate_delivery,
        "backfill_ledger": backfill_ledger})

    history_index = view_stage.build_history_index(
        authorities, load_result.receipts)
    resolutions, view_cache, staleness_fp = view_stage.build_view(
        config, load_result.slots, authorities, requests, history_index)
    stage_paths[STAGE_VIEW] = store.write_stage(STAGE_VIEW, {
        "staleness_fingerprint": staleness_fp,
        "resolutions": [
            {"key": [key[0], key[1] or "", key[2]],
             **resolution.__dict__}
            for key, resolution in sorted(
                resolutions.items(),
                key=lambda item: (item[0][0], item[0][1] or "", item[0][2]))],
        "view_cache": view_cache})

    cache_book = previous_book if previous_book is not None else CacheBook()
    _record_views(cache_book, view_cache)
    cache_book.mark_invalidated_groups(covered_groups)
    score_entries, executable_by_request, pending, quota_failures, ledger = \
        scheduler_stage.run(
            config, requests, load_result.slots, resolutions, staleness_fp,
            cache_book, concurrency_override=concurrency_override)
    ranking_results = scheduler_stage.rank_requests(
        config, requests, score_entries, load_result.slots, resolutions,
        cache_book)

    configured_concurrency = int(config["scheduler"]["max_concurrency"])
    max_concurrency = (configured_concurrency if concurrency_override is None
                       else max(1, min(configured_concurrency,
                                       int(concurrency_override))))
    all_failures = [*load_result.failures, *quota_failures]
    all_failures.extend(preflight_failures)
    ledger_with_runtime = dict(ledger,
                               max_concurrency_used=max_concurrency)
    cache_summary = cache_book.summary()
    stage_paths[STAGE_SCORE] = store.write_stage(STAGE_SCORE, {
        "score_entries": score_entries,
        "ranking_results": ranking_results,
        "pending_queue": pending,
        "failures": all_failures,
        "budget_ledger": ledger})

    manifest = store.read_manifest()
    completed_cursor = max(load_result.batches) if load_result.batches else -1
    input_fingerprint = content_fingerprint([
        store.config_hash(config), completed_cursor,
        sorted(load_result.batches),
        store.artifact_versions(stage_paths)])
    if manifest and manifest.get("input_fingerprint") == input_fingerprint:
        global_version = manifest.get("global_version", 1)
    else:
        global_version = (manifest or {}).get("global_version", 0) + 1
    report = build_report(
        load_result, requests, authorities, rejections,
        duplicate_delivery, resolutions, ranking_results, pending,
        all_failures, ledger_with_runtime,
        _cache_book_with_summary(cache_book, cache_summary),
        {"requested_mode": mode, **recovery})

    new_manifest = {
        "global_version": global_version,
        "format_version": config.get("format_version"),
        "config_hash": store.config_hash(config),
        "completed_batch_cursor": completed_cursor,
        "consumed_points": ledger["consumed_points"],
        "stage_artifact_versions": store.artifact_versions(stage_paths),
        "stage_paths": {stage: str(Path(path).relative_to(store.root))
                        for stage, path in stage_paths.items()},
        "pending_request_ids": [item["request_id"] for item in pending],
        "input_fingerprint": input_fingerprint,
    }
    store.commit_manifest(new_manifest)
    _persist_cache_book(store, cache_book)

    canonical = {
        "manifest": new_manifest,
        "load": {"failures": load_result.failures,
                 "receipt_count": len(load_result.receipts)},
        "authority": stage_payload(store, STAGE_AUTHORITY),
        "view": stage_payload(store, STAGE_VIEW),
        "score": stage_payload(store, STAGE_SCORE),
        "report": report,
    }
    return canonical, cache_book, store


def stage_payload(store: SnapshotStore, stage: str) -> dict:
    path = Path(store.root) / stage / "artifact.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def full_replay(config_path, batch_dir, requests_path, state_dir,
                concurrency_override=None):
    clean = Path(state_dir)
    if clean.exists():
        for child in clean.iterdir():
            if child.is_dir():
                __import__("shutil").rmtree(child)
            else:
                child.unlink()
    return run_pipeline(config_path, batch_dir, requests_path, state_dir,
                        mode="full",
                        concurrency_override=concurrency_override)


def incremental_replay(config_path, batch_dir, requests_path, state_dir,
                       previous_book, concurrency_override=None):
    return run_pipeline(config_path, batch_dir, requests_path, state_dir,
                        mode="auto",
                        concurrency_override=concurrency_override,
                        previous_book=previous_book)


def _persist_cache_book(store: SnapshotStore, book: CacheBook) -> None:
    payload = {
        "view_cache": book.view_cache,
        "score_cache": book.score_cache,
        "ranking_cache": book.ranking_cache,
        "ranking_references": book.ranking_references,
    }
    target = Path(store.root) / CACHE_BOOK_NAME
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(canonical_json(payload), encoding="utf-8")
    os.replace(tmp, target)


def _cache_book_with_summary(book: CacheBook, summary: dict) -> CacheBook:
    # The summary is attached only for the in-memory report; it must never be
    # part of a stage artifact, otherwise cache hits would perturb canonical
    # equivalence between full and incremental runs.
    book.report_summary = summary
    return book
