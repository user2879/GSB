from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, digest_value, monotonic_ns
from .errors import ReplayError
from .incremental import read_fast_once
from .ledger import LEDGER_FORMAT_VERSION, Ledger
from .pipeline import compute_conclusion, first_divergence
from .workspace import Workspace


def batch_outcome(batch: dict[str, Any]) -> dict[str, Any]:
    docs = batch.get("docs", [])
    return {
        "format_version": 1,
        "batch_id": batch["batch_id"],
        "domain_id": batch["domain_id"],
        "batch_seq": int(batch["batch_seq"]),
        "stream_ordinal": int(batch.get("stream_ordinal", 0)),
        "batch_digest": batch.get("batch_digest") or digest_value(batch),
        "doc_count": len(docs),
        "doc_digests": [doc["content_digest"] for doc in docs],
        "status": "processed",
    }


def open_initialized_ledger(workspace: Workspace) -> Ledger:
    workspace.ensure_dirs()
    ledger = Ledger(workspace.ledger_path)
    ledger.initialize()
    bundle = workspace.load_bundle()
    ledger.init_batch_plan(bundle.batches)
    return ledger


def process_pending_batches(
    workspace: Workspace,
    ledger: Ledger,
    *,
    crash_batch_id: str | None = None,
    marker_path: Path | None = None,
) -> list[str]:
    bundle = workspace.load_bundle()
    batches_by_id = {batch["batch_id"]: batch for batch in bundle.batches}
    pending = ledger.pending_batches()
    processed: list[str] = []
    for row in pending:
        batch_id = row["batch_id"]
        batch = batches_by_id[batch_id]
        outcome = batch_outcome(batch)
        now_ms = monotonic_ns() // 1_000_000
        if crash_batch_id == batch_id and marker_path is not None:
            ledger.insert_uncommitted_batch_before_kill(
                batch_id=batch_id,
                ordinal=int(row["ordinal"]),
                domain_id=row["domain_id"],
                batch_seq=int(row["batch_seq"]),
                outcome=outcome,
                config_version=int(bundle.config["config_version"]),
                monotonic_ms_value=now_ms,
                marker_path=marker_path,
            )
            continue  # pragma: no cover
        ledger.commit_batch_outcome(
            batch_id=batch_id,
            ordinal=int(row["ordinal"]),
            domain_id=row["domain_id"],
            batch_seq=int(row["batch_seq"]),
            outcome=outcome,
            config_version=int(bundle.config["config_version"]),
            monotonic_ms_value=now_ms,
        )
        processed.append(batch_id)
    return processed


def record_fast_read(workspace: Workspace, ledger: Ledger) -> dict[str, Any]:
    read = read_fast_once(workspace)
    start_ns = read.get("start_ns")
    end_ns = read.get("end_ns")
    read_id = ledger.record_ui_read(
        config_version=int(read["config_version"]),
        cache_digest=str(read["cache_digest"]),
        elapsed_ms=float(read["elapsed_ms"]),
        bytes_read=int(read["bytes_read"]),
        start_ns=start_ns,
        end_ns=end_ns,
    )
    read["read_id"] = read_id
    return read


def _doc_to_batch_id(conclusion: dict[str, Any]) -> dict[str, str]:
    return {item["doc_id"]: item["batch_id"] for item in conclusion["weighted_sampling"]["selected"]}


def _batch_order(bundle: Any) -> dict[str, int]:
    return {batch["batch_id"]: int(batch.get("stream_ordinal", index)) for index, batch in enumerate(bundle.batches, start=1)}


def _latest_batch_for_docs(doc_ids: list[str], doc_batch: dict[str, str], order: dict[str, int], fallback: str) -> str:
    candidates = [doc_batch[doc_id] for doc_id in doc_ids if doc_id in doc_batch]
    if not candidates:
        return fallback
    return max(candidates, key=lambda batch_id: order.get(batch_id, 0))


def build_stage_items(workspace: Workspace, conclusion: dict[str, Any], propagation: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    bundle = workspace.load_bundle()
    order = _batch_order(bundle)
    fallback_batch = max(order, key=order.get) if order else "BATCH:UNKNOWN"
    doc_batch = _doc_to_batch_id(conclusion)
    items: list[dict[str, Any]] = []

    for domain_id, summary in sorted(conclusion["weighted_sampling"]["domains"].items()):
        doc_ids = [item["doc_id"] for item in conclusion["weighted_sampling"]["selected"] if item["domain_id"] == domain_id]
        items.append(
            {
                "item_key": f"weighted:{domain_id}",
                "stage": "weighted_sampling",
                "domain_id": domain_id,
                "shard_id": None,
                "batch_id": _latest_batch_for_docs(doc_ids, doc_batch, order, fallback_batch),
                "field_name": "quota_points",
                "value": summary,
            }
        )

    for shard in conclusion["shards"]:
        items.append(
            {
                "item_key": f"dedup:{shard['shard_id']}",
                "stage": "near_duplicate_dedup",
                "domain_id": None,
                "shard_id": shard["shard_id"],
                "batch_id": _latest_batch_for_docs(shard["member_ids"], doc_batch, order, fallback_batch),
                "field_name": "shard_digest",
                "value": shard,
            }
        )

    for domain_item in conclusion["ratio_normalization"]["domains"]:
        kept_ids = list(domain_item.get("kept_ids", []))
        # Ratio domain items do not duplicate kept ids; derive from shard results.
        if not kept_ids:
            for shard in conclusion["shards"]:
                if domain_item["domain_id"] in shard.get("domains", []):
                    kept_ids.extend(doc_id for doc_id in shard["kept_ids"] if doc_id.startswith(domain_item["domain_id"] + ":"))
        items.append(
            {
                "item_key": f"ratio:{domain_item['domain_id']}",
                "stage": "ratio_normalization",
                "domain_id": domain_item["domain_id"],
                "shard_id": None,
                "batch_id": _latest_batch_for_docs(kept_ids, doc_batch, order, fallback_batch),
                "field_name": "deviation_bp",
                "value": domain_item,
            }
        )

    items.append(
        {
            "item_key": "error_attribution:all",
            "stage": "error_attribution",
            "domain_id": None,
            "shard_id": None,
            "batch_id": fallback_batch,
            "field_name": "total_bound_micro_bp",
            "value": conclusion["error_attribution"],
        }
    )
    return items


def persist_current_view(workspace: Workspace, ledger: Ledger, *, record_read: bool = False) -> dict[str, Any]:
    view = workspace.load_fast_view()
    conclusion = view["conclusion"]
    propagation = view.get("last_invalidation")
    bundle = workspace.load_bundle()
    existing_outcomes = {row["batch_id"]: row for row in ledger.latest_batch_outcomes()}
    outcome_revisions: list[dict[str, Any]] = []
    for batch in bundle.batches:
        if batch["batch_id"] not in existing_outcomes:
            continue
        outcome = batch_outcome(batch)
        if existing_outcomes[batch["batch_id"]]["payload_digest"] != digest_value(outcome):
            outcome_revisions.append(
                {
                    "batch_id": batch["batch_id"],
                    "ordinal": int(batch.get("stream_ordinal", 0)),
                    "domain_id": batch["domain_id"],
                    "batch_seq": int(batch["batch_seq"]),
                    "status": "committed_revision",
                    "outcome": outcome,
                }
            )
    if propagation is None:
        shard_ids = [shard["shard_id"] for shard in conclusion["shards"]]
        propagation = {
            "format_version": 1,
            "config_version": int(conclusion["config_version"]),
            "change": {"kind": "initialize", "field": "$", "old_value": None, "new_value": int(conclusion["config_version"])},
            "affected_domains": sorted(conclusion["weighted_sampling"]["domains"]),
            "affected_shards": shard_ids,
            "recomputed_shards": shard_ids,
            "preserved_shards": [],
            "affected_ratio_domains": [item["domain_id"] for item in conclusion["ratio_normalization"]["domains"]],
            "steps": [{"seq": 1, "from": "raw_batches", "to": "full_cache", "edge": "initial_materialization", "basis": {"shards": len(shard_ids)}}],
        }
        propagation["propagation_digest"] = digest_value(propagation)
    items = build_stage_items(workspace, conclusion, propagation)
    read_metrics = read_fast_once(workspace) if record_read else None
    ui_read = None
    if read_metrics is not None:
        ui_read = {
            "config_version": read_metrics["config_version"],
            "cache_digest": read_metrics["cache_digest"],
            "elapsed_ms": read_metrics["elapsed_ms"],
            "bytes_read": read_metrics["bytes_read"],
            "start_ns": read_metrics["start_ns"],
            "end_ns": read_metrics["end_ns"],
        }
    ledger.commit_invalidation_and_items(
        propagation=propagation,
        items=items,
        monotonic_ms_value=monotonic_ns() // 1_000_000,
        batch_outcomes=outcome_revisions,
        ui_read=ui_read,
    )
    return {
        "config_version": conclusion["config_version"],
        "items": len(items),
        "revised_batches": [row["batch_id"] for row in outcome_revisions],
        "propagation_digest": propagation["propagation_digest"],
        "read_elapsed_ms": read_metrics["elapsed_ms"] if read_metrics else None,
        "read_bytes": read_metrics["bytes_read"] if read_metrics else None,
    }


def _compare_ratio_field(actual: dict[str, Any], ledger_value: dict[str, Any]) -> str | None:
    for field in ("actual_bp", "target_bp", "deviation_bp", "kept_count", "selected_count"):
        if actual.get(field) != ledger_value.get(field):
            return field
    if canonical_bytes(actual) != canonical_bytes(ledger_value):
        return "canonical_json"
    return None


def replay_ledger(workspace: Workspace, ledger: Ledger, *, config_version: int | None = None) -> dict[str, Any]:
    unknown_or_pending = ledger.pending_batches()
    if unknown_or_pending:
        first = unknown_or_pending[0]
        raise ReplayError(
            "ledger contains unknown or pending batches; rerun them before replay",
            batch_id=first["batch_id"],
            field="status",
            pending_count=len(unknown_or_pending),
            unknown_batch_ids=[row["batch_id"] for row in unknown_or_pending],
        )

    bundle = workspace.load_bundle()
    candidates = workspace.load_candidates()
    conclusion = compute_conclusion(bundle, candidates)
    raw_batches = {batch["batch_id"]: batch for batch in bundle.batches}

    checked_batches = 0
    for row in ledger.latest_batch_outcomes():
        batch_id = row["batch_id"]
        raw_batch = raw_batches.get(batch_id)
        if raw_batch is None:
            raise ReplayError("ledger batch is absent from raw stream", batch_id=batch_id, field="batch_id")
        expected = batch_outcome(raw_batch)
        if row["payload_digest"] != digest_value(expected):
            ledger_payload = json.loads(row["payload_json"])
            field = "payload_digest"
            for key in ("batch_digest", "doc_count", "doc_digests", "stream_ordinal"):
                if ledger_payload.get(key) != expected.get(key):
                    field = key
                    break
            raise ReplayError(
                "batch outcome replay mismatch",
                code="E_REPLAY",
                batch_id=batch_id,
                field=field,
                ledger_digest=row["payload_digest"],
                authoritative_digest=digest_value(expected),
            )
        checked_batches += 1

    version = config_version or ledger.latest_config_version_with_items()
    if version is None:
        raise ReplayError("ledger has no stage items to replay", field="stage_items")
    items = ledger.stage_items(version)
    actual_shards = {shard["shard_id"]: shard for shard in conclusion["shards"]}
    actual_ratio = {item["domain_id"]: item for item in conclusion["ratio_normalization"]["domains"]}
    actual_weighted = conclusion["weighted_sampling"]["domains"]
    checked_items = 0
    for row in items:
        key = row["item_key"]
        if key.startswith("dedup:"):
            shard_id = row["shard_id"]
            actual_item = actual_shards.get(shard_id)
            stage_probe = first_divergence({"shards": [actual_item]}, {"shards": [json.loads(row["canonical_json"])]}) if actual_item else None
            field = (stage_probe or {}).get("field", "shard_presence")
            if actual_item is None or row["canonical_json"].encode("utf-8") != canonical_bytes(actual_item):
                raise ReplayError(
                    "stage item replay mismatch",
                    batch_id=row["batch_id"],
                    config_version=version,
                    stage=row["stage"],
                    shard_id=shard_id,
                    domain_id=row["domain_id"],
                    item_key=key,
                    field=field,
                    ledger_digest=row["digest"],
                    authoritative_digest=digest_value(actual_item) if actual_item else None,
                )
        elif key.startswith("ratio:"):
            domain_id = row["domain_id"]
            actual_item = actual_ratio.get(domain_id)
            ledger_item = json.loads(row["canonical_json"])
            field = _compare_ratio_field(actual_item or {}, ledger_item) or row["field_name"]
            if actual_item is None or canonical_bytes(actual_item) != row["canonical_json"].encode("utf-8"):
                raise ReplayError(
                    "ratio replay mismatch",
                    batch_id=row["batch_id"],
                    config_version=version,
                    stage=row["stage"],
                    domain_id=domain_id,
                    item_key=key,
                    field=field,
                    ledger_digest=row["digest"],
                    authoritative_digest=digest_value(actual_item) if actual_item else None,
                )
        elif key.startswith("weighted:"):
            domain_id = row["domain_id"]
            actual_item = actual_weighted.get(domain_id)
            if actual_item is None or row["canonical_json"].encode("utf-8") != canonical_bytes(actual_item):
                raise ReplayError(
                    "weighted stage replay mismatch",
                    batch_id=row["batch_id"],
                    config_version=version,
                    stage=row["stage"],
                    domain_id=domain_id,
                    item_key=key,
                    field=row["field_name"],
                    ledger_digest=row["digest"],
                    authoritative_digest=digest_value(actual_item) if actual_item else None,
                )
        else:
            actual_item = conclusion["error_attribution"]
            if row["canonical_json"].encode("utf-8") != canonical_bytes(actual_item):
                raise ReplayError(
                    "error attribution replay mismatch",
                    batch_id=row["batch_id"],
                    config_version=version,
                    stage=row["stage"],
                    item_key=key,
                    field=row["field_name"],
                    ledger_digest=row["digest"],
                    authoritative_digest=digest_value(actual_item),
                )
        checked_items += 1

    return {
        "identical": True,
        "config_version": version,
        "checked_batches": checked_batches,
        "checked_items": checked_items,
        "ledger_format_version": LEDGER_FORMAT_VERSION,
        "conclusion_digest": conclusion["conclusion_digest"],
    }
