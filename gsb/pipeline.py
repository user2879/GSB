"""End-to-end deterministic diagnostic pipeline."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any

from .attribution import compare_compression_declaration, compare_index_statistics
from .codec import encode_block, write_block_file
from .engine import verification_summary, verify_query
from .hashing import canonical_hash
from .invalidation import plan_invalidation
from .ledger import Ledger
from .scheduler import Scheduler
from .snapshot import read_snapshot, write_snapshot
from .spec import load_spec, spec_digest


def _json_block(block: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in block.items() if key != "bytes"}


def _load_spec_source(spec: dict[str, Any] | str | Path | None, spec_path: str | Path | None) -> Any:
    if spec is not None:
        return load_spec(spec)
    if spec_path is not None:
        return load_spec(spec_path)
    raise ValueError("either spec or spec_path is required")


def _empty_recovery(session_id: str, query_ids: list[str], spec_digest: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "status": "requires_rerun",
        "requires_rerun": True,
        "snapshot_version": 1,
        "spec_digest": spec_digest,
        "unknown_batches": [],
        "committed_batches_replayed": [],
        "query_results_replayed": [],
        "missing_query_results": query_ids,
        "first_inconsistency": None,
    }


def build_verification_details(spec: dict[str, Any], blocks_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {group["id"]: group for group in spec["block_groups"]}
    details = []
    for block_group_id in sorted(blocks_by_id):
        block = blocks_by_id[block_group_id]
        group = groups[block_group_id]
        index_mismatches = compare_index_statistics(group.get("skip_index", {}), block.get("column_meta", {}))
        compression_mismatches, encoding_mismatches = compare_compression_declaration(group, block)
        storage_mismatches = []
        if block.get("content_digest") and block.get("header"):
            # Re-encoding the same logical columns must reproduce digest metadata.
            rebuilt = encode_block(group, block["columns"])
            if rebuilt["content_digest"] != block["content_digest"]:
                storage_mismatches.append({"field": "content_digest", "expected": rebuilt["content_digest"], "actual": block["content_digest"]})
        details.append(
            {
                "block_group_id": block_group_id,
                "declared_rows": group["row_count"],
                "decoded_rows": block["row_count"],
                "decoded_columns": len(block["columns"]),
                "content_digest": block["content_digest"],
                "logical_digest": block["logical_digest"],
                "index_distorted": bool(index_mismatches),
                "index_mismatches": index_mismatches,
                "compression_mismatch": bool(compression_mismatches),
                "encoding_mismatch": bool(encoding_mismatches),
                "storage_mismatches": storage_mismatches,
            }
        )
    return details


def run_pipeline(
    *,
    spec: dict[str, Any] | None = None,
    spec_path: str | Path | None = None,
    output_dir: str | Path = "artifacts/run",
    session_id: str | None = None,
    ledger_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    use_scheduler: bool = True,
    scheduler: Scheduler | None = None,
    decode_block_groups: list[str] | None = None,
    resume_committed: bool = True,
    crash_after_commit: int | None = None,
    write_files: bool = True,
    force_null_order: bool = False,
) -> dict[str, Any]:
    spec_model = _load_spec_source(spec, spec_path)
    spec = dict(spec_model)
    digest = spec_digest(spec)
    session_id = session_id or spec.get("session_id") or "session"
    output_dir = Path(output_dir)
    block_dir = output_dir / "blocks"
    ledger_path = Path(ledger_path) if ledger_path else output_dir / "ledger.sqlite3"
    cache_path = Path(cache_path) if cache_path else output_dir / "cache.json"
    report_path = output_dir / "report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    block_dir.mkdir(parents=True, exist_ok=True)

    old_cache = read_snapshot(cache_path)
    cache_snapshot_info = {
        "existed": old_cache is not None,
        "replaced": False,
        "session_id": session_id,
        "snapshot_version": 1,
        "spec_digest": digest,
    }
    snapshot_version = 1
    invalidation = None
    valid_old_cache = False
    if old_cache:
        cache_snapshot_info["snapshot_version"] = int(old_cache.get("snapshot_version", 1))
        valid_old_cache = (
            old_cache.get("format_version") == 1
            and old_cache.get("content_digest") is not None
            and old_cache.get("spec") is not None
            and old_cache.get("spec_digest") is not None
        )
        from .snapshot import snapshot_content_digest

        valid_old_cache = valid_old_cache and old_cache.get("content_digest") == snapshot_content_digest(old_cache)
        if valid_old_cache and old_cache.get("spec_digest") != digest:
            snapshot_version = int(old_cache.get("snapshot_version", 1)) + 1
            invalidation = plan_invalidation(
                old_cache.get("spec"),
                spec,
                old_spec_digest=old_cache.get("spec_digest"),
                new_spec_digest=digest,
            )
        elif valid_old_cache:
            snapshot_version = int(old_cache.get("snapshot_version", 1))
        else:
            cache_snapshot_info["replaced"] = True
            cache_snapshot_info["reason"] = "content_digest"
            snapshot_version = int(old_cache.get("snapshot_version", 0) or 0) + 1

    group_ids = [group["id"] for group in spec["block_groups"]]
    query_ids = [query["id"] for query in spec["queries"]]

    ledger = Ledger(ledger_path)
    ledger.initialize_session(
        session_id,
        digest,
        group_ids,
        snapshot_version=snapshot_version,
        format_version=1,
    )
    recovery_before = ledger.recover(session_id, query_ids)
    committed_ids = ledger.committed_batch_ids(session_id) if resume_committed else set()

    # Determine the minimal decode set.
    candidate_decode_ids = list(group_ids)
    forced_decode: set[str] = set(decode_block_groups or [])
    if invalidation is not None:
        forced_decode.update(invalidation.get("redecode_block_groups", []))
    if not valid_old_cache:
        # A corrupt/missing snapshot does not imply block bytes are unusable;
        # deterministic regeneration and ledger payloads still bound the work.
        pass

    reused_from_cache: dict[str, str] = {}
    blocks_by_id: dict[str, dict[str, Any]] = {}

    # Reuse valid cache blocks without decoding.
    if valid_old_cache:
        old_blocks = old_cache.get("blocks", {})
        for block_group_id, block in old_blocks.items():
            if block_group_id in group_ids and block_group_id not in forced_decode:
                blocks_by_id[block_group_id] = block
                reused_from_cache[block_group_id] = "cache"

    # Reuse committed ledger payload blocks for crash resume.
    for entry in recovery_before.get("committed_batches_replayed", []):
        block_group_id = entry["block_group_id"]
        if block_group_id in forced_decode or block_group_id in blocks_by_id:
            continue
        row = ledger.connection.execute(
            "SELECT payload_json FROM batches WHERE session_id=? AND block_group_id=?",
            (session_id, block_group_id),
        ).fetchone()
        if row:
            import json

            payload = json.loads(row["payload_json"])
            if payload.get("block"):
                blocks_by_id[block_group_id] = payload["block"]
                reused_from_cache[block_group_id] = "ledger"

    decode_ids: list[str] = []
    for block_group_id in candidate_decode_ids:
        if block_group_id in forced_decode:
            decode_ids.append(block_group_id)
        elif block_group_id in committed_ids and block_group_id in blocks_by_id:
            continue
        elif block_group_id in blocks_by_id:
            continue
        else:
            decode_ids.append(block_group_id)

    # Build deterministic bytes and local block files for decode jobs.
    generated_blocks: dict[str, dict[str, Any]] = {}
    group_by_id = {group["id"]: group for group in spec["block_groups"]}
    for block_group_id in decode_ids:
        block = encode_block(group_by_id[block_group_id])
        generated_blocks[block_group_id] = block
        if write_files:
            write_block_file(block_dir / f"{block_group_id}.gsb", block)

    receipts: dict[str, dict[str, Any]] = {}
    lease_peaks: dict[str, Any] = {}
    active_scheduler = scheduler
    owns_scheduler = scheduler is None and use_scheduler
    if use_scheduler:
        if active_scheduler is None:
            active_scheduler = Scheduler(
                global_decode_budget=spec["scheduler"]["global_decode_budget"],
                per_column_lease=spec["scheduler"]["per_column_lease"],
                socket_timeout_ms=spec["scheduler"]["socket_timeout_ms"],
                worker_start_timeout_ms=spec["scheduler"]["worker_start_timeout_ms"],
                worker_response_delay_ms=spec["scheduler"]["worker_response_delay_ms"],
            )
        jobs = []
        for block_group_id in decode_ids:
            block = generated_blocks[block_group_id]
            jobs.append(
                {
                    "block_group_id": block_group_id,
                    "columns": list(block["columns"]),
                    "block_bytes": block["bytes"],
                    "row_count": block["row_count"],
                    "response_delay_ms": spec["scheduler"]["worker_response_delay_ms"],
                }
            )
        receipt_list = active_scheduler.run(jobs)
        receipts = {receipt["block_group_id"]: receipt for receipt in receipt_list}
        lease_peaks = active_scheduler.lease_peaks()
        if owns_scheduler:
            active_scheduler.close()
    else:
        for block_group_id in decode_ids:
            block = generated_blocks[block_group_id]
            receipts[block_group_id] = {
                "block_group_id": block_group_id,
                "decoded_rows": block["row_count"],
                "elapsed_ms": 0,
                "transport": "local/direct",
                "worker_port": None,
            }

    for block_group_id, block in generated_blocks.items():
        receipt = receipts.get(block_group_id, {})
        if int(receipt.get("decoded_rows", block["row_count"])) != block["row_count"]:
            raise ValueError(f"worker receipt row mismatch for {block_group_id}")
        blocks_by_id[block_group_id] = _json_block(block)

    planned_batches = len(group_ids)
    committed_this_run = 0
    for sequence, block_group_id in enumerate(group_ids, start=1):
        if block_group_id not in decode_ids:
            continue
        block = generated_blocks[block_group_id]
        receipt = receipts[block_group_id]
        existing_count = ledger.connection.execute(
            "SELECT COUNT(*) AS n FROM batches WHERE session_id=?",
            (session_id,),
        ).fetchone()["n"]
        cursor = {
            "sequence": sequence,
            "last_sequence": sequence,
            "completed_batches": existing_count + 1,
            "planned_batches": planned_batches,
        }
        occupancy = {
            "decoded_rows": block["row_count"],
            "column_count": len(block["columns"]),
            "stored_bytes": sum(meta["stored_size"] for meta in block["column_meta"].values()),
            "receipt_elapsed_ms": receipt.get("elapsed_ms"),
        }
        lease = {
            "status": "released",
            "columns": sorted(block["columns"]),
            "worker_port": receipt.get("worker_port"),
            "transport": receipt.get("transport"),
            "global_budget": spec["scheduler"]["global_decode_budget"],
            "per_column_lease": spec["scheduler"]["per_column_lease"],
        }
        ledger.commit_batch(
            session_id,
            sequence,
            block_group_id,
            content_digest=block["content_digest"],
            logical_digest=block["logical_digest"],
            decoded_rows=block["row_count"],
            payload={"block": _json_block(block), "receipt": receipt},
            occupancy=occupancy,
            lease=lease,
            cursor=cursor,
            row_version=1,
        )
        committed_this_run += 1
        if crash_after_commit is not None and committed_this_run >= crash_after_commit:
            ledger.connection.commit()
            os.kill(os.getpid(), signal.SIGKILL)

    # Build query payloads, reusing unaffected canonical payloads byte-for-byte.
    query_payloads: dict[str, dict[str, Any]] = {}
    reused_queries: list[str] = []
    unaffected = set((invalidation or {}).get("unaffected_queries", []))
    old_query_payloads = (old_cache or {}).get("query_payloads", {}) if valid_old_cache else {}
    global_rows = sum(block["row_count"] for block in blocks_by_id.values())
    for query in spec["queries"]:
        query_id = query["id"]
        if query_id in unaffected and query_id in old_query_payloads:
            query_payloads[query_id] = old_query_payloads[query_id]
            reused_queries.append(query_id)
            continue
        payload = verify_query(
            query,
            spec["block_groups"],
            blocks_by_id,
            global_decoded_block_groups=len(blocks_by_id),
            global_decoded_rows=global_rows,
            force_null_order=force_null_order,
        )
        query_payloads[query_id] = payload
        result_digest = canonical_hash(payload)
        ledger.commit_query_result(session_id, query_id, result_digest, payload, digest)

    def verify_batch_row(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
        block_payload = payload.get("block")
        if not block_payload:
            return {"field": "payload.block", "expected": "object", "actual": None}
        group = group_by_id[row["block_group_id"]]
        rebuilt = encode_block(group, block_payload["columns"])
        if rebuilt["content_digest"] != row["content_digest"]:
            return {"field": "content_digest", "expected": rebuilt["content_digest"], "actual": row["content_digest"]}
        if rebuilt["logical_digest"] != row["logical_digest"]:
            return {"field": "logical_digest", "expected": rebuilt["logical_digest"], "actual": row["logical_digest"]}
        if rebuilt["row_count"] != row["decoded_rows"]:
            return {"field": "decoded_rows", "expected": rebuilt["row_count"], "actual": row["decoded_rows"]}
        return None

    recovery = ledger.recover(
        session_id,
        query_ids,
        verify_batch=verify_batch_row,
    )

    verification = verification_summary(blocks_by_id, spec)
    all_issues = [issue for payload in query_payloads.values() for issue in payload["issues"]]
    all_attribution = [entry for payload in query_payloads.values() for entry in payload["attribution"]]
    all_attribution.sort(key=lambda e: (e["block_group_id"], e["query_id"]))
    dominant_sources: dict[str, int] = {}
    for entry in all_attribution:
        dominant_sources[entry["dominant_source"]] = dominant_sources.get(entry["dominant_source"], 0) + 1

    planned_skip_union = sorted({gid for payload in query_payloads.values() for gid in payload["plan"]["skip_block_groups"]})
    execution = {
        "decode_jobs_this_run": decode_ids,
        "reused_block_groups": reused_from_cache,
        "reused_queries": reused_queries,
        "worker_receipts": dict(sorted(receipts.items())),
        "lease_peaks": lease_peaks,
    }
    report = {
        "ok": True,
        "session_id": session_id,
        "format_version": 1,
        "snapshot_version": snapshot_version,
        "spec_digest": digest,
        "summary": {
            "queries": len(query_ids),
            "plans": len(query_ids),
            "false_skips": sum(issue["kind"] == "false_skip" for issue in all_issues),
            "false_retains": sum(issue["kind"] == "false_retain" for issue in all_issues),
            "attribution_entries": len(all_attribution),
            "decoded_block_groups_this_run": len(decode_ids),
            "reused_block_groups": len(blocks_by_id) - len(decode_ids),
            "reused_query_payloads": len(reused_queries),
            "dominant_sources": dict(sorted(dominant_sources.items())),
        },
        "verification": {
            **verification,
            "declared_verification_scope": "full",
            "planned_skip_block_groups": len(planned_skip_union),
            "planned_skip_block_groups_verified": len(planned_skip_union),
            "unverified_planned_skip_block_groups": [],
            "block_groups": build_verification_details(spec, blocks_by_id),
        },
        "plans": {qid: {"digest": payload["plan_digest"], "skip_block_groups": payload["plan"]["skip_block_groups"], "verify_block_groups": payload["plan"]["verify_block_groups"]} for qid, payload in sorted(query_payloads.items())},
        "queries": [query_payloads[qid] for qid in sorted(query_payloads)],
        "issues": all_issues,
        "attribution": all_attribution,
        "diagnostic_digest": canonical_hash({"issues": all_issues, "attribution": all_attribution}),
        "invalidation": invalidation,
        "execution": execution,
        "ledger": {
            "schema": "sqlite",
            "path": str(ledger_path),
            "new_batches_committed": committed_this_run,
            "recovery": recovery,
            "snapshot": {
                **cache_snapshot_info,
                "snapshot_version": snapshot_version,
                "spec_digest": digest,
                "existed": old_cache is not None,
                "replaced": bool(cache_snapshot_info.get("replaced")),
            },
        },
    }

    cache_snapshot = {
        "format_version": 1,
        "session_id": session_id,
        "snapshot_version": snapshot_version,
        "spec_digest": digest,
        "spec": spec,
        "blocks": {gid: _json_block(blocks_by_id[gid]) for gid in sorted(blocks_by_id)},
        "query_digests": {qid: payload["digest"] for qid, payload in sorted(query_payloads.items())},
        "query_payloads": {qid: payload for qid, payload in sorted(query_payloads.items())},
    }
    if write_files:
        write_snapshot(cache_path, cache_snapshot)
        report["cache_path"] = str(cache_path.resolve())
        report["report_path"] = str(report_path.resolve())
        import json

        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        report["cache_snapshot"] = cache_snapshot

    # Recovery information after snapshot/query commit, with query verification.
    final_recovery = ledger.recover(session_id, query_ids)
    report["ledger"]["recovery"] = final_recovery
    ledger.close()
    return report


def recover_pipeline(
    *,
    output_dir: str | Path = "artifacts/run",
    session_id: str = "session",
    spec: dict[str, Any] | None = None,
    spec_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    ledger_path = output_dir / "ledger.sqlite3"
    if spec is None and spec_path is None:
        cache_path = output_dir / "cache.json"
        cached = read_snapshot(cache_path)
        if cached and cached.get("spec"):
            spec = cached["spec"]
    spec_model = _load_spec_source(spec, spec_path)
    spec = dict(spec_model)
    digest = spec_digest(spec)
    ledger = Ledger(ledger_path)
    groups = {group["id"]: group for group in spec["block_groups"]}

    def verify_batch(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
        block_payload = payload.get("block")
        if not block_payload:
            return {"field": "payload.block", "expected": "object", "actual": None}
        rebuilt = encode_block(groups[row["block_group_id"]], block_payload["columns"])
        if rebuilt["content_digest"] != row["content_digest"]:
            return {"field": "content_digest", "expected": rebuilt["content_digest"], "actual": row["content_digest"]}
        if rebuilt["logical_digest"] != row["logical_digest"]:
            return {"field": "logical_digest", "expected": rebuilt["logical_digest"], "actual": row["logical_digest"]}
        return None

    recovery = ledger.recover(
        session_id,
        [q["id"] for q in spec["queries"]],
        verify_batch=verify_batch,
    )
    snapshot = read_snapshot(output_dir / "cache.json")
    snapshot_info = {
        "existed": snapshot is not None,
        "replaced": False,
        "session_id": session_id,
        "snapshot_version": recovery.get("snapshot_version", 1),
        "spec_digest": digest,
    }
    result = {"ok": not recovery["requires_rerun"], "recovery": recovery, "snapshot": snapshot_info}
    ledger.close()
    return result
