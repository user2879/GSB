from __future__ import annotations

import copy
from typing import Any

from .canonical import canonical_bytes, sha256_hex
from .diagnosis import diagnose
from .invalidation import derive_invalidation, recompute_incremental
from .ledger import Ledger
from .model import bundle_digest
from .runtime import ConcurrencyGovernor, ShaperProcessManager, execute_stage_calls


def spec_snapshot(bundle: dict[str, Any]) -> dict[str, Any]:
    snapshot = copy.deepcopy(bundle)
    snapshot.pop("spec_dir", None)
    snapshot["spec_digest"] = bundle_digest(snapshot)
    return snapshot


def diagnosis_artifacts(diagnosis: dict[str, Any]) -> dict[str, Any]:
    return {
        "path_selection": diagnosis["path_selection"],
        "stage_stats": diagnosis["stage_stats"],
        "error_bounds": diagnosis["error_bounds"],
        "windows": diagnosis["windows"],
    }


def run_full_into_ledger(
    bundle: dict[str, Any],
    ledger: Ledger,
    manager: ShaperProcessManager,
    *,
    revision: int = 1,
    revision_info: dict[str, Any] | None = None,
    batch_size: int | None = None,
    forced_path: str = "auto",
    governor: ConcurrencyGovernor | None = None,
    kill_shaper_before_batch: tuple[int, str] | None = None,
) -> dict[str, Any]:
    diagnosis = diagnose(bundle, forced_path=forced_path)
    results = diagnosis["events"]
    artifacts = diagnosis_artifacts(diagnosis)
    if revision_info is None:
        revision_info = {
            "parent_revision": 0,
            "reason": "initial_run",
            "spec_digest": bundle_digest(bundle),
            "change": {},
            "propagation": {},
            "selection": diagnosis["path_selection"],
        }
    batch_size = max(1, int(batch_size or bundle["config"]["batch_size"]))
    governor = governor or ConcurrencyGovernor(
        int(bundle["config"]["global_max_inflight"]),
        int(bundle["config"]["pipeline_max_inflight"]),
    )
    aggregate_errors: list[dict[str, Any]] = []
    call_count = 0
    for batch_index, start in enumerate(range(0, len(results), batch_size)):
        batch = results[start : start + batch_size]
        seq_start = int(batch[0]["event_seq"])
        seq_end = int(batch[-1]["event_seq"])
        batch_id = f"r{revision}-b{batch_index:03d}"
        ledger.plan_batch(batch_id, revision, seq_start, seq_end)
        if kill_shaper_before_batch and kill_shaper_before_batch[0] == batch_index:
            manager.kill(kill_shaper_before_batch[1])
        output = execute_stage_calls(bundle, batch, governor=governor)
        call_count += int(output["task_count"])
        if output["errors"]:
            aggregate_errors.extend(output["errors"])
            return {
                "ok": False,
                "failed_batch_id": batch_id,
                "failed_batch_index": batch_index,
                "call_output": output,
                "governor": governor.snapshot(),
                "errors": aggregate_errors,
                "diagnosis": diagnosis,
            }
        ledger.commit_event_batch(
            batch_id=batch_id,
            revision=revision,
            results=batch,
            call_results=output["calls"],
            revision_info=revision_info if batch_index == 0 else None,
            artifacts=artifacts if batch_index == 0 else None,
            sealed_seq=seq_end,
        )
    return {
        "ok": True,
        "revision": revision,
        "events": len(results),
        "calls": call_count,
        "governor": governor.snapshot(),
        "diagnosis": diagnosis,
    }


def recover_into_ledger(
    bundle: dict[str, Any],
    ledger: Ledger,
    manager: ShaperProcessManager,
    *,
    forced_path: str = "auto",
    governor: ConcurrencyGovernor | None = None,
) -> dict[str, Any]:
    unknown_batches = ledger.mark_unfinished_unknown()
    cursor = ledger.cursor()
    revision = int(cursor["revision"])
    sealed_seq = int(cursor["sealed_seq"])
    before_results = ledger.load_event_results(revision)
    sealed_prefix_before = [item for item in before_results if int(item["event_seq"]) <= sealed_seq]
    before_bytes = canonical_bytes(sealed_prefix_before)
    before_hash = sha256_hex(sealed_prefix_before)

    diagnosis = diagnose(bundle, forced_path=forced_path)
    results = diagnosis["events"]
    artifacts = diagnosis_artifacts(diagnosis)
    start_seq = max(0, sealed_seq + 1)
    suffix = [item for item in results if int(item["event_seq"]) >= start_seq]
    batch_size = max(1, int(bundle["config"]["batch_size"]))
    governor = governor or ConcurrencyGovernor(
        int(bundle["config"]["global_max_inflight"]),
        int(bundle["config"]["pipeline_max_inflight"]),
    )
    recovery_batches: list[str] = []
    errors: list[dict[str, Any]] = []
    if suffix:
        revision_row = ledger.conn.execute("SELECT revision FROM revisions WHERE revision=?", (revision,)).fetchone()
        revision_info = None
        if revision_row is None:
            revision_info = {
                "parent_revision": max(0, revision - 1),
                "reason": "recovery",
                "spec_digest": bundle_digest(bundle),
                "change": {},
                "propagation": {},
                "selection": diagnosis["path_selection"],
            }
        for offset, start in enumerate(range(0, len(suffix), batch_size)):
            batch = suffix[start : start + batch_size]
            seq_start = int(batch[0]["event_seq"])
            seq_end = int(batch[-1]["event_seq"])
            batch_id = f"r{revision}-recovery-{seq_start:03d}-{seq_end:03d}"
            ledger.plan_batch(
                batch_id,
                revision,
                seq_start,
                seq_end,
                kind="recovery",
                recovered_from=",".join(item["batch_id"] for item in unknown_batches) or None,
            )
            output = execute_stage_calls(bundle, batch, governor=governor)
            if output["errors"]:
                errors.extend(output["errors"])
                return {"ok": False, "errors": errors, "governor": governor.snapshot()}
            ledger.commit_event_batch(
                batch_id=batch_id,
                revision=revision,
                results=batch,
                call_results=output["calls"],
                revision_info=revision_info if offset == 0 else None,
                artifacts=artifacts if offset == 0 else None,
                sealed_seq=seq_end,
                recovered_from=",".join(item["batch_id"] for item in unknown_batches) or None,
            )
            recovery_batches.append(batch_id)

    after_results = ledger.load_event_results(revision)
    sealed_prefix_after = [item for item in after_results if int(item["event_seq"]) <= sealed_seq]
    after_bytes = canonical_bytes(sealed_prefix_after)
    prefix_identical = before_bytes == after_bytes
    return {
        "ok": prefix_identical and not errors,
        "unknown_batches": unknown_batches,
        "sealed_seq_before": sealed_seq,
        "rerun_event_count": len(suffix),
        "recovery_batches": recovery_batches,
        "sealed_prefix_byte_identical": prefix_identical,
        "sealed_prefix_sha256": before_hash,
        "governor": governor.snapshot(),
        "errors": errors,
    }


def _preserved_call_map(ledger: Ledger, revision: int) -> dict[str, dict[str, Any]]:
    rows = ledger.load_judgment_rows(revision)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        receipt = None
        if row.get("receipt_json"):
            import json

            receipt = json.loads(row["receipt_json"])
        result[f"{int(row['event_seq'])}:{row['stage']}"] = {
            "wait_start_ms": row.get("wait_start_ms"),
            "wait_end_ms": row.get("wait_end_ms"),
            "wait_ms": row.get("wait_ms"),
            "receipt": receipt,
            "receipt_digest": row.get("receipt_digest"),
        }
    return result


def commit_edit_revision(
    old_bundle: dict[str, Any],
    new_bundle: dict[str, Any],
    ledger: Ledger,
    manager: ShaperProcessManager,
    *,
    edit_manifest: dict[str, Any] | None = None,
    forced_path: str = "auto",
    governor: ConcurrencyGovernor | None = None,
) -> dict[str, Any]:
    parent_revision = ledger.current_revision()
    revision = parent_revision + 1
    parent_results = ledger.load_event_results(parent_revision)
    if not parent_results:
        parent_results = diagnose(old_bundle, forced_path=forced_path)["events"]
    closure = derive_invalidation(old_bundle, new_bundle)
    incremental = recompute_incremental(old_bundle, new_bundle, parent_results, closure)
    if not incremental["unchanged_check"]["all_unaffected_byte_identical"]:
        return {"ok": False, "error": "incremental unchanged-byte check failed", "incremental": incremental}
    diagnosis = diagnose(new_bundle, forced_path=forced_path)
    full_bytes = canonical_bytes(diagnosis["events"])
    incremental_bytes = canonical_bytes(incremental["results"])
    if full_bytes != incremental_bytes:
        return {"ok": False, "error": "incremental result differs from full replay", "closure": closure}

    affected_seqs = set(int(seq) for seq in closure["affected_event_seqs"])
    affected_results = [item for item in incremental["results"] if int(item["event_seq"]) in affected_seqs]
    only_stage_keys = {
        (int(cell["event_seq"]), cell["stage"])
        for cell in closure["affected_cells"]
        if int(cell["event_seq"]) in affected_seqs
    }
    governor = governor or ConcurrencyGovernor(
        int(new_bundle["config"]["global_max_inflight"]),
        int(new_bundle["config"]["pipeline_max_inflight"]),
    )
    output = execute_stage_calls(
        new_bundle,
        affected_results,
        governor=governor,
        only_stage_keys=only_stage_keys,
    )
    if output["errors"]:
        return {"ok": False, "call_output": output, "closure": closure}

    call_results = _preserved_call_map(ledger, parent_revision)
    call_results.update(output["calls"])
    for event_seq, stage in only_stage_keys:
        key = f"{event_seq}:{stage}"
        if key not in output["calls"]:
            call_results[key] = {
                "wait_start_ms": 0,
                "wait_end_ms": 0,
                "wait_ms": 0,
                "receipt": None,
                "receipt_digest": None,
            }
    revision_info = {
        "parent_revision": parent_revision,
        "reason": "spec_edit_incremental_recompute",
        "spec_digest": bundle_digest(new_bundle),
        "change": {
            "edit": True,
            "manifest": edit_manifest or {},
            "old_spec": spec_snapshot(old_bundle),
            "new_spec": spec_snapshot(new_bundle),
        },
        "propagation": closure,
        "selection": diagnosis["path_selection"],
    }
    ledger.commit_event_batch(
        batch_id=f"r{revision}-incremental",
        revision=revision,
        results=affected_results,
        call_results=call_results,
        revision_info=revision_info,
        artifacts=diagnosis_artifacts(diagnosis),
        sealed_seq=len(new_bundle["events"]) - 1,
        copy_from_revision=parent_revision,
    )
    return {
        "ok": True,
        "revision": revision,
        "closure": closure,
        "unchanged_check": incremental["unchanged_check"],
        "governor": governor.snapshot(),
        "call_output": {"task_count": output["task_count"], "governor": output["governor"]},
    }
