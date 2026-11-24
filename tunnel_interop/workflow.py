"""End-to-end orchestration for real forwarding, evolution and recovery."""

from __future__ import annotations

import base64
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .errors import E_FIELD, E_REF, E_SNAPSHOT, E_SPEC, TunnelError, fail
from .fragmenter import (
    FragmentBundle,
    FragmentRecord,
    MessageBundle,
    build_bundle,
    context_for_plan,
    migrate_all_variants,
    reassemble_fragments,
)
from .invalidation import evolve_bundle
from .ledger import ReplayResult, SessionLedger
from .logging_utils import StructuredLogger
from .scheduler import ScheduledTask, run_scheduler
from .spec import TunnelSpec, load_spec
from .transport import ReceiverPool, SendReceipt, decode_receipts, send_fragment
from .util import atomic_write_json, canonical_json, monotonic_ms, sha256_json
from .wire import decode_fragment


def batch_context(plan, target_format: str, total_fragments: int) -> dict[str, Any]:
    return {
        "message_key": plan.message_key,
        "tunnel_id": plan.tunnel_id,
        "destination_prefix": plan.destination_prefix,
        "next_hop": plan.endpoint.endpoint_id,
        "endpoint": plan.endpoint.canonical(),
        "route": plan.route.canonical(),
        "payload_b64": base64.b64encode(plan.payload).decode("ascii"),
        "payload_sha256": plan.payload_sha256,
        "message_id": plan.message_id,
        "message_id_basis": plan.message_id_basis,
        "rng_seed": plan.rng_seed,
        "rng_seed_basis": plan.rng_seed_basis,
        "generated_at_ms": plan.generated_at_ms,
        "fragment_size": plan.fragment_size,
        "total_fragments": total_fragments,
        "target_format": target_format,
        "encapsulation_version": 1,
    }


def _receipt_row(
    *,
    plan,
    record: FragmentRecord,
    receipt: SendReceipt,
    total_fragments: int,
) -> dict[str, Any]:
    normalized = decode_fragment(receipt.wire, context_for_plan(plan, total_fragments))
    return {
        "fragment_index": record.fragment_index,
        "message_key": record.message_key,
        "fragment_offset": record.fragment_offset,
        "fragment_length": record.fragment_length,
        "total_fragments": total_fragments,
        "message_id": record.message_id,
        "tunnel_id": record.tunnel_id,
        "target_format": record.target_format,
        "retry_count": normalized.retry_count,
        "fragment_version": record.fragment_version,
        "wire_b64": base64.b64encode(receipt.wire).decode("ascii"),
        "wire_sha256": receipt.wire_sha256,
        "payload_sha256": normalized.payload_sha256,
        "header_json": canonical_json(normalized.to_dict()).decode("utf-8"),
    }


def _reassembly_row(message: MessageBundle, actual) -> dict[str, Any]:
    return {
        "message_key": message.plan.message_key,
        "message_id": actual.message_id,
        "tunnel_id": actual.tunnel_id,
        "payload_sha256": actual.payload_sha256,
        "payload_b64": base64.b64encode(actual.payload).decode("ascii"),
        "reassembly_version": actual.reassembly_version,
        "complete": actual.complete,
    }


def _task_list(
    message: MessageBundle,
    scenario: str,
) -> tuple[list[ScheduledTask], dict[str, FragmentRecord]]:
    fragments = {fragment.fragment_index: fragment for fragment in message.fragments}
    records: dict[str, FragmentRecord] = {}
    tasks: list[ScheduledTask] = []
    if scenario == "normal":
        entries = [(index, 0) for index in sorted(fragments)]
    elif scenario == "reverse":
        entries = [(index, 0) for index in sorted(fragments, reverse=True)]
    elif scenario == "duplicate":
        entries = []
        for index in sorted(fragments):
            entries.append((index, 0))
            entries.append((index, 1))
    else:
        fail(E_FIELD, "forward.scenario", "unknown forwarding scenario", actual=scenario)
    for order, (index, copy) in enumerate(entries):
        record = fragments[index]
        uid = f"{message.plan.message_key}:{index}:copy{copy}"
        records[uid] = record
        tasks.append(
            ScheduledTask(
                task_uid=uid,
                message_key=message.plan.message_key,
                tunnel_id=message.plan.tunnel_id,
                fragment_index=index,
                order=order,
                payload=record,
            )
        )
    return tasks, records


def forward_bundle(
    *,
    spec: TunnelSpec,
    bundle: FragmentBundle,
    ledger_path: str | Path,
    scenario: str = "normal",
    logger: StructuredLogger | None = None,
    message_keys: Iterable[str] | None = None,
    ack_mode: str = "normal",
    drop_indices: set[int] | None = None,
    drop_all: bool = False,
    crash_after_batch: int | None = None,
) -> dict[str, Any]:
    logger = logger or StructuredLogger()
    selected = set(message_keys) if message_keys is not None else None
    messages = [message for message in bundle.messages if selected is None or message.plan.message_key in selected]
    if not messages:
        fail(E_FIELD, "forward.message_keys", "no selected messages exist in the bundle")
    if selected is not None and selected - {message.plan.message_key for message in messages}:
        fail(
            E_FIELD,
            "forward.message_keys",
            "message selection contains unknown keys",
            unknown=sorted(selected - {message.plan.message_key for message in messages}),
        )
    endpoints = tuple(
        spec.endpoint_map[message.plan.endpoint.endpoint_id]
        for message in messages
    )
    endpoints = tuple(sorted({endpoint.endpoint_id: endpoint for endpoint in endpoints}.values(), key=lambda item: item.endpoint_id))

    ledger = SessionLedger(ledger_path)
    open_result = ledger.open(spec)
    logger.event(
        "ledger_opened",
        ledger_path=str(ledger_path),
        generation=open_result.generation,
        replaced=open_result.replaced,
        unknown_batches=len(open_result.unknown_batches),
    )
    message_reports: list[dict[str, Any]] = []
    commits: list[dict[str, Any]] = []
    current_batch_id: str | None = None
    pool = ReceiverPool(
        endpoints,
        ack_mode=ack_mode,
        drop_indices=drop_indices,
        drop_all=drop_all,
        idle_timeout_ms=spec.runtime.receiver_idle_timeout_ms,
        log_file=None,
    )
    try:
        pool.start()
        for batch_index, message in enumerate(messages):
            plan = message.plan
            batch_id = plan.message_key
            if crash_after_batch is not None and batch_index >= crash_after_batch:
                pool.stop()
                ledger.begin_batch(
                    batch_id=batch_id,
                    message_key=plan.message_key,
                    context=batch_context(plan, bundle.target_format, len(message.fragments)),
                    fragment_indices=[fragment.fragment_index for fragment in message.fragments],
                    target_format=bundle.target_format,
                )
                logger.event("forced_exit_before_batch_commit", batch_id=batch_id)
                os._exit(97)
            current_batch_id = batch_id
            ledger.begin_batch(
                batch_id=batch_id,
                message_key=plan.message_key,
                context=batch_context(plan, bundle.target_format, len(message.fragments)),
                fragment_indices=[fragment.fragment_index for fragment in message.fragments],
                target_format=bundle.target_format,
            )
            tasks, record_map = _task_list(message, scenario)
            logger.event("batch_started", batch_id=batch_id, task_count=len(tasks), scenario=scenario)
            reassembly_deadline = (
                monotonic_ms() + spec.fragment_policy.reassembly_timeout_ms
                if spec.fragment_policy.reassembly_timeout_ms > 0
                else None
            )

            def worker(task: ScheduledTask) -> SendReceipt:
                record = record_map[task.task_uid]
                receipts = send_fragment(
                    plan=plan,
                    fragment=record.normalized,
                    target_format=bundle.target_format,
                    timeout_ms=spec.runtime.ack_timeout_ms,
                    max_retries=spec.fragment_policy.max_retries,
                    reassembly_deadline_monotonic_ms=reassembly_deadline,
                )
                return receipts[0]

            scheduler_result = run_scheduler(
                tasks,
                worker,
                global_inflight_limit=(
                    1 if scenario == "reverse" else spec.runtime.global_inflight_limit
                ),
                per_tunnel_inflight_quota=(
                    1 if scenario == "reverse" else spec.runtime.per_tunnel_inflight_quota
                ),
            )
            receipts = [outcome.value for outcome in scheduler_result.outcomes]
            if reassembly_deadline is not None and monotonic_ms() > reassembly_deadline:
                fail(
                    E_TIMEOUT,
                    "reassembly.deadline",
                    "declared reassembly timeout elapsed before the complete fragment set arrived",
                    batch_id=batch_id,
                    timeout_ms=spec.fragment_policy.reassembly_timeout_ms,
                )
            receipts_in_arrival_order = sorted(
                receipts,
                key=lambda item: (item.ack_received_monotonic_ms, item.fragment_index, item.duplicate_copy),
            )
            receipts_by_index: dict[int, list[SendReceipt]] = {}
            for receipt in receipts:
                receipts_by_index.setdefault(receipt.fragment_index, []).append(receipt)
            missing = sorted(set(record_map[uid].fragment_index for uid in record_map) - set(receipts_by_index))
            if missing:
                fail(E_FIELD, "forward.receipts", "one or more fragments have no acknowledgement", missing_indices=missing)
            decode_many = decode_receipts(
                plan,
                receipts_in_arrival_order,
                total_fragments=len(message.fragments),
            )
            reassembly = reassemble_fragments(
                decode_many,
                expected_message_key=plan.message_key,
                expected_sha256=plan.payload_sha256,
                expected_total=len(message.fragments),
                reassembly_version=message.reassembly.reassembly_version,
            )
            if reassembly.payload != plan.payload:
                fail(
                    E_FIELD,
                    f"forward[{plan.message_key}].reassembly",
                    "reassembled bytes differ from the deterministic inner sample",
                )
            fragment_rows = []
            for index in sorted(receipts_by_index):
                chosen = sorted(receipts_by_index[index], key=lambda item: item.ack_received_monotonic_ms)[0]
                fragment_rows.append(
                    _receipt_row(
                        plan=plan,
                        record=record_map[chosen.message_key + f":{index}:copy" + str(chosen.duplicate_copy)],
                        receipt=chosen,
                        total_fragments=len(message.fragments),
                    )
                )
            real_events = scheduler_result.real_trace.get("events", [])
            scheduling_events = [
                {"kind": "dispatch", **event}
                for event in real_events
            ]
            commit_digest = ledger.commit_batch(
                batch_id=batch_id,
                fragments=fragment_rows,
                reassembly=_reassembly_row(message, reassembly),
                scheduling_statistics=scheduler_result.deterministic_statistics,
                scheduling_events=scheduling_events,
                cursor={
                    "state": "committed",
                    "batch_id": batch_id,
                    "message_key": plan.message_key,
                    "completed_batches": batch_index + 1,
                },
            )
            current_batch_id = None
            commit = {"batch_id": batch_id, "commit_digest": commit_digest}
            commits.append(commit)
            logger.event("batch_committed", batch_id=batch_id, commit_digest=commit_digest)
            message_reports.append(
                {
                    "message_key": plan.message_key,
                    "scenario": scenario,
                    "target_format": bundle.target_format,
                    "receipt_count": len(receipts),
                    "fragment_count": len(message.fragments),
                    "arrival_indices": list(reassembly.arrival_indices),
                    "duplicate_indices": list(reassembly.duplicate_indices),
                    "payload_sha256": reassembly.payload_sha256,
                    "payload_matches_inner_sample": reassembly.payload == plan.payload,
                    "scheduler": scheduler_result.to_dict(),
                    "commit_digest": commit_digest,
                }
            )
    except BaseException as exc:
        if current_batch_id is not None:
            error = exc.to_dict() if isinstance(exc, TunnelError) else {"code": "E_INTERNAL", "message": repr(exc)}
            try:
                ledger.mark_batch_unknown(current_batch_id, error)
            except Exception:
                pass
        raise
    finally:
        pool.stop()
        ledger.close()

    return {
        "ok": True,
        "scenario": scenario,
        "target_format": bundle.target_format,
        "ledger": {
            "path": str(ledger_path),
            "open": open_result.to_dict(),
            "commits": commits,
        },
        "messages": message_reports,
    }


def interop_report(bundle: FragmentBundle) -> dict[str, Any]:
    rows = []
    all_equal = True
    for message in bundle.messages:
        for fragment in message.fragments:
            variants = migrate_all_variants(fragment, message=message.plan)
            signatures = {
                name: sha256_json(normalized.semantic_key())
                for name, normalized in variants.items()
            }
            equal = len(set(signatures.values())) == 1
            all_equal = all_equal and equal
            rows.append(
                {
                    "message_key": message.plan.message_key,
                    "fragment_index": fragment.fragment_index,
                    "fragment_offset": fragment.fragment_offset,
                    "fragment_length": fragment.fragment_length,
                    "total_fragments": fragment.total_fragments,
                    "message_id": fragment.message_id,
                    "message_id_basis": message.plan.message_id_basis,
                    "semantic_signatures": signatures,
                    "normalized_key_fields_equal": equal,
                    "padded_fields_by_format": {
                        name: list(normalized.padded_fields)
                        for name, normalized in variants.items()
                    },
                    "padding_basis_by_format": {
                        name: dict(sorted(normalized.padding_basis.items()))
                        for name, normalized in variants.items()
                    },
                }
            )
    return {
        "ok": all_equal,
        "check": "f1/f2/f3 normalization and migration key-field equality",
        "fragment_count": len(rows),
        "all_equal": all_equal,
        "fragments": rows,
    }


def scheduler_determinism_report(spec: TunnelSpec, bundle: FragmentBundle) -> dict[str, Any]:
    tasks = [
        ScheduledTask(
            task_uid=f"{message.plan.message_key}:{fragment.fragment_index}",
            message_key=message.plan.message_key,
            tunnel_id=message.plan.tunnel_id,
            fragment_index=fragment.fragment_index,
            order=fragment.fragment_index,
        )
        for message in bundle.messages
        for fragment in message.fragments
    ]

    def worker(task: ScheduledTask) -> str:
        return task.task_uid

    first = run_scheduler(
        tasks,
        worker,
        global_inflight_limit=spec.runtime.global_inflight_limit,
        per_tunnel_inflight_quota=spec.runtime.per_tunnel_inflight_quota,
    )
    second = run_scheduler(
        tasks,
        worker,
        global_inflight_limit=spec.runtime.global_inflight_limit,
        per_tunnel_inflight_quota=spec.runtime.per_tunnel_inflight_quota,
    )
    first_bytes = canonical_json(first.deterministic_statistics)
    second_bytes = canonical_json(second.deterministic_statistics)
    return {
        "ok": first_bytes == second_bytes,
        "byte_identical": first_bytes == second_bytes,
        "statistics_sha256": first.deterministic_statistics["statistics_sha256"],
        "statistics": first.deterministic_statistics,
        "thread_names_first_run": first.real_trace["threads_used"],
        "thread_names_second_run": second.real_trace["threads_used"],
    }


def evolution_report(old_spec: TunnelSpec, new_spec: TunnelSpec) -> dict[str, Any]:
    old_bundle = build_bundle(old_spec)
    result = evolve_bundle(old_bundle, old_spec=old_spec, new_spec=new_spec)
    payload = result.to_dict()
    payload["ok"] = all(result.unchanged_checks.values())
    payload["recomputed_bundle"] = result.bundle.to_dict(include_wire=False, include_payload=False)
    return payload


def resume_session(
    *,
    spec: TunnelSpec,
    ledger_path: str | Path,
    logger: StructuredLogger | None = None,
    rerun: bool = False,
    scenario: str = "normal",
) -> dict[str, Any]:
    logger = logger or StructuredLogger()
    ledger = SessionLedger(ledger_path)
    open_result = ledger.open(spec)
    replay: ReplayResult | None = None
    final_open = open_result
    if open_result.replaced:
        report = {
            "ok": True,
            "open": open_result.to_dict(),
            "replay": None,
            "message": "snapshot replaced wholesale; committed data from the old snapshot is not reused",
        }
        ledger.close()
        return report
    replay = ledger.replay_committed(spec)
    pending = list(open_result.unknown_batches)
    ledger.close()
    rerun_report = None
    if rerun and pending:
        bundle = build_bundle(spec)
        rerun_report = forward_bundle(
            spec=spec,
            bundle=bundle,
            ledger_path=ledger_path,
            scenario=scenario,
            logger=logger,
            message_keys=[item["message_key"] for item in pending],
        )
        with SessionLedger(ledger_path) as replay_ledger:
            final_open = replay_ledger.open(spec)
            replay_result_after = replay_ledger.replay_committed(spec)
        rerun_report["replay_after_rerun"] = replay_result_after.to_dict()
    return {
        "ok": True,
        "open": open_result.to_dict(),
        "final_open": final_open.to_dict(),
        "replay": replay.to_dict(),
        "rerun_requested": rerun,
        "rerun": rerun_report,
        "can_claim_exit_report": final_open.can_claim_exit_report,
    }


def snapshot_replacement_report(
    *,
    old_spec: TunnelSpec,
    new_spec: TunnelSpec,
    ledger_path: str | Path,
) -> dict[str, Any]:
    with SessionLedger(ledger_path) as ledger:
        first = ledger.open(old_spec)
        second = ledger.open(new_spec)
    return {
        "ok": second.replaced,
        "initialized": first.initialized,
        "replacement": second.to_dict(),
        "replacement_mode": "whole_snapshot" if second.replaced else "unchanged",
    }


def inject_anomaly(
    *,
    kind: str,
    spec_path: str | Path,
    output_dir: str | Path,
    ledger_path: str | Path | None = None,
    logger: StructuredLogger | None = None,
) -> dict[str, Any]:
    logger = logger or StructuredLogger()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if kind == "corrupt-spec":
        corrupted = output / "spec.corrupt.json"
        corrupted.write_text('{"endpoints": [', encoding="utf-8")
        load_spec(corrupted)
    elif kind == "dangling-route":
        raw = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        if not raw.get("routes"):
            fail(E_SPEC, "$.routes", "cannot inject a dangling route into an empty route table")
        raw["routes"][0]["next_hop"] = "missing-endpoint"
        mutated = output / "spec.dangling.json"
        atomic_write_json(mutated, raw)
        load_spec(mutated, reference_error_code=E_REF)
    elif kind == "snapshot-digest":
        if ledger_path is None:
            fail(E_SNAPSHOT, "ledger.path", "snapshot-digest injection requires a ledger path")
        source = Path(ledger_path)
        if not source.exists():
            fail(E_SNAPSHOT, "ledger.path", "ledger file does not exist", file=str(source))
        target = output / "ledger.tampered.db"
        shutil.copy2(source, target)
        connection = sqlite3.connect(target)
        try:
            connection.execute(
                "UPDATE meta SET value=? WHERE key='ledger_content_digest'",
                ("0" * 64,),
            )
            connection.commit()
        finally:
            connection.close()
        spec = load_spec(spec_path)
        with SessionLedger(target) as ledger:
            ledger.open(spec)
    else:
        fail(E_SPEC, "inject.kind", "unknown injection kind", actual=kind)
    return {"ok": False, "injection": kind, "unreachable": True}


def verify_all(
    *,
    spec_path: str | Path,
    evolved_spec_path: str | Path | None,
    output_dir: str | Path,
    with_udp: bool = True,
    logger: StructuredLogger | None = None,
) -> dict[str, Any]:
    logger = logger or StructuredLogger()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    spec = load_spec(spec_path)
    report: dict[str, Any] = {
        "ok": True,
        "spec": str(spec_path),
        "versions": spec.versions.canonical(),
        "format_interop": None,
        "scheduler_determinism": None,
        "udp_scenarios": [],
        "evolution": None,
        "snapshot_replacement": None,
    }
    bundle = build_bundle(spec)
    interop = interop_report(bundle)
    atomic_write_json(output / "format_interop.json", interop)
    report["format_interop"] = {
        "ok": interop["ok"],
        "fragment_count": interop["fragment_count"],
        "all_equal": interop["all_equal"],
    }
    report["ok"] = report["ok"] and interop["ok"]

    scheduler_report = scheduler_determinism_report(spec, bundle)
    atomic_write_json(output / "scheduler-determinism.json", scheduler_report)
    report["scheduler_determinism"] = {
        "ok": scheduler_report["ok"],
        "byte_identical": scheduler_report["byte_identical"],
        "statistics_sha256": scheduler_report["statistics_sha256"],
        "global_peak_inflight": scheduler_report["statistics"]["global_peak_inflight"],
        "per_tunnel_peak": scheduler_report["statistics"]["per_tunnel_peak"],
        "queued_fragments": scheduler_report["statistics"]["queued_fragments"],
    }
    report["ok"] = report["ok"] and scheduler_report["ok"]

    if with_udp:
        for scenario in ("normal", "reverse", "duplicate"):
            ledger_path = output / f"session-{scenario}.db"
            if ledger_path.exists():
                ledger_path.unlink()
            scene_report = forward_bundle(
                spec=spec,
                bundle=bundle,
                ledger_path=ledger_path,
                scenario=scenario,
                logger=logger,
            )
            replay = SessionLedger(ledger_path)
            try:
                replay.open(spec)
                replay_result = replay.replay_committed(spec)
            finally:
                replay.close()
            scene_report["replay"] = replay_result.to_dict()
            atomic_write_json(output / f"udp-{scenario}.json", scene_report)
            report["udp_scenarios"].append(
                {
                    "scenario": scenario,
                    "ok": scene_report["ok"] and replay_result.checked_batches == len(spec.routes),
                    "receipts": sum(item["receipt_count"] for item in scene_report["messages"]),
                    "payloads_match": all(item["payload_matches_inner_sample"] for item in scene_report["messages"]),
                    "replay": replay_result.to_dict(),
                }
            )
            report["ok"] = report["ok"] and report["udp_scenarios"][-1]["ok"]

    if evolved_spec_path is not None:
        new_spec = load_spec(evolved_spec_path)
        evolution = evolution_report(spec, new_spec)
        atomic_write_json(output / "evolution.json", evolution)
        report["evolution"] = {
            "ok": evolution["ok"],
            "invalid_fragment_count": evolution["invalid_fragment_count"],
            "invalid_reassemblies": evolution["invalid_reassemblies"],
            "preserved_fragment_count": evolution["preserved_fragment_count"],
            "unchanged_checks": evolution["unchanged_checks"],
        }
        replacement_db = output / "session-replacement.db"
        if replacement_db.exists():
            replacement_db.unlink()
        replacement = snapshot_replacement_report(
            old_spec=spec,
            new_spec=new_spec,
            ledger_path=replacement_db,
        )
        atomic_write_json(output / "snapshot-replacement.json", replacement)
        report["snapshot_replacement"] = replacement
        report["ok"] = report["ok"] and evolution["ok"] and replacement["ok"]

    atomic_write_json(output / "verification.json", report)
    return report
