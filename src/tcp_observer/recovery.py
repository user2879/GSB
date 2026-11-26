from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .congestion import CongestionState, ErrorBounds
from .engine import ObservationWorkspace
from .errors import ReplayError
from .ledger import Ledger
from .replay import compare_batches, compare_items, first_difference


def state_from_item(spec: Any, item: dict[str, Any]) -> CongestionState:
    state_after = item["state_after"]
    raw_bounds = item["error_bounds_raw"]
    bounds = ErrorBounds(
        cwnd_scaled=int(raw_bounds["cwnd_scaled"]),
        ssthresh_scaled=int(raw_bounds["ssthresh_scaled"]),
        srtt_scaled=int(raw_bounds["srtt_scaled"]),
        rttvar_scaled=int(raw_bounds["rttvar_scaled"]),
        rto_scaled=int(raw_bounds["rto_scaled"]),
    )
    return CongestionState(
        scale=spec.scale,
        window_max_scaled=spec.window_max * spec.scale,
        cwnd_scaled=int(state_after["cwnd_scaled"]),
        ssthresh_scaled=int(state_after["ssthresh_scaled"]),
        srtt_scaled=int(state_after["srtt_scaled"]),
        rttvar_scaled=int(state_after["rttvar_scaled"]),
        rto_scaled=int(state_after["rto_scaled"]),
        outstanding_packets=int(state_after["outstanding_packets"]),
        dupack_count=int(state_after["dupack_count"]),
        phase=str(state_after["phase"]),
        fast_retransmit_count=int(state_after["fast_retransmit_count"]),
        timeout_count=int(state_after["timeout_count"]),
        saturation_count=int(state_after["saturation_count"]),
        bounds=bounds,
    )


def load_committed_items(ledger: Ledger) -> list[dict[str, Any]]:
    rows = ledger.conn.execute("SELECT payload_json FROM observation_items ORDER BY event_seq").fetchall()
    return [json.loads(row["payload_json"]) for row in rows]


def replay_from_ledger(spec: Any, ledger: Ledger) -> dict[str, Any]:
    progress = ledger.last_progress()
    workspace = ObservationWorkspace.__new__(ObservationWorkspace)
    workspace.spec = spec
    workspace.exchange = None
    workspace.response_by_seq = {}
    workspace.pre_states = []
    workspace.items = load_committed_items(ledger)
    if progress is None:
        workspace.state = CongestionState.initial(spec)
        last_event_seq = spec.initial_seq - 1
    else:
        last_event_seq = int(progress["last_committed_event_seq"])
        row = ledger.conn.execute(
            "SELECT payload_json FROM observation_items WHERE event_seq=?",
            (last_event_seq,),
        ).fetchone()
        if row is None:
            raise ReplayError(
                "progress cursor references an event without a committed item",
                event_seq=last_event_seq,
                field_name="progress.last_committed_event_seq",
            )
        workspace.state = state_from_item(spec, json.loads(row["payload_json"]))

    for seq in range(last_event_seq + 1, spec.last_event_seq + 1):
        if spec.entry_for_seq(seq) is None:
            from .scripting import assert_no_gap

            assert_no_gap(spec)
        workspace.append_event(seq)
    return workspace.rebuild_views()


def recover_and_replay(
    spec: Any,
    db_path: str | Path,
    expected_result: dict[str, Any],
) -> tuple[Ledger, dict[str, Any], dict[str, Any]]:
    ledger = Ledger(db_path)
    progress = ledger.last_progress()
    last_batch = -1 if progress is None else int(progress["last_committed_batch_index"])
    unknown_batches = [last_batch + 1] if last_batch + 1 < len(expected_result["batches"]) else []
    for batch_index in unknown_batches:
        ledger.record_recovery_event(
            "unknown_uncommitted_batch_marked",
            {
                "integer_format_version": 1,
                "batch_index": batch_index,
                "status": "unknown_and_rerun",
                "last_committed_batch_index": last_batch,
            },
        )

    replayed = replay_from_ledger(spec, ledger)
    item_comparison = compare_items(expected_result, replayed)
    batch_comparison = compare_batches(expected_result, replayed)

    from .pipeline import schedule_payload_for_batch
    from .canonical import sha256_hex

    items_by_batch: dict[int, list[dict[str, Any]]] = {}
    for item in replayed["items"]:
        items_by_batch.setdefault(int(item["batch_index"]), []).append(item)
    for batch in replayed["batches"]:
        index = int(batch["batch_index"])
        if index <= last_batch:
            continue
        schedule_payload = schedule_payload_for_batch(index, batch, spec)
        cursor = {
            "integer_format_version": 1,
            "last_committed_batch_index": index,
            "last_committed_event_seq": int(batch["end_seq"]),
            "next_event_seq": int(batch["end_seq"]) + 1,
            "summary_content_digest": expected_result["summary"]["content_digest"],
        }
        ledger.commit_batch(batch, items_by_batch[index], schedule_payload, cursor)

    ledger.record_recovery_event(
        "replay_completed",
        {
            "integer_format_version": 1,
            "last_committed_batch_index": len(expected_result["batches"]) - 1,
            "matching_item_count": len(expected_result["items"]),
            "matching_batch_count": len(expected_result["batches"]),
        },
    )
    report = {
        "last_committed_batch_before_recovery": last_batch,
        "unknown_batches_marked_and_rerun": unknown_batches,
        "replay_comparison": {**item_comparison, **batch_comparison},
        "committed_batch_indices_after_recovery": ledger.committed_batch_indices(),
        "progress_after_recovery": ledger.last_progress(),
        "replay_ok": True,
    }
    return ledger, report, replayed
