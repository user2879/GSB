"""Authoritative ledger assembly, persistence, and integrity verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .budget import compute_budget_report
from .canonical import (
    atomic_write_json,
    hash_value,
    hash_without,
    read_json,
)
from .constants import (
    ENGINE_VERSION,
    LEDGER_SCHEMA_VERSION,
    PIPELINE_VERSION,
    RNG_VERSION,
    SCHEDULER_VERSION,
    SCHEMA_VERSION,
)
from .errors import PipelineError, require
from .events import events_digest
from .generator import assign_event_indices
from .rng import verify_records
from .rules import faction_hash, rules_hash, validate_rules


def version_stamp() -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "rng_version": RNG_VERSION,
        "scheduler_version": SCHEDULER_VERSION,
    }


def _business_game(game: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in game.items() if key != "chunk_ticks"}


def build_ledger(
    manifest: dict[str, Any],
    scheduler_result: dict[str, Any],
    rules: dict[str, Any],
    game: dict[str, Any],
    *,
    document_kind: str,
) -> dict[str, Any]:
    require(
        scheduler_result.get("status") == "ok",
        "E_INTERNAL",
        "cannot build a ledger from an incomplete scheduler result",
        scheduler_status=scheduler_result.get("status"),
    )
    normalized_rules = validate_rules(rules)
    results = sorted(
        scheduler_result["results"], key=lambda item: (item["start_tick"], item["chunk_id"])
    )
    expected_start = manifest["tick_start"]
    previous_exit_hash = manifest["initial_snapshot_hash"]
    tick_records: list[dict[str, Any]] = []
    raw_events: list[dict[str, Any]] = []
    raw_violations: list[dict[str, Any]] = []
    chunk_boundaries: list[dict[str, Any]] = []
    for result in results:
        if result["start_tick"] != expected_start:
            raise PipelineError(
                "E_CHUNK_BOUNDARY",
                "chunk coverage has a gap or overlap",
                expected_start_tick=expected_start,
                actual_start_tick=result["start_tick"],
                chunk_id=result["chunk_id"],
            )
        if result["entry_snapshot_hash"] != previous_exit_hash:
            raise PipelineError(
                "E_CHUNK_BOUNDARY",
                "chunk entry snapshot does not match the previous exit",
                chunk_id=result["chunk_id"],
                expected_entry_hash=previous_exit_hash,
                actual_entry_hash=result["entry_snapshot_hash"],
            )
        if result["computed_exit_snapshot_hash"] != result["declared_exit_snapshot_hash"]:
            raise PipelineError(
                "E_CHUNK_BOUNDARY",
                "chunk computed exit differs from declared exit",
                chunk_id=result["chunk_id"],
            )
        tick_records.extend(result["tick_records"])
        raw_events.extend(result["event_records"])
        raw_violations.extend(result["violations"])
        chunk_boundaries.append(
            {
                "chunk_id": result["chunk_id"],
                "start_tick": result["start_tick"],
                "end_tick": result["end_tick"],
                "entry_snapshot": result["entry_snapshot"],
                "entry_snapshot_hash": result["entry_snapshot_hash"],
                "exit_snapshot": result["computed_exit_snapshot"],
                "exit_snapshot_hash": result["computed_exit_snapshot_hash"],
            }
        )
        previous_exit_hash = result["computed_exit_snapshot_hash"]
        expected_start = result["end_tick"] + 1
    if expected_start != manifest["tick_end"] + 1:
        raise PipelineError(
            "E_CHUNK_BOUNDARY",
            "chunks do not cover the complete game tick range",
            expected_end_tick=manifest["tick_end"],
            covered_until_tick=expected_start - 1,
        )

    tick_records.sort(key=lambda item: item["tick"])
    expected_ticks = list(range(manifest["tick_start"], manifest["tick_end"] + 1))
    actual_ticks = [record["tick"] for record in tick_records]
    if actual_ticks != expected_ticks:
        raise PipelineError(
            "E_CHUNK_BOUNDARY",
            "merged tick records are not complete and contiguous",
            expected_tick_count=len(expected_ticks),
            actual_tick_count=len(actual_ticks),
        )
    event_records, violations, rng_records = assign_event_indices(
        raw_events, raw_violations
    )
    budget_report = compute_budget_report(
        event_records, tick_records, normalized_rules, violations
    )
    result_codes: list[str] = []
    if violations:
        result_codes.append("E_RESOURCE_OVERFLOW")
    exit_snapshot = chunk_boundaries[-1]["exit_snapshot"]
    game_from_manifest = {
        key: manifest.get(key)
        for key in ("tick_start", "tick_end", "chunk_ticks", "event_count", "events_hash")
    }
    payload: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "document_kind": document_kind,
        "version_stamp": version_stamp(),
        "game": game,
        "game_dimensions": game_from_manifest,
        "rules_hash": rules_hash(normalized_rules),
        "events_hash": events_digest(
            [
                {
                    "tick": record["tick"],
                    "actor": record["actor"],
                    "action": record["action"],
                    "target": record["target"],
                    "amount": record["amount"],
                    "rng_seed_delta": record["rng_seed_delta"],
                    "local_seq": record["local_seq"],
                }
                for record in event_records
            ]
        ),
        "chunks_manifest_hash": manifest["manifest_hash"],
        "scheduler": scheduler_result["report"],
        "chunk_boundaries": chunk_boundaries,
        "initial_snapshot": manifest["initial_snapshot"],
        "initial_snapshot_hash": manifest["initial_snapshot_hash"],
        "exit_snapshot": exit_snapshot,
        "exit_snapshot_hash": hash_value(exit_snapshot),
        "tick_records": tick_records,
        "event_records": event_records,
        "rng_records": rng_records,
        "violations": violations,
        "budget_report": budget_report,
        "result_codes": result_codes,
        "summary": {
            "full_tick_count": len(tick_records),
            "event_count": len(event_records),
            "chunk_count": len(chunk_boundaries),
            "faction_count": len(normalized_rules["factions"]),
            "rng_record_count": len(rng_records),
            "overflow_count": len(violations),
            "business_result": result_codes[0] if result_codes else "OK",
            "execution_complete": True,
        },
    }
    business_payload = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "game": _business_game(game),
        "rules_hash": payload["rules_hash"],
        "events_hash": payload["events_hash"],
        "initial_snapshot_hash": payload["initial_snapshot_hash"],
        "exit_snapshot_hash": payload["exit_snapshot_hash"],
        "tick_records": tick_records,
        "event_records": event_records,
        "rng_records": rng_records,
        "violations": violations,
        "budget_report": budget_report,
        "result_codes": result_codes,
    }
    payload["business_hash"] = hash_value(business_payload)
    payload["ledger_hash"] = hash_without(payload, "ledger_hash")
    return payload


def persist_ledger(path: Path, ledger: dict[str, Any]) -> None:
    expected = ledger.get("ledger_hash")
    actual = hash_without(ledger, "ledger_hash")
    if expected != actual:
        raise PipelineError(
            "E_LEDGER_HASH",
            "refusing to persist a ledger with an invalid self hash",
            expected=expected,
            actual=actual,
        )
    atomic_write_json(path, ledger)


def load_and_verify_ledger(path: Path) -> dict[str, Any]:
    ledger = read_json(path)
    verify_ledger(ledger)
    return ledger


def verify_ledger(ledger: Any) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        raise PipelineError("E_LEDGER_HASH", "ledger must be a JSON object")
    required = (
        "schema_version",
        "version_stamp",
        "game",
        "rules_hash",
        "events_hash",
        "initial_snapshot",
        "initial_snapshot_hash",
        "exit_snapshot",
        "exit_snapshot_hash",
        "chunk_boundaries",
        "tick_records",
        "event_records",
        "rng_records",
        "violations",
        "budget_report",
        "result_codes",
        "business_hash",
        "ledger_hash",
    )
    missing = [field for field in required if field not in ledger]
    if missing:
        raise PipelineError("E_LEDGER_HASH", "ledger is incomplete", missing=missing)

    game = ledger["game"]
    seed = game.get("seed") if isinstance(game, dict) else None
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise PipelineError("E_LEDGER_HASH", "ledger game seed is missing")
    tick_start = game.get("tick_start")
    tick_end = game.get("tick_end")
    if (
        not isinstance(tick_start, int)
        or isinstance(tick_start, bool)
        or not isinstance(tick_end, int)
        or isinstance(tick_end, bool)
        or tick_end < tick_start
    ):
        raise PipelineError("E_LEDGER_HASH", "ledger game tick range is invalid")
    actual_ticks = [record.get("tick") for record in ledger["tick_records"]]
    expected_ticks = list(range(tick_start, tick_end + 1))
    if actual_ticks != expected_ticks:
        raise PipelineError(
            "E_LEDGER_HASH",
            "ledger tick records are incomplete, duplicated, or out of order",
            expected_count=len(expected_ticks),
            actual_count=len(actual_ticks),
        )
    if game.get("rules_hash") != ledger["rules_hash"]:
        raise PipelineError("E_LEDGER_HASH", "game rules hash disagrees with ledger")
    if game.get("events_hash") != ledger["events_hash"]:
        raise PipelineError("E_LEDGER_HASH", "game events hash disagrees with ledger")
    event_view = [
        {
            "tick": record.get("tick"),
            "actor": record.get("actor"),
            "action": record.get("action"),
            "target": record.get("target"),
            "amount": record.get("amount"),
            "rng_seed_delta": record.get("rng_seed_delta"),
            "local_seq": record.get("local_seq"),
        }
        for record in ledger["event_records"]
    ]
    if events_digest(event_view) != ledger["events_hash"]:
        raise PipelineError("E_LEDGER_HASH", "event records do not reproduce the event hash")
    if hash_value(ledger["initial_snapshot"]) != ledger["initial_snapshot_hash"]:
        raise PipelineError("E_LEDGER_HASH", "initial snapshot hash mismatch")
    if hash_value(ledger["exit_snapshot"]) != ledger["exit_snapshot_hash"]:
        raise PipelineError("E_LEDGER_HASH", "exit snapshot hash mismatch")
    if ledger["tick_records"][-1]["snapshot"] != ledger["exit_snapshot"]:
        raise PipelineError("E_LEDGER_HASH", "exit snapshot disagrees with final tick")

    expected_chunk_start = tick_start
    previous_exit_hash = ledger["initial_snapshot_hash"]
    for boundary in ledger["chunk_boundaries"]:
        if boundary.get("start_tick") != expected_chunk_start:
            raise PipelineError(
                "E_LEDGER_HASH",
                "chunk boundary coverage is not contiguous",
                expected_start_tick=expected_chunk_start,
                actual_start_tick=boundary.get("start_tick"),
            )
        if hash_value(boundary.get("entry_snapshot")) != boundary.get("entry_snapshot_hash"):
            raise PipelineError("E_LEDGER_HASH", "chunk entry snapshot hash mismatch")
        if boundary.get("entry_snapshot_hash") != previous_exit_hash:
            raise PipelineError("E_LEDGER_HASH", "chunk entry does not connect to previous exit")
        if hash_value(boundary.get("exit_snapshot")) != boundary.get("exit_snapshot_hash"):
            raise PipelineError("E_LEDGER_HASH", "chunk exit snapshot hash mismatch")
        expected_chunk_start = boundary.get("end_tick", -1) + 1
        previous_exit_hash = boundary.get("exit_snapshot_hash")
    if expected_chunk_start != tick_end + 1:
        raise PipelineError("E_LEDGER_HASH", "chunk boundaries do not cover the game")
    if previous_exit_hash != ledger["exit_snapshot_hash"]:
        raise PipelineError("E_LEDGER_HASH", "final chunk exit disagrees with ledger exit")
    verify_records(ledger["rng_records"], seed)

    for record in ledger["tick_records"]:
        snapshot = record.get("snapshot")
        if hash_value(snapshot) != record.get("state_hash"):
            raise PipelineError(
                "E_LEDGER_HASH",
                "tick state hash mismatch",
                tick=record.get("tick"),
            )
        if set(record.get("faction_hashes", {})) != set(snapshot.get("factions", {})):
            raise PipelineError(
                "E_LEDGER_HASH",
                "tick faction hash set is incomplete",
                tick=record.get("tick"),
            )
        for faction, state in snapshot["factions"].items():
            if faction_hash(state) != record["faction_hashes"].get(faction):
                raise PipelineError(
                    "E_LEDGER_HASH",
                    "faction resource/queue hash mismatch",
                    tick=record.get("tick"),
                    faction=faction,
                )

    expected_event_indices = list(range(1, len(ledger["event_records"]) + 1))
    actual_event_indices = [record.get("event_index") for record in ledger["event_records"]]
    if actual_event_indices != expected_event_indices:
        raise PipelineError(
            "E_LEDGER_HASH",
            "event index sequence is not contiguous",
            expected_count=len(expected_event_indices),
            actual_count=len(actual_event_indices),
        )
    for record in ledger["event_records"]:
        expected_rng = record.get("rng")
        if expected_rng is None:
            raise PipelineError("E_RNG_DRIFT", "event has no derivation record", event_index=record.get("event_index"))
    rng_by_index = {
        record["event_index"]: record for record in ledger["rng_records"]
    }
    if [record["event_index"] for record in ledger["rng_records"]] != expected_event_indices:
        raise PipelineError("E_RNG_DRIFT", "RNG record event indexes are not contiguous")
    for record in ledger["event_records"]:
        rng = rng_by_index.get(record["event_index"])
        rng_view = (
            {key: value for key, value in rng.items() if key != "event_index"}
            if rng is not None
            else None
        )
        if rng_view != record["rng"]:
            raise PipelineError(
                "E_RNG_DRIFT",
                "event derivation record disagrees with the replay log",
                event_index=record["event_index"],
            )

    if "business_hash" in ledger:
        business_payload = {
            "schema_version": ledger["schema_version"],
            "game": _business_game(ledger["game"]),
            "rules_hash": ledger.get("rules_hash"),
            "events_hash": ledger.get("events_hash"),
            "initial_snapshot_hash": ledger.get("initial_snapshot_hash"),
            "exit_snapshot_hash": ledger.get("exit_snapshot_hash"),
            "tick_records": ledger["tick_records"],
            "event_records": ledger["event_records"],
            "rng_records": ledger["rng_records"],
            "violations": ledger["violations"],
            "budget_report": ledger["budget_report"],
            "result_codes": ledger["result_codes"],
        }
        if ledger["business_hash"] != hash_value(business_payload):
            raise PipelineError(
                "E_LEDGER_HASH",
                "ledger business hash mismatch",
                expected=hash_value(business_payload),
                actual=ledger["business_hash"],
            )

    expected_hash = hash_without(ledger, "ledger_hash")
    if ledger["ledger_hash"] != expected_hash:
        raise PipelineError(
            "E_LEDGER_HASH",
            "ledger self hash mismatch",
            expected=expected_hash,
            actual=ledger["ledger_hash"],
        )
    return {
        "valid": True,
        "ledger_hash": ledger["ledger_hash"],
        "event_count": len(ledger["event_records"]),
        "tick_count": len(ledger["tick_records"]),
        "rng_record_count": len(ledger["rng_records"]),
        "result_codes": ledger["result_codes"],
    }
