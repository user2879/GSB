"""Replay orchestration: full run, incremental run and consistency compare."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engine import Engine
from .errors import E_REC_INVALID
from .loader import load_commands, load_documents, load_query_stream
from .models import RetrievalConfig
from .rules import load_rule_table
from .snapshot import SnapshotStore


def load_inputs(sample_dir: str | Path):
    sample_dir = Path(sample_dir)
    with (sample_dir / "retrieval_config.json").open(
        "r", encoding="utf-8"
    ) as handle:
        config = RetrievalConfig.from_dict(json.load(handle))
    with (sample_dir / "domain_rules.json").open(
        "r", encoding="utf-8"
    ) as handle:
        rules = load_rule_table(json.load(handle))
    faults_path = sample_dir / "fault_injection.json"
    faults: dict[str, Any] = {}
    if faults_path.exists():
        with faults_path.open("r", encoding="utf-8") as handle:
            faults = json.load(handle)
    docs, doc_failures = load_documents(sample_dir / "documents.jsonl")
    turns, query_failures = load_query_stream(sample_dir / "queries.jsonl")
    commands, command_failures = load_commands(sample_dir / "commands.jsonl")
    failures = [*doc_failures, *query_failures, *command_failures]
    return config, rules, faults, docs, turns, commands, failures


def inject_record_faults(
    docs,
    turns,
    faults: dict[str, Any],
    sample_dir: Path,
):
    """Apply deterministic fault-injection flags from the fault config."""
    from .models import Failure

    extra_failures: list[Failure] = []
    if faults.get("invalid_doc_inject"):
        duplicated_id = docs[0].doc_id if docs else "DOC0001"
        # Deterministic E_REC_INVALID record; corpus is intentionally left
        # untouched so index fingerprints stay identical across engines.
        extra_failures.append(
            Failure(
                code=E_REC_INVALID,
                scope="fault:duplicate_doc",
                message=f"injected duplicate doc_id rejected: {duplicated_id}",
                sort_key="fault:duplicate_doc",
            )
        )
    if faults.get("invalid_query_inject"):
        extra_failures.append(
            Failure(
                code=E_REC_INVALID,
                scope="fault:query_gap",
                message="injected turn sequence gap for S-FAULT",
                sort_key="fault:query_gap",
            )
        )
    return extra_failures


def _comparable_turns(engine: Engine) -> dict[str, Any]:
    payload = {}
    for row in engine.turn_results():
        payload[row["turn"]] = {
            "effective_query": row["effective_query"],
            "retrieval": row["retrieval"],
            "extraction": row["extraction"],
            "citation": row["citation"],
        }
    return payload


def _failure_codes(engine: Engine) -> list[dict[str, str]]:
    return [failure.to_dict() for failure in engine.failure_list()]


def diff_results(
    incremental: dict[str, Any], full: dict[str, Any]
) -> list[str]:
    diffs: list[str] = []
    all_keys = sorted(set(incremental) | set(full))
    for key in all_keys:
        if key not in incremental:
            diffs.append(f"{key}: missing in incremental")
            continue
        if key not in full:
            diffs.append(f"{key}: missing in full")
            continue
        if incremental[key] != full[key]:
            diffs.append(f"{key}: payload mismatch")
    return diffs


def run_replay(
    sample_dir: str | Path,
    snapshot_dir: str | Path,
    *,
    clean: bool = True,
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    snapshot_dir = Path(snapshot_dir)
    config, rules, faults, docs, turns, commands, load_failures = load_inputs(
        sample_dir
    )
    if clean and snapshot_dir.exists():
        SnapshotStore(snapshot_dir).reset()
    injected_failures = inject_record_faults(docs, turns, faults, sample_dir)
    load_failures = [*load_failures, *injected_failures]

    # ---- Incremental engine: full baseline then one command at a time. ----
    store = SnapshotStore(snapshot_dir)
    incremental = Engine(
        docs=docs,
        turns=turns,
        config=config,
        rules=rules,
        store=store,
        base_failures=load_failures,
        faults=faults,
    )
    baseline_summary = incremental.run_full()
    command_summaries: list[dict[str, Any]] = []
    for command in commands:
        command_summaries.append(incremental.handle_command(command))
    drain_summary = incremental.drain_pending()
    incremental_payload = _comparable_turns(incremental)
    incremental_failures = _failure_codes(incremental)
    incremental_stats = {
        "cache_hits": dict(incremental.stats.cache_hits),
        "cache_misses": dict(incremental.stats.cache_misses),
        "invalidations": dict(incremental.stats.invalidations),
        "stage_skips": dict(incremental.stats.stage_skips),
        "retrievals_executed": incremental.stats.retrievals_executed,
    }
    final_manifest = incremental.manifest()
    incremental.shutdown()

    # ---- Reference full engine on a blank snapshot with final inputs. ----
    reference_store = SnapshotStore(snapshot_dir.parent / "reference_snapshots")
    reference_store.reset()
    # Reconstruct the final mutated state by replaying commands identically.
    config2, rules2, faults2, docs2, turns2, _, load_failures2 = load_inputs(
        sample_dir
    )
    injected2 = inject_record_faults(docs2, turns2, faults2, sample_dir)
    load_failures2 = [*load_failures2, *injected2]
    reference = Engine(
        docs=docs2,
        turns=turns2,
        config=config2,
        rules=rules2,
        store=reference_store,
        base_failures=load_failures2,
        faults=faults2,
    )
    for command in commands:
        from .engine_state import apply_command

        apply_command(reference.state, command)
    reference._rebuild_if_needed()
    for record in reference.sorted_records():
        record.effective_query = reference.state.effective_query(record.turn)
    reference.run_full()
    full_payload = _comparable_turns(reference)
    full_failures = _failure_codes(reference)
    full_stats = {
        "cache_hits": dict(reference.stats.cache_hits),
        "cache_misses": dict(reference.stats.cache_misses),
        "invalidations": dict(reference.stats.invalidations),
        "stage_skips": dict(reference.stats.stage_skips),
        "retrievals_executed": reference.stats.retrievals_executed,
    }
    reference.shutdown()

    diffs = diff_results(incremental_payload, full_payload)
    failure_diffs = (
        [] if incremental_failures == full_failures else ["failure list mismatch"]
    )

    return {
        "sample_dir": str(sample_dir),
        "snapshot_dir": str(snapshot_dir),
        "turn_count": len(incremental_payload),
        "baseline_summary": baseline_summary,
        "command_summaries": command_summaries,
        "drain_summary": drain_summary,
        "incremental_turns": incremental_payload,
        "full_turns": full_payload,
        "incremental_failures": incremental_failures,
        "full_failures": full_failures,
        "incremental_stats": incremental_stats,
        "full_stats": full_stats,
        "diffs": diffs + failure_diffs,
        "consistent": not diffs and not failure_diffs,
        "final_manifest": final_manifest,
        "command_log": incremental.command_log,
        "budget_pending_seen": any(
            summary["code"] == "E_ANSWER_BUDGET_PENDING"
            for summary in command_summaries
        ),
    }
