"""Atomic ledger persistence, semantic snapshots and recovery reports."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from . import __version__
from .config import RuntimeState
from .errors import E_LEDGER_CORRUPT, E_LEDGER_VERSION, QLabError
from .util import atomic_write_json, canonical_json, digest_json, read_json

SCHEMA_VERSION = 1


def worker_to_dict(worker: Any) -> dict[str, Any]:
    return worker.to_dict()


def state_to_dict(state: RuntimeState) -> dict[str, Any]:
    return {
        "workers": [worker.to_dict() for worker in state.workers],
        "operator": {
            "revision": state.revision,
            "frac_bits": state.frac_bits,
            "quant_level": state.quant_level,
            "layers": [layer.to_dict() for layer in state.layers],
        },
        "batches": [batch.to_dict() for batch in state.batches],
    }


def empty_ledger(base_config_hash: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "engine_version": __version__,
        "base_config_hash": base_config_hash,
        "commit_generation": 0,
        "applied_commands": [],
        "command_issues": [],
        "state": {},
        "run": {"batches": {}},
        "events": [],
        "next_event_seq": 1,
        "pending_batches": [],
        "deliveries": [],
        "receipts": [],
        "arbitration_decisions": [],
        "authority": {},
        "downgrades": [],
        "roundtrip_checks": [],
        "cache": {"layers": {}, "ledger_entries": {}},
        "lease_events": [],
        "diagnostics": {},
    }


class LedgerStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            ledger = read_json(self.path)
        except (OSError, json.JSONDecodeError) as exc:
            raise QLabError(E_LEDGER_CORRUPT, f"cannot read ledger: {exc}") from exc
        if not isinstance(ledger, dict):
            raise QLabError(E_LEDGER_CORRUPT, "ledger root must be an object")
        if ledger.get("schema_version") != SCHEMA_VERSION:
            raise QLabError(
                E_LEDGER_VERSION,
                f"ledger schema {ledger.get('schema_version')!r} is not supported",
                expected=SCHEMA_VERSION,
            )
        return ledger

    def save(self, ledger: dict[str, Any]) -> None:
        ledger["commit_generation"] = int(ledger.get("commit_generation", 0)) + 1
        atomic_write_json(self.path, ledger)

    def save_without_bump(self, ledger: dict[str, Any]) -> None:
        atomic_write_json(self.path, ledger)


def command_prefix_matches(stored: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> bool:
    if len(stored) > len(incoming):
        return False
    return canonical_json(stored) == canonical_json(incoming[: len(stored)])


def _sanitize_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "format",
        "batch_id",
        "layer_name",
        "output_summary",
        "revision",
        "worker_id",
        "stable_rank",
        "superseded_refs",
        "quant_params_summary",
        "completeness",
        "sidecar_used",
    }
    return {key: copy.deepcopy(receipt[key]) for key in sorted(allowed) if key in receipt}


def _sanitize_delivery(delivery: dict[str, Any]) -> dict[str, Any]:
    receipt = delivery.get("receipt", {})
    return {
        "origin": delivery.get("origin"),
        "command_seq": delivery.get("command_seq"),
        "receipt": {
            key: copy.deepcopy(receipt[key])
            for key in (
                "batch_id",
                "layer_name",
                "output_summary",
                "revision",
                "worker_id",
                "stable_rank",
                "superseded_refs",
                "quant_params_summary",
            )
            if key in receipt
        },
    }


def semantic_snapshot(ledger: dict[str, Any]) -> dict[str, Any]:
    """Deterministic verification projection.

    Monotonic elapsed milliseconds and commit-generation counters are runtime
    telemetry. Every semantic field, intermediate integer, dequantized output,
    authority decision, invalidation/reuse event and cache key remains present.
    """

    run_batches: dict[str, Any] = {}
    for batch_id in sorted(ledger.get("run", {}).get("batches", {})):
        batch = ledger["run"]["batches"][batch_id]
        run_batches[batch_id] = {
            "batch_id": batch_id,
            "sample_digest": batch.get("sample_digest"),
            "revision": batch.get("revision"),
            "frac_bits": batch.get("frac_bits"),
            "quant_level": batch.get("quant_level"),
            "worker_id": batch.get("worker_id"),
            "status": batch.get("status"),
            "error": batch.get("error"),
            "output": batch.get("output"),
            "output_digest": batch.get("output_digest"),
            "ledger_cache_key": batch.get("ledger_cache_key"),
            "layers": batch.get("layers", []),
        }
    authority: dict[str, Any] = {}
    for batch_id in sorted(ledger.get("authority", {})):
        decision = ledger["authority"][batch_id]
        authority[batch_id] = {
            "batch_id": batch_id,
            "layer_name": decision.get("layer_name"),
            "winner": _sanitize_receipt(decision.get("winner", {})),
            "candidates": sorted(
                [
                {
                    "format": candidate.get("format"),
                    "completeness": candidate.get("completeness"),
                    "semantic_ref": digest_json(
                        {
                            "format": candidate.get("format"),
                            "values": {
                                key: value
                                for key, value in candidate.get("arbitration_values", {}).items()
                                if key not in {"receipt_ref", "duration_ms"}
                            },
                        }
                    ),
                }
                for candidate in decision.get("candidates", [])
                ],
                key=lambda item: (item["semantic_ref"], item["format"], canonical_json(item["completeness"])),
            ),
            "order": decision.get("rationale", {}).get("order"),
            "missing_policy": decision.get("rationale", {}).get("missing_policy"),
            "winner_semantic_key": digest_json(_sanitize_receipt(decision.get("winner", {}))),
        }
    return {
        "schema_version": ledger.get("schema_version"),
        "base_config_hash": ledger.get("base_config_hash"),
        "applied_commands": ledger.get("applied_commands", []),
        "command_issues": ledger.get("command_issues", []),
        "state": ledger.get("state", {}),
        "run": run_batches,
        "events": ledger.get("events", []),
        "cache": ledger.get("cache", {}),
        "deliveries": [_sanitize_delivery(item) for item in ledger.get("deliveries", [])],
        "receipts": sorted(
            [_sanitize_receipt(item) for item in ledger.get("receipts", [])],
            key=canonical_json,
        ),
        "authority": authority,
        "arbitration_decisions": sorted(
            [
            {
                "batch_id": item.get("batch_id"),
                "layer_name": item.get("layer_name"),
                "winner": _sanitize_receipt(item.get("winner", {})),
                "rationale": {
                    "order": item.get("rationale", {}).get("order"),
                    "missing_policy": item.get("rationale", {}).get("missing_policy"),
                },
            }
            for item in ledger.get("arbitration_decisions", [])
            ],
            key=lambda item: (item["batch_id"], item["layer_name"]),
        ),
        "downgrades": [
            {
                **{key: value for key, value in item.items() if key not in {"sidecar_key", "delivery_key", "q1"}},
                "q1": {
                    key: value
                    for key, value in item.get("q1", {}).items()
                    if key in {"q", "batch_id", "layer_name", "output_summary"}
                },
            }
            for item in ledger.get("downgrades", [])
        ],
        "roundtrip_conclusions": [item.get("conclusion_unchanged") for item in ledger.get("roundtrip_checks", [])],
        "diagnostics": {
            key: ledger.get("diagnostics", {}).get(key)
            for key in (
                "reuse_count",
                "invalidation_count",
                "pending_queue",
                "E_LEASE_EXPIRED",
                "E_RECONCILE_PENDING",
                "incomplete_batches",
            )
        },
    }


def snapshot_digest(ledger: dict[str, Any]) -> str:
    return digest_json(semantic_snapshot(ledger))


def recovery_report(ledger: dict[str, Any], *, ledger_path: str | Path | None = None) -> dict[str, Any]:
    pending = sorted(set(ledger.get("pending_batches", [])))
    incomplete = sorted(
        batch_id
        for batch_id, batch in ledger.get("run", {}).get("batches", {}).items()
        if batch.get("status") != "complete"
    )
    expiry_count = sum(1 for event in ledger.get("lease_events", []) if event.get("code") == "E_LEASE_EXPIRED")
    reconcile_count = sum(
        1 for event in ledger.get("lease_events", []) if event.get("code") == "E_RECONCILE_PENDING"
    )
    reconcile_count += len(incomplete)
    reuse_count = sum(1 for event in ledger.get("events", []) if event.get("kind") == "reuse")
    return {
        "recovery_source": str(ledger_path) if ledger_path else None,
        "atomic_ledger_only": True,
        "reuse_entries": reuse_count,
        "pending_queue": pending,
        "E_LEASE_EXPIRED": expiry_count,
        "E_RECONCILE_PENDING": reconcile_count,
        "incomplete_batches": incomplete,
        "applied_command_count": len(ledger.get("applied_commands", [])),
        "cache_layer_entries": len(ledger.get("cache", {}).get("layers", {})),
        "cache_ledger_entries": len(ledger.get("cache", {}).get("ledger_entries", {})),
        "safe_to_resume": not pending and not incomplete,
    }
