"""End-to-end workspace execution, reporting, and committed-ledger recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import (
    atomic_write_json,
    hash_without,
    read_json,
    sha256_bytes,
)
from .chunker import prepare_chunks
from .constants import (
    ERROR_EXIT_CODES,
    EXIT_REPORT_FILENAME,
    LEDGER_FILENAME,
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    SERIAL_LEDGER_FILENAME,
)
from .errors import PipelineError
from .events import events_digest, read_events
from .ledger import (
    build_ledger,
    load_and_verify_ledger,
    persist_ledger,
)
from .rules import rules_hash, validate_rules
from .scheduler import run_scheduler
from .verifier import compare_ledgers, compare_to_truth


def load_workspace(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = Path(root)
    rules = validate_rules(
        read_json(root / "input" / "rules.json", error_code="E_SCHEMA")
    )
    game = read_json(root / "input" / "game.json", error_code="E_SCHEMA")
    if not isinstance(game, dict):
        raise PipelineError("E_SCHEMA", "game metadata must be an object")
    events = read_events(root / "input" / "events.jsonl", rules)
    checks = {
        "rules_hash": rules_hash(rules),
        "events_hash": events_digest(events),
    }
    for field, expected in checks.items():
        if game.get(field) != expected:
            raise PipelineError(
                "E_SCHEMA",
                "input metadata does not match its content",
                field=field,
                expected=expected,
                actual=game.get(field),
            )
    if game.get("event_count") != len(events):
        raise PipelineError(
            "E_SCHEMA",
            "game event count does not match the event stream",
            expected=game.get("event_count"),
            actual=len(events),
        )
    return game, rules, events


def _write_error(root: Path, error: PipelineError, stage: str) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "stage": stage,
        "error": error.to_dict(),
        "exit_report_written": False,
    }
    payload["error_report_hash"] = hash_without(payload, "error_report_hash")
    atomic_write_json(Path(root) / "output" / "error.json", payload)


def _artifact_hashes(output_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir).as_posix()
        if relative in ("artifact_hashes.json", EXIT_REPORT_FILENAME):
            continue
        artifacts.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_bytes(path.read_bytes()),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    payload["artifact_manifest_hash"] = hash_without(payload, "artifact_manifest_hash")
    atomic_write_json(output_dir / "artifact_hashes.json", payload)
    return payload


def run_workspace(
    root: Path,
    *,
    threads: int,
    memory_budget_bytes: int,
    chunk_ticks: int | None = None,
) -> dict[str, Any]:
    root = Path(root)
    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for marker in ("exit_report.json", "recovery_report.json", "error.json"):
        (output_dir / marker).unlink(missing_ok=True)
    stage = "load"
    try:
        game, rules, events = load_workspace(root)
        effective_chunk_ticks = (
            chunk_ticks if chunk_ticks is not None else game["chunk_ticks"]
        )
        game = dict(game)
        game["chunk_ticks"] = effective_chunk_ticks
        stage = "chunk"
        manifest = prepare_chunks(root, events, rules, game, effective_chunk_ticks)

        stage = "parallel_synthesis"
        parallel_result = run_scheduler(
            root=root,
            manifest=manifest,
            rules=rules,
            requested_threads=threads,
            memory_budget_bytes=memory_budget_bytes,
            mode="parallel",
        )
        atomic_write_json(
            output_dir / "scheduler_parallel.json",
            parallel_result["report"],
            pretty=True,
        )
        if parallel_result["status"] != "ok":
            code = parallel_result.get("error_code") or "E_BUDGET_DENIED"
            error = PipelineError(
                code,
                "parallel synthesis did not complete",
                scheduler_status=parallel_result["status"],
                scheduler=parallel_result["report"],
            )
            _write_error(root, error, stage)
            return {
                "status": parallel_result["status"],
                "result_code": code,
                "exit_code": ERROR_EXIT_CODES.get(code, 1),
                "scheduler_report": parallel_result["report"],
                "exit_report_written": False,
            }

        parallel_ledger = build_ledger(
            manifest,
            parallel_result,
            rules,
            game,
            document_kind="parallel_ledger",
        )
        max_block = max(
            entry["estimated_memory_bytes"] for entry in manifest["chunks"]
        )
        stage = "serial_synthesis"
        serial_result = run_scheduler(
            root=root,
            manifest=manifest,
            rules=rules,
            requested_threads=1,
            memory_budget_bytes=max(memory_budget_bytes, max_block),
            mode="serial",
        )
        atomic_write_json(
            output_dir / "scheduler_serial.json",
            serial_result["report"],
            pretty=True,
        )
        if serial_result["status"] != "ok":
            code = serial_result.get("error_code") or "E_INTERNAL"
            error = PipelineError(
                code,
                "serial baseline synthesis did not complete",
                scheduler_status=serial_result["status"],
                scheduler=serial_result["report"],
            )
            _write_error(root, error, stage)
            return {
                "status": serial_result["status"],
                "result_code": code,
                "exit_code": ERROR_EXIT_CODES.get(code, 1),
                "scheduler_report": serial_result["report"],
                "exit_report_written": False,
            }
        serial_ledger = build_ledger(
            manifest,
            serial_result,
            rules,
            game,
            document_kind="serial_ledger",
        )

        stage = "verification"
        verification = compare_ledgers(parallel_ledger, serial_ledger)
        truth = read_json(root / "truth" / "ledger.json")
        truth_comparison = compare_to_truth(parallel_ledger, truth)
        if not verification["full_consistency"]:
            result_code = verification["result_code"]
        elif not truth_comparison["full_consistency"]:
            result_code = "E_LEDGER_HASH"
        elif parallel_ledger["result_codes"]:
            result_code = parallel_ledger["result_codes"][0]
        else:
            result_code = "OK"

        stage = "persistence"
        persist_ledger(output_dir / LEDGER_FILENAME, parallel_ledger)
        persist_ledger(output_dir / SERIAL_LEDGER_FILENAME, serial_ledger)
        atomic_write_json(
            output_dir / "budget_report.json", parallel_ledger["budget_report"]
        )
        atomic_write_json(output_dir / "verification.json", verification)
        atomic_write_json(output_dir / "truth_comparison.json", truth_comparison)
        run_report = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "scenario": game.get("scenario"),
            "seed": game.get("seed"),
            "event_count": len(events),
            "tick_count": len(parallel_ledger["tick_records"]),
            "chunk_count": len(manifest["chunks"]),
            "chunk_ticks": effective_chunk_ticks,
            "parallel_threads_requested": threads,
            "parallelism_effective": parallel_result["report"]["effective_parallelism"],
            "memory_budget_bytes": memory_budget_bytes,
            "parallel_ledger_hash": parallel_ledger["ledger_hash"],
            "serial_ledger_hash": serial_ledger["ledger_hash"],
            "business_hash": parallel_ledger["business_hash"],
            "verification_hash": verification["verification_hash"],
            "truth_comparison_hash": truth_comparison["truth_comparison_hash"],
            "full_consistency": verification["full_consistency"],
            "truth_match": truth_comparison["full_consistency"],
            "result_code": result_code,
            "overflow_count": parallel_ledger["budget_report"]["global_overflow_count"],
        }
        run_report["run_report_hash"] = hash_without(run_report, "run_report_hash")
        atomic_write_json(output_dir / "run_report.json", run_report)
        stage = "artifact_manifest"
        artifact_hashes = _artifact_hashes(output_dir)
        exit_report = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "scenario": game.get("scenario"),
            "status": "committed",
            "result_code": result_code,
            "parallel_ledger_hash": parallel_ledger["ledger_hash"],
            "serial_ledger_hash": serial_ledger["ledger_hash"],
            "business_hash": parallel_ledger["business_hash"],
            "verification_hash": verification["verification_hash"],
            "truth_comparison_hash": truth_comparison["truth_comparison_hash"],
            "artifact_manifest_hash": artifact_hashes["artifact_manifest_hash"],
            "summary": {
                "full_consistency": verification["full_consistency"],
                "truth_match": truth_comparison["full_consistency"],
                "event_count": len(parallel_ledger["event_records"]),
                "tick_count": len(parallel_ledger["tick_records"]),
                "overflow_count": parallel_ledger["budget_report"]["global_overflow_count"],
                "effective_parallelism": parallel_result["report"]["effective_parallelism"],
                "degradation_count": len(parallel_result["report"]["degradations"]),
                "business_hash": parallel_ledger["business_hash"],
            },
            "artifacts": {
                "parallel_ledger": "parallel_ledger.json",
                "serial_ledger": "serial_ledger.json",
                "budget_report": "budget_report.json",
                "verification": "verification.json",
                "truth_comparison": "truth_comparison.json",
                "run_report": "run_report.json",
                "artifact_hashes": "artifact_hashes.json",
            },
        }
        exit_report["exit_report_hash"] = hash_without(
            exit_report, "exit_report_hash"
        )
        atomic_write_json(output_dir / EXIT_REPORT_FILENAME, exit_report)
        return {
            "status": "ok",
            "result_code": result_code,
            "exit_code": ERROR_EXIT_CODES.get(result_code, 1),
            "exit_report_written": True,
            "exit_report": exit_report,
            "verification": verification,
            "truth_comparison": truth_comparison,
            "parallel_ledger": parallel_ledger,
            "serial_ledger": serial_ledger,
            "manifest": manifest,
        }
    except PipelineError as exc:
        _write_error(root, exc, stage)
        return {
            "status": "failed",
            "result_code": exc.code,
            "exit_code": exc.exit_code,
            "error": exc.to_dict(),
            "exit_report_written": False,
        }
    except Exception as exc:
        error = PipelineError(
            "E_INTERNAL",
            "unexpected pipeline failure",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        _write_error(root, error, stage)
        return {
            "status": "failed",
            "result_code": error.code,
            "exit_code": error.exit_code,
            "error": error.to_dict(),
            "exit_report_written": False,
        }


def recover_workspace(root: Path) -> dict[str, Any]:
    root = Path(root)
    ledger_path = root / "output" / LEDGER_FILENAME
    ledger = load_and_verify_ledger(ledger_path)
    report = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "status": "recovered_from_committed_ledger",
        "source": "output/parallel_ledger.json",
        "ledger_hash": ledger["ledger_hash"],
        "game": ledger["game"],
        "summary": ledger["summary"],
        "result_codes": ledger["result_codes"],
        "exit_report_reconstructed": False,
    }
    report["recovery_hash"] = hash_without(report, "recovery_hash")
    atomic_write_json(root / "output" / "recovery_report.json", report)
    return report


def read_exit_report(root: Path) -> dict[str, Any] | None:
    path = Path(root) / "output" / EXIT_REPORT_FILENAME
    if not path.exists():
        return None
    report = read_json(path)
    if report.get("exit_report_hash") != hash_without(report, "exit_report_hash"):
        raise PipelineError("E_LEDGER_HASH", "exit report hash mismatch", path=str(path))
    return report
