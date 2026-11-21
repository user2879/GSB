from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from .engine import DiagnosticEngine, comparable_report
from .errors import Failure, sorted_failures
from .fingerprint import canonical_json, digest
from .loading import load_workspace, validate_exposure
from .models import EVENT_FIELDS, Exposure, WorkspaceState


def apply_batch(state: WorkspaceState, batch: dict[str, Any]) -> tuple[WorkspaceState, list[str]]:
    updated = state.clone()
    operation_names: list[str] = []
    extra_failures: list[Failure] = []
    batch_no = batch.get("batch")
    for operation in batch.get("operations", []):
        op_type = operation.get("type")
        operation_names.append(str(op_type))
        if op_type == "add_exposures":
            seen = set(updated.exposures)
            for raw in operation.get("events", []):
                exposure, failures = validate_exposure(raw, updated.config, seen)
                if failures:
                    for failure in failures:
                        extra_failures.append(
                            Failure(
                                failure.code,
                                f"batch {batch_no}: {failure.message}",
                                field_path=failure.field_path,
                                request_id=failure.request_id,
                                context={"batch": batch_no, "operation": op_type},
                            )
                        )
                    continue
                assert exposure is not None
                updated.exposures[exposure.request_id] = exposure
                seen.add(exposure.request_id)
        elif op_type == "fix_exposure":
            request_id = operation.get("request_id")
            patch = operation.get("patch", {})
            current = updated.exposures.get(request_id) if isinstance(request_id, str) else None
            if current is None:
                extra_failures.append(
                    Failure(
                        "E_LOG_INVALID",
                        f"batch {batch_no}: fix_exposure target not found",
                        field_path="request_id",
                        request_id=request_id if isinstance(request_id, str) else None,
                    )
                )
                continue
            if any(key not in EVENT_FIELDS for key in patch):
                extra_failures.append(
                    Failure(
                        "E_LOG_INVALID",
                        f"batch {batch_no}: fix_exposure patch contains unknown fields",
                        field_path="patch",
                        request_id=request_id,
                    )
                )
                continue
            raw = current.to_dict()
            raw.update(patch)
            if raw.get("request_id") != request_id:
                extra_failures.append(
                    Failure(
                        "E_LOG_INVALID",
                        f"batch {batch_no}: fix_exposure cannot change request_id",
                        field_path="patch.request_id",
                        request_id=request_id,
                    )
                )
                continue
            seen = set(updated.exposures) - {request_id}
            exposure, failures = validate_exposure(raw, updated.config, seen)
            if failures:
                for failure in failures:
                    extra_failures.append(
                        Failure(
                            failure.code,
                            f"batch {batch_no}: {failure.message}",
                            field_path=f"patch.{failure.field_path}",
                            request_id=request_id,
                            context={"batch": batch_no, "operation": op_type},
                        )
                    )
                continue
            assert exposure is not None
            updated.exposures[exposure.request_id] = exposure
        elif op_type == "switch_spec_version":
            version = operation.get("spec_version")
            if version not in updated.specs:
                extra_failures.append(
                    Failure(
                        "E_CONFIG_INVALID",
                        f"batch {batch_no}: active spec version {version!r} is absent",
                        field_path="active_spec_version",
                        context={"batch": batch_no, "operation": op_type},
                    )
                )
                continue
            updated.config = replace(updated.config, active_spec_version=version)
        elif op_type == "change_bucket_seed":
            seed = operation.get("seed")
            if not isinstance(seed, str) or not seed:
                extra_failures.append(
                    Failure(
                        "E_CONFIG_INVALID",
                        f"batch {batch_no}: bucket seed must be a non-empty string",
                        field_path="bucket_hash_seed",
                        context={"batch": batch_no, "operation": op_type},
                    )
                )
                continue
            updated.config = replace(updated.config, bucket_hash_seed=seed)
    updated.failures = sorted_failures(list(updated.failures) + extra_failures)
    return updated, operation_names


def _batch_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(plan.get("batches", []), key=lambda batch: batch.get("batch", 0))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _full_rebuild_from_disk(state: WorkspaceState) -> dict[str, Any]:
    """Materialize the current state and run the full path through the loader."""
    with tempfile.TemporaryDirectory(prefix="driftbench-full-rebuild-") as temporary:
        root = Path(temporary)
        _write_jsonl(root / "exposures.jsonl", [event.to_dict() for event in state.sorted_exposures()])
        _write_jsonl(root / "specs.jsonl", [state.specs[version].to_dict() for version in state.spec_versions()])
        (root / "config.json").write_text(
            json.dumps(state.config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "user_properties.json").write_text(
            json.dumps({"users": state.properties}, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        loaded = load_workspace(root)
        return DiagnosticEngine(full_rebuild=True, check_snapshot=False).run(loaded)


def run_edit_sequence(
    initial_state: WorkspaceState,
    *,
    snapshot_dir: str | None = None,
) -> dict[str, Any]:
    engine = DiagnosticEngine(snapshot_dir=snapshot_dir) if snapshot_dir else DiagnosticEngine()

    def compare(state: WorkspaceState, batch_no: int, operations: list[str], force: bool = False) -> dict[str, Any]:
        incremental = engine.run(state, batch_no=batch_no, force_commit=force)
        full = _full_rebuild_from_disk(state)
        left = canonical_json(comparable_report(incremental))
        right = canonical_json(comparable_report(full))
        left_digest = digest(comparable_report(incremental))
        right_digest = digest(comparable_report(full))
        return {
            "batch": batch_no,
            "operations": operations,
            "equal": left == right,
            "incremental_digest": left_digest,
            "full_digest": right_digest,
            "reuse": incremental["reuse"]["counts"],
            "reuse_by_stage": incremental["reuse"]["by_stage"],
            "reuse_consistent": incremental["reuse"]["consistent"],
            "incremental_total_ms": incremental["timing_ms"]["total_ms"],
            "full_total_ms": full["timing_ms"]["total_ms"],
            "final_report": incremental,
        }

    verification_rows: list[dict[str, Any]] = []
    initial_row = compare(initial_state, 0, ["__initial__"])
    verification_rows.append(initial_row)
    state = initial_state
    batches = _batch_rows(state.plan)
    final_report = initial_row["final_report"]
    for index, batch in enumerate(batches):
        state, operations = apply_batch(state, batch)
        row = compare(
            state,
            int(batch.get("batch", index + 1)),
            operations,
            force=index == len(batches) - 1,
        )
        verification_rows.append(row)
        final_report = row["final_report"]

    report_rows = []
    for row in verification_rows:
        report_rows.append({key: value for key, value in row.items() if key != "final_report"})
    return {
        "schema_version": 1,
        "report_type": "incremental_full_equivalence",
        "all_equal": all(row["equal"] for row in verification_rows),
        "comparison_scope": [
            "bucketing",
            "metrics",
            "drift",
            "attribution",
            "failures",
        ],
        "batches": report_rows,
        "final_state_fingerprint": state.state_fingerprint(),
        "final_report": final_report,
        "final_state": state,
    }
