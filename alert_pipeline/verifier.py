from __future__ import annotations

from typing import Any

from .config import PipelineConfig
from .correlate import build_ledger
from .dedup import deduplicate_serial
from .errors import PipelineError
from .logging_utils import StructuredLogger
from .spec import LoadedDataset
from .storage import LedgerStore
from .util import digest_json, write_json_file


def _declared_refs(aggregate: Any) -> tuple[str, ...]:
    refs: set[str] = set()
    for record in aggregate.source_records:
        refs.update(record.get("root_candidate_refs", []))
    return tuple(sorted(refs))


def _first_diff(
    before: Any, after: Any, prefix: str
) -> tuple[str, Any, Any] | None:
    if type(before) is not type(after):
        return prefix, before, after
    if isinstance(before, dict):
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before:
                return path, None, after[key]
            if key not in after:
                return path, before[key], None
            found = _first_diff(before[key], after[key], path)
            if found is not None:
                return found
        return None
    if isinstance(before, list):
        if len(before) != len(after):
            return prefix, before, after
        for index, (left, right) in enumerate(zip(before, after)):
            found = _first_diff(left, right, f"{prefix}[{index}]")
            if found is not None:
                return found
        return None
    if before != after:
        return prefix, before, after
    return None


def _verify_failure(
    alert_id: str | None,
    field: str,
    expected: Any,
    actual: Any,
    evidence: str,
    **extra: Any,
) -> PipelineError:
    details = {
        "alert_id": alert_id,
        "expected": expected,
        "actual": actual,
        "evidence": evidence,
        **extra,
    }
    return PipelineError(
        "E_VERIFY",
        "verification found a mismatch in the current conclusion ledger",
        field=field,
        details=details,
    )


def _ledger_content_digest(content: dict[str, Any]) -> str:
    semantic = {
        key: value
        for key, value in content.items()
        if key not in {"content_digest", "candidate_services", "paths"}
    }
    return digest_json(semantic)


def verify_pipeline(
    config: PipelineConfig,
    dataset: LoadedDataset,
    *,
    logger: StructuredLogger | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    logger = logger or StructuredLogger(config.log_path)
    with LedgerStore(config.database_path) as store:
        generation = store.current_generation()
        if generation is None:
            raise _verify_failure(
                None,
                "generation",
                "a complete current generation",
                None,
                "no complete current generation exists",
            )
        requested_digest = config.semantic_digest(dataset)
        if (
            generation.declaration_version != config.declaration_version
            or generation.seed != config.seed
            or generation.semantic_digest != requested_digest
        ):
            raise _verify_failure(
                None,
                "declaration_version/seed/semantic_digest",
                {
                    "declaration_version": generation.declaration_version,
                    "seed": generation.seed,
                    "semantic_digest": generation.semantic_digest,
                },
                {
                    "declaration_version": config.declaration_version,
                    "seed": config.seed,
                    "semantic_digest": requested_digest,
                },
                "original inputs do not match the declared current generation",
                generation_id=generation.generation_id,
            )
        unknown = store.unknown_attempts(generation.generation_id)
        if unknown:
            raise _verify_failure(
                None,
                "unknown_attempts",
                "no unresolved batch attempts",
                list(unknown),
                "the current generation has unresolved batches and cannot be verified",
                generation_id=generation.generation_id,
            )

        expected_aggregates = deduplicate_serial(dataset.alerts)
        refs = {item.alert_id: _declared_refs(item) for item in expected_aggregates}
        expected_ledger = build_ledger(
            expected_aggregates,
            dataset.graph,
            dataset.rules,
            refs,
        )
        expected_aggregate_by_id = {
            item.alert_id: item.to_public_dict() for item in expected_aggregates
        }
        expected_ledger_by_id = {
            item.alert_id: item.to_public_dict() for item in expected_ledger
        }

        stored_aggregate_records = {
            row["alert_id"]: row for row in store.load_current_aggregate_records()
        }
        stored_ledger_records = {
            row["alert_id"]: row for row in store.load_current_ledger_records()
        }

        for alert_id in sorted(
            set(expected_aggregate_by_id) | set(stored_aggregate_records)
        ):
            expected = expected_aggregate_by_id.get(alert_id)
            actual_row = stored_aggregate_records.get(alert_id)
            actual = actual_row["content"] if actual_row else None
            diff = (
                _first_diff(expected, actual, "aggregate")
                if expected is not None and actual is not None
                else None
            )
            if expected is None or actual is None or diff is not None:
                field = diff[0] if diff is not None else "aggregate.presence"
                error = _verify_failure(
                    alert_id,
                    field,
                    expected,
                    actual,
                    "raw alert stream recomputation differs from the aggregate ledger",
                )
                line_source = expected if expected is not None else actual
                if line_source and line_source.get("participant_lines"):
                    error.line = int(line_source["participant_lines"][0])
                raise error
            digest = digest_json(actual)
            if actual_row["content_digest"] != digest:
                raise _verify_failure(
                    alert_id,
                    "aggregate.content_digest",
                    digest,
                    actual_row["content_digest"],
                    "stored aggregate digest does not match its canonical content",
                )

        for alert_id in sorted(
            set(expected_ledger_by_id) | set(stored_ledger_records)
        ):
            expected = expected_ledger_by_id.get(alert_id)
            actual_row = stored_ledger_records.get(alert_id)
            actual = actual_row["content"] if actual_row else None
            diff = (
                _first_diff(expected, actual, "ledger")
                if expected is not None and actual is not None
                else None
            )
            if expected is None or actual is None or diff is not None:
                field = diff[0] if diff is not None else "ledger.presence"
                error = _verify_failure(
                    alert_id,
                    field,
                    expected,
                    actual,
                    "raw graph/rule recomputation differs from the correlation ledger",
                )
                aggregate_source = (expected or actual or {}).get("aggregate", {})
                if aggregate_source.get("participant_lines"):
                    error.line = int(aggregate_source["participant_lines"][0])
                raise error
            digest = _ledger_content_digest(actual)
            if actual_row["content_digest"] != digest:
                raise _verify_failure(
                    alert_id,
                    "ledger.content_digest",
                    digest,
                    actual_row["content_digest"],
                    "stored ledger digest does not match its canonical content",
                )

        batches = store.committed_batches(generation.generation_id)
        stale = store.stale_counts()
        report = {
            "business_code": "OK",
            "generation_id": generation.generation_id,
            "declaration_version": generation.declaration_version,
            "seed": generation.seed,
            "format_version": generation.format_version,
            "semantic_digest": generation.semantic_digest,
            "alerts_checked": dataset.alert_count,
            "aggregates_checked": len(expected_aggregates),
            "ledger_entries_checked": len(expected_ledger),
            "batches_checked": len(batches),
            "suppressed_total": sum(item.suppressed_count for item in expected_aggregates),
            "source_format_versions": sorted(
                {item.format_version for item in dataset.alerts}
            ),
            "stale_counts": stale,
            "unknown_batches": [],
            "first_mismatch": None,
            "verification_digest": digest_json(
                {
                    "generation": generation.generation_id,
                    "semantic_digest": generation.semantic_digest,
                    "aggregates": [item.to_public_dict() for item in expected_aggregates],
                    "ledger": [item.to_public_dict() for item in expected_ledger],
                }
            ),
        }
        if write_report:
            report_path = config.output_dir / "verification_report.json"
            write_json_file(report_path, report)
            report["report_path"] = str(report_path.resolve())
        logger.write(
            "pipeline_verify",
            business_code="OK",
            generation_id=generation.generation_id,
            alerts_checked=dataset.alert_count,
            aggregates_checked=len(expected_aggregates),
            verification_digest=report["verification_digest"],
        )
        return report
