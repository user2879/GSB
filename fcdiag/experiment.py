"""Reusable experiment helpers: incremental runs, full rebuilds and injections."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .errors import E_INPUT_INVALID, InputInvalidError
from .generator import PIPELINE_YAML
from .injection import Injection, InjectionReport, load_injections
from .loader import load_config, load_pipeline, load_samples
from .models import Artifact, ReceiptRecord
from .receipts import parse_receipts
from .runner import RunResult, Runner
from .state import WorkState, apply_command, apply_commands


def result_signature(result: RunResult) -> dict:
    """Serializable field-level signature used for incremental/full equality."""

    return {
        "artifacts": {
            key: artifact.to_dict()
            for key, artifact in sorted(result.artifacts.items())
        },
        "cache_keys": dict(sorted(result.cache_keys.items())),
        "ledger": [entry.to_dict() for entry in result.ledger],
        "events": list(result.events),
        "failures": list(result.failures),
        "conversion_diffs": list(result.conversion_diffs),
        "receipts": [record.to_dict() for record in result.receipts],
        "quant_logs": {key: list(value) for key, value in sorted(result.quant_logs.items())},
        "pending": [task.sort_key for task in result.pending],
        "budget_used": dict(sorted(result.budget_used.items())),
    }


def incremental_runs(
    initial_state: WorkState,
    commands: Tuple[dict, ...],
    baseline_receipts: Tuple[ReceiptRecord, ...],
    max_workers: int = 1,
) -> Tuple[RunResult, ...]:
    states = apply_commands(initial_state, commands)
    results = []
    for state in states:
        results.append(
            Runner(
                state,
                baseline_receipts,
                baseline_artifacts=None,
                max_workers=max_workers,
            ).run()
        )
    return tuple(results)


def full_rebuild(
    final_state: WorkState,
    baseline_receipts: Tuple[ReceiptRecord, ...],
    max_workers: int = 1,
) -> RunResult:
    return Runner(
        final_state,
        baseline_receipts,
        baseline_artifacts=None,
        max_workers=max_workers,
    ).run()


def assert_run_equal(left: RunResult, right: RunResult) -> None:
    left_sig = result_signature(left)
    right_sig = result_signature(right)
    if left_sig != right_sig:
        for key in left_sig:
            if left_sig[key] != right_sig[key]:
                raise AssertionError(f"incremental/full mismatch in {key}")
        raise AssertionError("incremental/full mismatch")


def _mutate_pipeline_text(text: str, injection: Injection) -> str:
    if injection.injection_id == "INF-002":
        return text.replace("bucket_count: 5", "bucket_count: 0")
    if injection.injection_id == "INF-003":
        return text.replace(
            "    upstream:\n      - standardize\n    params:\n      degree: 2",
            "    upstream:\n      - standardize\n      - missing_stage\n    params:\n      degree: 2",
        )
    return text


def run_injections(
    sample_text: str,
    pipeline_text: str,
    config_doc: dict,
    receipt_texts: Dict[int, str],
    injection_doc: dict,
) -> Tuple[InjectionReport, ...]:
    injections = load_injections(injection_doc)
    reports: List[InjectionReport] = []
    for injection in injections:
        observed_code = None
        locator = injection.target
        message = ""
        try:
            if injection.injection_id == "INF-001":
                bad_config = copy.deepcopy(config_doc)
                # The weight is injected into the pipeline, matching the target locator.
                bad_pipeline = pipeline_text.replace("- 1.0", "- NaN", 1)
                load_pipeline(bad_pipeline)
            elif injection.injection_id in ("INF-002", "INF-003"):
                load_pipeline(_mutate_pipeline_text(pipeline_text, injection))
            elif injection.injection_id == "INF-004":
                bad_text = receipt_texts[3]
                lines = bad_text.splitlines()
                lines[1] = lines[1][:-1]
                parse_receipts(3, "\n".join(lines) + "\n")
            elif injection.injection_id == "INF-005":
                bad_text = receipt_texts[2]
                lines = bad_text.splitlines()
                lines.append(lines[1])
                parse_receipts(2, "\n".join(lines) + "\n")
            else:
                raise KeyError(injection.injection_id)
        except InputInvalidError as exc:
            observed_code = E_INPUT_INVALID
            locator = exc.violations[0].locator
            message = "; ".join(violation.message for violation in exc.violations)
        reports.append(
            InjectionReport(
                injection_id=injection.injection_id,
                kind=injection.kind,
                expected_code=injection.expected_code,
                observed_code=observed_code or "NO_ERROR",
                hit=observed_code == injection.expected_code,
                locator=locator,
                message=message,
            )
        )
    return tuple(reports)
