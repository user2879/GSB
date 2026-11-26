"""Repeatable end-to-end acceptance suite for the local GSB demo."""

from __future__ import annotations

import copy
import json
import platform
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import pytest
import yaml

from gsb.commands import apply_commands, load_command_stream, run_command_stream
from gsb.context import issues_payload, load_project
from gsb.models import EditCommand
from gsb.pipeline import run_pipeline
from gsb.samples import scan_png_directory
from gsb.snapshot import SnapshotStore
from gsb.synth import FLIP_TEMPLATE_VERSION
from gsb.trace import analyze_attribution
from gsb.util import sha256_obj
from gsb.verification import compare_core_results, truth_accuracy


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    **details: Any,
) -> None:
    checks.append({"name": name, "passed": bool(passed), **details})


def _recognition_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        recognition["keypoint_id"]: recognition
        for sample in report.get("samples", [])
        for recognition in sample.get("recognitions", [])
    }


def _status_counts(
    invalidation: list[dict[str, Any]],
    artifact_type: str,
) -> dict[str, int]:
    return dict(
        Counter(
            item["status"]
            for item in invalidation
            if item["artifact_type"] == artifact_type
        )
    )


def _tk_status() -> dict[str, Any]:
    try:
        import tkinter  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"importable": False, "error": str(exc)}
    return {"importable": True, "error": None}


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "Pillow": PIL.__version__,
        "PyYAML": yaml.__version__,
        "pytest": pytest.__version__,
        "tkinter": _tk_status(),
        "network_policy": "local files and 127.0.0.1 only; no network client is used",
    }


def run_acceptance(
    root: str | Path,
    *,
    output_path: str | Path | None = "artifacts/validation.json",
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    checks: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": 1,
        "environment": _environment(),
    }

    context = load_project(root=root_path)
    templates = context.require_templates()
    png_issues = scan_png_directory(root_path / "samples", config=context.config)
    _check(
        checks,
        "configuration_valid",
        not context.template_issues and not context.dataset_issues and not png_issues,
        template_issues=issues_payload(context.template_issues),
        dataset_issues=issues_payload(context.dataset_issues),
        png_issues=issues_payload(png_issues),
    )
    validation = {
        "template_issues": issues_payload(context.template_issues),
        "dataset_issues": issues_payload(context.dataset_issues),
        "png_issues": issues_payload(png_issues),
        "sample_count": len(context.dataset.sample_order),
        "png_count": len(list((root_path / "samples").rglob("*.png"))),
    }
    report["validation"] = validation

    baseline = run_pipeline(
        context.dataset,
        config=context.config,
        templates=templates,
    )
    baseline_accuracy = truth_accuracy(baseline.report, context.dataset)
    _check(
        checks,
        "baseline_truth_accuracy",
        baseline_accuracy["all_keypoints_correct"]
        and baseline_accuracy["all_recognitions_correct"]
        and baseline_accuracy["all_reading_correct"],
        accuracy=baseline_accuracy,
        failures=baseline.report.get("failures", []),
        pending=len(baseline.pending_ids),
    )
    observed_workers = max(
        [
            sample["concurrency"]["candidate_max_observed"]
            for sample in baseline.report["samples"]
        ]
        + [baseline.report["budget"]["max_observed_workers"]]
    )
    _check(
        checks,
        "baseline_complete_within_budget",
        not baseline.pending_ids
        and baseline.report["budget"]["consumed"] <= baseline.report["budget"]["limit"],
        budget=baseline.report["budget"],
    )
    _check(
        checks,
        "worker_limit_respected",
        observed_workers <= context.config.recognition.max_workers,
        observed_workers=observed_workers,
        configured_limit=context.config.recognition.max_workers,
    )

    serial = run_pipeline(
        context.dataset,
        config=context.config.with_workers(1),
        templates=templates,
    )
    serial_comparison = compare_core_results(baseline.report, serial.report)
    _check(
        checks,
        "serial_matches_concurrent",
        serial_comparison["equal"],
        configured_workers=context.config.recognition.max_workers,
        serial_workers=1,
    )

    quantization: dict[str, Any] = {}
    confidence_fingerprints: dict[str, str] = {}
    for level in ("Q4", "Q6", "Q8", "Q10"):
        result = run_pipeline(
            context.dataset,
            config=context.config.with_quantization(level).with_budget(10**12),
            templates=templates,
        )
        accuracy = truth_accuracy(result.report, context.dataset)
        recognitions = _recognition_map(result.report)
        confidence_vector = {
            keypoint_id: [
                item.get("char"),
                item.get("score"),
                item.get("confidence"),
                item.get("kp_residual_ppm"),
            ]
            for keypoint_id, item in sorted(recognitions.items())
        }
        confidence_fingerprints[level] = sha256_obj(confidence_vector)
        quantization[level] = {
            "accuracy": accuracy,
            "confidence_fingerprint": confidence_fingerprints[level],
            "recognition_characters": {
                keypoint_id: item.get("char")
                for keypoint_id, item in sorted(recognitions.items())
            },
        }
        _check(
            checks,
            f"quantization_{level}_truth_accuracy",
            accuracy["all_keypoints_correct"]
            and accuracy["all_recognitions_correct"]
            and accuracy["all_reading_correct"],
            accuracy=accuracy,
        )
    _check(
        checks,
        "quantization_changes_fixed_values",
        len(set(confidence_fingerprints.values())) == len(confidence_fingerprints),
        confidence_fingerprints=confidence_fingerprints,
    )
    report["quantization_levels"] = {
        "levels": quantization,
        "confidence_changed": len(set(confidence_fingerprints.values())) > 1,
    }

    flip_context = load_project(
        root=root_path,
        config_path="configs/quantization-flip.yaml",
        dataset_path="samples/quantization_flip.dataset.yaml",
    )
    flip_templates = flip_context.require_templates()
    flip_outputs: dict[str, Any] = {}
    for level in ("Q4", "Q6", "Q8", "Q10"):
        result = run_pipeline(
            flip_context.dataset,
            config=flip_context.config.with_quantization(level),
            templates=flip_templates,
        )
        sampled = result.report["samples"][0]
        recognition = sampled["recognitions"][0]
        flip_outputs[level] = {
            "char": recognition["char"],
            "score": recognition["score"],
            "confidence": recognition["confidence"],
            "block_bbox": recognition["block_bbox"],
            "quantized_fixed": sampled["keypoints"][0]["quantized_fixed"],
            "residual_ppm": recognition["kp_residual_ppm"],
        }
    _check(
        checks,
        "quantization_flip_fixture",
        flip_outputs["Q4"]["char"] == "己"
        and flip_outputs["Q6"]["char"] == "曰"
        and flip_outputs["Q8"]["char"] == "曰"
        and flip_outputs["Q10"]["char"] == "曰"
        and flip_outputs["Q4"]["confidence"] != flip_outputs["Q8"]["confidence"]
        and flip_outputs["Q4"]["block_bbox"] == flip_outputs["Q8"]["block_bbox"]
        and flip_outputs["Q4"]["quantized_fixed"] != flip_outputs["Q8"]["quantized_fixed"],
        template_version=FLIP_TEMPLATE_VERSION,
        outputs=flip_outputs,
    )

    flip_base = run_pipeline(
        flip_context.dataset,
        config=flip_context.config.with_quantization("Q8"),
        templates=flip_templates,
    )
    flip_attribution = analyze_attribution(
        flip_context.dataset,
        config=flip_context.config.with_quantization("Q8"),
        templates=flip_templates,
        base_result=flip_base,
        alternate_levels=("Q4",),
    )
    flip_sources = [item["source"] for item in flip_attribution["flips"]]
    _check(
        checks,
        "flip_attribution_identifies_score_rounding",
        flip_attribution["flip_count"] == 1
        and flip_sources == ["template_match_score_rounding"],
        flip_count=flip_attribution["flip_count"],
        sources=flip_sources,
        flips=flip_attribution["flips"],
    )
    report["quantization_flip"] = {
        "outputs": flip_outputs,
        "attribution": flip_attribution,
    }

    default_attribution = analyze_attribution(
        context.dataset,
        config=context.config,
        templates=templates,
        base_result=baseline,
        alternate_levels=("Q4", "Q6", "Q10"),
    )
    _check(
        checks,
        "reference_error_trace_complete",
        len(default_attribution["per_character"]) == sum(
            item["recognition_total"] for item in baseline_accuracy["per_sample"]
        )
        and default_attribution["reference"]["reference_bits"]
        == context.config.fixed_bits + 1,
        reference=default_attribution["reference"],
        character_records=len(default_attribution["per_character"]),
        default_flip_count=default_attribution["flip_count"],
    )
    report["reference_error_trace"] = {
        "reference": default_attribution["reference"],
        "per_character": default_attribution["per_character"],
        "flip_count": default_attribution["flip_count"],
    }

    annotation_command = [
        EditCommand(
            seq=1,
            op="replace_annotation_version",
            payload={
                "sample_id": "page_a",
                "annotation": "annotations/page_a.ann-v2.json",
            },
        )
    ]
    annotation_dataset, annotation_config, annotation_log = apply_commands(
        context.dataset,
        context.config,
        annotation_command,
    )
    annotation_result = run_pipeline(
        annotation_dataset,
        config=annotation_config,
        templates=templates,
        cache=baseline.cache,
        prior_recognition_results=baseline.recognition_results,
    )
    annotation_counts = {
        artifact: _status_counts(annotation_result.report["invalidation"], artifact)
        for artifact in ("annotation_version", "keypoint", "recognition", "reading")
    }
    page_b_recognition_reused = all(
        item["status"] == "reused"
        for item in annotation_result.report["invalidation"]
        if item["artifact_type"] == "recognition"
        and item["artifact_id"].startswith("char:page_b-")
    )
    _check(
        checks,
        "annotation_version_precise_invalidation",
        annotation_log[0]["status"] == "applied"
        and annotation_counts["keypoint"] == {"invalidated": 2, "reused": 34}
        and annotation_counts["recognition"] == {"invalidated": 2, "reused": 34}
        and annotation_counts["reading"] == {"invalidated": 2, "reused": 34}
        and page_b_recognition_reused
        and annotation_result.report["incremental"]["recognition_reuses"] == 34,
        counts=annotation_counts,
        incremental=annotation_result.report["incremental"],
        page_b_recognition_reused=page_b_recognition_reused,
    )
    report["annotation_invalidation"] = {
        "counts": annotation_counts,
        "incremental": annotation_result.report["incremental"],
        "page_b_recognition_reused": page_b_recognition_reused,
    }

    commands, command_issues = load_command_stream(root_path / "configs" / "commands.jsonl")
    command_seed = run_pipeline(
        context.dataset,
        config=context.config,
        templates=templates,
    )
    incremental_result, command_log, _ = run_command_stream(
        dataset=context.dataset,
        config=context.config,
        templates=templates,
        commands=commands,
        cache=command_seed.cache,
        prior_result=command_seed,
    )
    fresh_dataset, fresh_config, fresh_log = apply_commands(
        context.dataset,
        context.config,
        commands,
    )
    fresh_result = run_pipeline(
        fresh_dataset,
        config=fresh_config,
        templates=templates,
    )
    command_comparison = compare_core_results(
        incremental_result.report,
        fresh_result.report,
    )
    _check(
        checks,
        "command_incremental_matches_full_rebuild",
        not command_issues
        and all(item["status"] == "applied" for item in command_log)
        and all(item["status"] == "applied" for item in fresh_log)
        and command_comparison["equal"],
        command_count=len(commands),
        command_issues=issues_payload(command_issues),
        command_log=command_log,
        incremental=incremental_result.report["incremental"],
        final_config_hash=fresh_config.config_hash,
    )
    report["command_stream"] = {
        "command_count": len(commands),
        "command_log": command_log,
        "incremental": incremental_result.report["incremental"],
        "full_rebuild_equal": command_comparison["equal"],
        "final_config_hash": fresh_config.config_hash,
    }

    budget_config = context.config.with_budget(15000)
    budget_cache = None
    prior_recognition: dict[str, Any] = {}
    pending_ids: tuple[str, ...] = ()
    pending_counts: list[int] = []
    batch_consumed: list[int] = []
    budget_result = None
    for _ in range(16):
        budget_result = run_pipeline(
            context.dataset,
            config=budget_config,
            templates=templates,
            cache=budget_cache,
            prior_recognition_results=prior_recognition,
            pending_ids=pending_ids,
        )
        budget_cache = budget_result.cache
        prior_recognition = budget_result.recognition_results
        pending_ids = budget_result.pending_ids
        pending_counts.append(len(pending_ids))
        batch_consumed.append(budget_result.report["budget"]["consumed"])
        if not pending_ids:
            break
    assert budget_result is not None
    budget_comparison = compare_core_results(
        budget_result.report,
        baseline.report,
    )
    _check(
        checks,
        "point_budget_batches_match_full_run",
        pending_counts == [22, 8, 0]
        and all(
            pending_counts[index] >= pending_counts[index + 1]
            for index in range(len(pending_counts) - 1)
        )
        and budget_comparison["equal"],
        pending_counts=pending_counts,
        batch_consumed=batch_consumed,
        final_budget=budget_result.report["budget"],
        full_equal=budget_comparison["equal"],
    )
    report["point_budget"] = {
        "limit": budget_config.budget.max_template_points,
        "pending_counts": pending_counts,
        "batch_consumed": batch_consumed,
        "final_budget": budget_result.report["budget"],
        "full_equal": budget_comparison["equal"],
    }

    with tempfile.TemporaryDirectory(prefix="gsb-snapshot-acceptance-") as temporary:
        store = SnapshotStore(Path(temporary) / "store")
        first_commit = store.commit(
            baseline.report,
            completed_command_cursor=0,
            consumed_budget=baseline.report["budget"]["consumed"],
        )
        first_load = store.load_current()
        second_report = copy.deepcopy(baseline.report)
        second_report["snapshot_marker"] = "second-version"
        second_commit = store.commit(
            second_report,
            completed_command_cursor=12,
            consumed_budget=12345,
        )
        second_load = store.load_current()
        (second_commit.snapshot_dir / "stages" / "keypoints.json").write_text(
            "{\"corrupted\":true}\n",
            encoding="utf-8",
        )
        recovery = store.load_current()
        _check(
            checks,
            "snapshot_atomic_commit_and_recovery",
            first_load.current_valid
            and second_load.current_valid
            and first_commit.global_version != second_commit.global_version
            and recovery.recovered
            and not recovery.current_valid
            and recovery.report == baseline.report
            and any("hash mismatch" in issue for issue in recovery.issues),
            first_version=first_commit.global_version,
            second_version=second_commit.global_version,
            first_current_valid=first_load.current_valid,
            second_current_valid=second_load.current_valid,
            recovered=recovery.recovered,
            current_valid=recovery.current_valid,
            recovery_issues=list(recovery.issues),
            recovered_version=recovery.manifest.get("global_version")
            if recovery.manifest
            else None,
        )
        report["snapshot"] = {
            "first_version": first_commit.global_version,
            "second_version": second_commit.global_version,
            "first_current_valid": first_load.current_valid,
            "second_current_valid": second_load.current_valid,
            "recovered": recovery.recovered,
            "current_valid": recovery.current_valid,
            "recovery_issues": list(recovery.issues),
            "recovered_version": recovery.manifest.get("global_version")
            if recovery.manifest
            else None,
        }

    from gsb.injections import run_injection_suite

    injection_result = run_injection_suite(
        root=root_path,
        config_path=context.config_path,
        dataset_path=context.dataset_path,
        templates_path=context.templates_path,
        injections_path=root_path / "injections" / "injections.yaml",
    )
    codes = {item["id"]: item["code"] for item in injection_result["injections"]}
    expected_codes = {
        "INJ-01": "E_SAMPLE_FORMAT",
        "INJ-02": "E_SAMPLE_FORMAT",
        "INJ-03": "E_TEMPLATE_VARIANT_DIMENSION",
        "INJ-04": "E_CONFIG_INVALID",
    }
    surviving = {
        item["id"]: item.get("surviving_samples")
        for item in injection_result["injections"]
        if item["id"] in {"INJ-01", "INJ-02"}
    }
    _check(
        checks,
        "numbered_injections_isolated",
        injection_result["all_hit"]
        and codes == expected_codes
        and surviving
        == {"INJ-01": ["page_b"], "INJ-02": ["page_b"]},
        all_hit=injection_result["all_hit"],
        codes=codes,
        surviving_samples=surviving,
        config_issues=injection_result["config_issues"],
    )
    report["injections"] = injection_result

    passed = all(item["passed"] for item in checks)
    report["checks"] = checks
    report["summary"] = {
        "passed": passed,
        "check_count": len(checks),
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
        "config_hash": context.config.config_hash,
        "template_version": templates.template_version,
        "template_fingerprint": templates.content_fingerprint,
        "baseline_budget_consumed": baseline.report["budget"]["consumed"],
        "baseline_worker_limit": context.config.recognition.max_workers,
        "baseline_observed_workers": observed_workers,
        "baseline_pending": len(baseline.pending_ids),
    }
    report["status"] = "passed" if passed else "failed"

    if output_path is not None:
        destination = Path(output_path)
        if not destination.is_absolute():
            destination = root_path / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report
