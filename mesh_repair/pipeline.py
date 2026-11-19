"""Fixed generation -> chunking -> repair -> merge -> verification sequence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig, config_from_dict, load_config
from .errors import EXIT_CODES, BusinessError, ErrorCode
from .generator import baseline_stats, generate_sample
from .hashing import sha256_file, sha256_value, write_json
from .io_ply import write_ply
from .preprocessing import partition_blocks, weld_vertices, write_chunk_manifest
from .quality import COMMON_CHECK_NAMES, compute_quality
from .repair import build_repair_plan, execute_repair
from .verification import validate_files


ARTIFACT_ORDER = (
    "baseline.ply",
    "defective.ply",
    "ground_truth.json",
    "welded.ply",
    "chunks.json",
    "merged.ply",
    "repair_actions.json",
    "verifier.json",
)


@dataclass(frozen=True)
class RunResult:
    status: str
    business_code: str
    exit_code: int
    report: dict[str, Any]
    hashes: dict[str, Any]
    output_dir: Path


def _cross_check_quality_and_verifier(
    quality: dict[str, Any], verifier: dict[str, Any]
) -> dict[str, Any]:
    discrepancies: list[dict[str, Any]] = []
    quality_checks = quality.get("checks_by_name", {})
    verifier_checks = verifier.get("checks_by_name", {})
    compared: list[str] = []
    for name in COMMON_CHECK_NAMES:
        if name not in quality_checks or name not in verifier_checks:
            discrepancies.append(
                {
                    "criterion": "check_presence",
                    "check": name,
                    "pipeline": name in quality_checks,
                    "verifier": name in verifier_checks,
                    "location": {},
                }
            )
            continue
        compared.append(name)
        pipeline_status = quality_checks[name].get("status")
        verifier_status = verifier_checks[name].get("status")
        if pipeline_status != verifier_status:
            failed_check = (
                quality_checks[name]
                if pipeline_status == "fail"
                else verifier_checks[name]
            )
            discrepancies.append(
                {
                    "criterion": "check_status_mismatch",
                    "check": name,
                    "pipeline": pipeline_status,
                    "verifier": verifier_status,
                    "location": failed_check.get("location", {}),
                }
            )

    quality_metrics = quality.get("metrics", {})
    verifier_metrics = verifier.get("metrics", {})
    integer_metrics = (
        "component_count",
        "face_count",
        "nonmanifold_edge_count",
        "boundary_edge_count",
        "inconsistent_winding_edge_count",
        "self_intersection_count",
    )
    for name in integer_metrics:
        if quality_metrics.get(name) != verifier_metrics.get(name):
            discrepancies.append(
                {
                    "criterion": "metric_mismatch",
                    "metric": name,
                    "pipeline": quality_metrics.get(name),
                    "verifier": verifier_metrics.get(name),
                    "location": {"criterion": name},
                }
            )
    for name in ("volume", "area"):
        pipeline_value = float(quality_metrics.get(name, float("nan")))
        verifier_value = float(verifier_metrics.get(name, float("nan")))
        scale = max(abs(pipeline_value), abs(verifier_value), 1.0e-15)
        if abs(pipeline_value - verifier_value) / scale > 1.0e-9:
            discrepancies.append(
                {
                    "criterion": "metric_mismatch",
                    "metric": name,
                    "pipeline": pipeline_value,
                    "verifier": verifier_value,
                    "location": {"criterion": name},
                }
            )
    if (
        quality.get("status") != verifier.get("status")
        or quality.get("business_code") != verifier.get("business_code")
    ):
        discrepancies.append(
            {
                "criterion": "conclusion_mismatch",
                "pipeline": {
                    "status": quality.get("status"),
                    "business_code": quality.get("business_code"),
                },
                "verifier": {
                    "status": verifier.get("status"),
                    "business_code": verifier.get("business_code"),
                },
                "location": verifier.get("first_failure", {}).get("location", {}),
            }
        )
    return {
        "status": "pass" if not discrepancies else "fail",
        "compared_check_count": len(compared),
        "discrepancy_count": len(discrepancies),
        "first_mismatch": discrepancies[0] if discrepancies else None,
        "discrepancies": discrepancies,
    }


def _effective_config(
    config_path: str | Path,
    *,
    threads: int | None,
    memory_budget_bytes: int | None,
    block_time_budget_ms: float | None,
) -> PipelineConfig:
    base = load_config(config_path)
    overridden = base.with_overrides(
        threads=threads,
        memory_budget_bytes=memory_budget_bytes,
        block_time_budget_ms=block_time_budget_ms,
    )
    return config_from_dict(overridden.to_dict())


def run_fixed_pipeline(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    threads: int | None = None,
    memory_budget_bytes: int | None = None,
    block_time_budget_ms: float | None = None,
) -> RunResult:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    steps: dict[str, dict[str, Any]] = {}
    error: BusinessError | None = None
    config: PipelineConfig | None = None
    report: dict[str, Any] = {
        "schema_version": 1,
        "sequence": [
            "generate",
            "weld",
            "partition_blocks",
            "plan_repair",
            "parallel_block_repair",
            "deterministic_merge",
            "independent_verification",
            "quality_cross_check",
        ],
    }

    def record(filename: str) -> None:
        target = output / filename
        if target.exists():
            steps[filename] = {
                "sha256": sha256_file(target),
                "size_bytes": int(target.stat().st_size),
            }

    try:
        config = _effective_config(
            config_path,
            threads=threads,
            memory_budget_bytes=memory_budget_bytes,
            block_time_budget_ms=block_time_budget_ms,
        )
        report["config"] = config.to_dict()
        report["config_sha256"] = sha256_value(config.to_dict())

        generation = generate_sample(config, output)
        record("baseline.ply")
        record("defective.ply")
        record("ground_truth.json")
        baseline = generation.baseline
        defective = generation.defective
        truth = generation.truth

        weld_result = weld_vertices(defective.vertices, defective.faces, config.weld_tolerance)
        welded = weld_result.mesh
        write_ply(output / "welded.ply", welded.vertices, welded.faces)
        record("welded.ply")

        manifest = partition_blocks(
            welded,
            weld_tolerance=config.weld_tolerance,
            max_faces_per_block=config.max_faces_per_block,
        )
        write_chunk_manifest(output / "chunks.json", manifest)
        record("chunks.json")

        plan = build_repair_plan(
            welded,
            isolated_component_face_threshold=config.isolated_component_face_threshold,
            intersection_epsilon=config.self_intersection_epsilon,
        )
        repair_result = execute_repair(
            welded,
            manifest,
            plan,
            requested_threads=config.threads,
            memory_budget_bytes=config.memory_budget_bytes,
            block_time_budget_ms=config.block_time_budget_ms,
            boundary_tolerance=config.boundary_tolerance,
        )
        merged = repair_result.mesh
        write_ply(output / "merged.ply", merged.vertices, merged.faces)
        record("merged.ply")

        verifier_report = validate_files(output / "merged.ply", output / "ground_truth.json")
        write_json(output / "verifier.json", verifier_report)
        record("verifier.json")

        quality = compute_quality(
            merged,
            baseline,
            truth,
            weld_report=weld_result.to_dict(),
            actions=repair_result.actions,
            intersection_epsilon=config.self_intersection_epsilon,
        )
        verdict_cross_check = _cross_check_quality_and_verifier(quality, verifier_report)
        repair_actions_payload = {
            "schema_version": 1,
            "weld": weld_result.to_dict(),
            "repair": repair_result.to_dict(),
            "defect_alignment": quality["defect_alignment"],
            "misdeleted_legal_faces": quality["misdeleted_legal_faces"],
            "legal_face_geometry_preserved": quality["legal_face_geometry_preserved"],
        }
        write_json(output / "repair_actions.json", repair_actions_payload)
        record("repair_actions.json")

        if verdict_cross_check["status"] != "pass":
            error = BusinessError(
                ErrorCode.E_VERIFIER_MISMATCH,
                "pipeline and independent verifier conclusions differ",
                location=verdict_cross_check["first_mismatch"].get("location", {}),
                details={"first_mismatch": verdict_cross_check["first_mismatch"]},
            )
        elif quality["status"] != "pass":
            code = ErrorCode(quality["business_code"])
            error = BusinessError(
                code,
                "repair quality thresholds were not met",
                location=quality.get("first_failure", {}).get("location", {}),
                details={"first_failure": quality.get("first_failure")},
            )
        elif verifier_report["status"] != "pass":
            code = ErrorCode(verifier_report.get("business_code", "E_RESIDUAL_TOPOLOGY"))
            error = BusinessError(
                code,
                "independent verifier rejected the merged mesh",
                location=verifier_report.get("first_failure", {}).get("location", {}),
                details={"first_failure": verifier_report.get("first_failure")},
            )

        report.update(
            {
                "generation": {
                    "baseline": truth["baseline"],
                    "defective_vertex_count": defective.vertex_count,
                    "defective_face_count": defective.face_count,
                    "injection_count": truth["injection_count"],
                    "defect_record_count": truth["defect_record_count"],
                    "counts_by_type": truth["counts_by_type"],
                },
                "preprocessing": {
                    "weld": weld_result.to_dict(),
                    "chunk_summary": {
                        "block_count": len(manifest.blocks),
                        "component_count": manifest.component_count,
                        "split_rule": manifest.split_rule,
                        "shared_boundary_vertex_count": len(manifest.shared_boundary_vertices),
                    },
                },
                "repair": {
                    "plan": plan.to_dict(),
                    "actions_summary": {
                        key: value
                        for key, value in repair_result.actions.items()
                        if key not in {"deleted_faces", "flipped_faces"}
                    },
                    "block_summaries": list(repair_result.block_summaries),
                    "boundary": repair_result.boundary_resolution.to_dict(),
                    "scheduler": repair_result.scheduler_report.to_dict(),
                    "defect_alignment": quality["defect_alignment"],
                    "misdeleted_legal_faces": quality["misdeleted_legal_faces"],
                    "legal_face_geometry_preserved": quality["legal_face_geometry_preserved"],
                },
                "quality": quality,
                "verifier": {
                    "status": verifier_report["status"],
                    "business_code": verifier_report["business_code"],
                    "metrics": verifier_report["metrics"],
                    "first_failure": verifier_report.get("first_failure"),
                },
                "quality_verifier_cross_check": verdict_cross_check,
            }
        )
    except BusinessError as exc:
        error = exc
    except Exception as exc:  # pragma: no cover - defensive process boundary
        error = BusinessError(
            ErrorCode.E_INTERNAL,
            "unhandled pipeline exception",
            details={"error": repr(exc)},
        )

    report["status"] = "fail" if error is not None else "pass"
    report["business_code"] = error.code.value if error is not None else "OK"
    report["exit_code"] = 0 if error is None else EXIT_CODES[error.code]
    if error is not None:
        report["error"] = error.to_dict()
    report["step_order"] = [name for name in ARTIFACT_ORDER if name in steps]
    report["step_hashes"] = {name: steps[name]["sha256"] for name in report["step_order"]}
    write_json(output / "report.json", report)
    report_hash = sha256_file(output / "report.json")
    hashes_payload = {
        "schema_version": 1,
        "algorithm": "sha256",
        "status": report["status"],
        "business_code": report["business_code"],
        "step_order": report["step_order"],
        "steps": {name: steps[name] for name in report["step_order"]},
        "report.json": {
            "sha256": report_hash,
            "size_bytes": int((output / "report.json").stat().st_size),
        },
    }
    write_json(output / "hashes.json", hashes_payload)
    return RunResult(
        status=report["status"],
        business_code=report["business_code"],
        exit_code=int(report["exit_code"]),
        report=report,
        hashes=hashes_payload,
        output_dir=output,
    )
