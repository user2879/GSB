"""End-to-end deterministic fixed-point pipeline."""

from __future__ import annotations

from typing import Any

from .accumulation import build_quadratics
from .attribution import build_attribution
from .cache import ContentCache
from .dataset import Dataset
from .features import correlation_analysis, quantize_dataset
from .hashing import canonical_sha256, short_key
from .ledger import ReuseLedger
from .pathing import summarize_path
from .reporting import to_jsonable
from .solvers import lasso_order_audit, solve_lasso_path_level, solve_ridge_path_level


def _solver_material(solver_config: dict[str, Any]) -> dict[str, Any]:
    return to_jsonable(solver_config)


def run_pipeline(
    dataset: Dataset,
    config: dict[str, Any],
    *,
    cache_dir: str | None = None,
    previous_lineage: dict[str, dict[str, Any]] | None = None,
    worker_count: int | None = None,
    input_errors: list[dict[str, Any]] | None = None,
    command_results: list[dict[str, Any]] | None = None,
    command_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cache = ContentCache(cache_dir)
    ledger = ReuseLedger(previous_lineage)
    fixed = config["fixed"]
    scheduler_config = config["scheduler"]
    workers = scheduler_config["worker_count"] if worker_count is None else worker_count
    sample_ids = dataset.ids
    columns = list(dataset.columns)
    column_fingerprints = dataset.column_fingerprints()
    sample_fingerprints = [[sample_id, dataset.rows[sample_id].fingerprint] for sample_id in sample_ids]
    common_refs_material = {
        "fixed": fixed.fingerprint_material(),
        "sample_fingerprints": sample_fingerprints,
        "column_fingerprints": column_fingerprints,
    }

    qd = quantize_dataset(dataset, fixed)

    quantization_columns: list[dict[str, Any]] = []
    for j, name in enumerate(columns):
        artifact_id = f"quant:column={j:02d}:{name}"
        material = {
            "kind": "quantization_column",
            "column_index": j,
            "column_name": name,
            "column_fingerprint": column_fingerprints[j],
            "fixed": fixed.fingerprint_material(),
        }
        cache_key = short_key("quant", material)
        cached = cache.get(cache_key)
        cache_hit = cached is not None
        if not cache_hit:
            payload = {"integers": [qd.x[row][j] for row in range(len(sample_ids))], "records": qd.feature_records[j]}
            cache.put(cache_key, payload)
        ledger.record(
            artifact_id=artifact_id,
            stage="feature_quantization",
            cache_key=cache_key,
            cache_hit=cache_hit,
            basis="column_content_fingerprint+fixed_width_gear_match",
            sample_ids=sample_ids,
            columns=[name],
            material=common_refs_material | {"column_fingerprints": [column_fingerprints[j]]},
        )
        step = qd.feature_records[j][0]["step"] if qd.feature_records[j] else None
        quantization_columns.append({
            "column_index": j,
            "column": name,
            "quantization_step": step,
            "integers": [qd.x[row][j] for row in range(len(sample_ids))],
            "residuals": [format(qd.residuals[row][j], "f") for row in range(len(sample_ids))],
            "cache_key": cache_key,
        })

    correlation = correlation_analysis(
        qd,
        config["correlation_threshold"],
        config["collinearity_threshold"],
    )
    for pair in correlation["pairs"]:
        i, j = pair["i"], pair["j"]
        artifact_id = f"correlation:pair={i:02d},{j:02d}"
        material = {
            "kind": "fixed_correlation_pair",
            "i": i,
            "j": j,
            "column_i_fingerprint": column_fingerprints[i],
            "column_j_fingerprint": column_fingerprints[j],
            "fixed": fixed.fingerprint_material(),
            "strong_threshold": str(config["correlation_threshold"]),
            "collinear_threshold": str(config["collinearity_threshold"]),
        }
        cache_key = short_key("corr", material)
        cached_pair = cache.get(cache_key)
        cache_hit = cached_pair is not None
        if not cache_hit:
            cache.put(cache_key, to_jsonable(pair))
        ledger.record(
            artifact_id=artifact_id,
            stage="correlation",
            cache_key=cache_key,
            cache_hit=cache_hit,
            basis="two_column_fingerprints+covariance_caliber+thresholds_match",
            sample_ids=sample_ids,
            columns=pair["columns"],
            material=common_refs_material | {"column_fingerprints": [column_fingerprints[i], column_fingerprints[j]]},
        )
        pair["cache_key"] = cache_key

    quadratics = build_quadratics(
        dataset,
        qd,
        cache=cache,
        ledger=ledger,
        worker_count=workers,
        mac_budget=scheduler_config["mac_budget_per_batch"],
        column_quota=scheduler_config["single_column_quota"],
    )
    quadratic_material = {
        "g": quadratics.g,
        "b": quadratics.b,
        "accum_digits": fixed.accum_digits,
        "fixed": fixed.fingerprint_material(),
    }
    quadratic_key = short_key("quadratics", quadratic_material)
    quadratic_artifact = "quadratics:normal_equations"
    cached_quad = cache.get(quadratic_key)
    if cached_quad is None:
        cache.put(quadratic_key, {"g": quadratics.g, "b": quadratics.b})
    ledger.record(
        artifact_id=quadratic_artifact,
        stage="quadratic_aggregation",
        cache_key=quadratic_key,
        cache_hit=cached_quad is not None,
        basis="all_referenced_mac_terms_and_order_match",
        sample_ids=sample_ids,
        columns=columns,
        depends_on=[entry["cache_key"] for entry in ledger.entries if entry["stage"] == "accumulation_mac"],
        material=common_refs_material,
    )

    paths_report: dict[str, Any] = {}
    attribution_report: dict[str, Any] = {}
    trace: dict[str, Any] = {
        "accumulation_operations": quadratics.operations,
        "execution_schedule": quadratics.schedule.get("execution_batches"),
        "solver_traces": {},
        "order_audits": {},
    }

    for path_name in sorted(config["paths"]):
        path_config = config["paths"][path_name]
        penalty = path_config["penalty"]
        alphas = path_config["alphas"]
        solver_config = dict(path_config["solver"])
        solver_config["attribution_tolerance"] = config["attribution_tolerance"]
        solver_fingerprint = canonical_sha256(_solver_material(solver_config))
        level_results: list[dict[str, Any]] = []
        path_trace: list[dict[str, Any]] = []
        order_audits: dict[str, dict[str, Any]] = {}

        for alpha in alphas:
            alpha_key = format(alpha, "f")
            coefficient_material = {
                "kind": f"{penalty}_coefficient_level",
                "path_name": path_name,
                "quadratic_key": quadratic_key,
                "alpha": alpha_key,
                "penalty": penalty,
                "solver_fingerprint": solver_fingerprint,
                "fixed": fixed.fingerprint_material(),
                "sample_fingerprints": sample_fingerprints,
                "column_fingerprints": column_fingerprints,
            }
            coefficient_key = short_key("coefficient", coefficient_material)
            coefficient_artifact = f"coefficient:{path_name}:alpha={alpha_key}"
            cached_level = cache.get(coefficient_key)
            cache_hit = cached_level is not None
            if cache_hit:
                level = cached_level
                level["requested_alpha_decimal"] = alpha
            elif penalty == "ridge":
                level = solve_ridge_path_level(quadratics.g, quadratics.b, fixed, alpha, solver_config)
                cache.put(coefficient_key, to_jsonable({k: v for k, v in level.items() if k != "requested_alpha_decimal"}))
            else:
                level = solve_lasso_path_level(quadratics.g, quadratics.b, qd, alpha, solver_config)
                cache.put(coefficient_key, to_jsonable({k: v for k, v in level.items() if k != "requested_alpha_decimal"}))
            ledger.record(
                artifact_id=coefficient_artifact,
                stage="coefficient_level",
                cache_key=coefficient_key,
                cache_hit=cache_hit,
                basis="sample_set+column_set+alpha+solver_fingerprint_match",
                sample_ids=sample_ids,
                columns=columns,
                depends_on=[quadratic_key],
                material=common_refs_material | {"alpha": alpha_key, "solver": _solver_material(solver_config)},
            )
            level_results.append(level)
            path_trace.append(to_jsonable({k: v for k, v in level.items() if k != "requested_alpha_decimal"}))

            solver_stage = "integer_elimination" if penalty == "ridge" else "coordinate_update"
            solver_artifact = f"{solver_stage}:{path_name}:alpha={alpha_key}"
            solver_key = short_key(solver_stage, coefficient_material | {"trace": "full"})
            if cache.get(solver_key) is None:
                trace_payload = {
                    key: level.get(key)
                    for key in [
                        "attempts",
                        "pivot_trace",
                        "elimination_trace",
                        "back_substitution_trace",
                        "equation_residual_vector",
                        "sweeps",
                        "iterations",
                        "status",
                    ]
                    if key in level
                }
                cache.put(solver_key, to_jsonable(trace_payload))
            ledger.record(
                artifact_id=solver_artifact,
                stage=solver_stage,
                cache_key=solver_key,
                cache_hit=cache_hit,
                basis="same coefficient content key; full deterministic solver trace",
                sample_ids=sample_ids,
                columns=columns,
                depends_on=[coefficient_key],
                material=common_refs_material | {"alpha": alpha_key},
            )

            if penalty == "lasso":
                order_material = coefficient_material | {"audit": "ascending_vs_descending_first_sweep"}
                order_key = short_key("coord-order", order_material)
                cached_audit = cache.get(order_key)
                if cached_audit is not None:
                    audit = cached_audit
                else:
                    audit = lasso_order_audit(
                        quadratics.g,
                        quadratics.b,
                        qd,
                        alpha,
                        solver_config,
                        correlation["coordinate_order_pairs"],
                    )
                    cache.put(order_key, to_jsonable(audit))
                order_audits[alpha_key] = audit
                trace["order_audits"][f"{path_name}:{alpha_key}"] = to_jsonable(audit)
                ledger.record(
                    artifact_id=f"coordinate_order_audit:{path_name}:alpha={alpha_key}",
                    stage="coordinate_order_audit",
                    cache_key=order_key,
                    cache_hit=cached_audit is not None,
                    basis="strong_pair_set+ascending_index_rule_match",
                    sample_ids=sample_ids,
                    columns=columns,
                    depends_on=[coefficient_key],
                    material=common_refs_material | {"alpha": alpha_key},
                )

        path_material = {
            "kind": "path_summary",
            "path_name": path_name,
            "penalty": penalty,
            "alpha_keys": [format(alpha, "f") for alpha in alphas],
            "coefficient_keys": [
                entry["cache_key"]
                for entry in ledger.entries
                if entry["stage"] == "coefficient_level" and entry["artifact_id"].startswith(f"coefficient:{path_name}:")
            ],
            "zero_policy": solver_config["zero_half_step_policy"],
            "fixed": fixed.fingerprint_material(),
            "dataset_fingerprint": dataset.fingerprint,
        }
        path_key = short_key("path-summary", path_material)
        path_artifact = f"path_summary:{path_name}"
        cached_path = cache.get(path_key)
        if cached_path is not None:
            path_report = cached_path
        else:
            path_report = summarize_path(
                dataset=dataset,
                qd=qd,
                params=fixed,
                penalty=penalty,
                level_results=level_results,
                solver_config=solver_config,
            )
            cache.put(path_key, to_jsonable(path_report))
        paths_report[path_name] = path_report
        ledger.record(
            artifact_id=path_artifact,
            stage="path_summary",
            cache_key=path_key,
            cache_hit=cached_path is not None,
            basis="ordered_alpha_key_set+coefficient_keys+zero_policy_match",
            sample_ids=sample_ids,
            columns=columns,
            depends_on=path_material["coefficient_keys"],
            material=common_refs_material | {"alpha_keys": path_material["alpha_keys"]},
        )

        attribution_material = {
            "kind": "staged_attribution",
            "path_key": path_key,
            "quadratic_key": quadratic_key,
            "tolerance": str(config["attribution_tolerance"]),
            "dataset_fingerprint": dataset.fingerprint,
            "fixed": fixed.fingerprint_material(),
        }
        attribution_key = short_key("attribution", attribution_material)
        attribution_artifact = f"error_attribution:{path_name}"
        cached_attribution = cache.get(attribution_key)
        if cached_attribution is not None:
            attribution = cached_attribution
        else:
            attribution = build_attribution(
                dataset=dataset,
                qd=qd,
                quadratics=quadratics,
                penalty=penalty,
                alphas=alphas,
                solver_config=solver_config,
                path_report=path_report,
                order_audits=order_audits if penalty == "lasso" else None,
            )
            cache.put(attribution_key, to_jsonable(attribution))
        attribution_report[path_name] = attribution
        ledger.record(
            artifact_id=attribution_artifact,
            stage="error_attribution",
            cache_key=attribution_key,
            cache_hit=cached_attribution is not None,
            basis="float64_reference+staged_chain+tolerance_match",
            sample_ids=sample_ids,
            columns=columns,
            depends_on=[path_key, quadratic_key],
            material=common_refs_material,
        )
        trace["solver_traces"][path_name] = path_trace

    current_artifacts = set(ledger._artifacts)
    removed_events = ledger.invalidate_removed(current_artifacts)
    lineage_checklist = [
        {
            "artifact_id": entry["artifact_id"],
            "stage": entry["stage"],
            "cache_key": entry["cache_key"],
            "sample_refs": entry["sample_refs"],
            "column_refs": entry["column_refs"],
            "depends_on": entry["depends_on"],
        }
        for entry in ledger.lineage_manifest()
    ]

    report = {
        "status": "E_INPUT_INVALID" if (input_errors or command_errors) else "OK",
        "schema_version": config["schema_version"],
        "dataset": {
            "fingerprint": dataset.fingerprint,
            "sample_count": len(sample_ids),
            "sample_ids": sample_ids,
            "columns": columns,
            "column_fingerprints": column_fingerprints,
        },
        "config_fingerprint": config["config_fingerprint"],
        "fixed_point": fixed.fingerprint_material(),
        "quantization": {
            "caliber": "fine fixed scale then coarse grid snap; ROUND_HALF_UP ties away from zero",
            "columns": quantization_columns,
        },
        "correlation": {key: value for key, value in correlation.items() if key != "pairs"} | {"pairs": correlation["pairs"]},
        "accumulation": {
            "caliber": "sample id ascending inside j/i stable order; Python arbitrary-precision integers",
            "configured_caliber": config.get("accumulation"),
            "accum_digits": fixed.accum_digits,
            "accum_scale": str(fixed.accum_scale),
            "accum_divisor": fixed.accum_divisor,
            "task_order": quadratics.schedule["task_order"],
            "budget_plan": quadratics.schedule["budget_plan"],
            "budget": quadratics.schedule["budget"],
        },
        "paths": paths_report,
        "attribution": attribution_report,
        "lineage_checklist": lineage_checklist,
        "input_errors": input_errors or [],
        "command_results": command_results or [],
        "command_errors": command_errors or [],
    }
    ledger_output = {
        "entries": ledger.lineage_manifest(),
        "events": ledger.ordered_events(),
        "removed_or_invalidated": removed_events,
        "summary": ledger.summary(),
    }
    return {"report": to_jsonable(report), "trace": to_jsonable(trace), "ledger": to_jsonable(ledger_output)}
