from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .canonical import content_digest, deep_field_diff, normalize
from .ledger import Ledger, compute_snapshot_content_digest
from .pipeline import build_estimates
from .spec import Spec, load_spec, parse_spec


VERIFY_MODEL_VERSION = "qplan-independent-verify-v1"


def _mismatch(query_id: str | None, field: str, ledger_value: Any, recomputed_value: Any) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "field": field,
        "ledger_value": ledger_value,
        "recomputed_value": recomputed_value,
    }


def _flatten_queries(view: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    queries: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    for batch in sorted(view.get("batches", []), key=lambda item: item["batch_no"]):
        if batch["status"] != "committed":
            issues.append(
                {
                    "code": "E_LEDGER",
                    "message": "non-committed batch appears in the current view",
                    "batch_no": batch["batch_no"],
                    "status": batch["status"],
                }
            )
            continue
        for query in batch["payload"].get("queries", []):
            query_id = query.get("query_id")
            if query_id in queries:
                issues.append(
                    {
                        "code": "E_LEDGER",
                        "message": "query appears in more than one committed batch",
                        "query_id": query_id,
                    }
                )
            queries[query_id] = query
    return queries, issues


def verify_ledger(
    *,
    spec_path: str | Path,
    ledger_path: str | Path,
    expected_snapshot_version: int | None = None,
    expected_snapshot_digest: str | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    spec: Spec = load_spec(spec_path)
    ledger = Ledger.read_only(ledger_path)
    try:
        view = ledger.load_current_view()
        snapshot = view.get("snapshot")
        if snapshot is None:
            return {
                "code": "E_VERIFY",
                "status": "NO_SNAPSHOT",
                "first_mismatch": _mismatch(None, "snapshot", None, "missing"),
            }
        if expected_snapshot_version is not None and int(expected_snapshot_version) != int(snapshot["snapshot_version"]):
            return {
                "code": "E_SNAPSHOT",
                "status": "SNAPSHOT_MISMATCH",
                "expected_snapshot_version": int(expected_snapshot_version),
                "actual_snapshot_version": int(snapshot["snapshot_version"]),
                "message": "requested snapshot version does not match the current ledger snapshot",
            }
        if expected_snapshot_digest is not None and expected_snapshot_digest != snapshot["content_digest"]:
            return {
                "code": "E_SNAPSHOT",
                "status": "SNAPSHOT_MISMATCH",
                "expected_snapshot_digest": expected_snapshot_digest,
                "actual_snapshot_digest": snapshot["content_digest"],
                "message": "requested snapshot digest does not match the current ledger snapshot",
            }
        if snapshot["status"] != "current":
            return {
                "code": "E_SNAPSHOT",
                "status": "STALE_SNAPSHOT",
                "first_mismatch": _mismatch(None, "snapshot.status", snapshot["status"], "current"),
            }

        storage_errors = ledger.verify_storage_digests(view)
        if storage_errors:
            return {
                "code": storage_errors[0]["code"],
                "status": "STORAGE_DIGEST_MISMATCH",
                "errors": storage_errors,
                "first_mismatch": _mismatch(None, "ledger_storage_digest", storage_errors[0].get("stored"), storage_errors[0].get("recomputed")),
            }
        disk_spec = normalize(spec.to_dict())
        structural_keys = ("spec_version", "name", "seed", "tables", "queries")
        structural_disk = {key: disk_spec.get(key) for key in structural_keys}
        structural_ledger = {key: snapshot["spec"].get(key) for key in structural_keys}
        if structural_disk != structural_ledger:
            diff = deep_field_diff(structural_ledger, structural_disk)
            return {
                "code": "E_SNAPSHOT",
                "status": "SPEC_MISMATCH",
                "first_mismatch": _mismatch(
                    None,
                    "snapshot.spec." + (diff[0] if diff else "$"),
                    diff[1] if diff else None,
                    diff[2] if diff else None,
                ),
                "message": "declaration data contract differs from the committed snapshot",
            }
        session_keys = ("stats", "cost_weights", "search", "execution", "bias")
        declaration_session_fields = {
            key: disk_spec.get(key) for key in session_keys if disk_spec.get(key) != snapshot["spec"].get(key)
        }
        effective_dict = copy.deepcopy(disk_spec)
        for key in ("stats", "search", "execution", "bias"):
            if key in snapshot["spec"]:
                effective_dict[key] = snapshot["spec"][key]
        effective_dict["cost_weights"] = snapshot["weights"]
        effective_spec = parse_spec(effective_dict)
        recomputed_snapshot_digest = compute_snapshot_content_digest(
            effective_spec.to_dict(), snapshot["stats"], snapshot["weights"]
        )
        if recomputed_snapshot_digest != snapshot["content_digest"]:
            return {
                "code": "E_SNAPSHOT",
                "status": "SNAPSHOT_DIGEST_MISMATCH",
                "first_mismatch": _mismatch(
                    None, "snapshot.content_digest", snapshot["content_digest"], recomputed_snapshot_digest
                ),
            }

        unknown_batches = view.get("unknown_batches") or []
        running_session = view.get("session") and view["session"]["status"] == "running"
        partial = bool(unknown_batches) or running_session or view.get("code") == "E_RECOVERY"
        if partial and not allow_partial:
            return {
                "code": "E_RECOVERY",
                "status": "UNKNOWN_BATCHES",
                "snapshot_version": snapshot["snapshot_version"],
                "session": view.get("session"),
                "unknown_batches": unknown_batches,
                "message": "run recovery first or pass --allow-partial to verify only committed batches",
            }

        recomputed = build_estimates(effective_spec, snapshot["stats"])
        ledger_queries, batch_issues = _flatten_queries(view)
        if batch_issues:
            return {
                "code": "E_LEDGER",
                "status": "BATCH_ISSUE",
                "errors": batch_issues,
                "first_mismatch": _mismatch(None, "batches", None, batch_issues[0]),
            }

        all_batches = sorted(view.get("batches", []), key=lambda item: item["batch_no"])
        ledger_search = all_batches[0]["payload"].get("index_search") if all_batches else None
        recomputed_search = recomputed["index_search"]
        if ledger_search is None:
            return {
                "code": "E_VERIFY",
                "status": "MISMATCH",
                "first_mismatch": _mismatch(None, "index_search", None, recomputed_search["selected_indexes"]),
            }
        for field_name in ("selected_indexes", "truth_source", "primary_mode"):
            if ledger_search.get(field_name) != recomputed_search.get(field_name):
                return {
                    "code": "E_VERIFY",
                    "status": "MISMATCH",
                    "first_mismatch": _mismatch(
                        None,
                        f"index_search.{field_name}",
                        ledger_search.get(field_name),
                        recomputed_search.get(field_name),
                    ),
                }

        checked_paths = 0
        checked_queries = 0
        skipped_queries: list[str] = []
        for query in effective_spec.queries:
            stored = ledger_queries.get(query.id)
            if stored is None:
                if partial and allow_partial:
                    skipped_queries.append(query.id)
                    continue
                return {
                    "code": "E_RECOVERY",
                    "status": "MISSING_QUERY",
                    "first_mismatch": _mismatch(query.id, "batch_commit", None, "missing"),
                }
            checked_queries += 1
            stored_estimates = stored.get("path_estimates", {})
            recomputed_estimates = recomputed["estimates_by_query"][query.id]
            expected_path_ids = sorted(recomputed_estimates)
            actual_path_ids = sorted(stored_estimates)
            if expected_path_ids != actual_path_ids:
                return {
                    "code": "E_VERIFY",
                    "status": "MISMATCH",
                    "first_mismatch": _mismatch(query.id, "path_estimates.keys", actual_path_ids, expected_path_ids),
                }
            for path_id in expected_path_ids:
                checked_paths += 1
                mismatch = deep_field_diff(
                    stored_estimates[path_id], recomputed_estimates[path_id], prefix=""
                )
                if mismatch is not None:
                    return {
                        "code": "E_VERIFY",
                        "status": "MISMATCH",
                        "snapshot_version": snapshot["snapshot_version"],
                        "first_mismatch": _mismatch(
                            query.id,
                            f"path_estimates.{path_id}.{mismatch[0]}",
                            mismatch[1],
                            mismatch[2],
                        ),
                    }
            recomputed_selected = recomputed_search["query_choices"][query.id]["selected_path_id"]
            if stored.get("selected_path_id") != recomputed_selected:
                return {
                    "code": "E_VERIFY",
                    "status": "MISMATCH",
                    "first_mismatch": _mismatch(
                        query.id, "selected_path_id", stored.get("selected_path_id"), recomputed_selected
                    ),
                }
            selected_mismatch = deep_field_diff(
                stored.get("selected_estimate"), recomputed_estimates[recomputed_selected]
            )
            if selected_mismatch is not None:
                return {
                    "code": "E_VERIFY",
                    "status": "MISMATCH",
                    "first_mismatch": _mismatch(
                        query.id,
                        f"selected_estimate.{selected_mismatch[0]}",
                        selected_mismatch[1],
                        selected_mismatch[2],
                    ),
                }

        execution_issues = []
        for query_id, stored in ledger_queries.items():
            executions = stored.get("executions", {})
            baseline_digest = stored.get("baseline_digest")
            if baseline_digest is None:
                execution_issues.append({"query_id": query_id, "issue": "missing baseline digest"})
                continue
            for path_id, execution in executions.items():
                if execution.get("digest") != baseline_digest:
                    execution_issues.append(
                        {
                            "query_id": query_id,
                            "path_id": path_id,
                            "issue": "result digest differs from baseline",
                        }
                    )
                if execution.get("sql_digest") != next(
                    (
                        path["sql_digest"]
                        for path in stored.get("path_specs", [])
                        if path.get("id") == path_id
                    ),
                    None,
                ):
                    execution_issues.append(
                        {
                            "query_id": query_id,
                            "path_id": path_id,
                            "issue": "execution SQL digest differs from stored path specification",
                        }
                    )
        return {
            "code": "OK" if not execution_issues else "E_RESULT",
            "status": "VERIFIED" if not execution_issues else "RESULT_INTEGRITY_ISSUE",
            "verification_model_version": VERIFY_MODEL_VERSION,
            "snapshot_version": snapshot["snapshot_version"],
            "plan_version": snapshot["plan_version"],
            "snapshot_digest": snapshot["content_digest"],
            "session_id": view["session"]["session_id"] if view.get("session") else None,
            "queries_checked": checked_queries,
            "paths_checked": checked_paths,
            "partial": partial,
            "skipped_queries": skipped_queries,
            "unknown_batches": unknown_batches,
            "first_mismatch": None,
            "execution_integrity_issues": execution_issues,
            "scope": "recomputed estimates from declaration plus archived statistics only; generated SQLite data was not read",
            "session_config_source": "ledger snapshot for stats/search/weights/execution/bias; declaration for tables and queries",
            "declaration_session_field_differences": declaration_session_fields,
        }
    finally:
        ledger.close()
