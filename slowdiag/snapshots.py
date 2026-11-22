from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json, sha256_canonical
from .canonical import true_int
from .config import WorkbenchConfig, load_config_data
from .constants import METRICS
from .errors import E_REPLAY, SlowDiagError
from .pipeline import recompute_view_from_config
from .storage import Ledger


SNAPSHOT_FORMAT = "slowdiag.snapshot"
SNAPSHOT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SnapshotData:
    path: Path
    payload: dict[str, Any]
    config: WorkbenchConfig
    digest_valid: bool
    diffs: tuple[dict[str, Any], ...]


def _snapshot_body(view: dict[str, Any], config: WorkbenchConfig) -> dict[str, Any]:
    return {
        "format": SNAPSHOT_FORMAT,
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "plan_version": view["plan_version"],
        "config": config.to_dict(),
        "config_sha256": config.fingerprint_sha256(),
        "source_summaries": view["source_summaries"],
        "merged": view["queries"],
        "migration": view["migration"],
        "attribution": view["attribution"],
        "invalidation": view["invalidation"],
    }


def create_snapshot(
    workspace: Path,
    view: dict[str, Any],
    config: WorkbenchConfig,
    name: str | None = None,
) -> Path:
    workspace = workspace.resolve(strict=False)
    directory = workspace / "snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    body = _snapshot_body(view, config)
    payload = dict(body)
    payload["content_sha256"] = sha256_canonical(body)
    if name is None:
        name = (
            f"session-p{view['plan_version']:04d}-"
            f"{payload['content_sha256'][:12]}.json"
        )
    safe_name = Path(name).name
    if not safe_name.endswith(".json"):
        safe_name += ".json"
    path = directory / safe_name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _source_summary_diffs(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    expected_map = {item["source_id"]: item for item in expected}
    actual_map = {item["source_id"]: item for item in actual}
    diffs: list[dict[str, Any]] = []
    for source_id in sorted(set(expected_map) | set(actual_map)):
        left = expected_map.get(source_id)
        right = actual_map.get(source_id)
        if left is None:
            diffs.append(
                {
                    "path": f"source_summaries.{source_id}",
                    "expected": None,
                    "actual": right,
                }
            )
            continue
        if right is None:
            diffs.append(
                {
                    "path": f"source_summaries.{source_id}",
                    "expected": left,
                    "actual": None,
                }
            )
            continue
        for field in (
            "content_sha256",
            "format_versions",
            "format_version",
            "sample_rows",
            "sample_period_ms",
            "elapsed_semantics",
            "scan_semantics",
            "config_fingerprint_sha256",
        ):
            if left.get(field) != right.get(field):
                diffs.append(
                    {
                        "path": f"source_summaries.{source_id}.{field}",
                        "expected": left.get(field),
                        "actual": right.get(field),
                    }
                )
    return diffs


def load_snapshot(
    path: Path | str,
    workspace: Path | str,
) -> SnapshotData:
    snapshot_path = Path(path)
    try:
        text = snapshot_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SlowDiagError(
            E_REPLAY,
            f"cannot read snapshot: {exc}",
            path="$",
            details={"file": str(snapshot_path)},
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SlowDiagError(
            E_REPLAY,
            f"snapshot JSON is invalid at line {exc.lineno}, column {exc.colno}",
            path="$",
            field="json",
        ) from exc
    if not isinstance(payload, dict):
        raise SlowDiagError(
            E_REPLAY, "snapshot root must be an object", path="$", field="root"
        )
    diffs: list[dict[str, Any]] = []
    if payload.get("format") != SNAPSHOT_FORMAT:
        diffs.append(
            {
                "path": "format",
                "expected": SNAPSHOT_FORMAT,
                "actual": payload.get("format"),
            }
        )
    version = payload.get("format_version")
    if not true_int(version) or version != SNAPSHOT_FORMAT_VERSION:
        diffs.append(
            {
                "path": "format_version",
                "expected": SNAPSHOT_FORMAT_VERSION,
                "actual": version,
            }
        )
    plan_version = payload.get("plan_version")
    if not true_int(plan_version) or plan_version <= 0:
        raise SlowDiagError(
            E_REPLAY,
            "snapshot plan_version must be a positive integer",
            path="plan_version",
            field="plan_version",
            details={"actual": plan_version, "diffs": diffs},
        )
    recorded_sha = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    actual_sha = sha256_canonical(body)
    digest_valid = recorded_sha == actual_sha
    if not digest_valid:
        diffs.append(
            {
                "path": "content_sha256",
                "expected": recorded_sha,
                "actual": actual_sha,
            }
        )
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise SlowDiagError(
            E_REPLAY,
            "snapshot does not contain a source declaration",
            path="config",
            field="config",
            details={"diffs": diffs},
        )
    try:
        config = load_config_data(raw_config, workspace)
    except SlowDiagError as exc:
        raise SlowDiagError(
            E_REPLAY,
            f"snapshot source declaration is invalid: {exc.message}",
            path=f"config.{exc.path}" if exc.path else "config",
            field=exc.path,
            details={"source_error": exc.to_dict(), "diffs": diffs},
        ) from exc
    diffs.sort(key=lambda item: item["path"])
    return SnapshotData(
        path=snapshot_path,
        payload=payload,
        config=config,
        digest_valid=digest_valid,
        diffs=tuple(diffs),
    )


def _first_field_difference(
    expected: dict[str, Any], actual: dict[str, Any]
) -> tuple[str | None, str | None, str, Any, Any]:
    expected_queries = {
        query["query_id"]: query for query in expected.get("merged", [])
    }
    actual_queries = {
        query["query_id"]: query for query in actual.get("queries", [])
    }
    for query_id in sorted(set(expected_queries) | set(actual_queries)):
        left = expected_queries.get(query_id)
        right = actual_queries.get(query_id)
        if left is None:
            return query_id, None, "query", None, right
        if right is None:
            return query_id, None, "query", left, None
        for metric in METRICS:
            left_metric = left.get("metrics", {}).get(metric, {})
            right_metric = right.get("metrics", {}).get(metric, {})
            for field in (
                "adopted.source_id",
                "adopted.value",
                "adopted.format_version",
                "comparisons",
                "basis",
            ):
                left_value: Any
                right_value: Any
                if field.startswith("adopted."):
                    key = field.split(".", 1)[1]
                    left_value = left_metric.get("adopted", {}).get(key)
                    right_value = right_metric.get("adopted", {}).get(key)
                else:
                    left_value = left_metric.get(field)
                    right_value = right_metric.get(field)
                if canonical_json(left_value) != canonical_json(right_value):
                    return query_id, metric, field, left_value, right_value
        for field in ("slow", "rule", "metric_slow", "thresholds", "basis"):
            left_value = left.get("diagnosis", {}).get(field)
            right_value = right.get("diagnosis", {}).get(field)
            if canonical_json(left_value) != canonical_json(right_value):
                return query_id, None, f"diagnosis.{field}", left_value, right_value

    expected_pairs = {
        pair["pair_id"]: pair for pair in expected.get("migration", {}).get("pairs", [])
    }
    actual_pairs = {
        pair["pair_id"]: pair for pair in actual.get("migration", {}).get("pairs", [])
    }
    for pair_id in sorted(set(expected_pairs) | set(actual_pairs)):
        left_pair = expected_pairs.get(pair_id)
        right_pair = actual_pairs.get(pair_id)
        if left_pair is None or right_pair is None:
            return None, None, f"migration.pairs.{pair_id}", left_pair, right_pair
        left_results = {
            item["query_id"]: item for item in left_pair.get("results", [])
        }
        right_results = {
            item["query_id"]: item for item in right_pair.get("results", [])
        }
        for query_id in sorted(set(left_results) | set(right_results)):
            left_result = left_results.get(query_id)
            right_result = right_results.get(query_id)
            if left_result is None or right_result is None:
                return query_id, None, f"migration.{pair_id}.result", left_result, right_result
            for field in ("status", "consistent", "conflicts", "thresholds"):
                left_value = left_result.get(field)
                right_value = right_result.get(field)
                if canonical_json(left_value) != canonical_json(right_value):
                    conflict_metric = None
                    if field == "conflicts":
                        left_metrics = [item.get("metric") for item in left_value or []]
                        right_metrics = [item.get("metric") for item in right_value or []]
                        conflict_metric = next(
                            (
                                metric
                                for metric in METRICS
                                if metric in left_metrics or metric in right_metrics
                            ),
                            None,
                        )
                    return query_id, conflict_metric, f"migration.{field}", left_value, right_value
            for side in ("source_a", "source_b"):
                for metric in METRICS:
                    left_value = left_result.get(side, {}).get("metrics", {}).get(metric, {}).get("value")
                    right_value = right_result.get(side, {}).get("metrics", {}).get(metric, {}).get("value")
                    if left_value != right_value:
                        return (
                            query_id,
                            metric,
                            f"migration.{side}.metrics.{metric}.value",
                            left_value,
                            right_value,
                        )
            for field in (
                "migrated_claim",
                "rejudged_on_b",
                "metric_comparisons",
                "insufficient_metrics",
                "reasons",
            ):
                left_value = left_result.get(field)
                right_value = right_result.get(field)
                if canonical_json(left_value) != canonical_json(right_value):
                    return query_id, None, f"migration.{field}", left_value, right_value
            if canonical_json(left_result) != canonical_json(right_result):
                return query_id, None, "migration.result", left_result, right_result

    expected_attr = {
        (item["pair_id"], item["query_id"]): item
        for item in expected.get("attribution", {}).get("results", [])
    }
    actual_attr = {
        (item["pair_id"], item["query_id"]): item
        for item in actual.get("attribution", {}).get("results", [])
    }
    for pair_id, query_id in sorted(set(expected_attr) | set(actual_attr)):
        left = expected_attr.get((pair_id, query_id))
        right = actual_attr.get((pair_id, query_id))
        if canonical_json(left) != canonical_json(right):
            return query_id, None, f"attribution.{pair_id}", left, right
    return None, None, "conclusions", expected, actual


def _replacement_view(
    recomputed_view: dict[str, Any],
    snapshot_diffs: list[dict[str, Any]],
) -> dict[str, Any]:
    invalidation = dict(recomputed_view["invalidation"])
    invalidation.update(
        {
            "reasons": ["snapshot_restore"],
            "replacement_mode": "whole_view",
            "snapshot_diff": snapshot_diffs,
            "snapshot_diff_count": len(snapshot_diffs),
        }
    )
    view = dict(recomputed_view)
    view["invalidation"] = invalidation
    return view


def _recorded_view(snapshot_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "slowdiag.view/1",
        "plan_version": int(snapshot_payload["plan_version"]),
        "config_sha256": snapshot_payload.get("config_sha256"),
        "source_summaries": snapshot_payload.get("source_summaries", []),
        "queries": snapshot_payload.get("merged", []),
        "migration": snapshot_payload.get("migration", {}),
        "attribution": snapshot_payload.get("attribution", {}),
        "invalidation": snapshot_payload.get("invalidation", {}),
    }


def restore_snapshot(
    ledger: Ledger,
    snapshot_path: Path | str,
    workspace: Path | str,
) -> dict[str, Any]:
    snapshot = load_snapshot(snapshot_path, workspace)
    plan_version = int(snapshot.payload["plan_version"])
    try:
        recomputed_view, datasets, timings = recompute_view_from_config(
            snapshot.config, plan_version
        )
    except SlowDiagError as exc:
        recorded = _recorded_view(snapshot.payload)
        ledger.replace_view(
            recorded,
            [],
            plan_version,
            run_metrics={"restore": 1, "replay_read_error": 1},
        )
        raise SlowDiagError(
            E_REPLAY,
            "snapshot replay could not read the recorded sources",
            query_id=exc.query_id,
            metric=exc.metric,
            field=exc.path or exc.field,
            details={
                "inner_code": exc.code,
                "inner_error": exc.to_dict(),
                "snapshot": str(snapshot.path),
                "snapshot_diffs": list(snapshot.diffs),
                "replacement_mode": "whole_view",
                "replaced": True,
            },
        ) from exc
    source_diffs = _source_summary_diffs(
        list(snapshot.payload.get("source_summaries", [])),
        list(recomputed_view["source_summaries"]),
    )
    diffs = list(snapshot.diffs)
    for item in source_diffs:
        if not any(existing["path"] == item["path"] for existing in diffs):
            diffs.append(item)
    diffs.sort(key=lambda item: item["path"])
    expected_core = {
        "merged": snapshot.payload.get("merged", []),
        "migration": snapshot.payload.get("migration", {}),
        "attribution": snapshot.payload.get("attribution", {}),
    }
    query_id, metric, field, expected_value, actual_value = _first_field_difference(
        expected_core, recomputed_view
    )
    if not diffs and field == "conclusions":
        replacement = _recorded_view(snapshot.payload)
    else:
        replacement = _replacement_view(recomputed_view, diffs)
    view_sha = ledger.replace_view(
        replacement,
        datasets,
        plan_version,
        run_metrics={**timings, "restore": 1, "clock": "perf_counter_ns"},
    )
    result = {
        "status": "restored",
        "snapshot": str(snapshot.path),
        "plan_version": snapshot.payload.get("plan_version"),
        "snapshot_diff_count": len(diffs),
        "snapshot_diffs": diffs,
        "view_sha256": view_sha,
    }
    if field != "conclusions":
        raise SlowDiagError(
            E_REPLAY,
            "snapshot replay differs from the recorded conclusions",
            query_id=query_id,
            metric=metric,
            field=field,
            details={
                **result,
                "expected": expected_value,
                "actual": actual_value,
                "replacement_mode": "whole_view",
            },
        )
    return result
