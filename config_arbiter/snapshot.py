from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

from .engine import ArbiterEngine
from .errors import ReplayError, SnapshotError, SpecError
from .ledger import Ledger
from .spec import Specs, load_specs
from .util import atomic_write_json, canonical_sha256, read_json

SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOT_KIND = "config-arbiter-session-snapshot"


def _state_digest(environment_document: dict[str, Any], view_document: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "environment_document": environment_document,
            "view_document": view_document,
        }
    )


def _view_summary_record(view: dict[str, Any]) -> dict[str, Any]:
    result = view["result"]
    return {
        "view_id": view["view_id"],
        "view_version": int(view["current_version"]),
        "result_sha256": view["result_sha256"],
        "authoritative_by_key": result["summary"]["authoritative_by_key"],
        "accepted_by_key": {
            key: [
                {
                    "environment_id": item["environment_id"],
                    "source_id": item["source_id"],
                }
                for item in key_result["accepted"]
            ]
            for key, key_result in sorted(result["keys"].items())
        },
        "per_key_decision_error_bounds": {
            key: key_result["error_model"]["comparison_error_bound_fixed_units"]
            for key, key_result in sorted(result["keys"].items())
        },
        "per_key_aggregation_error_upper_bounds": {
            key: key_result["key_aggregation_error_upper_fixed_units"]
            for key, key_result in sorted(result["keys"].items())
        },
        "view_aggregation_error_upper_bound_fixed_units": result[
            "view_aggregation_error_upper_bound_fixed_units"
        ],
    }


def make_snapshot(specs: Specs, ledger: Ledger) -> dict[str, Any]:
    current_views = [
        view for view in ledger.all_current_views() if view["view_id"] in specs.views
    ]
    completed = [
        {
            "view_id": view["view_id"],
            "view_version": int(view["current_version"]),
            "result_sha256": view["result_sha256"],
        }
        for view in sorted(current_views, key=lambda item: item["view_id"])
    ]
    summaries = {
        view["view_id"]: _view_summary_record(view)
        for view in sorted(current_views, key=lambda item: item["view_id"])
    }
    environment_document = copy.deepcopy(specs.environment_document)
    view_document = copy.deepcopy(specs.view_document)
    payload: dict[str, Any] = {
        "kind": SNAPSHOT_KIND,
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "environment_set_version": specs.environment_set_version,
        "source_priority_version": specs.source_priority_version,
        "scale_policy_version": specs.session.scale_policy_version,
        "arbitration_policy_version": specs.session.arbitration_policy_version,
        "state_digest": _state_digest(environment_document, view_document),
        "environment_document": environment_document,
        "view_document": view_document,
        "completed_views": completed,
        "completed_view_ids": [item["view_id"] for item in completed],
        "view_summaries": summaries,
        "ledger_content_digest": ledger.meta()["content_digest"],
    }
    payload["content_digest"] = canonical_sha256(payload)
    return payload


def write_snapshot(path: str | Path, specs: Specs, ledger: Ledger) -> dict[str, Any]:
    payload = make_snapshot(specs, ledger)
    atomic_write_json(path, payload)
    return payload


def load_snapshot(path: str | Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise SnapshotError(
            "cannot load snapshot",
            details={"path": str(path), "reason": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise SnapshotError("snapshot root must be an object", details={"path": str(path)})
    if payload.get("kind") != SNAPSHOT_KIND:
        raise SnapshotError(
            "snapshot kind mismatch",
            details={"path": str(path), "expected": SNAPSHOT_KIND, "actual": payload.get("kind")},
        )
    if payload.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        raise SnapshotError(
            "unsupported snapshot format version",
            details={
                "path": str(path),
                "expected_format_version": SNAPSHOT_FORMAT_VERSION,
                "actual_format_version": payload.get("format_version"),
            },
        )
    supplied_digest = payload.get("content_digest")
    without_digest = {key: value for key, value in payload.items() if key != "content_digest"}
    actual_digest = canonical_sha256(without_digest)
    if supplied_digest != actual_digest:
        raise SnapshotError(
            "snapshot content digest mismatch",
            details={
                "path": str(path),
                "expected": supplied_digest,
                "actual": actual_digest,
            },
        )
    environment_document = payload.get("environment_document")
    view_document = payload.get("view_document")
    if not isinstance(environment_document, dict) or not isinstance(view_document, dict):
        raise SnapshotError(
            "snapshot must embed environment and view documents",
            details={"path": str(path)},
        )
    expected_state_digest = _state_digest(environment_document, view_document)
    if payload.get("state_digest") != expected_state_digest:
        raise SnapshotError(
            "snapshot state digest mismatch",
            details={
                "path": str(path),
                "expected": payload.get("state_digest"),
                "actual": expected_state_digest,
            },
        )
    if not isinstance(payload.get("environment_set_version"), int) or payload["environment_set_version"] < 0:
        raise SnapshotError("snapshot environment_set_version must be a non-negative integer", details={"path": str(path)})
    if not isinstance(payload.get("source_priority_version"), int) or payload["source_priority_version"] < 0:
        raise SnapshotError("snapshot source_priority_version must be a non-negative integer", details={"path": str(path)})
    for field in ("scale_policy_version", "arbitration_policy_version"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise SnapshotError(f"snapshot {field} must be a non-empty string", details={"path": str(path)})
    completed = payload.get("completed_views")
    completed_ids = payload.get("completed_view_ids")
    summaries = payload.get("view_summaries")
    if not isinstance(completed, list) or not isinstance(completed_ids, list) or not isinstance(summaries, dict):
        raise SnapshotError("snapshot completed view metadata is malformed", details={"path": str(path)})
    seen_ids: set[str] = set()
    for item in completed:
        if not isinstance(item, dict):
            raise SnapshotError("snapshot completed view entry must be an object", details={"path": str(path)})
        view_id = item.get("view_id")
        if not isinstance(view_id, str) or not view_id or view_id in seen_ids:
            raise SnapshotError("snapshot completed view_id is missing or duplicated", details={"path": str(path)})
        if not isinstance(item.get("view_version"), int) or item["view_version"] < 1:
            raise SnapshotError("snapshot completed view_version must be a positive integer", details={"path": str(path), "view_id": view_id})
        if not isinstance(item.get("result_sha256"), str) or len(item["result_sha256"]) != 64:
            raise SnapshotError("snapshot completed result_sha256 is malformed", details={"path": str(path), "view_id": view_id})
        seen_ids.add(view_id)
    if completed_ids != [item["view_id"] for item in completed] or set(summaries) != seen_ids:
        raise SnapshotError(
            "snapshot completed view summaries are inconsistent",
            details={"path": str(path), "completed_view_ids": completed_ids, "summary_ids": sorted(summaries)},
        )
    return payload


def _first_replay_difference(expected: dict[str, Any], actual: dict[str, Any]) -> tuple[str, Any, Any] | None:
    expected_keys = expected.get("keys", {})
    actual_keys = actual.get("keys", {})
    if set(expected_keys) != set(actual_keys):
        return "keys", sorted(expected_keys), sorted(actual_keys)
    checks = (
        "authoritative.value_fixed",
        "authoritative.value_text",
        "authoritative.environment_id",
        "authoritative.source_id",
    )
    for key in sorted(expected_keys):
        expected_key = expected_keys[key]
        actual_key = actual_keys[key]
        for field in checks:
            if field == "keys":
                continue
            left: Any = expected_key
            right: Any = actual_key
            for part in field.split("."):
                left = left.get(part) if isinstance(left, dict) else None
                right = right.get(part) if isinstance(right, dict) else None
            if left != right:
                return f"{key}:{field}", left, right
        expected_accepted = [
            (item.get("environment_id"), item.get("source_id"))
            for item in expected_key.get("accepted", [])
        ]
        actual_accepted = [
            (item.get("environment_id"), item.get("source_id"))
            for item in actual_key.get("accepted", [])
        ]
        if expected_accepted != actual_accepted:
            return f"{key}:accepted", expected_accepted, actual_accepted
        expected_bound = expected_key.get("error_model", {}).get("comparison_error_bound_fixed_units")
        actual_bound = actual_key.get("error_model", {}).get("comparison_error_bound_fixed_units")
        if expected_bound != actual_bound:
            return f"{key}:comparison_error_bound_fixed_units", expected_bound, actual_bound
        expected_aggregation = expected_key.get("key_aggregation_error_upper_fixed_units")
        actual_aggregation = actual_key.get("key_aggregation_error_upper_fixed_units")
        if expected_aggregation != actual_aggregation:
            return f"{key}:key_aggregation_error_upper_fixed_units", expected_aggregation, actual_aggregation
    return None


def replay_ledger(engine: ArbiterEngine) -> dict[str, Any]:
    """Recompute committed views and locate the first replay mismatch."""
    compared: list[dict[str, Any]] = []
    for view_id in sorted(engine.specs.views):
        current = engine.ledger.current_view(view_id)
        if current is None:
            continue
        expected_fingerprint = engine._view_fingerprint(engine.specs.views[view_id])
        if current["fingerprint"] != expected_fingerprint:
            continue
        actual = engine.recompute_view(view_id, int(current["current_version"]))
        difference = _first_replay_difference(current["result"], actual)
        if difference is not None:
            field, expected, observed = difference
            key = field.split(":", 1)[0] if ":" in field else None
            field_name = field.split(":", 1)[1] if ":" in field else field
            raise ReplayError(
                "ledger replay mismatch",
                details={
                    "view_id": view_id,
                    "key": key,
                    "field": field_name,
                    "expected": expected,
                    "actual": observed,
                    "view_version": int(current["current_version"]),
                },
            )
        compared.append(
            {
                "view_id": view_id,
                "view_version": int(current["current_version"]),
                "result_sha256": current["result_sha256"],
            }
        )
    return {
        "compared_views": compared,
        "compared_count": len(compared),
    }


def _validate_snapshot_specs(snapshot: dict[str, Any]) -> Specs:
    with tempfile.TemporaryDirectory(prefix="arbiter-snapshot-validate-") as temporary:
        environment_path = Path(temporary) / "environments.json"
        view_path = Path(temporary) / "views.json"
        atomic_write_json(environment_path, snapshot["environment_document"])
        atomic_write_json(view_path, snapshot["view_document"])
        try:
            specs = load_specs(environment_path, view_path)
            if (
                specs.environment_set_version != snapshot.get("environment_set_version")
                or specs.source_priority_version != snapshot.get("source_priority_version")
                or specs.session.scale_policy_version != snapshot.get("scale_policy_version")
                or specs.session.arbitration_policy_version != snapshot.get("arbitration_policy_version")
            ):
                raise SnapshotError(
                    "snapshot version fields do not match embedded specifications",
                    details={"path": str(environment_path.parent)},
                )
            return specs
        except SpecError as exc:
            raise SnapshotError(
                "snapshot embeds an invalid specification",
                details={"cause": exc.as_dict()},
            ) from exc


def _ledger_snapshot_summary(specs: Specs, ledger: Ledger) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    current_views = [
        view for view in ledger.all_current_views() if view["view_id"] in specs.views
    ]
    completed = [
        {
            "view_id": view["view_id"],
            "view_version": int(view["current_version"]),
            "result_sha256": view["result_sha256"],
        }
        for view in sorted(current_views, key=lambda item: item["view_id"])
    ]
    summaries = {
        view["view_id"]: _view_summary_record(view)
        for view in sorted(current_views, key=lambda item: item["view_id"])
    }
    return completed, summaries


def _snapshot_summary_matches(snapshot: dict[str, Any], specs: Specs, ledger: Ledger) -> bool:
    completed, summaries = _ledger_snapshot_summary(specs, ledger)
    return (
        completed == snapshot.get("completed_views")
        and summaries == snapshot.get("view_summaries")
    )


def _replace_from_snapshot(
    *,
    snapshot: dict[str, Any],
    snapshot_path: str | Path,
    environment_path: str | Path,
    views_path: str | Path,
    ledger: Ledger,
    reason: str,
    previous_state_digest: str,
) -> dict[str, Any]:
    replacement_specs = _validate_snapshot_specs(snapshot)
    atomic_write_json(environment_path, snapshot["environment_document"])
    atomic_write_json(views_path, snapshot["view_document"])
    ledger.reset_all()
    engine = ArbiterEngine(replacement_specs, ledger)
    run_report = engine.run_all(verify_digest=True)
    write_snapshot(snapshot_path, replacement_specs, ledger)
    return {
        "mode": "replaced",
        "reason": reason,
        "previous_state_digest": previous_state_digest,
        "restored_state_digest": snapshot["state_digest"],
        "replay": None,
        "run": run_report,
        "ledger_content_digest": ledger.meta()["content_digest"],
    }


def restore_workspace(
    *,
    snapshot_path: str | Path,
    environment_path: str | Path,
    views_path: str | Path,
    db_path: str | Path,
    expected_ledger_digest: str | None = None,
) -> dict[str, Any]:
    """Restore by whole-state replacement, then replay and resume remaining views."""
    snapshot = load_snapshot(snapshot_path)
    ledger = Ledger(db_path)
    try:
        ledger.verify_integrity(expected_digest=expected_ledger_digest)
        try:
            current_environment = read_json(environment_path)
            current_views = read_json(views_path)
            current_state_digest = _state_digest(current_environment, current_views)
        except (OSError, UnicodeError, ValueError):
            current_state_digest = "<missing-or-invalid-input-state>"
        context = ledger.meta()
        context_matches = (
            context.get("environment_set_version") == snapshot.get("environment_set_version")
            and context.get("source_priority_version") == snapshot.get("source_priority_version")
            and context.get("scale_policy_version") == snapshot.get("scale_policy_version")
            and context.get("arbitration_policy_version") == snapshot.get("arbitration_policy_version")
        )
        state_matches = current_state_digest == snapshot["state_digest"] and context_matches

        if not state_matches:
            return _replace_from_snapshot(
                snapshot=snapshot,
                snapshot_path=snapshot_path,
                environment_path=environment_path,
                views_path=views_path,
                ledger=ledger,
                reason="snapshot_state_or_version_mismatch",
                previous_state_digest=current_state_digest,
            )

        specs = load_specs(environment_path, views_path)
        engine = ArbiterEngine(specs, ledger)
        replay_report = replay_ledger(engine)
        if not _snapshot_summary_matches(snapshot, specs, ledger):
            return _replace_from_snapshot(
                snapshot=snapshot,
                snapshot_path=snapshot_path,
                environment_path=environment_path,
                views_path=views_path,
                ledger=ledger,
                reason="snapshot_conclusion_summary_mismatch",
                previous_state_digest=current_state_digest,
            )
        run_report = engine.run_all(verify_digest=True)
        write_snapshot(snapshot_path, specs, ledger)
        return {
            "mode": "replayed",
            "reason": "state_and_versions_match",
            "previous_state_digest": current_state_digest,
            "restored_state_digest": snapshot["state_digest"],
            "replay": replay_report,
            "run": run_report,
            "ledger_content_digest": ledger.meta()["content_digest"],
        }
    finally:
        ledger.close()
