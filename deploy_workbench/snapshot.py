from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import atomic_write_json, canonical_json, digest_value
from .errors import E_REPLAY, WorkbenchError
from .model import VersionState, WorkbenchSpec


SNAPSHOT_FORMAT_VERSION = 1


def build_snapshot(
    spec: WorkbenchSpec,
    versions: VersionState,
    *,
    run_id: int,
    environments: Mapping[str, Mapping[str, Any]],
    completed_environments: Iterable[str],
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    completed = sorted(completed_environments)
    environment_payload = {env_id: environments[env_id] for env_id in completed}
    snapshot: dict[str, Any] = {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "run_id": run_id,
        "environment_version": versions.environment_version,
        "priority_version": versions.priority_version,
        "scale_version": versions.scale_version,
        "environment_fingerprint": versions.environment_fingerprint,
        "priority_fingerprint": versions.priority_fingerprint,
        "scale_fingerprint": versions.scale_fingerprint,
        "spec_digest": versions.spec_digest,
        "scale_power": spec.scale_power,
        "overflow_policy": spec.overflow_policy,
        "completed_environments": completed,
        "environments": environment_payload,
        "stats": dict(stats),
    }
    snapshot["conclusion_digest"] = digest_value(
        {
            "completed_environments": completed,
            "environments": environment_payload,
        }
    )
    return snapshot


def write_snapshot(path: str | Path, snapshot: Mapping[str, Any]) -> None:
    atomic_write_json(path, snapshot)


def read_snapshot(path: str | Path) -> dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
        snapshot = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkbenchError(E_REPLAY, f"cannot read snapshot: {exc}", path="$") from exc
    if not isinstance(snapshot, dict):
        raise WorkbenchError(E_REPLAY, "snapshot root must be an object", path="$")
    if snapshot.get("snapshot_format_version") != SNAPSHOT_FORMAT_VERSION:
        raise WorkbenchError(E_REPLAY, "unsupported snapshot format version", path="$.snapshot_format_version")
    for field in (
        "environment_version",
        "priority_version",
        "scale_version",
        "completed_environments",
        "environments",
    ):
        if field not in snapshot:
            raise WorkbenchError(E_REPLAY, f"snapshot is missing {field!r}", path=f"$.{field}")
    return snapshot


def snapshot_versions(snapshot: Mapping[str, Any]) -> tuple[int, int, int]:
    try:
        return (
            int(snapshot["environment_version"]),
            int(snapshot["priority_version"]),
            int(snapshot["scale_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkbenchError(E_REPLAY, "snapshot versions are malformed", path="$.environment_version") from exc


def _mismatch(
    environment: str | None,
    key: str | None,
    field: str,
    expected: Any,
    actual: Any,
    basis: str,
) -> None:
    path = "$"
    if environment is not None:
        path += f".environments.{environment}"
    if key is not None:
        path += f".keys.{key}"
    path += f".{field}"
    raise WorkbenchError(
        E_REPLAY,
        f"replay mismatch at environment={environment!r}, key={key!r}, field={field!r}",
        environment=environment,
        path=path,
        details={
            "key": key,
            "field": field,
            "expected": expected,
            "actual": actual,
            "basis": basis,
        },
    )


KEY_FIELDS = (
    "resolved",
    "status",
    "chain",
    "chain_digest",
    "effective_source_id",
    "effective_source_type",
    "effective_raw",
    "effective_fixed",
    "quantization_error",
    "quantization_error_abs",
    "quantization_error_bound",
    "comparison_error_bound",
)


TASK_FIELDS = (
    "status",
    "fixed_result",
    "result",
    "error_bound",
    "theoretical_error_bound",
    "actual_error",
    "bounds_valid",
    "saturated",
    "saturation_gap",
    "truncation_remainder",
    "intermediates",
)


def _compare_mapping_field(
    environment: str,
    key: str,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    field: str,
    basis: str,
) -> None:
    expected_value = expected.get(field)
    actual_value = actual.get(field)
    if canonical_json(expected_value) != canonical_json(actual_value):
        _mismatch(environment, key, field, expected_value, actual_value, basis)


def compare_replay(
    spec: WorkbenchSpec,
    snapshot: Mapping[str, Any],
    replay: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_completed = sorted(str(value) for value in snapshot.get("completed_environments", []))
    expected_environments = snapshot.get("environments")
    if not isinstance(expected_environments, dict):
        raise WorkbenchError(E_REPLAY, "snapshot environments must be an object", path="$.environments")
    actual_completed = sorted(replay)
    if expected_completed != actual_completed:
        first_missing = next((env for env in expected_completed if env not in replay), None)
        first_extra = next((env for env in actual_completed if env not in expected_completed), None)
        _mismatch(
            first_missing or first_extra,
            None,
            "completed_environments",
            expected_completed,
            actual_completed,
            "snapshot completed set differs from replayed set",
        )

    for env in spec.environment_order():
        expected_env = expected_environments.get(env.id)
        actual_env = replay.get(env.id)
        if not isinstance(expected_env, dict) or actual_env is None:
            _mismatch(env.id, None, "environment", expected_env, actual_env, "environment missing from replay or snapshot")

        for field in ("depth", "available_sources", "source_priority"):
            _compare_mapping_field(
                env.id,
                None,
                expected_env,
                actual_env,
                field,
                "environment metadata replay comparison",
            )

        expected_keys = expected_env.get("keys")
        actual_keys = actual_env.get("keys")
        if not isinstance(expected_keys, dict) or not isinstance(actual_keys, dict):
            _mismatch(env.id, None, "keys", expected_keys, actual_keys, "key maps are malformed")
        key_union = sorted(set(expected_keys) | set(actual_keys))
        for key in key_union:
            if key not in expected_keys:
                _mismatch(env.id, key, "key", None, actual_keys[key], "replay produced a key absent from snapshot")
            if key not in actual_keys:
                _mismatch(env.id, key, "key", expected_keys[key], None, "snapshot key absent from replay")
            for field in KEY_FIELDS:
                _compare_mapping_field(
                    env.id,
                    key,
                    expected_keys[key],
                    actual_keys[key],
                    field,
                    "per-key replay comparison",
                )

        expected_tasks = expected_env.get("tasks", {})
        actual_tasks = actual_env.get("tasks", {})
        if not isinstance(expected_tasks, dict) or not isinstance(actual_tasks, dict):
            _mismatch(env.id, None, "tasks", expected_tasks, actual_tasks, "task maps are malformed")
        for task_id in sorted(set(expected_tasks) | set(actual_tasks)):
            if task_id not in expected_tasks:
                _mismatch(env.id, task_id, "task", None, actual_tasks[task_id], "replay produced a task absent from snapshot")
            if task_id not in actual_tasks:
                _mismatch(env.id, task_id, "task", expected_tasks[task_id], None, "snapshot task absent from replay")
            for field in TASK_FIELDS:
                _compare_mapping_field(
                    env.id,
                    task_id,
                    expected_tasks[task_id],
                    actual_tasks[task_id],
                    field,
                    "per-task replay comparison",
                )
            expected_segments = expected_tasks[task_id].get("attribution", {}).get("segments")
            actual_segments = actual_tasks[task_id].get("attribution", {}).get("segments")
            if canonical_json(expected_segments) != canonical_json(actual_segments):
                _mismatch(
                    env.id,
                    task_id,
                    "attribution.segments",
                    expected_segments,
                    actual_segments,
                    "task attribution differs",
                )
            expected_threshold = expected_tasks[task_id].get("attribution", {}).get("threshold")
            actual_threshold = actual_tasks[task_id].get("attribution", {}).get("threshold")
            if canonical_json(expected_threshold) != canonical_json(actual_threshold):
                _mismatch(
                    env.id,
                    task_id,
                    "attribution.threshold",
                    expected_threshold,
                    actual_threshold,
                    "task attribution threshold differs",
                )
            expected_dominant = expected_tasks[task_id].get("attribution", {}).get("dominant_segment")
            actual_dominant = actual_tasks[task_id].get("attribution", {}).get("dominant_segment")
            if canonical_json(expected_dominant) != canonical_json(actual_dominant):
                _mismatch(
                    env.id,
                    task_id,
                    "attribution.dominant_segment",
                    expected_dominant,
                    actual_dominant,
                    "task attribution dominance differs",
                )

        for field in ("source_chain_digest", "conclusion_digest", "resolved_key_count", "unresolved_key_count"):
            _compare_mapping_field(
                env.id,
                None,
                expected_env,
                actual_env,
                field,
                "per-environment replay comparison",
            )

    expected_digest = digest_value(
        {
            "completed_environments": expected_completed,
            "environments": expected_environments,
        }
    )
    if snapshot.get("conclusion_digest") != expected_digest:
        _mismatch(
            None,
            None,
            "conclusion_digest",
            snapshot.get("conclusion_digest"),
            expected_digest,
            "snapshot conclusion digest does not cover its recorded conclusions",
        )

    replay_digest = digest_value(
        {
            "completed_environments": actual_completed,
            "environments": {env_id: replay[env_id] for env_id in actual_completed},
        }
    )
    if snapshot.get("conclusion_digest") != replay_digest:
        _mismatch(
            None,
            None,
            "conclusion_digest",
            snapshot.get("conclusion_digest"),
            replay_digest,
            "replay conclusion digest differs from the snapshot",
        )
    return {
        "status": "verified",
        "completed_environments": actual_completed,
        "environment_count": len(actual_completed),
        "key_count": sum(len(replay[env]["keys"]) for env in actual_completed),
        "snapshot_conclusion_digest": snapshot.get("conclusion_digest"),
        "replay_conclusion_digest": replay_digest,
    }
