"""Snapshot file helpers and strict envelope validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import ESTIMATOR_VERSION, SNAPSHOT_FORMAT_VERSION
from .errors import SnapshotError
from .models import AggregationSpec, ReplayParams
from .util import read_json, sha256_file
from .validation import validate_aggregation_spec, validate_replay_params, validate_snapshot_envelope


def snapshot_sequence(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "param_version",
        "commit_seq",
        "replay_completed_ns",
        "delay_ns",
        "fixed_delay",
    )
    return [
        {field: sample[field] for field in fields}
        for sample in sorted(samples, key=lambda item: int(item["commit_seq"]))
    ]


def load_snapshot_document(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    try:
        document = read_json(source)
    except Exception as exc:  # noqa: BLE001 - convert every read problem to E_SNAPSHOT
        raise SnapshotError(
            f"无法读取快照 JSON: {exc}", field="snapshot", details={"path": str(source)}
        ) from exc
    try:
        checked = validate_snapshot_envelope(document, SNAPSHOT_FORMAT_VERSION)
    except SnapshotError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SnapshotError(f"快照格式非法: {exc}", field="snapshot") from exc
    return checked, sha256_file(source)


def parse_snapshot_contract(
    document: dict[str, Any],
) -> tuple[ReplayParams, AggregationSpec]:
    estimator_raw = document.get("estimator")
    try:
        spec = validate_aggregation_spec(estimator_raw)
    except Exception as exc:  # noqa: BLE001
        raise SnapshotError(
            "快照估算口径不兼容",
            field="estimator",
            details={"reason": str(exc), "expected_version": ESTIMATOR_VERSION},
        ) from exc
    if not isinstance(estimator_raw, dict) or estimator_raw.get("spec_digest") != spec.digest:
        raise SnapshotError(
            "快照估算口径摘要不一致",
            field="estimator.spec_digest",
            details={"computed": spec.digest},
        )
    replay_raw = document.get("replay_params")
    try:
        params = validate_replay_params(replay_raw, spec)
    except Exception as exc:  # noqa: BLE001
        raise SnapshotError(
            "快照回放参数不合法", field="replay_params", details={"reason": str(exc)}
        ) from exc
    if not isinstance(replay_raw, dict) or replay_raw.get("version") != params.version:
        raise SnapshotError(
            "快照回放参数版本摘要不一致",
            field="replay_params.version",
            details={
                "expected": params.version,
                "actual": replay_raw.get("version") if isinstance(replay_raw, dict) else None,
            },
        )
    return params, spec


def snapshot_doc_digest_without_volatile(document: dict[str, Any]) -> str:
    from .util import sha256_json

    stable = {key: value for key, value in document.items() if key != "created_at_utc"}
    return sha256_json(stable)


def loads_canonical(value: str) -> Any:
    return json.loads(value)
