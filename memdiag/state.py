"""Conversion between runtime diagnostics and declarative persisted state."""

from __future__ import annotations

from typing import Any, Mapping

from .errors import SnapshotError
from .models import DiagnosticRun
from .plan import plan_from_dict, sample_set_from_dict
from .util import canonical_digest


def state_content_digest(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("content_digest", None)
    normalized.pop("snapshot_format_version", None)
    return canonical_digest(normalized)


def run_from_snapshot_payload(payload: Mapping[str, Any]) -> DiagnosticRun:
    if not isinstance(payload, Mapping):
        raise SnapshotError("snapshot payload must be an object")
    if "plan" not in payload or "samples" not in payload:
        raise SnapshotError("snapshot is missing plan or samples", field="snapshot")
    snapshot_format_version = payload.get("snapshot_format_version", 1)
    if not isinstance(snapshot_format_version, int) or isinstance(snapshot_format_version, bool):
        raise SnapshotError("snapshot format version must be an integer", field="snapshot_format_version")
    if snapshot_format_version != 1:
        raise SnapshotError(
            "unsupported snapshot format version",
            field="snapshot_format_version",
            expected=1,
            actual=snapshot_format_version,
        )
    recorded_digest = str(payload.get("content_digest", ""))
    actual_digest = state_content_digest(payload)
    if recorded_digest and recorded_digest != actual_digest:
        raise SnapshotError(
            "snapshot content digest does not match its payload",
            field="content_digest",
            recorded=recorded_digest,
            actual=actual_digest,
        )
    plan = plan_from_dict(payload["plan"])
    samples = sample_set_from_dict(payload["samples"])
    run = DiagnosticRun.from_dict(payload, plan=plan, samples=samples)
    if not run.content_digest:
        run.content_digest = actual_digest
    return run
