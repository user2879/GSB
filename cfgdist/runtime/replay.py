"""Authoritative ledger replay and snapshot reproducibility checks."""
from __future__ import annotations

from typing import Any

from ..config.compiler import CompiledVersion, compile_version
from ..config.grid import Grid
from ..config.incremental import derive_invalidation
from ..config.manifest import Manifest, ManifestVersion
from ..config.planner import build_plan
from ..errors import E_REPLAY, ErrorDetail, PlatformError
from ..runtime.converge import expected_node_summary
from ..runtime.view import ViewRow, build_view
from ..util.canonical import digest


def replay_compilation(
    manifest_version: ManifestVersion, grid: Grid
) -> CompiledVersion:
    return compile_version(manifest_version, grid)


def replay_invalidation(
    previous_manifest: ManifestVersion | None, manifest_version: ManifestVersion
) -> tuple[set[str], list[dict[str, str]]]:
    invalidated, steps = derive_invalidation(previous_manifest, manifest_version)
    return invalidated, [
        {"key": step.key, "trigger": step.trigger, "source": step.source} for step in steps
    ]


def replay_plan(
    manifest_version: ManifestVersion, compiled: CompiledVersion, grid: Grid
) -> dict[str, Any]:
    plan = build_plan(manifest_version, compiled, grid.q)
    return plan.to_dict()


def replay_expected_summaries(
    manifest_version: ManifestVersion,
    compiled: CompiledVersion,
    registry: Any,
    plan_requests: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for request in plan_requests:
        summaries[request["node_id"]] = expected_node_summary(
            node_id=request["node_id"],
            request_keys=list(request["keys"]),
            manifest_version=manifest_version,
            compiled=compiled,
            registry=registry,
        )
    return summaries


def verify_snapshot_replay(
    manifest: Manifest,
    registry: Any,
    grid: Grid,
    snapshot: dict[str, Any],
    previous_manifest: ManifestVersion | None,
) -> dict[str, Any]:
    """Re-derive every recorded artifact and compare field-by-field."""
    version = snapshot["version"]
    manifest_version = next(v for v in manifest.versions if v.version == version)
    compiled = replay_compilation(manifest_version, grid)
    details: list[ErrorDetail] = []

    # 1. Artifact / error-bound reproducibility per configuration item.
    for key in manifest_version.order:
        recorded_artifact = snapshot["artifacts"].get(key)
        if recorded_artifact is None:
            details.append(ErrorDetail(reason="snapshot missing item artifact", key=key, source="replay"))
            continue
        live_artifact = compiled.items[key]
        if recorded_artifact["artifact_id"] != live_artifact.artifact_id:
            details.append(
                ErrorDetail(reason="artifact_id not reproducible", key=key, source="replay.artifact_id")
            )
        recorded_error = recorded_artifact["error"]
        live_error = live_artifact.error.to_dict()
        if digest(recorded_error) != digest(live_error):
            for field_name in ("total_bound", "total_decimal", "immediate", "paths"):
                if digest(recorded_error.get(field_name)) != digest(live_error.get(field_name)):
                    details.append(
                        ErrorDetail(
                            reason=f"error field '{field_name}' not reproducible",
                            key=key,
                            source="replay.error",
                        )
                    )

    # 2. Invalidation set reproducibility.
    invalidated, steps = replay_invalidation(previous_manifest, manifest_version)
    if steps != snapshot["invalidation_steps"]:
        details.append(
            ErrorDetail(
                reason="invalidation steps not reproducible",
                key=",".join(sorted(invalidated)),
                source="replay.invalidation",
            )
        )

    # 3. Plan reproducibility.
    live_plan = replay_plan(manifest_version, compiled, grid)
    if digest(live_plan) != digest(snapshot["plan"]):
        details.append(
            ErrorDetail(reason="distribution plan not reproducible", key=str(version), source="replay.plan")
        )

    # 4. Receipt convergence reproducibility against expected summaries.
    live_summaries = replay_expected_summaries(
        manifest_version, compiled, registry, snapshot["plan"]["requests"]
    )
    for node_id, receipt in snapshot["receipts"].items():
        expected = live_summaries.get(node_id)
        observed = receipt.get("summary")
        if expected is None or digest(observed) != digest(expected):
            details.append(
                ErrorDetail(reason="receipt summary not reproducible", key=node_id, source="replay.receipt")
            )

    # 5. View reproducibility.
    if snapshot.get("view") is not None:
        convergence_rows = [
            type(
                "NodeLike",
                (),
                {
                    "node_id": node_id,
                    "match": digest(receipt.get("summary")) == digest(live_summaries[node_id]),
                },
            )()
            for node_id, receipt in snapshot["receipts"].items()
        ]
        convergence = type(
            "ConvergenceLike", (), {"nodes": convergence_rows, "version": version}
        )()
        live_rows = build_view(manifest_version, compiled, convergence)
        live_view = {"rows": [row.to_dict() for row in live_rows]}
        recorded_view = {k: v for k, v in snapshot["view"].items() if k != "view_digest"}
        if digest(live_view) != digest(recorded_view):
            details.append(
                ErrorDetail(reason="view payload not reproducible", key=str(version), source="replay.view")
            )

    PlatformError.raise_if(
        E_REPLAY,
        "snapshot replay diverged from the recorded ledger",
        details,
    )
    return {
        "version": version,
        "replayed_artifacts": len(manifest_version.order),
        "status": "reproducible",
        "digest": digest(snapshot),
    }
