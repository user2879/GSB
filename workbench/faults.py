"""Deterministic fault-injection entry points for diagnostics."""

from __future__ import annotations

from dataclasses import replace

from .errors import WorkbenchError
from .loader import load_declaration
from .ledger import Ledger
from .logging_utils import JsonlLogger
from .models import ClusterSpec


def inject_declaration_corruption(path: str, logger: JsonlLogger):
    """Load a corrupt declaration; the loader must reject it via E_SCHEMA."""

    try:
        return load_declaration(path)
    except WorkbenchError as exc:
        logger.error("fault_injected", injection="declaration_corruption",
                     **exc.as_dict())
        raise


def inject_length_mismatch(spec: ClusterSpec, logger: JsonlLogger) -> ClusterSpec:
    """Return a spec whose second service series has one extra minute."""

    services = list(spec.services)
    target = services[1] if len(services) > 1 else services[0]
    patched = replace(target, requests=target.requests + (0,))
    services[services.index(target)] = patched
    tampered = ClusterSpec(
        services=tuple(services),
        global_replica_budget=spec.global_replica_budget,
        fixed_point_scale=spec.fixed_point_scale,
        latency_budget_ms=spec.latency_budget_ms,
        incremental_threshold_minutes=spec.incremental_threshold_minutes,
        window_size=spec.window_size,
    )
    logger.error(
        "fault_injected",
        injection="load_length_mismatch",
        code="E_LENGTH",
        service_id=target.service_id,
        expected_length=len(spec.services[0].requests),
        actual_length=len(patched.requests),
    )
    return tampered


def inject_snapshot_mismatch(ledger: Ledger, version: int, logger: JsonlLogger):
    """Force digest recomputation to detect a tampered snapshot (E_SNAPSHOT)."""

    try:
        ledger.verify_snapshots(version, inject_corrupt=True)
    except WorkbenchError as exc:
        logger.error("fault_injected", injection="snapshot_digest_mismatch",
                     **exc.as_dict())
        raise
