"""Deterministic fault injection hooks (command-line / test use only).

Three failure classes are supported and each is raised with its dedicated
machine code and field path before any ledger mutation occurs:
* load sequence gap            -> E_LOAD
* illegal replica budget       -> E_BUDGET
* snapshot version conflict    -> E_VERSION
"""

from __future__ import annotations

import copy
from typing import Any

from .common import is_pos_int
from .errors import BudgetError, LoadError, VersionConflictError


def inject_load_gap(raw_spec: dict[str, Any], service_name: str,
                    minute: int) -> dict[str, Any]:
    mutated = copy.deepcopy(raw_spec)
    target_path = None
    for index, svc in enumerate(mutated.get("services", [])):
        if svc.get("name") == service_name:
            if not (0 <= minute < len(svc.get("requests", []))):
                raise LoadError(
                    f"cannot inject gap: minute {minute} is outside the load horizon",
                    fields=[f"/services/{index}/requests"],
                )
            svc["requests"][minute] = None
            target_path = f"/services/{index}/requests[{minute}]"
            break
    if target_path is None:
        raise LoadError(
            f"service {service_name!r} not found for load-gap injection",
            fields=["/services"],
        )
    _raise_if_load_gap(mutated)
    return mutated


def _raise_if_load_gap(raw_spec: dict[str, Any]) -> None:
    for index, svc in enumerate(raw_spec.get("services", [])):
        requests = svc.get("requests", [])
        for minute, value in enumerate(requests):
            if value is None:
                raise LoadError(
                    f"load sequence gap detected for service {svc.get('name')!r} at minute {minute}",
                    fields=[f"/services/{index}/requests[{minute}]"],
                    context={"service": svc.get("name"), "minute": minute},
                )
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise LoadError(
                    f"non-integer load entry for service {svc.get('name')!r} at minute {minute}",
                    fields=[f"/services/{index}/requests[{minute}]"],
                )


def inject_budget_violation(raw_spec: dict[str, Any], bad_budget: int = 0) -> dict[str, Any]:
    mutated = copy.deepcopy(raw_spec)
    mutated["global_replica_budget"] = bad_budget
    if not is_pos_int(bad_budget):
        raise BudgetError(
            f"injected illegal global replica budget: {bad_budget!r}",
            fields=["/global_replica_budget"],
            context={"global_replica_budget": bad_budget},
        )
    return mutated


def inject_snapshot_version_conflict(ledger: Any, snapshot_id: str, run_id: str,
                                     current_spec_version: int, target_minute: int) -> None:
    """Persist a snapshot that deliberately points at a stale spec version."""
    stale_version = max(0, current_spec_version - 1)
    if stale_version == 0:
        raise VersionConflictError(
            "cannot inject a version conflict before any spec version exists",
            fields=["/spec_versions"],
        )
    order = int(ledger.get_meta("snapshot_counter", "0")) + 1
    with ledger.conn:
        ledger.conn.execute(
            "INSERT INTO snapshots(snapshot_id, run_id, spec_version, target_minute, "
            "created_order, note) VALUES(?, ?, ?, ?, ?, ?)",
            (snapshot_id, run_id, stale_version, target_minute, order,
             "injected stale-version snapshot"),
        )
        ledger.set_meta("snapshot_counter", str(order))
    ledger.assert_snapshot_version(snapshot_id, current_spec_version)
