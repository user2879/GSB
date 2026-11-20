from __future__ import annotations

import copy

from .model import (
    E_ORDER_OVERRIDDEN,
    E_SUPPLY_OVERFLOW,
    E_TIE_UNRESOLVED,
    EXCLUSIVE_PAIRS,
    EXCLUSIVE_TYPES,
    ORDER_COST,
    Config,
    content_hash,
)
from .storage import CrashAbort, MatchStore


class RecoveryMismatch(Exception):
    """A committed snapshot does not match the deterministic replay."""


def adjudication_queue(orders, cfg: Config):
    """Physical-time order: (abs_time, -priority, commander, seq).

    Arrival order and thread completion order never influence the queue.
    """
    active = [o for o in orders if not o.expired]
    return sorted(
        active,
        key=lambda o: (o.abs_time_ms(cfg.tick_ms), -o.priority, o.commander, o.seq),
    )


def _unit(units, unit_id, cfg: Config):
    unit = units.get(unit_id)
    if unit is None:
        unit = {
            "pos": None,
            "suppression": 0,
            "balance": cfg.initial_supply,
            "status": "active",
        }
        units[unit_id] = unit
    return unit


def adjudicate(match_id, orders, parse_anomalies, cfg: Config,
               store: MatchStore | None = None, crash_after: int | None = None):
    """Adjudicate one match strictly serially.

    Every step produces an immutable content-hashed snapshot. When ``store``
    is given, steps are committed atomically; already-committed steps are
    verified hash-by-hash against the deterministic replay (recovery path).
    """
    queue = adjudication_queue(orders, cfg)
    units = {}
    exclusive = {}
    anomalies = list(parse_anomalies)
    overflow_count = 0
    first_overflow = None
    ledger_entries = []
    steps = []

    committed = 0
    prev_hash = content_hash({"genesis": match_id})
    if store is not None:
        manifest = store.read_manifest()
        if manifest:
            committed = int(manifest["committed"])
            store.truncate_ledger(committed)

    prev_marker = None
    for step_idx, order in enumerate(queue):
        abs_time = order.abs_time_ms(cfg.tick_ms)

        if prev_marker is not None and prev_marker[0] == abs_time and prev_marker[1] == order.priority:
            anomalies.append({
                "code": E_TIE_UNRESOLVED,
                "step": step_idx,
                "abs_time": abs_time,
                "priority": order.priority,
                "orders": [prev_marker[2], order.order_id],
                "rule": "same abs_time and priority: resolved by commander asc, then seq asc",
                "winner": prev_marker[2],
            })

        status = "applied"
        override = None
        unit = _unit(units, order.unit_id, cfg)
        ex = exclusive.get(order.unit_id)
        if (
            order.order_type in EXCLUSIVE_TYPES
            and ex is not None
            and 0 <= abs_time - ex["time"] <= cfg.adjudication_window_ms
            and frozenset({order.order_type, ex["order_type"]}) in EXCLUSIVE_PAIRS
        ):
            status = "overridden"
            override = {
                "code": E_ORDER_OVERRIDDEN,
                "step": step_idx,
                "loser": order.order_id,
                "winner": ex["order_id"],
                "winner_time": ex["time"],
                "reason": "mutually exclusive order inside adjudication window; first in adjudication order wins",
                "window_ms": cfg.adjudication_window_ms,
            }
            anomalies.append(override)

        delta = 0
        if status == "applied":
            delta = ORDER_COST[order.order_type]
            if order.order_type == "MOVE":
                unit["pos"] = list(order.target_cell)
                exclusive[order.unit_id] = {
                    "time": abs_time, "order_id": order.order_id, "order_type": order.order_type,
                }
            elif order.order_type == "RETREAT":
                unit["status"] = "retreated"
                exclusive[order.unit_id] = {
                    "time": abs_time, "order_id": order.order_id, "order_type": order.order_type,
                }
            elif order.order_type == "SUPPRESS":
                unit["suppression"] += 1
            unit["balance"] += delta
            if unit["balance"] < 0 or unit["balance"] > cfg.supply_cap:
                overflow_count += 1
                is_first = first_overflow is None
                if is_first:
                    first_overflow = {
                        "step": step_idx, "unit_id": order.unit_id, "balance": unit["balance"],
                    }
                anomalies.append({
                    "code": E_SUPPLY_OVERFLOW,
                    "step": step_idx,
                    "unit_id": order.unit_id,
                    "balance": unit["balance"],
                    "cap": cfg.supply_cap,
                    "first": is_first,
                })

        content = {
            "match_id": match_id,
            "step": step_idx,
            "order_id": order.order_id,
            "abs_time": abs_time,
            "status": status,
            "override": override,
            "units": copy.deepcopy(units),
            "exclusive": copy.deepcopy(exclusive),
            "anomalies": copy.deepcopy(anomalies),
            "overflow_count": overflow_count,
            "first_overflow": first_overflow,
            "prev_hash": prev_hash,
        }
        digest = content_hash(content)
        entry = {
            "step": step_idx,
            "unit_id": order.unit_id,
            "order_id": order.order_id,
            "delta": delta,
            "balance": unit["balance"],
            "status": status,
        }

        if store is not None:
            if step_idx < committed:
                existing = store.read_snapshot(step_idx)
                if existing["hash"] != digest:
                    raise RecoveryMismatch(
                        f"{match_id} step {step_idx}: committed {existing['hash']} != replay {digest}"
                    )
            else:
                store.commit(content, entry)
                if crash_after is not None and step_idx + 1 >= crash_after:
                    store.leave_partial_artifact(step_idx + 1)
                    raise CrashAbort(
                        f"simulated crash after {step_idx + 1} commits in {match_id}"
                    )

        ledger_entries.append(entry)
        steps.append({
            "step": step_idx,
            "hash": digest,
            "order_id": order.order_id,
            "abs_time": abs_time,
            "status": status,
        })
        prev_hash = digest
        prev_marker = (abs_time, order.priority, order.order_id)

    return {
        "match_id": match_id,
        "steps": steps,
        "final_hash": prev_hash,
        "anomalies": anomalies,
        "ledger_entries": ledger_entries,
        "overflow_count": overflow_count,
        "first_overflow": first_overflow,
    }
