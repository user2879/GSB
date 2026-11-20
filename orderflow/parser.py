from __future__ import annotations

import json

from .model import (
    E_CELL_BOUNDS,
    E_DUPLICATE_ORDER,
    E_LATE_ORDER,
    E_ORDER_TYPE,
    E_PARSE,
    E_SEQ_GAP,
    ORDER_TYPES,
    Config,
    Order,
)


def _cell_ok(cell, cfg: Config) -> bool:
    return (
        isinstance(cell, list)
        and len(cell) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in cell)
        and 0 <= cell[0] < cfg.grid_w
        and 0 <= cell[1] < cfg.grid_h
    )


def parse_lines(lines, cfg: Config):
    """Parse JSONL order lines.

    Returns (orders, errors, anomalies). Seq continuity is checked per
    commander on the *set* of sequence numbers (sorted), so the result is
    independent of the arrival/line order of the input file.
    """
    orders = []
    errors = []
    anomalies = []
    by_commander = {}

    for idx, raw in enumerate(lines, start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append({"code": E_PARSE, "line": idx, "commander": None, "detail": str(exc)})
            continue
        commander = rec.get("commander")
        order_type = rec.get("order_type")
        if order_type not in ORDER_TYPES:
            errors.append({
                "code": E_ORDER_TYPE, "line": idx, "commander": commander,
                "order_type": order_type,
            })
            continue
        cell = rec.get("target_cell")
        if not _cell_ok(cell, cfg):
            errors.append({
                "code": E_CELL_BOUNDS, "line": idx, "commander": commander, "cell": cell,
            })
            continue
        try:
            order = Order(
                line=idx,
                tick=int(rec["tick"]),
                commander=str(commander),
                unit_id=str(rec["unit_id"]),
                order_type=order_type,
                target_cell=(cell[0], cell[1]),
                priority=int(rec["priority"]),
                latency_ms=int(rec["latency_ms"]),
                seq=int(rec["seq"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"code": E_PARSE, "line": idx, "commander": commander, "detail": str(exc)})
            continue
        by_commander.setdefault(order.commander, []).append(order)

    for commander in sorted(by_commander):
        group = sorted(by_commander[commander], key=lambda o: (o.seq, o.line))
        seen = {}
        expected = 0
        for order in group:
            if order.seq in seen:
                prev = seen[order.seq]
                if prev.fingerprint() == order.fingerprint():
                    anomalies.append({
                        "code": E_DUPLICATE_ORDER, "line": order.line,
                        "commander": commander, "seq": order.seq,
                        "kept_line": prev.line,
                        "rule": "identical redelivery: first copy kept, later copy ignored",
                    })
                    continue
                errors.append({
                    "code": E_SEQ_GAP, "line": order.line, "commander": commander,
                    "detail": "conflicting duplicate seq", "seq": order.seq,
                })
                continue
            if order.seq != expected:
                errors.append({
                    "code": E_SEQ_GAP, "line": order.line, "commander": commander,
                    "expected": expected, "got": order.seq,
                })
            seen[order.seq] = order
            expected = order.seq + 1
            if order.latency_ms > cfg.max_latency_ms:
                anomalies.append({
                    "code": E_LATE_ORDER, "line": order.line, "commander": commander,
                    "seq": order.seq, "latency_ms": order.latency_ms,
                    "window_ms": cfg.max_latency_ms,
                    "rule": "latency beyond declared window: marked expired, excluded from adjudication queue",
                })
                order.expired = True
            orders.append(order)

    return orders, errors, anomalies
