from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20261120

MATCH_IDS = ["M01_NORMAL", "M02_LATE", "M03_TIE", "M04_SUPPLY", "M05_DENIED"]

_META = {
    "M01_NORMAL": {"sched_priority": 10, "est_mem_mb": 9},
    "M02_LATE": {"sched_priority": 20, "est_mem_mb": 9},
    "M03_TIE": {"sched_priority": 30, "est_mem_mb": 9},
    "M04_SUPPLY": {"sched_priority": 40, "est_mem_mb": 9},
    "M05_DENIED": {"sched_priority": 50, "est_mem_mb": 9},
}


def _order(tick, commander, unit_id, order_type, cell, priority, latency_ms, seq):
    return {
        "tick": int(tick),
        "commander": commander,
        "unit_id": unit_id,
        "order_type": order_type,
        "target_cell": [int(cell[0]), int(cell[1])],
        "priority": int(priority),
        "latency_ms": int(latency_ms),
        "seq": int(seq),
    }


def _cycled(rng, n_per_cmdr=12, late_every=None):
    lines = []
    for commander in ("CMD_A", "CMD_B", "CMD_C"):
        for seq in range(n_per_cmdr):
            order_type = ("MOVE", "SUPPRESS", "SUPPLY")[seq % 3]
            unit_id = f"U{1 + seq % 4}"
            tick = int(rng.integers(0, 25))
            latency = int(rng.integers(0, 200))
            if late_every and seq % late_every == late_every - 1:
                latency = int(rng.integers(300, 650))
            cell = (int(rng.integers(0, 64)), int(rng.integers(0, 64)))
            priority = int(rng.integers(1, 6))
            lines.append(_order(tick, commander, unit_id, order_type, cell, priority, latency, seq))
    return lines


def _gen_normal(rng):
    return _cycled(rng)


def _gen_late(rng):
    lines = _cycled(rng, late_every=5)
    lines.append(dict(lines[7]))  # identical redelivery -> E_DUPLICATE_ORDER
    perm = rng.permutation(len(lines))  # deterministic arrival shuffle
    return [lines[int(i)] for i in perm]


def _gen_tie(rng):
    return [
        _order(5, "CMD_A", "U1", "MOVE", (10, 10), 3, 100, 0),
        _order(5, "CMD_B", "U1", "RETREAT", (0, 0), 3, 100, 0),
        _order(6, "CMD_A", "U1", "SUPPLY", (10, 10), 2, 50, 1),
        _order(6, "CMD_B", "U2", "MOVE", (5, 5), 2, 60, 1),
        _order(7, "CMD_A", "U2", "SUPPRESS", (5, 5), 2, 20, 2),
        _order(7, "CMD_B", "U1", "SUPPLY", (10, 10), 1, 20, 2),
        _order(8, "CMD_C", "U3", "MOVE", (7, 7), 1, 0, 0),
        _order(8, "CMD_C", "U3", "SUPPLY", (7, 7), 1, 10, 1),
    ]


def _gen_supply(rng):
    lines = []
    for i in range(4):
        lines.append(_order(1 + i, "CMD_A", "U8", "SUPPLY", (3, 3), 2, 10, i))
    for i in range(6):
        lines.append(_order(1 + i, "CMD_B", "U9", "SUPPRESS", (4, 4), 2, 20, i))
    lines.append(_order(9, "CMD_C", "U7", "MOVE", (2, 2), 1, 0, 0))
    return lines


def _gen_denied(rng):
    lines = []
    for commander in ("CMD_A", "CMD_B"):
        for seq in range(3):
            order_type = ("MOVE", "SUPPLY", "SUPPRESS")[seq % 3]
            lines.append(_order(2 + seq, commander, "U5", order_type, (9, 9), 1, 10, seq))
    return lines


_BUILDERS = {
    "M01_NORMAL": _gen_normal,
    "M02_LATE": _gen_late,
    "M03_TIE": _gen_tie,
    "M04_SUPPLY": _gen_supply,
    "M05_DENIED": _gen_denied,
}


def gen_samples(samples_dir):
    """Deterministically (re)generate the sample set. Idempotent."""
    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    metas = []
    for match_id in MATCH_IDS:
        lines = _BUILDERS[match_id](rng)
        path = samples_dir / f"{match_id}.jsonl"
        payload = "".join(json.dumps(rec, sort_keys=True) + "\n" for rec in lines)
        path.write_text(payload, encoding="utf-8")
        meta = {"match_id": match_id, **_META[match_id]}
        (samples_dir / f"{match_id}.meta.json").write_text(
            json.dumps(meta, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        metas.append((meta, path))
    return metas
