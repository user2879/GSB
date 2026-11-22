"""Deterministic local sample generation (fixed integer seed).
Generates columnar blocks covering int64 columns, float64 columns and
nullable columns, and pre-stages blocks in all three historical formats:
  * blk-a1, blk-a2 : stored as b1 (no types / no null bitmap in payload)
  * blk-b1, blk-b2 : stored as b2 (types + null bitmap)
  * blk-c1, blk-c2 : stored directly as b3 (current version)
The canonical b3 payload of every block is written to canonical.json so the
upgraded bytes can be verified for byte-identity against a direct b3 write.
Also emits the expression task list JSON with batch split parameters.
"""

import json
import os

import numpy as np

from .formats import DEFAULT_BATCH_MAX_ROWS, DEFAULT_QUANT_SCALE_EXP
from .formats import encode_block, make_bitmap
from .store import BlockStore, block_from_columns

SEED = 20261123


def _rng():
    return np.random.RandomState(SEED)


def _ints(rng, n, lo, hi):
    return [int(v) for v in rng.randint(lo, hi + 1, size=n)]


def _floats(rng, n, lo, hi, ndigits=3):
    vals = rng.uniform(lo, hi, size=n)
    return [round(float(v), ndigits) for v in vals]


def _nulls(rng, n, prob):
    return make_bitmap([i for i in range(n) if rng.rand() < prob], n)


def build_canonical_blocks():
    """Return dict block_id -> canonical b3 Block (deterministic)."""
    rng = _rng()
    blocks = {}
    d = dict(quant_scale_exp=DEFAULT_QUANT_SCALE_EXP,
             batch_max_rows=DEFAULT_BATCH_MAX_ROWS)

    n = 250
    blocks["blk-a1"] = block_from_columns("blk-a1", [
        ("id", "int64", make_bitmap([], n), list(range(1, n + 1))),
        ("qty", "int64", make_bitmap([], n), _ints(rng, n, 0, 500)),
        ("price", "float64", make_bitmap([], n), _floats(rng, n, 0.5, 999.5)),
    ], **d)

    n = 180
    blocks["blk-a2"] = block_from_columns("blk-a2", [
        ("k", "int64", make_bitmap([], n), _ints(rng, n, -1000, 1000)),
        ("x", "float64", make_bitmap([], n), _floats(rng, n, -50.0, 50.0)),
        ("y", "float64", make_bitmap([], n), _floats(rng, n, 0.0, 10.0)),
    ], **d)

    n = 320
    blocks["blk-b1"] = block_from_columns("blk-b1", [
        ("id", "int64", make_bitmap([], n), list(range(1, n + 1))),
        ("amount", "int64", _nulls(rng, n, 0.15), _ints(rng, n, 0, 2000)),
        ("rate", "float64", _nulls(rng, n, 0.10), _floats(rng, n, 0.01, 2.5)),
    ], **d)

    n = 140
    blocks["blk-b2"] = block_from_columns("blk-b2", [
        ("sku", "int64", make_bitmap([], n), _ints(rng, n, 100, 999)),
        ("qty", "int64", _nulls(rng, n, 0.20), _ints(rng, n, 0, 60)),
        ("price", "float64", make_bitmap([], n), _floats(rng, n, 1.0, 400.0)),
    ], **d)

    n = 500
    blocks["blk-c1"] = block_from_columns("blk-c1", [
        ("id", "int64", make_bitmap([], n), list(range(1, n + 1))),
        ("amount", "int64", _nulls(rng, n, 0.12), _ints(rng, n, 0, 3000)),
        ("rate", "float64", _nulls(rng, n, 0.08), _floats(rng, n, 0.01, 3.0)),
        ("flag", "int64", make_bitmap([], n), _ints(rng, n, 0, 1)),
    ], quant_scale_exp=DEFAULT_QUANT_SCALE_EXP, batch_max_rows=128)

    n = 96
    vcol = _floats(rng, n, -2.0, 3.0)
    for i in range(0, n, 16):  # force exact divisors that hit zero
        vcol[i] = 1.5
    blocks["blk-c2"] = block_from_columns("blk-c2", [
        ("k", "int64", make_bitmap([], n), _ints(rng, n, 1, 100)),
        ("v", "float64", _nulls(rng, n, 0.10), vcol),
    ], quant_scale_exp=DEFAULT_QUANT_SCALE_EXP, batch_max_rows=128)
    return blocks


def build_tasks():
    return {
        "global_memory_budget_units": 6,
        "max_workers": 4,
        "tasks": [
            {"task_id": "t1", "block_id": "blk-a1",
             "expression": "qty * price - 10",
             "aggregation": "none", "batch_max_rows": 64, "quota": 2},
            {"task_id": "t2", "block_id": "blk-b1",
             "expression": "if(amount > 500 and rate < 0.5, amount * rate, amount / 2)",
             "aggregation": "sum", "batch_max_rows": 48, "quota": 2},
            {"task_id": "t3", "block_id": "blk-c1",
             "expression": "(amount + 7) * rate / 3",
             "aggregation": "avg", "batch_max_rows": 64, "quota": 3},
            {"task_id": "t4", "block_id": "blk-c2",
             "expression": "k * v + k / (v - 1.5)",
             "aggregation": "none", "batch_max_rows": 32, "quota": 1},
            {"task_id": "t5", "block_id": "blk-b2",
             "expression": "qty * price >= 100 or not (qty < 5)",
             "aggregation": "count", "batch_max_rows": 40, "quota": 2},
        ],
    }


def init_sample_workspace(base_dir):
    """Create data dir (blocks.db with b1/b2/b3 blocks), canonical.json and
    tasks/expressions.json under base_dir. Returns dict of written paths."""
    data_dir = os.path.join(base_dir, "data")
    blocks = build_canonical_blocks()
    store = BlockStore(data_dir)
    try:
        for block_id, version in (("blk-a1", "b1"), ("blk-a2", "b1"),
                                  ("blk-b1", "b2"), ("blk-b2", "b2"),
                                  ("blk-c1", "b3"), ("blk-c2", "b3")):
            store.put_block(blocks[block_id], version=version)
    finally:
        store.close()
    canonical_path = os.path.join(base_dir, "canonical.json")
    with open(canonical_path, "w", encoding="utf-8") as fh:
        json.dump({bid: encode_block(b, "b3") for bid, b in blocks.items()},
                  fh, indent=1, sort_keys=True)
        fh.write("\n")
    tasks_path = os.path.join(base_dir, "tasks", "expressions.json")
    os.makedirs(os.path.dirname(tasks_path), exist_ok=True)
    with open(tasks_path, "w", encoding="utf-8") as fh:
        json.dump(build_tasks(), fh, indent=1, sort_keys=True)
        fh.write("\n")
    return {"data_dir": data_dir, "canonical": canonical_path,
            "tasks": tasks_path}
