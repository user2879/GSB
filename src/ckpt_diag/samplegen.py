"""Generate deterministic local declaration JSON and on-disk receipt files."""
from __future__ import annotations

import json
from pathlib import Path

from .protocol import pick_free_port
from .receipts import encode_receipt


def build_spec_dict(*, global_budget: int = 6) -> dict:
    ports = [pick_free_port() for _ in range(3)]
    return {
        "global_budget": global_budget,
        "shards": [
            {
                "shard_id": "shard-alpha", "host": "127.0.0.1", "port": ports[0],
                "lease_cap": 3, "concurrency_quota": 2, "seed": 101,
                "param_version": 1, "input_dim": 4, "sample_count": 6,
                "layers": [
                    {"layer_id": "L0", "dim_in": 4, "dim_out": 4, "scale": 4,
                     "parent": None},
                    {"layer_id": "L1", "dim_in": 4, "dim_out": 3, "scale": 3,
                     "parent": "L0"},
                    {"layer_id": "L2", "dim_in": 3, "dim_out": 2, "scale": 2,
                     "parent": "L1"},
                ],
            },
            {
                "shard_id": "shard-beta", "host": "127.0.0.1", "port": ports[1],
                "lease_cap": 2, "concurrency_quota": 1, "seed": 202,
                "param_version": 1, "input_dim": 3, "sample_count": 5,
                "layers": [
                    {"layer_id": "M0", "dim_in": 3, "dim_out": 3, "scale": 4,
                     "parent": None},
                    {"layer_id": "M1", "dim_in": 3, "dim_out": 2, "scale": 3,
                     "parent": "M0"},
                ],
            },
            {
                "shard_id": "shard-gamma", "host": "127.0.0.1", "port": ports[2],
                "lease_cap": 2, "concurrency_quota": 2, "seed": 303,
                "param_version": 1, "input_dim": 2, "sample_count": 4,
                "layers": [
                    {"layer_id": "G0", "dim_in": 2, "dim_out": 2, "scale": 4,
                     "parent": None},
                ],
            },
        ],
        "checkpoints": [
            {"checkpoint_id": "cp-a-1", "shard_id": "shard-alpha",
             "generation": 1, "param_version": 1,
             "loss_baseline": "baseline-alpha-0001"},
            {"checkpoint_id": "cp-a-2", "shard_id": "shard-alpha",
             "generation": 2, "param_version": 1,
             "loss_baseline": "baseline-alpha-0002"},
            {"checkpoint_id": "cp-b-1", "shard_id": "shard-beta",
             "generation": 1, "param_version": 1,
             "loss_baseline": "baseline-beta-0001"},
        ],
        "events": [
            {"seq": 1, "kind": "dispatch", "shard_id": "shard-alpha",
             "batch_id": "batch-a-1", "param_version": 1, "cost_units": 2,
             "sample_ids": ["s-0", "s-1", "s-2"], "scope_layers": ["L0", "L1"]},
            {"seq": 2, "kind": "dispatch", "shard_id": "shard-alpha",
             "batch_id": "batch-a-2", "param_version": 1, "cost_units": 1,
             "sample_ids": ["s-3", "s-4"], "scope_layers": ["L2"]},
            {"seq": 3, "kind": "dispatch", "shard_id": "shard-beta",
             "batch_id": "batch-b-1", "param_version": 1, "cost_units": 1,
             "sample_ids": ["s-0", "s-1"], "scope_layers": ["M0"]},
            {"seq": 4, "kind": "lease_expire", "shard_id": "shard-alpha",
             "batch_id": "batch-a-2", "param_version": 1},
            {"seq": 5, "kind": "quota_change", "shard_id": "shard-beta",
             "lease_cap": 3, "concurrency_quota": 2},
            {"seq": 6, "kind": "rollback", "shard_id": "shard-alpha",
             "target_checkpoint_id": "cp-a-1", "param_version": 1,
             "batch_id": "batch-a-1"},
            {"seq": 7, "kind": "param_release", "shard_id": "shard-alpha",
             "param_version": 1, "batch_id": "batch-a-1"},
            {"seq": 8, "kind": "lease_expire", "shard_id": "shard-beta",
             "batch_id": "batch-b-1", "param_version": 1},
            {"seq": 9, "kind": "dispatch", "shard_id": "shard-gamma",
             "batch_id": "batch-g-1", "param_version": 1, "cost_units": 1,
             "sample_ids": ["s-0", "s-1"], "scope_layers": ["G0"]}
        ],
    }


def write_spec(path: str | Path, spec_dict: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec_dict or build_spec_dict(), indent=2,
                               ensure_ascii=False), encoding="utf-8")
    return path


def write_historical_receipts(directory: str | Path, results: dict) -> Path:
    """Emit c1/c2/c3 receipts on disk, including a mixed-format batch."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "receipts.jsonl"
    lines: list[str] = []
    items = list(results.values())
    # c1 legacy for first batch (only the three base fields).
    first = items[0]
    lines.append(encode_receipt(1, batch_id=first["batch_id"],
                                shard_id=first["shard_id"],
                                loss_summary=first["loss_summary"]))
    # c2 for second batch.
    if len(items) > 1:
        second = items[1]
        lines.append(encode_receipt(
            2, batch_id=second["batch_id"], shard_id=second["shard_id"],
            loss_summary=second["loss_summary"],
            param_version=second["param_version"], generation=1,
            elapsed_ms=second["elapsed_ms"]))
    # Mixed-format for the first batch: c2 and c3 alongside the c1 above.
    lines.append(encode_receipt(
        2, batch_id=first["batch_id"], shard_id=first["shard_id"],
        loss_summary=first["loss_summary"], param_version=first["param_version"],
        generation=1, elapsed_ms=first["elapsed_ms"]))
    lines.append(encode_receipt(
        3, batch_id=first["batch_id"], shard_id=first["shard_id"],
        loss_summary=first["loss_summary"], param_version=first["param_version"],
        generation=2, elapsed_ms=first["elapsed_ms"],
        sample_ids=first["sample_ids"], param_summary=first["param_summary"],
        lease_id=first["lease_id"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_bad_receipts(directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "bad_receipts.jsonl"
    path.write_text(
        '{"format_version": 9, "batch_id": "batch-x", "shard_id": "shard-alpha",'
        ' "loss_summary": "abc"}\n'
        '{"format_version": 2, "batch_id": "batch-y", "shard_id": "shard-alpha"}\n'
        '{"format_version": 3, "batch_id": "batch-z", "shard_id": "shard-alpha",'
        ' "loss_summary": "0123456789abcdef", "param_version": 1, "generation": 1,'
        ' "elapsed_ms": -5, "sample_ids": [], "param_summary": "UNKNOWN-SUMMARY",'
        ' "lease_id": "NO_LEASE"}\n'
        '{"format_version": 3, "batch_id": "short", "shard_id": "shard-alpha",'
        ' "loss_summary": "0123456789abcdef", "param_version": 1, "generation": 1,'
        ' "elapsed_ms": 1, "sample_ids": [], "param_summary": "UNKNOWN-SUMMARY",'
        ' "lease_id": "NO_LEASE"}\n',
        encoding="utf-8")
    return path
