#!/usr/bin/env python3
"""Generate the deterministic local fixture set and three receipt formats."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantledger.config import load_base_config
from quantledger.quant import evaluate_chain, output_summary, quant_summary
from quantledger.util import atomic_write_json

CONFIG_DIR = ROOT / "configs"
RECEIPT_DIR = CONFIG_DIR / "receipts"


def build_base() -> dict:
    return {
        "fixture": "quantledger-local-v1",
        "workers": [
            {
                "worker_id": "w1",
                "host": "127.0.0.1",
                "port": 19101,
                "concurrency_quota": 2,
                "lease_limit": 2,
                "frac_bits": 8,
                "quant_level": 8,
            },
            {
                "worker_id": "w2",
                "host": "127.0.0.1",
                "port": 19102,
                "concurrency_quota": 2,
                "lease_limit": 3,
                "frac_bits": 8,
                "quant_level": 8,
            },
            {
                "worker_id": "w3",
                "host": "127.0.0.1",
                "port": 19103,
                "concurrency_quota": 1,
                "lease_limit": 1,
                "frac_bits": 8,
                "quant_level": 8,
            },
        ],
        "operator": {
            "revision": 10,
            "frac_bits": 8,
            "quant_level": 8,
            "layers": [
                {"name": "l1", "rows": 3, "cols": 4, "group_size": 2, "weight_seed": 1101},
                {"name": "l2", "rows": 2, "cols": 3, "group_size": 3, "weight_seed": 1102},
                {"name": "l3", "rows": 2, "cols": 2, "group_size": 2, "weight_seed": 1103},
            ],
        },
        "batches": [
            {"batch_id": "b1", "sample_seed": 2101, "batch_size": 3},
            {"batch_id": "b2", "sample_seed": 2102, "batch_size": 2},
            {"batch_id": "b3", "sample_seed": 2103, "batch_size": 4},
        ],
    }


def build_commands() -> dict:
    return {
        "commands": [
            {"seq": 1, "type": "append_batch", "batch": {"batch_id": "b4", "sample_seed": 2104, "batch_size": 3}},
            {"seq": 2, "type": "replace_weight", "layer": "l1", "weight_seed": 3101, "revision": 11},
            {"seq": 3, "type": "set_quant_level", "value": 16, "revision": 12},
            {"seq": 4, "type": "set_frac_bits", "value": 10, "revision": 13},
            {"seq": 5, "type": "replace_weight", "layer": "l2", "weight_seed": 3102, "revision": 14},
            {"seq": 6, "type": "set_lease_limit", "worker_id": "w2", "value": 2},
            {"seq": 7, "type": "rollback_revision", "value": 1},
            {"seq": 8, "type": "set_lease_limit", "worker_id": "w1", "value": 3},
            {"seq": 9, "type": "append_batch", "batch": {"batch_id": "b5", "sample_seed": 2105, "batch_size": 2}},
        ]
    }


def build_injections() -> dict:
    return {
        "injections": [
            {"id": "INJ-001", "kind": "dimension_mismatch", "batch_id": "b1"},
            {"id": "INJ-002", "kind": "level_out_of_range", "batch_id": "b2", "value": 99},
            {"id": "INJ-003", "kind": "nonfinite_weight", "batch_id": "b3", "layer": "l1"},
            {"id": "INJ-004", "kind": "stale_generation", "batch_id": "b4"},
        ]
    }


def generate_receipts(base: dict) -> None:
    config_path = CONFIG_DIR / "base.json"
    atomic_write_json(config_path, base)
    loaded = load_base_config(config_path)
    state = loaded.state
    batch = next(item for item in state.batches if item.batch_id == "b1")
    output, records, _, _ = evaluate_chain(
        state.layers,
        batch.samples,
        revision=state.revision,
        frac_bits=state.frac_bits,
        quant_level=state.quant_level,
        batch_id=batch.batch_id,
    )
    summary = output_summary(output)
    quant = quant_summary(records)
    q1 = [
        {
            "q": 1,
            "batch_id": "b1",
            "layer_name": "l3",
            "output_summary": summary,
        }
    ]
    q2 = [
        {
            "q": 2,
            "batch_id": "b1",
            "layer_name": "l3",
            "output_summary": summary,
            "revision": 1,
            "duration_ms": 7,
            "worker_id": "w3",
        },
        {
            "q": 2,
            "batch_id": "b2",
            "layer_name": "l3",
            "output_summary": summary,
            "revision": 1,
            "duration_ms": 9,
            "worker_id": "w1",
        },
    ]
    q3 = [
        {
            "q": 3,
            "batch_id": "b1",
            "layer_name": "l3",
            "output_summary": summary,
            "revision": 30,
            "duration_ms": 5,
            "worker_id": "w2",
            "stable_rank": 1,
            "superseded_refs": ["history:q1:b1:l3", "history:q2:b1:l3"],
            "quant_params_summary": quant,
        }
    ]
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(RECEIPT_DIR / "q1_receipts.json", q1)
    atomic_write_json(RECEIPT_DIR / "q2_receipts.json", q2)
    atomic_write_json(RECEIPT_DIR / "q3_receipts.json", q3)


def main() -> int:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    base = build_base()
    atomic_write_json(CONFIG_DIR / "base.json", base)
    atomic_write_json(CONFIG_DIR / "commands.json", build_commands())
    atomic_write_json(CONFIG_DIR / "injections.json", build_injections())
    generate_receipts(base)
    manifest = {
        "fixture": base["fixture"],
        "workers": [worker["worker_id"] for worker in base["workers"]],
        "ports": [worker["port"] for worker in base["workers"]],
        "layers": [layer["name"] for layer in base["operator"]["layers"]],
        "base_batches": sorted(batch["batch_id"] for batch in base["batches"]),
        "command_count": len(build_commands()["commands"]),
        "instructions": [
            "python -m quantledger.cli validate --config configs/base.json --commands configs/commands.json",
            "python -m quantledger.cli verify --config configs/base.json --commands configs/commands.json --runtime runtime/verify",
            "python -m quantledger.cli diagnose --config configs/base.json --injections configs/injections.json",
        ],
    }
    atomic_write_json(CONFIG_DIR / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
