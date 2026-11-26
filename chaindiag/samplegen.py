from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .exchange import e2_to_e1


def default_bundle() -> dict[str, Any]:
    config = {
        "schema_version": "1.0",
        "seed": 20261127,
        "vocab_size": 32,
        "embedding_dim": 6,
        "heads": 2,
        "head_dim": 3,
        "frac_bits": 8,
        "quant_tier": "q2",
        "exchange_version": "e2",
        "point_budget": 512,
        "operator_quota": 128,
        "max_workers": 4,
        "width_tolerance": "0.0005",
        "tie_rule": "half_away_from_zero",
        "temperature_permille": 900,
    }
    tokens = [
        {"seq": index, "uid": f"t{index:04d}", "token": token}
        for index, token in enumerate([3, 7, 11, 5, 13, 17, 19, 23])
    ]
    boundaries = [0, 3, 6, 8]
    chain_e2 = {
        "format": "chain-e2",
        "version": "e2-1.0",
        "chain_seq": 1,
        "name": "local-fixedpoint-demo",
        "operators": [
            {
                "seq": 0,
                "kind": "embedding",
                "name": "embedding",
                "inputs": [],
                "scale_float": 1.0,
                "scale_permille": 1000,
                "int_bits": 16,
                "quant_scheme": "qint16",
                "params": {"table": "splitmix64", "dim": 6},
                "exchange_meta": {"precision_tag": "sample-exact-1.000"},
            },
            {
                "seq": 1,
                "kind": "segment_pool",
                "name": "segment_pool",
                "inputs": [0],
                "scale_float": 1.125,
                "scale_permille": 1125,
                "int_bits": 16,
                "quant_scheme": "qint16",
                "params": {"pool": "ordered_sum"},
                "exchange_meta": {"precision_tag": "sample-exact-1.125-from-permille"},
            },
            {
                "seq": 2,
                "kind": "attention",
                "name": "multi_head_attention",
                "inputs": [0, 1],
                "scale_float": 1.0004,
                "scale_permille": 1000,
                "int_bits": 24,
                "quant_scheme": "qint24",
                "params": {"temperature_permille": 900},
                "attention_snapshot": {"heads": 2, "dim": 3, "temperature_permille": 900},
                "exchange_meta": {"precision_tag": "sample-lossy-1.0004"},
            },
            {
                "seq": 3,
                "kind": "layer_norm",
                "name": "layer_norm",
                "inputs": [2],
                "scale_float": 1.094,
                "scale_permille": 1094,
                "int_bits": 16,
                "quant_scheme": "qint16",
                "params": {"bias_int": 0, "gamma_int": 256},
                "exchange_meta": {"precision_tag": "sample-exact-1.094"},
            },
        ],
    }
    edits = {
        "description": "覆盖令牌替换/删除、参数、链增删、段、量化与版本切换",
        "commands": [
            {"op": "replace_token", "token_seq": 1, "token": 29},
            {"op": "update_op_param", "op_seq": 1, "path": "scale_float", "value": 1.25},
            {"op": "set_segments", "boundaries": [0, 2, 5, 8]},
            {"op": "set_quant", "quant_tier": "q3"},
            {"op": "set_frac_bits", "frac_bits": 16},
            {"op": "set_version", "exchange_version": "e1"},
            {"op": "add_operator", "operator": {
                "seq": 4,
                "kind": "layer_norm",
                "name": "diagnostic_re_norm",
                "inputs": [3],
                "scale_float": 1.0,
                "scale_permille": 1000,
                "int_bits": 16,
                "params": {"bias_int": 1},
            }},
            {"op": "remove_operator", "op_seq": 4},
            {"op": "delete_token", "token_seq": 7},
            {"op": "set_version", "exchange_version": "e2"},
            {"op": "set_quant", "quant_tier": "q2"},
            {"op": "set_frac_bits", "frac_bits": 8},
        ],
    }
    injections = {
        "injections": [
            {"id": "INJ-DIM-001", "type": "dimension_mismatch", "op_seq": 0, "token_uid": "t0003", "delta": 1},
            {"id": "INJ-NAN-002", "type": "non_finite_feature", "op_seq": 0, "token_uid": "t0005"},
            {"id": "INJ-SEG-003", "type": "segment_out_of_bounds", "op_seq": 1, "token_uid": "t0006", "delta": 1},
        ]
    }
    budget_cases = {
        "serial": {"max_workers": 1, "point_budget": 512, "operator_quota": 128},
        "concurrent": {"max_workers": 4, "point_budget": 512, "operator_quota": 128},
        "pending": {"max_workers": 3, "point_budget": 11, "operator_quota": 128},
        "quota": {"max_workers": 2, "point_budget": 512, "operator_quota": 3},
    }
    return {
        "config": config,
        "tokens": tokens,
        "segments": {"boundaries": boundaries},
        "chain_e2": chain_e2,
        "edits": edits,
        "injections": injections,
        "budget_cases": budget_cases,
    }


def write_sample_bundle(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    bundle = default_bundle()
    paths: dict[str, Path] = {}

    def write_json(name: str, value: Any) -> Path:
        path = root / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths[name] = path
        return path

    write_json("config.json", bundle["config"])
    write_json("tokens.json", {"tokens": bundle["tokens"]})
    write_json("segments.json", bundle["segments"])
    e2_path = write_json("chain.e2.json", bundle["chain_e2"])
    write_json("edits.json", bundle["edits"])
    write_json("injections.json", bundle["injections"])
    write_json("budget_cases.json", bundle["budget_cases"])

    e1_text, sidecar = e2_to_e1(bundle["chain_e2"])
    e1_path = root / "chain.e1.txt"
    e1_path.write_text(e1_text, encoding="utf-8")
    paths["chain.e1.txt"] = e1_path
    write_json("chain.e1.sidecar.json", sidecar)

    invalid_e1 = root / "invalid_chain.e1.txt"
    invalid_e1.write_text("E1000000100\nXX001001000016\n", encoding="utf-8")
    paths["invalid_chain.e1.txt"] = invalid_e1
    invalid_chain = {
        "version": "e2-bad",
        "operators": [
            {"seq": 0, "kind": "embedding", "inputs": []},
            {"seq": 2, "kind": "mystery_tree", "inputs": [3]},
        ],
    }
    write_json("invalid_chain.e2.json", invalid_chain)
    return paths
