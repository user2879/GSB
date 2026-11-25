"""Deterministic local sample generator.

Produces config, token sequence, segment boundaries, operator graph, e1/e2
quant files, bypass metadata, an edit command stream and a handful of
intentionally invalid entries to prove non-fatal E_SCHEMA handling.
"""

from __future__ import annotations

import json
from pathlib import Path

from .exchange import (
    QuantRecord,
    e1_to_e2,
    e2_to_e1,
    read_bypass,
    restore_e2_from_bypass,
    write_bypass,
    write_e1,
    write_e2,
)
from .config import QuantSpec, WIDTH_TO_SCHEME


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def generate(sample_root: str | Path, seed: int = 20261126) -> dict[str, Path]:
    root = Path(sample_root)
    root.mkdir(parents=True, exist_ok=True)

    config = {
        "heads": 2,
        "dim": 8,
        "frac_bits": 16,
        "gear": "g1",
        "point_budget": 520,
        "operator_quota": 260,
        "max_workers": 3,
        "vocab_size": 12,
        "batch_id": "deterministic-batch",
    }
    tokens = [3, 7, 1, 9, 4, 11, 2, 6]
    segments = [[0, 3], [3, 5], [5, 8]]
    graph = {
        "chains": [
            {
                "chain_id": 1,
                "ops": [
                    {"seq": 0, "op_type": "embed", "inputs": [],
                     "params": {"seed": seed, "table": "synthetic"}},
                    {"seq": 1, "op_type": "segment_pool", "inputs": [0],
                     "params": {"reduction": "sum"}},
                    {"seq": 2, "op_type": "multihead_attn", "inputs": [0],
                     "params": {"causal": False, "weight_scheme": "permille"}},
                    {"seq": 3, "op_type": "layer_norm", "inputs": [2],
                     "params": {"epsilon_q": 4096}},
                    {"seq": 4, "op_type": "final", "inputs": [3, 1],
                     "params": {"emit": "per_token"}},
                ],
            },
            {
                "chain_id": 2,
                "ops": [
                    {"seq": 0, "op_type": "embed", "inputs": [],
                     "params": {"seed": seed + 1, "table": "synthetic"}},
                    {"seq": 1, "op_type": "layer_norm", "inputs": [0],
                     "params": {"epsilon_q": 2048}},
                    {"seq": 2, "op_type": "final", "inputs": [1],
                     "params": {"emit": "per_token"}},
                ],
            },
        ]
    }

    # Canonical quant settings shared by e1 and e2 (equivalent semantics).
    settings = {
        (1, 0): (1000, 32),
        (1, 1): (1250, 32),
        (1, 2): (875, 16),
        (1, 3): (1000, 16),
        (1, 4): (1000, 32),
        (2, 0): (1000, 32),
        (2, 1): (1000, 16),
        (2, 2): (1100, 32),
    }
    e1_records = [
        QuantRecord(chain_id=cid, seq=seq,
                    quant=QuantSpec(permille=permille, width=width,
                                    scheme=WIDTH_TO_SCHEME[width],
                                    precision_label="e1-exact",
                                    completeness="complete"),
                    source_version="e1")
        for (cid, seq), (permille, width) in sorted(settings.items())
    ]
    # e2 derived through the real promotion path, then a downgrade/restore cycle
    # exercises the bypass metadata requirements.
    e2_records = [e1_to_e2(record) for record in e1_records]
    downgraded, bypass_map = [], {}
    for record in e2_records:
        e1_record, bypass = e2_to_e1(record)
        bypass_map[record.node_id()] = bypass
        downgraded.append(e1_record)
    restored = [restore_e2_from_bypass(record, bypass_map[record.node_id()]) for record in downgraded]
    # Attach deterministic per-token summaries on the e2 side.
    for record in restored:
        record.summary = {
            "token_count": len(tokens),
            "scale_float": record.quant.permille / 1000.0,
            "scheme": record.quant.scheme,
        }

    edits = {
        "edits": [
            {"cmd": "update_op_param", "chain_id": 1, "seq": 2,
             "params": {"causal": True, "weight_scheme": "permille"}},
            {"cmd": "replace_token", "index": 4, "token": 5},
            {"cmd": "set_quant", "chain_id": 1, "seq": 1,
             "quant": {"permille": 1300, "width": 32, "scheme": "qint32",
                       "precision_label": "manual", "completeness": "complete"}},
            {"cmd": "set_segments", "segments": [[0, 2], [2, 5], [5, 8]]},
            {"cmd": "set_gear", "gear": "g2"},
            {"cmd": "set_version", "version": "e1"},
            {"cmd": "set_frac_bits", "frac_bits": 24},
            {"cmd": "set_version", "version": "e2"},
            {"cmd": "set_gear", "gear": "g1"},
        ]
    }

    invalid_graph = {
        "chains": [
            {"chain_id": 9, "ops": [
                {"seq": 0, "op_type": "embed", "inputs": [5], "params": {}},
                {"seq": 1, "op_type": "final", "inputs": [0], "params": {}},
            ]},
        ]
    }
    invalid_quant = [
        QuantRecord(chain_id=9, seq=0,
                    quant=QuantSpec(permille=1000, width=32, scheme="qint32",
                                    precision_label="e1-exact", completeness="complete"),
                    source_version="e1"),
        QuantRecord(chain_id=9, seq=1,
                    quant=QuantSpec(permille=1000, width=32, scheme="qint32",
                                    precision_label="e1-exact", completeness="complete"),
                    source_version="e1"),
    ]
    invalid_e1_lines = [
        "00000100000000000100000000032",   # truncated width
        "0000010000000000abc000000000016",  # illegal chars in permille
    ]

    paths = {
        "config": root / "config.json",
        "tokens": root / "tokens.json",
        "segments": root / "segments.json",
        "graph": root / "graph.json",
        "e1": root / "chain.e1.txt",
        "e2": root / "chain.e2.json",
        "bypass": root / "bypass.json",
        "edits": root / "edits.json",
        "invalid_graph": root / "invalid_graph.json",
        "invalid_e1": root / "invalid_chain.e1.txt",
        "invalid_quant": root / "invalid_quant.e2.json",
    }
    _write_json(paths["config"], config)
    _write_json(paths["tokens"], {"tokens": tokens})
    _write_json(paths["segments"], {"segments": segments})
    _write_json(paths["graph"], graph)
    write_e1(paths["e1"], e1_records)
    write_e2(paths["e2"], restored, {"seed": seed})
    write_bypass(paths["bypass"], bypass_map)
    _write_json(paths["edits"], edits)
    _write_json(paths["invalid_graph"], invalid_graph)
    paths["invalid_e1"].write_text("\n".join(invalid_e1_lines) + "\n", encoding="utf-8")
    write_e2(paths["invalid_quant"], [e1_to_e2(record) for record in invalid_quant])
    return paths
