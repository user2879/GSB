"""Deterministic local sample generation (fixed seed, no network, no weights)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20261121


def build_config() -> dict:
    return {
        "seed": SEED,
        "schemes": ["standard", "linear", "ntk"],
        "scheme": "standard",
        "base": 10000.0,
        "train_len": 512,
        "target_len": 2048,
        "dim": 64,
        "scale": 4.0,
        "precision": "float64",
        "dim_block_size": 16,
        "point_budget": 430,
        "concurrency": 4,
        "snapshot_commit_interval": 1,
        "prune_tolerance": 0.001,
        "error_tolerance": 1e-6,
        "base_scan_factors": [0.5, 0.75, 1.0, 1.5, 2.0, 1000000000.0],
        "scale_scan_factors": [0.001, 0.5, 1.0, 2.0, 4.0],
    }


def build_samples(config: dict, n_samples: int = 8) -> list[dict]:
    dim = int(config["dim"])
    train_len = int(config["train_len"])
    target_len = int(config["target_len"])
    samples = []
    for idx in range(n_samples):
        rng = np.random.default_rng(SEED + idx)
        sample_id = f"S{idx:03d}"
        query_pos = int(rng.integers(train_len, target_len))
        n_keys = 24 + int(rng.integers(0, 9))
        in_train = [int(query_pos - d) for d in rng.integers(1, train_len, size=n_keys // 2)]
        extra = [
            int(query_pos - d)
            for d in rng.integers(train_len + 1, target_len, size=n_keys - n_keys // 2)
        ]
        key_positions = sorted(p for p in in_train + extra if p >= 0)
        n_keys = len(key_positions)
        query = rng.standard_normal(dim).tolist()
        keys = rng.standard_normal((n_keys, dim)).tolist()
        samples.append(
            {
                "sample_id": sample_id,
                "query_pos": query_pos,
                "key_positions": key_positions,
                "query": query,
                "keys": keys,
            }
        )
    return samples


def build_edits(samples: list[dict]) -> dict:
    rng = np.random.default_rng(SEED + 999)
    replacement = dict(samples[3])
    replacement["query"] = rng.standard_normal(len(samples[3]["query"])).tolist()
    return {
        "commands": [
            {"op": "switch_scheme", "scheme": "ntk"},
            {"op": "set_base", "base": 20000.0},
            {"op": "set_scale", "scale": 2.0},
            {"op": "set_dim_block_size", "dim_block_size": 32},
            {"op": "set_precision", "precision": "float32"},
            {"op": "replace_sample", "sample_id": samples[3]["sample_id"], "sample": replacement},
        ]
    }


def build_injections(samples: list[dict]) -> dict:
    return {
        "injections": [
            {
                "injection_id": "INJ-001",
                "sample_id": samples[1]["sample_id"],
                "stage": "load_validate",
                "fault": "nonfinite_feature",
            },
            {
                "injection_id": "INJ-002",
                "sample_id": samples[2]["sample_id"],
                "stage": "load_validate",
                "fault": "key_pos_oob",
                "oob_position": 10**9,
            },
            {
                "injection_id": "INJ-003",
                "sample_id": samples[3]["sample_id"],
                "stage": "position_encoding",
                "fault": "base_zero",
            },
            {
                "injection_id": "INJ-004",
                "sample_id": samples[4]["sample_id"],
                "stage": "position_encoding",
                "fault": "scale_nonpositive",
            },
        ]
    }


def build_policy_table(config: dict) -> dict:
    dim = int(config["dim"])
    return {
        "policies": [
            {
                "policy_id": "P_FULL_16",
                "dim_block_size": 16,
                "min_prune_tolerance": 0.0,
                "scheme_priority": ["standard", "linear", "ntk"],
                "dim_range": [0, dim],
                "score_chunk": 16,
            },
            {
                "policy_id": "P_PRUNE_32",
                "dim_block_size": 32,
                "min_prune_tolerance": 0.0005,
                "scheme_priority": ["ntk", "linear", "standard"],
                "dim_range": [0, dim * 3 // 4],
                "score_chunk": 32,
            },
            {
                "policy_id": "P_PRUNE_8",
                "dim_block_size": 8,
                "min_prune_tolerance": 0.005,
                "scheme_priority": ["linear", "ntk", "standard"],
                "dim_range": [dim // 4, dim],
                "score_chunk": 8,
            },
        ]
    }


def apply_edits(config: dict, samples: list[dict], edits: dict) -> tuple[dict, list[dict]]:
    """Apply an edit command stream, returning new (config, samples)."""
    config = dict(config)
    samples = [dict(s) for s in samples]
    for cmd in edits["commands"]:
        op = cmd["op"]
        if op == "switch_scheme":
            config["scheme"] = cmd["scheme"]
            schemes = list(config["schemes"])
            if cmd["scheme"] in schemes:
                schemes.remove(cmd["scheme"])
            config["schemes"] = [cmd["scheme"]] + schemes
        elif op == "set_base":
            config["base"] = float(cmd["base"])
        elif op == "set_scale":
            config["scale"] = float(cmd["scale"])
        elif op == "set_dim_block_size":
            config["dim_block_size"] = int(cmd["dim_block_size"])
        elif op == "set_precision":
            config["precision"] = cmd["precision"]
        elif op == "replace_sample":
            for i, s in enumerate(samples):
                if s["sample_id"] == cmd["sample_id"]:
                    samples[i] = cmd["sample"]
                    break
        else:
            raise ValueError(f"unknown edit op {op!r}")
    return config, samples


def generate_all(sample_dir: str | Path) -> dict:
    """Generate every deterministic input artifact into ``sample_dir``."""
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    config = build_config()
    samples = build_samples(config)
    edits = build_edits(samples)
    injections = build_injections(samples)
    policy_table = build_policy_table(config)
    (sample_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (sample_dir / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, sort_keys=True) + "\n")
    (sample_dir / "edits.json").write_text(
        json.dumps(edits, indent=2, sort_keys=True), encoding="utf-8"
    )
    (sample_dir / "injections.json").write_text(
        json.dumps(injections, indent=2, sort_keys=True), encoding="utf-8"
    )
    (sample_dir / "policy_table.json").write_text(
        json.dumps(policy_table, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "config": config,
        "samples": samples,
        "edits": edits,
        "injections": injections,
        "policy_table": policy_table,
    }
