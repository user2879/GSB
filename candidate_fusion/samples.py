"""Deterministic local fixture generation."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

from .io_utils import write_bytes_fsynced, write_json


SEED = 20261122


def _candidate(channel: str, candidate_id: str, rng: random.Random) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "channel": channel,
        "features": [round(rng.gauss(0.0, 1.0), 6) for _ in range(8)],
        "score": round(rng.gauss(0.0, 1.0), 6),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        for record in sorted(records, key=lambda item: item["candidate_id"])
    ]
    write_bytes_fsynced(path, ("\n".join(lines) + "\n").encode("utf-8"))


def generate_samples(root: Path | str, *, force: bool = False) -> Path:
    root = Path(root)
    marker = root / "fixture.json"
    if marker.exists() and not force:
        return root
    if root.exists() and force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    base = root / "base_workspace"
    fragments = base / "fragments"
    config = {
        "channel_quota": 10000,
        "channels": {
            "ch_alpha": {"capacity": 80, "weight": 1.5},
            "ch_beta": {"capacity": 70, "weight": 1.0},
            "ch_delta": {"capacity": 40, "weight": 0.8},
            "ch_gamma": {"capacity": 60, "weight": 1.2},
        },
        "feature_dim": 8,
        "global_budget": 100000,
        "max_trunc_k": 200,
        "max_workers": 4,
        "normalization": "minmax",
        "quant_bits": 8,
        "rounding": "half_even",
        "trunc_k": 20,
    }
    write_json(base / "config.json", config)

    rng = random.Random(SEED)
    pool = [f"cand_{index:03d}" for index in range(120)]
    rng.shuffle(pool)
    assignments = {
        "ch_alpha": sorted(pool[:55]),
        "ch_beta": sorted(pool[20:60]),
        "ch_gamma": sorted(pool[40:67]),
        "ch_delta": sorted(pool[50:62]),
    }
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for channel in sorted(assignments):
        records[channel] = {
            candidate_id: _candidate(channel, candidate_id, rng)
            for candidate_id in assignments[channel]
        }

    _write_jsonl(
        fragments / "ch_alpha" / "frag_000.jsonl",
        [records["ch_alpha"][candidate_id] for candidate_id in sorted(records["ch_alpha"])[:35]],
    )
    _write_jsonl(
        fragments / "ch_alpha" / "frag_001.jsonl",
        [records["ch_alpha"][candidate_id] for candidate_id in sorted(records["ch_alpha"])[35:]],
    )
    for channel in ("ch_beta", "ch_delta", "ch_gamma"):
        _write_jsonl(
            fragments / channel / "frag_000.jsonl",
            [records[channel][candidate_id] for candidate_id in sorted(records[channel])],
        )

    edits_dir = root / "edits"
    patch_dir = edits_dir / "patch"
    append_ids = sorted(pool[70:76])
    append_records = [_candidate("ch_alpha", candidate_id, rng) for candidate_id in append_ids]
    _write_jsonl(patch_dir / "alpha_append.jsonl", append_records)

    replacement_records: list[dict[str, Any]] = []
    for candidate_id in sorted(records["ch_beta"]):
        replacement = _candidate("ch_beta", candidate_id, rng)
        replacement_records.append(replacement)
    _write_jsonl(patch_dir / "beta_replacement.jsonl", replacement_records)

    sequence = {
        "edits": [
            {
                "channel": "ch_alpha",
                "destination": "fragments/ch_alpha/frag_002.jsonl",
                "id": "E01",
                "source": "patch/alpha_append.jsonl",
                "type": "append_fragment",
            },
            {
                "channel": "ch_beta",
                "destination": "fragments/ch_beta/frag_000.jsonl",
                "id": "E02",
                "source": "patch/beta_replacement.jsonl",
                "type": "replace_fragment",
            },
            {"channel": "ch_alpha", "id": "E03", "type": "set_weight", "value": 2.5},
            {"id": "E04", "type": "set_trunc_k", "value": 14},
            {"id": "E05", "type": "set_normalization", "value": "zscore"},
            {"id": "E06", "type": "set_trunc_k", "value": 12},
            {"id": "E07", "type": "set_quant_bits", "value": 12},
        ],
        "schema_version": 1,
        "seed": SEED,
    }
    write_json(edits_dir / "sequence.json", sequence)

    injections_dir = root / "injections"
    injection_specs = [
        {
            "id": "INJ-001",
            "type": "nonfinite_score",
            "stage": "load",
            "channel": "ch_alpha",
            "candidate_id": sorted(records["ch_alpha"])[3],
            "value": "NaN",
        },
        {
            "id": "INJ-002",
            "type": "duplicate_candidate",
            "stage": "load",
            "channel": "ch_beta",
            "candidate_id": sorted(records["ch_beta"])[5],
        },
        {
            "id": "INJ-003",
            "type": "trunc_limit_out_of_range",
            "stage": "config",
            "value": -3,
        },
        {
            "id": "INJ-004",
            "type": "zero_capacity",
            "stage": "config",
            "channel": "ch_gamma",
        },
    ]
    for spec in injection_specs:
        write_json(injections_dir / f"{spec['id']}.json", {"injections": [spec]})

    fixture = {
        "channel_candidate_counts": {
            channel: len(assignments[channel]) for channel in sorted(assignments)
        },
        "edits": [entry["id"] for entry in sequence["edits"]],
        "feature_dim": 8,
        "injections": [spec["id"] for spec in injection_specs],
        "seed": SEED,
        "version": 1,
    }
    write_json(marker, fixture)
    return root
