from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .receipts import encode_line
from .utils import write_json


def digest(seed: int, label: str) -> str:
    return hashlib.sha256(f"{seed}:{label}".encode("utf-8")).hexdigest()[:16]


def _worker(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed * 17 + 11)
    base_port = 43000 + (seed % 700)
    dims = [32, 48, 64]
    hidden = [64, 96, 128]
    return [
        {
            "worker_id": "W1",
            "host": "127.0.0.1",
            "port": base_port + 1,
            "model_dim": dims[rng.randrange(len(dims))],
            "hidden_dim": hidden[rng.randrange(len(hidden))],
            "quota": 4,
            "param_versions": [1, 2],
            "current_param_version": 1,
        },
        {
            "worker_id": "W2",
            "host": "127.0.0.1",
            "port": base_port + 2,
            "model_dim": dims[rng.randrange(len(dims))],
            "hidden_dim": hidden[rng.randrange(len(hidden))],
            "quota": 4,
            "param_versions": [1],
            "current_param_version": 1,
        },
        {
            "worker_id": "W3",
            "host": "127.0.0.1",
            "port": base_port + 3,
            "model_dim": dims[rng.randrange(len(dims))],
            "hidden_dim": hidden[rng.randrange(len(hidden))],
            "quota": 5,
            "param_versions": [1, 2, 3],
            "current_param_version": 2,
        },
    ]


def _base_inputs(seed: int) -> dict[str, Any]:
    workers = _worker(seed)
    batches = [
        {"batch_id": "B101", "worker_id": "W1", "batch_size": 3, "param_version": 1},
        {"batch_id": "B102", "worker_id": "W1", "batch_size": 3, "param_version": 1},
        {"batch_id": "B109", "worker_id": "W1", "batch_size": 2, "param_version": 1},
        {"batch_id": "B201", "worker_id": "W2", "batch_size": 4, "param_version": 1},
        {"batch_id": "B209", "worker_id": "W2", "batch_size": 2, "param_version": 1},
        {"batch_id": "B301", "worker_id": "W3", "batch_size": 4, "param_version": 2},
        {"batch_id": "B302", "worker_id": "W3", "batch_size": 4, "param_version": 2},
        {"batch_id": "B309", "worker_id": "W3", "batch_size": 2, "param_version": 2},
    ]
    assignment_groups = [
        (1, 3, 10, 7, "B101", "W1"),
        (4, 3, 40, 6, "B102", "W1"),
        (7, 4, 70, 5, "B201", "W2"),
        (11, 4, 110, 4, "B301", "W3"),
        (15, 3, 150, 6, "B302", "W3"),
    ]
    assignments = []
    for start, count, base_arrival, step, batch_id, worker_id in assignment_groups:
        for offset in range(count):
            seq = start + offset
            assignments.append((
                f"REQ-{seq:04d}",
                seq,
                base_arrival + offset * step,
                batch_id,
                worker_id,
            ))
    assignments.append(("REQ-0018", 18, 181, "B309", "W3"))
    requests = []
    for request_id, seq, arrived_ms, batch_id, worker_id in assignments:
        requests.append({
            "seq": seq,
            "request_id": request_id,
            "arrived_ms": arrived_ms,
            "batch_id": batch_id,
            "worker_id": worker_id,
            "output_digest": digest(seed, request_id),
        })
    rules = {
        "rules_id": "merge-rules-v1",
        "rule_version": 1,
        "allowed_workers": ["W1", "W2", "W3"],
        "worker_sets": {
            "cpu_pool": ["W1", "W2"],
            "accelerator_pool": ["W3"],
        },
        "policy": {
            "authority_order": [
                "r3_override_reference",
                "param_version_desc",
                "batch_id_asc",
                "latency_ms_asc",
                "receipt_uid_asc",
            ],
            "missing_r3_override_is_undecidable": True,
            "fixed_point_scale": 1_000_000,
        },
        "budgets": {
            "batch_budget_units": 2,
            "worker_quota_units": 4,
        },
        "resource": {
            "mac_factor": 2,
            "version_weight_step_q": 10_000,
        },
    }
    commands = {
        "schema_version": "commands.v1",
        "seed": seed,
        "commands": [
            {"command_id": "C001-normalize-constrained", "op": "normalize", "reason": "initial constrained normalization"},
            {"command_id": "C002-grant-budget", "op": "grant_budget", "batch_budget": 8, "worker_quota": 16},
            {"command_id": "C003-normalize-full", "op": "normalize", "reason": "budget granted"},
            {
                "command_id": "C004-w3-version-digest",
                "op": "set_param_version",
                "worker_id": "W3",
                "param_version": 2,
                "param_digest": digest(seed, "W3-v2-new"),
            },
            {"command_id": "C005-normalize-version-change", "op": "normalize", "reason": "propagate W3 version invalidation"},
            {
                "command_id": "C006-replace-req-0001",
                "op": "replace_request",
                "request_id": "REQ-0001",
                "request": {"output_digest": digest(seed, "REQ-0001-replacement")},
            },
            {"command_id": "C007-normalize-request-replacement", "op": "normalize", "reason": "propagate one request replacement"},
            {
                "command_id": "C008-switch-rules-v2",
                "op": "switch_rules",
                "rules": {
                    **rules,
                    "rules_id": "merge-rules-v2",
                    "rule_version": 2,
                    "budgets": {"batch_budget_units": 8, "worker_quota_units": 16},
                    "policy": {**rules["policy"], "tie_breaker_note": "same deterministic order; rule hash changes"},
                },
            },
            {"command_id": "C009-normalize-rules-change", "op": "normalize", "reason": "propagate rule switch"},
            {"command_id": "C010-checkpoint", "op": "checkpoint"},
        ],
    }
    return {
        "schema_version": "workspace-inputs.v1",
        "seed": seed,
        "allowed_port_range": [43000, 43999],
        "workers_payload": {"schema_version": "workers.v1", "seed": seed, "allowed_port_range": [43000, 43999], "workers": workers},
        "request_stream_payload": {"schema_version": "request-stream.v1", "seed": seed, "batches": batches, "requests": requests},
        "rules_payload": rules,
        "commands_payload": commands,
    }


def _receipt(seed: int, batch_id: str, request_id: str, version: str, **overrides: Any) -> dict[str, Any]:
    worker_id = overrides.pop("worker_id", "W1")
    param_version = overrides.pop("param_version", 1)
    latency_ms = overrides.pop("latency_ms", 20 + (int(request_id.split("-")[1]) * 3) % 70)
    record = {
        "batch_id": batch_id,
        "request_id": request_id,
        "output_digest": overrides.pop("output_digest", digest(seed, f"{batch_id}:{request_id}:{version}")),
        "param_version": param_version,
        "latency_ms": latency_ms,
        "worker_id": worker_id,
        "rank": overrides.pop("rank", int(request_id.split("-")[1]) % 10),
        "overrides": overrides.pop("overrides", None),
        "overrides_state": overrides.pop("overrides_state", "explicit_none"),
        "param_digest": overrides.pop("param_digest", digest(seed, f"{worker_id}:v{param_version}")),
    }
    record.update(overrides)
    return record


def generate_receipt_files(seed: int) -> dict[str, str]:
    r1_records = [
        _receipt(seed, "B101", "REQ-0001", "r1", worker_id="W1"),
        _receipt(seed, "B101", "REQ-0002", "r1", worker_id="W1"),
        _receipt(seed, "B101", "REQ-0003", "r1", worker_id="W1"),
    ]
    r2_records = [
        _receipt(seed, "B102", "REQ-0004", "r2", worker_id="W1", param_version=1, latency_ms=31),
        _receipt(seed, "B102", "REQ-0005", "r2", worker_id="W1", param_version=1, latency_ms=29),
        _receipt(seed, "B102", "REQ-0006", "r2", worker_id="W1", param_version=1, latency_ms=33),
        _receipt(seed, "B201", "REQ-0007", "r2", worker_id="W2", latency_ms=44),
        _receipt(seed, "B201", "REQ-0008", "r2", worker_id="W2", latency_ms=41),
        _receipt(seed, "B201", "REQ-0009", "r2", worker_id="W2", latency_ms=43),
        _receipt(seed, "B201", "REQ-0010", "r2", worker_id="W2", latency_ms=40),
        _receipt(seed, "B109", "REQ-0004", "r2", worker_id="W1", param_version=1, latency_ms=35),
        _receipt(seed, "B209", "REQ-0015", "r2", worker_id="W2", param_version=1, latency_ms=88),
    ]
    r3_in_order = [
        _receipt(seed, "B301", "REQ-0011", "r3", worker_id="W3", param_version=2, latency_ms=22, rank=3),
        _receipt(seed, "B301", "REQ-0012", "r3", worker_id="W3", param_version=2, latency_ms=19, rank=1),
        _receipt(seed, "B301", "REQ-0013", "r3", worker_id="W3", param_version=2, latency_ms=24, rank=4),
        _receipt(seed, "B301", "REQ-0014", "r3", worker_id="W3", param_version=2, latency_ms=20, rank=2),
        _receipt(seed, "B302", "REQ-0015", "r3", worker_id="W3", param_version=2, latency_ms=18, rank=1,
                 overrides="B209@W2", overrides_state="reference"),
        _receipt(seed, "B302", "REQ-0016", "r3", worker_id="W3", param_version=2, latency_ms=21, rank=2,
                 output_digest=digest(seed, "conflict-a")),
        _receipt(seed, "B302", "REQ-0016", "r3", worker_id="W3", param_version=2, latency_ms=21, rank=2,
                 output_digest=digest(seed, "conflict-b")),
        _receipt(seed, "B302", "REQ-0017", "r3", worker_id="W3", param_version=2, latency_ms=23, rank=3),
        _receipt(seed, "B309", "REQ-0017", "r3", worker_id="W3", param_version=2, latency_ms=26, rank=1,
                 overrides=None, overrides_state="missing"),
        _receipt(seed, "B309", "REQ-0018", "r3", worker_id="W3", param_version=2, latency_ms=17, rank=2),
    ]
    # Deliberately disturb file order; ranks must restore B301 order 12,14,11,13.
    r3_records = [r3_in_order[i] for i in [2, 0, 3, 1, 6, 4, 7, 8, 5, 9]]
    return {
        "receipts-r1.log": "".join(encode_line(record, "r1") + "\n" for record in r1_records),
        "receipts-r2.log": "".join(encode_line(record, "r2") + "\n" for record in r2_records),
        "receipts-r3.log": "".join(encode_line(record, "r3") + "\n" for record in r3_records),
    }


def injection_config(seed: int) -> dict[str, Any]:
    return {
        "schema_version": "injections.v1",
        "seed": seed,
        "injections": [
            {
                "id": "INJ-001-truncated-receipt",
                "kind": "truncate_line",
                "file": "receipts-r1.log",
                "line_no": 2,
                "drop_suffix": 9,
                "expected_code": "E_RECEIPT_TRUNCATED",
            },
            {
                "id": "INJ-002-unknown-param-version",
                "kind": "set_field",
                "file": "receipts-r2.log",
                "line_no": 1,
                "field": "param_version",
                "value": 999,
                "expected_code": "E_VERSION_UNKNOWN",
            },
            {
                "id": "INJ-003-nonfinite-latency",
                "kind": "set_field",
                "file": "receipts-r2.log",
                "line_no": 2,
                "field": "latency_ms",
                "value": "NaN",
                "expected_code": "E_RECEIPT_NONFINITE_LATENCY",
            },
            {
                "id": "INJ-004-duplicate-request-id",
                "kind": "duplicate_request_id",
                "source": "request_stream.json",
                "seq": 9001,
                "request_id": "REQ-0001",
                "arrived_ms": 999,
                "batch_id": "B101",
                "worker_id": "W1",
                "expected_code": "E_SCHEMA_INVALID",
            },
        ],
    }


def invalid_validation_sample() -> dict[str, Any]:
    return {
        "workers.json": {
            "workers": [
                {"worker_id": "W-OK", "host": "127.0.0.1", "port": 43501, "model_dim": 32, "hidden_dim": 64},
                {"worker_id": "W-DUP", "host": "127.0.0.1", "port": 43502, "model_dim": 32, "hidden_dim": 64},
                {"worker_id": "W-DUP", "host": "127.0.0.1", "port": 43503, "model_dim": 32, "hidden_dim": 64},
                {"worker_id": "W-PORT", "host": "127.0.0.1", "port": 70000, "model_dim": 32, "hidden_dim": 64},
                {"worker_id": "W-DIM", "host": "127.0.0.1", "port": 43504, "model_dim": float("nan"), "hidden_dim": -2},
                {"worker_id": "W-UNKNOWN", "host": "127.0.0.1", "port": 43505, "model_dim": 32, "hidden_dim": 64, "mystery": True},
            ]
        },
        "request_stream.json": {
            "batches": [
                {"batch_id": "B-OK", "worker_id": "W-OK", "batch_size": 2, "param_version": 1},
                {"batch_id": "B-NEG", "worker_id": "W-OK", "batch_size": -1, "param_version": 1},
            ],
            "requests": [
                {"seq": 2, "request_id": "Q2", "arrived_ms": 3, "batch_id": "B-OK", "worker_id": "W-OK"},
                {"seq": 1, "request_id": "Q1", "arrived_ms": 3, "batch_id": "B-OK", "worker_id": "W-OK"},
                {"seq": 3, "request_id": "Q3", "arrived_ms": -1, "batch_id": "B-OK", "worker_id": "W-OK"},
            ],
        },
        "rules.json": {
            "rules_id": "bad-rules",
            "allowed_workers": ["W-OK", "W-GHOST"],
            "worker_sets": {"ghost_set": ["W-GHOST-2"]},
        },
    }


def init_workspace(root: str | Path, seed: int = 234) -> dict[str, Path]:
    root_path = Path(root)
    data_dir = root_path / "data"
    receipts_dir = data_dir / "receipts"
    invalid_dir = data_dir / "invalid_cases"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir.mkdir(parents=True, exist_ok=True)

    inputs = _base_inputs(seed)
    paths = {
        "workers": data_dir / "workers.json",
        "requests": data_dir / "request_stream.json",
        "rules": data_dir / "rules.json",
        "commands": data_dir / "commands.json",
        "injections": data_dir / "injections.json",
    }
    write_json(paths["workers"], inputs["workers_payload"])
    write_json(paths["requests"], inputs["request_stream_payload"])
    write_json(paths["rules"], inputs["rules_payload"])
    write_json(paths["commands"], inputs["commands_payload"])
    write_json(paths["injections"], injection_config(seed))

    receipt_texts = generate_receipt_files(seed)
    receipt_paths: dict[str, Path] = {}
    for name, text in receipt_texts.items():
        path = receipts_dir / name
        path.write_text(text, encoding="utf-8", newline="\n")
        receipt_paths[name] = path

    for name, payload in invalid_validation_sample().items():
        path = invalid_dir / name
        # Deliberately permit NaN here: this fixture tests non-finite dimension rejection.
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    readme = data_dir / "SAMPLE_LAYOUT.md"
    readme.write_text(
        "# Deterministic sample\n\n"
        "Receipt columns are fixed-width pipe-padded fields defined in "
        "`receipt_ledger/receipts.py`.\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"data_dir": data_dir, "receipts_dir": receipts_dir, **paths, **receipt_paths}
