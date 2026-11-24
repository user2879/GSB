"""Reproducible local sample generation from an integer seed."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def basic_sample(
    *,
    seed: int = 20261125,
    port_base: int = 43171,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed)
    request_count = 10
    model_dim = 4
    hidden_dim = 6
    request_ids = [f"req-{index:04d}" for index in range(request_count)]
    requests = []
    for index, request_id in enumerate(request_ids):
        duplicate_refs = ["req-0000"] if index == 8 else []
        requests.append(
            {
                "request_id": request_id,
                "arrival_seq": index,
                "input": [rng.randint(-9, 9) for _ in range(model_dim)],
                "duplicate_refs": duplicate_refs,
            }
        )
    routes = [
        {
            "worker_id": "w-alpha",
            "request_ids": request_ids[0::2],
        },
        {
            "worker_id": "w-beta",
            "request_ids": request_ids[1::2],
        },
    ]
    request_document = {
        "format_version": 1,
        "generator": {"seed": seed, "schema": "receipt-diag-sample-v1"},
        "requests": requests,
    }
    worker_document = {
        "format_version": 1,
        "workers": [
            {
                "worker_id": "w-alpha",
                "host": "127.0.0.1",
                "port": port_base,
                "model_dim": model_dim,
                "hidden_dim": hidden_dim,
                "seed": 17,
                "param_version": 3,
                "batch_size": 2,
                "concurrency": 2,
            },
            {
                "worker_id": "w-beta",
                "host": "127.0.0.1",
                "port": port_base + 1,
                "model_dim": model_dim,
                "hidden_dim": hidden_dim,
                "seed": 29,
                "param_version": 1,
                "batch_size": 3,
                "concurrency": 1,
            },
        ],
    }
    rules_document = {
        "format_version": 1,
        "global_concurrency": 2,
        "socket_timeout_ms": 5000,
        "worker_start_timeout_ms": 10000,
        "receipt_format": "r3",
        "routes": routes,
    }
    return request_document, worker_document, rules_document


def changed_sample(
    base: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    request_document, worker_document, rules_document = json.loads(json.dumps(base))
    worker_document["workers"][0]["seed"] = 18
    worker_document["workers"][0]["param_version"] = 4
    return request_document, worker_document, rules_document


def write_sample(
    output_dir: str | Path,
    *,
    preset: str = "basic",
    seed: int = 20261125,
    port_base: int = 43171,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    base = basic_sample(seed=seed, port_base=port_base)
    documents = changed_sample(base) if preset == "changed" else base
    if preset not in ("basic", "changed"):
        raise ValueError("preset must be basic or changed")
    for name, document in zip(
        ("request_stream.json", "worker_spec.json", "merge_rules.json"),
        documents,
    ):
        _write(target / name, document)
    return {
        "sample_dir": str(target.resolve()),
        "preset": preset,
        "seed": seed,
        "port_base": port_base,
        "files": [
            str((target / name).resolve())
            for name in ("request_stream.json", "worker_spec.json", "merge_rules.json")
        ],
    }
