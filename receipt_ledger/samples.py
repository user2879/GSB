from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .compute import forward_batch
from .planner import build_plan, verify_coverage
from .receipts import encode_receipt_line
from .spec import load_bundle
from .compute import deterministic_vectors


FIXED_SAMPLE_SEED = 276


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def generate_sample_directory(directory: str | Path) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    dim = 4
    vectors = deterministic_vectors(FIXED_SAMPLE_SEED, 6, dim)
    workers = {
        "format": "worker-decl/v1",
        "workers": [
            {
                "worker_id": "w-alpha",
                "host": "127.0.0.1",
                "port": 49621,
                "model_dim": dim,
                "hidden_dim": 3,
                "seed": 27601,
                "param_version": 1,
                "quota": 1,
                "batch_size": 2,
            },
            {
                "worker_id": "w-beta",
                "host": "127.0.0.1",
                "port": 49622,
                "model_dim": dim,
                "hidden_dim": 3,
                "seed": 27602,
                "param_version": 1,
                "quota": 2,
                "batch_size": 2,
            },
            {
                "worker_id": "w-gamma",
                "host": "127.0.0.1",
                "port": 49623,
                "model_dim": dim,
                "hidden_dim": 3,
                "seed": 27603,
                "param_version": 1,
                "quota": 1,
                "batch_size": 2,
            },
        ],
    }
    requests = {
        "format": "request-stream/v1",
        "requests": [
            {
                "request_id": f"req-{index:03d}",
                "worker_id": ["w-alpha", "w-beta", "w-gamma"][index % 3],
                "arrival_seq": index,
                "input_vector": list(vectors[index]),
            }
            for index in range(6)
        ],
    }
    rules = {
        "format": "merge-rules/v1",
        "default_batch_size": 2,
        "global_concurrency": 2,
        "lease_ttl_ms": 5000,
        "lease_renew_ms": 1000,
        "digest_algorithm": "sha256-v1",
        "batch_id_serialization": "canonical JSON of {worker_id,batch_index,arrival_seqs}, SHA-256 prefix rb1_",
    }
    _write_json(target / "workers.json", workers)
    _write_json(target / "requests.json", requests)
    _write_json(target / "rules.json", rules)

    bundle = load_bundle(target)
    plan = build_plan(bundle)
    verify_coverage(plan, bundle.requests)
    worker_map = bundle.worker_map()
    request_map = bundle.request_map()
    results = {
        batch.batch_id: forward_batch(worker_map[batch.worker_id], batch, request_map, elapsed_ms=9 + batch.batch_index * 4)
        for batch in plan.batches
    }

    r1_lines: list[str] = []
    r2_lines: list[str] = []
    r3_lines: list[str] = []
    formats = ["r1", "r2", "r3"]
    for batch_index, batch in enumerate(plan.batches):
        result = results[batch.batch_id]
        for rank, request_id in enumerate(batch.request_ids):
            receipt = {
                "source_format_version": formats[min(batch_index, 2)],
                "batch_id": batch.batch_id,
                "request_id": request_id,
                "output_digest": result["output_digest"],
                "param_version": result["param_version"],
                "elapsed_ms": result["elapsed_ms"] + rank,
                "worker_id": batch.worker_id,
                "rank": rank,
                "supersedes_ref": "req-000" if batch_index == 2 and rank == 0 else None,
                "param_digest": result["param_digest"],
            }
            line = encode_receipt_line(receipt, formats[min(batch_index, 2)])
            if batch_index == 0:
                r1_lines.append(line)
            elif batch_index == 1:
                r2_lines.append(line)
            else:
                r3_lines.append(line)

    (target / "receipts_r1.jsonl").write_text("\n".join(r1_lines) + "\n", encoding="utf-8")
    (target / "receipts_r2.jsonl").write_text("\n".join(r2_lines) + "\n", encoding="utf-8")
    (target / "receipts_r3.jsonl").write_text("\n".join(r3_lines) + "\n", encoding="utf-8")
    manifest = {
        "fixed_seed": FIXED_SAMPLE_SEED,
        "loopback_only": True,
        "plan": [batch.to_json() for batch in plan.batches],
        "batch_ledger": [
            {
                "batch_id": batch.batch_id,
                "worker_id": batch.worker_id,
                "batch_index": batch.batch_index,
                "request_ids": list(batch.request_ids),
                "param_digest": results[batch.batch_id]["param_digest"],
                "output_digest": results[batch.batch_id]["output_digest"],
            }
            for batch in plan.batches
        ],
    }
    _write_json(target / "sample_manifest.json", manifest)
    return target
