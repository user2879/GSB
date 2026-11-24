"""Deterministic local sample generation (fixed integer seeds)."""

import json
from pathlib import Path

from .model import build_parameters, param_digest, run_batch
from .planner import build_plan
from .spec import load_spec

MODEL_DIM = 4
BASE_SEED = 20261125


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def generate(samples_dir):
    samples_dir = Path(samples_dir)
    workers = [
        {"worker_id": "w-alpha", "model_dim": MODEL_DIM, "hidden_dim": 3,
         "seed": BASE_SEED + 1, "param_version": 1, "host": "127.0.0.1",
         "port": 39101, "batch_size": 2, "quota": 2},
        {"worker_id": "w-bravo", "model_dim": MODEL_DIM, "hidden_dim": 4,
         "seed": BASE_SEED + 2, "param_version": 1, "host": "127.0.0.1",
         "port": 39102, "batch_size": 3, "quota": 1},
        {"worker_id": "w-gamma", "model_dim": MODEL_DIM, "hidden_dim": 2,
         "seed": BASE_SEED + 3, "param_version": 1, "host": "127.0.0.1",
         "port": 39103, "batch_size": 2, "quota": 1},
    ]
    rules = {"global_concurrency": 2, "socket_timeout_ms": 800}

    requests = []
    for index in range(12):
        vector = [
            (BASE_SEED + index * 7 + component * 13) % 101 - 50
            for component in range(MODEL_DIM)
        ]
        requests.append({
            "request_id": f"req-{index:03d}",
            "arrival": 100 + index,
            "vector": vector,
        })

    _write(samples_dir / "requests.json", requests)
    _write(samples_dir / "workers.json", workers)
    _write(samples_dir / "merge_rules.json", rules)

    spec = load_spec(samples_dir / "requests.json",
                     samples_dir / "workers.json",
                     samples_dir / "merge_rules.json")
    plan = build_plan(spec)

    r1, r2, r3 = [], [], []
    duplicate_map = {"req-005": "req-001"}
    for batch in plan.batches:
        worker = spec.worker(batch.worker_id)
        params = build_parameters(worker)
        result = run_batch(worker, batch, params)
        digest_snapshot = param_digest(worker, params)
        for rank, request_id in enumerate(batch.requests):
            base = {
                "batch_id": batch.batch_id,
                "request_id": request_id,
                "output_digest": result.output_digest,
            }
            r1.append(dict(base, format="r1"))
            r2.append(dict(base, format="r2",
                           param_version=worker.param_version,
                           duration_ms=12 + rank * 3,
                           worker_id=worker.worker_id))
            r3.append(dict(base, format="r3",
                           param_version=worker.param_version,
                           duration_ms=12 + rank * 3,
                           worker_id=worker.worker_id,
                           rank=rank,
                           duplicate_ref=duplicate_map.get(request_id),
                           param_digest=digest_snapshot))

    _write(samples_dir / "receipts_r1.json", r1)
    _write(samples_dir / "receipts_r2.json", r2)
    _write(samples_dir / "receipts_r3.json", r3)

    bad_spec = [dict(raw) for raw in requests]
    bad_spec[3]["arrival"] = -4
    _write(samples_dir / "bad_requests_negative_arrival.json", bad_spec)

    bad_r3 = json.loads(json.dumps(r3))
    bad_r3[0]["duration_ms"] = -9
    _write(samples_dir / "bad_receipts_field.json", bad_r3)

    bad_chain = json.loads(json.dumps(r3))
    bad_chain[1]["duplicate_ref"] = "req-ghost"
    _write(samples_dir / "bad_receipts_chain.json", bad_chain)

    return {
        "requests": len(requests),
        "workers": len(workers),
        "batches": len(plan.batches),
        "r1": len(r1),
        "r2": len(r2),
        "r3": len(r3),
        "samples_dir": str(samples_dir),
    }


def generate_changed_workers(samples_dir, out_path,
                             bump_worker="w-bravo", new_version=2,
                             new_seed=None):
    """Emit a modified worker declaration for invalidation demonstrations."""
    data = json.loads(Path(samples_dir, "workers.json").read_text())
    for worker in data:
        if worker["worker_id"] == bump_worker:
            worker["param_version"] = new_version
            if new_seed is not None:
                worker["seed"] = new_seed
    _write(out_path, data)
    return out_path
