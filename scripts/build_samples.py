"""Generate the deterministic local sample corpus."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inference_attribution.model import output_digest, parameter_digest
from inference_attribution.planner import build_plan
from inference_attribution.spec import parse_merge_rules, parse_requests, parse_workers


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    path.write_text(text, encoding="utf-8")


def worker_document(worker_set_version: int = 3) -> dict[str, object]:
    return {
        "schema_version": 1,
        "worker_set_version": worker_set_version,
        "workers": [
            {
                "worker_id": "worker-alpha",
                "host": "127.0.0.1",
                "port": 28431,
                "seed": 41001,
                "input_dim": 4,
                "hidden_dim": 5,
                "param_version_seq": 2,
                "batch_size": 2,
                "concurrency_quota": 1,
            },
            {
                "worker_id": "worker-beta",
                "host": "127.0.0.1",
                "port": 28432,
                "seed": 41002,
                "input_dim": 4,
                "hidden_dim": 3,
                "param_version_seq": 1,
                "batch_size": 2,
                "concurrency_quota": 1,
            },
        ],
    }


def request_document() -> dict[str, object]:
    rng = np.random.Generator(np.random.PCG64(90125))
    assignments = [
        ("req-001", 0, "worker-alpha"),
        ("req-002", 1, "worker-alpha"),
        ("req-003", 2, "worker-beta"),
        ("req-004", 3, "worker-alpha"),
        ("req-005", 4, "worker-beta"),
        ("req-006", 5, "worker-alpha"),
    ]
    requests = []
    for request_id, arrival_seq, worker_id in assignments:
        vector = np.round(rng.uniform(-1.0, 1.0, 4), 6)
        requests.append(
            {
                "request_id": request_id,
                "arrival_seq": arrival_seq,
                "worker_id": worker_id,
                "input": [float(value) for value in vector],
            }
        )
    return {"schema_version": 1, "requests": requests}


def merge_rules_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "global_concurrency": 3,
        "formats": {
            "r1": {
                "fill": {
                    "worker_id": "batch.worker_id",
                    "param_version_seq": "batch.param_version_seq",
                    "latency_ms": "constant:0",
                    "param_digest": "batch.param_digest",
                    "rank_in_batch": "request.arrival_rank",
                    "superseded_request_ids": "empty_list",
                }
            },
            "r2": {
                "fill": {
                    "param_digest": "batch.param_digest",
                    "rank_in_batch": "request.arrival_rank",
                    "superseded_request_ids": "empty_list",
                }
            },
            "r3": {"fill": {}},
        },
    }


def build_receipts() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    workers_doc = worker_document()
    requests_doc = request_document()
    workers = parse_workers(workers_doc, object_name="samples/workers.json")
    requests = parse_requests(
        requests_doc, workers, object_name="samples/requests.json"
    )
    plan = build_plan(workers, requests)
    worker_map = workers.by_id()
    request_map = requests.by_id()
    batches = {(batch.worker_id, batch.batch_seq): batch for batch in plan.batches}
    r1_batch = batches[("worker-alpha", 0)]
    r2_batch = batches[("worker-alpha", 1)]
    r3_batch = batches[("worker-beta", 0)]
    alpha = worker_map["worker-alpha"]
    beta = worker_map["worker-beta"]

    r1: list[dict[str, object]] = []
    for rank, request_id in enumerate(r1_batch.request_ids):
        r1.append(
            {
                "batch_id": r1_batch.batch_id,
                "request_id": request_id,
                "output_digest": output_digest(
                    alpha, request_map[request_id].input
                ),
            }
        )

    alpha_digest = parameter_digest(alpha)
    r2: list[dict[str, object]] = []
    for rank, request_id in enumerate(r2_batch.request_ids):
        r2.append(
            {
                "batch_id": r2_batch.batch_id,
                "request_id": request_id,
                "output_digest": output_digest(
                    alpha, request_map[request_id].input
                ),
                "param_version_seq": alpha.param_version_seq,
                "latency_ms": 11 + rank,
                "worker_id": alpha.worker_id,
            }
        )

    beta_digest = parameter_digest(beta)
    r3: list[dict[str, object]] = []
    for rank, request_id in enumerate(r3_batch.request_ids):
        r3.append(
            {
                "batch_id": r3_batch.batch_id,
                "request_id": request_id,
                "output_digest": output_digest(
                    beta, request_map[request_id].input
                ),
                "param_version_seq": beta.param_version_seq,
                "latency_ms": 17 + rank,
                "worker_id": beta.worker_id,
                "rank_in_batch": rank,
                "superseded_request_ids": ["req-003"] if rank == 1 else [],
                "param_digest": beta_digest,
            }
        )
    if alpha_digest == beta_digest:
        raise RuntimeError("sample workers unexpectedly share a parameter digest")
    return r1, r2, r3


def main() -> None:
    workers = worker_document()
    requests = request_document()
    rules = merge_rules_document()
    write_json(SAMPLES / "workers.json", workers)
    write_json(SAMPLES / "requests.json", requests)
    write_json(SAMPLES / "merge_rules.json", rules)

    updated = worker_document(worker_set_version=4)
    updated["workers"][0]["param_version_seq"] = 3
    write_json(SAMPLES / "workers_param_update.json", updated)

    repartitioned = worker_document(worker_set_version=5)
    repartitioned["workers"][1]["batch_size"] = 1
    write_json(SAMPLES / "workers_repartition.json", repartitioned)

    r1, r2, r3 = build_receipts()
    write_jsonl(SAMPLES / "receipts_r1.jsonl", r1)
    write_jsonl(SAMPLES / "receipts_r2.jsonl", r2)
    write_jsonl(SAMPLES / "receipts_r3.jsonl", r3)
    print("samples generated")


if __name__ == "__main__":
    main()
