"""按固定整数种子生成全部本地样本与异常注入配置。

运行：python tools/generate_samples.py
产物：samples/ 与 samples_faulty/，全部确定性、无网络依赖。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil

SEED = 20241125
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")
FAULTY = os.path.join(ROOT, "samples_faulty")


def digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def receipt_ref(batch_id: str, request_id: str, output_digest: str, integrity: str) -> str:
    identity = [batch_id, request_id, output_digest, integrity]
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_requests(rng: random.Random) -> list[dict]:
    rows = []
    arrival = 0
    for index in range(1, 13):
        arrival += rng.randint(5, 40)
        rows.append(
            {
                "seq": index,
                "request_id": f"req-{index:03d}",
                "arrival_ms": arrival,
                "token_count": rng.randint(8, 64),
            }
        )
    return rows


def build_workers() -> dict:
    return {
        "batch_size": 4,
        "workers": [
            {
                "worker_id": "worker-a",
                "port": 9101,
                "model_dim": 64,
                "hidden_dim": 128,
                "param_version": 3,
                "quant": "int8",
            },
            {
                "worker_id": "worker-b",
                "port": 9102,
                "model_dim": 48,
                "hidden_dim": 96,
                "param_version": 2,
                "quant": "int4",
            },
        ],
    }


def build_rules() -> dict:
    return {
        "rules": [
            {"rule_id": "rule-main", "worker_ids": ["worker-a", "worker-b"], "batch_size": 4},
            {"rule_id": "rule-a-only", "worker_ids": ["worker-a"], "batch_size": 2},
        ]
    }


def build_receipts(rows: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    """生成跨三版本、跨批次、含重复投递与覆盖引用的回执。"""

    assignments = []
    for index, row in enumerate(rows):
        primary_batch = f"batch-{index // 4 + 1:02d}"
        worker = "worker-a" if index % 2 == 0 else "worker-b"
        assignments.append((row, primary_batch, worker, 3 if worker == "worker-a" else 2))

    r1_rows: list[dict] = []
    r2_rows: list[dict] = []
    r3_rows: list[dict] = []

    for idx, (row, batch_id, worker, version) in enumerate(assignments[:4]):
        r1_rows.append(
            {
                "version": "r1",
                "batch_id": batch_id,
                "request_id": row["request_id"],
                "output_digest": digest(f"{row['request_id']}|{batch_id}|r1"),
            }
        )

    for idx, (row, batch_id, worker, version) in enumerate(assignments[4:8]):
        r2_rows.append(
            {
                "version": "r2",
                "batch_id": batch_id,
                "request_id": row["request_id"],
                "output_digest": digest(f"{row['request_id']}|{batch_id}|r2"),
                "param_version": version,
                "duration_ms": rng.randint(20, 120),
                "worker_id": worker,
            }
        )

    duplicate_target = assignments[8][0]["request_id"]
    for idx, (row, batch_id, worker, version) in enumerate(assignments[8:]):
        r3_rows.append(
            {
                "version": "r3",
                "batch_id": batch_id,
                "request_id": row["request_id"],
                "output_digest": digest(f"{row['request_id']}|{batch_id}|r3"),
                "param_version": version,
                "duration_ms": rng.randint(20, 120),
                "worker_id": worker,
                "rank_in_batch": idx % 4,
                "supersedes": [],
                "param_summary": f"params-v{version}-{worker}",
            }
        )

    old_dup = {
        "version": "r3",
        "batch_id": "batch-09",
        "request_id": duplicate_target,
        "output_digest": digest(f"{duplicate_target}|batch-09|old"),
        "param_version": 1,
        "duration_ms": 150,
        "worker_id": "worker-a",
        "rank_in_batch": 0,
        "supersedes": [],
        "param_summary": "params-v1-worker-a",
    }
    r3_rows.append(old_dup)
    old_tag = "sha256:" + hashlib.sha256(
        json.dumps(
            {key: value for key, value in old_dup.items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    old_ref = receipt_ref(
        old_dup["batch_id"], old_dup["request_id"], old_dup["output_digest"], old_tag
    )
    r3_rows.append(
        {
            "version": "r3",
            "batch_id": "batch-10",
            "request_id": duplicate_target,
            "output_digest": digest(f"{duplicate_target}|batch-10|new"),
            "param_version": 3,
            "duration_ms": 60,
            "worker_id": "worker-a",
            "rank_in_batch": 1,
            "supersedes": [old_ref],
            "param_summary": "params-v3-worker-a",
        }
    )
    return {"r1": r1_rows, "r2": r2_rows, "r3": r3_rows}


def build_commands(receipts: dict[str, list[dict]]) -> dict:
    first_r3 = next(row for row in receipts["r3"] if row["request_id"] == "req-010")
    tag_payload = {key: value for key, value in first_r3.items()}
    tag = "sha256:" + hashlib.sha256(
        json.dumps(tag_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    target_ref = receipt_ref(
        first_r3["batch_id"], first_r3["request_id"], first_r3["output_digest"], tag
    )
    return {
        "commands": [
            {
                "type": "switch_param_version",
                "worker_id": "worker-b",
                "param_version": 5,
                "cursor": 1,
            },
            {
                "type": "replace_receipt",
                "file": "receipt_r2.jsonl",
                "line": 2,
                "version": "r2",
                "payload": {
                    "version": "r2",
                    "batch_id": "batch-02",
                    "request_id": "req-006",
                    "output_digest": digest("req-006|batch-02|r2-replaced"),
                    "param_version": 5,
                    "duration_ms": 44,
                    "worker_id": "worker-b",
                },
                "cursor": 2,
            },
            {
                "type": "add_batch",
                "payload": {
                    "version": "r3",
                    "batch_id": "batch-11",
                    "request_id": "req-012",
                    "output_digest": digest("req-012|batch-11|r3"),
                    "param_version": 3,
                    "duration_ms": 77,
                    "worker_id": "worker-a",
                    "rank_in_batch": 0,
                    "supersedes": [],
                    "param_summary": "params-v3-worker-a",
                },
                "cursor": 3,
            },
            {"type": "adjust_batch_size", "batch_size": 6, "cursor": 4},
            {
                "type": "modify_merge_rule",
                "rule_id": "rule-a-only",
                "worker_ids": ["worker-a", "worker-b"],
                "batch_size": 6,
                "cursor": 5,
            },
            {"type": "convert_receipt", "target": "r3", "cursor": 6},
            {
                "type": "convert_receipt",
                "target": "r1",
                "receipt_ref": target_ref,
                "cursor": 7,
            },
            {
                "type": "convert_receipt",
                "target": "r3",
                "receipt_ref": target_ref,
                "cursor": 8,
            },
        ]
    }


def build_injection() -> dict:
    return {
        "injections": [
            {
                "id": "INJ-001",
                "type": "non_finite_duration",
                "file": "receipt_r2.jsonl",
                "line": 1,
                "description": "将 r2 首行耗时替换为 NaN",
            },
            {
                "id": "INJ-002",
                "type": "unknown_param_version",
                "worker_id": "worker-a",
                "param_version": 99,
                "file": "receipt_r3.jsonl",
                "line": 1,
                "description": "r3 首行声明工作器不存在的参数版本 v99",
            },
            {
                "id": "INJ-003",
                "type": "truncated_line",
                "file": "receipt_r3.jsonl",
                "line": 3,
                "description": "截断 r3 第三行为非法 JSON",
            },
            {
                "id": "INJ-004",
                "type": "duplicate_request_id",
                "request_id": "req-001",
                "arrival_ms": 999,
                "description": "重放已存在的请求标识 req-001",
            },
        ]
    }


def materialize_faulty(base: str, faulty: str) -> None:
    shutil.copytree(base, faulty, dirs_exist_ok=True)
    r2_path = os.path.join(faulty, "receipts", "receipt_r2.jsonl")
    with open(r2_path, encoding="utf-8") as handle:
        r2_lines = handle.readlines()
    row = json.loads(r2_lines[0])
    row["duration_ms"] = "NaN"
    r2_lines[0] = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with open(r2_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(r2_lines)

    r3_path = os.path.join(faulty, "receipts", "receipt_r3.jsonl")
    with open(r3_path, encoding="utf-8") as handle:
        r3_lines = handle.readlines()
    row = json.loads(r3_lines[0])
    row["param_version"] = 99
    r3_lines[0] = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    r3_lines[2] = '{"version": "r3", "batch_id": "batch-03", "request_id":\n'
    with open(r3_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(r3_lines)


def main() -> None:
    rng = random.Random(SEED)
    for directory in (SAMPLES, FAULTY):
        if os.path.exists(directory):
            shutil.rmtree(directory)

    requests = build_requests(rng)
    write_jsonl(os.path.join(SAMPLES, "requests.jsonl"), requests)
    write_json(os.path.join(SAMPLES, "workers.json"), build_workers())
    write_json(os.path.join(SAMPLES, "merge_rules.json"), build_rules())
    receipts = build_receipts(requests, rng)
    write_jsonl(os.path.join(SAMPLES, "receipts", "receipt_r1.jsonl"), receipts["r1"])
    write_jsonl(os.path.join(SAMPLES, "receipts", "receipt_r2.jsonl"), receipts["r2"])
    write_jsonl(os.path.join(SAMPLES, "receipts", "receipt_r3.jsonl"), receipts["r3"])
    write_json(os.path.join(SAMPLES, "commands.json"), build_commands(receipts))

    materialize_faulty(SAMPLES, FAULTY)
    write_json(os.path.join(FAULTY, "injection.json"), build_injection())
    print(
        json.dumps(
            {
                "seed": SEED,
                "samples": os.path.relpath(SAMPLES, ROOT),
                "samples_faulty": os.path.relpath(FAULTY, ROOT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
