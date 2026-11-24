from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .fixedpoint import deterministic_forward
from .hashing import sha256_json
from .model import receipt_ref_for
from .state import new_state

FIXED_SEED = 20261125


def _digest(request_number: int, config: dict[str, Any]) -> str:
    result = deterministic_forward(
        seed=int(config.get("seed", FIXED_SEED)),
        arrival_seq=request_number,
        model_dim=int(config["model_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        quant_bits=int(config.get("quant_bits", 8)),
    )
    return result.digest


def _param_digest(worker_id: str, version: int, config: dict[str, Any]) -> str:
    return "param:" + sha256_json(
        {
            "seed": config.get("seed", FIXED_SEED),
            "worker_id": worker_id,
            "param_version": version,
            "model_dim": config["model_dim"],
            "hidden_dim": config["hidden_dim"],
            "quant_bits": config.get("quant_bits", 8),
        }
    ).split(":", 1)[1][:24]


def make_sample_state() -> dict[str, Any]:
    state = new_state()
    config = state["config"]
    config["seed"] = FIXED_SEED
    config["batch_size"] = 4

    version_map = {"W1": 11, "W2": 21, "W3": 31}
    state["baseline_worker_versions"] = dict(version_map)
    state["workers"] = [
        {
            "worker_id": "W1",
            "host": "127.0.0.1",
            "port": 29101,
            "param_version": 11,
            "model_dim": config["model_dim"],
            "hidden_dim": config["hidden_dim"],
        },
        {
            "worker_id": "W2",
            "host": "127.0.0.1",
            "port": 29102,
            "param_version": 21,
            "model_dim": config["model_dim"],
            "hidden_dim": config["hidden_dim"],
        },
        {
            "worker_id": "W3",
            "host": "127.0.0.1",
            "port": 29103,
            "param_version": 31,
            "model_dim": config["model_dim"],
            "hidden_dim": config["hidden_dim"],
        },
        {
            "worker_id": "W-BAD-PORT",
            "host": "127.0.0.1",
            "port": 70000,
            "param_version": 1,
            "model_dim": config["model_dim"],
            "hidden_dim": config["hidden_dim"],
        },
    ]

    duplicate_ids = {29: "Q0005", 30: "Q0009", 31: "Q0017", 32: "Q0021"}
    requests: list[dict[str, Any]] = []
    for seq in range(1, 33):
        request_id = duplicate_ids.get(seq, f"Q{seq:04d}")
        requests.append(
            {
                "seq": seq,
                "request_id": request_id,
                "arrival_ms": (seq - 1) * 7 + ((seq - 1) // 4) * 3,
            }
        )
    requests.append(
        {
            "seq": 33,
            "request_id": "Q9000",
            "arrival_ms": 1,
            "_source": "requests_invalid.jsonl",
        }
    )
    state["requests"] = requests

    receipts: list[dict[str, Any]] = []
    record_counts = {"r1": 0, "r2": 0, "r3": 0}

    for batch_index in range(1, 9):
        start = (batch_index - 1) * 4
        members = requests[start : start + 4]
        worker_id = f"W{((batch_index - 1) % 3) + 1}"
        param_version = version_map[worker_id]
        schema_version = "r1" if batch_index <= 2 else "r2" if batch_index <= 5 else "r3"
        elapsed_ms = 12 + batch_index * 4

        for rank, request in enumerate(members):
            request_id = request["request_id"]
            request_number = int(request_id[1:])
            digest = _digest(request_number, config)
            doc: dict[str, Any] = {
                "v": schema_version,
                "batch_id": f"B-{batch_index:04d}",
                "request_id": request_id,
                "output_digest": digest,
            }
            if schema_version in ("r2", "r3"):
                doc.update(
                    {
                        "param_version": param_version,
                        "elapsed_ms": elapsed_ms + rank,
                        "worker_id": worker_id,
                    }
                )
            if schema_version == "r3":
                supersedes = "$missing"
                if batch_index == 8 and rank == 0:
                    loser_digest = _digest(5, config)
                    supersedes = receipt_ref_for("B-0002", "Q0005", loser_digest)
                doc.update(
                    {
                        "rank_in_batch": rank,
                        "supersedes_ref": supersedes,
                        "param_digest": _param_digest(worker_id, param_version, config),
                    }
                )
            doc["_source"] = f"receipts/{schema_version}.jsonl"
            doc["_record_index"] = record_counts[schema_version]
            record_counts[schema_version] += 1
            receipts.append(doc)

    unknown_worker_receipt = {
        "v": "r2",
        "batch_id": "B-9001",
        "request_id": "Q9001",
        "output_digest": _digest(9001, config),
        "param_version": 1,
        "elapsed_ms": 5,
        "worker_id": "W999",
        "_source": "receipts/invalid_worker.jsonl",
        "_record_index": 0,
    }
    bad_version_receipt = {
        "v": "r9",
        "batch_id": "B-9002",
        "request_id": "Q9002",
        "output_digest": "sha256:bad-version-sample",
        "_source": "receipts/bad_version.jsonl",
        "_record_index": 0,
    }
    receipts.extend([unknown_worker_receipt, bad_version_receipt])
    state["receipts"] = receipts

    target = next(
        item
        for item in receipts
        if item.get("batch_id") == "B-0007" and item.get("request_id") == "Q0025"
    )
    replacement = {key: value for key, value in target.items() if not key.startswith("_")}
    replacement["elapsed_ms"] = 77
    replacement["param_digest"] = _param_digest("W1", 11, config)

    new_batch_requests = [
        {"seq": 101 + offset, "request_id": f"Q{101 + offset:04d}", "arrival_ms": 600 + offset * 9}
        for offset in range(4)
    ]
    new_batch_receipts = []
    for rank, request in enumerate(new_batch_requests):
        number = int(request["request_id"][1:])
        new_batch_receipts.append(
            {
                "v": "r3",
                "batch_id": "B-9001",
                "request_id": request["request_id"],
                "output_digest": _digest(number, config),
                "param_version": 22,
                "elapsed_ms": 19 + rank,
                "worker_id": "W2",
                "rank_in_batch": rank,
                "supersedes_ref": "$missing",
                "param_digest": _param_digest("W2", 22, config),
            }
        )

    state["command_stream"] = [
        {
            "type": "replace_receipt",
            "match": {"batch_id": "B-0007", "request_id": "Q0025"},
            "receipt": replacement,
        },
        {"type": "set_batch_size", "batch_size": 3},
        {
            "type": "switch_param_version",
            "worker_id": "W2",
            "param_version": 22,
            "param_digest": _param_digest("W2", 22, config),
        },
        {
            "type": "add_batch",
            "batch": {
                "batch_id": "B-9001",
                "requests": new_batch_requests,
                "receipts": new_batch_receipts,
            },
        },
        {"type": "set_merge_rule", "merge_rule": "version_then_batch_v1"},
    ]
    state["negative_commands"] = [{"type": "set_batch_size", "batch_size": 0}]
    return state


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]], *, strip_private: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in records:
        item = {k: v for k, v in record.items() if not strip_private or not k.startswith("_")}
        lines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_sample_workspace(root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    state = make_sample_state()

    _write_json(root / "config.json", state["config"])
    _write_json(root / "workers.json", state["workers"])
    _write_json(root / "baseline_worker_versions.json", state["baseline_worker_versions"])
    _write_json(root / "commands.json", state["command_stream"])
    _write_json(root / "sidecar.json", state["sidecar"])

    valid_requests = [item for item in state["requests"] if item.get("_source") != "requests_invalid.jsonl"]
    invalid_requests = [item for item in state["requests"] if item.get("_source") == "requests_invalid.jsonl"]
    _write_jsonl(root / "requests.jsonl", valid_requests)
    _write_jsonl(root / "requests_invalid.jsonl", invalid_requests)

    by_version: dict[str, list[dict[str, Any]]] = {"r1": [], "r2": [], "r3": []}
    invalid_files: dict[str, list[dict[str, Any]]] = {}
    for receipt in state["receipts"]:
        source = receipt.get("_source", "")
        if source.startswith("receipts/") and source.endswith(".jsonl"):
            name = Path(source).name
            if name in ("r1.jsonl", "r2.jsonl", "r3.jsonl"):
                by_version[name[:-6]].append(receipt)
            else:
                invalid_files.setdefault(name, []).append(receipt)
    for version, records in by_version.items():
        _write_jsonl(root / "receipts" / f"{version}.jsonl", records)
    for name, records in invalid_files.items():
        _write_jsonl(root / "receipts" / name, records)

    _write_json(root / "workspace_state.generated.json", state)
    return root
