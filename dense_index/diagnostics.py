from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .commands import CommandState
from .config import validate_config
from .corpus import load_corpus_file, load_vocabulary
from .engine import IndexEngine
from .quant import bucket_key_from_vector, diagnose_pair, quantize_vector, routing_tie_flip
from .receipts import load_receipt_file
from .sample import base_config
from .snapshot import SnapshotStore
from .errors import DiagnosticError


INJECTION_SPECS: List[Dict[str, Any]] = [
    {
        "id": "FI-001",
        "kind": "corpus_invalid",
        "title": "unknown term, duplicate paragraph id, empty document id and truncated JSONL line",
        "expected_code": "E_CORPUS_INVALID",
        "config": {"corpus_fixture": "fixtures/corpus_invalid.jsonl", "vocabulary": "config/vocabulary.json"},
    },
    {
        "id": "FI-002",
        "kind": "overlap_not_smaller_than_window",
        "title": "overlap length is not smaller than chunk window",
        "expected_code": "E_CONFIG_INVALID",
        "config": {"chunk_window": 8, "chunk_overlap": 10},
    },
    {
        "id": "FI-003",
        "kind": "bucket_bits_out_of_range",
        "title": "bucket_bits exceeds supported range and projection dimension",
        "expected_code": "E_CONFIG_INVALID",
        "config": {"bucket_bits": 99},
    },
    {
        "id": "FI-004",
        "kind": "receipt_truncated",
        "title": "truncate a v1 receipt file in the middle of JSON",
        "expected_code": "E_RECEIPT_INVALID",
        "config": {"source_receipt": "receipts/receipts_v1.json", "truncate_to_bytes": 48},
    },
    {
        "id": "FI-005",
        "kind": "nonfinite_vector",
        "title": "inject a NaN projection component into one isolated block task",
        "expected_code": "E_VECTOR_NONFINITE",
        "config": {"target": "first_block"},
    },
    {
        "id": "FI-006",
        "kind": "snapshot_truncated",
        "title": "truncate newest manifest and leave an interrupted staging fragment",
        "expected_code": "E_SNAPSHOT_RECOVERY",
        "config": {"committed_versions": 2},
    },
    {
        "id": "FI-007",
        "kind": "boundary_flip_quantization",
        "title": "coarse truncation and one-bit-wider reference cross threshold differently",
        "expected_code": "E_BOUNDARY_FLIP",
        "config": {"quant_level": "Q8", "fractional_bits": 4, "threshold": 0.90},
    },
    {
        "id": "FI-008",
        "kind": "boundary_flip_tie_rule",
        "title": "zero-sign tie rule changes query bucket routing",
        "expected_code": "E_BOUNDARY_FLIP",
        "config": {"bucket_bits": 2},
    },
    {
        "id": "FI-009",
        "kind": "point_budget_pending",
        "title": "two-point projection budget preserves stable-order pending work",
        "expected_code": "E_POINT_BUDGET_PENDING",
        "config": {"budget": 2},
    },
]


def write_injection_configs(root: str | Path) -> List[Path]:
    root = Path(root)
    directory = root / "config" / "injections"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    index = {"schema": "fault-injection-index/v1", "injections": []}
    for spec in INJECTION_SPECS:
        path = directory / f"{spec['id'].lower()}_{spec['kind']}.yaml"
        payload = {"injection_id": spec["id"], "kind": spec["kind"], "expected_code": spec["expected_code"], **spec["config"]}
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), encoding="utf-8", newline="\n")
        paths.append(path)
        index["injections"].append({"id": spec["id"], "path": str(path.relative_to(root)), "expected_code": spec["expected_code"]})
    (directory / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return paths


def _item(spec: Dict[str, Any], hit: bool, actual_code: str | None, detail: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "injection_id": spec["id"],
        "kind": spec["kind"],
        "title": spec["title"],
        "expected_code": spec["expected_code"],
        "actual_code": actual_code,
        "hit": bool(hit),
        "detail": detail,
    }


def run_fault_injections(root: str | Path) -> Dict[str, Any]:
    root = Path(root)
    write_injection_configs(root)
    output_root = root / "outputs" / "injections"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []

    for spec in INJECTION_SPECS:
        try:
            if spec["kind"] == "corpus_invalid":
                vocab = load_vocabulary(root / "config" / "vocabulary.json")
                try:
                    load_corpus_file(root / spec["config"]["corpus_fixture"], vocab)
                    items.append(_item(spec, False, None, {"error": "validation did not raise"}))
                except Exception as exc:
                    code = getattr(exc, "code", None)
                    failures = getattr(exc, "failures", [])
                    items.append(_item(spec, code == spec["expected_code"], code, {"failure_count": len(failures), "first": failures[0].to_dict() if failures else None}))

            elif spec["kind"] in {"overlap_not_smaller_than_window", "bucket_bits_out_of_range"}:
                data = base_config()
                data.update(spec["config"])
                try:
                    validate_config(data)
                    items.append(_item(spec, False, None, {"error": "validation did not raise"}))
                except Exception as exc:
                    code = getattr(exc, "code", None)
                    items.append(_item(spec, code == spec["expected_code"], code, {"failure_count": len(getattr(exc, "failures", []))}))

            elif spec["kind"] == "receipt_truncated":
                source = root / spec["config"]["source_receipt"]
                target = output_root / spec["id"] / "truncated_receipt.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                text = source.read_text(encoding="utf-8")[: int(spec["config"]["truncate_to_bytes"])]
                target.write_text(text, encoding="utf-8", newline="")
                try:
                    load_receipt_file(target, forced_version="v1")
                    items.append(_item(spec, False, None, {"error": "receipt parse did not raise"}))
                except Exception as exc:
                    code = getattr(exc, "code", None)
                    items.append(_item(spec, code == spec["expected_code"], code, {"truncated_chars": len(text), "message": str(exc)}))

            elif spec["kind"] == "nonfinite_vector":
                state = CommandState.from_sample_dir(root)
                blocks = state.blocks
                target_id = blocks[0]["block_id"]
                engine = IndexEngine(state.config)
                injection = {"id": spec["id"], "kind": "nonfinite_vector", "target_block_id": target_id}
                try:
                    engine.run_batch(blocks, injection=injection)
                    items.append(_item(spec, False, None, {"target_block_id": target_id}))
                except Exception as exc:
                    code = getattr(exc, "code", None)
                    items.append(_item(spec, code == spec["expected_code"], code, {"target_block_id": target_id, "injection_id": getattr(exc, "injection_id", None)}))

            elif spec["kind"] == "snapshot_truncated":
                store_root = output_root / spec["id"] / "snapshots"
                store = SnapshotStore(store_root)
                common = {"config_hash": "fixture-config-hash", "param_fingerprint": "fixture-param", "quant_level": "Q16"}
                store.commit({**common, "completed_blocks": ["B1"], "cache": {"n": 1}})
                store.commit({**common, "completed_blocks": ["B1", "B2"], "cache": {"n": 2}})
                truncated_path = store.truncate_current_manifest(24)
                staging = store.create_staging_fragment()
                recovered = store.load_current()
                hit = bool(recovered and recovered["manifest"]["global_version"] == 1)
                items.append(
                    _item(
                        spec,
                        hit,
                        spec["expected_code"] if hit else None,
                        {
                            "truncated_manifest": truncated_path,
                            "interrupted_stage": str(staging),
                            "recovered_global_version": recovered["manifest"]["global_version"] if recovered else None,
                            "recovery_rule": "newest_valid_committed_manifest_and_config_version",
                        },
                    )
                )

            elif spec["kind"] == "boundary_flip_quantization":
                config = validate_config({**base_config(), **spec["config"], "bucket_bits": 2})
                width = 32
                query_raw = [1.0] + [0.0] * (width - 1)
                block_raw = [1.0, 0.49] + [0.0] * (width - 2)
                query_quant = quantize_vector(query_raw, config)
                block_quant = quantize_vector(block_raw, config)
                diagnosis = diagnose_pair(
                    query_raw,
                    query_quant["fixed_vector"],
                    block_raw,
                    block_quant["fixed_vector"],
                    "BLOCK-QUANT-FLIP",
                    config,
                    threshold=float(spec["config"]["threshold"]),
                    query_id="QUERY-QUANT-FLIP",
                )
                hit = bool(diagnosis["flip"] and diagnosis["cause"] == "quantization_truncation")
                items.append(_item(spec, hit, diagnosis["code"], diagnosis))

            elif spec["kind"] == "boundary_flip_tie_rule":
                config = validate_config({**base_config(), "bucket_bits": 2})
                width = int(config.data["projection_dim"])
                query_raw = [0.0, 0.0] + [0.0] * (width - 2)
                block_raw = [1.0, 1.0] + [0.0] * (width - 2)
                query_quant = quantize_vector(query_raw, config)
                block_quant = quantize_vector(block_raw, config)
                current_query = bucket_key_from_vector(query_raw, query_quant["fixed_vector"], config)
                current_block = bucket_key_from_vector(block_raw, block_quant["fixed_vector"], config)
                alternative_query = bucket_key_from_vector(query_raw, query_quant["fixed_vector"], config, alternative_tie_rule="negative")
                alternative_block = bucket_key_from_vector(block_raw, block_quant["fixed_vector"], config, alternative_tie_rule="negative")
                routing = routing_tie_flip(current_query, current_block, alternative_query, alternative_block)
                hit = bool(routing["flip"] and routing["cause"] == "tie_rule")
                items.append(_item(spec, hit, routing["code"], routing))

            elif spec["kind"] == "point_budget_pending":
                state = CommandState.from_sample_dir(root)
                engine = IndexEngine(state.config)
                report = engine.run_batch(state.blocks, budget=int(spec["config"]["budget"]))
                expected_first_two = [block["block_id"] for block in sorted(state.blocks, key=lambda item: (item["document_id"], item["block_seq"]))[:2]]
                hit = report["status"] == spec["expected_code"] and report["scheduled_ids"] == expected_first_two and len(report["pending_ids"]) == len(state.blocks) - 2
                items.append(_item(spec, hit, report["status"], {"scheduled_ids": report["scheduled_ids"], "pending_count": len(report["pending_ids"]), "expected_first_two": expected_first_two}))

        except Exception as exc:
            items.append(_item(spec, False, getattr(exc, "code", type(exc).__name__), {"fatal": str(exc)}))

    report = {
        "schema": "fault-injection-report/v1",
        "all_hit": all(item["hit"] for item in items),
        "hit_count": sum(item["hit"] for item in items),
        "total": len(items),
        "items": items,
    }
    path = root / "outputs" / "fault_injection_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return report


def build_failure_manifest(root: str | Path) -> Dict[str, Any]:
    root = Path(root)
    cases: List[Dict[str, Any]] = []

    def record_case(case_id: str, expected_code: str, callback) -> None:
        try:
            callback()
            cases.append({"case_id": case_id, "expected_code": expected_code, "hit": False, "failures": []})
        except DiagnosticError as exc:
            cases.append(
                {
                    "case_id": case_id,
                    "expected_code": expected_code,
                    "actual_code": exc.code,
                    "hit": exc.code == expected_code,
                    "message": exc.message,
                    "failures": [failure.to_dict() for failure in exc.failures],
                }
            )

    vocab = load_vocabulary(root / "config" / "vocabulary.json")
    record_case(
        "corpus_jsonl",
        "E_CORPUS_INVALID",
        lambda: load_corpus_file(root / "fixtures" / "corpus_invalid.jsonl", vocab),
    )
    record_case(
        "config_overlap",
        "E_CONFIG_INVALID",
        lambda: validate_config({**base_config(), "chunk_window": 8, "chunk_overlap": 9}),
    )
    record_case(
        "config_bucket_bits",
        "E_CONFIG_INVALID",
        lambda: validate_config({**base_config(), "bucket_bits": 32}),
    )
    record_case(
        "receipt_v1",
        "E_RECEIPT_INVALID",
        lambda: load_receipt_file(root / "fixtures" / "receipt_invalid_v1.json", forced_version="v1"),
    )

    manifest = {"schema": "input-failure-manifest/v1", "all_hit": all(item["hit"] for item in cases), "cases": cases}
    path = root / "outputs" / "failure_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest
