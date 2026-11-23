"""Deterministic local sample-data generation for the demo and tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .config import FingerprintConfig, validate_config_payload
from .features import normalize_and_window
from .hashing import compute_minhash, pair_similarity, signed_projection_bucket
from .text_store import TextRecord


VOCABULARY = [
    "甲",
    "乙",
    "丙",
    "丁",
    "戊",
    "己",
    "庚",
    "辛",
    "壬",
    "癸",
    "春",
    "夏",
    "秋",
    "冬",
    "东",
    "南",
    "西",
    "北",
    "天",
    "地",
    "人",
    "和",
    "风",
    "云",
]


BASE_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "hash_seed": "uidfc5909d3-v1",
    "window_count": 4,
    "signature_dimension": 32,
    "similarity_threshold": 0.75,
    "bucket_bits": 10,
    "fixed_decimal_places": 4,
    "quantization_levels": 1000,
    "signature_batch_budget": 2048,
    "max_workers": 4,
    "snapshot_commit_interval": 3,
    "source_labels": ["forum", "wiki", "news"],
    "vocabulary": VOCABULARY,
}


def _record(
    text_id: str,
    terms: list[str],
    source: str,
    version: int = 1,
) -> dict[str, Any]:
    return {"text_id": text_id, "terms": terms, "source": source, "version": version}


def build_initial_payloads() -> list[dict[str, Any]]:
    all_jia = ["甲"] * 12
    all_yi = ["乙"] * 12
    all_bing = ["丙"] * 12
    all_ding = ["丁"] * 12
    return [
        _record("T001", all_jia, "forum"),
        _record("T002", all_jia, "wiki"),
        _record("T003", all_jia, "news"),
        _record("T004", all_jia, "forum"),
        _record("T005", ["乙", "乙", "乙", "甲", "甲", "甲", "甲", "甲", "甲", "甲", "甲", "甲"], "wiki"),
        _record("T006", all_yi, "forum"),
        _record("T007", all_yi, "wiki"),
        _record("T008", all_bing, "news"),
        _record("T009", ["丙", "丁", "丙", "丙", "丙", "丙", "丙", "丙", "丁", "丙", "丙", "丙"], "forum"),
        _record("T010", all_ding, "wiki"),
        _record("T011", ["戊"] * 12, "forum"),
        _record("T012", list(VOCABULARY[:12]), "news"),
        _record("T013", ["己"] * 10, "wiki"),
        _record("T014", list(reversed(VOCABULARY[12:24])), "forum", version=2),
    ]


def build_edit_plan() -> dict[str, Any]:
    replacement_terms = ["壬"] * 12
    return {
        "schema_version": 1,
        "batches": [
            {
                "batch": 1,
                "operations": [
                    {
                        "op": "append",
                        "text": _record("T015", ["辛"] * 12, "forum", version=3),
                    },
                    {
                        "op": "replace",
                        "text_id": "T002",
                        "text": _record("T002", replacement_terms, "wiki", version=3),
                    },
                    {"op": "delete", "text_id": "T011"},
                    {
                        "op": "append",
                        "text": _record("T016", ["乙"] * 12, "news", version=3),
                    },
                ],
            },
            {
                "batch": 2,
                "operations": [
                    {"op": "set_config", "key": "window_count", "value": 5},
                    {"op": "set_config", "key": "quantization_levels", "value": 500},
                ],
            },
            {
                "batch": 3,
                "operations": [
                    {"op": "set_config", "key": "signature_dimension", "value": 40},
                    {"op": "set_config", "key": "bucket_bits", "value": 12},
                    {"op": "set_config", "key": "fixed_decimal_places", "value": 5},
                ],
            },
            {
                "batch": 4,
                "operations": [
                    {"op": "set_config", "key": "similarity_threshold", "value": 0.70},
                ],
            },
            {
                "batch": 5,
                "operations": [
                    {"op": "set_config", "key": "window_count", "value": 6},
                    {"op": "set_config", "key": "quantization_levels", "value": 2000},
                ],
            },
        ],
    }


def build_injections() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "injections": [
            {
                "injection_id": "I01-ZERO-DIM",
                "kind": "zero_signature_dimension",
                "field": "signature_dimension",
                "value": 0,
            },
            {
                "injection_id": "I02-NONFINITE-SIM",
                "kind": "non_finite_similarity",
                "field": "similarity_threshold",
                "value": "NaN",
            },
            {
                "injection_id": "I03-BUCKET-OVERFLOW",
                "kind": "bucket_bits_out_of_range",
                "field": "bucket_bits",
                "value": 64,
            },
            {
                "injection_id": "I04-CLUSTER-COLLISION",
                "kind": "cluster_id_conflict",
                "text_ids": ["T001", "T008"],
                "forced_cluster_id": "INJECTED-CONFLICT",
            },
        ],
    }


def build_invalid_text_lines() -> str:
    records = [
        {"text_id": "T001", "terms": ["甲"], "source": "forum", "version": 1},
        {"text_id": "T001", "terms": ["乙"], "source": "forum", "version": 1},
        {"text_id": "BAD_EMPTY", "terms": [], "source": "forum", "version": 1},
        {"text_id": "BAD_TERM", "terms": ["未收录词"], "source": "forum", "version": 1},
        {"text_id": "BAD_SOURCE", "terms": ["甲"], "source": "unknown", "version": 1},
        {"text_id": "BAD_VERSION_DOWN", "terms": ["乙"], "source": "forum", "version": 0},
        {"text_id": "BAD_VERSION_DOWN", "terms": ["丙"], "source": "forum", "version": 0},
    ]
    lines = [canonical_json(record) for record in records]
    lines.append('{"text_id": "BAD_JSON", "terms": [}')
    return "\n".join(lines) + "\n"


def build_invalid_plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "batches": [
            {
                "batch": 1,
                "operations": [
                    {"op": "unknown", "text_id": "T001"},
                    {"op": "delete"},
                    {"op": "set_config", "key": "not_a_field", "value": 1},
                    {
                        "op": "append",
                        "text": {"text_id": "T900", "terms": ["未收录词"], "source": "forum", "version": 4},
                    },
                ],
            }
        ],
    }


def _config_yaml(config: dict[str, Any]) -> str:
    lines = [
        "schema_version: 1",
        f"hash_seed: {config['hash_seed']}",
        f"window_count: {config['window_count']}",
        f"signature_dimension: {config['signature_dimension']}",
        f"similarity_threshold: {config['similarity_threshold']}",
        f"bucket_bits: {config['bucket_bits']}",
        f"fixed_decimal_places: {config['fixed_decimal_places']}",
        f"quantization_levels: {config['quantization_levels']}",
        f"signature_batch_budget: {config['signature_batch_budget']}",
        f"max_workers: {config['max_workers']}",
        f"snapshot_commit_interval: {config['snapshot_commit_interval']}",
        "source_labels:",
    ]
    lines.extend(f"  - {value}" for value in config["source_labels"])
    lines.append("vocabulary:")
    lines.extend(f"  - {value}" for value in config["vocabulary"])
    return "\n".join(lines) + "\n"


def _assert_sample_semantics(config: FingerprintConfig, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    records = {
        item["text_id"]: TextRecord(
            item["text_id"], tuple(item["terms"]), item["source"], int(item["version"])
        )
        for item in payloads
    }
    signatures = {
        text_id: compute_minhash(normalize_and_window(record, config), config)
        for text_id, record in records.items()
    }
    buckets = {
        text_id: signed_projection_bucket(result.signature, config)
        for text_id, result in signatures.items()
    }
    boundary = pair_similarity(
        signatures["T001"].signature,
        signatures["T005"].signature,
        config,
    )
    if not boundary["boundary"] or buckets["T004"] != buckets["T005"]:
        raise AssertionError("sample must contain a same-bucket exact-threshold boundary pair")
    repeated = pair_similarity(
        signatures["T001"].signature,
        signatures["T004"].signature,
        config,
    )
    if not repeated["qualifies"]:
        raise AssertionError("sample must contain a near-duplicate cluster")
    return {
        "boundary_pair": ["T001", "T005"],
        "boundary_similarity_fixed": boundary["similarity_fixed"],
        "boundary_bucket": buckets["T004"],
        "repeated_similarity_fixed": repeated["similarity_fixed"],
    }


def generate_samples(
    output_dir: str,
    *,
    config_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    config_raw = dict(config_payload or BASE_CONFIG)
    config, failures = validate_config_payload(config_raw, source="generated")
    if config is None:
        raise ValueError([failure.to_dict() for failure in failures])
    payloads = build_initial_payloads()
    semantics = _assert_sample_semantics(config, payloads)

    with (target / "texts.jsonl").open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(canonical_json(payload) + "\n")
    (target / "config.yaml").write_text(_config_yaml(config_raw), encoding="utf-8")
    (target / "edit_plan.json").write_text(
        json.dumps(build_edit_plan(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (target / "injections.json").write_text(
        json.dumps(build_injections(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (target / "invalid_texts.jsonl").write_text(build_invalid_text_lines(), encoding="utf-8")
    (target / "invalid_plan.json").write_text(
        json.dumps(build_invalid_plan(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "seed": 188,
                "text_count": len(payloads),
                "config_hash": config.config_hash,
                "semantics": semantics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(target),
        "text_count": len(payloads),
        "config_hash": config.config_hash,
        "semantics": semantics,
    }
