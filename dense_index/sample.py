from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .commands import CommandState
from .config import validate_config
from .engine import IndexEngine


VOCABULARY = [
    "<PAD>",
    "river", "delta", "vector", "index", "bucket", "query", "token", "hash",
    "shadow", "signal", "matrix", "quant", "fixed", "point", "ledger", "receipt",
    "document", "paragraph", "chunk", "window", "overlap", "seed", "stable", "digest",
    "copper", "amber", "granite", "willow", "cedar", "falcon", "otter", "lynx",
    "orbit", "beacon", "cipher", "lantern", "meadow", "harbor", "summit", "canyon",
    "prefix", "suffix", "ngram", "feature", "projection", "dimension", "threshold", "recall",
]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )


def base_config(seed: int = 239) -> Dict[str, Any]:
    return {
        "schema_version": "build-config/v1",
        "seed": seed,
        "chunk_window": 10,
        "chunk_overlap": 3,
        "pad_token": "<PAD>",
        "subword_min_n": 3,
        "subword_max_n": 4,
        "hash_buckets": 64,
        "projection_dim": 16,
        "projection_pool_dim": 32,
        "bucket_bits": 4,
        "quant_level": "Q16",
        "fractional_bits": 8,
        "sign_tie_rule": "positive",
        "similarity_threshold": 0.35,
        "max_concurrency": 4,
        "projection_point_budget": 100,
    }


def generate_sample(root: str | Path, seed: int = 239) -> Dict[str, Any]:
    root = Path(root)
    for directory in ("config", "data", "commands", "receipts", "fixtures", "config/injections"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    vocab = list(VOCABULARY)
    _write_json(root / "config" / "vocabulary.json", {"schema": "controlled-vocabulary/v1", "terms": vocab})

    records: List[Dict[str, Any]] = []
    normal_terms = [term for term in vocab if term != "<PAD>"]
    paragraph_number = 1
    for doc_index in range(1, 7):
        document_id = f"D{doc_index:02d}"
        for sequence_no in range(4):
            term_count = rng.randint(8, 16)
            terms = [rng.choice(normal_terms) for _ in range(term_count)]
            records.append(
                {
                    "paragraph_id": f"P{paragraph_number:04d}",
                    "document_id": document_id,
                    "sequence_no": sequence_no,
                    "terms": terms,
                }
            )
            paragraph_number += 1

    records[5]["terms"] = list(records[1]["terms"])
    records[5]["document_id"] = records[1]["document_id"]
    records[5]["sequence_no"] = 9
    _write_jsonl(root / "data" / "paragraphs.jsonl", records)

    config_data = base_config(seed)
    config_bundle = validate_config(config_data)
    config_text = yaml.safe_dump(config_data, allow_unicode=True, sort_keys=True)
    (root / "config" / "build_config.yaml").write_text(config_text, encoding="utf-8", newline="\n")

    initial_state = CommandState(config=config_bundle, records=records, vocabulary=vocab)
    initial_engine = IndexEngine(config_bundle)
    initial_report = initial_engine.run_batch(initial_state.blocks, budget=10**9)
    initial_records = initial_report["records"]

    bucket_ids = sorted(initial_report["buckets"])[:4]
    v1_receipts: List[Dict[str, Any]] = []
    v2_receipts: List[Dict[str, Any]] = []
    v3_receipts: List[Dict[str, Any]] = []
    for receipt_index, bucket_id in enumerate(bucket_ids):
        member_ids = initial_report["buckets"][bucket_id]
        head = next(record for record in initial_records if record["block_id"] == member_ids[0])
        hit_rate = 742 + receipt_index * 53
        if hit_rate > 999:
            hit_rate = 911 - receipt_index * 41
        common = {"bucket_id": bucket_id, "hit_rate_permille": hit_rate}
        v1_receipts.append(dict(common))
        v2_receipts.append({**common, "param_version": 10 + receipt_index, "elapsed_ms": 17 + receipt_index * 6})
        v3_receipts.append(
            {
                **common,
                "param_version": 10 + receipt_index,
                "elapsed_ms": 17 + receipt_index * 6,
                "chunk_digest": head["content_digest"],
                "overridden_duplicate_refs": list(head.get("duplicate_refs") or []),
                "projection_params": {
                    "projection_dim": config_data["projection_dim"],
                    "hash_buckets": config_data["hash_buckets"],
                    "quant_level": config_data["quant_level"],
                    "fractional_bits": config_data["fractional_bits"],
                    "projection_fingerprint": config_bundle.projection_fingerprint,
                },
            }
        )

    _write_json(root / "receipts" / "receipts_v1.json", {"format": "v1", "receipts": v1_receipts})
    _write_json(root / "receipts" / "receipts_v2.json", {"format": "v2", "receipts": v2_receipts})
    _write_json(root / "receipts" / "receipts_v3.json", {"format": "v3", "receipts": v3_receipts})

    replace_target = next(block["block_id"] for block in initial_state.blocks if block["document_id"] == "D03")
    delete_target = next(block["block_id"] for block in initial_state.blocks if block["document_id"] == "D04")
    replacement_terms = [rng.choice(normal_terms) for _ in range(config_data["chunk_window"])]

    commands: List[Dict[str, Any]] = [
        {
            "id": "cmd-001-append-batch",
            "op": "append_batch",
            "paragraphs": [
                {"paragraph_id": "P9001", "document_id": "D07", "sequence_no": 0, "terms": [rng.choice(normal_terms) for _ in range(11)]},
                {"paragraph_id": "P9002", "document_id": "D07", "sequence_no": 1, "terms": [rng.choice(normal_terms) for _ in range(13)]},
            ],
        },
        {
            "id": "cmd-002-replace-paragraph",
            "op": "replace_paragraph",
            "target_id": "P0003",
            "paragraph": {"paragraph_id": "P0003", "document_id": "D01", "sequence_no": 2, "terms": [rng.choice(normal_terms) for _ in range(12)]},
        },
        {
            "id": "cmd-003-replace-block",
            "op": "replace_block",
            "target_id": replace_target,
            "terms": replacement_terms,
        },
        {"id": "cmd-004-delete-block", "op": "delete_block", "target_id": delete_target},
        {"id": "cmd-005-delete-paragraph", "op": "delete_paragraph", "target_id": "P0004"},
        {"id": "cmd-006-window-overlap", "op": "set_config", "values": {"chunk_window": 8, "chunk_overlap": 2}},
        {"id": "cmd-007-projection-dim", "op": "set_config", "values": {"projection_dim": 12}},
        {"id": "cmd-008-threshold-loose", "op": "set_threshold", "threshold": 0.42},
        {"id": "cmd-009-quant-level", "op": "set_config", "values": {"quant_level": "Q8"}},
    ]

    preview_state = CommandState.from_sample_dir(root)
    for command in commands:
        preview_state.apply(command)
    preview_engine = IndexEngine(preview_state.config)
    preview_report = preview_engine.run_batch(preview_state.blocks, budget=10**9)
    exact_block = preview_report["records"][0]

    queries = {
        "queries": [
            {"query_id": "Q-EXACT", "terms": list(exact_block["terms"])},
            {"query_id": "Q-RANDOM-1", "terms": [rng.choice(normal_terms) for _ in range(7)]},
            {"query_id": "Q-RANDOM-2", "terms": [rng.choice(normal_terms) for _ in range(6)]},
            {"query_id": "Q-RANDOM-3", "terms": [rng.choice(normal_terms) for _ in range(8)]},
        ]
    }
    _write_json(root / "data" / "queries.json", queries)

    commands.extend(
        [
            {"id": "cmd-010-threshold-exact-boundary", "op": "set_threshold", "threshold": 1.0},
            {"id": "cmd-011-receipt-v2", "op": "switch_receipt_version", "target_version": "v2"},
            {"id": "cmd-012-receipt-v1", "op": "switch_receipt_version", "target_version": "v1"},
            {"id": "cmd-013-receipt-v3", "op": "switch_receipt_version", "target_version": "v3"},
            {"id": "cmd-014-budget-pending", "op": "set_point_budget", "budget": 3},
        ]
    )
    _write_json(root / "commands" / "command_stream.json", {"schema": "command-stream/v1", "commands": commands})

    invalid_corpus = "\n".join(
        [
            json.dumps({"paragraph_id": "BAD1", "document_id": "D9", "sequence_no": 0, "terms": ["river", "not_a_vocab_term"]}),
            json.dumps({"paragraph_id": "BAD1", "document_id": "D9", "sequence_no": 1, "terms": ["delta"]}),
            json.dumps({"paragraph_id": "BAD2", "document_id": "", "sequence_no": 2, "terms": ["bucket"]}),
            "{not valid json",
        ]
    ) + "\n"
    (root / "fixtures" / "corpus_invalid.jsonl").write_text(invalid_corpus, encoding="utf-8", newline="\n")
    _write_json(
        root / "fixtures" / "receipt_invalid_v1.json",
        {"format": "v1", "receipts": [{"bucket_id": "", "hit_rate_permille": 1234}]},
    )

    return {
        "root": str(root),
        "paragraph_count": len(records),
        "initial_block_count": len(initial_records),
        "bucket_ids": bucket_ids,
        "replace_target": replace_target,
        "delete_target": delete_target,
        "exact_block_id": exact_block["block_id"],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate deterministic local sample workspace")
    parser.add_argument("--root", default="sample_workspace")
    parser.add_argument("--seed", type=int, default=239)
    args = parser.parse_args()
    summary = generate_sample(args.root, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
