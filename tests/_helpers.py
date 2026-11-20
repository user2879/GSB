"""测试辅助：在临时目录构造最小确定性输入。"""
from __future__ import annotations

import json
import os

from sft_diag.runner import run_workbench


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_json(path: str, obj) -> None:
    write(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def write_jsonl(path: str, rows) -> None:
    write(path, "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                          for row in rows) + "\n")


def minimal_sample(
    root: str,
    *,
    threshold: float = 0.5,
    boundary_rule: str = "merge",
    max_workers: int = 4,
    interval: int = 1,
    extra_seed_rows: str = "",
    extra_slot_rows: str = "",
    extra_batches: list | None = None,
) -> str:
    sample_dir = os.path.join(root, "sample")
    seeds = [
        {
            "seed_id": "seed-1",
            "task_type": "alpha",
            "template_id": "tpl-a0",
            "template": "Classify the {subject} note about {topic} now.",
            "slots": ["subject", "topic"],
            "answer_template": "answer:{subject}:{topic}",
        },
        {
            "seed_id": "seed-2",
            "task_type": "alpha",
            "template_id": "tpl-a1",
            "template": "Summarize a document on {topic} for {subject} quickly.",
            "slots": ["subject", "topic"],
            "answer_template": "summary:{topic}:{subject}",
        },
    ]
    write_jsonl(os.path.join(sample_dir, "seeds.jsonl"), seeds)
    slots = [
        {"source_id": "src-1", "task_type": "alpha", "slot_name": "subject",
         "values": ["alpha", "beta"]},
        {"source_id": "src-2", "task_type": "alpha", "slot_name": "subject",
         "values": ["gamma", "delta"]},
        {"source_id": "src-3", "task_type": "alpha", "slot_name": "topic",
         "values": ["finance", "sports", "science"]},
        {"source_id": "src-4", "task_type": "alpha", "slot_name": "topic",
         "values": ["music", "travel", "health"]},
    ]
    write_jsonl(os.path.join(sample_dir, "slot_sources.jsonl"), slots)
    write_json(
        os.path.join(sample_dir, "vocab.json"),
        {
            "tokens": sorted({
                "classify", "summarize", "the", "a", "note", "about", "topic",
                "subject", "document", "on", "for", "now", "quickly", "answer",
                "alpha", "beta", "gamma", "delta", "finance", "sports", "science",
                "music", "travel", "health",
            }),
            "rules": {"match": "longest_prefix", "unknown": "char_fallback"},
        },
    )
    write_json(
        os.path.join(sample_dir, "build_config.json"),
        {
            "task_types": ["alpha"],
            "target_counts": {"alpha": 10},
            "template_weights": {"alpha": {"tpl-a0": 2.0, "tpl-a1": 1.0}},
            "slot_weights": {
                "alpha": {"src-1": 1.0, "src-2": 1.0, "src-3": 1.0, "src-4": 1.0}
            },
            "dedup_threshold": threshold,
            "boundary_rule": boundary_rule,
            "max_workers": max_workers,
            "snapshot_interval_batches": interval,
        },
    )
    write_json(
        os.path.join(sample_dir, "score_params.json"),
        {
            "dimensions": ["clarity", "correctness", "format", "length"],
            "specs": {
                "clarity": {"weight": 0.3, "baseline": 0.5},
                "correctness": {"weight": 0.4, "baseline": 0.6},
                "format": {"weight": 0.2, "baseline": 0.4},
                "length": {"weight": 0.1, "baseline": 0.3},
            },
            "length_lower": 4,
            "length_upper": 30,
        },
    )
    batches = extra_batches or []
    write_json(
        os.path.join(sample_dir, "batch_plan.json"),
        {"batches": batches},
    )
    if extra_seed_rows:
        with open(os.path.join(sample_dir, "seeds.jsonl"), "a", encoding="utf-8") as h:
            h.write(extra_seed_rows)
    if extra_slot_rows:
        with open(os.path.join(sample_dir, "slot_sources.jsonl"), "a",
                  encoding="utf-8") as h:
            h.write(extra_slot_rows)
    return sample_dir


def run(sample_dir: str, output_dir: str, **kwargs):
    return run_workbench(sample_dir, output_dir, shadow_check=False, **kwargs)
