"""固定种子程序：从受控词表生成确定性本地样本目录。

无任何随机源：所有文本均由固定词表与确定性循环拼接生成，重复执行产物逐字节一致。
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "samples", "basic")

TASK_TYPES = ["classification", "summarization", "translation"]

SUBJECTS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu",
]
ACTIONS = [
    "review", "summarize", "translate", "classify", "annotate", "extract",
    "rewrite", "evaluate",
]
FORMATS = ["table", "paragraph", "bullet", "outline", "memo", "dialogue"]
LANGUAGES = ["english", "french", "german", "spanish", "italian", "japanese"]
TOPICS = [
    "finance", "sports", "science", "music", "travel", "cooking", "health",
    "history", "gaming", "weather",
]
TONES = ["formal", "neutral", "concise", "casual", "academic", "technical"]

VOCAB_TOKENS = sorted(
    {
        token
        for words in (
            SUBJECTS, ACTIONS, FORMATS, LANGUAGES, TOPICS, TONES,
            ["the", "a", "an", "please", "into", "with", "and", "of",
             "about", "return", "one", "two", "single", "sentence",
             "sentences", "within", "keep", "document", "output",
             "ing", "tion", "ly", "ed", "class", "ify", "ize", "summary",
             "translate", "answer", "label", "text", "report", "note",
             "classify", "summarize", "tone", "in", "to"],
        )
        for token in words
    }
)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_jsonl(path: str, rows: list[dict]) -> None:
    write_text(path, "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                               for row in rows) + "\n")


def build_seeds() -> list[dict]:
    seeds: list[dict] = []
    counter = 0
    for task_index, task in enumerate(TASK_TYPES):
        for variant in range(3):
            counter += 1
            if task == "classification":
                slots = ["subject", "topic"]
                classification_templates = (
                    "Please classify the {subject} report about {topic} and return one label.",
                    "Task: assign a single category label to the {subject} document covering {topic}.",
                    "Read the {subject} article on {topic}; output only its category tag as the answer.",
                )
                template = classification_templates[variant]
                answer = "label::{subject}:{topic}"
            elif task == "summarization":
                slots = ["topic", "tone"]
                summarization_templates = (
                    "Summarize the text about {topic} in a {tone} tone within two sentences.",
                    "Produce a {tone} headline plus bullet outline for the passage discussing {topic}.",
                    "Condense the {topic} report into one {tone} paragraph without extra commentary.",
                )
                template = summarization_templates[variant]
                answer = "summary::{topic}:{tone}"
            else:
                slots = ["language", "subject"]
                translation_templates = (
                    "Translate the note of {subject} into {language} and keep the answer concise.",
                    "Render the {subject} passage in {language}; preserve meaning and omit explanations.",
                    "Convert the {subject} draft to {language}, then return only the translated sentence.",
                )
                template = translation_templates[variant]
                answer = "translation::{language}:{subject}"
            seeds.append(
                {
                    "seed_id": f"seed-{counter:03d}",
                    "task_type": task,
                    "template_id": f"tpl-{task}-{variant:02d}",
                    "template": template,
                    "slots": slots,
                    "answer_template": answer,
                }
            )
    return seeds


def build_slot_sources() -> list[dict]:
    rows: list[dict] = []
    source_index = 0
    catalog = {
        "classification": {
            "subject": (SUBJECTS, 2),
            "topic": (TOPICS, 2),
        },
        "summarization": {
            "topic": (TOPICS, 2),
            "tone": (TONES, 2),
        },
        "translation": {
            "language": (LANGUAGES, 2),
            "subject": (SUBJECTS, 2),
        },
    }
    for task, slots in catalog.items():
        for slot_name, (values, source_count) in slots.items():
            chunk = max(1, len(values) // source_count)
            for source_no in range(source_count):
                source_index += 1
                start = (source_no * chunk) % len(values)
                picked = [values[(start + offset) % len(values)]
                          for offset in range(chunk)]
                rows.append(
                    {
                        "source_id": f"src-{source_index:03d}",
                        "task_type": task,
                        "slot_name": slot_name,
                        "values": picked,
                    }
                )
    return rows


def build_config() -> dict:
    template_weights = {}
    slot_weights = {}
    targets = {"classification": 16, "summarization": 14, "translation": 12}
    for task_index, task in enumerate(TASK_TYPES):
        template_weights[task] = {
            f"tpl-{task}-{variant:02d}": 1.0 + variant
            for variant in range(3)
        }
    source_rows = build_slot_sources()
    for task in TASK_TYPES:
        ids = sorted(row["source_id"] for row in source_rows if row["task_type"] == task)
        slot_weights[task] = {
            source_id: 1.0 + (index % 3) for index, source_id in enumerate(ids)
        }
    return {
        "task_types": TASK_TYPES,
        "target_counts": targets,
        "template_weights": template_weights,
        "slot_weights": slot_weights,
        "dedup_threshold": 0.55,
        "boundary_rule": "merge",
        "max_workers": 4,
        "snapshot_interval_batches": 2,
    }


def build_score_params() -> dict:
    return {
        "dimensions": ["clarity", "correctness", "format", "length"],
        "specs": {
            "clarity": {"weight": 0.30, "baseline": 0.55},
            "correctness": {"weight": 0.35, "baseline": 0.60},
            "format": {"weight": 0.20, "baseline": 0.50},
            "length": {"weight": 0.15, "baseline": 0.45},
        },
        "length_lower": 6,
        "length_upper": 24,
    }


def build_batch_plan() -> dict:
    return {
        "batches": [
            {
                "batch_no": 1,
                "operations": [
                    {
                        "op": "replace_template",
                        "seed": {
                            "seed_id": "seed-001",
                            "task_type": "classification",
                            "template_id": "tpl-classification-000",
                            "template": "Classify the {subject} document about {topic} and output a single answer label.",
                            "slots": ["subject", "topic"],
                            "answer_template": "label::{subject}:{topic}",
                        },
                    },
                    {"op": "set_target_count", "task_type": "translation", "value": 12},
                ],
            },
            {
                "batch_no": 2,
                "operations": [
                    {
                        "op": "add_slot_source",
                        "source": {
                            "source_id": "src-901",
                            "task_type": "summarization",
                            "slot_name": "tone",
                            "values": ["formal", "brief", "analytical"],
                        },
                    },
                    {"op": "set_slot_weight", "task_type": "summarization",
                     "source_id": "src-901", "value": 2.5},
                ],
            },
            {
                "batch_no": 3,
                "operations": [
                    {"op": "delete_template", "seed_id": "seed-005"},
                    {"op": "set_score_weight", "dimension": "clarity", "value": 0.42},
                    {"op": "set_score_baseline", "dimension": "clarity", "value": 0.70},
                ],
            },
            {
                "batch_no": 4,
                "operations": [
                    {"op": "delete_slot_source", "source_id": "src-003"},
                    {"op": "set_dedup_threshold", "value": 0.5},
                    {"op": "set_target_count", "task_type": "classification", "value": 16},
                ],
            },
            {
                "batch_no": 5,
                "operations": [
                    {"op": "set_boundary_rule", "value": "split"},
                    {"op": "set_score_weight", "dimension": "length", "value": 0.22},
                    {"op": "set_template_weight", "task_type": "translation",
                     "template_id": "tpl-translation-001", "value": 4.0},
                ],
            },
        ]
    }


def build_vocab() -> dict:
    return {
        "tokens": VOCAB_TOKENS,
        "rules": {"match": "longest_prefix", "unknown": "char_fallback"},
    }


def main() -> int:
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    write_jsonl(os.path.join(SAMPLE_DIR, "seeds.jsonl"), build_seeds())
    write_jsonl(os.path.join(SAMPLE_DIR, "slot_sources.jsonl"), build_slot_sources())
    write_text(
        os.path.join(SAMPLE_DIR, "vocab.json"),
        json.dumps(build_vocab(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write_text(
        os.path.join(SAMPLE_DIR, "build_config.json"),
        json.dumps(build_config(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write_text(
        os.path.join(SAMPLE_DIR, "score_params.json"),
        json.dumps(build_score_params(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    write_text(
        os.path.join(SAMPLE_DIR, "batch_plan.json"),
        json.dumps(build_batch_plan(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(f"sample directory written: {SAMPLE_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
