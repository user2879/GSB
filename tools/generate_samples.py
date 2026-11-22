#!/usr/bin/env python3
"""Regenerate the bundled deterministic local sample set.

The program has no clock, network, or process-global random dependency.  The
sample corpus is intentionally small so every construction decision can be
audited by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FIXED_SEED = "uidfc5909d3-family-local-samples-v1"

SEEDS = [
    {
        "seed_id": "seed-classify-intent",
        "task_type": "classify",
        "template_variant_id": "classify.intent",
        "instruction_template": "判断指令“{topic}”的目标类别，并给出判断依据。",
        "slots": ["topic"],
        "reference_answer": "类别：{topic}；依据：关键词与上下文一致。",
    },
    {
        "seed_id": "seed-classify-risk",
        "task_type": "classify",
        "template_variant_id": "classify.risk",
        "instruction_template": "识别“{topic}”中的主要风险，输出风险等级与处理建议。",
        "slots": ["topic"],
        "reference_answer": "风险等级：中；建议对“{topic}”执行双人复核。",
    },
    {
        "seed_id": "seed-summarize-concise",
        "task_type": "summarize",
        "template_variant_id": "summarize.concise",
        "instruction_template": "用一句话概括“{topic}”并点出关键风险。",
        "slots": ["topic"],
        "reference_answer": "“{topic}”需要简明概括并标注关键风险。",
    },
    {
        "seed_id": "seed-summarize-bullets",
        "task_type": "summarize",
        "template_variant_id": "summarize.bullets",
        "instruction_template": "将“{topic}”概括为两条要点，每条不超过十五字。",
        "slots": ["topic"],
        "reference_answer": "要点一：说明主题；要点二：提示风险。",
    },
    {
        "seed_id": "seed-rewrite-polite",
        "task_type": "rewrite",
        "template_variant_id": "rewrite.polite",
        "instruction_template": "将“{topic}”改写为礼貌、明确的中文请求。",
        "slots": ["topic"],
        "reference_answer": "请围绕“{topic}”给出清晰、可执行的答复。",
    },
    {
        "seed_id": "seed-rewrite-short",
        "task_type": "rewrite",
        "template_variant_id": "rewrite.short",
        "instruction_template": "把“{topic}”压缩为不超过二十字的短句。",
        "slots": ["topic"],
        "reference_answer": "请简要处理“{topic}”。",
    },
    {
        "seed_id": "seed-extract-fields",
        "task_type": "extract",
        "template_variant_id": "extract.fields",
        "instruction_template": "从“{topic}”中抽取主体、动作与约束三个字段。",
        "slots": ["topic"],
        "reference_answer": "主体：执行人；动作：复核；约束：最小权限。",
    },
    {
        "seed_id": "seed-extract-risk",
        "task_type": "extract",
        "template_variant_id": "extract.risk",
        "instruction_template": "列出“{topic}”中的实体与风险线索。",
        "slots": ["topic"],
        "reference_answer": "实体：{topic}；线索：异常访问与权限变化。",
    },
]

LEXICON = {
    "schema_version": 1,
    "sources": {
        "core": ["数据安全", "访问控制", "审计追踪", "最小权限", "异常登录", "变更复核"],
        "support": ["备份恢复", "密钥轮换", "日志留存", "基线检查", "离线演练"],
        "edge": ["边界主机", "受限终端", "离线环境", "临时账号"],
    },
    "subword": {
        "lowercase": True,
        "ngram_size": 2,
        "drop_chars": " \t\r\n，。！？；：,.!?;:“”\"'（）()【】[]",
        "min_chars": 1,
    },
}

SCORING = {
    "schema_version": 1,
    "dimension_order": ["clarity", "correctness", "format", "length"],
    "dimensions": {
        "clarity": {"weight": 0.30, "baseline": 76.0},
        "correctness": {"weight": 0.35, "baseline": 78.0},
        "format": {"weight": 0.20, "baseline": 80.0},
        "length": {"weight": 0.15, "baseline": 74.0},
    },
    "pass_threshold": 87.6,
    "ideal_length_chars": 26,
    "length_penalty_scale": 1.2,
}

CONFIG = {
    "schema_version": 1,
    "task_types": {
        "classify": {
            "target_count": 5,
            "template_weights": {"classify.intent": 1.0, "classify.risk": 1.0},
            "slot_source_weights": {"core": 3.0, "support": 2.0, "edge": 1.0},
        },
        "rewrite": {
            "target_count": 5,
            "template_weights": {"rewrite.polite": 1.0, "rewrite.short": 1.0},
            "slot_source_weights": {"core": 2.0, "support": 2.0, "edge": 1.0},
        },
        "summarize": {
            "target_count": 6,
            "template_weights": {"summarize.concise": 3.0, "summarize.bullets": 1.0},
            "slot_source_weights": {"core": 3.0, "support": 1.0, "edge": 1.0},
        },
    },
    "task_quotas": {"classify": 6, "rewrite": 6, "summarize": 7},
    "near_duplicate_threshold": 0.65,
    "fixed_decimal_places": 3,
    "quantization_step": 0.01,
    "max_concurrency": 4,
    "global_score_budget": 90,
    "per_batch_generation_points": 25,
    "snapshot_commit_interval_batches": 2,
}

PLAN = {
    "schema_version": 1,
    "batches": [
        {
            "batch_id": "b01-target-classify",
            "operations": [{"op": "set_target", "task_type": "classify", "target_count": 4}],
        },
        {
            "batch_id": "b02-classify-template-weights",
            "operations": [
                {
                    "op": "set_template_weights",
                    "task_type": "classify",
                    "template_weights": {"classify.intent": 2.0, "classify.risk": 1.0},
                }
            ],
        },
        {
            "batch_id": "b03-replace-rewrite-template",
            "operations": [
                {
                    "op": "replace_template",
                    "task_type": "rewrite",
                    "variant_id": "rewrite.short",
                    "instruction_template": "请把“{topic}”压缩成不超过二十字的明确短句。",
                    "reference_answer": "请用二十字以内说明“{topic}”。",
                    "slots": ["topic"],
                }
            ],
        },
        {
            "batch_id": "b04-remove-summarize-edge",
            "operations": [{"op": "remove_slot_source", "task_type": "summarize", "source": "edge"}],
        },
        {
            "batch_id": "b05-score-weight-and-target",
            "operations": [
                {"op": "set_score_weight", "dimension": "clarity", "weight": 0.38},
                {"op": "set_score_weight", "dimension": "length", "weight": 0.08},
                {"op": "set_target", "task_type": "summarize", "target_count": 7},
            ],
        },
        {
            "batch_id": "b06-delete-classify",
            "operations": [{"op": "delete_task", "task_type": "classify"}],
        },
        {
            "batch_id": "b07-add-extract-task",
            "operations": [
                {
                    "op": "add_task",
                    "task_type": "extract",
                    "target_count": 3,
                    "template_weights": {"extract.fields": 1.0, "extract.risk": 1.0},
                    "slot_source_weights": {"core": 1.0, "support": 1.0},
                    "task_quota": 4,
                }
            ],
        },
        {
            "batch_id": "b08-final-summarize-target",
            "operations": [{"op": "set_target", "task_type": "summarize", "target_count": 7}],
        },
    ],
}


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _seed_text() -> str:
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in SEEDS) + "\n"


def _expected_files() -> dict[str, str]:
    return {
        "seeds.jsonl": _seed_text(),
        "lexicon.json": _json_text(LEXICON),
        "scoring.json": _json_text(SCORING),
        "config.json": _json_text(CONFIG),
        "plan.json": _json_text(PLAN),
    }


def _semantic(value: Any) -> Any:
    return value


def check(output_dir: Path) -> bool:
    expected = _expected_files()
    ok = True
    for filename, text in expected.items():
        path = output_dir / filename
        if not path.exists():
            print(f"MISSING {path}")
            ok = False
            continue
        if filename.endswith(".json"):
            same = _semantic(json.loads(path.read_text(encoding="utf-8"))) == _semantic(json.loads(text))
        else:
            same = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ] == [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        print(f"{'OK' if same else 'MISMATCH'} {path}")
        ok = ok and same
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    print(f"fixed_seed={FIXED_SEED}")
    if args.check:
        return 0 if check(output_dir) else 1
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in _expected_files().items():
        (output_dir / filename).write_text(text, encoding="utf-8", newline="\n")
        print(f"WROTE {output_dir / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

