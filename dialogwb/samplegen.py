from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


SEED = 20261118

TOKENS = [
    "我们",
    "团队",
    "用户",
    "负责人",
    "目标",
    "计划",
    "方案",
    "预算",
    "风险",
    "进度",
    "反馈",
    "接口",
    "测试",
    "质量",
    "交付",
    "发布",
    "时间",
    "问题",
    "结论",
    "记录",
    "讨论",
    "决定",
    "确认",
    "检查",
    "优化",
    "跟进",
    "保持",
    "延期",
    "里程碑",
    "要点",
    "与",
    "明天",
    "需要",
    "结果",
    "最终",
    "we",
    "team",
    "user",
    "owner",
    "goal",
    "plan",
    "option",
    "budget",
    "risk",
    "progress",
    "feedback",
    "api",
    "interface",
    "test",
    "quality",
    "delivery",
    "release",
    "time",
    "issue",
    "conclusion",
    "record",
    "discuss",
    "decide",
    "confirm",
    "check",
    "optimize",
    "follow",
    "keep",
    "delay",
    "milestone",
    "point",
    "tomorrow",
    "need",
    "result",
    "final",
    "and",
    "with",
    "for",
    "the",
    "to",
    "a",
    "is",
    "of",
    "on",
    "this",
    "will",
    "review",
    "next",
    "dead",
    "line",
    "s",
    "deadline",
    "deadlines",
    "system",
    "summary",
    "update",
]


ZH_SENTENCES = [
    "团队讨论预算。",
    "负责人确认里程碑。",
    "用户反馈需要优化接口。",
    "负责人跟进测试。",
    "风险结论需要确认。",
    "我们决定保持进度。",
    "最终方案记录交付与质量。",
    "明天检查发布计划。",
]

EN_SENTENCES = [
    "we review plan and risk.",
    "user feedback need optimize api.",
    "final plan confirm budget.",
    "risk issue need follow.",
    "team confirm delivery and quality.",
    "owner check test progress.",
    "option decide delay release.",
    "team review deadlines and delivery.",
]


def generate_sample(destination: str | Path) -> dict[str, Path]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    lexicon = {
        "version": "lexicon-2026.11.1",
        "tokens": list(TOKENS),
        "rules": {
            "case_sensitive": False,
            "punctuation": [
                "，",
                "。",
                "！",
                "？",
                "；",
                "：",
                "、",
                "…",
                "“",
                "”",
                "‘",
                "’",
                "（",
                "）",
            ],
            "subwords": {"deadlines": ["dead", "line", "s"]},
        },
    }
    config = {
        "compression_window": 3,
        "concurrency": 4,
        "dedupe_similarity_threshold": 0.58,
        "max_candidates_per_round": 4,
        "max_summary_length": 42,
        "point_weights": {
            "base": 1.5,
            "position_decay": 0.82,
            "sentence_bonus": 0.25,
            "term_weights": {
                "decide": 1.6,
                "delivery": 1.4,
                "risk": 2.0,
                "交付": 1.4,
                "决定": 1.6,
                "风险": 2.0,
            },
        },
        "snapshot_interval": 2,
        "template_weights": [
            {"format": "记录要点：{point}", "id": "zh_record", "weight": 3.0},
            {"format": "讨论结论：{point}", "id": "zh_conclusion", "weight": 2.0},
            {"format": "Key point: {point}", "id": "en_summary", "weight": 2.0},
        ],
    }

    lines: list[str] = ["{this is not valid json}"]
    base_time = datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc)
    session_specs = {
        "s00_short": ("zh", 2),
        "s01_plan": ("mixed", 9),
        "s02_delivery": ("en", 9),
        "s03_feedback": ("mixed", 7),
    }
    for session_id in sorted(session_specs):
        language, turns = session_specs[session_id]
        for turn in range(1, turns + 1):
            if language == "zh":
                sentence_count = rng.randint(1, 3)
                pool = ZH_SENTENCES
            elif language == "en":
                sentence_count = rng.randint(1, 4)
                pool = EN_SENTENCES
            else:
                sentence_count = rng.randint(2, 4)
                pool = ZH_SENTENCES if turn % 2 else EN_SENTENCES
            text = "".join(
                rng.choice(pool) for _ in range(sentence_count)
            ).strip()
            event = {
                "session_id": session_id,
                "turn": turn,
                "speaker": "林舟" if turn % 3 == 0 else "Maya",
                "text": text,
                "timestamp": (
                    base_time + timedelta(minutes=turn * 3 + len(session_id))
                ).isoformat(),
            }
            lines.append(json.dumps(event, ensure_ascii=False, sort_keys=True))

    lines.extend(
        [
            json.dumps(
                {
                    "session_id": "s90_empty",
                    "turn": 1,
                    "speaker": "system",
                    "text": "我们讨论计划与风险。",
                    "timestamp": "2026-11-01T12:00:00+00:00",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            json.dumps(
                {
                    "session_id": "s90_empty",
                    "turn": 2,
                    "speaker": "system",
                    "text": "",
                    "timestamp": "2026-11-01T12:01:00+00:00",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            json.dumps(
                {
                    "session_id": "s90_empty",
                    "turn": 3,
                    "speaker": "system",
                    "text": "团队确认预算与质量。",
                    "timestamp": "2026-11-01T12:02:00+00:00",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            json.dumps(
                {
                    "session_id": "s91_unknown",
                    "turn": 1,
                    "speaker": "system",
                    "text": "记录量子。",
                    "timestamp": "2026-11-01T12:03:00+00:00",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            json.dumps(
                {
                    "session_id": "s91_unknown",
                    "turn": 2,
                    "speaker": "system",
                    "text": "用户反馈问题。",
                    "timestamp": "2026-11-01T12:04:00+00:00",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            json.dumps(
                {
                    "session_id": "s92_gap",
                    "turn": 1,
                    "speaker": "system",
                    "text": "用户反馈预算问题。",
                    "timestamp": "2026-11-01T12:05:00+00:00",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            json.dumps(
                {
                    "session_id": "s92_gap",
                    "turn": 3,
                    "speaker": "system",
                    "text": "我们讨论方案。",
                    "timestamp": "2026-11-01T12:06:00+00:00",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            json.dumps(
                {
                    "session_id": "s92_gap",
                    "turn": 4,
                    "speaker": "system",
                    "text": "负责人跟进质量与进度。",
                    "timestamp": "2026-11-01T12:07:00+00:00",
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        ]
    )

    edits = {
        "batches": [
            {
                "batch_id": "replace-and-reweight",
                "operations": [
                    {
                        "op": "replace",
                        "session_id": "s01_plan",
                        "turn": 2,
                        "speaker": "Maya",
                        "text": "风险结论需要确认。负责人优化接口测试。",
                        "timestamp": "2026-11-02T09:00:00+00:00",
                    },
                    {
                        "op": "replace",
                        "session_id": "s03_feedback",
                        "turn": 5,
                        "text": "风险交付决定质量。",
                    },
                ],
            },
            {
                "batch_id": "delete-add-window",
                "operations": [
                    {
                        "op": "delete",
                        "session_id": "s02_delivery",
                        "turn": 3,
                    },
                    {
                        "op": "add",
                        "session_id": "s02_delivery",
                        "turn": 9,
                        "speaker": "system",
                        "text": "final plan confirm quality and delivery.",
                        "timestamp": "2026-11-03T09:00:00+00:00",
                    },
                ],
                "set_compression_window": 2,
            },
            {
                "batch_id": "english-replace-window",
                "config_patch": {
                    "compression_window": 4,
                    "point_weights": {
                        "base": 1.8,
                        "position_decay": 0.8,
                    },
                },
                "operations": [
                    {
                        "op": "replace",
                        "session_id": "s03_feedback",
                        "turn": 4,
                        "text": "user feedback record risk and quality. team confirm final plan.",
                        "timestamp": "2026-11-04T09:00:00+00:00",
                    },
                    {
                        "op": "add",
                        "session_id": "s01_plan",
                        "turn": 10,
                        "speaker": "林舟",
                        "text": "最终结论记录交付与质量。风险计划保持进度。",
                        "timestamp": "2026-11-04T09:01:00+00:00",
                    },
                ],
            },
        ]
    }

    paths = {
        "events": root / "events.jsonl",
        "lexicon": root / "lexicon.json",
        "config": root / "config.json",
        "edits": root / "edits.json",
    }
    paths["events"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["lexicon"].write_text(
        json.dumps(lexicon, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["config"].write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["edits"].write_text(
        json.dumps(edits, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic sample inputs")
    parser.add_argument("--out", default="samples")
    args = parser.parse_args(argv)
    paths = generate_sample(args.out)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

