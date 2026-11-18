#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from session_pipeline.util import atomic_write_json, atomic_write_text  # noqa: E402


SEED = 20260818
VOCAB = {
    "book": ["我想预订一个位置", "麻烦帮我安排预订", "请预留两个人的位置"],
    "cancel": ["我想取消之前的预约", "请帮我取消这次预订", "预约需要撤销"],
    "inquiry": ["我想了解一下营业时间", "请介绍一下可用套餐", "想咨询预订规则"],
    "escalate": ["请转人工客服", "我需要人工帮助", "麻烦安排人工处理"],
    "confirm": ["确认，请继续", "好的，确认办理", "确认这个方案"],
    "finish": ["结束本次会话", "已经完成，结束吧", "好的，会话结束"],
    "filler": ["谢谢", "明白了", "请稍等", "收到"],
    "assistant": ["请问需要几位", "已记录您的需求", "正在为您处理", "还需要其他信息吗"],
}


def event(
    session_id: str,
    turn: int,
    role: str,
    text: str,
    timestamp_ms: int,
    updates: list[dict] | None = None,
    requested_target: str | None = None,
) -> dict:
    value = {
        "session_id": session_id,
        "turn": turn,
        "role": role,
        "text": text,
        "slot_updates": updates or [],
        "timestamp_ms": timestamp_ms,
    }
    if requested_target is not None:
        value["requested_target"] = requested_target
    return value


def build_events(rng: random.Random) -> list[dict]:
    def phrase(group: str) -> str:
        choices = VOCAB[group]
        return choices[rng.randrange(len(choices))]

    events = [
        event("S001", 1, "user", phrase("book"), 1000, [
            {"slot": "intent", "value": "book", "confidence": "0.800000", "source": "explicit"}
        ]),
        event("S002", 1, "user", phrase("cancel"), 1050, [
            {"slot": "intent", "value": "cancel", "confidence": "0.880000", "source": "explicit"}
        ]),
        event("S001", 2, "assistant", phrase("assistant"), 1100),
        event("S002", 2, "user", "请取消这次预订", 1150, requested_target="closed"),
        event("S001", 3, "user", "两位，预算三百", 1200, [
            {"slot": "party_size", "value": 2, "confidence": "0.900000", "source": "explicit"},
            {"slot": "budget", "value": "300.00", "confidence": "0.700000", "source": "explicit"},
        ]),
        event("S002", 3, "user", "改到明天", 1250),
        event("S001", 4, "user", "电话是 123-4567", 1300, [
            {"slot": "phone", "value": "123-4567", "confidence": "0.850000", "source": "explicit"}
        ]),
        event("S002", 4, "user", "想回到收集状态", 1350, requested_target="collecting"),
        event("S001", 5, "user", phrase("confirm"), 1400, [
            {"slot": "intent", "value": "book", "confidence": "0.950000", "source": "explicit"}
        ]),
        event("S002", 5, "assistant", phrase("filler"), 1450),
        event("S003", 1, "user", phrase("escalate"), 2100, [
            {"slot": "intent", "value": "escalate", "confidence": "0.910000", "source": "explicit"}
        ]),
        event("S003", 2, "assistant", phrase("assistant"), 2200),
        event("S003", 3, "user", "会员为真", 2300, [
            {"slot": "member", "value": True, "confidence": "0.700000", "source": "explicit"}
        ]),
        event("S003", 4, "user", "会员改为假", 2400, [
            {"slot": "member", "value": False, "confidence": "0.900000", "source": "inferred"}
        ]),
        event("S004", 1, "user", phrase("inquiry"), 3000, [
            {"slot": "intent", "value": "inquiry", "confidence": "0.840000", "source": "explicit"}
        ]),
        event("S004", 2, "user", "四个人，预算五百", 3100, [
            {"slot": "party_size", "value": 4, "confidence": "0.900000", "source": "explicit"},
            {"slot": "budget", "value": "500.00", "confidence": "0.800000", "source": "explicit"},
        ]),
        event("S004", 3, "user", "电话 234-5678", 3200, [
            {"slot": "phone", "value": "234-5678", "confidence": "0.860000", "source": "explicit"}
        ]),
        event("S004", 4, "user", "改成六个人", 3300, [
            {"slot": "party_size", "value": 6, "confidence": "0.850000", "source": "inferred"}
        ]),
        event("S004", 5, "user", phrase("confirm"), 3400),
        event("S004", 6, "user", phrase("finish"), 3500),
    ]
    return sorted(events, key=lambda item: item["timestamp_ms"])


def machine_definition() -> dict:
    return {
        "machine_id": "booking_support_v1",
        "initial_state": "idle",
        "states": ["idle", "greeted", "collecting", "ready", "escalated", "closed"],
        "slots": {
            "intent": {
                "type": "enum",
                "allowed": ["book", "cancel", "inquiry", "escalate"],
                "merge": "overwrite",
            },
            "party_size": {"type": "int", "min": 1, "max": 12, "merge": "max_value"},
            "budget": {
                "type": "decimal",
                "min": "0",
                "max": "10000",
                "decimal_places": 2,
                "merge": "weighted",
            },
            "phone": {
                "type": "string",
                "pattern": "^[0-9]{3}-[0-9]{4}$",
                "merge": "overwrite",
            },
            "member": {"type": "bool", "merge": "reject_overwrite"},
        },
        "merge_strategies": {
            "intent": "overwrite",
            "party_size": "max_value",
            "budget": "weighted",
            "phone": "overwrite",
            "member": "reject_overwrite",
        },
        "conflict_priority": ["explicit", "inferred", "default"],
        "transitions": [
            {"id": "R010", "from": "idle", "to": "greeted", "when": {"intent": {"op": "eq", "value": "book"}}},
            {"id": "R011", "from": "idle", "to": "greeted", "when": {"intent": {"op": "eq", "value": "inquiry"}}},
            {"id": "R012", "from": "idle", "to": "greeted", "when": {"intent": {"op": "eq", "value": "cancel"}}},
            {"id": "R020", "from": "idle", "to": "escalated", "when": {"intent": {"op": "eq", "value": "escalate"}}},
            {"id": "R030", "from": "greeted", "to": "collecting", "when": {"party_size": {"op": "exists"}}},
            {"id": "R031", "from": "greeted", "to": "collecting", "when": {"budget": {"op": "exists"}}},
            {"id": "R013", "from": "greeted", "to": "closed", "when": {"intent": {"op": "eq", "value": "cancel"}}},
            {"id": "R040", "from": "collecting", "to": "ready", "text_contains": "确认"},
            {"id": "R041", "from": "collecting", "to": "escalated", "text_contains": "人工"},
            {"id": "R050", "from": "collecting", "to": "closed", "when": {"intent": {"op": "eq", "value": "cancel"}}},
            {"id": "R060", "from": "ready", "to": "closed", "text_contains": "结束"},
        ],
    }


def pipeline_config() -> dict:
    return {
        "worker_count": 4,
        "max_events_per_batch": 32,
        "global_state_budget": 40,
        "session_state_quota": 8,
        "snapshot_interval_batches": 1,
        "confidence_decimals": 6,
        "conflict_priority": ["explicit", "inferred", "default"],
        "merge_strategy_overrides": {},
        "business_codes": {
            "input_invalid": "E_INPUT_INVALID",
            "transition_rejected": "E_TRANSITION_REJECTED",
            "slot_value_invalid": "E_SLOT_VALUE_INVALID",
            "session_quota_exceeded": "E_SESSION_QUOTA_EXCEEDED",
            "state_budget_exhausted": "E_STATE_BUDGET_EXHAUSTED",
            "internal_error": "E_INTERNAL_ERROR",
        },
    }


def edit_plan() -> dict:
    return {
        "description": "追加、替换、删除、槽位模式与合并策略切换的确定性编辑序列",
        "batches": [
            {
                "batch_id": 1,
                "operations": [
                    {
                        "op": "append",
                        "event": event(
                            "S001", 6, "user", "谢谢，保持这个安排", 4000,
                            [{"slot": "budget", "value": "360.00", "confidence": "0.760000", "source": "inferred"}],
                        ),
                    },
                    {
                        "op": "replace_utterance",
                        "session_id": "S002",
                        "turn": 3,
                        "text": "改到后天，已经确认",
                    },
                    {
                        "op": "patch_slot_schema",
                        "slot": "party_size",
                        "patch": {"max": 20},
                    },
                ],
            },
            {
                "batch_id": 2,
                "operations": [
                    {"op": "delete_turn", "session_id": "S003", "turn": 2},
                    {
                        "op": "append",
                        "event": event(
                            "S004", 7, "user", "请回到收集状态", 4100, requested_target="collecting"
                        ),
                    },
                    {"op": "set_merge_strategy", "slot": "intent", "strategy": "reject_overwrite"},
                    {
                        "op": "append",
                        "event": event(
                            "S005", 1, "user", "我想咨询套餐", 5000,
                            [{"slot": "intent", "value": "inquiry", "confidence": "0.800000", "source": "explicit"}],
                        ),
                    },
                    {
                        "op": "append",
                        "event": event(
                            "S005", 2, "user", "两个人，预算四百", 5100,
                            [
                                {"slot": "party_size", "value": 2, "confidence": "0.870000", "source": "explicit"},
                                {"slot": "budget", "value": "400.00", "confidence": "0.780000", "source": "explicit"},
                            ],
                        ),
                    },
                    {
                        "op": "append",
                        "event": event("S005", 3, "user", "确认，请继续", 5200),
                    },
                ],
            },
            {
                "batch_id": 3,
                "operations": [
                    {"op": "delete_turn", "session_id": "S004", "turn": 4},
                    {"op": "set_merge_strategy", "slot": "budget", "strategy": "max_value"},
                    {
                        "op": "patch_slot_schema",
                        "slot": "budget",
                        "patch": {"max": "20000"},
                    },
                    {
                        "op": "append",
                        "event": event(
                            "S002", 6, "user", "仍想回到收集状态", 5300, requested_target="collecting"
                        ),
                    },
                ],
            },
        ],
    }


def fault_injections() -> dict:
    return {
        "description": "独立异常注入；默认仅 fault demo 启用",
        "injections": [
            {
                "id": "INJ-UNKNOWN-TARGET",
                "batch": 1,
                "session_id": "S001",
                "turn": 4,
                "type": "unknown_transition_target",
                "target": "teleported",
            },
            {
                "id": "INJ-ILLEGAL-SLOT",
                "batch": 2,
                "session_id": "S002",
                "turn": 2,
                "type": "illegal_slot_value",
                "slot": "party_size",
                "value": 999,
            },
            {
                "id": "INJ-TURN-JUMP",
                "batch": 3,
                "session_id": "S003",
                "turn": 1,
                "type": "turn_jump",
                "offset": 2,
            },
            {
                "id": "INJ-QUOTA",
                "batch": 3,
                "session_id": "S005",
                "turn": 3,
                "type": "quota_overflow",
                "quota": 1,
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic offline conversation samples")
    parser.add_argument("--output", type=Path, default=ROOT / "samples" / "scenario_01")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    events = build_events(rng)
    atomic_write_text(
        output / "events.jsonl",
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in events),
    )
    atomic_write_text(
        output / "state_machine.yaml",
        yaml.safe_dump(machine_definition(), allow_unicode=True, sort_keys=False),
    )
    atomic_write_json(output / "pipeline_config.json", pipeline_config())
    atomic_write_json(output / "edit_plan.json", edit_plan())
    atomic_write_json(output / "fault_injections.json", fault_injections())
    atomic_write_json(
        output / "meta.json",
        {
            "generator": "scripts/generate_samples.py",
            "seed": args.seed,
            "event_count": len(events),
            "session_count": len({item["session_id"] for item in events}),
            "controlled_vocabulary_size": sum(len(values) for values in VOCAB.values()),
            "edit_batch_count": len(edit_plan()["batches"]),
        },
    )
    print(f"generated {len(events)} events at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
