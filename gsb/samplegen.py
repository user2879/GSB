from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


SEED = 383


def vocabulary() -> dict[str, Any]:
    terms: list[dict[str, Any]] = []

    def add(term: str, order: int, kind: str = "neutral", **extra: Any) -> None:
        row = {"term": term, "order": order, "kind": kind}
        row.update(extra)
        terms.append(row)

    positives = [
        ("满意", 88), ("及时", 64), ("专业", 72), ("顺利", 80), ("友好", 56),
        ("解决", 96), ("退款成功", 120), ("主动", 48), ("耐心", 32),
    ]
    negatives = [
        ("延迟", 92), ("故障", 128), ("投诉", 140), ("损坏", 132), ("等待", 60),
        ("推诿", 116), ("失败", 124), ("卡顿", 84),
    ]
    for index, (term, evidence) in enumerate(positives, 1):
        add(term, index, "sentiment", polarity="positive", evidence_points=evidence)
    for index, (term, evidence) in enumerate(negatives, 20):
        add(term, index, "sentiment", polarity="negative", evidence_points=evidence)

    neutral_terms = [
        "客服", "订单", "系统", "处理", "物流", "商品", "用户", "反馈", "页面", "电话",
        "原来", "完全", "几乎", "从来", "知道", "说", "很久", "问题", "并", "要求",
        "支付", "导致", "这次", "完毕", "就", "帮助", "且", "指导", "了", "很",
    ]
    for index, term in enumerate(neutral_terms, 40):
        add(term, index, "neutral")
    for index, term in enumerate(["但是", "然而", "不过", "反而", "偏偏", "谁知"], 70):
        add(term, index, "discourse")
    for index, term in enumerate(["不", "没", "未", "无", "别", "毫不"], 80):
        add(term, index, "neutral")
    for index, term in enumerate(["，", "。", "！", "？", "；"], 95):
        add(term, index, "punctuation")
    return {"version": "vocab-383", "seed": SEED, "terms": terms}


def triggers() -> dict[str, Any]:
    rows = [
        ("NEG_BU", "不", 1),
        ("NEG_MEI", "没", 2),
        ("NEG_WEI", "未", 3),
        ("NEG_WU", "无", 4),
        ("NEG_BIE", "别", 5),
    ]
    return {"version": "neg-383", "seed": SEED, "triggers": [
        {"trigger_id": ident, "term": term, "order": order} for ident, term, order in rows
    ]}


def conflicts() -> dict[str, Any]:
    rows = [
        ("CONF_DANSHI", "但是", 1),
        ("CONF_RANER", "然而", 2),
        ("CONF_BUGUO", "不过", 3),
        ("CONF_FANER", "反而", 4),
        ("CONF_PIANPIAN", "偏偏", 5),
    ]
    return {"version": "conflict-383", "seed": SEED, "conflicts": [
        {"conflict_id": ident, "term": term, "order": order} for ident, term, order in rows
    ]}


def channels() -> dict[str, Any]:
    return {"version": "channel-383", "channels": [
        {"channel_id": "app", "weight_permille": 1000},
        {"channel_id": "web", "weight_permille": 938},
        {"channel_id": "miniapp", "weight_permille": 850},
        {"channel_id": "hotline", "weight_permille": 980},
    ]}


def config(seed: int = SEED) -> dict[str, Any]:
    return {
        "seed": seed,
        "global_version": 1,
        "scope_max_tokens": 6,
        "cancellation_strength_points": 42.375,
        "score_lower_bound_points": 0,
        "length_penalty_bp_per_token": 750,
        "length_min_retention_bp": 5200,
        "positive_threshold_points": 30.015,
        "negative_threshold_points": -30.015,
        "risk_thresholds_points": {"medium": 17.504, "high": 48.0, "critical": 82.0},
        "fixed_point": {
            "bit_width": 64,
            "decimal_width": 6,
            "quantization_gear": "fine",
            "rounding_mode": "half_up",
        },
        "error_tolerances_points": {
            "scope_assignment": 0.0,
            "cancellation": 0.0005,
            "length_penalty": 0.0005,
            "composition": 0.002,
        },
        "boundary_half_step_decimal_width": 2,
        "batch_budget_points": 24,
        "max_workers": 4,
    }


def _special_token_patterns() -> list[list[str]]:
    return [
        ["完全", "不", "知道", "没", "解决", "但是", "退款成功", "。"],
        ["订单", "未", "及时", "处理", "反而", "延迟", "了", "。"],
        ["客服", "说", "别", "等待", "不过", "系统", "故障", "。"],
        ["物流", "延迟", "。", "客服", "耐心", "解决", "问题", "。"],
        ["不", "友好", "不过", "然而", "反而", "。"],
        ["页面", "无", "卡顿", "但是", "支付", "失败", "。"],
        ["商品", "损坏", "；", "用户", "投诉", "并", "要求", "退款成功", "。"],
        ["电话", "客服", "没", "专业", "反馈", "等待", "很久", "。"],
        ["几乎", "没", "等待", "就", "顺利", "处理", "完毕", "！"],
        ["从来", "不", "推诿", "然而", "这次", "订单", "延迟", "。"],
        ["退款成功", "很", "顺利", "，", "客服", "主动", "且", "专业", "。"],
        ["系统", "故障", "导致", "卡顿", "，", "用户", "反馈", "失败", "。"],
        ["耐心", "指导", "。"],
        ["客服", "主动", "帮助", "，", "问题", "顺利", "解决", "。"],
        ["不", "卡顿", "很久", "没", "等待", "反馈", "失败", "处理", "投诉", "。"],
    ]


def generate_tickets(seed: int = SEED) -> dict[str, Any]:
    rng = random.Random(seed)
    channels = ["app", "web", "miniapp", "hotline"]
    statuses = ["closed", "closed", "closed", "open"]
    patterns = _special_token_patterns()

    neutral_pool = ["客服", "订单", "系统", "处理", "物流", "商品", "用户", "反馈", "页面", "电话"]
    hit_pool = ["满意", "及时", "专业", "顺利", "友好", "延迟", "故障", "等待", "推诿", "失败", "卡顿"]
    trigger_pool = ["不", "没", "未", "无"]
    conflict_pool = ["但是", "然而", "不过", "反而"]

    while len(patterns) < 18:
        tokens: list[str] = [rng.choice(neutral_pool)]
        if rng.random() < 0.75:
            tokens.append(rng.choice(trigger_pool))
        for _ in range(rng.randint(2, 5)):
            tokens.append(rng.choice(hit_pool if rng.random() < 0.7 else neutral_pool))
            if rng.random() < 0.25:
                tokens.append(rng.choice(conflict_pool))
        tokens.append(rng.choice(["。", "！", "？"]))
        patterns.append(tokens)

    tickets: list[dict[str, Any]] = []
    for index, tokens in enumerate(patterns, 1):
        tickets.append({
            "ticket_id": f"TK-{index:04d}",
            "tokens": tokens,
            "channel_id": rng.choice(channels),
            "annotator_version": rng.randint(1, 3),
            "completion_status": rng.choice(statuses),
        })
    # Pinned fixtures make threshold/half-step acceptance checks reproducible.
    for ticket in tickets:
        if ticket["ticket_id"] == "TK-0002":
            ticket["channel_id"] = "app"
            ticket["annotator_version"] = 2
            ticket["completion_status"] = "closed"
        if ticket["ticket_id"] == "TK-0013":
            ticket["channel_id"] = "web"
            ticket["annotator_version"] = 2
            ticket["completion_status"] = "closed"
    return {"seed": seed, "tickets": tickets}


def edit_stream() -> list[dict[str, Any]]:
    return [
        {"seq": 1, "op": "replace_ticket", "ticket": {
            "ticket_id": "TK-0014",
            "tokens": ["客服", "未", "及时", "处理", "但是", "顺利", "解决", "。"],
            "channel_id": "app",
            "annotator_version": 4,
            "completion_status": "closed",
        }},
        {"seq": 2, "op": "upsert_neg_trigger", "trigger": {
            "trigger_id": "NEG_HAOBU", "term": "毫不", "order": 6
        }},
        {"seq": 3, "op": "upsert_neg_trigger", "trigger": {
            "trigger_id": "NEG_BIE", "term": "别", "order": 9
        }},
        {"seq": 4, "op": "upsert_conflict", "conflict": {
            "conflict_id": "CONF_SHUIZHI", "term": "谁知", "order": 6
        }},
        {"seq": 5, "op": "set_channel_weight", "channel_id": "miniapp", "weight_permille": 880},
        {"seq": 6, "op": "update_config", "patch": {"length_penalty_bp_per_token": 700}},
    ]


def build_sample(seed: int = SEED) -> dict[str, Any]:
    return {
        "controlled_vocabulary.json": vocabulary(),
        "negation_triggers.json": triggers(),
        "polarity_conflicts.json": conflicts(),
        "channel_weights.json": channels(),
        "config.json": config(seed),
        "tickets.json": generate_tickets(seed),
        "edits.jsonl": edit_stream(),
    }


def generate_sample_dir(target: str | Path, seed: int = SEED) -> Path:
    root = Path(target)
    root.mkdir(parents=True, exist_ok=True)
    for filename, payload in build_sample(seed).items():
        path = root / filename
        if filename.endswith(".jsonl"):
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in payload) + "\n", encoding="utf-8")
        else:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root
