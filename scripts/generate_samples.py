#!/usr/bin/env python3
"""Generate deterministic local fixtures for IntentLab."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
SEED = 20261127


VOCABULARY = [
    "北京",
    "上海",
    "深圳",
    "杭州",
    "今天",
    "明天",
    "后天",
    "天气",
    "预报",
    "气温",
    "附近",
    "周边",
    "咖啡",
    "店",
    "餐厅",
    "酒店",
    "药店",
    "播放",
    "音乐",
    "歌曲",
    "周杰伦",
    "陈奕迅",
    "导航",
    "去",
    "怎么",
    "走",
    "路线",
    "机场",
    "火车站",
    "预订",
    "房间",
    "价格",
    "优惠",
    "入住",
    "搜索",
    "新闻",
    "股票",
    "行情",
    "翻译",
    "英文",
    "中文",
    "句子",
    "帮我",
    "请",
    "想",
    "查",
    "一下",
    "打开",
    "设置",
    "提醒",
    "会议",
]


TRIGGERS = [
    ("t_weather_city", ("北京", "天气"), "weather", 1000, ["city", "weather"]),
    ("t_weather_tomorrow", ("明天", "天气"), "weather", 900, ["date", "weather"]),
    ("t_nearby_coffee", ("附近", "咖啡", "店"), "nearby", 1100, ["poi"]),
    ("t_music_play", ("播放", "音乐"), "music", 1000, ["action", "media"]),
    ("t_nav_airport", ("导航", "去", "机场"), "navigation", 1100, ["destination"]),
]


INTENT_TRIGGERS = {
    "weather": [("北京", "天气"), ("明天", "天气")],
    "nearby": [("附近", "咖啡", "店"), ("周边", "餐厅")],
    "music": [("播放", "周杰伦"), ("播放", "音乐")],
    "navigation": [("导航", "去", "机场")],
    "hotel": [("预订", "酒店")],
    "news": [("股票", "行情")],
    "translate": [("翻译", "英文")],
}


DECOYS = {
    "weather": ["nearby", "news"],
    "nearby": ["weather", "hotel"],
    "music": ["translate", "news"],
    "navigation": ["nearby", "weather"],
    "hotel": ["navigation", "nearby"],
    "news": ["translate", "weather"],
    "translate": ["music", "news"],
}


SESSION_INTENTS = {
    "s01": ["weather", "weather", "weather"],
    "s02": ["nearby", "nearby", "music"],
    "s03": ["navigation", "navigation", "hotel"],
    "s04": ["hotel", "hotel", "hotel"],
    "s05": ["news", "news", "translate"],
    "s06": ["nearby", "nearby", "nearby"],
    "s07": ["music", "music", "weather"],
    "s08": ["translate", "translate", "navigation"],
}


FILLERS = ["帮我", "请", "想", "查", "一下", "搜索", "打开", "设置", "提醒", "会议"]


def slots_for(intent: str, trigger: tuple[str, ...], start: int) -> list[dict[str, Any]]:
    if intent == "weather":
        first = trigger[0]
        if first in {"北京", "上海", "深圳", "杭州"}:
            return [
                {"name": "city", "start": start, "end": start + 1},
                {"name": "weather", "start": start + 1, "end": start + 2},
            ]
        return [
            {"name": "date", "start": start, "end": start + 1},
            {"name": "weather", "start": start + 1, "end": start + 2},
        ]
    if intent == "nearby":
        return [{"name": "poi", "start": start, "end": start + len(trigger)}]
    if intent == "music":
        return [
            {"name": "action", "start": start, "end": start + 1},
            {"name": "media", "start": start + 1, "end": start + len(trigger)},
        ]
    if intent == "navigation":
        return [
            {"name": "route", "start": start, "end": start + 1},
            {"name": "destination", "start": start + 2, "end": start + 3},
        ]
    if intent == "hotel":
        return [
            {"name": "action", "start": start, "end": start + 1},
            {"name": "lodging", "start": start + 1, "end": start + 2},
        ]
    if intent == "news":
        return [{"name": "topic", "start": start, "end": start + 2}]
    if intent == "translate":
        return [
            {"name": "action", "start": start, "end": start + 1},
            {"name": "target_language", "start": start + 1, "end": start + 2},
        ]
    return []


def build_query(
    rng: random.Random,
    *,
    query_id: str,
    session_id: str,
    turn: int,
    source: str,
    intent: str,
    trigger_index: int,
    before_count: int,
    after_count: int,
    conflict_slot: bool = False,
) -> dict[str, Any]:
    trigger = INTENT_TRIGGERS[intent][trigger_index % len(INTENT_TRIGGERS[intent])]
    before = [rng.choice(FILLERS) for _ in range(before_count)]
    after = [rng.choice(FILLERS) for _ in range(after_count)]
    terms = before + list(trigger) + after
    start = len(before)
    slots = slots_for(intent, trigger, start)
    if conflict_slot and len(trigger) >= 3:
        slots.append({"name": "wide_fragment", "start": start, "end": start + len(trigger)})
    candidates = [intent] + DECOYS[intent]
    return {
        "query_id": query_id,
        "text": " ".join(terms),
        "terms": terms,
        "session_id": session_id,
        "turn": turn,
        "source": source,
        "ground_truth_intent": intent,
        "candidates": candidates,
        "slots": slots,
        "priority": turn,
    }


def generate_queries() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    queries: list[dict[str, Any]] = []
    number = 1
    for session_id in sorted(SESSION_INTENTS):
        for turn, intent in enumerate(SESSION_INTENTS[session_id]):
            query_id = f"q{number:03d}"
            source = ["source-a", "source-b", "source-c"][(number - 1) % 3]
            before_count = rng.randint(0, 2)
            after_count = rng.randint(1, 2)
            conflict_slot = session_id == "s06" and turn == 2
            if conflict_slot:
                before_count = 2
                after_count = 3
            trigger_index = turn if intent in {"weather", "nearby"} else 0
            queries.append(
                build_query(
                    rng,
                    query_id=query_id,
                    session_id=session_id,
                    turn=turn,
                    source=source,
                    intent=intent,
                    trigger_index=trigger_index,
                    before_count=before_count,
                    after_count=after_count,
                    conflict_slot=conflict_slot,
                )
            )
            number += 1
    return queries


def config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "global_version": 1,
        "allowed": {
            "fraction_bits": [8, 12, 16, 20],
            "quantums": [1, 4, 16, 32, 64],
            "weights": [600, 700, 800, 900, 1000, 1100, 1200],
        },
        "fixed_point": {"fraction_bits": 12, "quantum": 16},
        "thresholds": {
            "default": 120,
            "tie_rule": "accept_on_tie",
            "intents": {
                "weather": 150,
                "nearby": 150,
                "music": 145,
                "navigation": 150,
                "hotel": 145,
                "news": 145,
                "translate": 140,
            },
        },
        "diagnostics": {"tolerance": 0.0005, "reference_extra_bits": 4},
        "scheduler": {
            "workers": 4,
            "match_budget_points": 100000,
            "source_quotas": {"source-a": 100, "source-b": 100, "source-c": 100},
            "batch_query_limit": 1000,
        },
        "merge": {"max_turn_gap": 1, "term_gap": 1},
        "fallback_intent": "unknown",
    }


def revision(
    revision_id: str,
    parent_id: str | None,
    seq: int,
    label: str,
    ops: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "revision_id": revision_id,
        "parent_id": parent_id,
        "seq": seq,
        "label": label,
        "ops": ops,
    }


def rules() -> dict[str, Any]:
    revisions = [
        revision(
            "r00",
            None,
            0,
            "base",
            [
                {
                    "op": "add",
                    "trigger_id": trigger_id,
                    "terms": list(terms),
                    "intent_id": intent,
                    "weight": weight,
                    "slot_pattern": pattern,
                }
                for trigger_id, terms, intent, weight, pattern in TRIGGERS
            ],
        ),
        revision(
            "r01",
            "r00",
            1,
            "add-nearby-hotel-and-adjust-weather",
            [
                {
                    "op": "add",
                    "trigger_id": "t_nearby_restaurant",
                    "terms": ["周边", "餐厅"],
                    "intent_id": "nearby",
                    "weight": 1000,
                    "slot_pattern": ["poi"],
                },
                {
                    "op": "add",
                    "trigger_id": "t_hotel_book",
                    "terms": ["预订", "酒店"],
                    "intent_id": "hotel",
                    "weight": 1100,
                    "slot_pattern": ["action", "lodging"],
                },
                {
                    "op": "update",
                    "trigger_id": "t_weather_city",
                    "weight": 1100,
                    "slot_pattern": ["city", "weather", "forecast"],
                },
            ],
        ),
        revision(
            "r02",
            "r01",
            2,
            "add-news-translate-and-route",
            [
                {
                    "op": "add",
                    "trigger_id": "t_news_stock",
                    "terms": ["股票", "行情"],
                    "intent_id": "news",
                    "weight": 1100,
                    "slot_pattern": ["topic"],
                },
                {
                    "op": "add",
                    "trigger_id": "t_translate",
                    "terms": ["翻译", "英文"],
                    "intent_id": "translate",
                    "weight": 1000,
                    "slot_pattern": ["action", "target_language"],
                },
                {
                    "op": "add",
                    "trigger_id": "t_nav_route",
                    "terms": ["怎么", "走"],
                    "intent_id": "navigation",
                    "weight": 700,
                    "slot_pattern": ["route"],
                },
                {
                    "op": "update",
                    "trigger_id": "t_music_play",
                    "slot_pattern": ["action", "media"],
                },
            ],
        ),
        revision(
            "r03",
            "r02",
            3,
            "adjust-nearby-delete-route",
            [
                {
                    "op": "update",
                    "trigger_id": "t_nearby_coffee",
                    "weight": 1200,
                    "slot_pattern": ["poi", "brand"],
                },
                {"op": "delete", "trigger_id": "t_nav_route"},
            ],
        ),
        revision(
            "r04",
            "r03",
            4,
            "music-artist-branch",
            [
                {
                    "op": "add",
                    "trigger_id": "t_music_artist",
                    "terms": ["播放", "周杰伦"],
                    "intent_id": "music",
                    "weight": 1200,
                    "slot_pattern": ["action", "artist"],
                },
                {"op": "update", "trigger_id": "t_weather_tomorrow", "weight": 1000},
            ],
        ),
        revision(
            "r05",
            "r03",
            5,
            "hotel-price-sibling",
            [
                {
                    "op": "add",
                    "trigger_id": "t_hotel_price",
                    "terms": ["酒店", "价格"],
                    "intent_id": "hotel",
                    "weight": 800,
                    "slot_pattern": ["lodging", "price"],
                },
                {"op": "update", "trigger_id": "t_weather_city", "weight": 1000},
            ],
        ),
        revision(
            "r06",
            "r04",
            6,
            "news-weight-and-delete-old-music",
            [
                {"op": "update", "trigger_id": "t_news_stock", "weight": 1200},
                {"op": "delete", "trigger_id": "t_music_play"},
            ],
        ),
    ]
    return {
        "schema_version": 1,
        "active_revision_ids": ["r06"],
        "revisions": revisions,
    }


def edits(queries: list[dict[str, Any]]) -> dict[str, Any]:
    query = next(item for item in queries if item["query_id"] == "q010")
    replacement = json.loads(json.dumps(query))
    replacement["terms"] = ["请", "想", "播放", "周杰伦", "查", "一下"]
    replacement["text"] = " ".join(replacement["terms"])
    replacement["slots"] = [
        {"name": "action", "start": 2, "end": 3},
        {"name": "media", "start": 3, "end": 4},
    ]
    replacement["ground_truth_intent"] = "music"
    replacement["candidates"] = ["music", "translate", "news"]
    return {
        "schema_version": 1,
        "commands": [
            {"op": "replace_query", "query": replacement},
            {
                "op": "add_revision",
                "revision": revision(
                    "r07",
                    "r06",
                    7,
                    "edit-music-artist-weight",
                    [{"op": "update", "trigger_id": "t_music_artist", "weight": 1100}],
                ),
            },
            {"op": "set_active_revisions", "revision_ids": ["r07"]},
            {"op": "set_fixed_point", "fraction_bits": 16, "quantum": 32},
            {"op": "set_threshold", "intent_id": "weather", "value": 140},
        ],
    }


def invalid_queries() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "queries": [
            {
                "query_id": "good-1",
                "terms": ["北京", "天气"],
                "session_id": "bad-session",
                "turn": 0,
                "source": "source-a",
                "ground_truth_intent": "weather",
                "candidates": ["weather"],
                "slots": [{"name": "city", "start": 0, "end": 1}],
            },
            {
                "query_id": "good-1",
                "terms": ["附近", "咖啡", "店"],
                "session_id": "bad-session",
                "turn": 1,
                "source": "source-a",
                "ground_truth_intent": "nearby",
                "candidates": ["nearby"],
                "slots": [],
            },
            {
                "query_id": "bad-term",
                "terms": ["北京", "不存在"],
                "session_id": "bad-session",
                "turn": 2,
                "source": "source-a",
                "ground_truth_intent": "weather",
                "candidates": ["weather"],
                "slots": [],
            },
            {
                "query_id": "bad-slot",
                "terms": ["明天", "天气"],
                "session_id": "bad-session",
                "turn": 3,
                "source": "source-b",
                "ground_truth_intent": "weather",
                "candidates": ["weather"],
                "slots": [{"name": "date", "start": 2, "end": 2}],
            },
        ],
    }


def invalid_rules() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "active_revision_ids": ["missing"],
        "revisions": [
            revision(
                "bad-parent",
                "no-such-parent",
                0,
                "bad-parent",
                [],
            ),
            revision(
                "bad-cycle-a",
                "bad-cycle-b",
                1,
                "cycle",
                [],
            ),
            revision(
                "bad-cycle-b",
                "bad-cycle-a",
                2,
                "cycle",
                [],
            ),
            revision(
                "bad-weight",
                None,
                3,
                "bad-weight",
                [
                    {
                        "op": "add",
                        "trigger_id": "bad",
                        "terms": ["北京", "天气"],
                        "intent_id": "weather",
                        "weight": 123,
                    }
                ],
            ),
        ],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    INPUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "snapshots").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    queries = generate_queries()
    write_json(INPUT / "vocabulary.json", {"schema_version": 1, "terms": VOCABULARY})
    write_json(INPUT / "queries.json", {"schema_version": 1, "queries": queries})
    write_json(INPUT / "rules.json", rules())
    write_json(INPUT / "config.json", config())
    write_json(INPUT / "edits.json", edits(queries))
    write_json(INPUT / "invalid_queries.json", invalid_queries())
    write_json(INPUT / "invalid_rules.json", invalid_rules())
    print(f"generated seed={SEED} queries={len(queries)} into {INPUT}")


if __name__ == "__main__":
    main()
