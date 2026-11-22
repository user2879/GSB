#!/usr/bin/env python3
"""Generate deterministic base and fault-injection datasets."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

from search_intent_workbench.canonical import file_digest


DEFAULT_SEED = 20261123


def _write_json(path: Path, value: Any, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=allow_nan,
        )
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_data(seed: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    vocabulary = [
        "iPhone",
        "15",
        "手机",
        "壳",
        "保护套",
        "Apple",
        "USB",
        "ＵＳＢ",
        "充电器",
        "数据线",
        "无线",
        "蓝牙",
        "耳机",
        "降噪",
        "跑步",
        "鞋",
        "女士",
        "男士",
        "运动",
        "新款",
        "联想",
        "笔记本",
        "电脑",
        "游戏",
        "红色",
        "黑色",
        "连衣裙",
        "夏季",
        "退货",
        "怎么",
        "发票",
        "保修",
        "便宜",
        "旗舰店",
        "正品",
        "儿童",
        "玩具",
        "乐高",
        "积木",
        "客厅",
        "AirPods",
        "Pro",
        "Type-C",
        "快充",
        "20W",
        "索尼",
        "WH-1000XM5",
        "通勤",
        "佳能",
        "相机",
        "镜头",
        "旅行",
        "轻薄",
        "办公",
        "申请",
        "对比",
        "哪个",
        "区别",
        "好",
        " 无线  耳机 ",
    ]
    rng.shuffle(vocabulary)

    config: dict[str, Any] = {
        "version": "intent-config-v1",
        "vocabulary": vocabulary,
        "intent_categories": [
            {
                "name": "product_search",
                "weight": 1.0,
                "features": [
                    {"term": "手机", "weight": 1.6},
                    {"term": "耳机", "weight": 1.5},
                    {"term": "壳", "weight": 1.1},
                    {"term": "连衣裙", "weight": 1.4},
                    {"term": "笔记本", "weight": 1.7},
                    {"term": "积木", "weight": 1.4},
                    {"term": "相机", "weight": 1.4},
                    {"term": "充电器", "weight": 1.3},
                    {"term": "数据线", "weight": 1.2},
                    {"term": "鞋", "weight": 1.2},
                    {"term": "镜头", "weight": 1.2},
                    {"term": "AirPods", "weight": 1.5},
                    {"term": "索尼", "weight": 1.1},
                ],
            },
            {
                "name": "browsing",
                "weight": 0.9,
                "features": [
                    {"term": "新款", "weight": 1.1},
                    {"term": "便宜", "weight": 0.9},
                    {"term": "正品", "weight": 0.8},
                    {"term": "旗舰店", "weight": 0.7},
                ],
            },
            {
                "name": "comparison",
                "weight": 1.0,
                "features": [
                    {"term": "对比", "weight": 1.8},
                    {"term": "哪个", "weight": 1.3},
                    {"term": "区别", "weight": 1.4},
                ],
            },
            {
                "name": "scenario_recommendation",
                "weight": 1.05,
                "features": [
                    {"term": "通勤", "weight": 1.2},
                    {"term": "旅行", "weight": 1.1},
                    {"term": "办公", "weight": 1.0},
                    {"term": "客厅", "weight": 0.9},
                    {"term": "夏季", "weight": 1.0},
                    {"term": "女士", "weight": 0.7},
                    {"term": "儿童", "weight": 0.8},
                ],
            },
            {
                "name": "support",
                "weight": 0.95,
                "features": [
                    {"term": "退货", "weight": 2.0},
                    {"term": "保修", "weight": 1.7},
                    {"term": "发票", "weight": 1.8},
                    {"term": "申请", "weight": 0.8},
                    {"term": "怎么", "weight": 0.6},
                ],
            },
        ],
        "entity_types": [
            "attribute",
            "audience",
            "brand",
            "category",
            "channel",
            "color",
            "product",
            "scenario",
            "service",
            "spec",
            "usage",
        ],
        "entities": [
            {"entry_id": "e_apple_iphone15", "term": "Apple iPhone 15", "entity_type": "product", "priority": 110},
            {"entry_id": "e_iphone15", "term": "iPhone 15", "entity_type": "product", "priority": 100},
            {"entry_id": "e_iphone15_new", "term": "iPhone 15 新款", "entity_type": "product", "priority": 115},
            {"entry_id": "e_phone_case", "term": "手机 壳", "entity_type": "category", "priority": 90},
            {"entry_id": "e_phone_cover", "term": "保护套", "entity_type": "category", "priority": 70},
            {"entry_id": "e_apple", "term": "Apple", "entity_type": "brand", "priority": 70},
            {"entry_id": "e_usb_c", "term": "USB Type-C", "entity_type": "spec", "priority": 95},
            {"entry_id": "e_fast_charge_cable", "term": "快充 数据线", "entity_type": "category", "priority": 85},
            {"entry_id": "e_wireless_headphones", "term": "无线 蓝牙 耳机", "entity_type": "product", "priority": 105},
            {"entry_id": "e_wireless_earphones", "term": "无线 耳机", "entity_type": "category", "priority": 80},
            {"entry_id": "e_wireless_basic", "term": "无线", "entity_type": "attribute", "priority": 40},
            {"entry_id": "e_noise_cancel", "term": "降噪", "entity_type": "attribute", "priority": 60},
            {"entry_id": "e_commute", "term": "通勤", "entity_type": "scenario", "priority": 50},
            {"entry_id": "e_running_shoes", "term": "跑步 鞋", "entity_type": "category", "priority": 88},
            {"entry_id": "e_women", "term": "女士", "entity_type": "audience", "priority": 40},
            {"entry_id": "e_lenovo_laptop", "term": "联想 笔记本", "entity_type": "product", "priority": 102},
            {"entry_id": "e_gaming_laptop", "term": "游戏 笔记本", "entity_type": "product", "priority": 95},
            {"entry_id": "e_red_dress", "term": "红色 夏季 连衣裙", "entity_type": "product", "priority": 108},
            {"entry_id": "e_summer", "term": "夏季", "entity_type": "scenario", "priority": 50},
            {"entry_id": "e_flagship", "term": "旗舰店", "entity_type": "channel", "priority": 55},
            {"entry_id": "e_invoice", "term": "发票", "entity_type": "service", "priority": 70},
            {"entry_id": "e_warranty", "term": "保修", "entity_type": "service", "priority": 70},
            {"entry_id": "e_return", "term": "退货", "entity_type": "service", "priority": 75},
            {"entry_id": "e_cheap", "term": "便宜", "entity_type": "attribute", "priority": 45},
            {"entry_id": "e_authentic", "term": "正品", "entity_type": "attribute", "priority": 45},
            {"entry_id": "e_sony_xm5", "term": "索尼 WH-1000XM5", "entity_type": "product", "priority": 106},
            {"entry_id": "e_canon_camera", "term": "佳能 相机", "entity_type": "product", "priority": 104},
            {"entry_id": "e_travel_lens", "term": "旅行 镜头", "entity_type": "category", "priority": 82},
            {"entry_id": "e_lego_bricks", "term": "乐高 积木", "entity_type": "product", "priority": 101},
            {"entry_id": "e_kids", "term": "儿童", "entity_type": "audience", "priority": 40},
            {"entry_id": "e_living_room", "term": "客厅", "entity_type": "scenario", "priority": 50},
            {"entry_id": "e_airpods_pro", "term": "AirPods Pro", "entity_type": "product", "priority": 107},
            {"entry_id": "e_office", "term": "办公", "entity_type": "scenario", "priority": 50},
            {"entry_id": "e_light_office", "term": "轻薄 办公", "entity_type": "usage", "priority": 65},
        ],
        "normalization": {
            "rule_ids": ["width_fold", "case_fold", "whitespace_collapse", "token_split"]
        },
        "intent_scoring": {
            "position_decay": 0.12,
            "decimal_places": 6,
            "quantum": "0.000001",
        },
        "max_workers": 4,
    }

    rules: dict[str, Any] = {
        "version": "rules-v1",
        "normalization_rules": [
            {"id": "width_fold", "kind": "width_fold", "enabled": True, "revision": 1},
            {"id": "case_fold", "kind": "case_fold", "enabled": True, "revision": 1},
            {"id": "whitespace_collapse", "kind": "whitespace_collapse", "enabled": True, "revision": 1},
            {"id": "token_split", "kind": "token_split", "enabled": True, "revision": 1},
        ],
        "rewrite_rules": [
            {
                "id": "rw_product",
                "intent": "product_search",
                "priority": 10,
                "revision": 1,
                "template": "搜索并推荐 {entity:product} {entity:category}，优先展示正品与旗舰店",
            },
            {
                "id": "rw_browsing",
                "intent": "browsing",
                "priority": 10,
                "revision": 1,
                "template": "浏览“{normalized_query}”相关的新款与优惠商品",
            },
            {
                "id": "rw_comparison",
                "intent": "comparison",
                "priority": 10,
                "revision": 1,
                "template": "对比 {entity:product} 的规格、价格与口碑，输出差异表",
            },
            {
                "id": "rw_scenario",
                "intent": "scenario_recommendation",
                "priority": 10,
                "revision": 1,
                "template": "围绕 {entity:scenario} 场景推荐 {entity:category}，兼顾 {entity:audience}",
            },
            {
                "id": "rw_support",
                "intent": "support",
                "priority": 10,
                "revision": 1,
                "template": "进入售后支持：“{normalized_query}”；关联订单、{entity:service} 与处理时效",
            },
        ],
    }

    queries: list[dict[str, Any]] = [
        {"query_id": "q001", "terms": ["Apple", "iPhone", "15", "手机", "壳"], "session_id": "s1", "turn": 1, "intent_label": "product_search"},
        {"query_id": "q002", "terms": ["ＵＳＢ", "Type-C", "快充", "数据线"], "session_id": "s1", "turn": 2, "intent_label": "product_search"},
        {"query_id": "q003", "terms": ["无线", "蓝牙", "耳机", "降噪", "通勤"], "session_id": "s1", "turn": 3, "intent_label": "product_search"},
        {"query_id": "q004", "terms": ["女士", "跑步", "鞋", "新款"], "session_id": "s2", "turn": 1, "intent_label": "product_search"},
        {"query_id": "q005", "terms": ["联想", "笔记本", "游戏", "办公", "轻薄"], "session_id": "s2", "turn": 2, "intent_label": "product_search"},
        {"query_id": "q006", "terms": ["红色", "夏季", "连衣裙", "旗舰店"], "session_id": "s3", "turn": 1, "intent_label": "product_search"},
        {"query_id": "q007", "terms": ["发票", "怎么", "申请"], "session_id": "s4", "turn": 1, "intent_label": "support"},
        {"query_id": "q008", "terms": ["退货", "保修", "怎么", "申请"], "session_id": "s4", "turn": 2, "intent_label": "support"},
        {"query_id": "q009", "terms": ["便宜", "正品", "iPhone", "15"], "session_id": "s5", "turn": 1, "intent_label": "browsing"},
        {"query_id": "q010", "terms": ["索尼", "WH-1000XM5", "降噪", "耳机"], "session_id": "s5", "turn": 2, "intent_label": "product_search"},
        {"query_id": "q011", "terms": ["佳能", "相机", "旅行", "镜头"], "session_id": "s6", "turn": 1, "intent_label": "product_search"},
        {"query_id": "q012", "terms": ["儿童", "乐高", "积木", "客厅"], "session_id": "s6", "turn": 2, "intent_label": "scenario_recommendation"},
        {"query_id": "q013", "terms": ["AirPods", "Pro", "对比", "索尼", "WH-1000XM5"], "session_id": "s7", "turn": 1, "intent_label": "comparison"},
        {"query_id": "q014", "terms": ["无线", "耳机", "哪个", "降噪", "好"], "session_id": "s7", "turn": 2, "intent_label": "comparison"},
        {"query_id": "q015", "terms": [" 无线  耳机 "], "session_id": "s8", "turn": 1, "intent_label": "product_search"},
        {"query_id": "q016", "terms": ["iPhone", "15", "新款", "旗舰店"], "session_id": "s8", "turn": 2, "intent_label": "browsing"},
    ]
    rng.shuffle(queries)

    commands: dict[str, Any] = {
        "version": "commands-v1",
        "commands": [
            {
                "command_id": "cmd001",
                "type": "add_query",
                "query": {"query_id": "q017", "terms": ["AirPods", "Pro", "无线", "耳机", "通勤"], "session_id": "s9", "turn": 1, "intent_label": "product_search"},
            },
            {
                "command_id": "cmd002",
                "type": "remove_query",
                "query_id": "q006",
            },
            {
                "command_id": "cmd003",
                "type": "update_normalization_rule",
                "rule_id": "case_fold",
                "patch": {"enabled": False},
            },
            {
                "command_id": "cmd004",
                "type": "update_intent_weights",
                "weights": {"product_search": 1.35, "support": 1.0},
            },
            {
                "command_id": "cmd005",
                "type": "replace_entity_dictionary_entry",
                "entry_id": "e_iphone15",
                "entry": {"entry_id": "e_iphone15", "term": "iPhone 15", "entity_type": "product", "priority": 120},
            },
            {
                "command_id": "cmd006",
                "type": "replace_entity_dictionary_entry",
                "entry_id": "e_airpods_pro",
                "entry": {"entry_id": "e_airpods_pro", "term": "AirPods Pro 2", "entity_type": "product", "priority": 107},
            },
            {
                "command_id": "cmd007",
                "type": "add_query",
                "query": {"query_id": "q018", "terms": ["索尼", "降噪", "耳机", "办公"], "session_id": "s9", "turn": 2, "intent_label": "product_search"},
            },
            {
                "command_id": "cmd008",
                "type": "update_normalization_rule",
                "rule_id": "whitespace_collapse",
                "patch": {"enabled": False},
            },
        ],
    }
    return config, rules, queries, commands


def generate(output_root: Path, seed: int) -> dict[str, Any]:
    base_dir = output_root / "base"
    injected_dir = output_root / "injected"
    config, rules, queries, commands = build_data(seed)

    _write_json(base_dir / "intent_config.json", config)
    _write_json(base_dir / "normalization_rules.json", rules)
    _write_jsonl(base_dir / "queries.jsonl", queries)
    _write_json(base_dir / "commands.json", commands)

    fault_config_weight = copy.deepcopy(config)
    for intent in fault_config_weight["intent_categories"]:
        if intent["name"] == "comparison":
            intent["weight"] = float("inf")
    _write_json(
        injected_dir / "I01_non_finite_weight_config.json",
        fault_config_weight,
        allow_nan=True,
    )

    fault_config_duplicate = copy.deepcopy(config)
    duplicate = copy.deepcopy(fault_config_duplicate["entities"][0])
    duplicate["entry_id"] = "e_duplicate_iphone15"
    fault_config_duplicate["entities"].append(duplicate)
    _write_json(
        injected_dir / "I02_duplicate_entity_config.json",
        fault_config_duplicate,
    )

    turn_gap_queries = copy.deepcopy(queries)
    turn_gap_queries.append(
        {"query_id": "q_fault_turn", "terms": ["发票", "退货"], "session_id": "s3", "turn": 3, "intent_label": "support"}
    )
    _write_jsonl(injected_dir / "I03_turn_gap_queries.jsonl", turn_gap_queries)

    oov_queries = copy.deepcopy(queries)
    oov_queries.append(
        {"query_id": "q_fault_oov", "terms": ["iPhone", "量子", "15"], "session_id": "s10", "turn": 1, "intent_label": "product_search"}
    )
    _write_jsonl(injected_dir / "I04_oov_queries.jsonl", oov_queries)

    injections = {
        "version": "fault-suite-v1",
        "items": [
            {
                "injection_id": "I01",
                "kind": "non_finite_intent_weight",
                "config_path": "I01_non_finite_weight_config.json",
                "rules_path": "../base/normalization_rules.json",
                "queries_path": "../base/queries.jsonl",
                "expected_codes": ["E_CONFIG_INVALID"],
            },
            {
                "injection_id": "I02",
                "kind": "duplicate_entity_dictionary_entry",
                "config_path": "I02_duplicate_entity_config.json",
                "rules_path": "../base/normalization_rules.json",
                "queries_path": "../base/queries.jsonl",
                "expected_codes": ["E_CONFIG_INVALID"],
            },
            {
                "injection_id": "I03",
                "kind": "session_turn_gap",
                "config_path": "../base/intent_config.json",
                "rules_path": "../base/normalization_rules.json",
                "queries_path": "I03_turn_gap_queries.jsonl",
                "expected_codes": ["E_QUERY_INVALID"],
            },
            {
                "injection_id": "I04",
                "kind": "out_of_vocabulary_term",
                "config_path": "../base/intent_config.json",
                "rules_path": "../base/normalization_rules.json",
                "queries_path": "I04_oov_queries.jsonl",
                "expected_codes": ["E_QUERY_INVALID"],
            },
        ],
    }
    _write_json(injected_dir / "injections.json", injections)

    manifest = {
        "seed": seed,
        "dependency": "numpy (runtime classifier only)",
        "network_required": False,
        "base_files": {
            name: {"sha256": file_digest(base_dir / name)}
            for name in (
                "queries.jsonl",
                "intent_config.json",
                "normalization_rules.json",
                "commands.json",
            )
        },
        "fault_files": {
            name: {"sha256": file_digest(injected_dir / name)}
            for name in (
                "injections.json",
                "I01_non_finite_weight_config.json",
                "I02_duplicate_entity_config.json",
                "I03_turn_gap_queries.jsonl",
                "I04_oov_queries.jsonl",
            )
        },
        "counts": {"queries": len(queries), "commands": len(commands["commands"]), "injections": 4},
        "rng_usage": "fixed-seed Fisher-Yates shuffle of vocabulary and JSONL query order; semantic labels/terms are fixed",
        "non_finite_encoding": "I01 uses Python JSON Infinity; the loader rejects it as non-finite",
    }
    _write_json(output_root / "dataset_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", help="dataset output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = generate(Path(args.out), args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
