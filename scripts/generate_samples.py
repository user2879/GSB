#!/usr/bin/env python3
"""Generate the deterministic local sample corpus and edit stream."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import yaml


SEED = 20261122
ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


ENTITY_NAMES = [
    "云海市发改委", "星河能源集团", "东港区交通局", "远山农业合作社",
    "北辰科技公司", "临江气象台", "南川区教育局", "海岳港口集团",
]
DATES = ["3月12日", "4月7日", "5月18日", "6月2日", "7月21日", "8月9日", "9月15日", "10月3日"]
TOPICS = ["城市更新", "新能源公交", "春耕保障", "数字政务", "港区物流", "校园餐管理"]
ACTIONS = ["表示", "指出", "宣布", "强调"]
VERBS = ["推进", "完成", "启动", "优化", "发布"]

VOCABULARY = [
    "云海市发改委", "星河能源集团", "东港区交通局", "远山农业合作社", "北辰科技公司",
    "临江气象台", "南川区教育局", "海岳港口集团", "新闻发布会", "城市更新", "新能源公交",
    "春耕保障", "数字政务", "港区物流", "校园餐管理", "相关负责人", "重点项目", "服务窗口",
    "居民出行", "应急物资", "调度平台", "年度计划", "现场检查", "风险隐患", "下一步",
    "表示", "指出", "宣布", "强调", "推进", "完成", "启动", "优化", "发布", "记者", "目前",
    "此外", "预计", "覆盖", "提升", "保障", "建设", "运行", "群众", "部门", "项目", "工作",
]


def _documents() -> list[dict]:
    rng = random.Random(SEED)
    documents = []
    for index in range(1, 11):
        chosen = [ENTITY_NAMES[(index + offset) % len(ENTITY_NAMES)] for offset in range(5)]
        date = DATES[(index - 1) % len(DATES)]
        topic = TOPICS[(index - 1) % len(TOPICS)]
        action = ACTIONS[(index - 1) % len(ACTIONS)]
        paragraphs = [
            f"{chosen[0]}在{date}举行新闻发布会。{chosen[0]}{action}，{topic}年度计划已经完成。"
            f"{chosen[1]}表示，重点项目将覆盖更多居民出行需求。",
            f"记者在现场看到，{chosen[2]}与{chosen[1]}共同推进调度平台建设。"
            f"{chosen[2]}指出，目前服务窗口运行平稳，风险隐患已完成排查。",
            f"此外，{chosen[3]}宣布启动应急物资保障工作。预计{date}之后，{chosen[4]}将发布现场检查结果。"
            f"{chosen[0]}强调，下一步要把群众意见纳入年度计划。",
        ]
        if index % 3 == 0:
            paragraphs.append(
                f"{chosen[4]}与{chosen[2]}表示，将围绕{topic}优化流程。"
                f"记者了解到，相关部门预计提升港区物流效率。"
            )
        documents.append({
            "document_id": f"news-{index:04d}",
            "title": f"{topic}工作推进会召开",
            "paragraphs": paragraphs,
            "entities": [
                {"id": f"e{index}-{offset + 1}", "text": name,
                 "type": ["ORG", "GOV", "COMPANY"][offset % 3]}
                for offset, name in enumerate(chosen)
            ],
            "events": [
                {"subject": chosen[0], "predicate": VERBS[(index - 1) % len(VERBS)],
                 "object": topic},
                {"subject": chosen[2], "predicate": "检查", "object": chosen[3]},
            ],
        })
    return documents


def _config() -> dict:
    v1_entries = []
    for template in ("official", "brief", "factual"):
        for source, candidates in (
            ("表示", ["指出", "强调"]),
            ("指出", ["说明", "表示"]),
            ("宣布", ["公布", "宣布"]),
            ("强调", ["重申", "强调"]),
            ("推进", ["开展", "推动"]),
            ("完成", ["办结", "完成"]),
            ("启动", ["开启", "启动"]),
            ("优化", ["改进", "优化"]),
            ("发布", ["披露", "发布"]),
            ("目前", ["当前", "眼下"]),
            ("此外", ["同时", "另外"]),
            ("记者", ["记者", "通讯员"]),
        ):
            v1_entries.append({"template_id": template, "source": source,
                               "candidates": candidates})
    # Duplicate source entries exercise the table-order tie break.
    v1_entries.insert(15, {"template_id": "brief", "source": "表示",
                           "candidates": ["称", "回应"]})
    v2_entries = copy.deepcopy(v1_entries)
    for entry in v2_entries:
        entry["candidates"] = list(reversed(entry["candidates"]))
    return {
        "schema_version": 1,
        "max_summary_sentences": 3,
        "scoring": {
            "word_frequency_weight": "0.45",
            "entity_coverage_weight": "0.35",
            "position_decay_weight": "0.20",
            "entity_coverage_min": "0.16",
            "position_decay_base": "0.96",
        },
        "fixed_decimal_places": 4,
        "thread_pool_size": 4,
        "snapshot_commit_interval_docs": 2,
        "vocabulary": VOCABULARY,
        "rewrite": {
            "style_template_id": "official",
            "synonym_table_version": "v1",
            "max_chars": 62,
        },
        "synonym_tables": {
            "v1": {
                "templates": {
                    "official": {"prefix": "【要闻】", "suffix": "", "max_chars": 60},
                    "brief": {"prefix": "简讯：", "suffix": "", "max_chars": 50},
                    "factual": {"prefix": "事实：", "suffix": "", "max_chars": 46},
                },
                "entries": v1_entries,
            },
            "v2": {
                "templates": {
                    "official": {"prefix": "【深度】", "suffix": "", "max_chars": 56},
                    "brief": {"prefix": "快讯｜", "suffix": "", "max_chars": 44},
                    "factual": {"prefix": "记录：", "suffix": "", "max_chars": 40},
                },
                "entries": v2_entries,
            },
        },
    }


def _commands(documents: list[dict]) -> list[dict]:
    replacement = copy.deepcopy(documents[3])
    replacement["title"] = "城市更新项目完成阶段验收"
    replacement["paragraphs"] = [
        "北辰科技公司发布城市更新项目阶段结果。北辰科技公司表示，调度平台已完成联调。"
        "海岳港口集团指出，服务窗口将覆盖更多企业。",
        "临江气象台与东港区交通局共同开展现场检查。东港区交通局强调，风险隐患已经完成整改。",
    ]
    return [
        {"type": "add_paragraph", "document_id": "news-0002", "index": 1,
         "paragraph": "云海市发改委表示，数字政务项目将优化居民出行服务。"},
        {"type": "set_max_summary_sentences", "value": 2},
        {"type": "set_style_template", "style_template_id": "brief"},
        {"type": "set_synonym_table_version", "version": "v2"},
        {"type": "replace_document", "document": replacement},
        {"type": "add_paragraph", "document_id": "news-9999", "paragraph": "无效目标"},
        {"type": "set_rewrite_style", "rewrite": {"style_template_id": "official", "max_chars": 30}},
        {"type": "append_batch", "batch_id": "batch-shift", "commands": [
            {"type": "remove_paragraph", "document_id": "news-0005", "index": 0},
            {"type": "add_paragraph", "document_id": "news-0005", "index": 0,
             "paragraph": "海岳港口集团表示，港区物流调度平台已经启动。"},
        ]},
        {"type": "rollback_last_batch", "batch_id": "batch-shift"},
        {"type": "remove_paragraph", "document_id": "news-0007", "index": 0},
        {"type": "set_scoring", "scoring": {"entity_coverage_min": "0.15"}},
    ]


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    documents = _documents()
    config = _config()
    commands = _commands(documents)
    with (SAMPLES / "news.jsonl").open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    with (SAMPLES / "commands.jsonl").open("w", encoding="utf-8") as handle:
        for command in commands:
            handle.write(json.dumps(command, ensure_ascii=False) + "\n")
    with (SAMPLES / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False, width=1000)

    injections = {
        "schema_version": 1,
        "injections": [
            {"id": "INJ-001", "document_id": "news-0002", "stage": "split",
             "kind": "empty_paragraph", "code": "E_DOC_INVALID"},
            {"id": "INJ-002", "document_id": "news-0003", "stage": "scoring",
             "kind": "illegal_entity", "code": "E_DOC_INVALID"},
            {"id": "INJ-003", "document_id": "news-0005", "stage": "selection",
             "kind": "zero_summary_length", "code": "E_CONFIG_INVALID"},
            {"id": "INJ-004", "document_id": "news-0007", "stage": "scoring",
             "kind": "decimal_width_out_of_range", "code": "E_CONFIG_INVALID"},
            {"id": "INJ-005", "document_id": "news-0009", "stage": "rewrite",
             "kind": "missing_template", "code": "E_CONFIG_INVALID"},
        ],
    }
    with (SAMPLES / "injections.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(injections, handle, allow_unicode=True, sort_keys=False, width=1000)

    invalid_docs = [
        {"document_id": "bad-1", "title": "", "paragraphs": ["段落"], "entities": [], "events": []},
        {"document_id": "bad-2", "title": "标题", "paragraphs": ["", "正文"], "entities": [], "events": []},
        {"document_id": "bad-3", "title": "标题", "paragraphs": ["正文"],
         "entities": [{"id": "", "text": "", "type": ""}],
         "events": [{"subject": "甲", "predicate": "", "object": "乙"}]},
        {"document_id": "dup", "title": "标题", "paragraphs": ["正文一"], "entities": [], "events": []},
        {"document_id": "dup", "title": "标题", "paragraphs": ["正文二"], "entities": [], "events": []},
    ]
    with (SAMPLES / "invalid_documents.jsonl").open("w", encoding="utf-8") as handle:
        for document in invalid_docs:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")

    invalid_config = copy.deepcopy(config)
    invalid_config["max_summary_sentences"] = 0
    invalid_config["fixed_decimal_places"] = 12
    invalid_config["thread_pool_size"] = 0
    invalid_config["rewrite"]["style_template_id"] = "not-declared"
    with (SAMPLES / "invalid_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(invalid_config, handle, allow_unicode=True, sort_keys=False, width=1000)

    invalid_commands = [
        {"type": "add_paragraph", "document_id": "missing-doc", "paragraph": "内容"},
        {"type": "unknown_operation", "value": 1},
        {"type": "remove_paragraph", "document_id": "news-0001", "index": 99},
    ]
    with (SAMPLES / "invalid_commands.jsonl").open("w", encoding="utf-8") as handle:
        handle.write("{not-json}\n")
        for command in invalid_commands:
            handle.write(json.dumps(command, ensure_ascii=False) + "\n")

    print(f"generated {len(documents)} documents, {len(commands)} commands in {SAMPLES}")


if __name__ == "__main__":
    main()
