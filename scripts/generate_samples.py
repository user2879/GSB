#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


SEED = 20261122

ORGANIZATIONS = [
    "市交通委",
    "市应急局",
    "市气象台",
    "东城公交集团",
    "水务集团",
    "蓝天救援队",
    "市中心医院",
]
LOCATIONS = ["东站", "沿江路", "中心城区", "北郊枢纽", "城南客运站"]
TOPICS = ["暴雨", "公交线路调整", "积水", "应急响应", "客流组织", "安全隐患"]
PASSENGER_TERMS = ["乘客", "市民", "通勤人员"]


CONFIG_YAML = """max_summary_sentences: 3
word_frequency_weight: 0.40
entity_coverage_weight: 0.38
position_decay_coefficient: 0.22
entity_coverage_min: 0.80
style_template_id: formal_v1
synonym_table_version: v1
synonym_table_path: synonym_table.yaml
fixed_decimals: 6
quantization_levels: 5
max_workers: 4
"""


SYNONYM_TABLE_YAML = """version: v1
templates:
  formal_v1:
    display_name: 正式通报
  concise_v1:
    display_name: 简明快讯
rules:
  - rule_id: R001
    template_id: formal_v1
    from: 召开
    to: 举行
  - rule_id: R002
    template_id: formal_v1
    from: 表示
    to: 指出
  - rule_id: R003
    template_id: formal_v1
    from: 已经
    to: 已
  - rule_id: R004
    template_id: formal_v1
    from: 排查
    to: 检查
  - rule_id: R005
    template_id: formal_v1
    from: 排查风险
    to: 开展风险检查
  - rule_id: R006
    template_id: formal_v1
    from: 公交线路
    to: 公共交通线路
  - rule_id: R007
    template_id: formal_v1
    from: 公交
    to: 公共交通
  - rule_id: R008
    template_id: formal_v1
    from: 恢复
    to: 重新恢复
  - rule_id: R009
    template_id: formal_v1
    from: 启动
    to: 启动专项
  - rule_id: R010
    template_id: formal_v1
    from: 交通委
    to: 交通运输委员会
  - rule_id: R011
    template_id: formal_v1
    from: 市交通委
    to: 市交通运输委员会
  - rule_id: R012
    template_id: concise_v1
    from: 表示
    to: 称
"""


def _alias(term: str) -> str:
    if term.startswith("市") and len(term) > 2:
        return term[1:]
    if term.endswith("集团") and len(term) > 2:
        return term[:-2]
    return term


def _entity(entity_id: str, name: str, entity_type: str, alias: str | None = None) -> dict[str, Any]:
    aliases = [alias] if alias and alias != name else []
    return {
        "entity_id": entity_id,
        "name": name,
        "aliases": aliases,
        "type": entity_type,
    }


def _variant_sentences(
    variant: int,
    org1: str,
    org2: str,
    location: str,
    topic: str,
    passenger: str,
) -> tuple[str, str]:
    if variant == 0:
        paragraph1 = f"{org1}与{org2}在{location}召开协调会。{org1}表示已经启动{topic}应急响应。"
        paragraph2 = f"{org2}组织人员排查风险，公交线路恢复正常通行。{passenger}可以关注后续通知。"
    elif variant == 1:
        paragraph1 = f"{org1}与{org2}召开专题会议。会议在{location}举行，重点讨论{topic}。"
        paragraph2 = f"{org2}组织人员排查风险，公交线路恢复正常通行。{passenger}可以关注后续通知。"
    elif variant == 2:
        paragraph1 = f"{org1}介绍{topic}处置情况。{org2}在{location}召开协调会。"
        paragraph2 = f"{org2}表示已经启动应急响应。{passenger}可以关注后续通知。"
    else:
        paragraph1 = f"{org1}在{location}检查防汛准备。{org2}介绍{topic}处置进展。"
        paragraph2 = f"{org2}组织人员排查风险，公交线路恢复正常通行。{passenger}可以关注后续通知。"
    return paragraph1, paragraph2


def generate_news(seed: int = SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    organizations = list(ORGANIZATIONS)
    documents: list[dict[str, Any]] = []
    for number in range(1, 11):
        org1, org2 = rng.sample(organizations, 2)
        location = rng.choice(LOCATIONS)
        topic = rng.choice(TOPICS)
        passenger = rng.choice(PASSENGER_TERMS)
        variant = (number - 1) % 4
        paragraph1, paragraph2 = _variant_sentences(
            variant, org1, org2, location, topic, passenger
        )
        entities = [
            _entity("E1", org1, "ORG", _alias(org1)),
            _entity("E2", org2, "ORG", _alias(org2)),
            _entity("E3", location, "LOC", None),
            _entity("E4", topic, "EVENT", None),
            _entity("E5", passenger, "GROUP", "市民" if passenger != "市民" else None),
        ]
        events = [
            {
                "event_id": "V1",
                "subject": "E1",
                "predicate": "召开",
                "object": "E2",
            },
            {
                "event_id": "V2",
                "subject": "E1",
                "predicate": "启动",
                "object": "E4",
            },
            {
                "event_id": "V3",
                "subject": "E2",
                "predicate": "排查",
                "object": "E3",
            },
        ]
        documents.append(
            {
                "doc_id": f"DOC-{number:03d}",
                "title": f"{org1}{location}{topic}协调处置情况通报",
                "paragraphs": [paragraph1, paragraph2],
                "entities": entities,
                "events": events,
            }
        )
    return documents


def injections() -> dict[str, Any]:
    return {
        "injections": [
            {
                "injection_id": "INJ-FREQ-001",
                "type": "frequency_delta",
                "doc_id": "DOC-002",
                "sentence_index": 2,
                "delta": 6.0,
            },
            {
                "injection_id": "INJ-ENT-001",
                "type": "entity_coverage_delta",
                "doc_id": "DOC-003",
                "sentence_index": 1,
                "delta": 0.40,
            },
            {
                "injection_id": "INJ-FREQ-002",
                "type": "frequency_delta",
                "doc_id": "DOC-002",
                "sentence_index": 3,
                "delta": 30.0,
            },
            {
                "injection_id": "INJ-ENT-002",
                "type": "entity_coverage_delta",
                "doc_id": "DOC-001",
                "sentence_index": 1,
                "delta": 0.20,
            },
            {
                "injection_id": "INJ-ENT-003",
                "type": "entity_coverage_delta",
                "doc_id": "DOC-002",
                "sentence_index": 3,
                "delta": 0.40,
            },
            {
                "injection_id": "INJ-ENT-004",
                "type": "entity_coverage_delta",
                "doc_id": "DOC-004",
                "sentence_index": 2,
                "delta": 0.80,
            },
            {
                "injection_id": "INJ-ILLEGAL-001",
                "type": "illegal_entity",
                "doc_id": "DOC-007",
                "paragraph_index": 1,
                "entity": {
                    "entity_id": "E-BAD",
                    "name": "",
                    "aliases": [],
                    "type": "ORG",
                },
            },
            {
                "injection_id": "INJ-TEMPLATE-001",
                "type": "missing_template_ref",
                "rule_id": "R002",
                "missing_template_id": "ghost_v1",
            },
            {
                "injection_id": "INJ-MISS-001",
                "type": "frequency_delta",
                "doc_id": "DOC-999",
                "sentence_index": 0,
                "delta": 1.0,
            },
            {
                "injection_id": "INJ-MISS-002",
                "type": "entity_coverage_delta",
                "doc_id": "DOC-004",
                "sentence_index": 99,
                "delta": 0.10,
            },
        ]
    }


def write_samples(output_dir: Path, seed: int = SEED) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = generate_news(seed)
    with (output_dir / "news.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    (output_dir / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8", newline="\n")
    (output_dir / "synonym_table.yaml").write_text(
        SYNONYM_TABLE_YAML, encoding="utf-8", newline="\n"
    )
    (output_dir / "injections.json").write_text(
        json.dumps(injections(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic sample inputs")
    parser.add_argument("--output", type=Path, default=Path("samples"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    write_samples(args.output, args.seed)
    print(f"generated samples in {args.output} with seed {args.seed}")


if __name__ == "__main__":
    main()
