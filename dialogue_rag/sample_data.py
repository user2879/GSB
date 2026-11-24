"""Deterministic local sample generation; numpy is used only for seeded data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEED = 23609


def build_sample(seed: int = SEED) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    # Consume seeded jitter so the seed genuinely participates in generation.
    jitter = [round(float(value), 3) for value in rng.uniform(-0.02, 0.02, size=8)]

    evidence_rows = [
        ("E001", "退款政策、适用订单与退款条件", 1.40),
        ("E002", "退款到账时效与支付渠道说明", 1.20),
        ("E003", "订单取消和退款申请流程", 1.10),
        ("E004", "收货地址与配送地址修改流程", 1.30),
        ("E005", "配送时效、仓配承诺与延误说明", 1.25),
        ("E006", "物流轨迹查询和派送节点", 1.15),
        ("E007", "账户资料、发票抬头与账单设置", 1.05),
        ("E008", "人工客服介入和异常工单处理", 0.95),
    ]
    evidence = [
        {"evidence_id": evidence_id, "text": text, "weight": weight + jitter[index] * 0.01}
        for index, (evidence_id, text, weight) in enumerate(evidence_rows)
    ]
    # Keep the principal weights exact enough for readable assertions.
    for row, (_, _, weight) in zip(evidence, evidence_rows):
        row["weight"] = weight

    channels = [
        {"channel_id": "ch_graph", "name": "knowledge_graph", "weight": 0.90},
        {"channel_id": "ch_lex", "name": "lexical", "weight": 1.00},
        {"channel_id": "ch_sem", "name": "semantic", "weight": 1.15},
    ]
    turns = [
        {"turn_id": "T1", "seq": 1, "query": "退款 政策 到账 时间 银行"},
        {"turn_id": "T2", "seq": 2, "query": "退款 进度 查询 到账 处理"},
        {"turn_id": "T3", "seq": 3, "query": "物流 地址 修改 配送 位置"},
        {"turn_id": "T4", "seq": 4, "query": "配送 时效 送达 物流 进度"},
        {"turn_id": "T5", "seq": 5, "query": "退款 到账 申请 订单 取消"},
        {"turn_id": "T6", "seq": 6, "query": "退款 到账 原路 返回 支付"},
    ]

    hit_table = {
        "T1": [
            ("ch_lex", "E001", 2.00),
            ("ch_sem", "E001", 1.85),
            ("ch_lex", "E002", 1.80),
            ("ch_graph", "E002", 2.00),
            ("ch_sem", "E003", 1.60),
            ("ch_lex", "E003", 1.70),
            ("ch_lex", "E003", 1.70),
        ],
        "T2": [
            ("ch_lex", "E002", 2.10),
            ("ch_sem", "E002", 1.70),
            ("ch_sem", "E003", 1.90),
            ("ch_graph", "E003", 2.20),
            ("ch_lex", "E008", 1.40),
            ("ch_graph", "E008", 1.70),
        ],
        "T3": [
            ("ch_lex", "E004", 2.20),
            ("ch_sem", "E004", 1.80),
            ("ch_graph", "E007", 2.40),
            ("ch_lex", "E007", 2.00),
            ("ch_sem", "E007", 1.70),
        ],
        "T4": [
            ("ch_lex", "E005", 2.30),
            ("ch_sem", "E005", 2.00),
            ("ch_graph", "E006", 2.50),
            ("ch_lex", "E006", 2.10),
            ("ch_sem", "E004", 1.60),
        ],
        "T5": [
            ("ch_sem", "E003", 2.20),
            ("ch_lex", "E003", 2.00),
            ("ch_lex", "E001", 1.90),
            ("ch_graph", "E001", 2.30),
            ("ch_sem", "E002", 1.80),
            ("ch_lex", "E002", 1.70),
        ],
        "T6": [
            ("ch_lex", "E002", 2.40),
            ("ch_sem", "E002", 2.00),
            ("ch_graph", "E001", 2.20),
            ("ch_lex", "E001", 1.80),
            ("ch_lex", "E008", 1.60),
            ("ch_sem", "E008", 1.50),
        ],
    }
    declarations = []
    for turn in turns:
        turn_id = turn["turn_id"]
        by_channel: dict[str, list[dict]] = {}
        for channel_id, evidence_id, raw_score in hit_table[turn_id]:
            by_channel.setdefault(channel_id, []).append(
                {"evidence_id": evidence_id, "raw_score": raw_score}
            )
        for channel_id, hits in sorted(by_channel.items()):
            declarations.append(
                {
                    "session_id": "S1",
                    "turn_id": turn_id,
                    "channel_id": channel_id,
                    "hits": hits,
                }
            )

    config = {
        "schema_version": "1.0",
        "seed": seed,
        "budget_points": 240,
        "workers": 2,
        "session_id": "S1",
        "turns": turns,
        "evidence": evidence,
        "channels": channels,
        "recall_declarations": declarations,
        "max_evidence_per_turn": 4,
        "topic": {"similarity_threshold": "0.25", "equal_rule": "merge"},
        "scoring": {
            "distance_decay": 0.82,
            "cross_topic_penalty": 0.55,
            "normalization_weight": 9.0,
            "answer_threshold": 0.60,
            "error_tolerance": 0.000001,
        },
        "fixed_point": {"fixed_width": "F16", "quant_gear": "G1", "rounding": "half_up"},
    }

    commands = {
        "commands": [
            {
                "op": "replace_query",
                "session_id": "S1",
                "turn_id": "T3",
                "query": "物流 地址 修改 配送 电话",
            },
            {
                "op": "set_topic_rule",
                "session_id": "S1",
                "topic": {"similarity_threshold": "0.25", "equal_rule": "split"},
            },
            {"op": "set_channel_weight", "channel_id": "ch_sem", "weight": 1.05},
            {"op": "set_evidence_weight", "evidence_id": "E002", "weight": 1.75},
            {
                "op": "add_turn",
                "session_id": "S1",
                "turn": {
                    "turn_id": "T7",
                    "query": "发票 抬头 账单 信息 修改",
                    "channel_hits": [
                        {"channel_id": "ch_lex", "evidence_id": "E007", "raw_score": 1.90},
                        {"channel_id": "ch_graph", "evidence_id": "E007", "raw_score": 2.00},
                        {"channel_id": "ch_sem", "evidence_id": "E008", "raw_score": 1.40},
                    ],
                },
            },
            {"op": "delete_turn", "session_id": "S1", "turn_id": "T4"},
            {"op": "set_fixed_point", "fixed_point": {"fixed_width": "F8", "quant_gear": "G2"}},
        ]
    }
    return config, commands


def write_samples(output_dir: str | Path = "samples", seed: int = SEED) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config, commands = build_sample(seed)
    config_path = output_dir / "config.json"
    commands_path = output_dir / "commands.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    commands_path.write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [config_path, commands_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic local RAG samples")
    parser.add_argument("--out", default="samples")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    paths = write_samples(args.out, args.seed)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
