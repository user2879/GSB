from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .models import AppConfig, Session
from .util import sha256_json, write_json

DEFAULT_SEED = 20261126

EVIDENCE_TEXTS = {
    "e01": "退款申请提交成功后，平台会在一个工作日内完成审核。",
    "e02": "退款到账时间取决于支付渠道，通常为一至三个工作日。",
    "e03": "原路退回是默认退款方式，银行卡到账可能延迟。",
    "e04": "退款进度可在订单详情页的售后记录中查询。",
    "e05": "节假日期间退款处理与银行清算时间可能顺延。",
    "e06": "客服工单可以催办超时未到账的退款。",
    "e07": "发票抬头填写错误时可以申请作废后重开。",
    "e08": "发票重开需要提供订单号、原发票与正确抬头。",
    "e09": "电子发票可在发票中心下载 PDF 文件。",
    "e10": "账号登录异常时先检查密码与安全验证方式。",
    "e11": "验证失败可通过绑定手机号找回账号。",
    "e12": "连续多次验证失败会临时冻结登录。",
    "e13": "查看退款政策更新公告可以确认最新时限。",
    "e14": "账号申诉需要提交身份验证材料。",
}

TURN_BLUEPRINTS = [
    {
        "query": "退款申请后多久到账？",
        "channels": {
            "keyword": ["e01", "e02", "e02"],
            "dense": ["e02", "e03"],
            "graph": ["e04"],
        },
    },
    {
        "query": "退款到账时间受哪些因素影响？",
        "channels": {
            "dense": ["e02", "e03", "e05"],
            "keyword": ["e01", "e02"],
        },
    },
    {
        "query": "退款进度在哪里查询？",
        "channels": {
            "keyword": ["e01", "e04"],
            "dense": ["e03", "e04"],
        },
    },
    {
        "query": "发票抬头填错了可以重开吗？",
        "channels": {
            "keyword": ["e07", "e08"],
            "dense": ["e08", "e09"],
            "graph": ["e02"],
        },
    },
    {
        "query": "发票重开需要什么材料？",
        "channels": {
            "dense": ["e08", "e09"],
            "keyword": ["e07", "e08"],
            "graph": ["e02"],
        },
    },
    {
        "query": "电子发票如何下载？",
        "channels": {
            "keyword": ["e07", "e09"],
            "dense": ["e09", "e08"],
        },
    },
    {
        "query": "账号登录异常怎么处理？",
        "channels": {
            "graph": ["e10", "e02"],
            "keyword": ["e10", "e11"],
            "dense": ["e11", "e12"],
        },
    },
    {
        "query": "登录验证失败如何找回账号？",
        "channels": {
            "dense": ["e11", "e02"],
            "keyword": ["e10", "e12"],
            "graph": ["e08"],
        },
    },
]


def _weight_text(rng: random.Random, low: int, high: int) -> str:
    return f"{rng.randint(low, high) / 100:.2f}"


def build_sample(seed: int = DEFAULT_SEED) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(seed)
    weights = {
        evidence_id: _weight_text(rng, 45, 118)
        for evidence_id in sorted(EVIDENCE_TEXTS)
    }
    keyword_weight = _weight_text(rng, 160, 185)
    dense_weight = _weight_text(rng, 125, 150)
    graph_weight = _weight_text(rng, 78, 102)
    config = {
        "schema_version": 1,
        "channel_weights": {
            "dense": dense_weight,
            "graph": graph_weight,
            "keyword": keyword_weight,
        },
        "fixed_point": {
            "fractional_bits": 12,
            "quantization": 4,
            "rounding": "half_up",
        },
        "topic": {"switch_threshold_bp": 1200},
        "confidence": {
            "turn_decay": "0.90",
            "cross_topic_penalty": "0.55",
            "answer_threshold": "0.57",
        },
        "limits": {"max_workers": 4, "evidence_budget": 200},
        "diagnostics": {"tolerance": "0.0015"},
        "reference": {
            "fractional_bits": 24,
            "quantization": 1,
            "rounding": "half_up",
        },
    }
    evidence = [
        {
            "evidence_id": evidence_id,
            "text": text,
            "weight": weights[evidence_id],
        }
        for evidence_id, text in sorted(EVIDENCE_TEXTS.items())
    ]
    turns = [
        {
            "turn_id": index,
            "query": blueprint["query"],
            "channels": {
                channel: list(ids)
                for channel, ids in sorted(blueprint["channels"].items())
            },
        }
        for index, blueprint in enumerate(TURN_BLUEPRINTS, start=1)
    ]
    session = {
        "session_id": f"refund-invoice-security-{seed}",
        "turns": turns,
        "evidence": evidence,
    }
    commands = [
        {
            "seq": 1,
            "op": "add_turn",
            "turn": {
                "query": "退款政策更新后到账时间会变化吗？",
                "channels": {
                    "dense": ["e02", "e05", "e13"],
                    "keyword": ["e01", "e13"],
                    "graph": ["e04"],
                },
            },
        },
        {
            "seq": 2,
            "op": "replace_turn_query",
            "turn_id": 2,
            "query": "退款到账受支付渠道和节假日影响吗？",
        },
        {
            "seq": 3,
            "op": "set_channel_weight",
            "channel": "dense",
            "weight": "1.58",
        },
        {
            "seq": 4,
            "op": "set_evidence_weight",
            "evidence_id": "e02",
            "weight": "0.41",
        },
        {
            "seq": 5,
            "op": "set_topic_threshold",
            "value_bp": 2500,
        },
        {
            "seq": 6,
            "op": "set_channel_declaration",
            "turn_id": 3,
            "channel": "dense",
            "evidence_ids": ["e03", "e04", "e12"],
        },
        {"seq": 7, "op": "delete_turn", "turn_id": 6},
        {
            "seq": 8,
            "op": "set_budget",
            "value": 5,
        },
        {
            "seq": 9,
            "op": "set_fixed_point",
            "fractional_bits": 8,
            "quantization": 8,
            "rounding": "half_up",
        },
        {"seq": 10, "op": "set_budget", "value": 200},
        {
            "seq": 11,
            "op": "set_fixed_point",
            "fractional_bits": 16,
            "quantization": 2,
            "rounding": "half_even",
        },
        {
            "seq": 12,
            "op": "set_evidence_text",
            "evidence_id": "e02",
            "text": "退款到账时间取决于支付渠道和银行清算，通常为一至三个工作日。",
        },
        {
            "seq": 13,
            "op": "set_channel_weight",
            "channel": "dense",
            "weight": "-1.00",
        },
        {
            "seq": 14,
            "op": "set_topic_threshold",
            "value_bp": 1200,
        },
        {
            "seq": 15,
            "op": "set_channel_declaration",
            "turn_id": 2,
            "channel": "keyword",
            "evidence_ids": ["e01", "e02", "e05"],
        },
        {
            "seq": 16,
            "op": "set_fixed_point",
            "fractional_bits": 8,
            "quantization": 4,
            "rounding": "half_up",
        },
        {
            "seq": 17,
            "op": "set_channel_weight",
            "channel": "keyword",
            "weight": "1.583",
        },
    ]
    normalized_config = AppConfig.from_dict(config).to_dict()
    normalized_session = Session.from_dict(session).to_dict()
    return normalized_config, normalized_session, commands


def generate_sample(root: str | Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    root_path = Path(root)
    config_dir = root_path / "config"
    sample_dir = root_path / "sample"
    snapshots_dir = root_path / "snapshots"
    reports_dir = root_path / "reports"
    for directory in (config_dir, sample_dir, snapshots_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)
    config, session, commands = build_sample(seed)
    write_json(config_dir / "demo.json", config)
    write_json(sample_dir / "session.json", session)
    with open(sample_dir / "commands.jsonl", "w", encoding="utf-8") as handle:
        for command in commands:
            handle.write(json.dumps(command, ensure_ascii=False, sort_keys=True) + "\n")
    generation_manifest = {
        "seed": seed,
        "config_file": "config/demo.json",
        "session_file": "sample/session.json",
        "commands_file": "sample/commands.jsonl",
        "config_fingerprint": sha256_json(config),
        "session_fingerprint": sha256_json(session),
        "command_stream_fingerprint": sha256_json(commands),
        "command_count": len(commands),
    }
    write_json(sample_dir / "generation.json", generation_manifest)
    return generation_manifest
