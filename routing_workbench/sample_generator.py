"""Deterministic local Chinese ticket sample generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .util import atomic_write_json, atomic_write_text


VOCABULARY = [
    "账号", "账户", "登录", "密码", "忘记", "重置", "修改", "手机号", "邮箱", "验证码",
    "收不到", "收到", "发送", "冻结", "解冻", "原因", "异地", "锁定", "退款", "进度", "到账",
    "订单号", "订单", "发票", "抬头", "发票号", "扣费", "重复扣费", "两次", "支付", "金额",
    "不是", "没有", "并未", "已经", "查询", "请", "帮我", "处理", "第一次", "失败",
    "无法", "需要", "确认", "充值", "银行卡", "尾号", "到账时间", "开票", "公司", "个人",
    "13800138000", "13900139000", "A1001", "A1002", "A1003", "A1004", "F2002", "F2003",
    "验证码是", "请查收", "工单", "客服", "本人",
]


def _u(
    utterance_id: str,
    session_id: str,
    turn: int,
    tokens: list[str],
    domain: str,
    intent: str,
    slots: dict[str, str],
) -> dict[str, Any]:
    return {
        "utterance_id": utterance_id,
        "session_id": session_id,
        "turn": turn,
        "tokens": tokens,
        "domain": domain,
        "intent": intent,
        "slots": slots,
    }


def sample_utterances() -> list[dict[str, Any]]:
    return [
        _u("S001-U1", "S001", 1, ["忘记", "密码", "手机号", "13800138000", "登录"], "account", "password_reset", {"phone": "13800138000"}),
        _u("S001-U2", "S001", 2, ["账号", "冻结", "原因", "异地", "登录", "解冻"], "account", "account_unfreeze", {"reason": "异地"}),
        _u("S001-U3", "S001", 3, ["验证码", "收不到", "手机号", "13900139000", "发送"], "account", "verification_code", {"phone": "13900139000"}),
        _u("S002-U1", "S002", 1, ["退款", "进度", "订单号", "A1001", "查询"], "billing", "refund_status", {"order_id": "A1001"}),
        _u("S002-U2", "S002", 2, ["发票", "抬头", "修改", "发票号", "F2002"], "billing", "invoice_change", {"invoice_id": "F2002"}),
        _u("S002-U3", "S002", 3, ["重复扣费", "两次", "订单号", "A1003"], "billing", "duplicate_charge", {"order_id": "A1003"}),
        _u("S003-U1", "S003", 1, ["重置", "密码", "手机号", "13800138000", "无法", "登录"], "account", "password_reset", {"phone": "13800138000"}),
        _u("S003-U2", "S003", 2, ["没有", "收到", "验证码", "手机号", "13900139000"], "account", "verification_code", {"phone": "13900139000"}),
        _u("S003-U3", "S003", 3, ["退款", "密码", "订单号", "A1004", "到账时间"], "billing", "refund_status", {"order_id": "A1004"}),
        _u("S004-U1", "S004", 1, ["发票", "公司", "抬头", "修改", "发票号", "F2003"], "billing", "invoice_change", {"invoice_id": "F2003"}),
        _u("S004-U2", "S004", 2, ["账号", "解冻", "原因", "确认", "本人", "锁定"], "account", "account_unfreeze", {"reason": "确认"}),
        _u("S004-U3", "S004", 3, ["扣费", "两次", "订单号", "A1002", "金额"], "billing", "duplicate_charge", {"order_id": "A1002"}),
        _u("S005-U1", "S005", 1, ["发票", "密码", "手机号", "13800138000", "修改"], "account", "password_reset", {"phone": "13800138000"}),
    ]


def sample_config() -> dict[str, Any]:
    return {
        "version": 1,
        "vocabulary": VOCABULARY,
        "domains": {
            "account": {
                "triggers": [
                    {"term": "账号", "weight": 3},
                    {"term": "登录", "weight": 2},
                    {"term": "密码", "weight": 4},
                    {"term": "验证码", "weight": 2},
                    {"term": "冻结", "weight": 3},
                    {"term": "解冻", "weight": 2},
                    {"term": "手机号", "weight": 1},
                    {"term": "发送", "weight": 1},
                    {"term": "收到", "weight": 1},
                ],
                "intents": {
                    "password_reset": {
                        "triggers": [
                            {"term": "忘记密码", "weight": 5},
                            {"term": "重置密码", "weight": 5},
                            {"term": "修改密码", "weight": 3},
                            {"term": "密码", "weight": 1},
                        ],
                        "slot_patterns": ["phone_after_label"],
                        "required_slots": ["phone"],
                    },
                    "account_unfreeze": {
                        "triggers": [
                            {"term": "账号冻结", "weight": 5},
                            {"term": "解冻", "weight": 5},
                            {"term": "冻结", "weight": 2},
                        ],
                        "slot_patterns": ["reason_after_label"],
                        "required_slots": ["reason"],
                    },
                    "verification_code": {
                        "triggers": [
                            {"term": "验证码", "weight": 5},
                            {"term": "收不到", "weight": 3},
                            {"term": "发送", "weight": 2},
                            {"term": "手机号", "weight": 1},
                        ],
                        "slot_patterns": ["phone_after_label"],
                        "required_slots": ["phone"],
                    },
                },
            },
            "billing": {
                "triggers": [
                    {"term": "账单", "weight": 3},
                    {"term": "扣费", "weight": 4},
                    {"term": "退款", "weight": 2},
                    {"term": "发票", "weight": 3},
                    {"term": "支付", "weight": 2},
                    {"term": "订单", "weight": 2},
                ],
                "intents": {
                    "refund_status": {
                        "triggers": [
                            {"term": "退款", "weight": 5},
                            {"term": "进度", "weight": 2},
                            {"term": "到账", "weight": 3},
                        ],
                        "slot_patterns": ["order_after_label"],
                        "required_slots": ["order_id"],
                    },
                    "invoice_change": {
                        "triggers": [
                            {"term": "发票", "weight": 5},
                            {"term": "抬头", "weight": 3},
                            {"term": "修改", "weight": 2},
                            {"term": "发票号", "weight": 4},
                        ],
                        "slot_patterns": ["invoice_after_label"],
                        "required_slots": ["invoice_id"],
                    },
                    "duplicate_charge": {
                        "triggers": [
                            {"term": "重复扣费", "weight": 5},
                            {"term": "扣费", "weight": 4},
                            {"term": "两次", "weight": 2},
                        ],
                        "slot_patterns": ["order_after_label"],
                        "required_slots": ["order_id"],
                    },
                },
            },
        },
        "slot_patterns": {
            "phone_after_label": {
                "name": "phone_after_label",
                "intent": "password_reset",
                "slot": "phone",
                "tokens": ["手机号", "{phone}"],
                "window": 2,
                "direction": "after",
            },
            "reason_after_label": {
                "name": "reason_after_label",
                "intent": "account_unfreeze",
                "slot": "reason",
                "tokens": ["原因", "{reason}"],
                "window": 2,
                "direction": "after",
            },
            "order_after_label": {
                "name": "order_after_label",
                "intent": "refund_status",
                "slot": "order_id",
                "tokens": ["订单号", "{order_id}"],
                "window": 2,
                "direction": "after",
            },
            "invoice_after_label": {
                "name": "invoice_after_label",
                "intent": "invoice_change",
                "slot": "invoice_id",
                "tokens": ["发票号", "{invoice_id}"],
                "window": 2,
                "direction": "after",
            },
        },
        "slots": {
            "phone": {"type": "phone"},
            "reason": {"type": "text"},
            "order_id": {"type": "order_id"},
            "invoice_id": {"type": "invoice_id"},
        },
        "required_slots": {
            "password_reset": ["phone"],
            "account_unfreeze": ["reason"],
            "verification_code": ["phone"],
            "refund_status": ["order_id"],
            "invoice_change": ["invoice_id"],
            "duplicate_charge": ["order_id"],
        },
        "negation_rules": [
            {"negators": ["不是", "没有", "并未"], "scope": 2, "mode": "cancel", "direction": "before"},
        ],
        "thresholds": {"domain": "0.20", "intent": "0.20"},
        "fixed_decimals": 5,
        "quantization_levels": 100000,
        "thread_pool_size": 4,
        "max_thread_pool_size": 8,
        "batch_point_budget": 4096,
        "snapshot_interval_commands": 2,
        "recalibration": {
            "completeness_weight": "0.12",
            "suppression_penalty": "0.03",
            "minimum": "0.0",
            "maximum": "1.0",
            "suppression_cap": 4,
        },
    }


def sample_commands() -> list[dict[str, Any]]:
    return [
        {"id": "C001", "type": "update_trigger", "scope": "domain", "domain": "account", "old_term": "密码", "weight": 6},
        {"id": "C002", "type": "add_trigger", "scope": "intent", "domain": "account", "intent": "password_reset", "trigger": "锁定", "weight": 2},
        {
            "id": "C003",
            "type": "replace_utterance",
            "utterance": _u("S001-U3", "S001", 3, ["验证码", "收不到", "手机号", "13900139000", "发送", "锁定"], "account", "verification_code", {"phone": "13900139000"}),
        },
        {"id": "C004", "type": "switch_slot_pattern", "pattern_name": "phone_after_label", "pattern": {"window": 4}},
        {"id": "C005", "type": "set_threshold", "domain": "0.18", "intent": "1/16"},
        {"id": "C006", "type": "set_recalibration", "completeness_weight": "0.18", "suppression_penalty": "0.04"},
        {
            "id": "C007",
            "type": "replace_utterance",
            "utterance": _u("S002-U2", "S002", 2, ["发票", "抬头", "修改", "发票号", "F2002", "公司"], "billing", "invoice_change", {"invoice_id": "F2002"}),
        },
        {"id": "C008", "type": "add_trigger", "scope": "intent", "domain": "billing", "intent": "refund_status", "trigger": "金额", "weight": 2},
        {"id": "C009", "type": "delete_trigger", "scope": "domain", "domain": "account", "term": "登录"},
        {
            "id": "C010",
            "type": "replace_utterance",
            "utterance": _u("S003-U1", "S003", 1, ["重置", "密码", "手机号", "13800138000", "无法", "登录", "请", "处理"], "account", "password_reset", {"phone": "13800138000"}),
        },
        {"id": "C011", "type": "switch_slot_pattern", "pattern_name": "order_after_label", "pattern": {"window": 3}},
        {"id": "C012", "type": "set_recalibration", "maximum": "0.98"},
    ]


def sample_injections() -> dict[str, Any]:
    return {
        "injections": [
            {
                "id": "INJ-001",
                "fault": "nonfinite_trigger_weight",
                "target": {"domain": "account", "term": "密码"},
                "expected_code": "E_ROUTE_INVALID",
            },
            {
                "id": "INJ-002",
                "fault": "undeclared_slot_pattern",
                "target": {"intent": "password_reset", "slot": "ghost_slot"},
                "expected_code": "E_ROUTE_INVALID",
            },
            {
                "id": "INJ-003",
                "fault": "turn_gap",
                "target": {"utterance_id": "S001-U2", "turn": 9},
                "expected_code": "E_ROUTE_INVALID",
            },
            {
                "id": "INJ-004",
                "fault": "out_of_vocab",
                "target": {"utterance_id": "S001-U3", "token": "未登记词项"},
                "expected_code": "E_ROUTE_INVALID",
            },
            {
                "id": "INJ-005",
                "fault": "runtime_exception",
                "target": {"utterance_id": "S002-U2", "stage": "intent"},
                "expected_code": "E_ROUTE_INJECTED_RUNTIME",
            },
        ]
    }


def generate_samples(target: str | Path) -> dict[str, str]:
    root = Path(target)
    root.mkdir(parents=True, exist_ok=True)
    utterances = sample_utterances()
    config = sample_config()
    commands = {"version": 1, "commands": sample_commands()}
    injections = sample_injections()
    atomic_write_text(
        root / "utterances.jsonl",
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in utterances),
    )
    atomic_write_text(root / "routing.yaml", yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    atomic_write_json(root / "commands.json", commands)
    atomic_write_text(root / "injections.yaml", yaml.safe_dump(injections, allow_unicode=True, sort_keys=False))
    atomic_write_text(root / "vocabulary.txt", "\n".join(VOCABULARY) + "\n")
    return {
        "utterances": str(root / "utterances.jsonl"),
        "config": str(root / "routing.yaml"),
        "commands": str(root / "commands.json"),
        "injections": str(root / "injections.yaml"),
        "vocabulary": str(root / "vocabulary.txt"),
    }
