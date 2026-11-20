"""使用 jsonschema 对输入做结构化校验（Draft 2020-12）。

schema 仅负责类型/形状约束，业务约束（权重和为正、ID 唯一、枚举引用等）
仍由 loader 中的确定性规则处理，错误统一进入失败清单。
"""
from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

SEED_SCHEMA = {
    "type": "object",
    "required": [
        "seed_id", "task_type", "template_id", "template",
        "slots", "answer_template",
    ],
    "properties": {
        "seed_id": {"type": "string", "minLength": 1},
        "task_type": {"type": "string", "minLength": 1},
        "template_id": {"type": "string", "minLength": 1},
        "template": {"type": "string", "minLength": 1},
        "answer_template": {"type": "string", "minLength": 1},
        "slots": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": True,
}

SLOT_SOURCE_SCHEMA = {
    "type": "object",
    "required": ["source_id", "task_type", "slot_name", "values"],
    "properties": {
        "source_id": {"type": "string", "minLength": 1},
        "task_type": {"type": "string", "minLength": 1},
        "slot_name": {"type": "string", "minLength": 1},
        "values": {
            "type": "array", "minItems": 1,
            "items": {"type": "string"},
        },
    },
    "additionalProperties": True,
}

CONFIG_SCHEMA = {
    "type": "object",
    "required": [
        "task_types", "target_counts", "template_weights",
        "slot_weights", "dedup_threshold", "max_workers",
        "snapshot_interval_batches",
    ],
    "properties": {
        "task_types": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "target_counts": {"type": "object"},
        "template_weights": {"type": "object"},
        "slot_weights": {"type": "object"},
        "dedup_threshold": {"type": "number"},
        "boundary_rule": {"type": "string", "enum": ["merge", "split"]},
        "max_workers": {"type": "integer"},
        "snapshot_interval_batches": {"type": "integer"},
    },
}

SCORE_PARAMS_SCHEMA = {
    "type": "object",
    "required": ["dimensions", "specs", "length_lower", "length_upper"],
    "properties": {
        "dimensions": {"type": "array", "items": {"type": "string"}},
        "specs": {"type": "object"},
        "length_lower": {"type": "integer"},
        "length_upper": {"type": "integer"},
    },
}

VOCAB_SCHEMA = {
    "type": "object",
    "required": ["tokens"],
    "properties": {
        "tokens": {"type": "array", "minItems": 1,
                   "items": {"type": "string", "minLength": 1}},
        "rules": {"type": "object"},
    },
}


def schema_errors(schema: dict[str, Any], instance: Any) -> list[str]:
    validator = Draft202012Validator(schema)
    return [
        f"{list(error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(instance), key=lambda e: e.message)
    ]
