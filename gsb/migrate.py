"""版本迁移表与幂等迁移。

迁移表每一项可审计：记录被补齐字段名、来源版本与默认值。
对同一事件重复迁移，内容哈希保持不变（幂等）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import E_SCHEMA_TOO_NEW, PipelineError
from .model import (
    CURRENT_SCHEMA_VERSION,
    MIN_SCHEMA_VERSION,
    PAYLOAD_ENCODING_CURRENT,
    TIME_BASE_CURRENT,
    Event,
)


@dataclass(frozen=True)
class MigrationRule:
    from_version: int
    to_version: int
    field: str
    default: object
    reason: str


# 固定迁移表：v1 -> v2 -> v3
MIGRATION_TABLE = [
    MigrationRule(1, 2, "local_seq", 0, "v1 无本地序号语义，按提交顺序由解析器赋予"),
    MigrationRule(1, 2, "payload_encoding", PAYLOAD_ENCODING_CURRENT, "v1 未声明 payload 编码版本"),
    MigrationRule(2, 3, "time_base", TIME_BASE_CURRENT, "v2 未声明时间基准，规范化为 tick"),
]


def audit_table() -> list[dict]:
    return [
        {
            "from_version": r.from_version,
            "to_version": r.to_version,
            "field": r.field,
            "default": r.default,
            "reason": r.reason,
        }
        for r in MIGRATION_TABLE
    ]


def migrate_event(event: Event) -> tuple[Event, list[dict]]:
    """把事件迁移到当前 schema 版本，返回 (事件, 审计记录)。

    - schema_version 高于当前支持上限：E_SCHEMA_TOO_NEW，不猜测字段语义。
    - 幂等：已迁移事件再次迁移时审计记录为空，内容哈希不变。
    """
    if event.schema_version > CURRENT_SCHEMA_VERSION:
        raise PipelineError(
            E_SCHEMA_TOO_NEW,
            f"schema_version {event.schema_version} 高于当前支持上限 {CURRENT_SCHEMA_VERSION}",
            field="schema_version",
        )
    if event.schema_version < MIN_SCHEMA_VERSION:
        raise PipelineError(
            "E_SCHEMA_TOO_OLD",
            f"schema_version {event.schema_version} 低于最低可迁移版本 {MIN_SCHEMA_VERSION}",
            field="schema_version",
        )

    audit: list[dict] = []
    declared_version = event.schema_version
    version = event.schema_version
    while version < CURRENT_SCHEMA_VERSION:
        for rule in MIGRATION_TABLE:
            if rule.from_version != version:
                continue
            # 仅当字段缺失（None）时补齐；已有值不覆盖，保证幂等。
            if getattr(event, rule.field, None) is None:
                setattr(event, rule.field, rule.default)
                audit.append(
                    {
                        "field": rule.field,
                        "from_version": rule.from_version,
                        "to_version": rule.to_version,
                        "default": rule.default,
                    }
                )
        version += 1
    # 位图未声明/承载未携带的可选字段：即使版本标记已是当前版本，也按迁移表默认值补齐
    latest_default = {}
    for rule in MIGRATION_TABLE:
        latest_default[rule.field] = rule.default
    for field, default in latest_default.items():
        if getattr(event, field, None) is None:
            setattr(event, field, default)
            audit.append({
                "field": field,
                "from_version": declared_version,
                "to_version": CURRENT_SCHEMA_VERSION,
                "default": default,
            })
    event.schema_version = CURRENT_SCHEMA_VERSION
    return event, audit
