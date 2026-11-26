from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ERROR_CATALOG = {
    "E_CONFIG_SCHEMA": "配置结构或字段类型不合法",
    "E_NODE_ID_DUPLICATE": "节点标识重复",
    "E_NODE_PORT_DUPLICATE": "节点回环端口重复",
    "E_NODE_PORT_INVALID": "节点端口不合法",
    "E_NODE_CAPACITY_INVALID": "节点容量槽位不是正整数",
    "E_NODE_LEASE_LIMIT_INVALID": "节点租约上限不是正整数",
    "E_SHARD_NODE_MISSING": "分片声明的节点不存在",
    "E_QUOTA_VERSION_NOT_MONOTONIC": "配额版本号未严格递增",
    "E_QUOTA_PARENT_MISSING": "配额版本显式引用的父版本不存在",
    "E_QUOTA_PARENT_NOT_EARLIER": "配额版本父版本号不早于当前版本",
    "E_QUOTA_CYCLE": "配额版本链存在环",
    "E_QUOTA_ACTIVE_MISSING": "当前配额版本不存在",
    "E_QUOTA_WATERMARK_INVALID": "配额水印不合法",
    "E_SHARD_ID_DUPLICATE": "分片标识重复",
    "E_SHARD_ID_MISSING": "分片标识缺失",
    "E_DEP_MISSING": "上游依赖指向不存在的分片",
    "E_DEP_CYCLE": "分片上游依赖存在环",
    "E_EVENT_SEQ_NOT_STRICT": "事件到达序号未严格递增",
    "E_EVENT_TYPE_UNKNOWN": "事件类型未知",
    "E_EVENT_REF_MISSING": "事件引用的对象不存在",
    "E_BATCH_ID_DUPLICATE": "分片内批次标识重复",
    "E_SAMPLE_DIMENSION": "样本维度与声明不一致",
    "E_NODE_CAPACITY_EXHAUSTED": "节点容量槽位已耗尽",
    "E_NODE_LEASE_LIMIT": "节点租约上限已耗尽",
    "E_LEASE_GENERATION_MISMATCH": "租约代次陈旧或不匹配",
    "E_LEASE_NOT_FOUND": "租约不存在或已释放",
    "E_LEASE_EXPIRED": "租约超过到期毫秒并由回收逻辑释放",
    "E_QUOTA_SOFT_LIMIT": "跨过配额软水印",
    "E_QUOTA_HARD_LIMIT": "跨过配额硬水印，拒绝新派发",
    "E_DEPENDENCY_BLOCKED": "上游尚未产出权威聚合摘要",
    "E_GLOBAL_BUDGET_EXHAUSTED": "全局资源单位预算不足",
    "E_BATCH_POINTS_EXCEEDED": "单批分片处理点数超过上限",
    "E_PROTOCOL": "本地回环协议请求或应答不合法",
    "E_NODE_UNAVAILABLE": "节点子进程不可用",
    "E_SNAPSHOT_CORRUPT": "提交快照不完整或配置哈希不匹配",
    "E_INTERNAL": "未分类内部错误",
}


@dataclass(frozen=True)
class BizError:
    code: str
    object_type: str
    object_id: str
    field: str
    message: str
    injection_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "code": self.code,
            "object_type": self.object_type,
            "object_id": self.object_id,
            "field": self.field,
            "message": self.message,
            "catalog": ERROR_CATALOG.get(self.code, "未知业务码"),
        }
        if self.injection_id is not None:
            result["injection_id"] = self.injection_id
        if self.details:
            result["details"] = self.details
        return result


class BizErrorException(Exception):
    def __init__(self, error: BizError):
        super().__init__(f"{error.code} {error.object_type}:{error.object_id}.{error.field}")
        self.error = error


def make_error(
    code: str,
    object_type: str,
    object_id: str,
    field: str,
    message: str,
    *,
    injection_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> BizError:
    return BizError(
        code=code,
        object_type=object_type,
        object_id=object_id,
        field=field,
        message=message,
        injection_id=injection_id,
        details=details or {},
    )
