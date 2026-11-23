"""确定性内容指纹与缓存键派生。

全部使用标准库 sha256，浮点先规整为规范 repr，保证跨进程、跨平台、
跨运行结果逐字段一致（不依赖 dict 顺序之外的哈希随机化）。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence

_FP_LEN = 16


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_fingerprint(obj: Any) -> str:
    """对任意可 JSON 化对象计算稳定短指纹。"""
    payload = _canonical(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_FP_LEN]


def fp_float(value: float) -> str:
    """单个浮点的稳定指纹（区分 -0.0 与 0.0 通过统一 repr 即可）。"""
    return hashlib.sha256(repr(float(value)).encode("utf-8")).hexdigest()[:_FP_LEN]


def fp_floats(values: Iterable[float]) -> str:
    joined = "|".join(repr(float(v)) for v in values)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:_FP_LEN]


def cache_key(stage: str, upstream: Sequence[str], **parts: Any) -> str:
    """由阶段名、上游指纹与命名参数派生缓存键。"""
    material: dict[str, Any] = {
        "stage": stage,
        "upstream": list(upstream),
        "parts": {k: parts[k] for k in sorted(parts)},
    }
    return f"{stage}:{content_fingerprint(material)}"


def version_id(source_version: int, content: Any) -> str:
    """转换产物/旁路元数据的版本标识：来源版本 + 内容指纹。"""
    return f"v{source_version}-src{source_version}-{content_fingerprint(content)}"
