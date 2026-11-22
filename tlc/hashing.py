"""稳定哈希与确定性随机源。

所有随机源均由内容哈希派生，不依赖全局 RNG，
保证并发与任意调度顺序下结果逐位一致。
"""

import hashlib
import json


def canonical(obj) -> str:
    """对象的规范 JSON 序列化（键排序、紧凑分隔符）。"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(obj) -> str:
    if not isinstance(obj, (str, bytes)):
        obj = canonical(obj)
    if isinstance(obj, str):
        obj = obj.encode("utf-8")
    return hashlib.sha256(obj).hexdigest()


def stable_seed(*parts) -> int:
    """由任意内容派生 63 位非负整数种子。"""
    digest = sha256_hex(canonical(list(parts)))
    return int(digest[:16], 16) % (2**63)
