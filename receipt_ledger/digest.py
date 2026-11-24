"""规范化编码与内容摘要。

所有持久化、逐字节比对与链式校验都基于 ``canonical_json`` 的输出：
排序键、无空白、固定分隔符，因此不依赖字典/哈希遍历顺序，跨进程稳定。
"""

import hashlib
import json


def canonical_json(obj) -> bytes:
    """生成确定性 JSON 字节串（sort_keys、紧凑分隔、保留 UTF-8）。"""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def content_digest(obj, prefix: str = "") -> str:
    """对象的内容摘要（blake2b 截断 32 字节十六进制），可带前缀区分域。"""
    h = hashlib.blake2b(digest_size=32)
    h.update(prefix.encode("utf-8"))
    h.update(b":")
    h.update(canonical_json(obj))
    return h.hexdigest()


def chain_digest(prev_digest: str, canonical_payload: bytes) -> str:
    """批次链式摘要：前一批摘要与当前规范化载荷的绑定摘要。"""
    h = hashlib.blake2b(digest_size=32)
    h.update(prev_digest.encode("ascii"))
    h.update(b"|")
    h.update(canonical_payload)
    return h.hexdigest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

