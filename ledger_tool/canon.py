"""规范化序列化与摘要工具.

所有落盘与摘要计算均使用 canonical_json: 键排序、紧凑分隔符,
不依赖字典插入顺序或哈希遍历顺序.
"""

import hashlib
import json

DIGEST_HEX_LEN = 64


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(*parts):
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def digest_of(obj):
    return sha256_hex(canonical_json(obj).encode("utf-8"))


def is_hex_digest(value, length=DIGEST_HEX_LEN):
    if not isinstance(value, str) or len(value) != length:
        return False
    return all(c in "0123456789abcdef" for c in value)
