"""规范化 JSON 序列化与 SHA-256 内容摘要。

:func:`canonical_json` 输出固定分隔符、固定键顺序（排序）、无多余空白的
字节串，保证结果序列化逐字节可复现，且与执行交错无关。
"""

import hashlib
import json


def canonical_json(obj) -> bytes:
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_digest(obj) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def dump_canonical(path, obj) -> None:
    with open(path, "wb") as handle:
        handle.write(canonical_json(obj))
        handle.write(b"\n")
