"""测试公共构造工具：直接在内存中构造合法/非法图声明。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from cache_invalidation.loader import build_graph


SAMPLE_NAMESPACES = {
    "namespaces": [
        {"name": "catalog", "parents": []},
        {"name": "pricing", "parents": ["catalog"]},
        {"name": "inventory", "parents": ["catalog"]},
        {"name": "storefront", "parents": ["pricing", "inventory"]},
        {"name": "search", "parents": ["storefront"]},
    ]
}

SAMPLE_ENTRIES = {
    "entries": [
        {"key": 100, "namespace": "catalog", "expire_ms": 10000,
         "digest": "a", "digest_algorithm": "crc32",
         "annotations": [
             {"namespace": "pricing", "key": 200},
             {"namespace": "inventory", "key": 300}]},
        {"key": 101, "namespace": "catalog", "expire_ms": 20000,
         "digest": "b", "digest_algorithm": "crc32",
         "annotations": [{"namespace": "search", "key": 500}]},
        {"key": 200, "namespace": "pricing", "expire_ms": 15000,
         "digest": "c", "digest_algorithm": "sha256",
         "annotations": [{"namespace": "storefront", "key": 400}]},
        {"key": 300, "namespace": "inventory", "expire_ms": 12000,
         "digest": "d", "digest_algorithm": "sha1",
         "annotations": [{"namespace": "storefront", "key": 400}]},
        {"key": 400, "namespace": "storefront", "expire_ms": 30000,
         "digest": "e", "digest_algorithm": "crc32",
         "annotations": [{"namespace": "search", "key": 500}]},
        {"key": 500, "namespace": "search", "expire_ms": 40000,
         "digest": "f", "digest_algorithm": "none", "annotations": []},
    ]
}


def sample_graph():
    return build_graph(copy.deepcopy(SAMPLE_NAMESPACES),
                       copy.deepcopy(SAMPLE_ENTRIES))


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path
