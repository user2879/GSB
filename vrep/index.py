"""检索索引构建与相似度查询（业务要求 3）。

决断规则（全量与增量共用同一纯函数，天然逐字段一致）：
- 相似度按固定 9 位小数四舍五入后再与阈值比较；
- 相似度 >= 阈值入选，相似度恰等于阈值的候选标注为边界样本；
- 结果按相似度降序，同分按图像名字典序升序；
- 未完成（预算待办、失败）的图像不进入索引。
"""

from __future__ import annotations

from .features import SIMILARITY_DECIMALS, cosine_similarity


def query(query_vector: list[int], entries: list[dict], threshold: float) -> dict:
    """对一组索引条目执行查询，返回候选与边界样本清单。"""
    candidates = []
    boundary = []
    threshold_rounded = round(float(threshold), SIMILARITY_DECIMALS)
    for entry in entries:
        similarity = cosine_similarity(query_vector, entry["vector"])
        if similarity < threshold_rounded:
            continue
        is_boundary = similarity == threshold_rounded
        item = {
            "image": entry["image"],
            "similarity": similarity,
            "boundary": is_boundary,
        }
        candidates.append(item)
        if is_boundary:
            boundary.append(entry["image"])
    candidates.sort(key=lambda item: (-item["similarity"], item["image"]))
    boundary.sort()
    return {"candidates": candidates, "boundary": boundary}


def build_index(embeddings: dict, threshold: float) -> dict:
    """由全部已完成图像嵌入构建索引，并为每个图像生成确定性检索视图。"""
    entries = [
        {
            "image": name,
            "vector": payload["vector"],
            "embedding_key": payload["embedding_key"],
            "chunk_keys": list(payload["chunk_keys"]),
        }
        for name, payload in sorted(embeddings.items())
    ]
    entry_keys = sorted(entry["embedding_key"] for entry in entries)
    index_doc = {
        "threshold": round(float(threshold), SIMILARITY_DECIMALS),
        "entry_keys": entry_keys,
        "images": [entry["image"] for entry in entries],
        "entries": entries,
        "views": {},
        "boundary_set": [],
    }

    boundary_union: set[str] = set()
    for entry in entries:
        result = query(entry["vector"], entries, threshold)
        # 检索视图排除查询图像自身。
        view = [item for item in result["candidates"] if item["image"] != entry["image"]]
        index_doc["views"][entry["image"]] = view
        for name in result["boundary"]:
            if name != entry["image"]:
                boundary_union.add(name)
    index_doc["boundary_set"] = sorted(boundary_union)
    return index_doc
