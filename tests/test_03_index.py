"""业务要求 3：嵌入、检索排序、阈值决断与边界样本。"""

from __future__ import annotations

import unittest

from vrep.index import build_index, query


class IndexRuleTests(unittest.TestCase):
    def _embeddings(self) -> dict:
        return {
            "a": {"vector": [1, 0], "embedding_key": "ka", "chunk_keys": ["c1"]},
            "b": {"vector": [1, 1], "embedding_key": "kb", "chunk_keys": ["c2"]},
            "c": {"vector": [0, 1], "embedding_key": "kc", "chunk_keys": ["c3"]},
            "d": {"vector": [-1, 0], "embedding_key": "kd", "chunk_keys": ["c4"]},
        }

    def test_similarity_desc_then_name_asc(self) -> None:
        embeddings = self._embeddings()
        index_doc = build_index(embeddings, threshold=0.0)
        view = index_doc["views"]["a"]
        similarities = [item["similarity"] for item in view]
        self.assertEqual(similarities, sorted(similarities, reverse=True))

    def test_boundary_equality_rule(self) -> None:
        embeddings = self._embeddings()
        # a=[1,0] 与 b=[1,1] 的余弦相似度固定为 1/sqrt(2)
        threshold = round(1.0 / (2.0 ** 0.5), 9)
        entries = [
            {"image": name, "vector": payload["vector"]}
            for name, payload in embeddings.items()
        ]
        result = query(embeddings["a"]["vector"], entries, threshold)
        boundary = [item for item in result["candidates"] if item["boundary"]]
        self.assertTrue(any(item["image"] == "b" for item in boundary))
        self.assertIn("b", result["boundary"])

    def test_excluded_below_threshold(self) -> None:
        embeddings = self._embeddings()
        entries = [
            {"image": name, "vector": payload["vector"]}
            for name, payload in embeddings.items()
        ]
        result = query(embeddings["a"]["vector"], entries, 0.9)
        self.assertTrue(all(item["similarity"] >= 0.9 for item in result["candidates"]))

    def test_tie_break_by_image_name(self) -> None:
        embeddings = {
            "z": {"vector": [2, 0], "embedding_key": "kz", "chunk_keys": []},
            "a": {"vector": [2, 0], "embedding_key": "ka", "chunk_keys": []},
            "m": {"vector": [1, 0], "embedding_key": "km", "chunk_keys": []},
        }
        index_doc = build_index(embeddings, threshold=0.5)
        view = index_doc["views"]["m"]
        names = [item["image"] for item in view]
        self.assertEqual(names, ["a", "z"])
