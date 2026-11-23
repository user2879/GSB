from rerank_lab.cache import DeterministicLRUCache


def test_lru_eviction_is_stable():
    cache = DeterministicLRUCache(2)
    cache.put("a", "fusion", {"value": 1})
    cache.put("b", "fusion", {"value": 2})
    assert cache.get("a") == {"value": 1}
    evictions = cache.put("c", "fusion", {"value": 3})
    assert [event["block_key"] for event in evictions] == ["b"]
    assert cache.keys() == ["a", "c"]
    assert cache.get("b") is None


def test_returned_values_are_copies():
    cache = DeterministicLRUCache(2)
    cache.put("a", "fusion", {"nested": [1, 2]})
    value = cache.get("a")
    assert value is not None
    value["nested"].append(3)
    assert cache.get("a") == {"nested": [1, 2]}
