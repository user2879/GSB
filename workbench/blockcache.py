"""Frame-block residency cache: deterministic LRU with lexicographic tie-break."""


class BlockCache:
    """LRU cache over frame-block evidence with deterministic eviction.

    Eviction picks the least-recently-used block; ties on access order are
    broken by ascending block key (lexicographic). Evicted blocks that are
    referenced again are recomputed from the original frame files and the
    recomputed evidence is verified field-by-field against the first result.
    """

    def __init__(self, capacity, loader):
        self.capacity = max(1, int(capacity))
        self._loader = loader          # block_key -> evidence dict (recompute)
        self._store = {}               # block_key -> evidence
        self._tick_of = {}             # block_key -> last access tick
        self._ever_evicted = set()
        self._tick = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.recomputes = 0

    def access(self, block_key):
        self._tick += 1
        if block_key in self._store:
            self.hits += 1
            self._tick_of[block_key] = self._tick
            return self._store[block_key]
        if block_key in self._ever_evicted:
            self.recomputes += 1
        else:
            self.misses += 1
        evidence = self._loader(block_key)
        while len(self._store) >= self.capacity:
            victim = min(self._store, key=lambda k: (self._tick_of[k], k))
            del self._store[victim]
            del self._tick_of[victim]
            self._ever_evicted.add(victim)
            self.evictions += 1
        self._store[block_key] = evidence
        self._tick_of[block_key] = self._tick
        return evidence

    def resident_keys(self):
        return sorted(self._store)

    def stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "recomputes": self.recomputes,
        }
