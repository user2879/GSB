"""Fixed-point integer token bucket.

Each (service, bucket_key) pair owns one bucket. Balances are quantized by
the session scale factor s in 0..6: one token == 10**s scaled units. Refill
is driven by the declared monotonic arrival_ms clock of the request script
(never the wall clock), so the token trajectory of a request sequence is
byte-identical under any processing order, provided each service evaluates
its own requests in seq order (enforced by the engine's ordered barrier).

Declared rules:
  * bucket starts full at capacity_tokens * scale units;
  * refill_tokens = elapsed_ms // refill_interval_ms (integer);
  * balance is clamped to capacity_tokens * scale after refill;
  * each request consumes cost_tokens * scale units when admitted.
"""

from dataclasses import dataclass

MIN_SCALE = 0
MAX_SCALE = 6


@dataclass(frozen=True)
class TokenDecision:
    before: int      # scaled units before refill
    refilled: int    # scaled units actually added after clamping
    allowed: bool
    after: int       # scaled units after consumption (if admitted)


class TokenBucket:
    def __init__(self, capacity_tokens, refill_interval_ms, cost_tokens,
                 scale):
        if not (MIN_SCALE <= scale <= MAX_SCALE):
            raise ValueError(f"scale must be in [{MIN_SCALE},{MAX_SCALE}]")
        self.scale = 10 ** scale
        self.capacity_units = capacity_tokens * self.scale
        self.refill_interval_ms = refill_interval_ms
        self.cost_units = cost_tokens * self.scale
        self.balance = self.capacity_units
        self.last_ms = None

    def evaluate(self, arrival_ms):
        elapsed = 0 if self.last_ms is None else arrival_ms - self.last_ms
        if elapsed < 0:
            raise ValueError("arrival_ms went backwards")
        refill_tokens = elapsed // self.refill_interval_ms
        before = self.balance
        self.balance = min(self.capacity_units,
                           self.balance + refill_tokens * self.scale)
        refilled = self.balance - before
        if self.balance >= self.cost_units:
            allowed = True
            self.balance -= self.cost_units
        else:
            allowed = False
        self.last_ms = arrival_ms
        return TokenDecision(before=before, refilled=refilled,
                             allowed=allowed, after=self.balance)
