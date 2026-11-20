"""Seed-driven, order-independent injection timeline generation.

Each injection item gets a stable key derived from a canonical hash of its
content (never from its position in the declaration file). The timeline is
sorted by that key, so re-ordering items in the JSON file yields a
byte-identical timeline. Start offsets come from random.Random(seed), so the
same declaration + seed always produces the same timeline and digest.
"""

import hashlib
import json
import random


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def stable_key(injection):
    return digest(injection.to_dict())[:16]


class TimelineEntry:
    __slots__ = ("seq", "stable_key", "type", "target", "params",
                 "duration_ms", "observe_ms", "planned_start_ms")

    def to_dict(self):
        return {
            "seq": self.seq,
            "stable_key": self.stable_key,
            "type": self.type,
            "target": self.target,
            "params": self.params,
            "duration_ms": self.duration_ms,
            "observe_ms": self.observe_ms,
            "planned_start_ms": self.planned_start_ms,
        }


def generate(spec, stagger_ms=200):
    """Build the deterministic timeline for a validated spec."""
    rng = random.Random(spec.seed)
    keyed = sorted(((stable_key(inj), inj) for inj in spec.injections),
                   key=lambda kv: kv[0])
    entries = []
    start = 0
    for seq, (key, inj) in enumerate(keyed):
        start += rng.randint(0, stagger_ms)
        e = TimelineEntry()
        e.seq = seq
        e.stable_key = key
        e.type = inj.type
        e.target = inj.target
        e.params = dict(inj.params)
        e.duration_ms = inj.duration_ms
        e.observe_ms = inj.observe_ms
        e.planned_start_ms = start
        entries.append(e)
    return entries


def timeline_digest(entries):
    return digest([e.to_dict() for e in entries])
