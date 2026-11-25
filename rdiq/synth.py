"""Deterministic local synthetic data generation from integer seeds.

The PRNG is a self-contained SplitMix64 implementation. No network or
external dataset is used; every generated document gets an explicit integer
seed and the resulting JSON files carry SHA-256 content digests.
"""

from .canonical import digest
from .models import write_json

MASK = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed):
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.state = seed & MASK

    def next_uint(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
        return value ^ (value >> 31)

    def index(self, modulus):
        return self.next_uint() % modulus

    def choose(self, values):
        return values[self.index(len(values))]


VOCABULARY_TERMS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
    "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
    "oscar", "papa", "quebec", "romeo", "sierra", "tango",
]
CHANNELS = ["feed", "index", "archive"]
TOPICS = {
    "diagnostics": ["alpha", "echo", "india"],
    "retrieval": ["bravo", "juliet", "november"],
    "quantization": ["charlie", "delta", "tango"],
}


def _payload(version, **body):
    document = {"version": version}
    document.update(body)
    document["digest"] = digest(document)
    return document


def generate_sample_set(output_dir, *, seed=20261126, precision_version=1,
                        weight_version=1, rule_version=1, channels=None,
                        scale=1024, width=16, rule_overrides=None,
                        weight_overrides=None):
    import os

    os.makedirs(output_dir, exist_ok=True)
    channels = list(channels or CHANNELS)

    vocabulary = _payload(1, terms=list(VOCABULARY_TERMS), seed=seed)
    rules = _payload(
        rule_version,
        topics={
            topic: {
                "anchors": list(rule_overrides.get(topic, anchors))
                if rule_overrides
                else list(anchors)
            }
            for topic, anchors in TOPICS.items()
        },
    )
    precision = _payload(
        precision_version, width=width, scale=scale, channels=channels
    )
    channel_weights = {"feed": 4200, "index": 3600, "archive": 2200}
    topic_weights = {
        "diagnostics": 1800,
        "retrieval": 2400,
        "quantization": 3000,
    }
    if weight_overrides:
        channel_weights.update(weight_overrides.get("channel_weights", {}))
        topic_weights.update(weight_overrides.get("topic_weights", {}))
    weights = _payload(
        weight_version,
        channel_weights={
            channel: value
            for channel, value in channel_weights.items()
            if channel in channels
        },
        topic_weights=topic_weights,
    )

    fragments = []
    for fragment_index in range(24):
        fragment_rng = SplitMix64(seed + 1000 + fragment_index)
        topic = list(TOPICS)[fragment_rng.index(len(TOPICS))]
        channel = fragment_rng.choose(channels)
        length = 4 + fragment_rng.index(5)
        terms = [fragment_rng.choose(VOCABULARY_TERMS) for _ in range(length)]
        anchor = TOPICS[topic][fragment_rng.index(len(TOPICS[topic]))]
        terms[0] = anchor
        fragments.append(
            {
                "fragment_id": f"frag-{fragment_index:03d}",
                "seed": seed + 1000 + fragment_index,
                "topic": topic,
                "channel": channel,
                "terms": terms,
            }
        )

    sessions = []
    for session_index in range(4):
        rounds = []
        for round_index in range(6):
            round_rng = SplitMix64(
                seed + 5000 + session_index * 100 + round_index
            )
            topic = list(TOPICS)[round_rng.index(len(TOPICS))]
            anchors = TOPICS[topic]
            query = [anchors[round_rng.index(len(anchors))]]
            for _ in range(2):
                query.append(round_rng.choose(VOCABULARY_TERMS))
            rounds.append(
                {
                    "round": round_index,
                    "query_terms": query,
                    "rule_version_ref": rule_version,
                    "precision_version_ref": precision_version,
                    "scale_decl": scale,
                }
            )
        sessions.append(
            {
                "session_id": f"sess-{session_index:02d}",
                "rule_version_ref": rule_version,
                "precision_version_ref": precision_version,
                "rounds": rounds,
            }
        )

    write_json(os.path.join(output_dir, "vocabulary.json"), vocabulary)
    write_json(os.path.join(output_dir, "rules.json"), rules)
    write_json(os.path.join(output_dir, "precision.json"), precision)
    write_json(os.path.join(output_dir, "weights.json"), weights)
    write_json(
        os.path.join(output_dir, "fragments.json"),
        _payload(1, seed=seed, fragments=fragments),
    )
    write_json(
        os.path.join(output_dir, "sessions.json"),
        _payload(1, seed=seed, sessions=sessions),
    )
    return output_dir
