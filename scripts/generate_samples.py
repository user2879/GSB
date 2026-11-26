"""Deterministically generate corpus, revision chain, config and command stream.

Everything derives from fixed integer seeds and the standard library ``random``
module, so re-running the script reproduces byte-identical YAML artifacts.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

# Controlled subword vocabulary: short Chinese subword tokens (1-2 chars).
VOCAB = [
    "订", "票", "机", "酒", "店", "退", "改", "查", "询", "账",
    "单", "发", "票", "客", "服", "投", "诉", "优", "惠", "券", "签",
    "预", "约", "取", "消", "价", "格", "行", "程", "问", "候",
]

INTENTS = ("INTENT_BOOK", "INTENT_REFUND", "INTENT_QUERY", "INTENT_SUPPORT")
CHANNELS = ("CH_WEB", "CH_APP", "CH_CALL")


def _yaml_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            payload,
            handle,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def build_trigger_terms(rng: random.Random) -> dict[str, dict]:
    """Hand-authored trigger terms per intent, plus a couple of generated."""
    terms = {
        "T_BOOK_1": {"term": ["预", "约"], "intent": "INTENT_BOOK", "weight": 6},
        "T_BOOK_2": {"term": ["订", "票"], "intent": "INTENT_BOOK", "weight": 7},
        "T_BOOK_3": {"term": ["机", "票"], "intent": "INTENT_BOOK", "weight": 5},
        "T_REFUND_1": {"term": ["退", "票"], "intent": "INTENT_REFUND", "weight": 8},
        "T_REFUND_2": {"term": ["退", "款"], "intent": "INTENT_REFUND", "weight": 9},
        "T_QUERY_1": {"term": ["查", "询"], "intent": "INTENT_QUERY", "weight": 6},
        "T_QUERY_2": {"term": ["账", "单"], "intent": "INTENT_QUERY", "weight": 5},
        "T_SUPPORT_1": {"term": ["客", "服"], "intent": "INTENT_SUPPORT", "weight": 7},
        "T_SUPPORT_2": {"term": ["投", "诉"], "intent": "INTENT_SUPPORT", "weight": 8},
    }
    # Two longer overlapping triggers to exercise longest-span suppression.
    terms["T_BOOK_4"] = {
        "term": ["预", "约", "订", "票"], "intent": "INTENT_BOOK", "weight": 12
    }
    terms["T_QUERY_3"] = {
        "term": ["查", "询", "账", "单"], "intent": "INTENT_QUERY", "weight": 11
    }
    return terms


def build_corpus(rng: random.Random, terms: dict[str, dict], n: int) -> list[dict]:
    texts: list[dict] = []
    term_list = list(terms.values())
    for i in range(n):
        channel = CHANNELS[i % len(CHANNELS)]
        gold = INTENTS[i % len(INTENTS)]
        length = rng.randint(5, 10)
        tokens: list[str] = []
        # Embed a trigger aligned with the gold intent most of the time.
        if rng.random() < 0.85:
            aligned = [t for t in term_list if t["intent"] == gold]
            chosen = rng.choice(aligned or term_list)
            tokens.extend(chosen["term"])
        while len(tokens) < length:
            tokens.append(VOCAB[rng.randrange(len(VOCAB))])
        rng.shuffle(tokens)
        texts.append(
            {
                "text_id": f"TEXT{i:04d}",
                "tokens": tokens,
                "channel": channel,
                "gold_label": gold,
            }
        )
    # Add a near-duplicate pair deliberately to exercise dedup.
    base = list(texts[2]["tokens"])
    twin = dict(texts[2])
    twin["text_id"] = "TEXT9001"
    twin["tokens"] = list(base)
    texts.append(twin)
    twin2 = dict(texts[2])
    twin2["text_id"] = "TEXT9002"
    mutated = list(base)
    if mutated:
        mutated[0] = VOCAB[(VOCAB.index(mutated[0]) + 1) % len(VOCAB)]
    twin2["tokens"] = mutated
    texts.append(twin2)
    return texts


def build_revisions(terms: dict[str, dict]) -> dict:
    root = "R-ROOT"
    # Revision 1: introduce half the entries.
    entry_ids = list(terms)
    half = entry_ids[: len(entry_ids) // 2]
    rev1_ops = [
        {
            "op": "upsert",
            "entry_id": eid,
            "term": terms[eid]["term"],
            "weight": terms[eid]["weight"],
            "intent": terms[eid]["intent"],
        }
        for eid in half
    ]
    # Revision 2: introduce the rest.
    rest = entry_ids[len(entry_ids) // 2:]
    rev2_ops = [
        {
            "op": "upsert",
            "entry_id": eid,
            "term": terms[eid]["term"],
            "weight": terms[eid]["weight"],
            "intent": terms[eid]["intent"],
        }
        for eid in rest
    ]
    # Revision 3: override one weight and reassign one intent.
    target = entry_ids[0]
    rev3_ops = [
        {
            "op": "upsert",
            "entry_id": target,
            "term": terms[target]["term"],
            "weight": terms[target]["weight"] + 3,
            "intent": terms[target]["intent"],
        }
    ]
    return {
        "root_id": root,
        "head_id": "R-3",
        "revisions": [
            {
                "revision_id": "R-1",
                "parent_id": root,
                "sequence": 1,
                "ops": rev1_ops,
            },
            {
                "revision_id": "R-2",
                "parent_id": "R-1",
                "sequence": 2,
                "ops": rev2_ops,
            },
            {
                "revision_id": "R-3",
                "parent_id": "R-2",
                "sequence": 3,
                "ops": rev3_ops,
            },
        ],
    }


def build_config() -> dict:
    return {
        "fraction_bits": 8,
        "gear": 2,
        "intents": list(INTENTS),
        "intent_thresholds": {
            "INTENT_BOOK": 5,
            "INTENT_REFUND": 6,
            "INTENT_QUERY": 4,
            "INTENT_SUPPORT": 5,
        },
        "dedup_threshold_permille": 700,
        "max_workers": 4,
        "global_match_budget": 60,
        "channel_quotas": {"CH_WEB": 30, "CH_APP": 30, "CH_CALL": 30},
        "per_batch_recompute_limit": 12,
        "fallback_intent": "INTENT_UNKNOWN",
    }


def build_commands(terms: dict[str, dict]) -> list[dict]:
    first_term = terms["T_BOOK_2"]
    return [
        {
            "seq": 1,
            "kind": "add_text",
            "payload": {
                "text_id": "TEXT8001",
                "tokens": ["客", "服", "投", "诉", "查", "询"],
                "channel": "CH_APP",
                "gold_label": "INTENT_SUPPORT",
            },
        },
        {
            "seq": 2,
            "kind": "replace_text",
            "payload": {
                "text_id": "TEXT0001",
                "tokens": ["退", "票", "改", "签", "查", "询", "账"],
                "channel": "CH_WEB",
                "gold_label": "INTENT_REFUND",
            },
        },
        {
            "seq": 3,
            "kind": "set_weight",
            "payload": {
                "entry_id": "T_BOOK_2",
                "term": first_term["term"],
                "weight": first_term["weight"] + 5,
                "intent": first_term["intent"],
            },
        },
        {"seq": 4, "kind": "set_dedup_threshold", "payload": {"dedup_threshold_permille": 650}},
        {"seq": 5, "kind": "set_gear", "payload": {"gear": 4}},
        {"seq": 6, "kind": "set_fraction_bits", "payload": {"fraction_bits": 16}},
        {"seq": 7, "kind": "delete_text", "payload": {"text_id": "TEXT0003"}},
        {
            "seq": 8,
            "kind": "delete_trigger",
            "payload": {"entry_id": "T_QUERY_2"},
        },
        {
            "seq": 9,
            "kind": "upsert_trigger",
            "payload": {
                "entry_id": "T_NEW_1",
                "term": ["优", "惠", "券"],
                "weight": 10,
                "intent": "INTENT_QUERY",
            },
        },
        {"seq": 10, "kind": "set_head_revision", "payload": {"head_id": "R-2"}},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument("--n", type=int, default=24)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    terms = build_trigger_terms(rng)
    out = Path(args.out)

    _yaml_dump(out / "corpus.yaml", {"vocab": VOCAB, "texts": build_corpus(rng, terms, args.n)})
    _yaml_dump(out / "revisions.yaml", build_revisions(terms))
    _yaml_dump(out / "config.yaml", build_config())
    _yaml_dump(out / "commands.yaml", {"commands": build_commands(terms)})
    print(f"generated deterministic artifacts in {out} (seed={args.seed})")


if __name__ == "__main__":
    main()
