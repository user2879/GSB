"""Fixed-seed local sample generator.

Generates the local sample directory without any external service:
  turns.jsonl      - dialogue turns (id, speaker, terms from a controlled
                     vocabulary, ms span, sentiment label, tool calls)
  acoustics.jsonl  - acoustic evidence summary per turn (1:1)
  tools.json       - tool manifest (name, param schema, formula, dimension)
  fusion.yaml      - fusion configuration
  vocab.json       - controlled vocabulary + sentiment trigger lexicon

Seed is fixed: identical output on every run.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import yaml

SEED = 20261122
N_TURNS = 24

NEUTRAL_TERMS = [
    "today", "meeting", "schedule", "report", "data", "system", "update",
    "window", "table", "signal", "channel", "buffer", "packet", "session",
    "folder", "record", "status", "metric", "module", "config", "driver",
    "sensor", "panel", "layout",
]
POSITIVE_TERMS = [
    "glad", "smooth", "clear", "stable", "bright", "calm", "crisp", "warm",
]
NEGATIVE_TERMS = [
    "rough", "broken", "dim", "harsh", "noisy", "stuck", "cold", "faint",
]

LEXICON = {**{t: 1.0 for t in POSITIVE_TERMS}, **{t: -1.0 for t in NEGATIVE_TERMS}}
VOCAB = NEUTRAL_TERMS + POSITIVE_TERMS + NEGATIVE_TERMS

SPEAKERS = ["agent", "caller"]


def generate(out_dir):
    rng = random.Random(SEED)
    os.makedirs(out_dir, exist_ok=True)

    turns = []
    acoustics = []
    cursor_ms = 0
    for idx in range(1, N_TURNS + 1):
        turn_id = f"T{idx:03d}"
        n_terms = rng.randint(6, 14)
        terms = []
        pos = neg = 0
        for _ in range(n_terms):
            roll = rng.random()
            if roll < 0.22:
                term = rng.choice(POSITIVE_TERMS)
                pos += 1
            elif roll < 0.44:
                term = rng.choice(NEGATIVE_TERMS)
                neg += 1
            else:
                term = rng.choice(NEUTRAL_TERMS)
            terms.append(term)
        duration = rng.randint(800, 4200)
        start_ms = cursor_ms
        end_ms = cursor_ms + duration
        cursor_ms = end_ms + rng.randint(120, 480)
        polarity = pos - neg
        sentiment = "pos" if polarity > 0 else ("neg" if polarity < 0 else "neu")
        turns.append({
            "turn_id": turn_id,
            "speaker": SPEAKERS[idx % 2],
            "terms": terms,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "sentiment": sentiment,
            "tool_calls": [{
                "tool": "sentiment_polarity",
                "params": {"pos_count": pos, "neg_count": neg, "term_count": n_terms},
            }],
        })
        pitch_mean = round(rng.uniform(90.0, 260.0), 6)
        pitch_var = round(rng.uniform(50.0, 2400.0), 6)
        acoustics.append({
            "turn_id": turn_id,
            "pitch_mean": pitch_mean,
            "pitch_var": pitch_var,
            "energy_quantiles": {
                "0.25": round(rng.uniform(0.05, 0.45), 6),
                "0.5": round(rng.uniform(0.30, 0.70), 6),
                "0.75": round(rng.uniform(0.55, 0.98), 6),
            },
        })

    tools = {
        "tools": [
            {
                "name": "sentiment_polarity",
                "param_schema": {"pos_count": "integer", "neg_count": "integer",
                                 "term_count": "integer"},
                "formula": "(pos_count - neg_count) / max(term_count, 1) * 0.5 + 0.5",
                "dimension": "sentiment_polarity",
            },
            {
                "name": "speech_rate",
                "param_schema": {"word_count": "integer", "duration_ms": "integer"},
                "formula": "word_count / max(duration_ms, 1) * 1000",
                "dimension": "ratio",
            },
        ]
    }

    config = {
        "weights": {"text": 0.5, "acoustic": 0.3, "tool": 0.2},
        "temperature": 1.0,
        "bias": 0.0,
        "precision": {
            "frac_bits": 12,
            "quantum": 2,
            "allowed_frac_bits": [8, 12, 16, 20],
            "allowed_quantum": [1, 2, 4],
        },
        "significant_digits": 6,
        "tolerances": {"abs": 1e-06, "rel": 1e-06},
        "concurrency": 4,
        "fusion_point_budget": 1000,
        "interactive_latency_ms": 200,
        "snapshot_commit_interval": 1,
        "confidence_interval": [0.0, 1.0],
        "sentiment_threshold": 0.5,
        "expected_tool_dimension": "sentiment_polarity",
        "acoustic_quantiles": ["0.25", "0.5", "0.75"],
    }

    vocab = {"terms": VOCAB, "sentiment_lexicon": LEXICON}

    def dump_jsonl(name, rows):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump_jsonl("turns.jsonl", turns)
    dump_jsonl("acoustics.jsonl", acoustics)
    with open(os.path.join(out_dir, "tools.json"), "w", encoding="utf-8") as handle:
        json.dump(tools, handle, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "fusion.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as handle:
        json.dump(vocab, handle, ensure_ascii=False, indent=1)
    return {"turns": len(turns), "seed": SEED, "out_dir": out_dir}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="samples", help="output sample directory")
    args = parser.parse_args()
    info = generate(args.out)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
