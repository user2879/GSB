"""Deterministic local sample generation from a controlled vocabulary."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from .util import atomic_write_bytes

WORDS = [f"word_{index:03d}" for index in range(64)]
ENTITIES = [f"ENT_{name}" for name in ("ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA")]
BASE_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

PREFERENCE_TEMPLATE = {
    "pair_id": "pair_{index:06d}",
    "prompt": None,
    "chosen": None,
    "rejected": None,
    "annotator_id": None,
    "timestamp": None,
}


def _rng(seed: int):
    """Small deterministic LCG so the generator never depends on global RNG state."""
    state = {"value": seed & 0x7FFFFFFF}

    def next_int(modulus: int) -> int:
        state["value"] = (1_103_515_245 * state["value"] + 12_345) & 0x7FFFFFFF
        return state["value"] % modulus

    return next_int


def _phrase(next_int, length: int) -> str:
    return " ".join(WORDS[next_int(len(WORDS))] for _ in range(length))


def generate_preferences(path: str, count: int, seed: int = 42) -> list:
    next_int = _rng(seed)
    rows = []
    for index in range(count):
        prompt = f"prompt {_phrase(next_int, 6)}"
        chosen = f"{_phrase(next_int, 10)} {ENTITIES[next_int(len(ENTITIES))]}"
        rejected = f"{_phrase(next_int, 7)} {ENTITIES[next_int(len(ENTITIES))]}"
        row = dict(PREFERENCE_TEMPLATE)
        row.update(
            pair_id=f"pair_{index:06d}",
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            annotator_id=f"annotator_{next_int(8):02d}",
            timestamp=(BASE_EPOCH + timedelta(minutes=index)).isoformat().replace("+00:00", "Z"),
        )
        rows.append(row)
    _write_jsonl(path, rows)
    return rows


def generate_evaluations(path: str, count: int, seed: int = 99) -> list:
    next_int = _rng(seed)
    rows = []
    for index in range(count):
        context_entities = ENTITIES[: 2 + next_int(3)]
        context = (
            f"context {_phrase(next_int, 8)} "
            + " ".join(context_entities)
            + f" metric {10 + next_int(80)} reference snippet_{index % 11}"
        )
        reference = (
            f"answer {_phrase(next_int, 5)} {context_entities[0]} value {10 + next_int(80)}"
        )
        mode = next_int(4)
        if mode == 0:  # entity hallucination
            answer = f"answer {_phrase(next_int, 4)} {ENTITIES[(next_int(len(ENTITIES)))]}"
            if answer.split()[-1] in context_entities:
                answer = f"answer {_phrase(next_int, 3)} ENT_UNKNOWN_{index}"
        elif mode == 1:  # numeric deviation
            answer = f"answer {_phrase(next_int, 4)} {context_entities[0]} value {900 + next_int(90)}"
        elif mode == 2:  # bad citation
            answer = f"answer {_phrase(next_int, 4)} [REF:missing_snippet_{index}]"
        else:  # clean sample aligned with reference
            answer = reference
        rows.append(
            {
                "sample_id": f"eval_{index:06d}",
                "context": context,
                "model_answer": answer,
                "reference_answer": reference,
            }
        )
    _write_jsonl(path, rows)
    return rows


def generate_rules(path: str) -> dict:
    payload = {
        "rules": [
            {
                "rule_id": "entity_consistency",
                "name": "entity overlap",
                "weight": 0.4,
                "priority": 1,
                "params": {"trigger_threshold": 0.6},
            },
            {
                "rule_id": "numeric_consistency",
                "name": "numeric deviation",
                "weight": 0.35,
                "priority": 2,
                "params": {"relative_tolerance": 0.05, "trigger_threshold": 0.5},
            },
            {
                "rule_id": "citation_consistency",
                "name": "citation existence",
                "weight": 0.25,
                "priority": 3,
                "params": {"trigger_threshold": 0.0},
            },
        ]
    }
    atomic_write_bytes(path, (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return payload


DEFAULT_CONFIG = """\
# Deterministic pipeline configuration (no network, no runtime randomness).
feature_dim: 16
hash_buckets: 512
precision_level: 2
tokenizer_seed: 7
preference_sample_ratio: 1.0
reward_boundary_threshold: 0.05
hallucination_threshold: 0.45
metrics:
  - sample_count
  - exact_match_rate
  - token_precision
  - token_recall
  - token_f1
  - numeric_mae
concurrency: 4
snapshot_interval: 50
tiers:
  - {name: small, max_samples: 200}
  - {name: medium, max_samples: 1200}
  - {name: large, max_samples: 6000}
  - {name: xlarge, max_samples: 30000}
"""


def generate_config(path: str) -> None:
    atomic_write_bytes(path, DEFAULT_CONFIG.encode("utf-8"))


DEFAULT_INJECTION = {
    "items": [
        {
            "item_no": 1,
            "enabled": False,
            "kind": "non_finite_feature",
            "target_id": "pair_000003",
            "detail": "replace reward diff with NaN",
        },
        {
            "item_no": 2,
            "enabled": False,
            "kind": "rule_weight_out_of_range",
            "target_id": "numeric_consistency",
            "detail": "set rule weight to 1.5 in an alternate rule table",
        },
        {
            "item_no": 3,
            "enabled": False,
            "kind": "tier_limit_zero",
            "target_id": "small",
            "detail": "set first tier max_samples to 0 in an alternate config",
        },
        {
            "item_no": 4,
            "enabled": False,
            "kind": "threshold_boundary",
            "target_id": "eval_000000",
            "detail": "set hallucination threshold exactly equal to a sample risk",
        },
    ]
}


def generate_injection(path: str) -> None:
    atomic_write_bytes(
        path, (json.dumps(DEFAULT_INJECTION, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )


def _write_jsonl(path: str, rows: list) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_bytes(path, payload.encode("utf-8"))


def generate_all(sample_dir: str, preference_count: int = 600, evaluation_count: int = 300,
                 seed: int = 42) -> dict:
    os.makedirs(sample_dir, exist_ok=True)
    paths = {
        "preferences": os.path.join(sample_dir, "preferences.jsonl"),
        "evaluations": os.path.join(sample_dir, "evaluations.jsonl"),
        "rules": os.path.join(sample_dir, "rules.json"),
        "config": os.path.join(sample_dir, "config.yaml"),
        "injection": os.path.join(sample_dir, "injection.json"),
    }
    generate_preferences(paths["preferences"], preference_count, seed)
    generate_evaluations(paths["evaluations"], evaluation_count, seed + 57)
    generate_rules(paths["rules"])
    generate_config(paths["config"])
    generate_injection(paths["injection"])
    return paths
