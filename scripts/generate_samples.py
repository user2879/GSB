"""Generate deterministic local samples: documents, rule config and commands.

All randomness comes from a seeded numpy default_rng; invalid documents and
boundary-tuned documents are constructed explicitly so every run reproduces
the same files byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20261120
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"

ENTITY_VOCAB = {
    "ORG": ["Acme", "Globex", "Initech"],
    "PER": ["Alice", "Bob", "Carol"],
    "LOC": ["Beijing", "Shanghai", "Shenzhen"],
}
RELATION_TRIGGERS = [
    {
        "entry_id": "rel-acquire",
        "trigger": "acquired",
        "relation": "acquire",
        "head_type": "ORG",
        "tail_type": "ORG",
        "compatibility": 1.0,
    },
    {
        "entry_id": "rel-partner",
        "trigger": "partnered",
        "relation": "partner_with",
        "head_type": "ORG",
        "tail_type": "PER",
        "compatibility": 0.8,
    },
    {
        "entry_id": "rel-locate",
        "trigger": "located",
        "relation": "located_in",
        "head_type": "ORG",
        "tail_type": "LOC",
        "compatibility": 0.9,
    },
]
EVENT_TRIGGERS = [
    {
        "entry_id": "evt-invest",
        "trigger": "invested",
        "event_type": "investment",
        "max_args": 3,
        "args": [
            {"role": "investor", "entity_type": "ORG", "required": True, "priority": 10},
            {"role": "target", "entity_type": "ORG", "required": True, "priority": 9},
            {"role": "place", "entity_type": "LOC", "required": False, "priority": 5},
        ],
    },
    {
        "entry_id": "evt-meet",
        "trigger": "met",
        "event_type": "meeting",
        "max_args": 2,
        "args": [
            {"role": "participant_a", "entity_type": "PER", "required": True, "priority": 8},
            {"role": "participant_b", "entity_type": "PER", "required": True, "priority": 7},
            {"role": "place", "entity_type": "LOC", "required": False, "priority": 4},
        ],
    },
]

TEMPLATES = [
    ["Acme", "acquired", "Globex", "in", "Shanghai"],
    ["Alice", "partnered", "with", "Initech", "near", "Beijing"],
    ["Globex", "invested", "in", "Acme", "at", "Shenzhen"],
    ["Bob", "met", "Carol", "in", "Beijing"],
    ["Initech", "located", "in", "Shenzhen", "and", "Acme", "located", "in", "Shanghai"],
]


def synthetic_positions(words: list[str]) -> list[list[int]]:
    positions = []
    cursor = 0
    for word in words:
        positions.append([cursor, cursor + len(word)])
        cursor += len(word) + 1
    return positions


def make_document(rng: np.random.Generator, index: int) -> dict:
    template = list(TEMPLATES[index % len(TEMPLATES)])
    extra = rng.choice(["recently", "officially", "quietly", "reportedly"], size=1)[0]
    insert_at = int(rng.integers(0, len(template) + 1))
    words = template[:insert_at] + [str(extra)] + template[insert_at:]
    return {
        "doc_id": f"doc-{index:03d}",
        "words": words,
        "positions": synthetic_positions(words),
        "text": " ".join(words),
    }


def build_documents() -> list[str]:
    rng = np.random.default_rng(SEED)
    lines: list[str] = []
    for index in range(12):
        doc = make_document(rng, index)
        lines.append(json.dumps(doc, ensure_ascii=False, sort_keys=True))

    boundary_words = ["Acme", "acquired", "Globex", "and", "Initech", "located", "in", "Beijing"]
    lines.append(
        json.dumps(
            {
                "doc_id": "doc-boundary",
                "words": boundary_words,
                "positions": synthetic_positions(boundary_words),
                "text": " ".join(boundary_words),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )

    invalid_lines = [
        json.dumps({"doc_id": "doc-empty", "words": [], "positions": []}, sort_keys=True),
        "{not valid json",
        json.dumps(
            {
                "doc_id": "doc-bad-pos",
                "words": ["Acme", "acquired"],
                "positions": [[0, 4], [99, 107]],
                "text": "Acme acquired",
            },
            sort_keys=True,
        ),
        json.dumps(
            {
                "doc_id": "doc-unknown-type",
                "words": ["Acme"],
                "positions": [[0, 4]],
                "text": "Acme",
                "entity_hints": [{"entity_type": "WEIRD", "start": 0, "end": 1}],
            },
            sort_keys=True,
        ),
    ]
    lines.extend(invalid_lines)
    return lines


def build_config() -> dict:
    return {
        "entity_types": ENTITY_VOCAB,
        "relation_triggers": RELATION_TRIGGERS,
        "event_triggers": EVENT_TRIGGERS,
        "relation_band_thresholds": [0.5495, 0.72],
        "event_band_thresholds": [0.5429, 0.7],
        "band_labels": ["low", "medium", "high"],
        "fixed_point_scale": 4,
        "interaction_budget_ms": 25,
        "thread_pool_workers": 3,
        "snapshot_commit_interval": 2,
        "relation_window": 4,
        "event_window": 4,
        "max_event_args": 3,
        "scoring_weights": {"trigger_decay": 0.5, "compatibility": 0.3, "distance": 0.2},
    }


def build_commands() -> list[dict]:
    replacement_words = ["Globex", "partnered", "with", "Alice", "in", "Shanghai", "last", "week"]
    return [
        {
            "command_id": "cmd-001",
            "type": "update_relation_trigger",
            "entry_id": "rel-partner",
            "trigger": "partnered",
            "relation": "partner_with",
            "head_type": "ORG",
            "tail_type": "PER",
            "compatibility": 0.95,
        },
        {
            "command_id": "cmd-002",
            "type": "replace_document",
            "document": {
                "doc_id": "doc-001",
                "words": replacement_words,
                "positions": synthetic_positions(replacement_words),
                "text": " ".join(replacement_words),
            },
        },
        {
            "command_id": "cmd-003",
            "type": "update_event_band_thresholds",
            "thresholds": [0.45, 0.72],
        },
        {
            "command_id": "cmd-004",
            "type": "update_relation_window",
            "window": 6,
        },
        {
            "command_id": "cmd-005",
            "type": "delete_document",
            "doc_id": "doc-005",
        },
        {
            "command_id": "cmd-006",
            "type": "replace_document",
            "document": {
                "doc_id": "doc-001",
                "words": ["Acme", "invested", "in", "Globex", "at", "Beijing"],
                "positions": synthetic_positions(["Acme", "invested", "in", "Globex", "at", "Beijing"]),
                "text": "Acme invested in Globex at Beijing",
            },
        },
        {
            "command_id": "cmd-007",
            "type": "update_event_trigger",
            "entry_id": "evt-meet",
            "trigger": "met",
            "event_type": "meeting",
            "max_args": 2,
            "args": [
                {"role": "participant_a", "entity_type": "PER", "required": True, "priority": 8},
                {"role": "participant_b", "entity_type": "PER", "required": True, "priority": 7},
                {"role": "place", "entity_type": "LOC", "required": False, "priority": 4},
            ],
        },
    ]


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLES_DIR / "documents.jsonl").write_text("\n".join(build_documents()) + "\n", encoding="utf-8")
    (SAMPLES_DIR / "config.json").write_text(
        json.dumps(build_config(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (SAMPLES_DIR / "commands.json").write_text(
        json.dumps(build_commands(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"samples generated under {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
