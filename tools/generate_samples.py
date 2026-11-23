"""Generate the deterministic local sample workspace.

The script intentionally uses only the standard library and a fixed seed.  It
also makes the corpus dependency on the controlled vocabulary explicit.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"

ENTITY_WORDS = {
    "PER": ["张三", "李四", "王五", "赵六", "张三丰", "李清照"],
    "LOC": ["北京", "上海", "广州", "深圳", "杭州", "南京"],
    "ORG": ["清华", "北大", "央行", "医院", "邮局", "学校"],
    "TIME": ["春天", "夏至", "周一", "今年", "昨天", "周末"],
}
FILLERS = list("在的和于与去到有为上下前后开会工作又也从很北京上海清华")
all_chars = sorted(set("".join(sum(ENTITY_WORDS.values(), [])) + "".join(FILLERS)))
LABELS = ["B", "I", "E", "S", "O"]


def make_sample(rng: random.Random, index: int) -> dict:
    length = rng.randint(8, 17)
    chars = [rng.choice(FILLERS) for _ in range(length)]
    labels = ["O"] * length
    types = ["O"] * length
    occupied = [False] * length
    entity_count = 1 if length < 11 else rng.choice([1, 2, 2])
    for _ in range(entity_count):
        for attempt in range(30):
            type_code = rng.choice(list(ENTITY_WORDS))
            word = rng.choice(ENTITY_WORDS[type_code])
            start = rng.randint(0, length - len(word))
            positions = list(range(start, start + len(word)))
            if any(occupied[pos] for pos in positions):
                continue
            for offset, char in enumerate(word):
                chars[start + offset] = char
                labels[start + offset] = (
                    "S" if len(word) == 1 else ("B" if offset == 0 else ("E" if offset == len(word) - 1 else "I"))
                )
                types[start + offset] = type_code
                occupied[start + offset] = True
            break
    return {
        "id": f"s{index:03d}",
        "chars": " ".join(chars),
        "labels": " ".join(labels),
        "types": " ".join(types),
    }


def make_model(samples: list[dict], seed: int) -> dict:
    rng = random.Random(seed)
    label_counts: dict[str, Counter] = defaultdict(Counter)
    controlled = set()
    for sample in samples:
        chars = sample["chars"].split()
        labels = sample["labels"].split()
        controlled.update(chars)
        for char, label in zip(chars, labels):
            label_counts[char][label] += 1

    embedding_dim = 12
    embeddings: dict[str, list[float]] = {}
    for char in sorted(controlled):
        counts = label_counts[char]
        total = sum(counts.values()) or 1
        vector = [round(counts[label] / total * 5.0, 6) for label in LABELS]
        vector.extend(round((rng.random() - 0.5) * 0.02, 6) for _ in range(embedding_dim - len(LABELS)))
        embeddings[char] = vector
    zero = [0.0] * embedding_dim
    for key in ["@BOS", "@EOS"] + [f"@rel:{value}" for value in ["single", "start", "middle", "end"]] + [f"@hit:{value}" for value in range(0, 9)]:
        embeddings[key] = list(zero)

    emission = [[0.0 for _ in LABELS] for _ in range(embedding_dim)]
    for label_index in range(len(LABELS)):
        emission[label_index][label_index] = 1.0
    for dimension in range(len(LABELS), embedding_dim):
        for label_index in range(len(LABELS)):
            emission[dimension][label_index] = round((rng.random() - 0.5) * 0.01, 6)

    transitions = [[-4.0 for _ in LABELS] for _ in LABELS]
    allowed = {
        "B": {"I": 3.0, "E": 2.5},
        "I": {"I": 3.0, "E": 3.0},
        "E": {"B": 1.0, "S": 1.0, "O": 2.0},
        "S": {"B": 1.0, "S": 1.0, "O": 2.0},
        "O": {"B": 2.0, "S": 2.0, "O": 1.0},
    }
    for from_label, destination in allowed.items():
        for to_label, score in destination.items():
            transitions[LABELS.index(from_label)][LABELS.index(to_label)] = score
    start_weights = [1.0, -5.0, -5.0, 1.0, 1.0]

    vocabulary = []
    entry_index = 0
    for type_code, words in ENTITY_WORDS.items():
        for word in words:
            entry_index += 1
            vocabulary.append({"id": f"v{entry_index:03d}", "text": word, "type": type_code})
    for word in ["开会", "工作", "从前", "南方", "北方", "大湖"]:
        entry_index += 1
        vocabulary.append({"id": f"v{entry_index:03d}", "text": word, "type": "ORG"})

    return {
        "model_version": "local-hash-bieso-v1",
        "feature_dim": embedding_dim,
        "hash_buckets": 97,
        "labels": LABELS,
        "char_embeddings": {key: embeddings[key] for key in sorted(embeddings)},
        "emission_weights": emission,
        "transition_weights": transitions,
        "start_weights": start_weights,
        "default_entity_type": "ORG",
        "vocabulary": vocabulary,
    }


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261124)
    samples = [make_sample(rng, index) for index in range(1, 19)]
    corpus_lines = [json.dumps(sample, ensure_ascii=False, separators=(",", ":")) for sample in samples]
    (SAMPLES / "corpus.jsonl").write_text("\n".join(corpus_lines) + "\n", encoding="utf-8")

    model = make_model(samples, 20261124)
    (SAMPLES / "model.json").write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    config = {
        "interaction_latency_budget_ms": 600,
        "fixed_decimals": 4,
        "quantization_level": 1,
        "quantization_levels": [1, 2, 8],
        "thread_pool_workers": 4,
        "snapshot_commit_interval_commands": 3,
        "boundary_score_tolerance": 4.50006,
        "temp_dir": "artifacts/tmp",
        "default_entity_type": "ORG",
    }
    (SAMPLES / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    replacement = dict(samples[2])
    replacement["id"] = "s003"
    replacement_chars = ["在", "杭", "州", "和", "李", "四", "开", "会"]
    replacement_labels = ["O", "B", "E", "O", "B", "E", "O", "O"]
    replacement_types = ["O", "LOC", "LOC", "O", "PER", "PER", "O", "O"]
    replacement.update(
        {
            "chars": " ".join(replacement_chars),
            "labels": " ".join(replacement_labels),
            "types": " ".join(replacement_types),
        }
    )
    commands = [
        {"seq": 1, "type": "noop"},
        {
            "seq": 2,
            "type": "vocab_add",
            "entry": {"id": "v100", "text": "南湖", "type": "LOC"},
        },
        {
            "seq": 3,
            "type": "vocab_replace",
            "entry_id": "v019",
            "entry": {"id": "v019", "text": "开会", "type": "TIME"},
        },
        {"seq": 4, "type": "corpus_replace", "sample_id": "s003", "sample": replacement},
        {
            "seq": 5,
            "type": "emission_adjust",
            "target": {"dimension": 0, "label": "B"},
            "delta": 0.2,
        },
        {
            "seq": 6,
            "type": "transition_adjust",
            "target": {"from": "O", "to": "B"},
            "delta": 0.15,
        },
        {"seq": 7, "type": "fixed_point_set", "fixed_decimals": 5},
        {"seq": 8, "type": "quantization_set", "quantization_level": 2},
        {"seq": 9, "type": "vocab_delete", "entry_id": "v020"},
        {
            "seq": 10,
            "type": "emission_adjust",
            "target": {"dimension": 5, "label": "S"},
            "delta": -0.03,
        },
        {"seq": 11, "type": "corpus_delete", "sample_id": "s018"},
        {"seq": 12, "type": "quantization_set", "quantization_level": 1},
    ]
    (SAMPLES / "commands.json").write_text(
        json.dumps(commands, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    invalid = SAMPLES / "invalid"
    invalid.mkdir(exist_ok=True)
    (invalid / "corpus_invalid.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "i001", "chars": "张 三", "labels": "B E", "types": "PER PER"}, ensure_ascii=False),
                json.dumps({"id": "i001", "chars": "李 四", "labels": "B E", "types": "PER PER"}, ensure_ascii=False),
                json.dumps({"id": "i002", "chars": "张 三", "labels": "B", "types": "PER"}, ensure_ascii=False),
                json.dumps({"id": "i003", "chars": "张 三", "labels": "B O", "types": "PER PER"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (invalid / "commands_invalid.json").write_text(
        json.dumps(
            [
                {"seq": 1, "type": "unknown_command"},
                {"seq": 2, "type": "emission_adjust", "target": {"dimension": 0, "label": "B"}},
                {"seq": 3, "type": "transition_adjust", "target": {"from": "O", "to": "B"}},
                {"seq": 4, "type": "corpus_delete"},
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (invalid / "model_invalid.json").write_text(
        json.dumps(
            {
                "model_version": "invalid-structural",
                "feature_dim": 0,
                "hash_buckets": 0,
                "labels": LABELS,
                "char_embeddings": {},
                "emission_weights": [],
                "transition_weights": [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
