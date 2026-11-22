"""固定种子生成全部本地样本，保证可复现。"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

TAGS = ["n", "v", "adj", "adv"]
PUNCT = "，。！？；："
HANZI = "天地人和山水风火云雨星月日花草树木鸟鱼虫路石田金银玉玉书笔墨纸砚春夏秋冬东西南北上下左右大小多少红黄蓝绿白黑青紫"


def build_charset() -> list[str]:
    chars = list(dict.fromkeys(list(HANZI) + list(PUNCT)))
    return chars


def gen_vocab(rng: random.Random, chars: list[str]) -> list[tuple[str, int, str]]:
    vocab: list[tuple[str, int, str]] = []
    used: set[str] = set()
    lengths = [2, 2, 2, 3, 3, 4]
    for idx in range(36):
        while True:
            length = rng.choice(lengths)
            word = "".join(rng.choice(chars[:32]) for _ in range(length))
            if word not in used and not any(ch in PUNCT for ch in word):
                used.add(word)
                break
        freq = rng.randint(1, 500)
        tag = TAGS[idx % len(TAGS)]
        vocab.append((word, freq, tag))
    return vocab


def gen_sentences(rng: random.Random, vocab: list[tuple[str, int, str]], chars: list[str], count: int) -> list[str]:
    lines: list[str] = []
    words = [item[0] for item in vocab]
    for idx in range(1, count + 1):
        parts: list[str] = []
        target = rng.randint(6, 14)
        length = 0
        while length < target:
            if rng.random() < 0.72:
                token = rng.choice(words)
            else:
                token = "".join(rng.choice(chars[:32]) for _ in range(rng.randint(1, 2)))
            parts.append(token)
            length += len(token)
        text = "".join(parts) + rng.choice(list(PUNCT))
        lines.append(f"S{idx:03d}\t{text}")
    return lines


def gen_prob_table(rng: random.Random, vocab: list[tuple[str, int, str]]) -> dict:
    tags = TAGS

    def log_distribution(n: int) -> list[float]:
        weights = [rng.uniform(0.05, 1.0) for _ in range(n)]
        total = sum(weights)
        return [math.log(w / total) for w in weights]

    start = dict(zip(tags, log_distribution(len(tags))))
    transition = {tag: dict(zip(tags, log_distribution(len(tags)))) for tag in tags}
    emission: dict[str, dict[str, float]] = {}
    for word, _freq, tag in vocab:
        row: dict[str, float] = {}
        for t in tags:
            base = rng.uniform(0.05, 1.0) if t == tag else rng.uniform(0.005, 0.2)
            row[t] = math.log(base)
        emission[word] = row
    return {
        "tags": tags,
        "start_logp": start,
        "transition_logp": transition,
        "emission_logp": emission,
        "smoothing": 1e-4,
    }


def write_invalid_scenario(out: Path, vocab: list[tuple[str, int, str]]) -> None:
    scenario = out / "invalid"
    scenario.mkdir(parents=True, exist_ok=True)
    (scenario / "charset.txt").write_text("".join(build_charset()), encoding="utf-8")
    (scenario / "sentences.txt").write_text(
        "S001\t天地人和。\n"
        "S002\t\n"
        "S001\t重复编号。\n"
        "S003\tabc非法字母。\n",
        encoding="utf-8",
    )
    (scenario / "lexicon.txt").write_text(
        f"{vocab[0][0]} 100 n\n"
        "好词 0 n\n"
        "坏词 abc v\n"
        "孤词 50 ghost\n"
        "\tn\n",
        encoding="utf-8",
    )
    bad_table = {
        "tags": TAGS,
        "start_logp": {"n": float("nan"), "v": -1.0, "adj": -1.2, "adv": -0.8},
        "transition_logp": {t: {u: -1.0 + 0.05 * i for i, u in enumerate(TAGS)} for t in TAGS},
        "emission_logp": {vocab[0][0]: {t: -1.0 for t in TAGS}},
        "smoothing": 1e-4,
    }
    (scenario / "prob_table.json").write_text(json.dumps(bad_table, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples/base")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--sentences", type=int, default=24)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    chars = build_charset()
    (out / "charset.txt").write_text("".join(chars), encoding="utf-8")

    vocab = gen_vocab(rng, chars)
    (out / "lexicon.txt").write_text(
        "\n".join(f"{word} {freq} {tag}" for word, freq, tag in vocab)
        + f"\n{vocab[0][0]} 7 {vocab[0][2]}\n",
        encoding="utf-8",
    )
    (out / "sentences.txt").write_text("\n".join(gen_sentences(rng, vocab, chars, args.sentences)), encoding="utf-8")
    table = gen_prob_table(rng, vocab)
    (out / "prob_table.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")

    config = {
        "concurrency": 4,
        "first_response_budget_ms": 50.0,
        "snapshot_commit_interval_commands": 3,
        "low_confidence_threshold": 0.45,
        "fixed_point_bits": 16,
        "quantization_levels": 1024,
        "smoothing": 1e-4,
        "oov_rule": "equal",
        "oov_max_length": 3,
        "simulated_task_delay_ms": 0.0,
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    new_word = "".join(rng.choice(chars[:32]) for _ in range(3))
    table_v2 = json.loads(json.dumps(table))
    table_v2["start_logp"] = {tag: value + 0.01 for tag, value in table_v2["start_logp"].items()}
    commands = [
        {"type": "add_entry", "word": new_word, "freq": 88, "tag": "n"},
        {"type": "adjust_freq", "word": vocab[1][0], "freq": 999},
        {"type": "remove_entry", "word": vocab[2][0]},
        {"type": "replace_prob_table", "table": "@prob_table_v2.json"},
        {"type": "replace_sentence", "sid": "S003", "text": "".join(rng.choice(chars[:32]) for _ in range(8)) + "。"},
        {"type": "add_entry", "word": "", "freq": 1, "tag": "n"},
        {"type": "adjust_freq", "word": "不存在词", "freq": 5},
    ]
    (out / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "prob_table_v2.json").write_text(json.dumps(table_v2, ensure_ascii=False, indent=2), encoding="utf-8")
    write_invalid_scenario(out.parent, vocab)
    print(f"samples generated at {out} (seed={args.seed})")


if __name__ == "__main__":
    main()
