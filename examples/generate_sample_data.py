"""生成确定性本地样本：文本行、基础词典、对数域概率表与增量配置。"""

from __future__ import annotations

import json
import math
from pathlib import Path

TAGS = ["n", "v", "a", "d"]

SENTENCES = [
    ("S001", "研究生物很有意思"),
    ("S002", "他从小学电脑"),
    ("S003", "我们用机器学习算法训练模型"),
    ("S004", "乒乓球拍卖完了"),
    ("S005", "这个新项目由专家组验收"),
    ("S006", "它们在野外快速奔跑"),
    ("S007", "村民们沿着河岸散步"),
    ("S008", "机器学习算法依赖高质量数据"),
    ("S009", "黑猫蹲在窗台晒太阳"),
    ("S010", "老师鼓励学生主动提问"),
    ("S011", "这家初创公司发布了新产品"),
    ("S012", "雨后的空气格外清新"),
]

INVALID_LINES = [
    "S013\t",
    "S001\t重复编号的句子",
]

DICT_ENTRIES = [
    ("研究", 120, "v"),
    ("研究生", 90, "n"),
    ("生物", 110, "n"),
    ("很", 80, "d"),
    ("有意思", 70, "a"),
    ("他", 60, "n"),
    ("从小", 45, "d"),
    ("小学", 95, "n"),
    ("电脑", 100, "n"),
    ("我们", 85, "n"),
    ("在", 75, "d"),
    ("自然", 60, "n"),
    ("自然语言", 55, "n"),
    ("语言", 70, "n"),
    ("处理", 65, "v"),
    ("实验室", 80, "n"),
    ("学习", 90, "v"),
    ("分词", 50, "v"),
    ("训练", 35, "v"),
    ("模型", 55, "n"),
    ("乒乓球", 88, "n"),
    ("乒乓球拍", 40, "n"),
    ("拍卖", 60, "v"),
    ("卖完", 35, "v"),
    ("了", 90, "d"),
    ("这个", 55, "d"),
    ("新", 40, "a"),
    ("项目", 75, "n"),
    ("由", 30, "d"),
    ("专家", 65, "n"),
    ("组", 25, "n"),
    ("验收", 45, "v"),
    ("它们", 50, "n"),
    ("野外", 45, "n"),
    ("快速", 55, "d"),
    ("奔跑", 40, "v"),
    ("村民", 50, "n"),
    ("们", 45, "d"),
    ("沿着", 35, "v"),
    ("河岸", 40, "n"),
    ("散步", 42, "v"),
    ("机器", 60, "n"),
    ("算法", 58, "n"),
    ("依赖", 38, "v"),
    ("高质量", 30, "a"),
    ("数据", 72, "n"),
    ("黑", 33, "a"),
    ("猫", 48, "n"),
    ("蹲", 28, "v"),
    ("窗台", 36, "n"),
    ("晒", 30, "v"),
    ("太阳", 66, "n"),
    ("老师", 78, "n"),
    ("鼓励", 44, "v"),
    ("学生", 82, "n"),
    ("主动", 40, "d"),
    ("提问", 36, "v"),
    ("这家", 32, "d"),
    ("初创", 26, "v"),
    ("公司", 76, "n"),
    ("发布", 52, "v"),
    ("了", 90, "d"),
    ("产品", 68, "n"),
    ("雨后", 28, "n"),
    ("的", 95, "d"),
    ("空气", 57, "n"),
    ("格外", 34, "d"),
    ("清新", 39, "a"),
]

CHAR_SET = sorted({char for word, _, _ in DICT_ENTRIES for char in word})


def seeded(index: int, salt: int) -> float:
    value = math.sin(index * 12.9898 + salt * 78.233) * 43758.5453
    return value - math.floor(value)


def log_distribution(weights: list[float]) -> list[float]:
    total = sum(weights)
    return [math.log(weight / total) for weight in weights]


def build_prob_table() -> dict:
    start_weights = [10.0 + seeded(index, 1) * 5 for index in range(len(TAGS))]
    start_logprob = dict(zip(TAGS, log_distribution(start_weights)))

    trans_logprob = {}
    for head_index, head in enumerate(TAGS):
        weights = [5.0 + seeded(head_index * 7 + tail_index, 2) * 8
                   for tail_index in range(len(TAGS))]
        trans_logprob[head] = dict(zip(TAGS, log_distribution(weights)))

    unique_words = sorted({word for word, _freq, _pos in DICT_ENTRIES})
    word_pos = {word: pos for word, _freq, pos in DICT_ENTRIES}
    emit_logprob: dict[str, dict[str, float]] = {}
    for word_index, word in enumerate(unique_words):
        gold_pos = word_pos[word]
        weights = []
        for tag_index, tag in enumerate(TAGS):
            base = 2.0 + seeded(word_index, 10 + tag_index) * 6
            weights.append(base * (6.0 if tag == gold_pos else 1.0))
        total = sum(weights)
        emit_logprob[word] = {
            tag: math.log(weight / total)
            for tag, weight in zip(TAGS, weights)
        }

    char_emit: dict[str, dict[str, float]] = {}
    for char_index, char in enumerate(CHAR_SET):
        weights = [
            1.0 + seeded(char_index, 30 + tag_index) * 4
            for tag_index in range(len(TAGS))
        ]
        total = sum(weights)
        char_emit[char] = {
            tag: math.log(weight / total)
            for tag, weight in zip(TAGS, weights)
        }

    return {
        "tags": TAGS,
        "start_logprob": start_logprob,
        "trans_logprob": trans_logprob,
        "emit_logprob": emit_logprob,
        "char_emit_logprob": char_emit,
    }


def main(output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = [f"{sid}\t{text}" for sid, text in SENTENCES]
    lines.extend(INVALID_LINES)
    (out / "sentences.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    seen = set()
    dict_lines = []
    for word, freq, pos in DICT_ENTRIES:
        if word in seen:
            continue
        seen.add(word)
        dict_lines.append(f"{word} {freq} {pos}")
    (out / "dictionary.txt").write_text("\n".join(dict_lines) + "\n", encoding="utf-8")

    (out / "prob_table.json").write_text(
        json.dumps(build_prob_table(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    config = {
        "workers": 4,
        "dict_change_mode": "entry",
        "snapshot_commit_interval": 4,
        "low_confidence_threshold": 0.35,
        "smoothing": -20.0,
    }
    (out / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"sample data written to {out}")


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "examples/sample_data"
    main(target)
