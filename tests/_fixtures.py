"""测试共用的小型确定性夹具。"""

from __future__ import annotations

import json
import math
from pathlib import Path

from segpos import ComponentConfig, Dictionary, ProbabilityTable

TAGS = ["n", "v", "a", "d"]

SENTENCES = {
    "S1": "研究生物很有意思",
    "S2": "他从小学电脑",
    "S3": "机器学习算法依赖数据",
    "S4": "乒乓球拍卖完了",
    "S5": "黑猫在窗台晒太阳",
}

DICT_LINES = [
    "研究 100 v",
    "研究生 80 n",
    "生物 90 n",
    "很 60 d",
    "有意思 50 a",
    "他 40 n",
    "从小 30 d",
    "小学 70 n",
    "电脑 80 n",
    "机器 50 n",
    "学习 70 v",
    "算法 55 n",
    "依赖 35 v",
    "数据 60 n",
    "乒乓球 70 n",
    "乒乓球拍 30 n",
    "拍卖 45 v",
    "卖完 25 v",
    "了 80 d",
    "黑 25 a",
    "猫 40 n",
    "在 50 d",
    "窗台 30 n",
    "晒 25 v",
    "太阳 55 n",
    "训练 20 v",
    "模型 45 n",
]


def _weights(seed: int, size: int):
    out = []
    for index in range(size):
        value = math.sin(seed * 13.1 + index * 7.7) * 43758.5
        out.append(1.0 + (value - math.floor(value)) * 5)
    return out


def build_table_data() -> dict:
    start_w = _weights(1, len(TAGS))
    start = dict(zip(TAGS, [math.log(w / sum(start_w)) for w in start_w]))
    trans = {}
    for head_index, head in enumerate(TAGS):
        weights = _weights(head_index + 2, len(TAGS))
        trans[head] = dict(
            zip(TAGS, [math.log(w / sum(weights)) for w in weights])
        )
    word_pos = {}
    for line in DICT_LINES:
        word, _freq, pos = line.split()
        word_pos[word] = pos
    emit = {}
    for index, (word, pos) in enumerate(sorted(word_pos.items())):
        weights = _weights(index + 10, len(TAGS))
        weights = [w * (5.0 if tag == pos else 1.0) for w, tag in zip(weights, TAGS)]
        total = sum(weights)
        emit[word] = {tag: math.log(w / total) for tag, w in zip(TAGS, weights)}
    chars = sorted({ch for word in word_pos for ch in word})
    char_emit = {}
    for index, char in enumerate(chars):
        weights = _weights(index + 40, len(TAGS))
        total = sum(weights)
        char_emit[char] = {
            tag: math.log(w / total) for tag, w in zip(TAGS, weights)
        }
    return {
        "tags": TAGS,
        "start_logprob": start,
        "trans_logprob": trans,
        "emit_logprob": emit,
        "char_emit_logprob": char_emit,
    }


def write_workspace(root: Path, workers: int = 1, interval: int = 2) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    sent_path = root / "sentences.txt"
    sent_path.write_text(
        "\n".join(f"{sid}\t{text}" for sid, text in sorted(SENTENCES.items()))
        + "\n",
        encoding="utf-8",
    )
    dict_path = root / "dictionary.txt"
    dict_path.write_text("\n".join(DICT_LINES) + "\n", encoding="utf-8")
    table_path = root / "table.json"
    table_path.write_text(
        json.dumps(build_table_data(), ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    config_path = root / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "workers": workers,
                "dict_change_mode": "entry",
                "snapshot_commit_interval": interval,
                "low_confidence_threshold": 0.35,
                "smoothing": -20.0,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "sentences": sent_path,
        "dictionary": dict_path,
        "table": table_path,
        "config": config_path,
    }


def make_pipeline(paths: dict, snapshot_dir: Path, table_version: int = 1):
    from segpos import IncrementalPipeline

    config = ComponentConfig.load(paths["config"])
    dictionary = Dictionary.load(paths["dictionary"], allowed_pos=set(TAGS))
    table = ProbabilityTable.load(paths["table"], version=table_version)
    return IncrementalPipeline(
        config, dictionary, table, snapshot_dir, paths["sentences"]
    )
