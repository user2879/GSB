"""固定种子生成确定性合成样本与全部本地资源。

不访问任何网络、站点或外部模型；全部文本来自受控字表与混淆字对。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

SEED = 20261118
ROOT = Path(__file__).resolve().parents[1]

# 受控字表（生成文本只允许使用这些字）
CHAR_TABLE = list("的一是不了人在我有他这中大来上个国和地到说为子道再入她个条")

# 混淆字对（错别字）：wrong -> correct
CONFUSION = [
    {"rule_id": "W001", "wrong": "在", "correct": "再"},
    {"rule_id": "W002", "wrong": "人", "correct": "入"},
    {"rule_id": "W003", "wrong": "他", "correct": "她"},
    {"rule_id": "W004", "wrong": "到", "correct": "道"},
    {"rule_id": "W005", "wrong": "大", "correct": "中"},
]
# 注：correct 侧也必须在受控字表内
for pair in CONFUSION:
    assert pair["wrong"] in CHAR_TABLE and pair["correct"] in CHAR_TABLE, pair

REPEAT_RULES = [
    {"rule_id": "R001", "pattern": "一一", "replacement": "一", "edit_type": "replace"},
    {"rule_id": "R002", "pattern": "个个", "replacement": "个", "edit_type": "replace"},
    {"rule_id": "R003", "pattern": "了了", "replacement": "了", "edit_type": "replace"},
    {"rule_id": "R004", "pattern": "在在", "replacement": "在", "edit_type": "replace"},
]

TERM_RULES = [
    {"rule_id": "T001", "pattern": "中国国人", "replacement": "中国人"},
    {"rule_id": "T002", "pattern": "大国", "replacement": "中国"},
]

OMISSION_RULES = [
    {"rule_id": "RULE_OMI_001", "pattern": "条国道", "replacement": "大", "insert_offset": 1},
    {"rule_id": "RULE_OMI_002", "pattern": "条人道", "replacement": "大", "insert_offset": 1},
]

RULE_WEIGHTS = {
    "W001": 1.05, "W002": 1.05, "W003": 1.10, "W004": 1.05, "W005": 1.00,
    "R001": 1.20, "R002": 1.20, "R003": 0.78, "R004": 1.20,
    "T001": 1.20, "T002": 1.15,
    "RULE_OMI_001": 1.20, "RULE_OMI_002": 1.20,
}

# 手工设计的确定性句子模板：原文含已知错误，truth 为期望修正文本
SAMPLE_SPECS = [
    # 设计原则：非错误位置不出现任何规则 pattern，规则命中即确定性真错误
    ("S001", "他在说", "她再说", ["W001", "W003"], "访谈记录"),
    ("S002", "人到了", "入道了", ["W002", "W004"], "通知正文"),
    ("S003", "他到了", "她道了", ["W003", "W004"], "访谈记录"),
    ("S004", "一一来", "一来", ["R001"], "公告"),
    ("S005", "这是个个", "这是个", ["R002"], "公告"),
    ("S006", "好了了吗", "好了了吗", ["R003"], "对话-边界样本"),
    ("S007", "中国国人来了", "中国人来了", ["T001"], "新闻稿"),
    ("S008", "大国这在", "中国这再", ["W001", "W005"], "新闻稿"),
    ("S009", "这条国道", "这条大国道", ["RULE_OMI_001"], "路标说明"),
    ("S010", "这条人道口", "这条大人道口", ["RULE_OMI_002"], "建筑说明"),
    # S011 重叠消解：R004(在在,0-2) 吞并 W001(在,0-1)/(在,1-2)
    ("S011", "在在说", "在说", ["R004", "W001"], "对话"),
    ("S012", "在在在", "在再", ["W001", "R004"], "对话"),
    ("S013", "大国条国道", "中国条大国道", ["W005", "RULE_OMI_001"], "路标说明"),
    ("S014", "个个是一", "个是一", ["R002"], "公告"),
    ("S015", "他的了", "她的了", ["W003"], "访谈记录"),
    ("S016", "人到条人道到", "入道条大人道道", ["W002", "W004", "RULE_OMI_002"], "建筑说明"),
    ("S017", "中国国人一一", "中国人一", ["T001", "R001"], "新闻稿"),
    ("S018", "到了到", "道了道", ["W004"], "通知正文"),
    ("S019", "他一一来", "她一来", ["W003", "R001"], "访谈记录"),
    ("S020", "大国这到了", "中国这道了", ["W004", "W005"], "新闻稿"),
    ("S021", "这个个条国道", "这个条大国道", ["R002", "RULE_OMI_001"], "路标说明"),
    ("S022", "了了这是", "了了这是", ["R003"], "对话-边界样本"),
]


def build_resources() -> dict:
    return {
        "char_table": CHAR_TABLE,
        "confusion_pairs": CONFUSION,
        "repeat_rules": REPEAT_RULES,
        "term_rules": TERM_RULES,
        "omission_rules": OMISSION_RULES,
        "rule_weights": RULE_WEIGHTS,
    }


def build_scoring() -> dict:
    return {
        "detect_weights": {
            "wrong_char": 1.20,
            "repeat": 1.00,
            "term": 1.10,
            "omission": 1.10,
        },
        "base_scores": {
            "wrong_char": 1.00,
            "repeat": 0.95,
            "term": 1.00,
            "omission": 0.98,
        },
        "decision_threshold": 0.2876953125,
    }


def build_config() -> dict:
    return {
        "ngram_order": 4,
        "similarity_threshold": 0.22,
        "fixed_point_bits": 8,
        "quant_levels": 32,
        "aggregation_weights": {"detect": 0.55, "confidence": 0.45},
        "precision": "float32",
        "tolerance": 1e-6,
        "max_workers": 4,
        "snapshot_interval": 5,
        "decision_threshold": 0.2876953125,
    }


def main() -> None:
    rng = random.Random(SEED)
    samples_dir = ROOT / "samples"
    resources_dir = ROOT / "resources"
    config_dir = ROOT / "config"
    for directory in (samples_dir, resources_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    for sample_id, text, truth, hints, context in SAMPLE_SPECS:
        records.append({
            "sample_id": sample_id,
            "text": text,
            "context": context,
            "truth": truth,
            "rule_hints": hints,
        })

    # 追加 4 条随机但确定性的组合样本，验证固定种子可重复
    rule_pool = (
        [(p["rule_id"], p["wrong"]) for p in CONFUSION]
        + [(r["rule_id"], r["pattern"]) for r in REPEAT_RULES[:3]]
    )
    for idx in range(4):
        sample_id = f"S{23 + idx:03d}"
        picks = rng.sample(rule_pool, 2)
        text = "".join(piece for _, piece in picks) + "来"
        records.append({
            "sample_id": sample_id,
            "text": text,
            "context": "随机组合样本",
            "truth": None,
            "rule_hints": [rule_id for rule_id, _ in picks],
        })

    jsonl_path = samples_dir / "sentences.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False,
                                    sort_keys=True) + "\n")

    (resources_dir / "rules.json").write_text(
        json.dumps(build_resources(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    (resources_dir / "scoring.json").write_text(
        json.dumps(build_scoring(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    with (config_dir / "proofread.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(build_config(), handle, allow_unicode=True, sort_keys=True)

    # 异常注入样本：覆盖 E_PARSE_INVALID / E_EMPTY_TEXT / E_RULE_UNKNOWN
    with (samples_dir / "sentences_invalid.jsonl").open("w", encoding="utf-8") as handle:
        handle.write('{"sample_id": "B001", "text": "正常一句", "truth": null}\n')
        handle.write('{"sample_id": "B002", "text": "   ", "truth": null}\n')
        handle.write('{"sample_id": "B003", broken json line\n')
        handle.write(json.dumps({
            "sample_id": "B004", "text": "他在说",
            "rule_hints": ["W999_NOT_EXIST"]}, ensure_ascii=False) + "\n")
        handle.write(json.dumps({
            "sample_id": "B001", "text": "重复ID样本"}, ensure_ascii=False) + "\n")

    print(f"已生成 {len(records)} 条正常样本 -> {jsonl_path}")
    print("异常注入样本 -> samples/sentences_invalid.jsonl")


if __name__ == "__main__":
    main()
