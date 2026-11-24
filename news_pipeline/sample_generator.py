"""Deterministic local sample generation from a fixed seed and controlled vocab."""

import json
import os
import random

SEED = 20261125

VOCAB = {
    "org": ["华新通讯社", "东方研究院", "北辰科技集团", "海湾港务公司"],
    "person": ["李明轩", "王雅婷", "赵启航", "陈思远"],
    "loc": ["上海", "深圳", "杭州", "成都"],
    "topic": ["人工智能", "量子计算", "新能源汽车", "跨境电商", "智能制造", "生物医药"],
    "action": ["发布", "签约", "投资", "研发", "投产", "验收"],
    "object": ["合作协议", "示范项目", "产业基金", "新一代平台", "区域中心"],
    "time": ["今日", "本周", "三季度", "年内"],
}

TEMPLATES = [
    "{time}{org}在{loc}{action}{topic}相关{object}",
    "{person}表示{topic}将推动{org}的{object}落地",
    "{loc}监管部门完成对{org}{topic}{object}的{action}流程",
    "{org}与{org}围绕{topic}{action}并推进{object}建设",
]


def _pick(rng, bucket):
    return bucket[rng.randrange(len(bucket))]


def _make_sentence(rng):
    template = TEMPLATES[rng.randrange(len(TEMPLATES))]
    values = {}
    for key in ("org", "person", "loc", "topic", "action", "object", "time"):
        values[key] = _pick(rng, VOCAB[key])
    return template.format(**values) + "。"


def _make_document(rng, doc_id, paragraph_count):
    paragraphs = []
    used_entities = []
    seen = set()
    for index in range(paragraph_count):
        sentence_count = rng.randint(2, 3)
        text = "".join(_make_sentence(rng) for _ in range(sentence_count))
        paragraphs.append({"pid": "p%d" % index, "text": text})
    flat_text = "".join(p["text"] for p in paragraphs)
    for bucket in ("org", "person", "loc", "topic"):
        for word in VOCAB[bucket]:
            if word in flat_text and word not in seen:
                used_entities.append({"text": word, "type": bucket})
                seen.add(word)
    events = []
    for _ in range(rng.randint(2, 4)):
        events.append(
            {
                "triple": [
                    _pick(rng, VOCAB["org"]),
                    _pick(rng, VOCAB["action"]),
                    _pick(rng, VOCAB["object"]),
                ]
            }
        )
    title = "%s%s观察" % (_pick(rng, VOCAB["loc"]), _pick(rng, VOCAB["topic"]))
    return {
        "doc_id": doc_id,
        "title": title,
        "paragraphs": paragraphs,
        "entities": used_entities,
        "events": events,
    }


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _config_yaml():
    return """# 摘要与重写配置（确定性样本）
summarization:
  max_sentences: 3
  term_weight: 0.4
  entity_weight: 0.4
  coverage_floor: 0.55
  position_decay: 0.2
  style_template_id: formal
  synonym_table_version: syn-v1
fixedpoint:
  decimal_width: 6
  quantization_tier: standard
runtime:
  workers: 3
  snapshot_interval_docs: 2
  sentence_budget_per_batch: 20
"""


def _rewrite_table_yaml():
    return """version: syn-v1
templates:
  - id: formal
    name: 正式通稿
  - id: brief
    name: 简讯
rules:
  - id: rule-ai
    template: formal
    source: 人工智能
    target: 人工智能技术
    enabled: true
  - id: rule-ai-short
    template: formal
    source: 人工
    target: 人力
    enabled: true
  - id: rule-qc
    template: formal
    source: 量子计算
    target: 量子计算研究
    enabled: true
  - id: rule-ev
    template: formal
    source: 新能源汽车
    target: 新能源汽车产业
    enabled: true
  - id: rule-org
    template: brief
    source: 华新通讯社
    target: 华新社
    enabled: true
  - id: rule-bad-template
    template: undeclared
    source: 智能制造
    target: 智能制造升级
    enabled: true
  - id: rule-empty-target
    template: formal
    source: 生物医药
    target: ""
    enabled: true
"""


def _edit_stream():
    commands = [
        {"command_id": "c01", "op": "update_paragraph", "doc_id": "doc-0002", "pid": "p0",
         "text": "今日华新通讯社在深圳发布人工智能合作协议。东方研究院推进新一代平台建设。"},
        {"command_id": "c02", "op": "insert_paragraph", "doc_id": "doc-0003", "pid": "p-new",
         "position": 1, "text": "本周北辰科技集团在杭州投产量子计算区域中心。"},
        {"command_id": "c03", "op": "delete_paragraph", "doc_id": "doc-0004", "pid": "p1"},
        {"command_id": "c04", "op": "set_max_sentences", "value": 2},
        {"command_id": "c05", "op": "set_coverage_floor", "value": 0.7},
        {"command_id": "c06", "op": "set_weights", "term_weight": 0.5, "entity_weight": 0.3},
        {"command_id": "c07", "op": "set_position_decay", "value": 0.35},
        {"command_id": "c08", "op": "set_style_template", "template_id": "brief"},
        {"command_id": "c09", "op": "add_rule", "rule": {"id": "rule-cross", "template": "brief",
         "source": "跨境电商", "target": "跨境电子商务", "enabled": True}},
        {"command_id": "c10", "op": "update_rule", "rule_id": "rule-ai", "target": "人工智能前沿技术"},
        {"command_id": "c11", "op": "delete_rule", "rule_id": "rule-qc"},
        {"command_id": "c12", "op": "set_quantization", "decimal_width": 8, "quantization_tier": "fine"},
        {"command_id": "c13", "op": "bump_table_version", "version": "syn-v2"},
        {"command_id": "c14", "op": "add_rule", "rule": {"id": "rule-bad-again", "template": "missing",
         "source": "杭州", "target": "杭州市", "enabled": True}},
    ]
    return commands


def generate_all(out_dir):
    rng = random.Random(SEED)
    os.makedirs(out_dir, exist_ok=True)
    documents = []
    for index in range(6):
        documents.append(_make_document(rng, "doc-%04d" % (index + 1), rng.randint(2, 3)))

    # Deterministic overlap fixture: "人工智能" (long key) and a standalone
    # "人工" occurrence coexist so suppression is observable end to end.
    overlap_doc = next(doc for doc in documents if doc["doc_id"] == "doc-0002")
    overlap_doc["paragraphs"].append(
        {"pid": "p-overlap", "text": "人工智能团队与人工审核流程今日在深圳同步运行。"}
    )
    for entity_text, entity_type in (("人工智能", "topic"), ("深圳", "loc")):
        if not any(entity["text"] == entity_text for entity in overlap_doc["entities"]):
            overlap_doc["entities"].append({"text": entity_text, "type": entity_type})
    from .hashing import fingerprint as _fingerprint
    overlap_doc["content_fingerprint"] = _fingerprint(
        {
            "doc_id": overlap_doc["doc_id"],
            "title": overlap_doc["title"],
            "paragraphs": overlap_doc["paragraphs"],
            "entities": overlap_doc["entities"],
            "events": overlap_doc["events"],
        }
    )

    invalid_documents = [
        {"doc_id": "doc-bad-1", "title": "缺少段落", "paragraphs": [],
         "entities": [{"text": "上海", "type": "loc"}],
         "events": [{"triple": ["华新通讯社", "发布", "合作协议"]}]},
        {"doc_id": "doc-bad-2", "title": "三元组损坏",
         "paragraphs": [{"pid": "p0", "text": "今日华新通讯社在深圳发布人工智能合作协议。"}],
         "entities": [{"text": "人工智能", "type": "topic"}],
         "events": [{"triple": ["华新通讯社", "发布"]}]},
    ]
    _write_jsonl(os.path.join(out_dir, "documents.jsonl"), documents + invalid_documents)

    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        handle.write(_config_yaml())
    with open(os.path.join(out_dir, "rewrite_table.yaml"), "w", encoding="utf-8") as handle:
        handle.write(_rewrite_table_yaml())
    _write_jsonl(os.path.join(out_dir, "edit_stream.jsonl"), _edit_stream())

    manifest = {
        "seed": SEED,
        "document_count": len(documents),
        "invalid_document_count": len(invalid_documents),
        "edit_count": len(_edit_stream()),
        "vocabulary_buckets": {key: list(values) for key, values in VOCAB.items()},
    }
    with open(os.path.join(out_dir, "sample_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
    return manifest


if __name__ == "__main__":
    import sys
    generate_all(sys.argv[1] if len(sys.argv) > 1 else "samples/data")
