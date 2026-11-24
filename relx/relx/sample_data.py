"""固定种子样本生成器。

所有样本均由本模块以固定随机种子与受控词表生成，不访问网络或外部数据集。
生成内容包括：模型参数包、句级标注 JSONL（含异常行）、抽取配置 YAML、
编辑命令流与一份用于全量对照的最终态配置。
"""

import json
import os

from . import mini_yaml
from .fingerprint import fnv1a64

SEED = 20261124

VOCABULARY = [
    "张伟", "李娜", "王强", "赵敏",
    "开源", "科技", "研究院", "有限", "公司",
    "北京", "上海", "深圳", "杭州",
    "就职", "于", "总部", "位于", "合作", "拜访", "访问", "参观",
    "在", "和", "与", "的", "年度", "峰会", "期间", "季度", "会议",
    "联合", "实验室", "园区", "大厦", "客户",
]

MULTI_TERM_ENTITIES = {
    "开源 科技": "ORG",
    "科技 有限 公司": "ORG",
    "联合 实验室": "ORG",
    "北京 园区": "LOC",
    "上海 大厦": "LOC",
}

ENTITY_TYPE_LEXICON = {
    "PER": ["张伟", "李娜", "王强", "赵敏"],
    "ORG": ["开源 科技", "科技 有限 公司", "联合 实验室", "研究院"],
    "LOC": ["北京", "上海", "深圳", "杭州", "北京 园区", "上海 大厦"],
}

PER_TERMS = ["张伟", "李娜", "王强", "赵敏"]
LOC_TERMS = ["北京", "上海", "深圳", "杭州"]
ORG_SINGLE = ["研究院"]
TRIGGERS = {
    "work_for": ["就职", "于"],
    "located_in": ["位于", "总部 位于"],
    "cooperate_with": ["合作", "联合"],
    "visit": ["拜访", "访问", "参观"],
}

CONFIG_DICT = {
    "entity_types": ["PER", "ORG", "LOC"],
    "relation_types": ["work_for", "located_in", "cooperate_with", "visit"],
    "max_term_length": 3,
    "score_threshold": 1.572,
    "fixedpoint_width": 16,
    "quant_level": 4,
    "thread_count": 4,
    "feature_budget_per_batch": 120,
    "snapshot_commit_batches": 2,
}


def _term_hash():
    return {term: int(fnv1a64(term.encode("utf-8")), 16) % 256 for term in VOCABULARY}


def build_model_package():
    templates = [
        {
            "template_id": "T01",
            "relation_type": "work_for",
            "trigger_terms": ["就职", "于"],
            "head_type": "PER",
            "tail_type": "ORG",
            "weight": 0.82,
            "version": 1,
        },
        {
            "template_id": "T02",
            "relation_type": "located_in",
            "trigger_terms": ["位于", "总部 位于"],
            "head_type": "ORG",
            "tail_type": "LOC",
            "weight": 0.74,
            "version": 1,
        },
        {
            "template_id": "T03",
            "relation_type": "cooperate_with",
            "trigger_terms": ["合作", "联合"],
            "head_type": "ORG",
            "tail_type": "ORG",
            "weight": 0.66,
            "version": 1,
        },
        {
            "template_id": "T04",
            "relation_type": "visit",
            "trigger_terms": ["拜访", "访问", "参观"],
            "head_type": "PER",
            "tail_type": "LOC",
            "weight": 0.58,
            "version": 1,
        },
    ]
    return {
        "model_version": "relx-model-2026.11",
        "feature_dim": 32,
        "hash_buckets": 256,
        "term_hash": _term_hash(),
        "entity_lexicon": ENTITY_TYPE_LEXICON,
        "templates": templates,
    }


def _entity(eid, start, end, etype, tokens):
    return {
        "entity_id": eid,
        "start": start,
        "end": end,
        "type": etype,
        "surface": " ".join(tokens[start:end]),
    }


def build_sentences():
    rows = []

    def add(sid, tokens, entities, relations):
        rows.append({
            "sentence_id": sid,
            "tokens": tokens,
            "entities": entities,
            "relations": relations,
        })

    # 1-4 PER work_for ORG
    for idx, person in enumerate(PER_TERMS, start=1):
        tokens = [person, "就职", "于", "开源", "科技"]
        add(f"S{idx:02d}", tokens,
            [_entity("E1", 0, 1, "PER", tokens),
             _entity("E2", 3, 5, "ORG", tokens)],
            [{"head_id": "E1", "tail_id": "E2", "relation_type": "work_for"}])

    # 5-8 ORG located_in LOC，含多词触发词“总部 位于”
    loc_cycle = LOC_TERMS
    for idx, loc in enumerate(loc_cycle, start=5):
        tokens = ["科技", "有限", "公司", "总部", "位于", loc]
        add(f"S{idx:02d}", tokens,
            [_entity("E1", 0, 3, "ORG", tokens),
             _entity("E2", 5, 6, "LOC", tokens)],
            [{"head_id": "E1", "tail_id": "E2", "relation_type": "located_in"}])

    # 9-12 ORG cooperate_with ORG
    org_forms = [
        (["开源", "科技"], ["科技", "有限", "公司"]),
        (["研究院"], ["联合", "实验室"]),
        (["联合", "实验室"], ["开源", "科技"]),
        (["科技", "有限", "公司"], ["研究院"]),
    ]
    for offset, (org_a, org_b) in enumerate(org_forms, start=9):
        tokens = org_a + ["与"] + org_b + ["合作"]
        end_a = len(org_a)
        start_b = end_a + 1
        end_b = start_b + len(org_b)
        add(f"S{offset:02d}", tokens,
            [_entity("E1", 0, end_a, "ORG", tokens),
             _entity("E2", start_b, end_b, "ORG", tokens)],
            [{"head_id": "E1", "tail_id": "E2", "relation_type": "cooperate_with"}])

    # 13-16 PER visit LOC
    visit_terms = ["拜访", "访问", "参观", "拜访"]
    for idx, (person, trigger) in enumerate(zip(PER_TERMS, visit_terms), start=13):
        loc = LOC_TERMS[(idx - 13) % len(LOC_TERMS)]
        tokens = [person, "在", "峰会", "期间", trigger, loc]
        add(f"S{idx:02d}", tokens,
            [_entity("E1", 0, 1, "PER", tokens),
             _entity("E2", 5, 6, "LOC", tokens)],
            [{"head_id": "E1", "tail_id": "E2", "relation_type": "visit"}])

    # 17-18 边界重叠消解：嵌套 ORG 跨度
    tokens = ["开源", "科技", "有限", "公司", "位于", "北京"]
    add("S17", tokens,
        [_entity("E1", 0, 4, "ORG", tokens),
         _entity("E2", 5, 6, "LOC", tokens)],
        [{"head_id": "E1", "tail_id": "E2", "relation_type": "located_in"}])

    tokens = ["张伟", "参观", "上海", "大厦"]
    add("S18", tokens,
        [_entity("E1", 0, 1, "PER", tokens),
         _entity("E2", 2, 4, "LOC", tokens)],
        [{"head_id": "E1", "tail_id": "E2", "relation_type": "visit"}])

    # 19-20 长距离衰减句
    tokens = ["李娜", "在", "年度", "季度", "会议", "期间", "就职", "于", "研究院"]
    add("S19", tokens,
        [_entity("E1", 0, 1, "PER", tokens),
         _entity("E2", 8, 9, "ORG", tokens)],
        [{"head_id": "E1", "tail_id": "E2", "relation_type": "work_for"}])

    tokens = ["王强", "和", "赵敏", "在", "北京", "园区", "拜访", "客户"]
    add("S20", tokens,
        [_entity("E1", 0, 1, "PER", tokens),
         _entity("E2", 2, 3, "PER", tokens),
         _entity("E3", 4, 6, "LOC", tokens)],
        [{"head_id": "E1", "tail_id": "E3", "relation_type": "visit"},
         {"head_id": "E2", "tail_id": "E3", "relation_type": "visit"}])

    # 21-22 无触发词/类型不兼容落空句
    tokens = ["张伟", "和", "李娜", "在", "会议"]
    add("S21", tokens,
        [_entity("E1", 0, 1, "PER", tokens),
         _entity("E2", 2, 3, "PER", tokens)],
        [])

    tokens = ["研究院", "位于", "深圳"]
    add("S22", tokens,
        [_entity("E1", 0, 1, "ORG", tokens),
         _entity("E2", 2, 3, "LOC", tokens)],
        [{"head_id": "E1", "tail_id": "E2", "relation_type": "located_in"}])

    # 23-24 无实体句与纯触发词句
    add("S23", ["在", "会议", "期间", "合作"], [], [])
    add("S24", ["峰会", "期间", "参观", "与", "访问"], [], [])

    return rows


def build_invalid_lines():
    valid_tokens = ["张伟", "就职", "于", "开源", "科技"]
    return [
        '{"sentence_id": "S25", "tokens": ["张伟"], "entities": []',  # 坏 JSON
        json.dumps({  # 重复 ID
            "sentence_id": "S01",
            "tokens": valid_tokens,
            "entities": [],
            "relations": [],
        }, ensure_ascii=False),
        json.dumps({  # OOV
            "sentence_id": "S26",
            "tokens": ["未知词"],
            "entities": [],
            "relations": [],
        }, ensure_ascii=False),
        json.dumps({  # 实体结构非法
            "sentence_id": "S27",
            "tokens": valid_tokens,
            "entities": [{"entity_id": "E9", "start": 0, "end": 9, "type": "PER"}],
            "relations": [],
        }, ensure_ascii=False),
        json.dumps({  # 关系引用不存在实体
            "sentence_id": "S28",
            "tokens": valid_tokens,
            "entities": [_entity("E1", 0, 1, "PER", valid_tokens)],
            "relations": [{"head_id": "E1", "tail_id": "E404",
                           "relation_type": "work_for"}],
        }, ensure_ascii=False),
    ]


def build_commands():
    return {
        "commands": [
            {"seq": 1, "kind": "edit_tokens",
             "payload": {"sentence_id": "S21",
                         "tokens": ["张伟", "就职", "于", "研究院"]},
             "comment": "修改 S21 用词使其产生 work_for 关系"},
            {"seq": 2, "kind": "add_entity",
             "payload": {"sentence_id": "S23", "entity_id": "G1",
                         "start": 1, "end": 2, "type": "ORG"},
             "comment": "为无实体句新增 ORG 实体"},
            {"seq": 3, "kind": "remove_entity",
             "payload": {"sentence_id": "S20", "entity_id": "E2"},
             "comment": "删除 S20 的一个 PER 实体"},
            {"seq": 4, "kind": "set_template_weight",
             "payload": {"template_id": "T04", "weight": 0.31},
             "comment": "下调 visit 模板权重并推进版本号"},
            {"seq": 5, "kind": "set_threshold",
             "payload": {"threshold": 1.572},
             "comment": "阈值恰好等于最低关系得分，触发边界决断"},
            {"seq": 6, "kind": "set_quant",
             "payload": {"fixedpoint_width": 8, "quant_level": 2},
             "comment": "切换到 8 位 Q2 档位验证整数变化"},
            {"seq": 7, "kind": "set_quant",
             "payload": {"fixedpoint_width": 16, "quant_level": 4},
             "comment": "恢复默认量化档位"},
            {"seq": 8, "kind": "set_threshold",
             "payload": {"threshold": 1.572},
             "comment": "恢复默认边界阈值"},
            {"seq": 9, "kind": "set_template_weight",
             "payload": {"template_id": "T04", "weight": 0.58},
             "comment": "恢复 visit 模板权重"},
        ]
    }


def generate(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    model = build_model_package()
    with open(os.path.join(data_dir, "model_package.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(model, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    sentences = build_sentences()
    invalid = build_invalid_lines()
    with open(os.path.join(data_dir, "sentences.jsonl"), "w",
              encoding="utf-8", newline="\n") as handle:
        for row in sentences:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        for line in invalid:
            handle.write(line + "\n")

    with open(os.path.join(data_dir, "config.yaml"), "w",
              encoding="utf-8", newline="\n") as handle:
        handle.write(mini_yaml.dumps(CONFIG_DICT))

    with open(os.path.join(data_dir, "commands.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(build_commands(), handle, ensure_ascii=False, indent=2,
                  sort_keys=True)
        handle.write("\n")

    return {
        "sentence_count": len(sentences),
        "invalid_line_count": len(invalid),
        "vocabulary_size": len(VOCABULARY),
        "seed": SEED,
    }
