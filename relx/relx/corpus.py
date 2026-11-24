"""句级标注 JSONL 的逐行加载与校验。

每行一个 JSON 对象，包含 sentence_id、tokens、entities、relations。
单句错误转换为 FailureRecord 进入失败清单并跳过该句，不中断其余句子。
"""

import json
from dataclasses import dataclass, field

from .errors import (
    E_ENTITY_REF_INVALID,
    E_ENTITY_STRUCTURE_INVALID,
    E_JSONL_INVALID,
    E_RELATION_REF_INVALID,
    E_SENTENCE_DUPLICATE,
    E_TOKEN_OOV,
    FailureRecord,
)


@dataclass(frozen=True)
class EntityMention:
    entity_id: str
    start: int
    end: int
    entity_type: str
    surface: str

    def to_dict(self):
        return {
            "entity_id": self.entity_id,
            "start": self.start,
            "end": self.end,
            "type": self.entity_type,
            "surface": self.surface,
        }


@dataclass(frozen=True)
class RelationLabel:
    head_id: str
    tail_id: str
    relation_type: str

    def to_dict(self):
        return {
            "head_id": self.head_id,
            "tail_id": self.tail_id,
            "relation_type": self.relation_type,
        }


@dataclass
class SentenceRecord:
    sentence_id: str
    tokens: list
    entities: list = field(default_factory=list)
    relations: list = field(default_factory=list)

    def to_dict(self):
        return {
            "sentence_id": self.sentence_id,
            "tokens": list(self.tokens),
            "entities": [mention.to_dict() for mention in self.entities],
            "relations": [label.to_dict() for label in self.relations],
        }


def _validate_sentence(raw, lineno, seen_ids, vocabulary, config):
    if not isinstance(raw, dict):
        return None, FailureRecord(E_JSONL_INVALID, "", "load",
                                   f"第 {lineno} 行不是 JSON 对象")
    sentence_id = raw.get("sentence_id")
    sid = str(sentence_id) if sentence_id is not None else ""

    def fail(code, message):
        return None, FailureRecord(code, sid or f"line:{lineno}", "load", message)

    if not sentence_id or not isinstance(sentence_id, str):
        return fail(E_JSONL_INVALID, "sentence_id 缺失或非字符串")
    if sid in seen_ids:
        return fail(E_SENTENCE_DUPLICATE, f"句子 ID 重复: {sid}")
    tokens = raw.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens):
        return fail(E_JSONL_INVALID, "tokens 必须为非空字符串列表")
    vocab_set = set(vocabulary)
    oov = sorted({token for token in tokens if token not in vocab_set})
    if oov:
        return fail(E_TOKEN_OOV, f"词项超出受控词表: {oov}")

    raw_entities = raw.get("entities")
    if not isinstance(raw_entities, list):
        return fail(E_ENTITY_STRUCTURE_INVALID, "entities 必须为列表")
    entities = []
    entity_ids = set()
    for raw_ent in raw_entities:
        if not isinstance(raw_ent, dict):
            return fail(E_ENTITY_STRUCTURE_INVALID, "实体元素必须为对象")
        ent_id = raw_ent.get("entity_id")
        start = raw_ent.get("start")
        end = raw_ent.get("end")
        ent_type = raw_ent.get("type")
        if not isinstance(ent_id, str) or not ent_id:
            return fail(E_ENTITY_STRUCTURE_INVALID, "实体缺少合法 entity_id")
        if ent_id in entity_ids:
            return fail(E_ENTITY_STRUCTURE_INVALID, f"实体 ID 重复: {ent_id}")
        if not (isinstance(start, int) and isinstance(end, int)
                and 0 <= start < end <= len(tokens)):
            return fail(E_ENTITY_STRUCTURE_INVALID,
                        f"实体 {ent_id} 边界非法: start={start}, end={end}")
        if ent_type not in config.entity_types:
            return fail(E_ENTITY_STRUCTURE_INVALID,
                        f"实体 {ent_id} 类型 {ent_type} 不在配置集合内")
        surface = raw_ent.get("surface", " ".join(tokens[start:end]))
        if not isinstance(surface, str):
            return fail(E_ENTITY_STRUCTURE_INVALID, f"实体 {ent_id} surface 非字符串")
        entities.append(EntityMention(ent_id, start, end, ent_type, surface))
        entity_ids.add(ent_id)

    raw_relations = raw.get("relations")
    if not isinstance(raw_relations, list):
        return fail(E_RELATION_REF_INVALID, "relations 必须为列表")
    relations = []
    for raw_rel in raw_relations:
        if not isinstance(raw_rel, dict):
            return fail(E_RELATION_REF_INVALID, "关系元素必须为对象")
        head_id = raw_rel.get("head_id")
        tail_id = raw_rel.get("tail_id")
        rel_type = raw_rel.get("relation_type")
        if head_id not in entity_ids or tail_id not in entity_ids:
            return fail(E_ENTITY_REF_INVALID,
                        f"关系引用的实体对 ({head_id}, {tail_id}) 不在实体列表中")
        if head_id == tail_id:
            return fail(E_ENTITY_REF_INVALID, "关系首尾实体不能相同")
        if rel_type not in config.relation_types:
            return fail(E_RELATION_REF_INVALID, f"关系类型 {rel_type} 不在配置集合内")
        relations.append(RelationLabel(head_id, tail_id, rel_type))

    seen_ids.add(sid)
    return SentenceRecord(sid, list(tokens), entities, relations), None


def load_corpus(path: str, vocabulary, config):
    records = []
    failures = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(FailureRecord(E_JSONL_INVALID, f"line:{lineno}",
                                              "load", f"JSON 解析失败: {exc}"))
                continue
            record, failure = _validate_sentence(raw, lineno, seen_ids, vocabulary, config)
            if failure is not None:
                failures.append(failure)
            else:
                records.append(record)
    records.sort(key=lambda item: item.sentence_id)
    failures.sort(key=FailureRecord.sort_key)
    return records, failures
