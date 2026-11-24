"""运行时可变状态：句子集合、模型模板版本与配置快照。

命令在状态上原地应用，非法命令返回 FailureRecord 而不抛出，保证命令流
中的坏命令进入失败清单后其余命令仍继续执行。
"""

from dataclasses import dataclass, field, replace

from .commands import Command
from .corpus import EntityMention, SentenceRecord
from .errors import (
    E_COMMAND_INVALID,
    E_ENTITY_REF_INVALID,
    E_ENTITY_STRUCTURE_INVALID,
    E_TOKEN_OOV,
    FailureRecord,
)
from .model import ModelPackage


@dataclass
class RuntimeState:
    sentences: dict
    model: ModelPackage
    config: object
    command_seq: int = 0
    edits_by_sentence: dict = field(default_factory=dict)
    last_edit_fingerprint: dict = field(default_factory=dict)

    def sentence_ids_sorted(self):
        return sorted(self.sentences.keys())

    def mark_sentence_edit(self, sentence_id, tag):
        self.edits_by_sentence.setdefault(sentence_id, []).append(
            (self.command_seq, tag)
        )


def _edit_tokens(state: RuntimeState, command: Command):
    payload = command.payload
    sentence_id = payload.get("sentence_id")
    if sentence_id not in state.sentences:
        return FailureRecord(E_COMMAND_INVALID, sentence_id, "command",
                             f"edit_tokens 目标句子不存在: {sentence_id}")
    vocab = set(state.model.vocabulary)
    oov = sorted({token for token in payload["tokens"] if token not in vocab})
    if oov:
        return FailureRecord(E_TOKEN_OOV, sentence_id, "command",
                             f"编辑词项超出受控词表: {oov}")
    record = state.sentences[sentence_id]
    new_tokens = list(payload["tokens"])
    kept_entities = [
        mention for mention in record.entities
        if mention.end <= len(new_tokens)
    ]
    dropped = [mention.entity_id for mention in record.entities
               if mention.end > len(new_tokens)]
    kept_ids = {mention.entity_id for mention in kept_entities}
    kept_relations = [
        label for label in record.relations
        if label.head_id in kept_ids and label.tail_id in kept_ids
    ]
    state.sentences[sentence_id] = SentenceRecord(
        sentence_id, new_tokens, kept_entities, kept_relations
    )
    state.mark_sentence_edit(sentence_id, "edit_tokens")
    if dropped:
        return FailureRecord(E_ENTITY_STRUCTURE_INVALID, sentence_id, "command",
                             f"新序列变短，丢弃越界实体: {dropped}")
    return None


def _add_entity(state: RuntimeState, command: Command):
    payload = command.payload
    sentence_id = payload.get("sentence_id")
    record = state.sentences.get(sentence_id)
    if record is None:
        return FailureRecord(E_COMMAND_INVALID, sentence_id, "command",
                             f"add_entity 目标句子不存在: {sentence_id}")
    ent_id = payload["entity_id"]
    if any(mention.entity_id == ent_id for mention in record.entities):
        return FailureRecord(E_ENTITY_STRUCTURE_INVALID, sentence_id, "command",
                             f"实体 ID 已存在: {ent_id}")
    start, end = payload["start"], payload["end"]
    if not (0 <= start < end <= len(record.tokens)):
        return FailureRecord(E_ENTITY_STRUCTURE_INVALID, sentence_id, "command",
                             f"实体边界越界: start={start}, end={end}")
    ent_type = payload["type"]
    if ent_type not in state.config.entity_types:
        return FailureRecord(E_ENTITY_STRUCTURE_INVALID, sentence_id, "command",
                             f"实体类型不在配置集合: {ent_type}")
    surface = payload.get("surface", " ".join(record.tokens[start:end]))
    record.entities.append(EntityMention(ent_id, start, end, ent_type, surface))
    record.entities.sort(key=lambda mention: (mention.start, mention.end,
                                              mention.entity_type, mention.entity_id))
    state.mark_sentence_edit(sentence_id, "add_entity")
    return None


def _remove_entity(state: RuntimeState, command: Command):
    payload = command.payload
    sentence_id = payload.get("sentence_id")
    record = state.sentences.get(sentence_id)
    if record is None:
        return FailureRecord(E_COMMAND_INVALID, sentence_id, "command",
                             f"remove_entity 目标句子不存在: {sentence_id}")
    ent_id = payload["entity_id"]
    if not any(mention.entity_id == ent_id for mention in record.entities):
        return FailureRecord(E_ENTITY_REF_INVALID, sentence_id, "command",
                             f"待删除实体不存在: {ent_id}")
    record.entities = [mention for mention in record.entities
                       if mention.entity_id != ent_id]
    record.relations = [
        label for label in record.relations
        if label.head_id != ent_id and label.tail_id != ent_id
    ]
    state.mark_sentence_edit(sentence_id, "remove_entity")
    return None


def _set_template_weight(state: RuntimeState, command: Command):
    payload = command.payload
    template_id = payload["template_id"]
    target = next((tpl for tpl in state.model.templates if tpl.template_id == template_id), None)
    if target is None:
        return FailureRecord(E_COMMAND_INVALID, "", "command",
                             f"模板不存在: {template_id}")
    new_weight = float(payload["weight"])
    new_templates = tuple(
        replace(tpl,
                weight=new_weight,
                version=tpl.version + (1 if tpl.weight != new_weight else 0))
        if tpl.template_id == template_id else tpl
        for tpl in state.model.templates
    )
    state.model = replace(state.model, templates=new_templates)
    return None


def _set_threshold(state: RuntimeState, command: Command):
    state.config = replace(state.config,
                           score_threshold=float(command.payload["threshold"]))
    return None


def _set_quant(state: RuntimeState, command: Command):
    state.config = replace(
        state.config,
        fixedpoint_width=command.payload["fixedpoint_width"],
        quant_level=command.payload["quant_level"],
    )
    return None


_HANDLERS = {
    "edit_tokens": _edit_tokens,
    "add_entity": _add_entity,
    "remove_entity": _remove_entity,
    "set_template_weight": _set_template_weight,
    "set_threshold": _set_threshold,
    "set_quant": _set_quant,
}


def apply_command(state: RuntimeState, command: Command):
    state.command_seq = command.seq
    return _HANDLERS[command.kind](state, command)
