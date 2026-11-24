"""实体候选抽取与边界重叠消解。

候选来源有两路：
1. 语料/命令维护的标注实体（带稳定 entity_id）；
2. 在词项序列上按受控词表与最长匹配长度生成的词典候选。
重叠消解按最长跨度优先、同长按起始位置升序、再按实体类型名升序；
标注实体与同跨度同类型的词典候选合并为同一实体。被抑制候选写入
抑制清单并记录吞并者候选 ID。
"""

from dataclasses import dataclass

from .fingerprint import content_fingerprint


@dataclass(frozen=True)
class EntityCandidate:
    candidate_id: str
    source: str
    start: int
    end: int
    entity_type: str
    surface: str
    gold_id: str

    def to_dict(self):
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "type": self.entity_type,
            "surface": self.surface,
            "gold_id": self.gold_id,
        }


@dataclass(frozen=True)
class SuppressedEntry:
    start: int
    end: int
    entity_type: str
    surface: str
    source: str
    swallowed_by: str

    def to_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "type": self.entity_type,
            "surface": self.surface,
            "source": self.source,
            "swallowed_by": self.swallowed_by,
        }


def _overlaps(left, right):
    return left.start < right.end and right.start < left.end


def _priority(candidate):
    return (-(candidate.end - candidate.start), candidate.start,
            candidate.entity_type, candidate.source, candidate.candidate_id)


def build_lexical_type_index(model):
    """实体类型词表 -> (词项元组 -> 类型集合)，候选仅按真实类型生成。"""
    index = {}
    for entity_type, terms in model.entity_lexicon.items():
        for term in terms:
            index.setdefault(tuple(term.split(" ")), set()).add(entity_type)
    return index


def generate_lexical_candidates(tokens, config, lexical_type_index):
    candidates = []
    max_len = min(config.max_term_length, len(tokens))
    for start in range(len(tokens)):
        for length in range(1, max_len + 1):
            end = start + length
            if end > len(tokens):
                continue
            span = tuple(tokens[start:end])
            type_set = lexical_type_index.get(span)
            if not type_set:
                continue
            for entity_type in sorted(type_set):
                if entity_type not in config.entity_types:
                    continue
                candidates.append(EntityCandidate(
                    candidate_id="",
                    source="lexical",
                    start=start,
                    end=end,
                    entity_type=entity_type,
                    surface=" ".join(span),
                    gold_id="",
                ))
    return candidates


def extract_entities(record, config, lexical_type_index):
    lexical = generate_lexical_candidates(record.tokens, config, lexical_type_index)
    gold = [
        EntityCandidate(
            candidate_id="",
            source="gold",
            start=mention.start,
            end=mention.end,
            entity_type=mention.entity_type,
            surface=mention.surface,
            gold_id=mention.entity_id,
        )
        for mention in record.entities
    ]
    pool = sorted(gold + lexical, key=_priority)

    accepted = []
    suppressed = []
    for candidate in pool:
        owner = next((kept for kept in accepted if _overlaps(kept, candidate)), None)
        if owner is not None:
            same_span = (owner.start == candidate.start and owner.end == candidate.end)
            merge = (same_span and candidate.source == "gold"
                     and owner.source == "lexical"
                     and owner.entity_type == candidate.entity_type)
            if not merge:
                suppressed.append((candidate, owner.candidate_id))
                continue
            accepted = [
                EntityCandidate(
                    candidate_id=candidate.gold_id,
                    source="gold",
                    start=candidate.start,
                    end=candidate.end,
                    entity_type=candidate.entity_type,
                    surface=candidate.surface,
                    gold_id=candidate.gold_id,
                )
                if kept is owner else kept
                for kept in accepted
            ]
            continue
        resolved_id = candidate.gold_id if candidate.source == "gold" else ""
        accepted.append(EntityCandidate(
            candidate_id=resolved_id,
            source=candidate.source,
            start=candidate.start,
            end=candidate.end,
            entity_type=candidate.entity_type,
            surface=candidate.surface,
            gold_id=candidate.gold_id,
        ))

    accepted.sort(key=lambda item: (item.start, item.end, item.entity_type,
                                    item.candidate_id))
    lex_seq = 0
    finalized = []
    id_map = {}
    for candidate in accepted:
        if candidate.source == "gold":
            new_id = candidate.gold_id
        else:
            lex_seq += 1
            new_id = f"C{lex_seq:02d}"
        id_map[(candidate.start, candidate.end, candidate.entity_type,
                candidate.source, candidate.gold_id)] = new_id
        finalized.append(EntityCandidate(
            candidate_id=new_id,
            source=candidate.source,
            start=candidate.start,
            end=candidate.end,
            entity_type=candidate.entity_type,
            surface=candidate.surface,
            gold_id=candidate.gold_id,
        ))

    owner_lookup = {}
    for candidate, owner_id in suppressed:
        for kept in finalized:
            if kept.candidate_id == owner_id:
                owner_lookup[owner_id] = kept.candidate_id
    suppressed_entries = [
        SuppressedEntry(
            start=candidate.start,
            end=candidate.end,
            entity_type=candidate.entity_type,
            surface=candidate.surface,
            source=candidate.source,
            swallowed_by=owner_lookup.get(owner_id, owner_id) or "SELF_SUPPRESSED",
        )
        for candidate, owner_id in suppressed
    ]
    suppressed_entries.sort(key=lambda item: (item.start, item.end,
                                              item.entity_type, item.source,
                                              item.swallowed_by))
    fingerprint = content_fingerprint([candidate.to_dict() for candidate in finalized])
    return finalized, suppressed_entries, fingerprint


def entity_task_points(record, config):
    """实体阶段特征点数：起始位置 × 探测跨度 × 实体类型，另加词项探测点。"""
    max_len = min(config.max_term_length, len(record.tokens))
    spans = sum(min(max_len, len(record.tokens) - start)
                for start in range(len(record.tokens)))
    return spans * len(config.entity_types) + len(record.tokens)
