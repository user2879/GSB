"""关系模板触发、定点关系打分与落空处理。

触发词命中集合由关系模板的最长词项长度（取配置 max_term_length）在
实体对区间内扫描得到；得分完全在定点整数域、按固定运算顺序合成。
得分恰等于量化阈值时判为保留，并标记 boundary=true。
"""

from dataclasses import dataclass

import numpy as np

from .features import feature_points_for_vector, pair_feature_vector, term_bucket
from .fixedpoint import half_up_divide, quantize, saturate
from .fingerprint import content_fingerprint

MISS_NO_COMPATIBLE_TEMPLATE = "MISS_NO_COMPATIBLE_TEMPLATE"
MISS_NO_TRIGGER = "MISS_NO_TRIGGER"
MISS_BELOW_THRESHOLD = "MISS_BELOW_THRESHOLD"
MISS_NO_ENTITY = "MISS_NO_ENTITY"


@dataclass(frozen=True)
class TriggerHit:
    template_id: str
    relation_type: str
    term: str
    position: int

    def to_dict(self):
        return {
            "template_id": self.template_id,
            "relation_type": self.relation_type,
            "term": self.term,
            "position": self.position,
        }


@dataclass(frozen=True)
class ScoredRelation:
    head_id: str
    tail_id: str
    relation_type: str
    template_id: str
    trigger_hits: tuple
    score: int
    threshold_q: int
    boundary: bool
    distance: int

    def to_dict(self):
        return {
            "head_id": self.head_id,
            "tail_id": self.tail_id,
            "relation_type": self.relation_type,
            "template_id": self.template_id,
            "trigger_hits": [hit.to_dict() for hit in self.trigger_hits],
            "score": self.score,
            "threshold_q": self.threshold_q,
            "boundary": self.boundary,
            "distance": self.distance,
        }


@dataclass(frozen=True)
class MissEntry:
    head_id: str
    tail_id: str
    relation_type: str
    template_id: str
    reason_code: str
    dependent_templates: tuple
    head_start: int
    tail_start: int

    def to_dict(self):
        return {
            "head_id": self.head_id,
            "tail_id": self.tail_id,
            "relation_type": self.relation_type,
            "template_id": self.template_id,
            "reason_code": self.reason_code,
            "dependent_templates": list(self.dependent_templates),
            "head_start": self.head_start,
            "tail_start": self.tail_start,
        }


def scan_triggers(tokens, start, end, template, max_term_length):
    hits = []
    upper = min(end, len(tokens))
    for position in range(start, upper):
        for length in range(1, min(max_term_length, upper - position) + 1):
            span = " ".join(tokens[position:position + length])
            if span in template.trigger_terms:
                hits.append(TriggerHit(template.template_id,
                                       template.relation_type, span, position))
    hits.sort(key=lambda hit: (hit.position, hit.term, hit.template_id))
    return tuple(hits)


def _compatible_templates(head, tail, model, config):
    return tuple(
        tpl for tpl in model.templates
        if tpl.head_type == head.entity_type and tpl.tail_type == tail.entity_type
        and tpl.relation_type in config.relation_types
    )


def score_relation(head, tail, tokens, template, hits, vector, model, config):
    width, quant = config.fixedpoint_width, config.quant_level
    weight_q = quantize(template.weight, width, quant)
    threshold_q = quantize(config.score_threshold, width, quant)

    bucket_weight = np.zeros(model.hash_buckets, dtype=np.int64)
    for term in template.trigger_terms:
        bucket_weight[term_bucket(term, model.hash_buckets)] += weight_q
    aligned = bucket_weight[:model.feature_dim]
    feature_energy = int(np.dot(aligned, vector))

    hit_count = len(hits)
    distance = max(0, tail.start - head.end)
    distance_q = half_up_divide(distance * (10 ** quant), (distance + 1) * (10 ** 2))
    distance_q = saturate(distance_q, width)

    compatibility = (10 ** quant) if hits else quantize(0.35, width, quant)
    score = feature_energy + weight_q * hit_count + compatibility - distance_q
    score = saturate(score, width)
    return score, threshold_q, distance


def miss_sort_key(entry: MissEntry):
    return (entry.head_start, entry.tail_start, entry.head_id,
            entry.tail_id, entry.template_id)


def relation_task_points(entity_count, model, feature_dim):
    pairs = entity_count * max(0, entity_count - 1) // 2
    if entity_count == 0:
        return feature_points_for_vector(feature_dim)
    return feature_points_for_vector(feature_dim) + pairs * len(model.templates)


def extract_relations(entities, tokens, model, config):
    """返回触发命中、保留关系、落空条目与依赖模板集合。"""
    all_hits = []
    relations = []
    misses = []
    referenced_templates = set()

    ordered = sorted(entities, key=lambda item: (item.start, item.end,
                                                 item.entity_type, item.candidate_id))
    if not ordered:
        no_entity = (MissEntry(
            head_id="", tail_id="", relation_type="", template_id="",
            reason_code=MISS_NO_ENTITY, dependent_templates=tuple(),
            head_start=-1, tail_start=-1,
        ),)
        return tuple(), tuple(), no_entity, set(), content_fingerprint([])

    for head_index, head in enumerate(ordered):
        for tail in ordered[head_index + 1:]:
            compatible = _compatible_templates(head, tail, model, config)
            if not compatible:
                misses.append(MissEntry(
                    head_id=head.candidate_id, tail_id=tail.candidate_id,
                    relation_type="", template_id="",
                    reason_code=MISS_NO_COMPATIBLE_TEMPLATE,
                    dependent_templates=tuple(),
                    head_start=head.start, tail_start=tail.start,
                ))
                continue
            vector = pair_feature_vector(head, tail, tokens,
                                         model.feature_dim, model.hash_buckets)
            pair_best = None
            for template in compatible:
                referenced_templates.add(template.template_id)
                hits = scan_triggers(tokens, head.start, tail.end,
                                     template, config.max_term_length)
                all_hits.extend(hits)
                score, threshold_q, distance = score_relation(
                    head, tail, tokens, template, hits, vector, model, config
                )
                if not hits:
                    misses.append(MissEntry(
                        head_id=head.candidate_id, tail_id=tail.candidate_id,
                        relation_type=template.relation_type,
                        template_id=template.template_id,
                        reason_code=MISS_NO_TRIGGER,
                        dependent_templates=(template.template_id,),
                        head_start=head.start, tail_start=tail.start,
                    ))
                    continue
                if score < threshold_q:
                    misses.append(MissEntry(
                        head_id=head.candidate_id, tail_id=tail.candidate_id,
                        relation_type=template.relation_type,
                        template_id=template.template_id,
                        reason_code=MISS_BELOW_THRESHOLD,
                        dependent_templates=(template.template_id,),
                        head_start=head.start, tail_start=tail.start,
                    ))
                    continue
                candidate = ScoredRelation(
                    head_id=head.candidate_id, tail_id=tail.candidate_id,
                    relation_type=template.relation_type,
                    template_id=template.template_id,
                    trigger_hits=hits, score=score, threshold_q=threshold_q,
                    boundary=(score == threshold_q), distance=distance,
                )
                candidate_key = (candidate.score, candidate.template_id)
                if pair_best is None or candidate_key > (pair_best.score, pair_best.template_id):
                    pair_best = candidate
            if pair_best is not None:
                relations.append(pair_best)

    all_hits = tuple(sorted(all_hits, key=lambda hit: (hit.position, hit.term,
                                                       hit.template_id)))
    relations = tuple(sorted(relations, key=lambda rel: (rel.head_id, rel.tail_id,
                                                         rel.template_id)))
    misses = tuple(sorted(misses, key=miss_sort_key))
    trigger_fingerprint = content_fingerprint([hit.to_dict() for hit in all_hits])
    return all_hits, relations, tuple(misses), referenced_templates, trigger_fingerprint
