"""阶段级内容指纹缓存。

四个阶段各有独立缓存域：entity、trigger、score、miss。缓存键由上游内容
指纹、参数指纹、模板版本号集合与量化档位派生，条目记录引用的模板与
实体候选集合，供增量失效判定与逐条审计。
"""

from dataclasses import dataclass, field

from .fingerprint import composite_fingerprint


STAGES = ("entity", "trigger", "score", "miss")


@dataclass
class CacheEntry:
    key: str
    stage: str
    sentence_id: str
    value: object
    referenced_templates: tuple = ()
    referenced_entities: tuple = ()
    template_versions: dict = field(default_factory=dict)
    quant_tag: str = ""

    def to_audit_dict(self):
        return {
            "key": self.key,
            "stage": self.stage,
            "sentence_id": self.sentence_id,
            "referenced_templates": list(self.referenced_templates),
            "referenced_entities": list(self.referenced_entities),
            "template_versions": dict(self.template_versions),
            "quant_tag": self.quant_tag,
        }


class StageCache:
    def __init__(self):
        self._stores = {stage: {} for stage in STAGES}

    def get(self, stage, sentence_id, key):
        entry = self._stores[stage].get(sentence_id)
        if entry is not None and entry.key == key:
            return entry
        return None

    def put(self, stage, sentence_id, key, value, referenced_templates=(),
            referenced_entities=(), template_versions=None, quant_tag=""):
        self._stores[stage][sentence_id] = CacheEntry(
            key=key,
            stage=stage,
            sentence_id=sentence_id,
            value=value,
            referenced_templates=tuple(referenced_templates),
            referenced_entities=tuple(referenced_entities),
            template_versions=dict(template_versions or {}),
            quant_tag=quant_tag,
        )

    def invalidate_sentence(self, sentence_id):
        removed = []
        for stage in STAGES:
            entry = self._stores[stage].pop(sentence_id, None)
            if entry is not None:
                removed.append((sentence_id, stage, entry.key))
        return removed

    def invalidate_template(self, template_id):
        removed = []
        for stage, store in self._stores.items():
            for sentence_id in list(store.keys()):
                entry = store[sentence_id]
                if template_id in entry.referenced_templates:
                    store.pop(sentence_id)
                    removed.append((sentence_id, stage, entry.key))
        return removed

    def invalidate_quant(self):
        removed = []
        for stage, store in self._stores.items():
            for sentence_id, entry in list(store.items()):
                store.pop(sentence_id)
                removed.append((sentence_id, stage, entry.key))
        return removed

    def invalidate_stage(self, stage, sentence_id):
        entry = self._stores[stage].pop(sentence_id, None)
        return None if entry is None else (sentence_id, stage, entry.key)

    def snapshot_prune(self, live_sentence_ids):
        live = set(live_sentence_ids)
        removed = []
        for stage, store in self._stores.items():
            for sentence_id in list(store.keys()):
                if sentence_id not in live:
                    entry = store.pop(sentence_id)
                    removed.append((sentence_id, stage, entry.key))
        return removed

    def audit_rows(self):
        rows = []
        for stage in STAGES:
            for sentence_id in sorted(self._stores[stage].keys()):
                rows.append(self._stores[stage][sentence_id].to_audit_dict())
        return rows


def make_stage_key(stage, *parts):
    return f"{stage}:{composite_fingerprint(stage, *parts)}"


def quant_tag(config):
    return f"w{config.fixedpoint_width}-q{config.quant_level}"
