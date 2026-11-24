"""模型参数包 JSON 的加载、校验与定点权重派生。"""

import json
from dataclasses import dataclass

from .errors import E_CONFIG_INVALID, E_MODEL_INVALID, RelxError


@dataclass(frozen=True)
class Template:
    template_id: str
    relation_type: str
    trigger_terms: tuple
    head_type: str
    tail_type: str
    weight: float
    version: int

    def fingerprint_material(self):
        return {
            "template_id": self.template_id,
            "relation_type": self.relation_type,
            "trigger_terms": list(self.trigger_terms),
            "head_type": self.head_type,
            "tail_type": self.tail_type,
            "weight": self.weight,
            "version": self.version,
        }


@dataclass(frozen=True)
class ModelPackage:
    model_version: str
    feature_dim: int
    hash_buckets: int
    vocabulary: tuple
    entity_lexicon: dict
    templates: tuple

    def fingerprint_material(self):
        return {
            "model_version": self.model_version,
            "feature_dim": self.feature_dim,
            "hash_buckets": self.hash_buckets,
            "vocabulary": list(self.vocabulary),
            "entity_lexicon": {key: list(value) for key, value in self.entity_lexicon.items()},
            "templates": [tpl.fingerprint_material() for tpl in self.templates],
        }


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise RelxError(E_MODEL_INVALID, f"模型参数包不存在: {path}")
    except json.JSONDecodeError as exc:
        raise RelxError(E_MODEL_INVALID, f"模型 JSON 解析失败: {exc}")


def load_model(path: str, config) -> ModelPackage:
    data = _read_json(path)
    for key in ("model_version", "feature_dim", "hash_buckets", "term_hash", "templates"):
        if key not in data:
            raise RelxError(E_MODEL_INVALID, f"模型参数包缺少字段: {key}")

    if not isinstance(data["feature_dim"], int) or data["feature_dim"] <= 0:
        raise RelxError(E_MODEL_INVALID, "feature_dim 必须为正整数")
    if not isinstance(data["hash_buckets"], int) or data["hash_buckets"] <= 0:
        raise RelxError(E_MODEL_INVALID, "hash_buckets 必须为正整数")

    term_hash = data["term_hash"]
    if not isinstance(term_hash, dict) or not term_hash:
        raise RelxError(E_MODEL_INVALID, "term_hash 必须为非空映射")
    vocabulary = tuple(sorted(term_hash.keys()))
    for term, bucket in term_hash.items():
        if not isinstance(term, str) or not isinstance(bucket, int) or bucket < 0:
            raise RelxError(E_MODEL_INVALID, "term_hash 键必须为字符串、值必须为非负整数")
        if bucket >= data["hash_buckets"]:
            raise RelxError(E_MODEL_INVALID, f"词项 {term} 的哈希桶越界")

    raw_templates = data["templates"]
    if not isinstance(raw_templates, list) or not raw_templates:
        raise RelxError(E_MODEL_INVALID, "templates 必须为非空列表")
    templates = []
    seen_ids = set()
    for raw in raw_templates:
        for key in ("template_id", "relation_type", "trigger_terms", "head_type",
                    "tail_type", "weight", "version"):
            if key not in raw:
                raise RelxError(E_MODEL_INVALID, f"模板缺少字段: {key}")
        tpl = Template(
            template_id=str(raw["template_id"]),
            relation_type=str(raw["relation_type"]),
            trigger_terms=tuple(raw["trigger_terms"]),
            head_type=str(raw["head_type"]),
            tail_type=str(raw["tail_type"]),
            weight=float(raw["weight"]),
            version=int(raw["version"]),
        )
        if tpl.template_id in seen_ids:
            raise RelxError(E_MODEL_INVALID, f"模板 ID 重复: {tpl.template_id}")
        seen_ids.add(tpl.template_id)
        if tpl.relation_type not in config.relation_types:
            raise RelxError(E_CONFIG_INVALID,
                            f"模板 {tpl.template_id} 的关系类型 {tpl.relation_type} 不在配置集合中")
        if tpl.head_type not in config.entity_types or tpl.tail_type not in config.entity_types:
            raise RelxError(E_CONFIG_INVALID,
                            f"模板 {tpl.template_id} 引用了未配置的实体类型")
        if not tpl.trigger_terms or any(not isinstance(term, str) for term in tpl.trigger_terms):
            raise RelxError(E_MODEL_INVALID, f"模板 {tpl.template_id} 触发词非法")
        if tpl.version < 0:
            raise RelxError(E_MODEL_INVALID, f"模板 {tpl.template_id} 版本号非法")
        templates.append(tpl)

    template_types = {tpl.relation_type for tpl in templates}
    if set(config.relation_types) != template_types:
        raise RelxError(
            E_CONFIG_INVALID,
            f"关系类型与模板权重必须一一对应，缺失: {set(config.relation_types) - template_types}，"
            f"多余: {template_types - set(config.relation_types)}",
        )

    lexicon = data.get("entity_lexicon", {})
    if not isinstance(lexicon, dict):
        raise RelxError(E_MODEL_INVALID, "entity_lexicon 必须为映射")
    for etype, terms in lexicon.items():
        if etype not in config.entity_types:
            raise RelxError(E_CONFIG_INVALID, f"entity_lexicon 含未配置类型: {etype}")
        if not isinstance(terms, list) or any(not isinstance(term, str) for term in terms):
            raise RelxError(E_MODEL_INVALID, f"entity_lexicon.{etype} 必须为字符串列表")
        for term in terms:
            for token in term.split(" "):
                if token not in term_hash:
                    raise RelxError(E_MODEL_INVALID,
                                    f"实体词表项 {term} 的词元 {token} 不在 term_hash 中")

    templates = tuple(sorted(templates, key=lambda tpl: tpl.template_id))
    return ModelPackage(
        model_version=str(data["model_version"]),
        feature_dim=data["feature_dim"],
        hash_buckets=data["hash_buckets"],
        vocabulary=vocabulary,
        entity_lexicon={etype: tuple(terms) for etype, terms in lexicon.items()},
        templates=templates,
    )
