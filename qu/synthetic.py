"""Deterministic synthetic data generation.

Everything is derived from a single :class:`random.Random` instance seeded by
``PipelineConfig.seed``. No external corpus or network access is involved.
Each scale tier gets its own JSONL files; tiers are nested supersets, so
incremental cache reuse across tiers is genuinely exercised.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


INTENTS = (
    ("phone_digital", ("手机", "耳机", "平板", "充电", "蓝牙", "智能手表",
                       "phone", "earphone", "tablet", "charger")),
    ("home_appliance", ("冰箱", "洗衣机", "空调", "烤箱", "吸尘器",
                        "fridge", "washer", "oven", "vacuum")),
    ("beauty_care", ("口红", "面膜", "香水", "护肤", "精华",
                     "lipstick", "perfume", "serum", "skincare")),
    ("food_grocery", ("零食", "牛奶", "坚果", "茶叶", "咖啡",
                      "snack", "milk", "coffee", "nut")),
    ("sports_outdoor", ("跑鞋", "瑜伽", "帐篷", "骑行", "哑铃",
                        "shoe", "yoga", "tent", "dumbbell")),
    ("book_media", ("图书", "小说", "文具", "钢笔", "book", "novel", "pen")),
)

ATTRIBUTE_WORDS = (
    "红色", "蓝色", "黑色", "白色", "大容量", "便携", "防水", "静音",
    "节能", "高清", "新款", "旗舰", "限量", "加厚", "red", "blue",
    "black", "white", "pro", "mini",
)

BRAND_CORE = (
    "星河", "蓝鲸", "北辰", "云杉", "朱雀", "玄铁", "麒麟", "凌霄",
    "aurora", "nebula", "vertex", "orbit",
)

# Deterministic malformed samples injected into every tier.
INVALID_LINES = (
    '{"query_id": "q-bad-json", "query": ',
    '{"query_id": "q-empty", "query": "   "}',
    '{"query_id": "q-punct", "query": "!!!？？？ ..."}',
)


@dataclass
class TierSpec:
    tier_id: str
    queries: int
    vocab: int
    entities: int
    rules: int


DEFAULT_TIERS = (
    TierSpec("tiny", 300, 200, 300, 24),
    TierSpec("small", 1_000, 1_500, 2_500, 60),
    TierSpec("medium", 5_000, 6_000, 10_000, 160),
    TierSpec("large", 20_000, 20_000, 30_000, 320),
)


def _write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False,
                                    sort_keys=True) + "\n")


def _write_raw_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


class SyntheticGenerator:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    # -- vocabularies --------------------------------------------------------
    def _extra_term(self, prefix: str, index: int) -> str:
        # Deterministic, collision-free filler terms.
        return f"{prefix}{index:05d}项"

    def build_vocab(self, size: int) -> List[dict]:
        rows: List[dict] = []
        core: List[Tuple[str, str]] = []
        for intent, words in INTENTS:
            for word in words:
                core.append((intent, word))
        for intent, word in core:
            rows.append({"intent": intent, "term": word,
                         "weight": round(self.rng.uniform(0.8, 2.4), 4)})
        # A few generic shared terms with lower weights.
        for word in ("正品", "包邮", "新款", "热卖", "official", "sale"):
            rows.append({"intent": INTENTS[0][0], "term": word,
                         "weight": round(self.rng.uniform(0.2, 0.8), 4)})
        index = 0
        while len(rows) < size:
            intent = INTENTS[index % len(INTENTS)][0]
            term = self._extra_term("词", index)
            rows.append({"intent": intent, "term": term,
                         "weight": round(self.rng.uniform(0.1, 1.5), 4)})
            index += 1
        return rows[:size]

    def build_entities(self, size: int) -> List[dict]:
        rows: List[dict] = []
        # Core brands/categories/attributes that queries actually reference.
        for index, word in enumerate(BRAND_CORE):
            rows.append({"entity_type": "brand", "term": word,
                         "entity_id": f"brand-core-{index:02d}",
                         "weight": round(self.rng.uniform(1.0, 2.0), 4)})
        for intent, words in INTENTS:
            for index, word in enumerate(words[:4]):
                rows.append({"entity_type": "category", "term": word,
                             "entity_id": f"cat-{intent}-{index:02d}",
                             "weight": round(self.rng.uniform(0.9, 1.8), 4)})
        for index, word in enumerate(ATTRIBUTE_WORDS):
            rows.append({"entity_type": "attribute", "term": word,
                         "entity_id": f"attr-{index:03d}",
                         "weight": round(self.rng.uniform(0.7, 1.6), 4)})
        index = 0
        type_cycle = ("brand", "category", "attribute")
        prefixes = {"brand": "牌", "category": "类", "attribute": "属"}
        while len(rows) < size:
            entity_type = type_cycle[index % 3]
            term = self._extra_term(prefixes[entity_type], index)
            rows.append({
                "entity_type": entity_type,
                "term": term,
                "entity_id": f"{entity_type}-gen-{index:06d}",
                "weight": round(self.rng.uniform(0.1, 1.4), 4),
            })
            index += 1
        return rows[:size]

    def build_rules(self, size: int, vocab_terms: List[str],
                    entity_terms: List[str]) -> List[dict]:
        known = set(vocab_terms) | set(entity_terms)
        rules: List[dict] = []
        templates = (
            ("synonym", "耳机", "耳麦"),
            ("synonym", "手机", "移动电话"),
            ("category_expand", "手机", "通讯数码"),
            ("category_expand", "冰箱", "家用电器"),
            ("category_expand", "口红", "美妆个护"),
            ("drop_recover", "保修", "全国联保"),
            ("drop_recover", "发票", "正规发票"),
            ("synonym", "咖啡", "咖啡豆"),
        )
        index = 0
        while len(rules) < size:
            rule_type, trigger, replacement = templates[index % len(templates)]
            rule_id = f"r-{index:05d}"
            if trigger not in known:
                # Guarantee referential integrity by anchoring to an existing
                # core term when the generator runs at tiny scale.
                trigger = next(iter(known))
                replacement = f"{trigger}优选"
            rules.append({
                "rule_id": rule_id,
                "type": rule_type,
                "trigger_term": trigger,
                "replacement": replacement,
                "weight": round(self.rng.uniform(0.6, 2.2), 4),
                "ref_terms": [trigger],
            })
            index += 1
        # One deliberately dangling rule lives in a separate fixture file in
        # tests rather than the generated valid corpus.
        return rules[:size]

    def build_queries(self, count: int, vocab_rows: List[dict],
                      entity_rows: List[dict]) -> List[dict]:
        brands = [r["term"] for r in entity_rows if r["entity_type"] == "brand"
                  and not r["term"].startswith(("牌",))]
        categories = [r["term"] for r in entity_rows
                      if r["entity_type"] == "category"
                      and not r["term"].startswith(("类",))]
        attributes = [r["term"] for r in entity_rows
                      if r["entity_type"] == "attribute"
                      and not r["term"].startswith(("属",))]
        samples: List[dict] = []
        for index in range(count):
            intent_name, words = INTENTS[index % len(INTENTS)]
            brand = self.rng.choice(BRAND_CORE)
            category = self.rng.choice(words[:4])
            attribute = self.rng.choice(ATTRIBUTE_WORDS)
            style = index % 6
            if style == 0:
                text = f"{brand} {category} {attribute}"
            elif style == 1:
                text = f"{brand}{category}新款  包邮!!"
            elif style == 2:
                text = f"ＡＢＣ {brand} {category} 128g"
            elif style == 3:
                text = f"  {attribute} {category}？？ "
            elif style == 4:
                text = f"{brand.upper()} {category} PRO 2026"
            else:
                text = f"想买{attribute}的{category}，{brand}怎么样"
            samples.append({
                "query_id": f"q-{index:06d}",
                "query": text,
                "context": {"session_pos": index % 5,
                            "prev_query": samples[-1]["query"] if samples and index % 5 else ""},
                "labels": {"intent": intent_name,
                           "brand": brand if style != 3 else None},
            })
        return samples

    def generate_tier(self, out_dir: str, spec: TierSpec) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        vocab = self.build_vocab(spec.vocab)
        entities = self.build_entities(spec.entities)
        vocab_terms = [row["term"] for row in vocab]
        entity_terms = [row["term"] for row in entities]
        rules = self.build_rules(spec.rules, vocab_terms, entity_terms)
        queries = self.build_queries(spec.queries, vocab, entities)

        paths = {
            "samples": os.path.join(out_dir, f"{spec.tier_id}_queries.jsonl"),
            "vocab": os.path.join(out_dir, f"{spec.tier_id}_vocab.jsonl"),
            "entities": os.path.join(out_dir, f"{spec.tier_id}_entities.jsonl"),
            "rules": os.path.join(out_dir, f"{spec.tier_id}_rules.jsonl"),
        }
        _write_jsonl(paths["vocab"], vocab)
        _write_jsonl(paths["entities"], entities)
        _write_jsonl(paths["rules"], rules)
        _write_jsonl(paths["samples"], queries)
        # A companion file exercising all loader-level failure codes.
        invalid_path = os.path.join(out_dir, f"{spec.tier_id}_invalid_queries.jsonl")
        invalid_lines = list(INVALID_LINES)
        if queries:
            invalid_lines.append(json.dumps(queries[0], ensure_ascii=False))
            invalid_lines.append(json.dumps(queries[0], ensure_ascii=False))
        _write_raw_lines(invalid_path, invalid_lines)
        paths["invalid_samples"] = invalid_path
        return paths
