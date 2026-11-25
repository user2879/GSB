"""Deterministic local sample generator (fixed integer seed)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .formats import encode_jv1, jv2_to_jv1


SEED = 3182026


POS_TAGS = [
    {"name": "NN", "code": 1, "label_zh": "名词"},
    {"name": "NR", "code": 2, "label_zh": "专有名词"},
    {"name": "VV", "code": 3, "label_zh": "动词"},
    {"name": "AJ", "code": 4, "label_zh": "形容词"},
    {"name": "AD", "code": 5, "label_zh": "副词"},
]


ENTRY_SPECS: list[tuple[str, str, str, list[str] | None, dict[str, float] | None]] = [
    ("E001", "政策", "NN", None, None),
    ("E002", "经济", "NN", None, None),
    ("E003", "发展", "VV", ["VV", "NN"], {"VV": 0.35, "NN": -0.10}),
    ("E004", "市场", "NN", None, None),
    ("E005", "监管", "VV", ["VV", "NN"], {"VV": 0.25, "NN": 0.05}),
    ("E006", "创新", "VV", ["VV", "NN"], {"VV": 0.15, "NN": 0.20}),
    ("E007", "企业", "NN", None, None),
    ("E008", "投资", "VV", ["VV", "NN"], {"VV": 0.40, "NN": 0.10}),
    ("E009", "增长", "VV", ["VV", "NN"], {"VV": 0.30, "NN": 0.15}),
    ("E010", "风险", "NN", None, None),
    ("E011", "管理", "VV", ["VV", "NN"], {"VV": 0.20, "NN": 0.00}),
    ("E012", "服务", "VV", ["VV", "NN"], {"VV": 0.18, "NN": 0.08}),
    ("E013", "数据", "NN", None, None),
    ("E014", "模型", "NN", None, None),
    ("E015", "系统", "NN", None, None),
    ("E016", "分析", "VV", ["VV", "NN"], {"VV": 0.28, "NN": 0.02}),
    ("E017", "预测", "VV", ["VV", "NN"], {"VV": 0.32, "NN": -0.05}),
    ("E018", "研究", "VV", ["VV", "NN"], {"VV": 0.22, "NN": 0.12}),
    ("E019", "研究生", "NN", None, {"NN": 0.45}),
    ("E020", "生命", "NN", None, None),
    ("E021", "科学", "NN", None, None),
    ("E022", "人工智能", "NN", None, {"NN": 0.55}),
    ("E023", "智能", "NN", ["NN", "AJ"], {"NN": 0.20, "AJ": -0.08}),
    ("E024", "技术", "NN", None, None),
    ("E025", "记录", "NN", ["NN", "VV"], {"NN": 0.0, "VV": 0.0}),
    ("E026", "人工", "AJ", ["AJ", "NN"], {"AJ": 0.10, "NN": -0.05}),
    ("E027", "计算", "VV", ["VV", "NN"], {"VV": 0.26, "NN": 0.04}),
]


DOC_SPECS = [
    ("D001", 1, "研究生命科学发展", 4),
    ("D002", 2, "人工智能技术创新", 4),
    ("D003", 3, "市场经济监管服务", 4),
    ("D004", 4, "企业投资风险管理", 4),
    ("D005", 5, "数据模型分析预测", 4),
    ("D006", 6, "记录系统管理", 3),
    ("D007", 7, "量子计算预测增长模型系统", 7),
]


def build_samples(seed: int = SEED) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    entries: list[dict[str, Any]] = []
    for index, (entry_id, term, pos, pos_tags, weights) in enumerate(ENTRY_SPECS):
        frequency = int(rng.randint(24, 180))
        if entry_id == "E025":
            frequency = 96
        entries.append(
            {
                "id": entry_id,
                "term": term,
                "frequency": frequency,
                "pos": pos,
                "pos_tags": pos_tags or [pos],
                "enabled": True,
                "version": 3,
                "tag_weights": weights or {pos: round(float(rng.uniform(-0.12, 0.22)), 3)},
            }
        )

    lexicon = {
        "lexicon_id": "LEX-318-SEED",
        "seed": seed,
        "revision": 3,
        "pos_tags": POS_TAGS,
        "entries": entries,
    }

    tag_names = [tag["name"] for tag in POS_TAGS]
    transition_weights: dict[str, dict[str, float]] = {}
    for source in tag_names:
        transition_weights[source] = {}
        for target in tag_names:
            value = round(float(rng.uniform(-1.15, 1.15)), 3)
            if source == target:
                value = round(value + 0.25, 3)
            transition_weights[source][target] = value

    weights = {
        "weights_id": "W-318",
        "seed": seed,
        "default_transition_weight": 0.0,
        "transition_weights": transition_weights,
    }

    documents = {
        "documents_id": "DOC-318",
        "seed": seed,
        "documents": [
            {"doc_id": doc_id, "doc_seq": seq, "text": text, "term_count": term_count}
            for doc_id, seq, text, term_count in DOC_SPECS
        ],
    }

    rules = {
        "rules_package_id": "RULES-318",
        "seed": seed,
        "rules": [
            {
                "rule_id": "R-SPLIT-AI-AT-2",
                "version": 2,
                "order": 0,
                "doc_id": "D002",
                "action": "split",
                "position": 2,
                "enabled": True,
                "reason": "force artificial|intelligence boundary",
            },
            {
                "rule_id": "R-MERGE-AI-CONFLICT",
                "version": 1,
                "order": 0,
                "doc_id": "D002",
                "action": "merge",
                "position": 2,
                "enabled": True,
                "reason": "lower-version conflict used for tie/version resolution",
            },
            {
                "rule_id": "R-MERGE-MARKET-ECONOMY",
                "version": 1,
                "order": 1,
                "doc_id": "D003",
                "action": "merge",
                "position": 2,
                "enabled": True,
                "reason": "merge market and economy",
            },
        ],
    }

    revisions = {
        "chain_id": "DICT-REV-318",
        "seed": seed,
        "head_revision": 3,
        "revisions": [
            {"revision": 0, "parent_revision": None, "label": "root"},
            {"revision": 1, "parent_revision": 0, "label": "initial-controlled-vocabulary"},
            {"revision": 2, "parent_revision": 1, "label": "ambiguity-rules"},
            {"revision": 3, "parent_revision": 2, "label": "seeded-local-sample"},
        ],
    }

    commands = {
        "command_stream_id": "CMDS-318",
        "seed": seed,
        "commands": [
            {
                "seq": 1,
                "op": "upsert_entry",
                "entry": {
                    "id": "E029",
                    "term": "量子",
                    "frequency": 88,
                    "pos": "NN",
                    "pos_tags": ["NN"],
                    "enabled": True,
                    "version": 4,
                    "tag_weights": {"NN": 0.10},
                },
            },
            {"seq": 2, "op": "set_frequency", "id": "E001", "frequency": 132, "entry_version": 4},
            {
                "seq": 3,
                "op": "update_weights",
                "transition_weights": {"NN": {"VV": 0.62, "NN": 0.31}},
            },
            {
                "seq": 4,
                "op": "upsert_rule",
                "rule": {
                    "rule_id": "R-SPLIT-MODEL-INTERNAL",
                    "version": 3,
                    "order": 2,
                    "doc_id": "D005",
                    "action": "split",
                    "position": 3,
                    "enabled": True,
                },
            },
            {"seq": 5, "op": "set_frequency", "id": "E001", "frequency": 1, "entry_version": 1},
            {"seq": 6, "op": "delete_entry", "id": "E024"},
            {"seq": 7, "op": "rollback_revision", "revision": 3},
            {"seq": 8, "op": "set_frequency", "id": "E010", "frequency": 211, "entry_version": 4},
            {"seq": 9, "op": "set_exchange_version", "version": "jv1"},
            {"seq": 10, "op": "set_exchange_version", "version": "jv2"},
            {"seq": 11, "op": "set_quant_tier", "tier": "fine"},
        ],
    }

    sample_receipts = _build_exchange_samples(rng, POS_TAGS)
    return {
        "lexicon.json": lexicon,
        "documents.json": documents,
        "weights.json": weights,
        "rules.json": rules,
        "revisions.json": revisions,
        "commands.json": commands,
        **sample_receipts,
    }


def _build_exchange_samples(rng: np.random.RandomState, pos_tags: list[dict[str, Any]]) -> dict[str, Any]:
    names = [tag["name"] for tag in pos_tags[:3]]
    codes = {tag["name"]: tag["code"] for tag in pos_tags}
    receipts: list[dict[str, Any]] = []
    for index in range(6):
        score = round(float(rng.uniform(0.52, 0.995)), 6)
        receipts.append(
            {
                "doc_seq": 1,
                "fragment_seq": index + 1,
                "score": score,
                "pos": names[index % len(names)],
                "rounding_mode": "ROUND_HALF_UP",
                "precision_label": "sample:float64",
                "precision_labels": ["sample", "float64"],
            }
        )
    jv2 = {"format": "jv2", "version": "2.0", "rounding_mode": "ROUND_HALF_UP", "receipts": receipts}
    downgrade = jv2_to_jv1(receipts, codes)
    valid_jv1 = downgrade.text or encode_jv1(
        [{"doc_seq": 1, "fragment_seq": i + 1, "score_permille": int(rng.randint(500, 1000)), "pos_code": (i % 3) + 1} for i in range(6)]
    ).text
    truncated_jv1 = (valid_jv1 or "").splitlines()[0] + "\n000100\n000100030098x0003\n"
    invalid_jv2 = {
        "format": "jv2",
        "version": "2.0",
        "receipts": [
            {"doc_seq": 1, "fragment_seq": 1, "pos": "NN", "rounding_mode": "ROUND_HALF_UP"},
            {"doc_seq": "2", "fragment_seq": 2, "score": "not-a-number", "pos": 7, "rounding_mode": "BAD_MODE"},
        ],
    }
    return {
        "jv2/sample_receipts.jv2.json": jv2,
        "jv2/invalid.jv2.json": invalid_jv2,
        "jv1/sample_receipts.jv1": {"__text__": valid_jv1},
        "jv1/truncated.jv1": {"__text__": truncated_jv1},
        "jv2/sample_sidecar.json": downgrade.sidecar,
    }


def generate_samples(output_dir: str | Path = "samples", seed: int = SEED) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = build_samples(seed)
    written: list[Path] = []
    for relative, value in samples.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, dict) and "__text__" in value:
            path.write_text(value["__text__"], encoding="utf-8")
        else:
            path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    paths = generate_samples()
    print(f"generated {len(paths)} sample files in samples/ with seed {SEED}")


if __name__ == "__main__":
    main()
