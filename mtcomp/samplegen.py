from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from .commands import apply_command
from .engine import Engine
from .model import LexiconEntry, ScoreParams, SourceSentence, TranslationState

SEED = 20261122
VOCABULARY = (
    "the red cat runs fast a blue bird sings quietly many small robots work "
    "together data flows safely robot unused"
).split()

BASE_LEXICON_LINES = [
    "the 这 100 0.96",
    "red 红 80 0.92",
    "red 红色 45 0.89",
    "cat 猫 95 0.98",
    "cat 猫咪 35 0.93",
    "runs 跑 70 0.90",
    "runs 奔跑 20 0.90",
    "runs 运行 30 0.89",
    "fast 快 60 0.88",
    "a 一 90 0.90",
    "blue 蓝 75 0.91",
    "bird 鸟 80 0.95",
    "sings 唱 55 0.89",
    "quietly 安静地 40 0.86",
    "many 许多 65 0.90",
    "small 小 72 0.91",
    "robots 机器人 85 0.93",
    "robot 机器人 52 0.91",
    "work 工作 75 0.90",
    "work 运转 20 0.87",
    "together 一起 50 0.88",
    "data 数据 90 0.94",
    "flows 流动 60 0.90",
    "safely 安全地 48 0.87",
    "unused 未用 1 0.50",
]

BASE_SENTENCES = [
    {"id": "S001", "tokens": "the red cat runs fast", "batch": 0},
    {"id": "S002", "tokens": "a blue bird sings quietly", "batch": 0},
    {"id": "S003", "tokens": "many small robots work together", "batch": 1},
    {"id": "S004", "tokens": "data flows safely", "batch": 1},
    {"id": "S005", "tokens": "the small cat runs", "batch": 2},
    {"id": "S006", "tokens": "many robots work fast", "batch": 2},
]

COMMANDS: list[dict[str, Any]] = [
    {
        "id": "E001",
        "op": "lex_delete",
        "source": "unused",
        "target": "未用",
    },
    {
        "id": "E002",
        "op": "lex_frequency",
        "source": "cat",
        "target": "猫",
        "value": 118,
    },
    {
        "id": "E003",
        "op": "lex_confidence",
        "source": "red",
        "target": "红",
        "value": 0.91,
    },
    {
        "id": "E004",
        "op": "lex_add",
        "source": "fast",
        "target": "迅速地",
        "frequency": 45,
        "alignment_confidence": 0.93,
    },
    {
        "id": "E005",
        "op": "param_set",
        "field": "order_weight",
        "value": 1.10,
    },
    {
        "id": "E006",
        "op": "sent_replace",
        "sentence": {"id": "S004", "tokens": "data flows", "batch": 1},
    },
    {
        "id": "E007",
        "op": "sent_delete",
        "sentence_id": "S002",
    },
    {
        "id": "E008",
        "op": "batch_append",
        "sentences": [
            {"id": "S007", "tokens": "a small robot runs fast", "batch": 3},
            {"id": "S008", "tokens": "the data flows safely", "batch": 3},
        ],
    },
]


def base_config() -> dict[str, Any]:
    return {
        "ngram_order": 3,
        "max_sentences_per_batch": 3,
        "thread_pool_concurrency": 4,
        "decode_points_per_batch": 40,
        "snapshot_commit_interval_batches": 1,
        "max_sentence_tokens": 12,
        "candidate_limit": 6,
        "controlled_vocabulary": list(VOCABULARY),
    }


def base_params(threshold: float = 0.5) -> dict[str, Any]:
    return {
        "alignment_weight": 1.25,
        "order_weight": 0.85,
        "length_penalty": 0.18,
        "fusion_temperature": 0.92,
        "confidence_threshold": threshold,
        "fixed_point_bits": 12,
        "quantization_tier": 16,
        "allowed_fixed_bits": [8, 12, 16, 20, 24],
        "allowed_quantization_tiers": [1, 2, 4, 8, 16, 32],
    }


def _state_from_raw(
    config: dict[str, Any], params: dict[str, Any]
) -> TranslationState:
    from .model import CompileConfig

    compile_config = CompileConfig(
        ngram_order=int(config["ngram_order"]),
        max_sentences_per_batch=int(config["max_sentences_per_batch"]),
        thread_pool_concurrency=int(config["thread_pool_concurrency"]),
        decode_points_per_batch=int(config["decode_points_per_batch"]),
        snapshot_commit_interval_batches=int(
            config["snapshot_commit_interval_batches"]
        ),
        max_sentence_tokens=int(config["max_sentence_tokens"]),
        vocabulary=tuple(config["controlled_vocabulary"]),
        candidate_limit=int(config.get("candidate_limit", 8)),
    )
    score_params = ScoreParams.from_mapping(params)
    lexicon = {}
    for line in BASE_LEXICON_LINES:
        source, target, frequency, confidence = line.split()
        entry = LexiconEntry(
            source=source,
            target=target,
            frequency=float(frequency),
            alignment_confidence=float(confidence),
        )
        lexicon[entry.key] = entry
    sentences = {
        item["id"]: SourceSentence(
            sentence_id=item["id"],
            tokens=tuple(item["tokens"].split()),
            batch=int(item["batch"]),
        )
        for item in BASE_SENTENCES
    }
    return TranslationState(
        sentences=sentences,
        lexicon=lexicon,
        params=score_params,
        config=compile_config,
    )


def _calibrate_threshold(config: dict[str, Any]) -> float:
    state = _state_from_raw(config, base_params(0.5))
    final_state = copy.deepcopy(state)
    for index, command in enumerate(COMMANDS):
        outcome = apply_command(final_state, command, index, final_state.config)
        if outcome.failure is not None:
            raise RuntimeError(f"sample command failed: {outcome.failure.message}")
    engine = Engine(final_state)
    engine.run_to_idle()
    rows = [row for row in engine.sentence_rows() if row["status"] != "ERROR"]
    if not rows:
        raise RuntimeError("sample does not produce any translated sentence")
    preferred = [
        row
        for row in rows
        if 100 <= int(row["confidence_fixed"]) < (1 << 12)
    ]
    # Prefer a unique non-saturated confidence. This yields ACCEPT above,
    # REJECT below and exactly one REVIEW boundary sentence.
    counts: dict[int, int] = {}
    for item in preferred:
        value = int(item["confidence_fixed"])
        counts[value] = counts.get(value, 0) + 1
    unique = [item for item in preferred if counts[int(item["confidence_fixed"])] == 1]
    if unique:
        row = max(unique, key=lambda item: int(item["confidence_fixed"]))
    elif preferred:
        row = max(preferred, key=lambda item: int(item["confidence_fixed"]))
    else:
        row = rows[0]
    if row["confidence_fixed"] is None:
        raise RuntimeError("cannot calibrate threshold from error row")
    return float(int(row["confidence_fixed"]) / (1 << 12))


def write_sample(output_dir: str | Path, seed: int = SEED) -> dict[str, Any]:
    if seed != SEED:
        raise ValueError(f"this deterministic sample is locked to seed {SEED}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = base_config()
    threshold = _calibrate_threshold(config)
    params = base_params(threshold)

    (output / "compile.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    (output / "params.json").write_text(
        json.dumps(params, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "lexicon.tsv").write_text(
        "\n".join(BASE_LEXICON_LINES) + "\n", encoding="utf-8"
    )
    (output / "sentences.jsonl").write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in BASE_SENTENCES
        ),
        encoding="utf-8",
    )
    (output / "edits.json").write_text(
        json.dumps(COMMANDS, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    injections = {
        "schema_version": 1,
        "injections": [
            {
                "id": "INJ-LEX-001",
                "type": "lexicon_invalid",
                "line": "broken_target zero_frequency 0 1.2",
            },
            {
                "id": "INJ-WEIGHT-002",
                "type": "nonfinite_weight",
                "field": "alignment_weight",
                "value": "NaN",
            },
            {
                "id": "INJ-SENT-003",
                "type": "overlong_sentence",
                "sentence_id": "INJ-LONG-SENTENCE",
                "batch": 99,
                "token": "the",
                "token_count": 99,
            },
        ],
    }
    (output / "injections.json").write_text(
        json.dumps(injections, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "seed": seed,
        "threshold": threshold,
        "files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
