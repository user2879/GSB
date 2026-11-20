from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_SEED = 20261120


VOCAB_ENTRIES: List[Dict[str, Any]] = [
    {"token": "<sil>", "pos": "SIL", "slot_start": False, "slot": None, "subwords": ["<sil>"]},
    {"token": "请", "pos": "FILLER", "slot_start": False, "slot": None, "subwords": ["请"]},
    {"token": "打开", "pos": "VERB", "slot_start": True, "slot": "command", "subwords": ["打", "开"]},
    {"token": "关闭", "pos": "VERB", "slot_start": True, "slot": "command", "subwords": ["关", "闭"]},
    {"token": "调高", "pos": "VERB", "slot_start": True, "slot": "command", "subwords": ["调", "高"]},
    {"token": "调低", "pos": "VERB", "slot_start": True, "slot": "command", "subwords": ["调", "低"]},
    {"token": "播放", "pos": "VERB", "slot_start": True, "slot": "command", "subwords": ["播", "放"]},
    {"token": "暂停", "pos": "VERB", "slot_start": True, "slot": "command", "subwords": ["暂", "停"]},
    {"token": "空调", "pos": "NOUN", "slot_start": True, "slot": "device", "subwords": ["空", "调"]},
    {"token": "灯光", "pos": "NOUN", "slot_start": True, "slot": "device", "subwords": ["灯", "光"]},
    {"token": "音量", "pos": "NOUN", "slot_start": True, "slot": "device", "subwords": ["音", "量"]},
    {"token": "音乐", "pos": "NOUN", "slot_start": True, "slot": "device", "subwords": ["音", "乐"]},
    {"token": "窗帘", "pos": "NOUN", "slot_start": True, "slot": "device", "subwords": ["窗", "帘"]},
    {"token": "模式", "pos": "NOUN", "slot_start": True, "slot": "mode", "subwords": ["模", "式"]},
    {"token": "制冷", "pos": "NOUN", "slot_start": True, "slot": "mode", "subwords": ["制", "冷"]},
    {"token": "制热", "pos": "NOUN", "slot_start": True, "slot": "mode", "subwords": ["制", "热"]},
    {"token": "一", "pos": "NUM", "slot_start": True, "slot": "value", "subwords": ["一"]},
    {"token": "二", "pos": "NUM", "slot_start": True, "slot": "value", "subwords": ["二"]},
    {"token": "三", "pos": "NUM", "slot_start": True, "slot": "value", "subwords": ["三"]},
    {"token": "四", "pos": "NUM", "slot_start": True, "slot": "value", "subwords": ["四"]},
    {"token": "五", "pos": "NUM", "slot_start": True, "slot": "value", "subwords": ["五"]},
    {"token": "度", "pos": "UNIT", "slot_start": False, "slot": None, "subwords": ["度"]},
    {"token": "到", "pos": "PREP", "slot_start": False, "slot": None, "subwords": ["到"]},
    {"token": "把", "pos": "PREP", "slot_start": False, "slot": None, "subwords": ["把"]},
    {"token": "并", "pos": "CONJ", "slot_start": False, "slot": None, "subwords": ["并"]},
    {"token": "吧", "pos": "PART", "slot_start": False, "slot": None, "subwords": ["吧"]},
    {"token": "自动", "pos": "ADJ", "slot_start": True, "slot": "mode", "subwords": ["自", "动"]},
]


INSTRUCTIONS: Dict[str, List[str]] = {
    "S01": ["请", "打开", "空调", "并", "把", "音量", "调高", "到", "三", "度"],
    "S02": ["请", "关闭", "灯光", "吧", "<sil>", "打开", "窗帘", "到", "二", "度"],
    "S03": ["调低", "空调", "到", "一", "度", "暂停", "音乐", "吧", "<sil>", "<sil>"],
    "S-INJ-SHARD": ["请", "打开", "灯光", "并", "把", "音量", "调低", "到", "四", "度"],
    "S-INJ-FRAME": ["调高", "空调", "到", "五", "度", "请", "关闭", "音乐", "吧", "<sil>"],
    "S-INJ-OVERLAP": ["打开", "窗帘", "到", "二", "度", "暂停", "音乐", "吧", "<sil>", "<sil>"],
    "S-INJ-LOOKAHEAD": ["请", "播放", "音乐", "并", "调高", "音量", "到", "三", "度", "吧"],
    "S04": ["打开", "模式", "制冷", "到", "三", "度", "关闭", "灯光", "吧", "<sil>"],
}

SHARD_LAYOUTS: Dict[str, List[int]] = {
    "S01": [5, 4, 5],
    "S02": [5, 4, 5],
    "S03": [6, 5, 6],
    "S-INJ-SHARD": [5, 4, 5],
    "S-INJ-FRAME": [5, 4, 5],
    "S-INJ-OVERLAP": [5, 4, 5],
    "S-INJ-LOOKAHEAD": [5, 4, 5],
    "S04": [5, 4, 5],
}


BIGRAM_TEMPLATES: Dict[Tuple[str, str], float] = {
    ("<s>", "请"): 1.1,
    ("<s>", "打开"): 0.8,
    ("<s>", "关闭"): 0.8,
    ("<s>", "调高"): 0.8,
    ("<s>", "调低"): 0.8,
    ("<s>", "播放"): 0.8,
    ("<s>", "暂停"): 0.8,
    ("请", "打开"): 1.0,
    ("请", "关闭"): 1.0,
    ("请", "播放"): 1.0,
    ("打开", "空调"): 1.8,
    ("打开", "灯光"): 1.7,
    ("打开", "窗帘"): 1.7,
    ("打开", "模式"): 1.6,
    ("关闭", "灯光"): 1.8,
    ("关闭", "空调"): 1.7,
    ("关闭", "音乐"): 1.7,
    ("调高", "音量"): 1.9,
    ("调低", "音量"): 1.9,
    ("调低", "空调"): 1.7,
    ("播放", "音乐"): 1.9,
    ("暂停", "音乐"): 1.9,
    ("空调", "到"): 1.2,
    ("灯光", "吧"): 1.1,
    ("音乐", "吧"): 1.1,
    ("音量", "到"): 1.2,
    ("窗帘", "到"): 1.2,
    ("模式", "制冷"): 1.8,
    ("模式", "制热"): 1.8,
    ("制冷", "到"): 1.3,
    ("制热", "到"): 1.3,
    ("到", "一"): 1.8,
    ("到", "二"): 1.8,
    ("到", "三"): 1.8,
    ("到", "四"): 1.8,
    ("到", "五"): 1.8,
    ("一", "度"): 2.0,
    ("二", "度"): 2.0,
    ("三", "度"): 2.0,
    ("四", "度"): 2.0,
    ("五", "度"): 2.0,
    ("度", "吧"): 1.0,
    ("度", "并"): 1.0,
    ("度", "暂停"): 0.8,
    ("度", "关闭"): 0.8,
    ("并", "把"): 1.7,
    ("把", "音量"): 1.7,
    ("把", "空调"): 1.5,
    ("音量", "调高"): 1.5,
    ("音量", "调低"): 1.5,
    ("吧", "<sil>"): 0.6,
    ("<sil>", "打开"): 0.3,
    ("<sil>", "<sil>"): 0.2,
}


@dataclass
class SamplePayloads:
    vocab: Dict[str, Any]
    config: Dict[str, Any]
    sessions: List[Dict[str, Any]]
    scores: List[Dict[str, Any]]
    injections: Dict[str, Any]
    extras: Dict[str, List[Dict[str, Any]]]


def _layout_global_frames(lengths: Sequence[int], overlap: int) -> List[Tuple[int, int, List[int]]]:
    result: List[Tuple[int, int, List[int]]] = []
    start = 0
    for index, length in enumerate(lengths):
        if index > 0:
            start = result[-1][1] - overlap + 1
        end = start + length - 1
        result.append((start, end, list(range(start, end + 1))))
    return result


def _score_vector(
    vocab_tokens: Sequence[str],
    intended: str,
    rng: random.Random,
    *,
    forced_missing: bool = False,
    shard_seq: int = 0,
) -> List[float]:
    if forced_missing:
        return [-100.0 for _ in vocab_tokens]
    token_index = {token: index for index, token in enumerate(vocab_tokens)}
    values = [-100.0 for _ in vocab_tokens]
    intended_id = token_index[intended]
    values[intended_id] = 2.4 + 0.001 * shard_seq + rng.random() * 0.012
    # A deterministic near competitor exercises tie-breaking and stable-margin
    # logic without changing the intended instruction.
    competitor = vocab_tokens[(intended_id + 1) % len(vocab_tokens)]
    if competitor == "<sil>":
        competitor = vocab_tokens[(intended_id + 2) % len(vocab_tokens)]
    values[token_index[competitor]] = 0.4 + rng.random() * 0.03
    if intended != "<sil>":
        values[token_index["<sil>"]] = -3.0
    return values


def make_sample_payloads(seed: int = DEFAULT_SEED) -> SamplePayloads:
    rng = random.Random(seed)
    vocab_tokens = [item["token"] for item in VOCAB_ENTRIES]
    vocab = {
        "version": f"zh-instruction-{seed}",
        "start_token": "<s>",
        "smoothing_token": "<sil>",
        "subword_rules": {
            "strategy": "restricted-longest-match",
            "max_subword_chars": 1,
            "token_separator": "",
            "subword_separator": "|",
            "join": "",
        },
        "entries": VOCAB_ENTRIES,
        "bigrams": [
            {"prev": previous, "token": token, "score": score}
            for (previous, token), score in sorted(BIGRAM_TEMPLATES.items())
        ],
    }
    config = {
        "overlap_frames": 2,
        "min_stable_frames": 3,
        "lm_weight": 0.65,
        "lm_backoff_score": -2.5,
        "slot_min_frames": 1,
        "slot_max_frames": 4,
        "response_budget_ms": 40,
        "thread_pool_concurrency": 4,
        "snapshot_interval_shards": 2,
        "strategy": "lookahead",
        "lookahead_frames": 3,
        "smoothing_score": -12.0,
        "candidate_missing_below": -50.0,
        "stable_margin": 0.05,
        "snapshot_root": "artifacts/snapshots",
        "report_path": "artifacts/diagnostic_report.json",
        "test_compute_delay_ms": 0.0,
    }

    session_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    forced_missing = {("S02", 4)}
    for session_id in sorted(INSTRUCTIONS):
        instruction = INSTRUCTIONS[session_id]
        lengths = SHARD_LAYOUTS[session_id]
        layout = _layout_global_frames(lengths, int(config["overlap_frames"]))
        total_frames = layout[-1][1] + 1
        if len(instruction) < total_frames:
            instruction = instruction + ["<sil>"] * (total_frames - len(instruction))
        intended_by_frame = instruction[:total_frames]
        for shard_seq, (start, end, frames) in enumerate(layout):
            session_rows.append(
                {
                    "session_id": session_id,
                    "shard_seq": shard_seq,
                    "start_frame": start,
                    "end_frame": end,
                    "arrived": True,
                }
            )
            for local, global_frame in enumerate(frames):
                values = _score_vector(
                    vocab_tokens,
                    intended_by_frame[global_frame],
                    rng,
                    forced_missing=(session_id, global_frame) in forced_missing,
                    shard_seq=shard_seq,
                )
                score_rows.append(
                    {
                        "session_id": session_id,
                        "shard_seq": shard_seq,
                        "frame_in_shard": local,
                        "scores": [round(value, 8) for value in values],
                    }
                )

    # Extra records are intentionally outside the initial session stream.  They
    # let the CLI exercise append/replace/add-session commands without touching
    # the generated source JSONL files.
    extras: Dict[str, List[Dict[str, Any]]] = {}
    append_frames = [8, 9, 10, 11]
    append_tokens = ["度", "吧", "<sil>", "<sil>"]
    extras["append_scores.jsonl"] = [
        {
            "frame_in_shard": local,
            "scores": [round(value, 8) for value in _score_vector(vocab_tokens, token, rng, shard_seq=3)],
        }
        for local, token in enumerate(append_tokens)
    ]

    replace_tokens = ["并", "把", "窗帘", "调高"]
    extras["replace_scores.jsonl"] = [
        {
            "frame_in_shard": local,
            "scores": [round(value, 8) for value in _score_vector(vocab_tokens, token, rng, shard_seq=1)],
        }
        for local, token in enumerate(replace_tokens)
    ]

    extras["new_session_scores.jsonl"] = [
        {
            "frame_in_shard": local,
            "scores": [round(value, 8) for value in _score_vector(vocab_tokens, token, rng, shard_seq=shard_seq)],
        }
        for shard_seq, (_, _, frames) in enumerate(_layout_global_frames(SHARD_LAYOUTS["S04"], 2))
        for local, frame in enumerate(frames)
        for token in [INSTRUCTIONS["S04"][frame]]
    ]

    injections = {
        "items": [
            {
                "id": "INJ-001-NONFINITE",
                "type": "non_finite_score",
                "session_id": "S-INJ-SHARD",
                "shard_seq": 1,
                "business_code": "E_INPUT_INVALID",
            },
            {
                "id": "INJ-002-FRAME-COUNT",
                "type": "frame_count_mismatch",
                "session_id": "S-INJ-FRAME",
                "shard_seq": 0,
                "delta": 1,
                "business_code": "E_INPUT_INVALID",
            },
            {
                "id": "INJ-003-OVERLAP",
                "type": "overlap_not_less_than_shard",
                "session_id": "S-INJ-OVERLAP",
                "business_code": "E_INPUT_INVALID",
            },
            {
                "id": "INJ-004-ZERO-LOOKAHEAD",
                "type": "zero_lookahead",
                "session_id": "S-INJ-LOOKAHEAD",
                "business_code": "E_DECODE_PARAM_INVALID",
            },
        ]
    }
    return SamplePayloads(
        vocab=vocab,
        config=config,
        sessions=session_rows,
        scores=score_rows,
        injections=injections,
        extras=extras,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            handle.write("\n")


def write_sample(root: str | Path, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    payloads = make_sample_payloads(seed)
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    _write_json(root_path / "vocab.json", payloads.vocab)
    _write_json(root_path / "decode_config.json", payloads.config)
    _write_json(root_path / "injection_config.json", payloads.injections)
    _write_jsonl(root_path / "sessions.jsonl", payloads.sessions)
    _write_jsonl(root_path / "scores.jsonl", payloads.scores)
    for name, rows in payloads.extras.items():
        _write_jsonl(root_path / "extras" / name, rows)
    metadata = {
        "seed": seed,
        "session_count": len({row["session_id"] for row in payloads.sessions}),
        "shard_count": len(payloads.sessions),
        "score_row_count": len(payloads.scores),
        "vocab_size": len(payloads.vocab["entries"]),
        "overlap_frames": payloads.config["overlap_frames"],
        "files": [
            "vocab.json",
            "decode_config.json",
            "injection_config.json",
            "sessions.jsonl",
            "scores.jsonl",
            "extras/append_scores.jsonl",
            "extras/replace_scores.jsonl",
            "extras/new_session_scores.jsonl",
        ],
    }
    _write_json(root_path / "sample_manifest.json", metadata)
    return metadata
