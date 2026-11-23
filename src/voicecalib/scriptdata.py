"""对话脚本 JSON 与声学证据摘要 JSONL 的加载校验。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import ErrorCollector, ErrorCode
from .features import FEATURE_DIM, FEATURE_NAMES


@dataclass
class ScriptRound:
    round: int
    audio_file: str
    speaker: str
    link: str
    emotion: str


@dataclass
class ScriptData:
    rounds: list[ScriptRound]
    speakers: list[str]


def load_script(path: str | Path, errors: ErrorCollector) -> ScriptData | None:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.add(ErrorCode.E_SCRIPT_STRUCTURE, "script",
                   f"cannot parse script: {exc}", file=str(path))
        return None

    entries = raw.get("rounds") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        errors.add(ErrorCode.E_SCRIPT_STRUCTURE, "script",
                   "script.rounds must be a non-empty list", file=str(path))
        return None

    rounds: list[ScriptRound] = []
    seen_audio: set[str] = set()
    expected = 0
    structure_bad = False
    for entry in entries:
        rnd = entry.get("round")
        # 轮次序号严格递增、从 0 开始、无空洞。
        if not isinstance(rnd, int) or rnd != expected:
            errors.add(ErrorCode.E_SCRIPT_STRUCTURE, "script",
                       f"round sequence error at position {expected}: got {rnd}",
                       round=rnd if isinstance(rnd, int) else None)
            structure_bad = True
            expected += 1
            continue
        expected += 1
        audio_file = entry.get("audio_file")
        speaker = entry.get("speaker")
        link = entry.get("link", "init" if rnd == 0 else "same_speaker")
        emotion = entry.get("emotion", "neutral")
        if not isinstance(audio_file, str) or not audio_file:
            errors.add(ErrorCode.E_SCRIPT_STRUCTURE, "script",
                       "audio_file missing", round=rnd)
            structure_bad = True
            continue
        if audio_file in seen_audio:
            errors.add(ErrorCode.E_SCRIPT_STRUCTURE, "script",
                       f"audio {audio_file} mapped to more than one round",
                       round=rnd, file=audio_file)
            structure_bad = True
            continue
        seen_audio.add(audio_file)
        if not isinstance(speaker, str) or not speaker.strip():
            errors.add(ErrorCode.E_SPEAKER_EMPTY, "script",
                       "speaker id must be non-empty", round=rnd, file=audio_file)
            structure_bad = True
            continue
        if link not in ("init", "same_speaker", "speaker_switch"):
            errors.add(ErrorCode.E_SCRIPT_STRUCTURE, "script",
                       f"unknown link type {link}", round=rnd, file=audio_file)
            structure_bad = True
            continue
        rounds.append(ScriptRound(rnd, audio_file, speaker, link, emotion))

    if structure_bad and not rounds:
        return None
    speakers = sorted({r.speaker for r in rounds})
    return ScriptData(rounds=rounds, speakers=speakers)


def load_evidence_jsonl(path: str | Path, errors: ErrorCollector,
                        valid_rounds: set[int]) -> dict[int, dict]:
    """校验三维证据为有限数；返回 round -> 证据条目。"""
    result: dict[int, dict] = {}
    path = Path(path)
    if not path.exists():
        errors.add(ErrorCode.E_EVIDENCE_FORMAT, "evidence",
                   "evidence jsonl not found", file=str(path))
        return result
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.add(ErrorCode.E_EVIDENCE_FORMAT, "evidence",
                       f"line {line_no} invalid json: {exc}", file=path.name)
            continue
        rnd = entry.get("round")
        if not isinstance(rnd, int) or rnd not in valid_rounds:
            errors.add(ErrorCode.E_EVIDENCE_FORMAT, "evidence",
                       f"line {line_no} round {rnd} not bound to a valid script round",
                       round=rnd if isinstance(rnd, int) else None, file=path.name)
            continue
        feats = entry.get("features")
        if not isinstance(feats, dict) or not all(
                name in feats for name in FEATURE_NAMES):
            errors.add(ErrorCode.E_EVIDENCE_FORMAT, "evidence",
                       f"line {line_no} must contain {FEATURE_NAMES}",
                       round=rnd, file=path.name)
            continue
        values = [feats[name] for name in FEATURE_NAMES]
        if len(values) != FEATURE_DIM or not all(
                isinstance(v, (int, float)) for v in values):
            errors.add(ErrorCode.E_EVIDENCE_FORMAT, "evidence",
                       f"line {line_no} feature dimensionality error",
                       round=rnd, file=path.name)
            continue
        if not all(np.isfinite(v) for v in values):
            errors.add(ErrorCode.E_NONFINITE_SAMPLE, "evidence",
                       f"line {line_no} contains non-finite feature",
                       round=rnd, file=path.name)
            continue
        if rnd in result:
            errors.add(ErrorCode.E_EVIDENCE_FORMAT, "evidence",
                       f"duplicate evidence for round {rnd}",
                       round=rnd, file=path.name)
            continue
        result[rnd] = entry
    return result
