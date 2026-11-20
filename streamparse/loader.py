from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .errors import (
    AppError,
    E_INPUT_INVALID,
    failure,
    sort_failures,
)
from .fingerprint import content_hash
from .models import (
    Dataset,
    DecodeConfig,
    SessionDataset,
    ShardRecord,
    VocabEntry,
    Vocabulary,
)


def read_json(path: str | Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AppError(
            E_INPUT_INVALID,
            f"cannot read JSON file {path}: {exc}",
            stage="file_load",
        ) from exc


def read_jsonl(path: str | Path) -> Tuple[List[Tuple[int, Any]], List[Dict[str, Any]]]:
    rows: List[Tuple[int, Any]] = []
    failures: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append((line_number, json.loads(stripped)))
                except json.JSONDecodeError as exc:
                    failures.append(
                        failure(
                            E_INPUT_INVALID,
                            "jsonl_parse",
                            f"invalid JSON on line {line_number}: {exc.msg}",
                            details={"line": line_number},
                        )
                    )
    except OSError as exc:
        raise AppError(
            E_INPUT_INVALID,
            f"cannot read JSONL file {path}: {exc}",
            stage="file_load",
        ) from exc
    return rows, failures


def load_vocab(path: str | Path) -> Vocabulary:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise AppError(E_INPUT_INVALID, "vocabulary root must be an object", stage="vocab_load")
    version = raw.get("version")
    if not isinstance(version, str) or not version.strip():
        raise AppError(E_INPUT_INVALID, "vocabulary version must be a non-empty string", stage="vocab_load")
    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise AppError(E_INPUT_INVALID, "vocabulary entries must be a non-empty list", stage="vocab_load")

    entries: List[VocabEntry] = []
    token_to_id: Dict[str, int] = {}
    for index, item in enumerate(entries_raw):
        if not isinstance(item, dict):
            raise AppError(E_INPUT_INVALID, f"vocabulary entry {index} is not an object", stage="vocab_load")
        token = item.get("token")
        pos = item.get("pos")
        slot_start = item.get("slot_start", False)
        slot = item.get("slot")
        subwords = item.get("subwords", [])
        if not isinstance(token, str) or not token:
            raise AppError(E_INPUT_INVALID, f"vocabulary entry {index} has an empty token", stage="vocab_load")
        if token in token_to_id:
            raise AppError(E_INPUT_INVALID, f"duplicate vocabulary token: {token}", stage="vocab_load")
        if not isinstance(pos, str) or not pos:
            raise AppError(E_INPUT_INVALID, f"vocabulary token {token} has no POS tag", stage="vocab_load")
        if not isinstance(slot_start, bool):
            raise AppError(E_INPUT_INVALID, f"vocabulary token {token} slot_start must be boolean", stage="vocab_load")
        if slot is not None and not isinstance(slot, str):
            raise AppError(E_INPUT_INVALID, f"vocabulary token {token} slot must be a string or null", stage="vocab_load")
        if not isinstance(subwords, list) or any(not isinstance(part, str) or not part for part in subwords):
            raise AppError(E_INPUT_INVALID, f"vocabulary token {token} has invalid subwords", stage="vocab_load")
        token_to_id[token] = index
        entries.append(
            VocabEntry(
                token=token,
                pos=pos,
                slot_start=slot_start,
                slot=slot,
                subwords=tuple(subwords) if subwords else (token,),
            )
        )

    smoothing_token = raw.get("smoothing_token")
    if not isinstance(smoothing_token, str) or smoothing_token not in token_to_id:
        raise AppError(
            E_INPUT_INVALID,
            "smoothing_token must identify an entry in the vocabulary",
            stage="vocab_load",
        )
    start_token = raw.get("start_token", "<s>")
    if not isinstance(start_token, str) or not start_token:
        raise AppError(E_INPUT_INVALID, "start_token must be a non-empty string", stage="vocab_load")

    subword_rules = raw.get("subword_rules", {})
    if not isinstance(subword_rules, dict):
        raise AppError(E_INPUT_INVALID, "subword_rules must be an object", stage="vocab_load")
    for key in ("token_separator", "subword_separator", "join"):
        if key in subword_rules and not isinstance(subword_rules[key], str):
            raise AppError(E_INPUT_INVALID, f"subword rule {key} must be a string", stage="vocab_load")

    bigrams: Dict[Tuple[str, str], float] = {}
    bigrams_raw = raw.get("bigrams", [])
    if isinstance(bigrams_raw, dict):
        iterator = []
        for prev, values in bigrams_raw.items():
            if not isinstance(values, dict):
                raise AppError(E_INPUT_INVALID, "bigram dictionary values must be objects", stage="vocab_load")
            for token, score in values.items():
                iterator.append({"prev": prev, "token": token, "score": score})
    elif isinstance(bigrams_raw, list):
        iterator = bigrams_raw
    else:
        raise AppError(E_INPUT_INVALID, "bigrams must be a list or object", stage="vocab_load")
    for item in iterator:
        if not isinstance(item, dict):
            raise AppError(E_INPUT_INVALID, "each bigram must be an object", stage="vocab_load")
        prev = item.get("prev")
        token = item.get("token")
        score = item.get("score")
        if not isinstance(prev, str) or not isinstance(token, str):
            raise AppError(E_INPUT_INVALID, "bigram prev/token must be strings", stage="vocab_load")
        try:
            numeric = float(score)
        except (TypeError, ValueError) as exc:
            raise AppError(E_INPUT_INVALID, "bigram score must be numeric", stage="vocab_load") from exc
        if not math.isfinite(numeric):
            raise AppError(E_INPUT_INVALID, "bigram score must be finite", stage="vocab_load")
        bigrams[(prev, token)] = numeric

    vocab = Vocabulary(
        version=version,
        entries=tuple(entries),
        token_to_id=token_to_id,
        bigrams=bigrams,
        start_token=start_token,
        smoothing_token=smoothing_token,
        smoothing_token_id=token_to_id[smoothing_token],
        subword_rules=dict(subword_rules),
        source_fingerprint=content_hash(
            {
                "version": version,
                "entries": [entry.as_dict() for entry in entries],
                "bigrams": sorted([[a, b, score] for (a, b), score in bigrams.items()]),
                "subword_rules": subword_rules,
                "smoothing_token": smoothing_token,
                "start_token": start_token,
            }
        ),
    )
    return vocab


def load_config(path: str | Path) -> DecodeConfig:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise AppError(E_INPUT_INVALID, "decode config root must be an object", stage="config_load")
    return DecodeConfig.from_mapping(raw)


def load_injection_config(path: str | Path) -> Dict[str, Any]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise AppError(E_INPUT_INVALID, "injection config root must be an object", stage="injection_load")
    items = raw.get("items", [])
    if not isinstance(items, list):
        raise AppError(E_INPUT_INVALID, "injection items must be a list", stage="injection_load")
    seen: Set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise AppError(E_INPUT_INVALID, f"injection item {index} must be an object", stage="injection_load")
        injection_id = item.get("id")
        if not isinstance(injection_id, str) or not injection_id:
            raise AppError(E_INPUT_INVALID, f"injection item {index} needs a non-empty id", stage="injection_load")
        if injection_id in seen:
            raise AppError(E_INPUT_INVALID, f"duplicate injection id: {injection_id}", stage="injection_load")
        seen.add(injection_id)
        if not isinstance(item.get("type"), str):
            raise AppError(E_INPUT_INVALID, f"injection {injection_id} needs a type", stage="injection_load")
    return raw


def _score_session_key(raw: Mapping[str, Any]) -> str:
    session_id = raw.get("session_id", "__GLOBAL__")
    return "" if session_id is None else str(session_id)


def load_scores(
    path: str | Path,
    vocab: Vocabulary,
) -> Tuple[Dict[str, Dict[int, Dict[int, Tuple[float, ...]]]], Dict[str, Set[int]], List[Dict[str, Any]]]:
    """Load frame score rows.

    The returned score mapping is keyed by session id (with ``__GLOBAL__`` for
    legacy rows without a session id), shard sequence, and local frame index.
    """

    rows, failures = read_jsonl(path)
    scores: Dict[str, Dict[int, Dict[int, Tuple[float, ...]]]] = {}
    invalid_shards: Dict[str, Set[int]] = {}
    seen: Set[Tuple[str, int, int]] = set()
    vocab_size = len(vocab)

    for line_number, raw in rows:
        if not isinstance(raw, dict):
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "score_load",
                    "score row must be an object",
                    details={"line": line_number},
                )
            )
            continue
        session_key = _score_session_key(raw)
        seq_raw = raw.get("shard_seq")
        frame_raw = raw.get("frame_in_shard")
        scores_raw = raw.get("scores", raw.get("emission_scores", raw.get("matrix")))
        try:
            seq = int(seq_raw)
            frame = int(frame_raw)
        except (TypeError, ValueError):
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "score_load",
                    "shard_seq and frame_in_shard must be integers",
                    session_id=session_key or None,
                    details={"line": line_number},
                )
            )
            if seq_raw is not None:
                try:
                    invalid_shards.setdefault(session_key, set()).add(int(seq_raw))
                except (TypeError, ValueError):
                    pass
            continue
        if seq < 0 or frame < 0:
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "score_load",
                    "shard_seq and frame_in_shard must be non-negative",
                    session_id=session_key or None,
                    shard_seq=seq,
                    details={"line": line_number},
                )
            )
            invalid_shards.setdefault(session_key, set()).add(seq)
            continue
        if not isinstance(scores_raw, list):
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "score_load",
                    "scores must be a list with one value per vocabulary entry",
                    session_id=session_key or None,
                    shard_seq=seq,
                    details={"line": line_number},
                )
            )
            invalid_shards.setdefault(session_key, set()).add(seq)
            continue
        if len(scores_raw) != vocab_size:
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "score_validation",
                    "score vector dimension does not match vocabulary size",
                    session_id=session_key or None,
                    shard_seq=seq,
                    details={
                        "line": line_number,
                        "expected": vocab_size,
                        "actual": len(scores_raw),
                    },
                )
            )
            invalid_shards.setdefault(session_key, set()).add(seq)
            continue
        try:
            values = tuple(float(value) for value in scores_raw)
        except (TypeError, ValueError):
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "score_validation",
                    "score vector contains a non-numeric value",
                    session_id=session_key or None,
                    shard_seq=seq,
                    details={"line": line_number},
                )
            )
            invalid_shards.setdefault(session_key, set()).add(seq)
            continue
        if not all(math.isfinite(value) for value in values):
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "score_validation",
                    "score vector contains a non-finite value",
                    session_id=session_key or None,
                    shard_seq=seq,
                    details={"line": line_number},
                )
            )
            invalid_shards.setdefault(session_key, set()).add(seq)
            continue
        key = (session_key, seq, frame)
        if key in seen:
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "score_load",
                    "duplicate score row",
                    session_id=session_key or None,
                    shard_seq=seq,
                    details={"line": line_number, "frame_in_shard": frame},
                )
            )
            invalid_shards.setdefault(session_key, set()).add(seq)
            continue
        seen.add(key)
        scores.setdefault(session_key, {}).setdefault(seq, {})[frame] = values

    return scores, invalid_shards, failures


def load_sessions(
    path: str | Path,
) -> Tuple[Dict[str, SessionDataset], List[Dict[str, Any]]]:
    rows, failures = read_jsonl(path)
    sessions: Dict[str, SessionDataset] = {}
    per_session_order: Dict[str, List[int]] = {}
    duplicate_seen: Set[Tuple[str, int]] = set()
    invalid_sessions: Set[str] = set()

    for line_number, raw in rows:
        if not isinstance(raw, dict):
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "session_load",
                    "session shard row must be an object",
                    details={"line": line_number},
                )
            )
            continue
        session_id = raw.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "session_load",
                    "session_id must be a non-empty string",
                    session_id=session_id if isinstance(session_id, str) else None,
                    details={"line": line_number},
                )
            )
            continue
        try:
            seq = int(raw["shard_seq"])
            start_frame = int(raw["start_frame"])
            end_frame = int(raw["end_frame"])
        except (KeyError, TypeError, ValueError):
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "session_load",
                    "shard_seq, start_frame and end_frame must be integers",
                    session_id=session_id,
                    details={"line": line_number},
                )
            )
            invalid_sessions.add(session_id)
            continue
        arrived = raw.get("arrived", True)
        if not isinstance(arrived, bool):
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "session_load",
                    "arrived must be boolean when present",
                    session_id=session_id,
                    shard_seq=seq,
                    details={"line": line_number},
                )
            )
            invalid_sessions.add(session_id)
            continue
        if seq < 0 or start_frame < 0 or end_frame < start_frame:
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "session_load",
                    "shard sequence/frame bounds are invalid",
                    session_id=session_id,
                    shard_seq=seq,
                    details={"line": line_number},
                )
            )
            invalid_sessions.add(session_id)
            continue
        key = (session_id, seq)
        if key in duplicate_seen:
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "session_validation",
                    "duplicate shard sequence within a session",
                    session_id=session_id,
                    shard_seq=seq,
                    details={"line": line_number},
                )
            )
            invalid_sessions.add(session_id)
            continue
        duplicate_seen.add(key)
        session = sessions.setdefault(session_id, SessionDataset(session_id=session_id))
        session.shards[seq] = ShardRecord(
            session_id=session_id,
            shard_seq=seq,
            start_frame=start_frame,
            end_frame=end_frame,
            arrived=arrived,
            revision=0,
        )
        session.ingest_order.append(seq)
        per_session_order.setdefault(session_id, []).append(seq)

    for session_id in sorted(sessions):
        order = per_session_order.get(session_id, [])
        if order != sorted(order):
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "session_validation",
                    "shard sequence is not strictly increasing in stream order",
                    session_id=session_id,
                )
            )
            invalid_sessions.add(session_id)
        expected = list(range(len(order)))
        if sorted(order) != expected:
            failures.append(
                failure(
                    E_INPUT_INVALID,
                    "session_validation",
                    "shard sequence contains a gap",
                    session_id=session_id,
                    details={"expected": expected, "actual": sorted(order)},
                )
            )
            invalid_sessions.add(session_id)

    for session_id in invalid_sessions:
        if session_id in sessions:
            sessions[session_id].input_valid = False

    return sessions, failures


def merge_scores(
    sessions: Dict[str, SessionDataset],
    score_rows: Dict[str, Dict[int, Dict[int, Tuple[float, ...]]]],
    invalid_shards: Dict[str, Set[int]],
) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for session_id in sorted(sessions):
        session = sessions[session_id]
        session_scores = score_rows.get(session_id, {})
        global_scores = score_rows.get("__GLOBAL__", {})
        for seq in sorted(session.shards):
            if seq in invalid_shards.get(session_id, set()) or seq in invalid_shards.get("__GLOBAL__", set()):
                session.scores[seq] = {}
                continue
            source = session_scores.get(seq, global_scores.get(seq))
            if source is None:
                failures.append(
                    failure(
                        E_INPUT_INVALID,
                        "score_validation",
                        "no score rows found for shard",
                        session_id=session_id,
                        shard_seq=seq,
                    )
                )
                session.scores[seq] = {}
                continue
            session.scores[seq] = dict(source)
    return failures


def load_dataset(
    sample_dir: str | Path,
    *,
    vocab_filename: str = "vocab.json",
    scores_filename: str = "scores.jsonl",
    sessions_filename: str = "sessions.jsonl",
    config_filename: str = "decode_config.json",
    injection_filename: str = "injection_config.json",
) -> Tuple[Dataset, DecodeConfig, Dict[str, Any], List[Dict[str, Any]]]:
    root = Path(sample_dir)
    vocab = load_vocab(root / vocab_filename)
    config = load_config(root / config_filename)
    score_rows, invalid_shards, score_failures = load_scores(root / scores_filename, vocab)
    sessions, session_failures = load_sessions(root / sessions_filename)
    merge_failures = merge_scores(sessions, score_rows, invalid_shards)
    injections: Dict[str, Any] = {"items": []}
    injection_path = root / injection_filename
    if injection_path.exists():
        injections = load_injection_config(injection_path)
    failures = sort_failures(score_failures + session_failures + merge_failures)
    dataset = Dataset(
        vocab=vocab,
        sessions=sessions,
        score_source=str(root / scores_filename),
    )
    return dataset, config, injections, failures


def append_score_rows(
    dataset: Dataset,
    session_id: str,
    shard_seq: int,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    target: Dict[int, Tuple[float, ...]] = {}
    for row in rows:
        try:
            frame = int(row["frame_in_shard"])
            values = tuple(float(value) for value in row["scores"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AppError(
                E_INPUT_INVALID,
                f"invalid inline score row: {exc}",
                stage="command_validation",
                session_id=session_id,
                shard_seq=shard_seq,
            ) from exc
        target[frame] = values
    dataset.sessions[session_id].scores[shard_seq] = target


def load_extra_scores(path: str | Path, vocab: Vocabulary) -> List[Dict[str, Any]]:
    rows, failures = read_jsonl(path)
    if failures:
        first = failures[0]
        raise AppError(
            E_INPUT_INVALID,
            "extra score file contains malformed JSONL",
            stage="command_validation",
            details={"line": first.get("details", {}).get("line")},
        )
    result: List[Dict[str, Any]] = []
    for _, raw in rows:
        if not isinstance(raw, dict):
            raise AppError(E_INPUT_INVALID, "extra score row must be an object", stage="command_validation")
        values = raw.get("scores")
        if not isinstance(values, list) or len(values) != len(vocab):
            raise AppError(
                E_INPUT_INVALID,
                "extra score vector dimension does not match vocabulary",
                stage="command_validation",
            )
        result.append(
            {
                "frame_in_shard": int(raw["frame_in_shard"]),
                "scores": [float(value) for value in values],
            }
        )
    return result
