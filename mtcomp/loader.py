from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .model import (
    CompileConfig,
    Failure,
    LexiconEntry,
    ScoreParams,
    SourceSentence,
    TranslationState,
    entry_key,
    stable_failures,
)

DEFAULT_FILES = {
    "sentences": "sentences.jsonl",
    "lexicon": "lexicon.tsv",
    "params": "params.json",
    "config": "compile.yaml",
    "edits": "edits.json",
    "injections": "injections.json",
}

FILE_RANK = {
    "compile.yaml": 0,
    "params.json": 1,
    "lexicon.tsv": 2,
    "sentences.jsonl": 3,
    "edits.json": 4,
    "injections.json": 5,
}


@dataclass
class InputBundle:
    state: TranslationState | None
    failures: list[Failure]
    commands: list[dict[str, Any]] = field(default_factory=list)
    injection_data: dict[str, Any] | None = None
    input_hashes: dict[str, str] = field(default_factory=dict)
    command_stream_hash: str = ""

    def sorted_failures(self) -> list[Failure]:
        return stable_failures(self.failures)


def _failure(
    code: str,
    phase: str,
    message: str,
    *,
    file: str,
    line: int = 0,
    sentence_id: str | None = None,
    entry: str | None = None,
    command_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> Failure:
    rank = FILE_RANK.get(file, 99)
    return Failure(
        code=code,
        phase=phase,
        message=message,
        file=file,
        line=line,
        sentence_id=sentence_id,
        entry_key=entry,
        command_id=command_id,
        details=details or {},
        order=(rank, line, code, sentence_id or "", entry or ""),
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 10)
        except ValueError:
            return None
    return None


def load_compile_config(path: Path, failures: list[Failure]) -> CompileConfig | None:
    file_name = path.name
    try:
        raw = yaml.safe_load(_read_text(path))
    except (OSError, yaml.YAMLError) as exc:
        failures.append(
            _failure(
                "E_CONFIG_INVALID",
                "load_config",
                f"cannot parse compile config: {exc}",
                file=file_name,
            )
        )
        return None
    if not isinstance(raw, dict):
        failures.append(
            _failure(
                "E_CONFIG_INVALID",
                "load_config",
                "compile config must be a mapping",
                file=file_name,
            )
        )
        return None

    def pick(*names: str, default: Any = None) -> Any:
        for name in names:
            if name in raw:
                return raw[name]
        return default

    ngram = _parse_int(pick("ngram_order", "n_gram_order"))
    max_batch = _parse_int(pick("max_sentences_per_batch", "batch_max_sentences"))
    concurrency = _parse_int(pick("thread_pool_concurrency", "concurrency"))
    budget = _parse_int(pick("decode_points_per_batch", "single_batch_decode_budget"))
    interval = _parse_int(
        pick("snapshot_commit_interval_batches", "snapshot_interval_batches")
    )
    max_tokens = _parse_int(pick("max_sentence_tokens", default=256))
    candidate_limit = _parse_int(pick("candidate_limit", default=8))
    vocabulary = pick("controlled_vocabulary", "vocabulary", default=[])

    missing = [
        name
        for name, value in (
            ("ngram_order", ngram),
            ("max_sentences_per_batch", max_batch),
            ("thread_pool_concurrency", concurrency),
            ("decode_points_per_batch", budget),
            ("snapshot_commit_interval_batches", interval),
            ("max_sentence_tokens", max_tokens),
            ("candidate_limit", candidate_limit),
        )
        if value is None
    ]
    if missing:
        failures.append(
            _failure(
                "E_CONFIG_INVALID",
                "load_config",
                f"missing or invalid fields: {', '.join(missing)}",
                file=file_name,
                details={"fields": missing},
            )
        )
        return None

    if isinstance(vocabulary, str):
        vocabulary = vocabulary.split()
    if not isinstance(vocabulary, (list, tuple)):
        failures.append(
            _failure(
                "E_CONFIG_INVALID",
                "load_config",
                "controlled_vocabulary must be a list or whitespace string",
                file=file_name,
            )
        )
        return None
    vocabulary_tuple = tuple(str(item) for item in vocabulary)
    if any(not item or any(ch.isspace() for ch in item) for item in vocabulary_tuple):
        failures.append(
            _failure(
                "E_CONFIG_INVALID",
                "load_config",
                "controlled_vocabulary entries must be non-empty tokens without whitespace",
                file=file_name,
            )
        )
        return None

    config = CompileConfig(
        ngram_order=int(ngram),
        max_sentences_per_batch=int(max_batch),
        thread_pool_concurrency=int(concurrency),
        decode_points_per_batch=int(budget),
        snapshot_commit_interval_batches=int(interval),
        max_sentence_tokens=int(max_tokens),
        vocabulary=vocabulary_tuple,
        candidate_limit=int(candidate_limit),
    )
    errors = config.validate()
    for error in errors:
        failures.append(
            _failure("E_CONFIG_INVALID", "load_config", error, file=file_name)
        )
    return config if not errors else None


def load_score_params(
    path: Path, failures: list[Failure]
) -> ScoreParams | None:
    file_name = path.name
    try:
        raw = json.loads(_read_text(path))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(
            _failure(
                "E_CONFIG_INVALID",
                "load_params",
                f"cannot parse scoring parameters: {exc}",
                file=file_name,
            )
        )
        return None
    if not isinstance(raw, dict):
        failures.append(
            _failure(
                "E_CONFIG_INVALID",
                "load_params",
                "scoring parameters must be a JSON object",
                file=file_name,
            )
        )
        return None
    try:
        params = ScoreParams.from_mapping(raw)
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(
            _failure(
                "E_CONFIG_INVALID",
                "load_params",
                f"invalid scoring parameter shape: {exc}",
                file=file_name,
            )
        )
        return None
    errors = params.validate()
    for error in errors:
        failures.append(
            _failure("E_CONFIG_INVALID", "load_params", error, file=file_name)
        )
    return params if not errors else None


def load_lexicon(
    path: Path, failures: list[Failure]
) -> dict[str, LexiconEntry]:
    file_name = path.name
    entries: dict[str, LexiconEntry] = {}
    try:
        lines = _read_text(path).splitlines()
    except OSError as exc:
        failures.append(
            _failure(
                "E_LEX_INVALID",
                "load_lexicon",
                f"cannot read lexicon: {exc}",
                file=file_name,
            )
        )
        return entries
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if len(parts) != 4:
            failures.append(
                _failure(
                    "E_LEX_INVALID",
                    "load_lexicon",
                    "lexicon line must contain source target frequency confidence",
                    file=file_name,
                    line=line_number,
                )
            )
            continue
        source, target, freq_text, conf_text = parts
        frequency = _finite_float(freq_text)
        confidence = _finite_float(conf_text)
        messages: list[str] = []
        if not source or not target:
            messages.append("source and target must be non-empty")
        if frequency is None or frequency <= 0:
            messages.append("frequency must be a positive finite number")
        if confidence is None or not 0 <= confidence <= 1:
            messages.append("alignment confidence must be finite and in [0, 1]")
        key = entry_key(source, target)
        if key in entries:
            messages.append("duplicate (source, target) lexicon entry")
        if messages:
            failures.append(
                _failure(
                    "E_LEX_INVALID",
                    "load_lexicon",
                    "; ".join(messages),
                    file=file_name,
                    line=line_number,
                    entry=key,
                    details={"source": source, "target": target},
                )
            )
            continue
        entries[key] = LexiconEntry(
            source=source,
            target=target,
            frequency=float(frequency),
            alignment_confidence=float(confidence),
        )
    return entries


def validate_sentence_payload(
    payload: Any,
    config: CompileConfig,
    *,
    existing_ids: Iterable[str] = (),
) -> tuple[SourceSentence | None, str | None, dict[str, Any]]:
    if not isinstance(payload, dict):
        return None, "sentence must be a JSON object", {}
    sentence_id = payload.get("id", payload.get("sentence_id"))
    batch_value = payload.get("batch", payload.get("batch_no"))
    tokens_value = payload.get("tokens")
    details: dict[str, Any] = {}
    if not isinstance(sentence_id, str) or not sentence_id:
        return None, "sentence id must be a non-empty string", details
    details["id"] = sentence_id
    if sentence_id in set(existing_ids):
        return None, "sentence id must be unique", details
    batch = _parse_int(batch_value)
    if batch is None or batch < 0:
        return None, "batch must be a non-negative integer", details
    details["batch"] = batch
    if isinstance(tokens_value, str):
        tokens = tuple(tokens_value.split())
    elif isinstance(tokens_value, list):
        if any(not isinstance(token, str) for token in tokens_value):
            return None, "tokens must all be strings", details
        tokens = tuple(tokens_value)
    else:
        return None, "tokens must be a whitespace string or string list", details
    if not tokens:
        return None, "sentence must contain at least one token", details
    if len(tokens) > config.max_sentence_tokens:
        return None, (
            f"sentence has {len(tokens)} tokens, exceeding max_sentence_tokens="
            f"{config.max_sentence_tokens}"
        ), details
    vocabulary = set(config.vocabulary)
    unknown = sorted({token for token in tokens if token not in vocabulary})
    if unknown:
        details["unknown_tokens"] = unknown
        return None, f"tokens outside controlled vocabulary: {unknown}", details
    return SourceSentence(sentence_id=sentence_id, tokens=tokens, batch=batch), None, details


def load_sentences(
    path: Path,
    config: CompileConfig,
    failures: list[Failure],
) -> dict[str, SourceSentence]:
    file_name = path.name
    sentences: dict[str, SourceSentence] = {}
    parsed: list[tuple[int, SourceSentence | None, Any]] = []
    try:
        lines = _read_text(path).splitlines()
    except OSError as exc:
        failures.append(
            _failure(
                "E_SENT_INVALID",
                "load_sentences",
                f"cannot read sentences: {exc}",
                file=file_name,
            )
        )
        return sentences
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            failures.append(
                _failure(
                    "E_SENT_INVALID",
                    "load_sentences",
                    f"invalid JSON: {exc.msg}",
                    file=file_name,
                    line=line_number,
                )
            )
            parsed.append((line_number, None, None))
            continue
        existing = set(sentences)
        sentence, error, details = validate_sentence_payload(
            payload, config, existing_ids=existing
        )
        if error:
            failures.append(
                _failure(
                    "E_SENT_INVALID",
                    "load_sentences",
                    error,
                    file=file_name,
                    line=line_number,
                    sentence_id=details.get("id"),
                    details=details,
                )
            )
            parsed.append((line_number, None, details.get("id")))
            continue
        assert sentence is not None
        sentences[sentence.sentence_id] = sentence
        parsed.append((line_number, sentence, sentence.sentence_id))

    # Apply the per-batch cardinality rule in stable file order. Only rows that
    # otherwise passed validation participate.
    by_batch: dict[int, list[tuple[int, SourceSentence]]] = {}
    for line_number, sentence, _ in parsed:
        if sentence is not None:
            by_batch.setdefault(sentence.batch, []).append((line_number, sentence))
    for batch, items in sorted(by_batch.items()):
        items.sort(key=lambda item: item[0])
        if len(items) <= config.max_sentences_per_batch:
            continue
        keep = {sentence.sentence_id for _, sentence in items[: config.max_sentences_per_batch]}
        for line_number, sentence in items[config.max_sentences_per_batch :]:
            if sentence.sentence_id in keep:
                continue
            failures.append(
                _failure(
                    "E_SENT_INVALID",
                    "load_sentences",
                    (
                        f"batch {batch} exceeds max_sentences_per_batch="
                        f"{config.max_sentences_per_batch}"
                    ),
                    file=file_name,
                    line=line_number,
                    sentence_id=sentence.sentence_id,
                    details={"batch": batch},
                )
            )
            sentences.pop(sentence.sentence_id, None)
    return sentences


def load_commands(
    path: Path, failures: list[Failure]
) -> tuple[list[dict[str, Any]], str]:
    file_name = path.name
    try:
        text = _read_text(path)
    except OSError as exc:
        failures.append(
            _failure(
                "E_EDIT_INVALID",
                "load_edits",
                f"cannot read edit stream: {exc}",
                file=file_name,
            )
        )
        return [], ""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        failures.append(
            _failure(
                "E_EDIT_INVALID",
                "load_edits",
                f"invalid edit stream JSON: {exc.msg}",
                file=file_name,
                line=exc.lineno,
            )
        )
        return [], ""
    if isinstance(raw, dict):
        commands = raw.get("commands", [])
    else:
        commands = raw
    if not isinstance(commands, list) or any(
        not isinstance(command, dict) for command in commands
    ):
        failures.append(
            _failure(
                "E_EDIT_INVALID",
                "load_edits",
                "edit stream must be a JSON array of objects or {commands: [...]}",
                file=file_name,
            )
        )
        return [], ""
    hashed = hashlib.sha256(
        json.dumps(commands, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return commands, hashed


def load_injection_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(_read_text(path))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def load_state(input_dir: str | Path) -> InputBundle:
    base = Path(input_dir)
    failures: list[Failure] = []
    paths = {name: base / relative for name, relative in DEFAULT_FILES.items()}
    input_hashes: dict[str, str] = {}
    for name in ("config", "params", "lexicon", "sentences", "edits"):
        path = paths[name]
        if path.exists():
            input_hashes[name] = file_sha256(path)

    config = load_compile_config(paths["config"], failures)
    params = load_score_params(paths["params"], failures)
    lexicon = load_lexicon(paths["lexicon"], failures)
    if config is not None:
        sentences = load_sentences(paths["sentences"], config, failures)
    else:
        sentences = {}
        if paths["sentences"].exists():
            failures.append(
                _failure(
                    "E_SENT_INVALID",
                    "load_sentences",
                    "cannot validate sentences because compile config is invalid",
                    file=paths["sentences"].name,
                )
            )
    commands, command_hash = load_commands(paths["edits"], failures)
    state = None
    if config is not None and params is not None:
        state = TranslationState(
            sentences=sentences,
            lexicon=lexicon,
            params=params,
            config=config,
        )
    return InputBundle(
        state=state,
        failures=failures,
        commands=commands,
        injection_data=load_injection_file(paths["injections"]),
        input_hashes=input_hashes,
        command_stream_hash=command_hash,
    )
