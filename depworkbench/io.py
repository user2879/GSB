from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .canonical import sha256_json
from .errors import ValidationError
from .models import (
    ACTION_NAMES,
    Config,
    Failure,
    Injection,
    Model,
    Project,
    Sentence,
    config_from_dict,
    model_from_dict,
    normalize_action_name,
    sentence_from_dict,
)


DEFAULT_SENTENCE_FILE = "sentences.jsonl"
DEFAULT_MODEL_FILE = "model.json"
DEFAULT_CONFIG_FILE = "config.json"
DEFAULT_VOCAB_FILE = "vocabulary.json"
DEFAULT_INJECTION_FILE = "injections.json"
INJECTION_STAGES = {
    "sentence_load",
    "load",
    "parse",
    "transition",
    "arc_aggregate",
    "tree_output",
}
INJECTION_KINDS = {
    "empty_words",
    "unknown_pos",
    "nonfinite_weight",
    "zero_max_steps",
}


def read_json(path: str | Path) -> Any:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ValidationError("E_IO", f"file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(
            "E_IO",
            f"invalid JSON in {path}: line {exc.lineno}, column {exc.colno}",
        ) from exc
    except OSError as exc:
        raise ValidationError("E_IO", f"cannot read {path}: {exc}") from exc


def write_json_atomic(path: str | Path, value: Any, *, indent: int | None = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=indent,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_json(path: str | Path, value: Any, *, indent: int | None = 2) -> None:
    write_json_atomic(path, value, indent=indent)


def write_jsonl_atomic(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _read_vocabulary(path: Path) -> tuple[list[str], list[str]]:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValidationError("E_VOCAB_INVALID", "vocabulary JSON must be an object")
    words_value = raw.get("words", raw.get("vocabulary"))
    pos_value = raw.get("pos_tags", raw.get("pos"))
    if isinstance(words_value, dict):
        words_value = list(words_value.keys())
    if isinstance(pos_value, dict):
        pos_value = list(pos_value.keys())
    if not isinstance(words_value, list) or not all(
        isinstance(item, str) and item for item in words_value
    ):
        raise ValidationError("E_VOCAB_INVALID", "vocabulary.words must be a string list")
    if not isinstance(pos_value, list) or not all(
        isinstance(item, str) and item for item in pos_value
    ):
        raise ValidationError("E_VOCAB_INVALID", "vocabulary.pos_tags must be a string list")
    if len(set(words_value)) != len(words_value):
        raise ValidationError("E_VOCAB_INVALID", "vocabulary words must be unique")
    if len(set(pos_value)) != len(pos_value):
        raise ValidationError("E_VOCAB_INVALID", "vocabulary pos_tags must be unique")
    return list(words_value), list(pos_value)


def _resolve_path(sample_dir: Path, value: str | Path | None, default_name: str) -> Path:
    if value is None:
        return sample_dir / default_name
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return sample_dir / path
    return path


def _load_sentence_file(
    path: Path,
    vocabulary: set[str],
    pos_tags: set[str],
) -> tuple[list[Sentence], list[Failure], list[dict[str, Any]]]:
    failures: list[Failure] = []
    candidates: list[tuple[int, dict[str, Any], Sentence | None, Failure | None]] = []
    raw_rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    failure = Failure(
                        sentence_id=f"__line_{line_no:06d}",
                        code="E_SENT_INVALID",
                        message=f"invalid JSON: {exc.msg}",
                        line_no=line_no,
                    )
                    failures.append(failure)
                    candidates.append((line_no, {}, None, failure))
                    continue
                if not isinstance(obj, dict):
                    failure = Failure(
                        sentence_id=f"__line_{line_no:06d}",
                        code="E_SENT_INVALID",
                        message="sentence row must be a JSON object",
                        line_no=line_no,
                    )
                    failures.append(failure)
                    candidates.append((line_no, {}, None, failure))
                    continue
                raw_rows.append(obj)
                try:
                    sentence = sentence_from_dict(obj, line_no)
                    _validate_sentence_tokens(sentence, vocabulary, pos_tags)
                    candidates.append((line_no, obj, sentence, None))
                except ValidationError as exc:
                    sid = obj.get("sentence_id", obj.get("id", obj.get("sid")))
                    failure = Failure(
                        sentence_id=sid if isinstance(sid, str) and sid else f"__line_{line_no:06d}",
                        code="E_SENT_INVALID",
                        message=exc.message,
                        line_no=line_no,
                    )
                    failures.append(failure)
                    candidates.append((line_no, obj, None, failure))
    except FileNotFoundError as exc:
        raise ValidationError("E_IO", f"sentence file not found: {path}") from exc
    except OSError as exc:
        raise ValidationError("E_IO", f"cannot read sentence file {path}: {exc}") from exc

    id_occurrences: dict[str, list[int]] = {}
    for index, (_, obj, sentence, _) in enumerate(candidates):
        sid: Any = None
        if sentence is not None:
            sid = sentence.sentence_id
        else:
            sid = obj.get("sentence_id", obj.get("id", obj.get("sid")))
        if isinstance(sid, str) and sid.strip():
            id_occurrences.setdefault(sid, []).append(index)
    duplicate_indexes: set[int] = set()
    for sid, indexes in id_occurrences.items():
        if len(indexes) > 1:
            duplicate_indexes.update(indexes)
            for index in indexes:
                line_no, _, _, old_failure = candidates[index]
                if old_failure is not None:
                    candidates[index] = (line_no, candidates[index][1], None, old_failure)
                    continue
                failure = Failure(
                    sentence_id=sid,
                    code="E_SENT_INVALID",
                    message=f"duplicate sentence id {sid!r}",
                    line_no=line_no,
                )
                failures.append(failure)
                candidates[index] = (line_no, candidates[index][1], None, failure)

    sentences: list[Sentence] = [
        sentence
        for index, (_, _, sentence, _) in enumerate(candidates)
        if sentence is not None and index not in duplicate_indexes
    ]
    sentences.sort(key=lambda sentence: sentence.sentence_id)
    failures.sort(key=lambda failure: failure.sort_key)
    raw_rows.sort(key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True))
    return sentences, failures, raw_rows


def _validate_sentence_tokens(
    sentence: Sentence,
    vocabulary: set[str],
    pos_tags: set[str],
) -> None:
    if not sentence.words:
        raise ValidationError("E_SENT_INVALID", "word sequence must not be empty")
    if len(sentence.words) != len(sentence.pos_tags):
        raise ValidationError("E_SENT_INVALID", "words and pos_tags lengths differ")
    for word in sentence.words:
        if word not in vocabulary:
            raise ValidationError("E_SENT_INVALID", f"word {word!r} is outside controlled vocabulary")
    for pos in sentence.pos_tags:
        if pos not in pos_tags:
            raise ValidationError("E_SENT_INVALID", f"pos tag {pos!r} is outside controlled POS set")
    if sentence.gold_heads is not None:
        if len(sentence.gold_heads) != len(sentence.words):
            raise ValidationError("E_SENT_INVALID", "gold_heads length differs from words length")
        for position, head in enumerate(sentence.gold_heads):
            if head < -1 or head >= len(sentence.words) or head == position:
                raise ValidationError("E_SENT_INVALID", "gold_heads contains an invalid head")
        if sum(1 for head in sentence.gold_heads if head == -1) != 1:
            raise ValidationError("E_SENT_INVALID", "gold_heads must contain exactly one root")
    if sentence.gold_relations is not None and len(sentence.gold_relations) != len(sentence.words):
        raise ValidationError("E_SENT_INVALID", "gold_relations length differs from words length")


def load_injections(path: str | Path | None) -> tuple[Injection, ...]:
    if path is None or not Path(path).exists():
        return ()
    raw = read_json(path)
    rows = raw.get("injections", []) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValidationError("E_INJECTION_INVALID", "injections must be a list")
    injections: list[Injection] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValidationError("E_INJECTION_INVALID", f"injection #{index} must be an object")
        item_id = row.get("injection_id", row.get("id", f"INJ-{index:03d}"))
        sid = row.get("sentence_id", row.get("sid"))
        stage = str(row.get("stage", "parse")).strip().lower()
        kind = str(row.get("kind", "")).strip().lower()
        if not isinstance(item_id, str) or not item_id:
            raise ValidationError("E_INJECTION_INVALID", "injection id must be non-empty")
        if item_id in seen:
            raise ValidationError("E_INJECTION_INVALID", f"duplicate injection id {item_id}")
        seen.add(item_id)
        if not isinstance(sid, str) or not sid:
            raise ValidationError("E_INJECTION_INVALID", f"injection {item_id} lacks sentence_id")
        if stage == "load":
            stage = "sentence_load"
        if stage == "transition":
            stage = "parse"
        if stage not in INJECTION_STAGES:
            raise ValidationError(
                "E_INJECTION_INVALID",
                f"injection {item_id} has unsupported stage {stage!r}",
            )
        if kind not in INJECTION_KINDS:
            raise ValidationError(
                "E_INJECTION_INVALID",
                f"injection {item_id} has unsupported kind {kind!r}",
            )
        try:
            token_index = int(row.get("token_index", row.get("position", 0)))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "E_INJECTION_INVALID", f"injection {item_id} token_index is invalid"
            ) from exc
        if token_index < 0:
            raise ValidationError("E_INJECTION_INVALID", f"injection {item_id} token_index is negative")
        try:
            weight_action = normalize_action_name(str(row.get("weight_action", "SHIFT")))
        except ValidationError as exc:
            raise ValidationError("E_INJECTION_INVALID", exc.message) from exc
        try:
            weight_index = int(row.get("weight_index", 0))
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "E_INJECTION_INVALID", f"injection {item_id} weight_index is invalid"
            ) from exc
        if weight_index < 0:
            raise ValidationError("E_INJECTION_INVALID", f"injection {item_id} weight_index is negative")
        injections.append(
            Injection(
                injection_id=item_id,
                sentence_id=sid,
                stage=stage,
                kind=kind,
                token_index=token_index,
                pos_tag=str(row.get("pos_tag", "__INJECTED_UNKNOWN_POS__")),
                weight_action=weight_action,
                weight_index=weight_index,
            )
        )
    injections.sort(key=lambda item: (item.sentence_id, item.injection_id))
    return tuple(injections)


def load_project(
    sample_dir: str | Path,
    *,
    sentence_file: str | Path | None = None,
    model_file: str | Path | None = None,
    config_file: str | Path | None = None,
    vocabulary_file: str | Path | None = None,
    injection_file: str | Path | None = None,
) -> Project:
    sample_path = Path(sample_dir).resolve()
    sentence_path = _resolve_path(sample_path, sentence_file, DEFAULT_SENTENCE_FILE)
    model_path = _resolve_path(sample_path, model_file, DEFAULT_MODEL_FILE)
    config_path = _resolve_path(sample_path, config_file, DEFAULT_CONFIG_FILE)
    vocab_path = _resolve_path(sample_path, vocabulary_file, DEFAULT_VOCAB_FILE)
    injection_path = _resolve_path(sample_path, injection_file, DEFAULT_INJECTION_FILE)

    model_raw = read_json(model_path)
    config_raw = read_json(config_path)
    if not isinstance(model_raw, dict):
        raise ValidationError("E_MODEL_INVALID", "model JSON must be an object")
    if not isinstance(config_raw, dict):
        raise ValidationError("E_CONFIG_INVALID", "config JSON must be an object")

    vocabulary: list[str] | None = None
    pos_tags: list[str] | None = None
    if vocab_path.exists():
        vocabulary, pos_tags = _read_vocabulary(vocab_path)
    model = model_from_dict(model_raw, vocabulary=vocabulary, pos_tags=pos_tags)
    config = config_from_dict(config_raw)
    if not model.vocabulary:
        raise ValidationError("E_VOCAB_INVALID", "controlled vocabulary is empty")
    if not model.pos_tags:
        raise ValidationError("E_VOCAB_INVALID", "controlled POS set is empty")

    vocab_set = set(model.vocabulary)
    pos_set = set(model.pos_tags)
    sentences, failures, raw_rows = _load_sentence_file(sentence_path, vocab_set, pos_set)
    injections = load_injections(injection_path if injection_path.exists() else None)
    corpus_fingerprint = sha256_json(
        {
            "sentences": raw_rows,
            "model": model.fingerprint,
            "config": config.fingerprint,
        }
    )
    return Project(
        sample_dir=sample_path,
        sentences=tuple(sentences),
        failures=tuple(failures),
        model=model,
        config=config,
        injections=injections,
        vocabulary=tuple(model.vocabulary),
        pos_tags=tuple(model.pos_tags),
        corpus_fingerprint=corpus_fingerprint,
        raw_sentences=tuple(raw_rows),
    )


def available_actions() -> tuple[str, ...]:
    return ACTION_NAMES
