from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .audio import AudioFragment, read_wav
from .config import DiagnosticConfig
from .errors import AudioFormatError, Issue, SchemaError
from .receipts import (
    ReceiptDocument,
    compare_normalized_receipts,
    load_receipt_a1,
    load_receipt_a2,
)
from .util import ensure_relative_contained, json_hash, load_json


_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{where} must be an object", where=where)
    return value


def _require_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaError(f"{where} must be an array", where=where)
    return value


def _require_str(value: Any, where: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise SchemaError(f"{where} must be a non-empty string" if nonempty else f"{where} must be a string", where=where)
    return value


def _require_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{where} must be an integer", where=where)
    if minimum is not None and value < minimum:
        raise SchemaError(f"{where} must be >= {minimum}", where=where)
    return value


def _require_identifier(value: Any, where: str) -> str:
    text = _require_str(value, where)
    if not _ID_PATTERN.fullmatch(text):
        raise SchemaError(f"{where} must use ASCII letters, digits, '_', '.', ':' or '-'", where=where)
    return text


@dataclass(frozen=True)
class Speaker:
    speaker_id: str
    display_name: str
    trust_ppm: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "display_name": self.display_name,
            "trust_ppm": self.trust_ppm,
        }


@dataclass(frozen=True)
class CommandDef:
    code: int
    name: str
    phrase: str
    allowed_chars: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "phrase": self.phrase,
            "allowed_chars": list(self.allowed_chars),
        }


@dataclass(frozen=True)
class CharSpan:
    index: int
    char: str
    start_sample: int
    end_sample: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "char": self.char,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
        }


@dataclass(frozen=True)
class Interval:
    interval_id: str
    speaker_id: str
    start_sample: int
    end_sample: int
    fragment_id: str
    truth_command_code: int
    text: str
    chars: tuple[CharSpan, ...]

    @property
    def sample_count(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def char_offset_fingerprint_payload(self) -> dict[str, Any]:
        return {
            "interval_id": self.interval_id,
            "speaker_id": self.speaker_id,
            "text": self.text,
            "chars": [char.as_dict() for char in self.chars],
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "interval_id": self.interval_id,
            "speaker_id": self.speaker_id,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "fragment_id": self.fragment_id,
            "truth_command_code": self.truth_command_code,
            "text": self.text,
            "chars": [char.as_dict() for char in self.chars],
        }


@dataclass(frozen=True)
class FragmentDecl:
    fragment_id: str
    interval_id: str
    audio_file: str
    arrival_index: int
    capture_start_ms: int
    arrival_time_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "interval_id": self.interval_id,
            "audio_file": self.audio_file,
            "arrival_index": self.arrival_index,
            "capture_start_ms": self.capture_start_ms,
            "arrival_time_ms": self.arrival_time_ms,
        }


@dataclass(frozen=True)
class CommandStreamItem:
    turn_index: int
    interval_id: str
    command_code: int
    char_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "interval_id": self.interval_id,
            "command_code": self.command_code,
            "char_indices": list(self.char_indices),
        }


@dataclass(frozen=True)
class ReceiptPaths:
    a1: str | None
    a2: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"a1": self.a1, "a2": self.a2}


@dataclass(frozen=True)
class Manifest:
    path: Path
    root: Path
    conversation_id: str
    seed: int
    speakers: tuple[Speaker, ...]
    commands: tuple[CommandDef, ...]
    fragments: tuple[FragmentDecl, ...]
    intervals: tuple[Interval, ...]
    command_stream: tuple[CommandStreamItem, ...]
    receipt_paths: ReceiptPaths
    structure_hash: str

    @property
    def speaker_by_id(self) -> dict[str, Speaker]:
        return {speaker.speaker_id: speaker for speaker in self.speakers}

    @property
    def command_by_code(self) -> dict[int, CommandDef]:
        return {command.code: command for command in self.commands}

    @property
    def command_vocabulary(self) -> dict[int, str]:
        return {command.code: command.name for command in self.commands}

    @property
    def interval_by_id(self) -> dict[str, Interval]:
        return {interval.interval_id: interval for interval in self.intervals}

    @property
    def fragment_by_id(self) -> dict[str, FragmentDecl]:
        return {fragment.fragment_id: fragment for fragment in self.fragments}

    @property
    def timeline_end(self) -> int:
        return max((interval.end_sample for interval in self.intervals), default=0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "seed": self.seed,
            "speakers": [speaker.as_dict() for speaker in self.speakers],
            "commands": [command.as_dict() for command in self.commands],
            "fragments": [fragment.as_dict() for fragment in self.fragments],
            "intervals": [interval.as_dict() for interval in self.intervals],
            "command_stream": [item.as_dict() for item in self.command_stream],
            "receipt_paths": self.receipt_paths.as_dict(),
            "structure_hash": self.structure_hash,
        }


@dataclass(frozen=True)
class LoadedSession:
    config: DiagnosticConfig
    manifest: Manifest
    fragments: tuple[AudioFragment, ...]
    issues: tuple[Issue, ...]
    skipped_fragment_ids: tuple[str, ...]
    receipts_a1: ReceiptDocument | None
    receipts_a2: ReceiptDocument | None
    receipt_comparison: dict[str, Any] | None
    parse_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def fragment_by_interval(self) -> dict[str, AudioFragment]:
        return {fragment.interval_id: fragment for fragment in self.fragments}


def _parse_speakers(raw: Any) -> tuple[Speaker, ...]:
    speakers: list[Speaker] = []
    seen: set[str] = set()
    for index, value in enumerate(_require_list(raw, "manifest.speakers")):
        item = _require_mapping(value, f"speakers[{index}]")
        speaker_id = _require_identifier(item.get("id"), f"speakers[{index}].id")
        display_name = _require_str(item.get("display_name", speaker_id), f"speakers[{index}].display_name")
        trust_ppm = _require_int(item.get("trust_ppm", 850_000), f"speakers[{index}].trust_ppm", minimum=0)
        if trust_ppm > 1_000_000:
            raise SchemaError("speaker trust_ppm must be <= 1000000", where=f"speakers[{index}]")
        if speaker_id in seen:
            raise SchemaError("duplicate speaker id", where=f"speakers[{index}]", details={"speaker_id": speaker_id})
        seen.add(speaker_id)
        speakers.append(Speaker(speaker_id, display_name, trust_ppm))
    if not speakers:
        raise SchemaError("at least one speaker must be declared", where="manifest.speakers")
    return tuple(speakers)


def _parse_commands(raw: Any) -> tuple[CommandDef, ...]:
    commands: list[CommandDef] = []
    seen_codes: set[int] = set()
    seen_names: set[str] = set()
    for index, value in enumerate(_require_list(raw, "manifest.commands")):
        item = _require_mapping(value, f"commands[{index}]")
        code = _require_int(item.get("code"), f"commands[{index}].code", minimum=1)
        name = _require_str(item.get("name"), f"commands[{index}].name")
        phrase = _require_str(item.get("phrase"), f"commands[{index}].phrase")
        allowed_raw = _require_list(item.get("allowed_chars"), f"commands[{index}].allowed_chars")
        allowed_chars = tuple(_require_str(char, f"commands[{index}].allowed_chars[]") for char in allowed_raw)
        if code in seen_codes or name in seen_names:
            raise SchemaError("duplicate command code or name", where=f"commands[{index}]")
        if any(len(char) != 1 for char in allowed_chars):
            raise SchemaError("allowed_chars entries must be single characters", where=f"commands[{index}]")
        if len(set(allowed_chars)) != len(allowed_chars):
            raise SchemaError("allowed_chars contains duplicates", where=f"commands[{index}]")
        if any(char not in set(allowed_chars) for char in phrase):
            raise SchemaError(
                "command phrase uses characters outside its controlled character table",
                where=f"commands[{index}].phrase",
            )
        seen_codes.add(code)
        seen_names.add(name)
        commands.append(CommandDef(code, name, phrase, allowed_chars))
    if not commands:
        raise SchemaError("at least one command must be declared", where="manifest.commands")
    return tuple(commands)


def _parse_fragments(raw: Any) -> tuple[FragmentDecl, ...]:
    fragments: list[FragmentDecl] = []
    seen_ids: set[str] = set()
    seen_arrival: set[int] = set()
    for index, value in enumerate(_require_list(raw, "manifest.fragments")):
        item = _require_mapping(value, f"fragments[{index}]")
        fragment_id = _require_identifier(item.get("id"), f"fragments[{index}].id")
        interval_id = _require_identifier(item.get("interval_id"), f"fragments[{index}].interval_id")
        audio_file = _require_str(item.get("audio_file"), f"fragments[{index}].audio_file")
        arrival_index = _require_int(item.get("arrival_index"), f"fragments[{index}].arrival_index", minimum=0)
        capture_start_ms = _require_int(
            item.get("capture_start_ms"), f"fragments[{index}].capture_start_ms", minimum=0
        )
        arrival_time_ms = _require_int(
            item.get("arrival_time_ms"), f"fragments[{index}].arrival_time_ms", minimum=0
        )
        if arrival_time_ms < capture_start_ms:
            raise SchemaError("arrival_time_ms must not precede capture_start_ms", where=f"fragments[{index}]")
        if fragment_id in seen_ids:
            raise SchemaError("duplicate fragment id", where=f"fragments[{index}]")
        if arrival_index in seen_arrival:
            raise SchemaError("duplicate arrival_index", where=f"fragments[{index}]")
        seen_ids.add(fragment_id)
        seen_arrival.add(arrival_index)
        fragments.append(
            FragmentDecl(
                fragment_id=fragment_id,
                interval_id=interval_id,
                audio_file=audio_file,
                arrival_index=arrival_index,
                capture_start_ms=capture_start_ms,
                arrival_time_ms=arrival_time_ms,
            )
        )
    if not fragments:
        raise SchemaError("at least one audio fragment must be declared", where="manifest.fragments")
    return tuple(fragments)


def _parse_intervals(
    raw: Any,
    *,
    speaker_ids: set[str],
    command_by_code: Mapping[int, CommandDef],
    fragment_by_interval: Mapping[str, FragmentDecl],
) -> tuple[Interval, ...]:
    intervals: list[Interval] = []
    seen: set[str] = set()
    for index, value in enumerate(_require_list(raw, "manifest.intervals")):
        item = _require_mapping(value, f"intervals[{index}]")
        interval_id = _require_identifier(item.get("id"), f"intervals[{index}].id")
        speaker_id = _require_identifier(item.get("speaker_id"), f"intervals[{index}].speaker_id")
        start_sample = _require_int(item.get("start_sample"), f"intervals[{index}].start_sample", minimum=0)
        end_sample = _require_int(item.get("end_sample"), f"intervals[{index}].end_sample", minimum=1)
        truth_code = _require_int(
            item.get("truth_command_code"), f"intervals[{index}].truth_command_code", minimum=1
        )
        text = _require_str(item.get("text"), f"intervals[{index}].text")
        if interval_id in seen:
            raise SchemaError("duplicate interval id", where=f"intervals[{index}]", details={"interval_id": interval_id})
        seen.add(interval_id)
        if speaker_id not in speaker_ids:
            raise SchemaError(
                "interval speaker is not declared",
                where=f"intervals[{index}].speaker_id",
                details={"speaker_id": speaker_id},
            )
        if end_sample <= start_sample:
            raise SchemaError("interval end_sample must be greater than start_sample", where=f"intervals[{index}]")
        if truth_code not in command_by_code:
            raise SchemaError("truth command is not declared", where=f"intervals[{index}].truth_command_code")
        if interval_id not in fragment_by_interval:
            raise SchemaError("interval has no declared audio fragment", where=f"intervals[{index}]")
        command = command_by_code[truth_code]
        allowed = set(command.allowed_chars)
        if any(char not in allowed for char in text):
            raise SchemaError(
                "interval transcript uses characters outside the controlled command character table",
                where=f"intervals[{index}].text",
                details={"text": text},
            )
        chars_raw = _require_list(item.get("chars"), f"intervals[{index}].chars")
        if "".join(_require_str(char_item.get("char"), f"intervals[{index}].chars[].char") for char_item in [_require_mapping(v, f"intervals[{index}].chars[]") for v in chars_raw]) != text:
            raise SchemaError("interval chars do not concatenate to text", where=f"intervals[{index}].chars")
        chars: list[CharSpan] = []
        previous_start = -1
        previous_end = -1
        duration = end_sample - start_sample
        for char_index, char_raw in enumerate(chars_raw):
            char_item = _require_mapping(char_raw, f"intervals[{index}].chars[{char_index}]")
            char_value = _require_str(char_item.get("char"), f"intervals[{index}].chars[{char_index}].char")
            declared_index = char_item.get("index", char_index)
            if declared_index != char_index:
                raise SchemaError("char index must be contiguous and zero based", where=f"intervals[{index}].chars[{char_index}]")
            char_start = _require_int(
                char_item.get("start_sample"),
                f"intervals[{index}].chars[{char_index}].start_sample",
                minimum=0,
            )
            char_end = _require_int(
                char_item.get("end_sample"),
                f"intervals[{index}].chars[{char_index}].end_sample",
                minimum=0,
            )
            if char_start > char_end:
                raise SchemaError("char start_sample must be <= end_sample", where=f"intervals[{index}].chars[{char_index}]")
            if char_end > duration:
                raise SchemaError(
                    "char offset exceeds interval sample range",
                    where=f"intervals[{index}].chars[{char_index}]",
                    details={"duration": duration, "end_sample": char_end},
                )
            if char_start < previous_start or char_end < previous_end:
                raise SchemaError("char offsets must be monotonically non-decreasing", where=f"intervals[{index}].chars[{char_index}]")
            previous_start = char_start
            previous_end = char_end
            chars.append(CharSpan(char_index, char_value, char_start, char_end))
        if not chars:
            raise SchemaError("interval must contain at least one character span", where=f"intervals[{index}]")
        intervals.append(
            Interval(
                interval_id=interval_id,
                speaker_id=speaker_id,
                start_sample=start_sample,
                end_sample=end_sample,
                fragment_id=fragment_by_interval[interval_id].fragment_id,
                truth_command_code=truth_code,
                text=text,
                chars=tuple(chars),
            )
        )
    if not intervals:
        raise SchemaError("at least one interval must be declared", where="manifest.intervals")
    return tuple(intervals)


def _parse_command_stream(
    raw: Any,
    *,
    interval_by_id: Mapping[str, Interval],
    command_by_code: Mapping[int, CommandDef],
) -> tuple[CommandStreamItem, ...]:
    items: list[CommandStreamItem] = []
    seen_turns: set[int] = set()
    for index, value in enumerate(_require_list(raw, "manifest.command_stream")):
        item = _require_mapping(value, f"command_stream[{index}]")
        turn_index = _require_int(item.get("turn_index"), f"command_stream[{index}].turn_index", minimum=0)
        interval_id = _require_identifier(item.get("interval_id"), f"command_stream[{index}].interval_id")
        command_code = _require_int(item.get("command_code"), f"command_stream[{index}].command_code", minimum=1)
        char_indices_raw = _require_list(item.get("char_indices"), f"command_stream[{index}].char_indices")
        if turn_index in seen_turns:
            raise SchemaError("duplicate command-stream turn index", where=f"command_stream[{index}]")
        seen_turns.add(turn_index)
        interval = interval_by_id.get(interval_id)
        if interval is None:
            raise SchemaError(
                "command stream references an unknown interval",
                where=f"command_stream[{index}].interval_id",
                details={"interval_id": interval_id},
            )
        if command_code not in command_by_code:
            raise SchemaError("command stream references an unknown command", where=f"command_stream[{index}].command_code")
        char_indices: list[int] = []
        for char_value in char_indices_raw:
            char_index = _require_int(char_value, f"command_stream[{index}].char_indices[]", minimum=0)
            if char_index >= len(interval.chars):
                raise SchemaError(
                    "command stream references a missing character item",
                    where=f"command_stream[{index}].char_indices[]",
                    details={"interval_id": interval_id, "char_index": char_index},
                )
            char_indices.append(char_index)
        if not char_indices:
            raise SchemaError("command stream entry must reference at least one character", where=f"command_stream[{index}]")
        items.append(CommandStreamItem(turn_index, interval_id, command_code, tuple(char_indices)))
    return tuple(sorted(items, key=lambda value: value.turn_index))


def _parse_receipt_paths(raw: Any) -> ReceiptPaths:
    if raw is None:
        return ReceiptPaths(None, None)
    item = _require_mapping(raw, "manifest.receipts")
    a1 = item.get("a1")
    a2 = item.get("a2")
    if a1 is not None:
        a1 = _require_str(a1, "manifest.receipts.a1")
    if a2 is not None:
        a2 = _require_str(a2, "manifest.receipts.a2")
    return ReceiptPaths(a1, a2)


def load_manifest(path: str | Path, config: DiagnosticConfig) -> Manifest:
    manifest_path = Path(path).resolve()
    root = manifest_path.parent
    try:
        raw = load_json(manifest_path)
    except FileNotFoundError as exc:
        raise SchemaError("manifest file does not exist", where=str(manifest_path)) from exc
    except Exception as exc:
        raise SchemaError(f"cannot parse manifest JSON: {exc}", where=str(manifest_path)) from exc
    item = _require_mapping(raw, "manifest")
    conversation_id = _require_str(item.get("conversation_id"), "manifest.conversation_id")
    seed = _require_int(item.get("seed"), "manifest.seed")
    if seed != config.seed:
        raise SchemaError("manifest seed must equal config seed", where="manifest.seed")
    speakers = _parse_speakers(item.get("speakers"))
    commands = _parse_commands(item.get("commands"))
    command_by_code = {command.code: command for command in commands}
    fragments = _parse_fragments(item.get("fragments"))
    fragment_by_interval: dict[str, FragmentDecl] = {}
    for fragment in fragments:
        if fragment.interval_id in fragment_by_interval:
            raise SchemaError("multiple fragments reference the same interval", where=f"fragment:{fragment.fragment_id}")
        fragment_by_interval[fragment.interval_id] = fragment
    intervals = _parse_intervals(
        item.get("intervals"),
        speaker_ids={speaker.speaker_id for speaker in speakers},
        command_by_code=command_by_code,
        fragment_by_interval=fragment_by_interval,
    )
    interval_by_id = {interval.interval_id: interval for interval in intervals}
    for fragment in fragments:
        if fragment.interval_id not in interval_by_id:
            raise SchemaError(
                "fragment references an unknown interval",
                where=f"fragment:{fragment.fragment_id}",
            )
        interval = interval_by_id[fragment.interval_id]
        # A local fragment covers exactly the interval's global sample range.
        if not isinstance(fragment.capture_start_ms, int):
            raise SchemaError("invalid capture annotation")
    command_stream = _parse_command_stream(
        item.get("command_stream"), interval_by_id=interval_by_id, command_by_code=command_by_code
    )
    receipt_paths = _parse_receipt_paths(item.get("receipts"))
    normalized = {
        "conversation_id": conversation_id,
        "seed": seed,
        "speakers": [speaker.as_dict() for speaker in speakers],
        "commands": [command.as_dict() for command in commands],
        "fragments": [fragment.as_dict() for fragment in fragments],
        "intervals": [interval.as_dict() for interval in intervals],
        "command_stream": [entry.as_dict() for entry in command_stream],
        "receipts": receipt_paths.as_dict(),
    }
    return Manifest(
        path=manifest_path,
        root=root,
        conversation_id=conversation_id,
        seed=seed,
        speakers=speakers,
        commands=commands,
        fragments=fragments,
        intervals=intervals,
        command_stream=command_stream,
        receipt_paths=receipt_paths,
        structure_hash=json_hash(normalized),
    )


def _load_receipts(
    manifest: Manifest,
    config: DiagnosticConfig,
) -> tuple[ReceiptDocument | None, ReceiptDocument | None, dict[str, Any] | None]:
    vocab = manifest.command_vocabulary
    a1_doc: ReceiptDocument | None = None
    a2_doc: ReceiptDocument | None = None
    interval_ids = set(manifest.interval_by_id)
    if manifest.receipt_paths.a1:
        try:
            path = ensure_relative_contained(manifest.root, manifest.receipt_paths.a1)
        except ValueError as exc:
            raise SchemaError(str(exc), where="manifest.receipts.a1") from exc
        a1_doc = load_receipt_a1(path, vocabulary=vocab)
        _validate_receipt_interval_refs(a1_doc, manifest, "a1")
    if manifest.receipt_paths.a2:
        try:
            path = ensure_relative_contained(manifest.root, manifest.receipt_paths.a2)
        except ValueError as exc:
            raise SchemaError(str(exc), where="manifest.receipts.a2") from exc
        a2_doc = load_receipt_a2(
            path,
            vocabulary=vocab,
            allowed_quant_schemes=config.receipt.quant_schemes,
        )
        _validate_receipt_interval_refs(a2_doc, manifest, "a2")
    comparison = None
    if a1_doc is not None and a2_doc is not None:
        comparison = compare_normalized_receipts(a1_doc, a2_doc)
        if not comparison["ok"]:
            raise SchemaError(
                "a1 and a2 receipts do not normalize to the same interval evidence",
                where="manifest.receipts",
                details={
                    "missing_in_a1": comparison["missing_in_a1"],
                    "missing_in_a2": comparison["missing_in_a2"],
                    "mismatches": comparison["mismatches"],
                },
            )
    for version, document in (("a1", a1_doc), ("a2", a2_doc)):
        if document is None:
            continue
        declared = set(manifest.interval_by_id)
        actual = {record.interval_id for record in document.records}
        if actual != declared:
            raise SchemaError(
                f"{version} receipt does not cover exactly the declared intervals",
                where=f"manifest.receipts.{version}",
                details={"missing": sorted(declared - actual), "extra": sorted(actual - declared)},
            )
    return a1_doc, a2_doc, comparison


def _validate_receipt_interval_refs(document: ReceiptDocument, manifest: Manifest, version: str) -> None:
    intervals = manifest.interval_by_id
    for record in document.records:
        interval = intervals.get(record.interval_id)
        if interval is None:
            raise SchemaError(
                f"{version} receipt references an unknown interval",
                details={"interval_id": record.interval_id},
            )
        if record.speaker_id != interval.speaker_id:
            raise SchemaError(
                f"{version} receipt speaker does not match the interval",
                details={"interval_id": record.interval_id},
            )


def load_session(
    config_path: str | Path,
    manifest_path: str | Path,
    config: DiagnosticConfig,
    *,
    arrival_limit: int | None = None,
) -> LoadedSession:
    manifest = load_manifest(manifest_path, config)
    interval_by_id = manifest.interval_by_id
    loaded: list[AudioFragment] = []
    issues: list[Issue] = []
    skipped: list[str] = []
    for declaration in manifest.fragments:
        if arrival_limit is not None and declaration.arrival_index >= arrival_limit:
            continue
        interval = interval_by_id[declaration.interval_id]
        try:
            audio_path = ensure_relative_contained(manifest.root, declaration.audio_file)
        except ValueError as exc:
            raise SchemaError(
                "audio_file escapes the manifest workspace",
                where=f"fragment:{declaration.fragment_id}",
            ) from exc
        try:
            wav = read_wav(
                audio_path,
                expected_samples=interval.sample_count,
                where=str(audio_path),
            )
        except AudioFormatError as exc:
            issues.append(exc.as_issue())
            skipped.append(declaration.fragment_id)
            continue
        loaded.append(
            AudioFragment(
                fragment_id=declaration.fragment_id,
                interval_id=declaration.interval_id,
                speaker_id=interval.speaker_id,
                path=audio_path,
                global_start=interval.start_sample,
                global_end=interval.end_sample,
                arrival_index=declaration.arrival_index,
                capture_start_ms=declaration.capture_start_ms,
                arrival_time_ms=declaration.arrival_time_ms,
                wav=wav,
            )
        )
    a1_doc, a2_doc, comparison = _load_receipts(manifest, config)
    valid_ids = {fragment.fragment_id for fragment in loaded}
    all_arrival_indices = sorted(fragment.arrival_index for fragment in manifest.fragments)
    total_arrivals = len(all_arrival_indices)
    cursor = total_arrivals
    if arrival_limit is not None:
        cursor = max(0, min(int(arrival_limit), total_arrivals))
    loaded_arrival_indices = sorted(fragment.arrival_index for fragment in loaded)
    completed_arrival = cursor
    all_arrived = cursor >= total_arrivals
    parse_summary = {
        "conversation_id": manifest.conversation_id,
        "structure_hash": manifest.structure_hash,
        "config_hash": config.config_hash,
        "interval_count": len(manifest.intervals),
        "fragment_count": len(manifest.fragments),
        "valid_fragment_count": len(loaded),
        "skipped_fragment_ids": skipped,
        "fragment_cursor": completed_arrival,
        "arrival_cursor": completed_arrival,
        "total_fragment_count": total_arrivals,
        "all_fragments_arrived": all_arrived,
        "loaded_fragment_ids": sorted(valid_ids),
        "speaker_count": len(manifest.speakers),
        "command_count": len(manifest.commands),
        "command_stream_count": len(manifest.command_stream),
        "receipt_a1_count": len(a1_doc.records) if a1_doc is not None else 0,
        "receipt_a2_count": len(a2_doc.records) if a2_doc is not None else 0,
        "receipt_comparison_ok": comparison["ok"] if comparison is not None else None,
        "audio_issues": [issue.as_dict() for issue in issues],
    }
    return LoadedSession(
        config=config,
        manifest=manifest,
        fragments=tuple(loaded),
        issues=tuple(issues),
        skipped_fragment_ids=tuple(skipped),
        receipts_a1=a1_doc,
        receipts_a2=a2_doc,
        receipt_comparison=comparison,
        parse_summary=parse_summary,
    )
