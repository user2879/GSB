"""Clip discovery and per-fragment truth/confusion label validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import RegistrationConfig
from .errors import ConfigFailure, DiagnosticFailure
from .jsonutil import parse_jsonl
from .wavio import scan_wav_files


def load_clip_labels(
    config: RegistrationConfig,
    failures: list[DiagnosticFailure],
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    clips = scan_wav_files(config.clip_dir)
    known_words = {word["command_word"] for word in config.words}
    labels_path = config.labels_path
    source = str(labels_path)
    try:
        text = labels_path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(ConfigFailure(source, None, "file", f"cannot read labels: {exc}").failure)
        return clips, {}
    rows = parse_jsonl(labels_path, text, failures)
    labels: dict[str, dict[str, Any]] = {}
    seen_clip_ids: set[str] = set()
    for line_no, row in rows:
        if not isinstance(row, dict):
            failures.append(ConfigFailure(source, line_no, "label", "expected object").failure)
            continue
        clip_id = row.get("clip_id")
        relative_path = row.get("path")
        true_command = row.get("true_command")
        confusion_with = row.get("confusion_with")
        kind = row.get("kind", "positive")
        valid = True
        if not isinstance(clip_id, str) or not clip_id.strip():
            failures.append(ConfigFailure(source, line_no, "clip_id", "expected non-empty string").failure)
            valid = False
        elif clip_id in seen_clip_ids:
            failures.append(
                ConfigFailure(source, line_no, "clip_id", f"duplicate clip id {clip_id!r}").failure
            )
            valid = False
        else:
            seen_clip_ids.add(clip_id)
        if not isinstance(relative_path, str) or not relative_path.strip():
            failures.append(ConfigFailure(source, line_no, "path", "expected non-empty string").failure)
            valid = False
        if not isinstance(true_command, str) or not true_command.strip():
            failures.append(ConfigFailure(source, line_no, "true_command", "expected non-empty string").failure)
            valid = False
        elif true_command != "NEGATIVE" and true_command not in known_words:
            failures.append(
                ConfigFailure(source, line_no, "true_command", f"unknown command word {true_command!r}").failure
            )
            valid = False
        if confusion_with is not None:
            if not isinstance(confusion_with, str) or not confusion_with.strip():
                failures.append(ConfigFailure(source, line_no, "confusion_with", "expected string or null").failure)
                valid = False
            elif confusion_with not in known_words:
                failures.append(
                    ConfigFailure(source, line_no, "confusion_with", f"unknown command word {confusion_with!r}").failure
                )
                valid = False
        if kind not in ("positive", "negative"):
            failures.append(ConfigFailure(source, line_no, "kind", "must be positive or negative").failure)
            valid = False
        if not valid:
            continue
        raw_candidate = (config.workspace / relative_path).resolve()
        try:
            normalized_path = raw_candidate.relative_to(config.workspace.resolve()).as_posix()
        except ValueError:
            normalized_path = Path(relative_path).as_posix()
        if normalized_path in labels:
            failures.append(
                ConfigFailure(source, line_no, "path", f"duplicate clip label for {normalized_path!r}").failure
            )
            continue
        labels[normalized_path] = {
            "clip_id": clip_id,
            "path": normalized_path,
            "true_command": true_command,
            "confusion_with": confusion_with,
            "kind": kind,
            "line": line_no,
        }

    clip_relatives = {path.relative_to(config.workspace.resolve()).as_posix(): path for path in clips}
    for relative_path in sorted(clip_relatives):
        if relative_path not in labels:
            failures.append(
                ConfigFailure(source, None, "path", f"missing clip label for {relative_path!r}").failure
            )
    for relative_path in sorted(labels):
        if relative_path not in clip_relatives:
            label = labels[relative_path]
            failures.append(
                ConfigFailure(source, label["line"], "path", f"label points to missing clip {relative_path!r}").failure
            )
    return clips, labels
