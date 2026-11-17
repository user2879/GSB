"""Interactive, command-driven edits with automatic incremental execution."""

from __future__ import annotations

import copy
import json
import shlex
import sys
import threading
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json
from .cli import _SignalGuard, print_summary
from .data import read_structured_file
from .pipeline import run_pipeline


def _tasks_list(document: Any) -> list[Any]:
    if isinstance(document, list):
        return document
    return document.setdefault("tasks", [])


def _find_task(document: Any, name: str) -> dict[str, Any] | None:
    for task in _tasks_list(document):
        if isinstance(task, dict) and task.get("name") == name:
            return task
    return None


def _samples_mapping(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("feature document must be an object")
    samples = document.get("samples")
    if not isinstance(samples, dict):
        raise ValueError("feature document.samples must be an object")
    return samples


def _write_workspace(directory: str, documents: dict[str, Any]) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    mapping = {
        "tasks.json": documents["tasks"],
        "features.json": documents["features"],
        "encoder.json": documents["encoder"],
        "adapter_config.json": documents["config"],
        "injections.json": documents["injections"],
    }
    for filename, value in mapping.items():
        if value is not None:
            atomic_write_json(target / filename, value)


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def run_repl(args: Any) -> int:
    documents = {
        "tasks": copy.deepcopy(read_structured_file(args.tasks)),
        "features": copy.deepcopy(read_structured_file(args.features)),
        "encoder": copy.deepcopy(read_structured_file(args.encoder)),
        "config": copy.deepcopy(read_structured_file(args.config)),
        "injections": copy.deepcopy(
            read_structured_file(args.injections) if args.injections else None
        ),
    }
    stop_event = threading.Event()

    def execute() -> int:
        stop_event.clear()
        with _SignalGuard(stop_event):
            result = run_pipeline(
                documents["tasks"],
                documents["features"],
                documents["encoder"],
                documents["config"],
                state_directory=args.state_dir,
                injection_document=documents["injections"],
                report_path=args.report,
                stop_event=stop_event,
            )
        print_summary(dict(result.report))
        return result.exit_code

    def maybe_execute() -> None:
        if not args.manual:
            execute()

    print("transfer-pipeline repl; type 'help' for commands", file=sys.stderr)
    while True:
        try:
            line = input("transfer> ")
        except EOFError:
            print(file=sys.stderr)
            return 0
        except KeyboardInterrupt:
            print("\nuse 'quit' to exit", file=sys.stderr)
            continue
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            print(f"parse error: {exc}", file=sys.stderr)
            continue
        if not parts:
            continue
        command = parts[0].lower()
        try:
            if command in {"quit", "exit"}:
                return 0
            if command == "help":
                print(
                    "run | show | add-task FILE | delete-task NAME | "
                    "set-classes NAME c1,c2 | replace-sample FILE | "
                    "set-hyper TARGET KEY VALUE | save DIR | quit"
                )
                continue
            if command == "run":
                execute()
                continue
            if command == "show":
                names = [
                    task.get("name")
                    for task in _tasks_list(documents["tasks"])
                    if isinstance(task, dict)
                ]
                print(
                    json.dumps(
                        {
                            "tasks": names,
                            "sample_count": len(_samples_mapping(documents["features"])),
                            "config": documents["config"],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                continue
            if command == "add-task":
                if len(parts) != 2:
                    raise ValueError("usage: add-task FILE")
                loaded = read_structured_file(parts[1])
                if isinstance(loaded, dict) and isinstance(loaded.get("tasks"), list):
                    if len(loaded["tasks"]) != 1:
                        raise ValueError("add-task file must contain exactly one task")
                    task = copy.deepcopy(loaded["tasks"][0])
                else:
                    task = copy.deepcopy(loaded)
                _tasks_list(documents["tasks"]).append(task)
                maybe_execute()
                continue
            if command == "delete-task":
                if len(parts) != 2:
                    raise ValueError("usage: delete-task NAME")
                raw_tasks = _tasks_list(documents["tasks"])
                before = len(raw_tasks)
                raw_tasks[:] = [
                    task
                    for task in raw_tasks
                    if not (isinstance(task, dict) and task.get("name") == parts[1])
                ]
                if len(raw_tasks) == before:
                    raise ValueError(f"task {parts[1]!r} not found")
                maybe_execute()
                continue
            if command == "set-classes":
                if len(parts) != 3:
                    raise ValueError("usage: set-classes NAME class1,class2,...")
                task = _find_task(documents["tasks"], parts[1])
                if task is None:
                    raise ValueError(f"task {parts[1]!r} not found")
                task["classes"] = [value for value in parts[2].split(",") if value]
                maybe_execute()
                continue
            if command == "replace-sample":
                if len(parts) != 2:
                    raise ValueError("usage: replace-sample FILE")
                replacement = read_structured_file(parts[1])
                sample_id = replacement.get("sample_id", replacement.get("id"))
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError("replacement file needs sample_id")
                samples = _samples_mapping(documents["features"])
                if sample_id not in samples:
                    raise ValueError(f"sample {sample_id!r} does not exist")
                if "features" not in replacement:
                    raise ValueError("replacement file needs features")
                samples[sample_id]["features"] = copy.deepcopy(replacement["features"])
                if "label" in replacement:
                    samples[sample_id]["label"] = replacement["label"]
                maybe_execute()
                continue
            if command == "set-hyper":
                if len(parts) != 4:
                    raise ValueError("usage: set-hyper TASK|* KEY VALUE")
                target, key, raw_value = parts[1], parts[2], _parse_value(parts[3])
                config = documents["config"]
                if target == "*":
                    config[key] = raw_value
                else:
                    overrides = config.setdefault("task_overrides", {})
                    overrides.setdefault(target, {})[key] = raw_value
                maybe_execute()
                continue
            if command == "save":
                if len(parts) != 2:
                    raise ValueError("usage: save DIR")
                _write_workspace(parts[1], documents)
                print(f"saved workspace to {parts[1]}")
                continue
            print(f"unknown command: {command}", file=sys.stderr)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
