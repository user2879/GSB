from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
PYTHON = Path(sys.executable)


def copy_samples(destination: Path, replacements: list[tuple[str, str]] | None = None) -> Path:
    shutil.copytree(SAMPLES, destination)
    if replacements:
        config_path = destination / "peft_config.yaml"
        config = config_path.read_text(encoding="utf-8")
        for original, replacement in replacements:
            if original not in config:
                raise AssertionError(f"configuration fragment not found: {original!r}")
            config = config.replace(original, replacement)
        config_path.write_text(config, encoding="utf-8")
    return destination


def run_workspace(samples: Path, workspace: Path, base_directory: Path | None = None):
    from peft_lora.pipeline import Pipeline

    pipeline = Pipeline(samples, workspace, base_directory=base_directory)
    return pipeline.run()


def read_report(workspace: Path) -> dict[str, Any]:
    with open(workspace / "report.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def adapter_payloads(report: dict[str, Any], workspace: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for task in report["tasks"]:
        relative = task.get("adapter_export")
        if not relative:
            continue
        with open(workspace / relative, "r", encoding="utf-8") as handle:
            result[task["task_name"]] = json.load(handle)
    return result


def current_manifest(workspace: Path) -> dict[str, Any]:
    with open(workspace / "snapshot" / "CURRENT.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def wait_for(predicate: Callable[[], bool], *, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if predicate():
                return
        except (OSError, ValueError, TypeError):
            pass
        time.sleep(0.05)
    raise AssertionError("condition was not met before timeout")


def cli_command(samples: Path, workspace: Path) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "peft_lora",
        "run",
        "--samples",
        str(samples),
        "--workspace",
        str(workspace),
    ]


def cli_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    for key in (
        "PEFT_TEST_HOLD_STAGE",
        "PEFT_TEST_HOLD_MS",
        "PEFT_TEST_HOLD_AFTER_STAGE",
        "PEFT_TEST_HOLD_AFTER_MS",
    ):
        environment.pop(key, None)
    return environment


def subprocess_signal_flags() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {}


def send_graceful_signal(process: Any) -> None:
    import signal

    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGTERM)
