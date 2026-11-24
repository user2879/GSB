from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state import new_state


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            value = {"raw_value": value}
        value["_source"] = str(path.name if path.parent.name == "." else path.relative_to(path.parents[1]))
        value["_record_index"] = line_number
        records.append(value)
    return records


def load_workspace_dir(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    state = new_state()
    config_path = root / "config.json"
    if config_path.exists():
        loaded_config = json.loads(config_path.read_text(encoding="utf-8"))
        state["config"].update(loaded_config)
    state["global_version"] = int(state["config"].get("global_version", 1))

    workers_path = root / "workers.json"
    if workers_path.exists():
        state["workers"] = json.loads(workers_path.read_text(encoding="utf-8"))

    baseline_path = root / "baseline_worker_versions.json"
    if baseline_path.exists():
        state["baseline_worker_versions"] = json.loads(baseline_path.read_text(encoding="utf-8"))
    else:
        state["baseline_worker_versions"] = {
            worker["worker_id"]: worker.get("param_version", 0)
            for worker in state["workers"]
            if isinstance(worker, dict) and worker.get("worker_id")
        }

    requests: list[dict[str, Any]] = []
    for path in sorted(root.glob("requests*.jsonl")):
        for record in _read_jsonl(path):
            record["_source"] = path.name
            requests.append(record)
    state["requests"] = requests

    receipts: list[dict[str, Any]] = []
    receipt_root = root / "receipts"
    if receipt_root.exists():
        for path in sorted(receipt_root.rglob("*.jsonl")):
            relative = path.relative_to(root)
            for record in _read_jsonl(path):
                record["_source"] = str(relative)
                receipts.append(record)
    state["receipts"] = receipts

    sidecar_path = root / "sidecar.json"
    if sidecar_path.exists():
        state["sidecar"] = json.loads(sidecar_path.read_text(encoding="utf-8"))

    commands_path = root / "commands.json"
    if commands_path.exists():
        state["command_stream"] = json.loads(commands_path.read_text(encoding="utf-8"))
    return state
