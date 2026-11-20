"""全量重建与交互增量两种运行方式的公共装配逻辑。"""

from __future__ import annotations

import json
from pathlib import Path

from rerank_core.clock import MonotonicClock
from rerank_core.commands import Command, parse_commands
from rerank_core.config import load_config
from rerank_core.engine import RerankEngine
from rerank_core.loader import load_candidates
from rerank_core.snapshot import SnapshotStore


def load_workspace(input_dir: str | Path, snapshot_dir: str | Path, *,
                   resume: bool = False, config_name: str = "rerank_config.json"):
    input_dir = Path(input_dir)
    config = load_config(input_dir / config_name, input_dir / "feature_groups.json")
    candidates, failures = load_candidates(input_dir / "candidates.jsonl", config)
    store = SnapshotStore(snapshot_dir)
    engine = RerankEngine(
        config, candidates, failures, store, MonotonicClock(), resume=resume
    )
    return config, engine


def run_full_rebuild(input_dir: str | Path, snapshot_dir: str | Path) -> dict:
    store_path = Path(snapshot_dir)
    if store_path.exists():
        import shutil

        shutil.rmtree(store_path)
    config, engine = load_workspace(input_dir, store_path, resume=False)
    # 从空白工作区一次性全量运行：所有阶段均为重算
    bootstrap = Command(seq=-1, type="poll", payload={})
    response = engine.handle_command(bootstrap)
    engine.flush_snapshot()
    report = engine.build_report()
    report["bootstrap_response"] = response.to_dict()
    return report


def run_interactive(input_dir: str | Path, snapshot_dir: str | Path,
                    commands_path: str | Path) -> dict:
    config, engine = load_workspace(input_dir, snapshot_dir, resume=True)
    raw = json.loads(Path(commands_path).read_text(encoding="utf-8"))
    commands = parse_commands(raw["commands"])
    responses = []
    for command in commands:
        if command.type == "poll":
            response = engine.drain_pending(command)
        else:
            response = engine.handle_command(command)
        responses.append(response.to_dict())
    engine.flush_snapshot()
    report = engine.build_report()
    report["interaction_responses"] = responses
    return report
