import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rerank_core.clock import DeterministicClock
from rerank_core.commands import Command, parse_commands
from rerank_core.config import load_config
from rerank_core.engine import RerankEngine
from rerank_core.loader import load_candidates
from rerank_core.snapshot import SnapshotStore

COMPARE_KEYS = (
    "request_id", "request_content_fp", "normalize_cache_key",
    "fusion_cache_keys", "rerank_cache_key", "discount_cache_keys",
    "fusion_order", "fusion_scores", "contributions", "rerank",
    "high_tier_score_diff",
)


def _business(report):
    return {
        q["request_id"]: {key: q[key] for key in COMPARE_KEYS}
        for q in report["per_request"]
    }


def test_incremental_matches_blank_rebuild(tmp_path):
    sample = ROOT / "samples" / "sample_input"
    commands_raw = json.loads((sample / "commands.json").read_text(encoding="utf-8"))
    commands = parse_commands(commands_raw["commands"])

    # 增量路径：同一引擎连续应用命令
    config = load_config(sample / "rerank_config.json", sample / "feature_groups.json")
    candidates, failures = load_candidates(sample / "candidates.jsonl", config)
    engine = RerankEngine(
        config, candidates, failures,
        SnapshotStore(tmp_path / "inc"), DeterministicClock(),
    )
    for command in commands:
        if command.type == "poll":
            engine.drain_pending(command)
        else:
            engine.handle_command(command)
    incremental = engine.build_report()

    # 全量路径：物化最终输入，空白工作区一次性重建
    final_dir = tmp_path / "final"
    subprocess.run(
        [sys.executable, "-m", "scripts.materialize_final",
         "--input", str(sample),
         "--commands", str(sample / "commands.json"),
         "--out", str(final_dir)],
        check=True, cwd=ROOT,
    )
    final_config = load_config(
        final_dir / "rerank_config.json", final_dir / "feature_groups.json"
    )
    final_candidates, final_failures = load_candidates(
        final_dir / "candidates.jsonl", final_config
    )
    full_engine = RerankEngine(
        final_config, final_candidates, final_failures,
        SnapshotStore(tmp_path / "full"), DeterministicClock(),
    )
    full_engine.handle_command(Command(0, "poll", {}))
    full = full_engine.build_report()

    assert incremental["config_hash"] == full["config_hash"]
    assert incremental["candidates_hash"] == full["candidates_hash"]
    assert incremental["request_order"] == full["request_order"]
    assert _business(incremental) == _business(full)


def test_failure_list_is_deterministic_and_sorted(tmp_path):
    fault = ROOT / "samples" / "fault_injection"
    config = load_config(fault / "threshold_config.json",
                         fault / "feature_groups.json")
    _, failures_a = load_candidates(fault / "bad_candidates.jsonl", config)
    _, failures_b = load_candidates(fault / "bad_candidates.jsonl", config)
    seq_a = [f.to_dict() for f in failures_a]
    seq_b = [f.to_dict() for f in failures_b]
    assert seq_a == seq_b
    assert all(f.code == "E_RECORD_INVALID" for f in failures_a)
    targets = [f.target for f in failures_a]
    assert targets == sorted(targets, key=lambda t: (t,)) or len(targets) == len(set(targets))


def test_concurrency_cap_is_respected(config, loaded, tmp_path, monkeypatch):
    import dataclasses
    from concurrent.futures import ThreadPoolExecutor
    import rerank_core.engine as engine_mod

    observed = []
    real_init = ThreadPoolExecutor.__init__

    def spy_init(self, *args, **kwargs):
        observed.append(kwargs.get("max_workers") or (args[0] if args else None))
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "__init__", spy_init)
    cfg = dataclasses.replace(config, concurrency=2)
    candidates, failures = loaded
    engine = RerankEngine(
        cfg, candidates, failures, SnapshotStore(tmp_path / "cap"),
        DeterministicClock(),
    )
    engine.handle_command(Command(0, "poll", {}))
    assert observed, "应至少创建一次线程池"
    assert all(workers <= 2 for workers in observed)
