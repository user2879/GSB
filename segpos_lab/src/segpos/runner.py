"""命令行入口：全量运行、交互命令流、快照恢复与诊断报告。"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from .errors import E_INTERACT_BUDGET_PENDING, E_SNAPSHOT_RECOVERED, Failure
from .loader import (
    load_charset,
    load_commands,
    load_config,
    load_lexicon,
    load_prob_table,
    load_sentences,
)
from .engine import IncrementalEngine
from .models import Config


def _resolve_table_ref(command: dict, base_dir: Path) -> dict:
    table_ref = command.get("table")
    if isinstance(table_ref, str) and table_ref.startswith("@"):
        path = base_dir / table_ref[1:]
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    return table_ref


def prepare_bundle(sample_dir: str | Path, config_override: dict | None = None):
    base = Path(sample_dir)
    charset = load_charset(base / "charset.txt")
    config, config_failures = load_config(base / "config.json")
    if config is None:
        return None, config_failures
    if config_override:
        config = Config(**{**config.__dict__, **config_override})
    table, table_failures = load_prob_table(base / "prob_table.json", config.smoothing)
    if table is None:
        return None, table_failures + config_failures
    lexicon, lexicon_failures = load_lexicon(base / "lexicon.txt", set(table.tags))
    sentences, sentence_failures = load_sentences(base / "sentences.txt", charset)
    failures = config_failures + table_failures + lexicon_failures + sentence_failures
    return {
        "config": config,
        "table": table,
        "lexicon": lexicon,
        "sentences": sentences,
        "charset": charset,
        "failures": failures,
        "base": base,
    }, failures


def build_engine(bundle: dict, snapshot_dir: str | Path) -> IncrementalEngine:
    return IncrementalEngine(
        config=bundle["config"],
        sentences=bundle["sentences"],
        lexicon=bundle["lexicon"],
        prob_table=bundle["table"],
        charset=bundle["charset"],
        failures=bundle["failures"],
        snapshot_dir=snapshot_dir,
    )


def _materialize_commands(commands: list[dict], bundle: dict) -> tuple[list[dict], list[Failure]]:
    failures: list[Failure] = []
    materialized: list[dict] = []
    base = bundle["base"]
    for command in commands:
        command = dict(command)
        if command.get("type") == "replace_prob_table":
            raw = _resolve_table_ref(command, base)
            tmp_path = base / ".resolved_table.json"
            tmp_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            table, table_failures = load_prob_table(tmp_path, bundle["config"].smoothing)
            tmp_path.unlink(missing_ok=True)
            if table is None or table_failures:
                failures.extend(table_failures)
                command["table"] = None
            else:
                command["table"] = table
        materialized.append(command)
    return materialized, failures


def run_full(sample_dir: str, snapshot_dir: str, report_path: str) -> dict:
    bundle, _ = prepare_bundle(sample_dir)
    if bundle is None:
        raise SystemExit("bundle preparation failed")
    engine = build_engine(bundle, snapshot_dir)
    response = engine.run_due(set(engine.states))
    engine.global_version = 1
    engine.commit_snapshot(budget_consumed=response["code"] == E_INTERACT_BUDGET_PENDING)
    report = engine.build_report(0)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    from .snapshot import atomic_write_json

    atomic_write_json(Path(report_path), report)
    return report


def run_interactive(sample_dir: str, snapshot_dir: str, report_path: str, config_override: dict | None = None) -> dict:
    bundle, _ = prepare_bundle(sample_dir, config_override)
    if bundle is None:
        raise SystemExit("bundle preparation failed")
    commands, command_failures = load_commands(bundle["base"] / "commands.json")
    materialized, resolve_failures = _materialize_commands(commands, bundle)
    engine = build_engine(bundle, snapshot_dir)
    engine.failures.extend(command_failures)
    engine.failures.extend(resolve_failures)

    terminate = {"flag": False}

    def _handler(signum, frame):  # pragma: no cover - 信号路径由测试模拟
        terminate["flag"] = True

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    budget_consumed = False
    for command in materialized:
        if terminate["flag"]:
            engine.commit_snapshot(budget_consumed)
            sys.exit(0)
        affected = engine.apply_command(command)
        engine.global_version += 1
        response = engine.run_due(affected)
        budget_consumed = budget_consumed or response["code"] == E_INTERACT_BUDGET_PENDING
        engine.commands_processed += 1
        if engine.commands_processed % engine.config.snapshot_commit_interval_commands == 0:
            engine.commit_snapshot(budget_consumed)
    engine.commit_snapshot(budget_consumed)
    report = engine.build_report(len(materialized))
    from .snapshot import atomic_write_json

    atomic_write_json(Path(report_path), report)
    return report


def run_restore(sample_dir: str, snapshot_dir: str, report_path: str) -> dict:
    bundle, _ = prepare_bundle(sample_dir)
    if bundle is None:
        raise SystemExit("bundle preparation failed")
    engine = build_engine(bundle, snapshot_dir)
    if engine.recovery_version is None:
        raise SystemExit("no recoverable snapshot found")
    report = engine.build_report(engine.commands_processed)
    report["recovery_code"] = E_SNAPSHOT_RECOVERED
    from .snapshot import atomic_write_json

    atomic_write_json(Path(report_path), report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="segpos interactive segmenter/tagger")
    parser.add_argument("mode", choices=["full", "interactive", "restore"])
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    if args.mode == "full":
        run_full(args.sample_dir, args.snapshot_dir, args.report)
    elif args.mode == "interactive":
        run_interactive(args.sample_dir, args.snapshot_dir, args.report)
    else:
        run_restore(args.sample_dir, args.snapshot_dir, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
