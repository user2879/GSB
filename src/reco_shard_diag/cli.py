from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .coordinator import ForcedTermination, run_pipeline
from .diagnostics import build_diagnostic_report
from .errors import BizErrorException
from .fixtures import write_workspace
from .models import Event, ProjectConfig
from .state import materialize
from .validation import validate_config


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_events(path: Path) -> list[Event]:
    events: list[Event] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            event, errors = Event.from_raw(raw, index)
            if event is None:
                raise ValueError(errors[0].message if errors else "event invalid")
            events.append(event)
    return events


def load_config(path: Path, events_path: Path | None = None) -> ProjectConfig:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("config root must be object")
    if events_path is not None:
        raw["events"] = [event.to_dict() for event in _read_events(events_path)]
    return ProjectConfig.from_raw(raw)


def _write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    else:
        text = canonical_json(value)
    path.write_text(text + "\n", encoding="utf-8")


def _write_samples(path: Path, state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for shard_id in sorted(state.shards):
            for sample in state.samples(shard_id):
                handle.write(
                    json.dumps(
                        {"shard_id": shard_id, **sample},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )


def _summary_markdown(result: dict[str, Any], report: dict[str, Any]) -> str:
    lines = [
        "# 分片下发诊断结果",
        "",
        f"- 全局版本: `{result.get('metadata', {}).get('global_version')}`",
        f"- 活跃配额版本: `{result.get('metadata', {}).get('active_quota_version')}`",
        f"- 定点位宽: `{result.get('metadata', {}).get('width')}`（参考 `{report.get('conclusion_consistency', {}).get('reference_width')}`）",
        f"- 完成游标: `{', '.join(result.get('completed_cursor', []))}`",
        f"- 已消耗资源单位: `{result.get('budget', {}).get('committed_units')}/{result.get('budget', {}).get('resource_budget')}`",
        f"- 业务码计数: `{json.dumps(result.get('error_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        f"- 注入命中: `{', '.join(report.get('injection_hits', []))}`",
        "",
        "## 分片结论",
        "",
        "| 分片 | 权威回执 | 聚合指纹 | 水位 | 复用判决 |",
        "|---|---|---|---|---|",
    ]
    reuse_by_shard = {
        entry["shard_id"]
        for entry in result.get("reuse_ledger", [])
        if entry.get("decision") == "REUSE"
    }
    invalid_by_shard = {
        entry["shard_id"]
        for entry in result.get("invalidation_ledger", [])
        if entry.get("decision") == "INVALIDATE"
    }
    for shard_id, shard_result in sorted(result.get("per_shard", {}).items()):
        receipt = shard_result.get("receipt") or {}
        aggregate = (shard_result.get("aggregate") or {}).get("summary") or {}
        watermark = shard_result.get("watermark") or {}
        decision = "REUSE" if shard_id in reuse_by_shard and shard_id not in invalid_by_shard else "REVALIDATE"
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                shard_id,
                receipt.get("cache_key", "N/A"),
                aggregate.get("summary_fingerprint", "N/A"),
                watermark.get("state", "N/A"),
                decision,
            )
        )
    lines.extend(
        [
            "",
            "## 归因",
            "",
            f"- 与高位宽参考存在差异的分片: `{', '.join(report.get('conclusion_consistency', {}).get('different_shards', []))}`",
            f"- 差异来源: `{json.dumps(report.get('conclusion_consistency', {}).get('sources', {}), ensure_ascii=False, sort_keys=True)}`",
            "- 账本按 `CURRENT` 原子指针只读取已提交快照；运行中快照标记为 `RUNNING`，重启时归类为强制终止恢复。",
        ]
    )
    return "\n".join(lines) + "\n"


def _execute(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    config_path = Path(args.config).resolve() if args.config else workdir / "config.json"
    if not config_path.exists():
        write_workspace(workdir, force=True)
    events_path = Path(args.events_file).resolve() if args.events_file else None
    try:
        config = load_config(config_path, events_path)
    except BizErrorException as exc:
        _write_json(workdir / "validation.json", {"ok": False, "error": exc.error.to_dict()})
        print(json.dumps(exc.error.to_dict(), ensure_ascii=False, sort_keys=True))
        return 3
    validation = validate_config(config)
    if not validation.ok:
        payload = validation.to_dict()
        _write_json(workdir / "validation.json", payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 3
    state = materialize(config)
    if state.errors:
        payload = {
            "ok": False,
            "state_errors": [error.to_dict() for error in state.errors],
            "state_fingerprint": state.state_fingerprint,
        }
        _write_json(workdir / "state-validation.json", payload)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 3
    mode = "full" if args.fresh else ("resume" if args.resume else "incremental")
    try:
        result = run_pipeline(
            config,
            state,
            workdir,
            mode=mode,
            use_previous=not args.fresh,
            abort_after=args.crash_after,
            checkpoint_on_each_shard=True,
        )
    except ForcedTermination as exc:
        snapshot = workdir / "ledger" / "CURRENT"
        print(
            json.dumps(
                {
                    "forced_termination": True,
                    "completed_count": exc.completed,
                    "snapshot_pointer": str(snapshot),
                    "resume_command": f"reco-shard-diag resume --workdir {workdir}",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        sys.stdout.flush()
        os._exit(137)
    outdir = Path(args.outdir).resolve() if args.outdir else workdir / "artifacts"
    report = build_diagnostic_report(state, result, output_dir=outdir)
    _write_json(outdir / "run.json", result)
    _write_json(outdir / "diagnostic-report.json", report)
    _write_samples(outdir / "samples.jsonl", state)
    node_ledger = result.get("execution", {}).get("node_ledgers", {})
    _write_json(outdir / "node-ledger.json", node_ledger)
    (outdir / "summary.md").write_text(_summary_markdown(result, report), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "phase": result.get("phase"),
                "state_fingerprint": state.state_fingerprint,
                "completed_cursor": result.get("completed_cursor"),
                "error_counts": result.get("error_counts"),
                "budget": result.get("budget"),
                "injection_hits": report.get("injection_hits"),
                "report": str(outdir / "diagnostic-report.json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reco-shard-diag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init", help="生成确定性本地样本和配置")
    init_parser.add_argument("--workdir", default="work/demo")
    init_parser.add_argument("--force", action="store_true")
    init_parser.add_argument("--no-events", action="store_true")
    for name in ("run", "resume"):
        run_parser = subparsers.add_parser(name, help="运行或恢复节点分片链路")
        run_parser.add_argument("--workdir", default="work/demo")
        run_parser.add_argument("--config")
        run_parser.add_argument("--events-file")
        run_parser.add_argument("--outdir")
        run_parser.add_argument("--fresh", action="store_true")
        run_parser.add_argument("--resume", action="store_true")
        run_parser.add_argument("--crash-after", type=int)
    acceptance_parser = subparsers.add_parser("acceptance", help="运行完整验收矩阵")
    acceptance_parser.add_argument("--workdir", default="work/acceptance")
    acceptance_parser.add_argument("--outdir", default="artifacts/acceptance")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        workdir = Path(args.workdir).resolve()
        write_workspace(workdir, include_events=not args.no_events, force=args.force)
        print(
            json.dumps(
                {
                    "ok": True,
                    "workdir": str(workdir),
                    "config": str(workdir / "config.json"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "acceptance":
        from .acceptance import run_acceptance

        return run_acceptance(Path(args.workdir).resolve(), Path(args.outdir).resolve())
    if args.command == "resume":
        args.resume = True
        args.fresh = False
    if args.command == "run" and args.resume:
        args.fresh = False
    return _execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
