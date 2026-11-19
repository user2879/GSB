"""确定性演示：三个场景顺序执行并输出稳定的账本总结。

场景 1: 首次全量失效传播（显式失效 catalog/100，沿注释图闭包传播）。
场景 2: 命名空间刷新 catalog，沿继承链展开下游命名空间，跨层传播。
场景 3: 独立会话写 v1 快照后升级恢复，整体替换并执行重放校验，
        v1 口径缺失按 E_REPLAY 如实报告。
"""

from __future__ import annotations

import json
from pathlib import Path

from .engine import Engine
from .errors import EReplay
from .loader import load_declarations
from .model import EntryID
from .snapshot import write_snapshot
from .store import Store


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_round(record) -> None:
    print(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))


def run_demo(workspace: str | Path,
             namespace_path: str | Path,
             entry_path: str | Path) -> dict:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    graph = load_declarations(namespace_path, entry_path)
    summary: dict = {"scenarios": []}

    _print_header("场景 1：首次全量失效传播")
    store_one = Store(workspace / "scene1.sqlite")
    engine_one = Engine(graph, store=store_one, format_version=3)
    first = engine_one.explicit_invalidate(EntryID("catalog", 100), 1000)
    assert first.round_record is not None
    _print_round(first.round_record)
    engine_one.save_snapshot(workspace / "snapshot_v3.json", 3)
    summary["scenarios"].append({
        "name": "full_propagation",
        "affected": list(first.affected),
        "rounds": [record.to_dict() for record in engine_one.records],
    })
    engine_one.close()

    _print_header("场景 2：命名空间刷新触发跨层传播")
    store_two = Store(workspace / "scene2.sqlite")
    engine_two = Engine(graph, store=store_two, format_version=3)
    refresh = engine_two.refresh_namespace("catalog", 2000)
    assert refresh.round_record is not None
    _print_round(refresh.round_record)
    summary["scenarios"].append({
        "name": "namespace_refresh",
        "affected": list(refresh.affected),
        "rounds": [record.to_dict() for record in engine_two.records],
    })
    engine_two.close()

    _print_header("场景 3：v1 快照升级后恢复与重放")
    store_three = Store(workspace / "scene3_before.sqlite")
    engine_three = Engine(graph, store=store_three, format_version=3)
    record_three = engine_three.explicit_invalidate(
        EntryID("catalog", 100), 3000)
    assert record_three.round_record is not None
    v1_path = workspace / "snapshot_v1.json"
    write_snapshot(v1_path, engine_three._snapshot_graph(),
                   engine_three.records, 1)
    engine_three.close()

    restored_store = Store(workspace / "scene3_restored.sqlite")
    restored, snapshot_data, report, replay_error = Engine.restore(
        v1_path, store=restored_store)
    print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    replay_payload: dict
    if isinstance(replay_error, EReplay):
        replay_payload = replay_error.to_dict()
        print(json.dumps(replay_payload, ensure_ascii=False,
                         sort_keys=True))
    else:
        replay_payload = {"code": "OK", "message": "replay matched"}
    restored.close()
    summary["scenarios"].append({
        "name": "v1_upgrade_restore_replay",
        "snapshot_format_version": snapshot_data.format_version,
        "restore_report": report.to_dict(),
        "replay": replay_payload,
    })

    summary_path = workspace / "ledger_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    print(f"\n账本总结已写入: {summary_path}")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    run_demo(
        root / "artifacts" / "demo",
        root / "samples" / "namespaces.json",
        root / "samples" / "entries.json",
    )


if __name__ == "__main__":
    main()
