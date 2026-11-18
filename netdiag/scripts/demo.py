"""端到端确定性演示：生成样本 -> 诊断 -> 重放 -> 快照 -> 账本校验。

用法（Windows / Linux 均可）::

    python scripts/demo.py [--port 41793] [--out output/demo]

全程仅使用本地回环与标准库 + NumPy，无外部网络依赖，可重复执行。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from netdiag import samples  # noqa: E402
from netdiag.logging_utils import StructuredLogger  # noqa: E402
from netdiag.pipeline import (persist_session, restore_session,  # noqa: E402
                              run_diagnosis, verify_ledger)
from netdiag.replay import replay_and_verify  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="netdiag 端到端演示")
    parser.add_argument("--port", type=int, default=41793)
    parser.add_argument("--out", default=str(ROOT / "output" / "demo"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(path=out / "demo.jsonl")

    print("== 1. 生成确定性样本 ==")
    meta = samples.generate_workspace(out / "sample")
    wire_path = out / "sample" / "capture.bin"
    event_paths = [out / "sample" / f"events_v{v}.json"
                   for v in (1, 2, 3)]
    print(f"抓包: {wire_path}（{meta['capture']['frame_count']} 帧）")
    print(f"事件清单: {[p.name for p in event_paths]}")

    print("\n== 2. 解码/合并/归因/统计 ==")
    # 账本存相对 workspace 的路径，保证 verify 时可在 workspace 内复现
    import os
    rel_wire = str(wire_path.relative_to(out))
    rel_events = [str(p.relative_to(out)) for p in event_paths]
    cwd = os.getcwd()
    os.chdir(out)
    try:
        session = run_diagnosis(
            wire_paths=[rel_wire],
            event_paths=rel_events,
            window_ms=100, min_samples=4, logger=logger)
    finally:
        os.chdir(cwd)
    for flow, conclusion in session.conclusions.items():
        print(f"{flow}  {conclusion.counts}")
        for gap in conclusion.gaps:
            print(f"  缺口 {gap.start_seq}-{gap.end_seq} -> {gap.cause}")

    print("\n== 3. 回环 TCP 重放校验 ==")
    frames_by_flow = {f: list(o.frames)
                      for f, o in session.model.flows.items()}
    report = replay_and_verify(frames_by_flow, session.conclusions,
                               port=args.port)
    print(f"发送 {report.frames_sent} 帧，对端观测 "
          f"{report.frames_observed} 帧，匹配缺口 {report.matched_gaps} 个")

    print("\n== 4. 快照与账本持久化 ==")
    paths = persist_session(session, out / "snapshot.json",
                            out / "ledger.jsonl", entry_id="demo-1")
    print(paths)

    print("\n== 5. 快照整体替换恢复并追加样本 ==")
    restored, restore_report = restore_session(paths["snapshot"],
                                               logger=logger)
    print(f"版本替换: {restore_report.replaced}，差异 {restore_report.differences}")
    restored.append_capture(wire_path)  # 追加支持绝对路径
    print(f"追加后流数: {len(restored.model.flows)}")

    print("\n== 6. 账本历史结论重放校验 ==")
    verified = verify_ledger(paths["ledger"], workspace=out, logger=logger)
    print(json.dumps(verified, ensure_ascii=False))

    logger.close()
    print(f"\n结构化日志: {out / 'demo.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
