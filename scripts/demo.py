"""端到端演示脚本（命令行启动）。

顺序执行：构造身份摘要 -> 盖章 -> 验章 -> 保存快照 ->
恢复重放 -> 异常注入场景。

用法：
  python scripts/demo.py
  python scripts/demo.py --inject corrupt-spec
  python scripts/demo.py --inject tamper-identity
  python scripts/demo.py --inject bad-snapshot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_stamper.app import (  # noqa: E402
    build_identities,
    recover_and_replay,
    run_threaded_demo,
    run_virtual_demo,
    stamp_all,
    verify_all,
)
from inference_stamper.errors import ITSError  # noqa: E402
from inference_stamper.ledger import (  # noqa: E402
    open_ledger,
    tamper_content_digest,
    tamper_stamp_checksum,
)
from inference_stamper.logging_utils import JsonlLogger  # noqa: E402
from inference_stamper.recovery import ForcedCrash  # noqa: E402
from inference_stamper.spec import load_spec  # noqa: E402
from inference_stamper.stamping import Stamper  # noqa: E402


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description="推理令牌盖章组件演示")
    parser.add_argument(
        "--inject",
        choices=("corrupt-spec", "tamper-identity", "bad-snapshot"),
        help="注入指定异常并验证对应业务码",
    )
    parser.add_argument(
        "--run-dir", default=str(ROOT / "run"),
        help="运行产物目录（账本、日志、备份）",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(run_dir / "demo.jsonl")

    try:
        if args.inject == "corrupt-spec":
            return demo_corrupt_spec(run_dir, logger)
        if args.inject == "tamper-identity":
            return demo_tamper_identity(run_dir, logger)
        if args.inject == "bad-snapshot":
            return demo_bad_snapshot(run_dir, logger)
        return demo_happy_path(run_dir, logger)
    except ITSError as exc:
        payload = exc.to_dict()
        payload.pop("code", None)
        logger.log("ERROR", "injected-failure", code=exc.code, **payload)
        print(f"[失败] 业务码 {exc.code}: {exc.message} "
              f"(标识={exc.item_id or '-'}, 字段={exc.field or '-'})")
        return 2
    finally:
        logger.close()


def demo_corrupt_spec(run_dir: Path, logger: JsonlLogger) -> int:
    print_header("异常注入：声明损坏（期望 E_SPEC）")
    try:
        load_spec(ROOT / "samples" / "spec.corrupt.json")
    except ITSError as exc:
        payload = exc.to_dict()
        payload.pop("code", None)
        logger.log("ERROR", "spec-corrupt", code=exc.code, **payload)
        print(f"[符合预期] {exc.code}: {exc.message} "
              f"(标识={exc.item_id}, 字段={exc.field})")
        return 0
    print("[异常] 损坏声明未被拒绝")
    return 1


def demo_tamper_identity(run_dir: Path, logger: JsonlLogger) -> int:
    print_header("异常注入：身份摘要/盖章校验值被篡改（期望 E_IDENTITY）")
    spec = load_spec(ROOT / "samples" / "spec.base.json")
    db_path = run_dir / "tamper.db"
    if db_path.exists():
        db_path.unlink()
    ledger, _ = open_ledger(db_path)
    stamper = Stamper(ledger, build_identities(spec))
    stamp_all(stamper, spec)
    target = spec.requests[0].token_id
    tamper_stamp_checksum(db_path, target)
    logger.log("WARN", "identity-tampered", code="E_IDENTITY",
               token_id=target, field="checksum")

    result = stamper.verify(target)
    if result.code == "E_IDENTITY":
        logger.log("ERROR", "verify-rejected", code=result.code,
                   token_id=result.token_id, field=result.field,
                   reason=result.reason)
        print(f"[符合预期] {result.code}: {result.reason} "
              f"(标识={result.token_id}, 字段={result.field})")
        ledger.close()
        return 0
    ledger.close()
    print("[异常] 被篡改的盖章未被拒绝")
    return 1


def demo_bad_snapshot(run_dir: Path, logger: JsonlLogger) -> int:
    print_header("异常注入：账本内容摘要不符（期望 E_SNAPSHOT）")
    spec = load_spec(ROOT / "samples" / "spec.base.json")
    db_path = run_dir / "snapshot.db"
    if db_path.exists():
        db_path.unlink()
    ledger, _ = open_ledger(db_path)
    stamper = Stamper(ledger, build_identities(spec))
    stamp_all(stamper, spec)
    ledger.close()

    tamper_content_digest(db_path)
    logger.log("WARN", "snapshot-tampered", code="E_SNAPSHOT")
    try:
        open_ledger(db_path, on_mismatch="raise")
    except ITSError as exc:
        payload = exc.to_dict()
        payload.pop("code", None)
        logger.log("ERROR", "snapshot-mismatch", code=exc.code, **payload)
        print(f"[符合预期] {exc.code}: {exc.message} "
              f"(字段={exc.field})")
        reopened, report = open_ledger(db_path, on_mismatch="replace")
        print(f"[整体替换] 备份={report.backup_path} "
              f"被替换盖章条目={len(report.stamp_token_ids)}")
        logger.log("INFO", "ledger-replaced",
                   backup=report.backup_path,
                   replaced_count=len(report.stamp_token_ids))
        reopened.close()
        return 0
    print("[异常] 摘要不符的账本未被识别")
    return 1


def demo_happy_path(run_dir: Path, logger: JsonlLogger) -> int:
    spec = load_spec(ROOT / "samples" / "spec.base.json")

    print_header("步骤 1：确定性构造参数矩阵与身份摘要")
    identities = build_identities(spec)
    for backend_id, identity in identities.items():
        print(f"\n后端 {backend_id} quant={identity.quant} "
              f"identity_version={identity.identity_version}")
        print(f"  身份摘要: {identity.identity_digest}")
        for layer in identity.layers:
            print(f"  层 {layer.index} {layer.name} 形状={layer.shape} "
                  f"scale={layer.scale:.6f} 摘要={layer.digest[:24]}...")
        print(f"  合成顺序: {list(identity.composition_order)}")
    logger.log("INFO", "identities-built",
               backends=[b.backend_id for b in spec.backends])

    print_header("步骤 2：盖章（含幂等与每后端单调序号）")
    db_path = run_dir / "main.db"
    if db_path.exists():
        db_path.unlink()
    ledger, _ = open_ledger(db_path)
    stamper = Stamper(ledger, identities)
    stamped = stamp_all(stamper, spec)
    for token in stamped:
        print(f"  {token.token_id} backend={token.backend_id} "
              f"stamp_seq={token.stamp_seq} checksum={token.checksum[:16]}...")
    first_request = spec.requests[0]
    repeat = stamper.stamp(first_request)
    print(f"\n  重复盖章 {repeat.token_id}: reused={repeat.reused} "
          f"stamp_seq 保持={repeat.stamp_seq}")
    logger.log("INFO", "stamped", count=len(stamped),
               repeat_reused=repeat.reused)

    print_header("步骤 3：基线验章（应全部通过）")
    results = verify_all(stamper, spec)
    for result in results:
        print(f"  {result.token_id}: {result.code} accepted={result.accepted}")
    assert all(r.accepted for r in results)

    print_header("步骤 4：身份/口径演进后的失效判定（E_STALE）")
    evolved_identity_spec = load_spec(
        ROOT / "samples" / "spec.evolved-identity.json")
    evolved = build_identities(evolved_identity_spec)
    evolved_stamper = Stamper(ledger, evolved)
    stale_identity = evolved_stamper.verify("token-001")
    print(f"  身份版本演进 -> token-001: {stale_identity.code} "
          f"field={stale_identity.field} reason={stale_identity.reason}")
    logger.log("WARN", "stale-identity", code=stale_identity.code,
               token_id=stale_identity.token_id, field=stale_identity.field)

    evolved_spec_spec = load_spec(
        ROOT / "samples" / "spec.evolved-token-spec.json")
    spec_evolved = build_identities(evolved_spec_spec)
    spec_stamper = Stamper(ledger, spec_evolved)
    stale_spec = spec_stamper.verify("token-002")
    print(f"  口径版本演进 -> token-002: {stale_spec.code} "
          f"field={stale_spec.field} reason={stale_spec.reason}")
    logger.log("WARN", "stale-token-spec", code=stale_spec.code,
               token_id=stale_spec.token_id, field=stale_spec.field)

    valid_after = [r for r in verify_all(stamper, spec) if r.accepted]
    print(f"  恢复基线声明后有效令牌数={len(valid_after)}（被拒令牌不计入）")

    print_header("步骤 5：并行调度轨迹（全局=2，每后端=1，虚拟时钟）")
    trace = run_virtual_demo(spec, spec_raw_seed(spec), 2, 1)
    for entry in trace:
        print(f"  {entry.task_id:14s} backend={entry.backend_id:14s} "
              f"wait={entry.wait_ms:3d}ms run=[{entry.run_start_ms},"
              f"{entry.run_end_ms}]")
    verify_budget_invariants(trace, 2, 1)
    print("  预算不变量校验通过：任意时刻全局<=2、单后端<=1")
    logger.log("INFO", "schedule-planned", tasks=len(trace))

    print_header("步骤 5b：真实线程池预算执行（全局=2，每后端=1）")
    threaded_trace = run_threaded_demo(stamper, spec, spec_raw_seed(spec),
                                       global_limit=2, per_backend_limit=1)
    for entry in threaded_trace[:6]:
        print(f"  {entry.task_id:14s} backend={entry.backend_id:14s} "
              f"等待={entry.wait_ms}ms 执行={entry.run_end_ms - entry.run_start_ms}ms")
    print(f"  共完成 {len(threaded_trace)} 个真实线程任务，预算未超限")
    logger.log("INFO", "schedule-threaded", tasks=len(threaded_trace))

    print_header("步骤 6：保存快照并重新打开（内容摘要一致，不替换）")
    ledger.close()
    reopened, report = open_ledger(db_path)
    print(f"  重新打开替换报告: {report}")
    assert report is None

    print_header("步骤 7：模拟强制终止与恢复重放")
    crash_db = run_dir / "crash.db"
    if crash_db.exists():
        crash_db.unlink()
    crash_ledger, _ = open_ledger(crash_db)
    crash_stamper = Stamper(crash_ledger, identities)
    crash_point = spec.requests[3].token_id
    try:
        stamp_all(crash_stamper, spec, crash_after_token_id=crash_point)
    except ForcedCrash as exc:
        print(f"  {exc}")
        logger.log("WARN", "forced-crash", token_id=crash_point)
    report, replay_items = recover_and_replay(crash_ledger, spec)
    print(f"  已提交令牌: {report.committed_token_ids}")
    print(f"  未知/待重盖令牌: {report.unknown_token_ids}")
    print(f"  各后端已用盖章序号: {report.stamp_seq_by_backend}")
    assert not replay_items or all(item.field == "" for item in replay_items)
    print(f"  重放校验通过 {len(replay_items)} 条已提交记录，无 E_REPLAY")
    for request in spec.requests:
        if request.token_id in report.unknown_token_ids:
            crash_stamper.stamp(request)
    report2, replay_items2 = recover_and_replay(crash_ledger, spec)
    print(f"  补盖后未知令牌: {report2.unknown_token_ids}，"
          f"重放记录数={len(replay_items2)}")
    logger.log("INFO", "recovered",
               unknown=list(report.unknown_token_ids))
    crash_ledger.close()
    reopened.close()

    print_header("演示完成")
    print(f"账本目录: {run_dir}")
    print(f"结构化日志: {run_dir / 'demo.jsonl'}")
    return 0


def spec_raw_seed(spec) -> int:
    data = json.loads(Path(spec.source).read_text(encoding="utf-8"))
    return int(data.get("master_seed", 20261123))


def verify_budget_invariants(trace, global_limit: int,
                             per_backend_limit: int) -> None:
    events: list[tuple[int, int, str, str]] = []
    for entry in trace:
        events.append((entry.run_start_ms, 1, entry.backend_id,
                       entry.task_id))
        events.append((entry.run_end_ms, -1, entry.backend_id,
                       entry.task_id))
    events.sort(key=lambda e: (e[0], e[1]))
    active = 0
    backend_active: dict[str, int] = {}
    for _, delta, backend_id, _ in events:
        active += delta
        backend_active[backend_id] = backend_active.get(backend_id, 0) + delta
        assert active <= global_limit
        assert backend_active[backend_id] <= per_backend_limit


if __name__ == "__main__":
    sys.exit(main())
