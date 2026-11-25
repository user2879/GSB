"""命令行入口：样本生成、校验、运行、重放、恢复与租约演示。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import run_pipeline
from .errors import SloError
from .lease_client import EvaluatorProcess
from .loader import load_batch_directory, load_service_specs
from .recovery import recover
from .replay import verify_ledger
from .sample import generate_workspace
from .scheduler import WindowScheduler


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slo-burn", description="本地 SLO 错误预算燃烧率诊断")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="生成确定性本地样本")
    sample.add_argument("workspace")

    validate = sub.add_parser("validate", help="强校验声明与批次（E_SPEC 定位）")
    validate.add_argument("workspace")

    run = sub.add_parser("run", help="执行双路径判定并提交账本")
    run.add_argument("workspace")
    run.add_argument("--ledger", required=True)
    run.add_argument("--global-limit", type=int, default=3)
    run.add_argument("--per-service-limit", type=int, default=2)
    run.add_argument("--no-scheduler", action="store_true")

    replay = sub.add_parser("replay", help="独立重放校验账本（E_REPLAY）")
    replay.add_argument("workspace")
    replay.add_argument("--ledger", required=True)

    recovery = sub.add_parser("recover", help="强制终止后恢复")
    recovery.add_argument("--ledger", required=True)

    lease = sub.add_parser("lease-demo", help="租约签发/续约/到期/裁决演示")
    lease.add_argument("--socket", required=True)
    lease.add_argument("--global-limit", type=int, default=2)
    lease.add_argument("--per-service-limit", type=int, default=1)

    dashboard = sub.add_parser("dashboard", help="启动 Tk 仪表盘（需 SLO_DISPLAY）")
    dashboard.add_argument("workspace")
    dashboard.add_argument("--ledger", required=True)
    return parser


def _cmd_validate(workspace: str) -> dict:
    specs = load_service_specs(Path(workspace) / "services.json")
    regular, late = load_batch_directory(Path(workspace) / "batches", specs)
    return {
        "ok": True,
        "services": [spec.service_id for spec in specs],
        "regular_batches": len(regular),
        "late_batches": len(late),
    }


def _cmd_lease_demo(socket_path: str, global_limit: int, per_service_limit: int) -> dict:
    from .arbitration import OwnershipEvent, RecycleReason
    from .lease_client import LeaseClient
    import time

    with EvaluatorProcess(socket_path, global_limit, per_service_limit) as process:
        with LeaseClient(socket_path) as client:
            first = client.grant("svc-a", ttl_ms=80)["lease"]
            second = client.grant("svc-b", ttl_ms=80)["lease"]
            # 错误版本续约：必须 E_LEASE 且不延长到期
            mismatch = None
            try:
                client.renew(first["lease_id"], first["version"] + 99, ttl_ms=500)
            except SloError as exc:
                mismatch = exc.to_dict()
            # 正确版本续约
            renewed = client.renew(first["lease_id"], first["version"], ttl_ms=300)["lease"]
            # 等租约自然到期后回收
            time.sleep(0.12)
            reaped = client.reap()
            # 撤销第二份租约
            revoked = client.revoke(second["lease_id"], second["version"])["lease"]
            events = [
                OwnershipEvent(
                    first["granted_ms"], "svc-a", first["slot_id"],
                    first["lease_id"], first["version"], RecycleReason.GRANT, 1,
                ),
                OwnershipEvent(
                    reaped["at_ms"], "svc-a", first["slot_id"],
                    first["lease_id"], first["version"], RecycleReason.EXPIRY, 2,
                ),
                OwnershipEvent(
                    renewed["granted_ms"], "svc-a", first["slot_id"],
                    first["lease_id"], first["version"], RecycleReason.REVOKE, 3,
                ),
            ]
            from .arbitration import OwnershipArbiter

            arbiter = OwnershipArbiter()
            decisions = arbiter.adjudicate_all(events)
            return {
                "ok": True,
                "mismatch_rejected": mismatch,
                "renewed_expires_ms": renewed["expires_ms"],
                "reaped": reaped["reaped"],
                "revoked_state": revoked["state"],
                "arbitration": [decision.__dict__ for decision in decisions],
                "final_slots": arbiter.slot_owners(),
            }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "sample":
            path = generate_workspace(args.workspace)
            _print({"ok": True, "workspace": str(path)})
        elif args.command == "validate":
            _print(_cmd_validate(args.workspace))
        elif args.command == "run":
            if args.no_scheduler:
                report = run_pipeline(args.workspace, args.ledger, scheduler=None)
            else:
                socket_path = str(Path(args.ledger).with_suffix(".sock"))
                with EvaluatorProcess(socket_path, args.global_limit, args.per_service_limit) as process:
                    from .lease_client import LeaseClient

                    with LeaseClient(socket_path) as client:
                        scheduler = WindowScheduler(
                            client, args.global_limit, args.per_service_limit
                        )
                        report = run_pipeline(
                            args.workspace, args.ledger, scheduler=scheduler
                        )
            _print(report)
        elif args.command == "replay":
            edits = Path(args.workspace) / "param_edits.json"
            _print(
                verify_ledger(
                    args.ledger,
                    Path(args.workspace) / "services.json",
                    Path(args.workspace) / "batches",
                    edits if edits.exists() else None,
                )
            )
        elif args.command == "recover":
            _print(recover(args.ledger).__dict__)
        elif args.command == "lease-demo":
            _print(_cmd_lease_demo(args.socket, args.global_limit, args.per_service_limit))
        elif args.command == "dashboard":
            from .ui import launch_dashboard

            launch_dashboard(args.workspace, args.ledger)
        else:
            parser.error("unknown command")
    except SloError as exc:
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
