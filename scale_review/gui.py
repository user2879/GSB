from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .errors import LeaseError, ScaleReviewError
from .ledger import Ledger
from .portcheck import check_loopback_port
from .propagation import VersionedPlan
from .replay import recover_and_replay
from .spec import load_spec_file
from .workbench import RoundRunner


class WorkbenchApp:
    def __init__(self, root: Any, workspace: Path) -> None:
        workspace = Path(workspace)
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.workspace = Path(workspace)
        self.spec_path = workspace / "configs" / "spec.json"
        self.batch_1 = workspace / "samples" / "request_batches" / "batch-001.json"
        self.batch_2 = workspace / "samples" / "request_batches" / "batch-002.json"
        self.db_path = workspace / "artifacts" / "gui-ledger.sqlite3"
        self.snapshot_path = workspace / "artifacts" / "gui-snapshot.json"
        self.spec = load_spec_file(self.spec_path)
        self.plan = VersionedPlan(self.spec)
        self.ledger: Ledger | None = None
        self.runner: RoundRunner | None = None
        self.last_round: dict[str, Any] | None = None

        root.title("推理副本伸缩复核工作台")
        root.geometry("1180x760")
        frame = ttk.Frame(root, padding=10)
        frame.pack(fill="both", expand=True)

        top = ttk.Frame(frame)
        top.pack(fill="x")
        ttk.Label(top, text=f"Spec: {self.spec_path}").pack(side="left")
        ttk.Button(top, text="重新加载样本", command=self.reload_spec).pack(side="right", padx=4)
        ttk.Button(top, text="回环端口检查", command=self.port_check).pack(side="right", padx=4)

        buttons = ttk.Frame(frame, padding=(0, 10))
        buttons.pack(fill="x")
        button_defs = [
            ("双算法校验", self.audit),
            ("运行第 7 调度轮", self.run_round),
            ("错误版本续约", self.renew_stale),
            ("当前版本续约", self.renew_current),
            ("等待真实 TTL 并裁决", self.schedule_adjudication),
            ("追加批次 1", lambda: self.append_batch(self.batch_1)),
            ("追加批次 2", lambda: self.append_batch(self.batch_2)),
            ("全局预算 6→7", self.increase_budget),
            ("缩放因子→1005", self.change_factor),
            ("快照/恢复校验", self.snapshot_and_replay),
        ]
        for label, command in button_defs:
            ttk.Button(buttons, text=label, command=command).pack(
                side="left", padx=3, pady=3
            )

        self.summary = ttk.Label(
            frame,
            text="就绪：双算法尚未运行；所有结论仅来自本地 JSON/SQLite。",
            anchor="w",
        )
        self.summary.pack(fill="x", pady=(0, 6))

        body = ttk.PanedWindow(frame, orient="horizontal")
        body.pack(fill="both", expand=True)
        self.text = tk.Text(body, wrap="none", font=("Consolas", 10))
        yscroll = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=yscroll.set)
        body.add(self.text, weight=4)
        body.add(yscroll, weight=0)
        self.text.insert("end", "点击“双算法校验”开始演示。\n")
        self.text.configure(state="disabled")

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def log(self, value: Any) -> None:
        self.text.configure(state="normal")
        if isinstance(value, str):
            self.text.insert("end", "\n" + value + "\n")
        else:
            self.text.insert(
                "end",
                "\n" + json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            )
        self.text.see("end")
        self.text.configure(state="disabled")

    def set_summary(self, text: str) -> None:
        self.summary.configure(text=text)

    def reload_spec(self) -> None:
        self.spec = load_spec_file(self.spec_path)
        self.plan = VersionedPlan(self.spec)
        self.log({"loaded": str(self.spec_path), "minutes": self.spec.minute_count})
        self.set_summary("样本已重新加载，方案版本=1")

    def audit(self) -> None:
        result = self.plan.audit_algorithm_paths()
        self.log(result)
        self.set_summary(
            f"双路径一致={result['equal']}；选择={result['selection']['selected_path']}；"
            f"摘要={result['full_digest'][:16]}"
        )

    def _ensure_runner(self) -> RoundRunner:
        if self.ledger is None:
            if self.db_path.exists():
                self.db_path.unlink()
            self.ledger = Ledger(self.db_path)
        if self.runner is None:
            self.runner = RoundRunner(self.plan.spec, self.ledger, self.plan)
        return self.runner

    def run_round(self) -> None:
        runner = self._ensure_runner()
        result = runner.run_round(7)
        self.last_round = result
        compact = {
            "commit": result["commit"],
            "max_observed_inflight": result["max_observed_inflight"],
            "wait_spans": result["wait_spans"],
            "global_held": result["quota_snapshot"]["global_held"],
            "global_remaining": result["quota_snapshot"]["global_remaining"],
            "leases": [lease.to_dict() for lease in result["leases"]],
        }
        self.log(compact)
        self.set_summary(
            f"轮次已提交：任务 {len(result['task_results'])}，"
            f"在途峰值 {result['max_observed_inflight']}，预算余量 "
            f"{result['quota_snapshot']['global_remaining']}"
        )

    def _first_lease(self) -> Any:
        if not self.last_round or not self.last_round["leases"]:
            raise ScaleReviewError("请先运行调度轮", field="round")
        return self.last_round["leases"][0]

    def renew_stale(self) -> None:
        runner = self._ensure_runner()
        lease = self._first_lease()
        try:
            runner.lease_manager.renew(lease.lease_id, 999_999)
        except LeaseError as exc:
            runner.persist_new_lease_events()
            self.log(exc.to_dict())
            self.set_summary("E_LEASE：旧版本续约被拒绝，到期时间未延长")

    def renew_current(self) -> None:
        runner = self._ensure_runner()
        lease = runner.lease_manager.get_lease(self._first_lease().lease_id)
        renewed = runner.lease_manager.renew(lease.lease_id, lease.version)
        runner.persist_new_lease_events()
        self.log(renewed.to_dict())
        self.set_summary(f"续约成功：新版本={renewed.version}")

    def schedule_adjudication(self) -> None:
        if self.runner is None or self.last_round is None:
            self.set_summary("请先运行调度轮")
            return
        ttl = self.spec.lease_policy.lease_ttl_ms / 1000.0 + 0.08
        self.set_summary(f"等待本地单调时钟真实到期：{ttl:.2f}s ...")

        def wait_then_run() -> None:
            time.sleep(ttl)
            self.root.after(0, self.adjudicate)

        import threading

        threading.Thread(target=wait_then_run, daemon=True).start()

    def adjudicate(self) -> None:
        if self.runner is None or self.last_round is None:
            return
        batch_id = self.last_round["payload"]["batch_id"]
        revoked = self.runner.lease_manager.revoke_batch(batch_id)
        expired = self.runner.lease_manager.reap_expired()
        self.runner.persist_new_lease_events()
        snapshot = self.runner.lease_manager.snapshot()
        result = {"revoke_calls": revoked, "expiry_calls": expired, "snapshot": snapshot}
        self.log(result)
        self.set_summary(
            f"裁决完成：预算余量={snapshot['global_remaining']}，"
            f"唯一属主集合大小={len(snapshot['owner_set'])}"
        )

    def append_batch(self, path: Path) -> None:
        batch = json.loads(path.read_text(encoding="utf-8"))["requests"]
        report = self.plan.append_minutes(batch)
        self.log(report)
        self.set_summary(
            f"追加完成：版本={self.plan.version}，受影响分钟={report['affected_minutes']}"
        )

    def increase_budget(self) -> None:
        report = self.plan.update_budget(global_replica_budget=7)
        if self.ledger is not None:
            self.ledger.save_plan_version(self.plan)
        self.log(report)
        self.set_summary(f"预算演进完成：版本={self.plan.version}")

    def change_factor(self) -> None:
        report = self.plan.update_scaling_factor(1005)
        if self.ledger is not None:
            self.ledger.save_plan_version(self.plan)
        self.log(report)
        self.set_summary(f"缩放因子演进完成：版本={self.plan.version}")

    def snapshot_and_replay(self) -> None:
        if self.ledger is None:
            self.ledger = Ledger(self.db_path)
        self.ledger.save_plan_version(self.plan)
        self.ledger.write_snapshot(self.snapshot_path)
        recovery = recover_and_replay(self.ledger, self.spec)
        self.log(
            {
                "snapshot": str(self.snapshot_path),
                "replayed_rounds": len(recovery["replay"]["rounds"]),
                "next_minute": recovery["next_minute"],
                "digest": recovery["replay"]["digest"],
            }
        )
        self.set_summary("快照与重放校验通过；篡改快照时会整体替换")

    def port_check(self) -> None:
        result = check_loopback_port(port=0)
        self.log(result)
        self.set_summary(f"回环端口可用：{result['available_port']}")

    def close(self) -> None:
        if self.ledger is not None:
            self.ledger.close()
        self.root.destroy()


def main() -> int:
    try:
        import tkinter as tk
    except ImportError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "E_TKINTER",
                        "message": "Tkinter is unavailable in this Python installation",
                        "detail": str(exc),
                        "windows_note": "Use the official Python 3.11+ installer which includes Tk/Tcl.",
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    workspace = Path.cwd()
    root = tk.Tk()
    WorkbenchApp(root, workspace)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
