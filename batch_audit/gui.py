"""Tkinter local workbench for generating, running, and auditing a workspace."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

from .audit import AuditMismatch, run_audit
from .constants import DEFAULT_OUTPUT_DIM, DEFAULT_SUPPORTED_SLOTS, NUMBERING_VERSION
from .demo import create_demo_workspace
from .engine import RunOptions, Scheduler, WorkspacePaths
from .feed import FeedError


class BatchAuditApp:
    def __init__(self, root: tk.Tk, workspace: Path) -> None:
        self.root = root
        self.root.title("确定性批次归属审计台")
        self.root.geometry("1180x760")
        self.workspace_var = tk.StringVar(value=str(Path(workspace).resolve()))
        self.seed_var = tk.StringVar(value="20261120")
        self.samples_var = tk.StringVar(value="12")
        self.input_dim_var = tk.StringVar(value="6")
        self.hidden_dim_var = tk.StringVar(value="8")
        self.output_dim_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIM))
        self.slots_var = tk.StringVar(value=",".join(str(item) for item in DEFAULT_SUPPORTED_SLOTS))
        self.numbering_var = tk.StringVar(value=NUMBERING_VERSION)
        self._build()

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        form = ttk.LabelFrame(outer, text="本地工作区", padding=10)
        form.pack(fill="x")
        ttk.Label(form, text="目录").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(form, textvariable=self.workspace_var, width=74).grid(
            row=0, column=1, columnspan=5, sticky="ew", padx=4, pady=3
        )
        ttk.Button(form, text="选择…", command=self._choose_workspace).grid(
            row=0, column=6, padx=4, pady=3
        )

        fields = [
            ("随机种子", self.seed_var),
            ("样本数", self.samples_var),
            ("输入维度", self.input_dim_var),
            ("隐藏维度", self.hidden_dim_var),
            ("输出维度", self.output_dim_var),
            ("槽位(逗号分隔)", self.slots_var),
            ("编号口径", self.numbering_var),
        ]
        for index, (label, variable) in enumerate(fields):
            row = 1 + index // 3
            col = (index % 3) * 2
            ttk.Label(form, text=label).grid(row=row, column=col, sticky="w", padx=4, pady=3)
            ttk.Entry(form, textvariable=variable, width=22).grid(
                row=row, column=col + 1, sticky="ew", padx=4, pady=3
            )
        for column in (1, 3, 5):
            form.columnconfigure(column, weight=1)

        actions = ttk.Frame(outer, padding=(0, 10))
        actions.pack(fill="x")
        ttk.Button(actions, text="生成演示工作区", command=self._init_demo).pack(side="left", padx=4)
        ttk.Button(actions, text="执行 / 恢复调度", command=self._run).pack(side="left", padx=4)
        ttk.Button(actions, text="独立复算审计", command=self._audit).pack(side="left", padx=4)
        ttk.Button(actions, text="查看进度", command=self._status).pack(side="left", padx=4)
        ttk.Button(actions, text="清空输出", command=self._clear).pack(side="right", padx=4)

        self.output = ScrolledText(outer, wrap="word", font=("Consolas", 10))
        self.output.pack(fill="both", expand=True)
        self._append(
            "工作台就绪。生成演示工作区后可执行调度；强制终止后的重启会从已提交槽位继续。"
        )

    def _workspace(self) -> Path:
        return Path(self.workspace_var.get()).expanduser().resolve()

    def _choose_workspace(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(self._workspace()))
        if selected:
            self.workspace_var.set(selected)

    def _slots(self) -> tuple[int, ...]:
        values = tuple(int(part.strip()) for part in self.slots_var.get().split(",") if part.strip())
        if not values:
            raise ValueError("槽位不能为空")
        return values

    def _init_demo(self) -> None:
        try:
            create_demo_workspace(
                self._workspace(),
                seed=int(self.seed_var.get()),
                sample_count=int(self.samples_var.get()),
                input_dim=int(self.input_dim_var.get()),
                hidden_dim=int(self.hidden_dim_var.get()),
                output_dim=int(self.output_dim_var.get()),
                supported_slots=self._slots(),
            )
            self._append(
                f"已生成确定性工作区：{self._workspace()}\n"
                "feed.json 已乱序，含同槽重投、准入失败与空槽位。"
            )
        except Exception as exc:
            self._show_error(exc)

    def _run(self) -> None:
        try:
            paths = WorkspacePaths.under(self._workspace())
            report = Scheduler(
                paths,
                RunOptions(
                    supported_slots=self._slots(),
                    numbering_version=self.numbering_var.get().strip(),
                ),
            ).run()
            summary = {
                "status": report["status"],
                "generation": report["generation"],
                "progress": report["progress"],
                "recovery": report["recovery"],
                "slots": [
                    {
                        "slot": item["batch_slot"],
                        "status": item["status"],
                        "members": len(item["members"]),
                        "error_code": item["error_code"],
                    }
                    for item in report["slots"]
                ],
                "unassigned_samples": report["unassigned_samples"],
                "report": str(paths.report),
            }
            self._append(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        except Exception as exc:
            self._show_error(exc)

    def _audit(self) -> None:
        try:
            paths = WorkspacePaths.under(self._workspace())
            result = run_audit(ledger_path=paths.ledger, feed_path=paths.feed)
            self._append(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        except AuditMismatch as exc:
            self._append(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True, indent=2))
        except Exception as exc:
            self._show_error(exc)

    def _status(self) -> None:
        try:
            paths = WorkspacePaths.under(self._workspace())
            snapshot = paths.snapshot.read_text(encoding="utf-8") if paths.snapshot.exists() else "{}"
            self._append(snapshot)
        except Exception as exc:
            self._show_error(exc)

    def _show_error(self, exc: Exception) -> None:
        if isinstance(exc, (FeedError, AuditMismatch)):
            payload: Any = exc.as_dict()
        else:
            payload = {"code": getattr(exc, "code", "E_STATE"), "message": str(exc)}
        self._append(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        messagebox.showerror("操作失败", str(payload))

    def _clear(self) -> None:
        self.output.delete("1.0", "end")

    def _append(self, text: str) -> None:
        self.output.insert("end", text.rstrip() + "\n\n")
        self.output.see("end")


def main(workspace: Path | None = None) -> int:
    root = tk.Tk()
    BatchAuditApp(root, workspace or Path("workspace"))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
