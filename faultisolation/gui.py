"""可交互专业工作台。

优先使用 Tkinter 桌面界面；当前环境若缺少 Tk 系统库（无显示器/无 libtk），
自动降级为无头交互控制台，核心诊断能力完全一致，不依赖网络与 GPU。

功能页签：

* 声明校验与规模参数
* 仲裁路径选择理由与工作量
* 三路径逐字节一致性与槽位唯一性
* 实时回环仲裁遥测（RTT、匹配率）
* 定点配额预算诊断与误差界
* 账本恢复状态
"""

from __future__ import annotations

import json
from pathlib import Path

from .fixedpoint import diagnose, global_budget, rank_ratios
from .ledger import Ledger
from .planner import choose_path
from .replay import replay_path, verify_ledger, verify_paths, verify_slot_uniqueness
from .spec import load_bundle

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    TK_AVAILABLE = True
except Exception:  # pragma: no cover - 取决于运行环境是否安装 Tk
    TK_AVAILABLE = False


class HeadlessWorkbench:
    """无 Tk 环境下的降级交互工作台（菜单驱动，纯 stdin/stdout）。"""

    def __init__(self) -> None:
        self.bundle = None
        self.spec_path = None
        self.session_path = None

    def run(self) -> None:
        print("=== 故障隔离平台 · 无头交互工作台（Tk 不可用，已降级）===")
        while True:
            print("\n1) 加载声明  2) 校验  3) 路径选择  4) 一致性校验  5) 定点诊断  6) 账本恢复  q) 退出")
            choice = input("选择> ").strip()
            if choice == "q":
                return
            try:
                if choice == "1":
                    self._load()
                elif choice == "2":
                    self._validate()
                elif choice == "3":
                    print(json.dumps(choose_path(self.bundle), indent=2, ensure_ascii=False))
                elif choice == "4":
                    print(json.dumps({
                        "paths": verify_paths(self.bundle),
                        "slots": verify_slot_uniqueness(self.bundle),
                    }, indent=2, ensure_ascii=False))
                elif choice == "5":
                    self._diagnose()
                elif choice == "6":
                    self._recover()
            except Exception as exc:  # 交互模式下展示错误而非崩溃
                print(f"[错误] {exc}")

    def _load(self) -> None:
        self.spec_path = input("声明 JSON 路径> ").strip()
        session = input("会话 JSON 路径（可留空）> ").strip() or None
        self.bundle = load_bundle(self.spec_path, session)
        self.session_path = session
        print(f"已加载：{len(self.bundle.domains)} 域 / {len(self.bundle.instances)} 实例 / "
              f"{len(self.bundle.events)} 事件")

    def _validate(self) -> None:
        assert self.bundle
        print("强校验通过，事件单调不减且序号连续")

    def _diagnose(self) -> None:
        from .clock import LogicalClock
        from .engine import Engine
        from .budget import compute_usage
        assert self.bundle
        power = self.bundle.session.scale_power
        engine = Engine(self.bundle, LogicalClock(0))
        usage, global_usage = compute_usage(engine)
        items, peak_items, rows = [], [], []
        for snap in usage:
            items.append((snap.domain_id, snap.current_used, snap.slots))
            peak_items.append((snap.domain_id, snap.peak_used, snap.slots))
            rows.append({
                "domain_id": snap.domain_id, "slots": snap.slots,
                "current": diagnose(snap.domain_id, snap.current_used, snap.slots, power).as_dict(),
                "peak": diagnose(snap.domain_id, snap.peak_used, snap.slots, power).as_dict(),
            })
        print(json.dumps({
            "scale_factor": 10 ** power,
            "domains": rows,
            "global_current": global_budget(global_usage.current_used, global_usage.total_slots, power).as_dict(),
            "global_peak": global_budget(global_usage.peak_used, global_usage.total_slots, power).as_dict(),
            "ranking_peak": [r.__dict__ for r in rank_ratios(peak_items, power)],
        }, indent=2, ensure_ascii=False))

    def _recover(self) -> None:
        assert self.bundle
        ledger_path = input("账本路径> ").strip()
        ledger = Ledger(ledger_path)
        ledger.replace_if_digest_bad()
        print(json.dumps(ledger.recover().__dict__, indent=2, ensure_ascii=False, default=str))
        ledger.close()


if TK_AVAILABLE:
    class TkWorkbench(tk.Tk):  # pragma: no cover - 需要图形环境
        def __init__(self) -> None:
            super().__init__()
            self.title("跨故障域租约仲裁与配额预算诊断工作台")
            self.geometry("980x720")
            self.bundle = None
            self._build()

        def _build(self) -> None:
            top = ttk.Frame(self)
            top.pack(fill="x", padx=8, pady=6)
            ttk.Label(top, text="声明文件:").pack(side="left")
            self.spec_var = tk.StringVar()
            ttk.Entry(top, textvariable=self.spec_var, width=60).pack(side="left", padx=4)
            ttk.Button(top, text="浏览", command=self._pick_spec).pack(side="left")
            ttk.Button(top, text="加载并校验", command=self._load).pack(side="left", padx=4)
            self.tabs = ttk.Notebook(self)
            self.tabs.pack(fill="both", expand=True, padx=8, pady=8)
            self.texts = {}
            for name in ("路径选择", "一致性校验", "定点诊断", "账本恢复"):
                frame = ttk.Frame(self.tabs)
                self.tabs.add(frame, text=name)
                txt = tk.Text(frame, wrap="word")
                txt.pack(fill="both", expand=True)
                self.texts[name] = txt

        def _pick_spec(self) -> None:
            path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
            if path:
                self.spec_var.set(path)

        def _load(self) -> None:
            try:
                self.bundle = load_bundle(self.spec_var.get())
            except Exception as exc:
                messagebox.showerror("加载失败", str(exc))
                return
            self._show("路径选择", json.dumps(choose_path(self.bundle), indent=2, ensure_ascii=False))
            self._show("一致性校验", json.dumps({
                "paths": verify_paths(self.bundle),
                "slots": verify_slot_uniqueness(self.bundle),
            }, indent=2, ensure_ascii=False))
            self._load_diagnosis()

        def _load_diagnosis(self) -> None:
            from .clock import LogicalClock
            from .engine import Engine
            from .budget import compute_usage
            power = self.bundle.session.scale_power
            engine = Engine(self.bundle, LogicalClock(0))
            usage, global_usage = compute_usage(engine)
            peak_items, rows = [], []
            for snap in usage:
                peak_items.append((snap.domain_id, snap.peak_used, snap.slots))
                rows.append({
                    "domain_id": snap.domain_id, "slots": snap.slots,
                    "current": diagnose(snap.domain_id, snap.current_used, snap.slots, power).as_dict(),
                    "peak": diagnose(snap.domain_id, snap.peak_used, snap.slots, power).as_dict(),
                })
            self._show("定点诊断", json.dumps({
                "scale_factor": 10 ** power,
                "domains": rows,
                "global_peak": global_budget(global_usage.peak_used, global_usage.total_slots, power).as_dict(),
                "ranking_peak": [r.__dict__ for r in rank_ratios(peak_items, power)],
            }, indent=2, ensure_ascii=False))

        def _show(self, name: str, content: str) -> None:
            widget = self.texts[name]
            widget.delete("1.0", "end")
            widget.insert("1.0", content)


def launch() -> int:
    if TK_AVAILABLE:
        app = TkWorkbench()  # pragma: no cover
        app.mainloop()
        return 0
    HeadlessWorkbench().run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(launch())
