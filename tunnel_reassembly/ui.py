"""Tkinter 诊断界面。

拆分为两层：
  build_view_model  纯函数，从 Session 结果生成稳定排序的视图数据，可无显示环境单测；
  DiagnosticView    Tk 渲染层，tkinter 不可用时导入即降级，调用 render() 给出明确提示。

展示内容（渲染顺序全部稳定）：通道列表与窗口状态、偏差估计版本与折算偏差、
翻转记录、槽位占用与丢弃组、业务码、快照列表。
"""

from __future__ import annotations

from .model import ReassemblyResult

try:  # Windows 标准安装勾选 Tkinter 后可用；Linux 精简环境可能缺失。
    import tkinter as tk
    from tkinter import ttk

    TK_AVAILABLE = True
except Exception:  # pragma: no cover - 环境相关
    tk = None
    ttk = None
    TK_AVAILABLE = False


BUSINESS_CODES = [
    ("E_CHANNEL", "通道声明/脚本强校验失败，整体拒绝"),
    ("E_ESTIMATE", "偏差样本不足，估计未定，拒绝开窗"),
    ("E_DIGEST", "重组摘要与登记值不一致"),
    ("E_SNAPSHOT", "快照版本或内容摘要不符，整体替换"),
    ("E_REPLAY", "恢复重放与账本逐项比对不一致"),
    ("E_PROTOCOL", "数据报片头/长度/CRC 自洽校验失败"),
]


def build_view_model(session) -> dict:
    """从 Session 结果生成稳定排序的只读视图模型。"""
    channels = []
    for cid in sorted(session.results):
        result: ReassemblyResult = session.results[cid]
        spec = next(s for s in session.specs if s.channel_id == cid)
        estimates = [
            {
                "version": v.version,
                "skew_ms": v.skew_ms,
                "samples": v.sample_count,
                "qerror_ms": v.qerror_ms,
            }
            for v in result.estimates
            if v.sample_count > 0
        ]
        windows = [
            {"seq": e.window_seq, "action": e.action, "base_ms": e.base_ms, "close_ms": e.close_ms}
            for e in result.window_events
        ]
        flips = [
            {
                "group": f.group_seq,
                "from_version": f.from_version,
                "to_version": f.to_version,
                "kind": f.flip_kind,
                "from_window": f.from_window,
                "to_window": f.to_window,
            }
            for f in result.flips
        ]
        drops = [
            {"group": o.group_seq, "status": o.status, "window": o.window_seq, "reason": o.reason}
            for o in result.outcomes
            if o.status in ("SLOT_DROPPED", "OFFSET_DROPPED")
        ]
        outcomes = [
            {"group": o.group_seq, "status": o.status, "window": o.window_seq}
            for o in result.outcomes
        ]
        channels.append(
            {
                "channel_id": cid,
                "host": spec.host,
                "port": spec.port,
                "slot_count": spec.slot_count,
                "max_inflight_ms": spec.max_inflight_ms,
                "window_ms": spec.window_ms,
                "estimates": estimates,
                "windows": windows,
                "flips": flips,
                "drops": drops,
                "outcomes": outcomes,
            }
        )
    return {
        "channels": channels,
        "business_codes": list(BUSINESS_CODES),
        "snapshots": session.ledger.list_snapshots(),
    }


class DiagnosticView:
    """Tk 窗口；无 tkinter 时 render() 退化为打印视图模型。"""

    def __init__(self, view_model: dict) -> None:
        self.view_model = view_model

    def render(self) -> None:
        if not TK_AVAILABLE:
            print("当前 Python 环境未安装 tkinter，已降级为文本视图（Windows 安装时勾选 Tcl/Tk 即可开窗）：")
            for channel in self.view_model["channels"]:
                print(f"\n[{channel['channel_id']}] {channel['host']}:{channel['port']} "
                      f"slots={channel['slot_count']} inflight={channel['max_inflight_ms']}ms")
                print("  估计版本:", channel["estimates"])
                print("  窗口:", channel["windows"])
                print("  翻转:", channel["flips"])
                print("  丢弃:", channel["drops"])
                print("  结局:", channel["outcomes"])
            print("\n业务码:", self.view_model["business_codes"])
            print("快照:", self.view_model["snapshots"])
            return
        self._build_window()

    def _build_window(self) -> None:  # pragma: no cover - 需要图形环境
        root = tk.Tk()
        root.title("隧道重组窗口裁决诊断")
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        for channel in self.view_model["channels"]:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=channel["channel_id"])
            text = tk.Text(frame, wrap="none")
            text.pack(fill="both", expand=True)
            text.insert("end", f"通道 {channel['channel_id']} @ {channel['host']}:{channel['port']}\n")
            text.insert("end", f"槽位={channel['slot_count']} 最大在途偏移={channel['max_inflight_ms']}ms "
                                f"窗口宽度={channel['window_ms']}ms\n\n")
            text.insert("end", "偏差估计版本:\n")
            for est in channel["estimates"]:
                text.insert("end", f"  v{est['version']} skew={est['skew_ms']}ms "
                                    f"samples={est['samples']} qerror<={est['qerror_ms']}ms\n")
            text.insert("end", "\n窗口开闭:\n")
            for win in channel["windows"]:
                text.insert("end", f"  窗口{win['seq']} {win['action']} base={win['base_ms']} close={win['close_ms']}\n")
            text.insert("end", "\n翻转记录:\n")
            for flip in channel["flips"]:
                text.insert("end", f"  组{flip['group']} {flip['kind']} v{flip['from_version']}->"
                                    f"v{flip['to_version']} 窗口{flip['from_window']}->{flip['to_window']}\n")
            text.insert("end", "\n槽位/偏移丢弃:\n")
            for drop in channel["drops"]:
                text.insert("end", f"  组{drop['group']} {drop['status']} {drop['reason']}\n")
            text.insert("end", "\n组结局:\n")
            for outcome in channel["outcomes"]:
                text.insert("end", f"  组{outcome['group']} {outcome['status']} 窗口{outcome['window']}\n")
            text.configure(state="disabled")

        code_frame = ttk.Frame(notebook)
        notebook.add(code_frame, text="业务码")
        code_text = tk.Text(code_frame)
        code_text.pack(fill="both", expand=True)
        for code, desc in self.view_model["business_codes"]:
            code_text.insert("end", f"{code}: {desc}\n")
        code_text.configure(state="disabled")

        snap_frame = ttk.Frame(notebook)
        notebook.add(snap_frame, text="快照")
        snap_text = tk.Text(snap_frame)
        snap_text.pack(fill="both", expand=True)
        for snap in self.view_model["snapshots"]:
            snap_text.insert("end", json_line(snap) + "\n")
        snap_text.configure(state="disabled")

        root.mainloop()


def json_line(data: dict) -> str:
    import json

    return json.dumps(data, sort_keys=True, ensure_ascii=False)
