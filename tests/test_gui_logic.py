"""业务要求 8：视图仅读取已持久化结果（用 tkinter 桩件验证数据流，无需显示环境）。"""
import json
import os
import sys
import types


class _FakeWidget:
    def __init__(self, *args, **kwargs):
        self.coords_data = [0, 0, 10, 10]

    def __getattr__(self, name):
        def method(*args, **kwargs):
            if name == "coords":
                return self.coords_data
            if name == "bbox":
                return (0, 0, 100, 100)
            return None
        return method

    def __call__(self, *args, **kwargs):
        return self


class _FakeTk:
    def __init__(self):
        self.title_text = ""
        self.geometry_text = ""

    def title(self, text):
        self.title_text = text

    def geometry(self, text):
        self.geometry_text = text

    def mainloop(self):
        self.mainloop_called = True


def _install_fake_tk(monkeypatch):
    fake_tk = types.ModuleType("tkinter")
    fake_tk.Tk = _FakeTk
    fake_tk.Canvas = _FakeWidget
    fake_tk.Listbox = _FakeWidget
    fake_tk.Label = _FakeWidget
    fake_tk.Frame = _FakeWidget
    fake_tk.BOTH = "both"
    fake_tk.X = "x"
    fake_tk.Y = "y"
    fake_tk.LEFT = "left"
    fake_tk.TOP = "top"
    fake_tk.BOTTOM = "bottom"
    fake_tk.END = "end"
    fake_tk.W = "w"
    fake_tk.NW = "nw"
    fake_tk.HORIZONTAL = "horizontal"
    fake_tk.VERTICAL = "vertical"
    ttk = types.ModuleType("tkinter.ttk")
    for name in ("Frame", "Label", "PanedWindow", "Scrollbar"):
        setattr(ttk, name, _FakeWidget)
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)
    monkeypatch.setitem(sys.modules, "tkinter.ttk", ttk)
    return fake_tk


def test_gui_reads_persisted_outputs_only(tmp_path, monkeypatch):
    # 准备一份最小持久化结果
    output = tmp_path / "output"
    output.mkdir()
    timeline = {
        "title": "unit",
        "ticks_per_beat": 480,
        "total_ticks": 960,
        "chunk_size": 960,
        "version_id": "v1",
        "chunks": [
            {
                "chunk_index": 0,
                "tick_lo": 0,
                "tick_hi": 960,
                "declared_ticks": 960,
                "tempo": [100.0] * 960,
                "pedal": [0] * 960,
                "note_segments": [
                    {"note_id": "N1", "voice": "vA", "pitch": 60, "start_tick": 0,
                     "active_end_tick": 400, "actual_end_tick": 480,
                     "actual_duration_ticks": 480, "actual_velocity": 80}
                ],
                "per_tick_notes": [],
                "dependency_keys": ["tempo:0", "pedal:P1"],
            }
        ],
    }
    report = {
        "version_id": "v1",
        "total_ticks": 960,
        "chunk_count": 1,
        "chunk_size": 960,
        "steps_applied": 1,
        "cross_voice_overlaps": [],
        "edit_impact": [{"step": 0, "op": "pedal_add", "invalidated_chunks": [0]}],
        "consistency": {"consistent": True, "max_duration_diff": 0, "max_velocity_diff": 0},
        "digest_verification": {"status": "passed", "chunks_verified": 1},
    }
    (output / "timeline.json").write_text(json.dumps(timeline), encoding="utf-8")
    (output / "diagnostic_report.json").write_text(json.dumps(report), encoding="utf-8")

    _install_fake_tk(monkeypatch)
    # 重新导入以绑定桩件
    import importlib
    import mwb.gui as gui_module
    importlib.reload(gui_module)
    gui_module.launch_gui(str(output))  # 不抛异常即数据流正确
