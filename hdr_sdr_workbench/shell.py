"""交互式工作台（纯 stdin 命令行，Windows/Linux 均可运行）。

命令::

    timeline [n]              显示时间轴前 n 帧缩略图与版本戳
    curves                    显示峰值/平均亮度 ASCII 曲线
    state [frame]             显示某帧（或全部）管线参数与状态 id
    set <帧区间> <k=v> ...    逐帧或区间调整，如 set 0-9 exposure_stops=0.5
    closure <帧区间> <k=v>... 仅计算失效闭包规模，不落地修改
    rebuild <version> [frames] 重放历史版本并实算重建，校验哈希
    verify                    全量哈希 + 实算校验
    help / exit
"""

from __future__ import annotations

import json

import numpy as np

from .display import create_display
from .errors import WorkbenchError
import os

from .state import DEFAULT_STATE


def parse_range(text: str, frame_count: int) -> list[int]:
    frames = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            frames.extend(range(int(lo), int(hi) + 1))
        elif part:
            frames.append(int(part))
    out = []
    for i in frames:
        if not (0 <= i < frame_count):
            raise WorkbenchError("E_STATE_VERSION", f"帧号越界: {i}")
        if i not in out:
            out.append(i)
    return out


def parse_changes(tokens: list[str]) -> dict:
    changes = {}
    for token in tokens:
        key, _, value = token.partition("=")
        key = key.strip()
        if key not in DEFAULT_STATE:
            raise WorkbenchError("E_TONEMAP_PARAM", f"未知参数: {key}")
        if key in ("colorspace", "tonemap", "oetf"):
            changes[key] = value.strip()
        else:
            changes[key] = float(value)
    return changes


class InteractiveShell:
    def __init__(self, engine, input_stream=None, output_stream=None):
        self.engine = engine
        self.inp = input_stream
        self.out = output_stream
        self.display = create_display()

    def log(self, text: str = "") -> None:
        if self.out is not None:
            self.out.write(text + "\n")

    def current_version(self) -> int:
        return len(self.engine.state_table.operations) - 1

    def render_timeline(self, limit: int = 12) -> str:
        n = min(limit, self.engine.frame_count)
        rendered = []
        for index in range(n):
            entry = self.engine.frame_entries.get(index)
            if entry is None:
                rendered.append(f"frame {index:03d}: <未计算>")
                continue
            u8 = self._load_u8(index)
            self.display.upload(index, u8)
            rendered.append(
                f"frame {index:03d}: v{entry['version']:06d} "
                f"{entry['output_hash'][:10]}")
        header = f"[backend={self.display.backend}] 时间轴 (版本戳 {self.current_version()})"
        return header + "\n" + self.display.render_timeline(list(range(n))) + \
            "\n" + "\n".join(rendered)

    def _load_u8(self, index: int) -> np.ndarray:
        entry = self.engine.frame_entries[index]
        path = os.path.join(self.engine.store.paths["outputs"], entry["output_file"])
        cfg = self.engine.config
        return np.frombuffer(open(path, "rb").read(), dtype=np.uint8).reshape(
            cfg.height, cfg.width, 3)

    def render_curves(self) -> str:
        truth = {f["frame"]: f for f in self.engine.source_manifest["frames"]}
        peaks = [truth[i]["peak_luminance"] for i in range(self.engine.frame_count)]
        means = [truth[i]["mean_luminance"] for i in range(self.engine.frame_count)]

        def ascii_curve(values, width: int = 48) -> list[str]:
            vmax = max(values) or 1.0
            lines = []
            for i, value in enumerate(values):
                bar = "#" * max(1, int(round(value / vmax * width)))
                lines.append(f"{i:03d} |{bar} {value:.4f}")
            return lines

        return ("峰值亮度 (线性 BT.2020 Y)\n" + "\n".join(ascii_curve(peaks)) +
                "\n\n平均亮度 (线性 BT.2020 Y)\n" + "\n".join(ascii_curve(means)))

    def render_state(self, frame: int | None = None) -> str:
        indexes = [frame] if frame is not None else range(self.engine.frame_count)
        lines = [f"当前版本戳: {self.current_version()}"]
        for i in indexes:
            state = self.engine.state_table.states[i]
            lines.append(f"frame {i:03d} id={state.state_id()[:12]} "
                         f"{json.dumps(state.to_dict(), sort_keys=True)}")
        return "\n".join(lines)

    def handle(self, line: str) -> str:
        parts = line.strip().split()
        if not parts:
            return ""
        cmd = parts[0]
        try:
            if cmd in ("help", "?"):
                return __doc__
            if cmd == "timeline":
                limit = int(parts[1]) if len(parts) > 1 else 12
                return self.render_timeline(limit)
            if cmd == "curves":
                return self.render_curves()
            if cmd == "state":
                frame = int(parts[1]) if len(parts) > 1 else None
                return self.render_state(frame)
            if cmd in ("set", "closure"):
                frames = parse_range(parts[1], self.engine.frame_count)
                changes = parse_changes(parts[2:])
                if cmd == "closure":
                    closure = self.engine.state_table.compute_closure(frames, changes)
                    return f"失效闭包规模={len(closure)} 帧={closure}"
                result = self.engine.apply_operation(frames, changes, note="interactive")
                return (f"已提交 v{result['version']} 闭包={result['closure_size']} "
                        f"帧={result['closure']}")
            if cmd == "rebuild":
                version = int(parts[1])
                frames = parse_range(parts[2], self.engine.frame_count) \
                    if len(parts) > 2 else None
                rebuilt = self.engine.rebuild_historical(version, frames)
                return f"历史重建 v{version} 完成，{len(rebuilt)} 帧实算哈希校验通过"
            if cmd == "verify":
                return json.dumps(self.engine.verify_all(), ensure_ascii=False)
            return f"未知命令: {cmd}（输入 help 查看用法）"
        except WorkbenchError as exc:
            return f"ERROR {exc.code}: {exc.message}"

    def run(self) -> None:
        assert self.inp is not None and self.out is not None
        self.log("HDR->SDR 工作台（输入 help 查看命令，exit 退出）")
        for line in self.inp:
            line = line.strip()
            if line in ("exit", "quit"):
                break
            if line:
                self.log(self.handle(line))
        self.display.close()
