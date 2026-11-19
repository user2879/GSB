"""全局常量、数据目录布局与确定性时钟。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

# 受支持的页大小集合(字节)
SUPPORTED_PAGE_SIZES: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
DEFAULT_PAGE_SIZE = 1024
DEFAULT_CAPACITY = 64

# 双写缓冲镜像魔数与定长头部
DWB_MAGIC = b"DWBM"
# magic(4s) page_no(I) page_size(I) seq(Q) length(I) digest(32s) = 56 字节
DWB_HEADER_SIZE = 4 + 4 + 4 + 8 + 4 + 32
DWB_DIGEST_ALGO = "sha256"
DWB_DIGEST_LEN = 32

# 页帧魔数: frame magic(4s) length(I) payload(...)
PAGE_MAGIC = b"PG01"
PAGE_HEADER_SIZE = 8

# 三阶段名称(顺序固定, 渲染/排序均依赖该顺序)
PHASES: tuple[str, ...] = ("buffer_write", "main_overwrite", "commit_mark")

# 固定随机种子(确定性样本)
DEMO_SEED = 20261120


@dataclass(frozen=True)
class DataLayout:
    """一次诊断会话在本地目录中的文件布局。"""

    root: Path

    @property
    def buffer_file(self) -> Path:
        return self.root / "doublewrite.dwb"

    @property
    def main_file(self) -> Path:
        return self.root / "main.pages"

    @property
    def version_map_file(self) -> Path:
        return self.root / "page_versions.json"

    @property
    def views_file(self) -> Path:
        return self.root / "views.json"

    @property
    def markers_file(self) -> Path:
        return self.root / "crash_markers.jsonl"

    @property
    def snapshot_db(self) -> Path:
        return self.root / "sessions.sqlite"

    @property
    def log_file(self) -> Path:
        return self.root / "events.log.jsonl"

    @property
    def pre_recovery_dir(self) -> Path:
        return self.root / "pre_recovery"

    def ensure(self) -> "DataLayout":
        self.root.mkdir(parents=True, exist_ok=True)
        return self


class Clock:
    """单调时钟毫秒封装; 演示模式下可用确定性时钟复现。"""

    def now_ms(self) -> int:
        # time.monotonic 单位为秒, 转毫秒
        return int(round(time.monotonic() * 1000.0))


class DeterministicClock(Clock):
    """固定步进时钟: 每次取值单调递增固定毫秒数, 保证样本可复现。"""

    def __init__(self, start_ms: int = 1_000_000, step_ms: int = 7) -> None:
        self._t = start_ms
        self._step = step_ms

    def now_ms(self) -> int:
        value = self._t
        self._t += self._step
        return value
