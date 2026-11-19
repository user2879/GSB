"""Tk 无关的只读诊断模型: 收集界面所需的稳定顺序数据。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .buffer import DoubleWriteBuffer
from .config import DataLayout
from .snapshot import SnapshotStore
from .util import StructuredLogger
from .versions import MarkerLog, VersionMap
from .paging import MainDataFile


def collect_model(root: Path | str, page_size: int, capacity: int) -> dict[str, Any]:
    layout = DataLayout(Path(root))
    model: dict[str, Any] = {"root": str(layout.root), "tabs": {}}

    # 缓冲镜像(按缓冲偏移稳定排序)
    dwb = DoubleWriteBuffer(layout.buffer_file, capacity)
    mirrors: list[dict[str, Any]] = []
    buffer_error: dict[str, Any] | None = None
    try:
        scan = dwb.scan(strict_digest=True)
        mirrors = [m.to_row() for m in scan.mirrors]
    except Exception as exc:  # noqa: BLE001
        buffer_error = exc.to_dict() if hasattr(exc, "to_dict") else {"code": "E_PAGE", "message": str(exc)}
        try:
            scan = dwb.scan(strict_digest=False)
            mirrors = [m.to_row() for m in scan.mirrors]
        except Exception as exc2:  # noqa: BLE001
            buffer_error = exc2.to_dict() if hasattr(exc2, "to_dict") else {"message": str(exc2)}
    model["tabs"]["buffer"] = {"mirrors": mirrors, "error": buffer_error}

    # 主文件页摘要(页号升序)
    main = MainDataFile(layout.main_file, page_size, capacity)
    versions = VersionMap.load(layout.version_map_file)
    page_rows: list[dict[str, Any]] = []
    for page_no in range(capacity):
        frame = main.read_slot(page_no)
        if frame is None:
            continue
        from .paging import Page

        actual = Page.digest_hex(frame)
        expected = versions.get(page_no)
        page_rows.append(
            {
                "page_no": page_no,
                "digest": actual,
                "expected": expected.digest if expected else None,
                "intact": expected is None or actual == expected.digest,
            }
        )
    model["tabs"]["main"] = {"pages": page_rows}

    # 崩溃点阶段(按文件顺序)
    markers = MarkerLog.load(layout.markers_file)
    model["tabs"]["markers"] = {"markers": [m.to_dict() for m in markers.markers]}

    # 业务码事件(日志顺序)
    codes: list[dict[str, Any]] = []
    if layout.log_file.exists():
        for line in layout.log_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            import json

            event = json.loads(line)
            if event.get("code"):
                codes.append(event)
    model["tabs"]["codes"] = {"events": codes}

    # 快照列表
    snaps: list[dict[str, Any]] = []
    if layout.snapshot_db.exists():
        snaps = SnapshotStore(layout.snapshot_db, StructuredLogger(Path(root) / ".gui-ignore.log")).list_snapshots()
    model["tabs"]["snapshots"] = {"snapshots": snaps}
    return model
