"""Tkinter 界面冒烟测试（有显示时真实建窗，无显示则跳过）。"""

import os

import pytest

from edge_cache.spec import load_spec
from edge_cache.ui import DiagnosticUI

try:
    import tkinter as tk
except ImportError:
    pytest.skip("tkinter 不可用", allow_module_level=True)


def test_ui_renders_read_only(tmp_path):
    if not os.environ.get("DISPLAY"):
        pytest.skip("无 DISPLAY，跳过 Tk 建窗测试")
    import tkinter as tk_
    from edge_cache.ledger import Ledger
    from edge_cache.spec import spec_digest
    # 复用集成产物或现场建最小账本
    spec = {
        "session": {"session_id": "ui", "global_lease_limit": 1,
                    "global_byte_budget": 256, "session_scale": 1000},
        "nodes": [{"node_id": "n1", "host": "127.0.0.1", "port": 19401,
                   "capacity_slots": 1, "per_node_lease_limit": 1,
                   "node_seed": 1, "quota_version": 1}],
        "contents": [{"content_key": "c", "total_bytes": 256,
                      "shard_count": 1, "content_seed": 1}],
        "events": [{"seq": 1, "type": "write", "node_id": "n1",
                    "content_key": "c", "shard_index": 0, "ttl_ms": 100}]}
    spec_path = tmp_path / "ui_spec.json"
    import json
    spec_path.write_text(json.dumps(spec))
    db = tmp_path / "ui.sqlite3"
    parsed = load_spec(spec_path)
    ledger = Ledger(db, parsed, spec_digest(parsed))
    ledger.close()

    root = tk_.Tk()
    try:
        ui = DiagnosticUI(root, spec_path, db)
        ui.refresh()
        root.update_idletasks()
        root.update()
        text = ui.meta_var.get()
        assert "账本格式版本" in text
        rows = ui.tree.get_children()
        assert rows
    finally:
        root.destroy()
