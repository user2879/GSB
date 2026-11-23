from __future__ import annotations

import json
from pathlib import Path

from walcrash.storage import Storage

from .conftest import run_cli


def test_three_cli_injections_return_business_codes_and_ndjson(tmp_path: Path) -> None:
    workspace = tmp_path / "injection-workspace"
    expectations = {
        "page_size": ("E_PAGE", 10, "size_bytes"),
        "order": ("E_ORDER", 11, "lsn"),
        "snapshot_digest": ("E_SNAPSHOT", 13, "content_digest"),
    }
    for mode, (code, exit_code, field) in expectations.items():
        result = run_cli(
            "--workspace",
            str(workspace),
            "--inject",
            mode,
            expect=exit_code,
        )
        payload = json.loads(result.stdout)
        assert payload["code"] == code
        assert payload["context"]["field"] == field
        assert payload["injection"] == mode
    log_path = Storage(workspace).logs_dir / "injections.ndjson"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 3
    assert {record["code"] for record in records} == {"E_PAGE", "E_ORDER", "E_SNAPSHOT"}
    assert all(record["injection"] in expectations for record in records)


def test_gui_module_imports_without_loading_tk(tmp_path: Path) -> None:
    import walcrash.gui

    assert callable(walcrash.gui.launch_gui)

