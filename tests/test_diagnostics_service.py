from __future__ import annotations

import json
import stat
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from acoustic_workbench.engine import IncrementalEngine
from acoustic_workbench.errors import E_SNAPSHOT_CONFLICT
from acoustic_workbench.service import DiagnosticService
from acoustic_workbench.webapp import create_server


def test_snapshot_is_immutable_and_service_has_no_recompute(workspace: Path) -> None:
    engine = IncrementalEngine.open(workspace)
    service = DiagnosticService.from_workspace(workspace)
    assert service.snapshot.revision == engine.scene["revision"]
    assert service.snapshot.json_bytes == service.snapshot_json()
    with pytest.raises(TypeError):
        service.snapshot.data["revision"] = 999
    assert not hasattr(service, "rebuild")
    assert not hasattr(service, "submit")
    snapshot_path = Path(service.snapshot.source_path)
    if snapshot_path.stat().st_mode & stat.S_IWUSR:
        pytest.skip("platform does not enforce read-only snapshot mode bits")
    assert service.projection_png().startswith(b"\x89PNG\r\n\x1a\n")


def test_http_dashboard_is_get_only_and_serves_snapshot(workspace: Path) -> None:
    service = DiagnosticService.from_workspace(workspace)
    server = create_server(service, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        with urlopen(base + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
        assert response.status == 200
        assert "read-only snapshot" in html
        with urlopen(base + "/api/snapshot", timeout=5) as response:
            body = response.read()
        assert body == service.snapshot_json()
        with urlopen(base + "/api/projection.png", timeout=5) as response:
            assert response.read().startswith(b"\x89PNG")
        request = Request(base + "/api/snapshot", data=b"{}", method="POST")
        with pytest.raises(HTTPError) as caught:
            urlopen(request, timeout=5)
        assert caught.value.code == 405
        assert json.loads(caught.value.read())["code"] == E_SNAPSHOT_CONFLICT
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
