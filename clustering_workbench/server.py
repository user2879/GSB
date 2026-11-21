"""Local JSON API and static-file server for the diagnostics console."""

from __future__ import annotations

import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml

from .codes import code_catalog
from .config import load_config
from .pipeline import WorkbenchPipeline
from .utils import json_safe, read_json


STATIC_ROOT = Path(__file__).resolve().parent / "static"


class WorkbenchServer:
    def __init__(self, config_path: str | Path, host: str = "127.0.0.1", port: int = 8765):
        self.config_path = Path(config_path).expanduser().resolve()
        self.host = host
        self.port = int(port)
        self.run_lock = threading.Lock()

    def make_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "ClusteringWorkbench/1.0"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, value: Any, status: int = HTTPStatus.OK) -> None:
                payload = json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def _send_file(self, path: Path) -> None:
                if not path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                payload = path.read_bytes()
                content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                if path.suffix in {".html", ".css", ".js"}:
                    content_type += "; charset=utf-8"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def _read_json_body(self) -> dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if length <= 0:
                    return {}
                payload = self.rfile.read(length)
                try:
                    value = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"Invalid JSON request: {error}") from error
                return value if isinstance(value, dict) else {}

            def _base_config(self) -> dict[str, Any]:
                config, _, _ = load_config(server.config_path)
                return config

            def do_GET(self) -> None:  # noqa: N802
                try:
                    parsed = urlparse(self.path)
                    if parsed.path in {"/", "/index.html"}:
                        self._send_file(STATIC_ROOT / "index.html")
                        return
                    if parsed.path.startswith("/static/"):
                        relative = unquote(parsed.path[len("/static/") :])
                        candidate = (STATIC_ROOT / relative).resolve()
                        if STATIC_ROOT.resolve() not in candidate.parents and candidate != STATIC_ROOT.resolve():
                            self.send_error(HTTPStatus.FORBIDDEN)
                            return
                        self._send_file(candidate)
                        return
                    if parsed.path == "/api/health":
                        self._send_json({"ok": True, "service": "clustering-diagnostics-workbench"})
                        return
                    if parsed.path == "/api/codes":
                        self._send_json({"codes": code_catalog()})
                        return
                    if parsed.path == "/api/config":
                        raw = yaml.safe_load(server.config_path.read_text(encoding="utf-8"))
                        config = self._base_config()
                        dataset_descriptor = yaml.safe_load(
                            Path(config["paths"]["dataset"]).read_text(encoding="utf-8")
                        )
                        dataset_block = dataset_descriptor.get("dataset", dataset_descriptor)
                        self._send_json(
                            {
                                "config_path": str(server.config_path),
                                "raw": raw,
                                "resolved_output": config["paths"]["outputs"],
                                "base_rows": int(dataset_block.get("rows", 0) or 0),
                                "sample_scale_options": [0.5, 1.0, 1.5, 2.0],
                                "method_names": ["KMeans", "DBSCAN", "GaussianMixture"],
                            }
                        )
                        return
                    if parsed.path == "/api/report":
                        config = self._base_config()
                        report = read_json(Path(config["paths"]["outputs"]) / "latest_report.json")
                        if report is None:
                            self._send_json(
                                {"available": False, "message": "No report yet. Run the workbench first."},
                                HTTPStatus.NOT_FOUND,
                            )
                        else:
                            self._send_json({"available": True, "report": report})
                        return
                    self.send_error(HTTPStatus.NOT_FOUND)
                except BrokenPipeError:
                    return
                except Exception as error:
                    self._send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

            def do_POST(self) -> None:  # noqa: N802
                try:
                    parsed = urlparse(self.path)
                    if parsed.path == "/api/run":
                        body = self._read_json_body()
                        if not server.run_lock.acquire(blocking=False):
                            self._send_json(
                                {"error": "A run is already in progress."},
                                HTTPStatus.CONFLICT,
                            )
                            return
                        try:
                            overrides = build_run_overrides(body)
                            report = WorkbenchPipeline(server.config_path, overrides=overrides).run()
                            self._send_json({"ok": True, "report": report})
                        finally:
                            server.run_lock.release()
                        return
                    if parsed.path == "/api/verify":
                        body = self._read_json_body()
                        if not server.run_lock.acquire(blocking=False):
                            self._send_json(
                                {"error": "A run is already in progress."},
                                HTTPStatus.CONFLICT,
                            )
                            return
                        try:
                            result = verify_serial_parallel(server.config_path, body)
                            self._send_json(result)
                        finally:
                            server.run_lock.release()
                        return
                    self.send_error(HTTPStatus.NOT_FOUND)
                except BrokenPipeError:
                    return
                except Exception as error:
                    self._send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        return Handler

    def serve_forever(self) -> None:
        httpd = ThreadingHTTPServer((self.host, self.port), self.make_handler())
        print(f"Clustering diagnostics workbench: http://{self.host}:{self.port}")
        print(f"Config: {self.config_path}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping workbench server.")
        finally:
            httpd.server_close()


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_run_overrides(body: dict[str, Any]) -> dict[str, Any]:
    method = str(body.get("preprocess_method", "standard"))
    if method not in {"standard", "quantile"}:
        method = "standard"
    sample_scale = min(4.0, max(0.05, _as_float(body.get("sample_scale"), 1.0)))
    max_workers = min(32, max(1, _as_int(body.get("max_workers"), 4)))
    global_budget = max(1, _as_int(body.get("global_budget"), 150))
    candidate_quota = max(1, _as_int(body.get("candidate_quota"), 18))
    cache_blocks = max(1, _as_int(body.get("cache_blocks"), 8))
    snapshot_version = max(1, _as_int(body.get("snapshot_version"), 1))
    overrides: dict[str, Any] = {
        "preprocessing": {"method": method},
        "data": {"sample_scale": sample_scale},
        "scheduler": {
            "max_workers": max_workers,
            "global_iteration_budget": global_budget,
            "per_candidate_iteration_quota": candidate_quota,
            "cache_resident_blocks": cache_blocks,
        },
        "snapshot": {"version": snapshot_version},
    }
    if body.get("injections_enabled") is False:
        overrides["injections"] = []
    return overrides


def verify_serial_parallel(config_path: str | Path, body: dict[str, Any]) -> dict[str, Any]:
    import tempfile

    base_overrides = build_run_overrides(body)
    serial_dir = tempfile.mkdtemp(prefix="cw-verify-serial-")
    parallel_dir = tempfile.mkdtemp(prefix="cw-verify-parallel-")
    serial_overrides = json.loads(json.dumps(base_overrides))
    parallel_overrides = json.loads(json.dumps(base_overrides))
    serial_overrides["scheduler"]["max_workers"] = 1
    parallel_overrides["scheduler"]["max_workers"] = max(2, _as_int(body.get("max_workers"), 4))
    serial = WorkbenchPipeline(config_path, overrides=serial_overrides, output_dir=serial_dir).run()
    parallel = WorkbenchPipeline(config_path, overrides=parallel_overrides, output_dir=parallel_dir).run()
    ranking_equal = serial.get("ranking") == parallel.get("ranking")
    labels_equal = [row.get("labels") for row in serial.get("candidates", [])] == [
        row.get("labels") for row in parallel.get("candidates", [])
    ]
    budget_equal = serial.get("scheduler", {}).get("budget_log") == parallel.get("scheduler", {}).get(
        "budget_log"
    )
    failures_equal = serial.get("failures") == parallel.get("failures")
    return {
        "ok": True,
        "serial_digest": serial.get("determinism_digest"),
        "parallel_digest": parallel.get("determinism_digest"),
        "digest_equal": serial.get("determinism_digest") == parallel.get("determinism_digest"),
        "ranking_equal": ranking_equal,
        "labels_equal": labels_equal,
        "budget_equal": budget_equal,
        "failures_equal": failures_equal,
        "serial_completion_order": serial.get("scheduler", {}).get("completion_order"),
        "parallel_completion_order": parallel.get("scheduler", {}).get("completion_order"),
        "serial_dir": serial_dir,
        "parallel_dir": parallel_dir,
    }
