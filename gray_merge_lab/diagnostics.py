from __future__ import annotations

import copy
import importlib
import json
import platform
import socket
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import canonical_json
from .errors import LabError
from .replay import authoritative_recompute, replay_batch
from .receipts import decode_receipt
from .snapshot import SnapshotStore, snapshot_document, verify_snapshot_document
from .specs import LoadedSpec


def environment_report(spec: LoadedSpec | None = None) -> dict[str, Any]:
    tkinter_status: dict[str, Any]
    try:
        tkinter = importlib.import_module("tkinter")
        tkinter_status = {"available": True, "version": tkinter.TkVersion}
    except Exception as exc:
        tkinter_status = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    bind_result: dict[str, Any]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        bind_result = {"ok": True, "ephemeral_port": int(sock.getsockname()[1])}
    except OSError as exc:
        bind_result = {"ok": False, "error": str(exc)}
    finally:
        sock.close()

    ports: list[dict[str, Any]] = []
    if spec is not None:
        for backend in spec.backends:
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.bind((backend.host, backend.port))
                ports.append({"backend_id": backend.backend_id, "port": backend.port, "bindable": True})
            except OSError as exc:
                ports.append(
                    {
                        "backend_id": backend.backend_id,
                        "port": backend.port,
                        "bindable": False,
                        "error": str(exc),
                    }
                )
            finally:
                probe.close()
    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "sqlite3": sqlite3.sqlite_version,
        "numpy": np.__version__,
        "tkinter": tkinter_status,
        "loopback_bind": bind_result,
        "declared_backend_ports": ports,
        "network_scope": "127.0.0.1 only",
    }


def _caught(label: str, code: str, operation: Any) -> dict[str, Any]:
    try:
        operation()
    except LabError as exc:
        return {
            "label": label,
            "expected": code,
            "caught": exc.code,
            "ok": exc.code == code,
            "field": exc.field,
            "batch_id": exc.batch_id,
            "message": exc.message,
            "details": exc.details,
        }
    return {"label": label, "expected": code, "caught": None, "ok": False, "message": "no error raised"}


def anomaly_suite(engine: Any) -> list[dict[str, Any]]:
    batches = engine.ledger.batches()
    if not batches:
        raise LabError("E_RUNTIME", "anomaly suite requires at least one ledger batch")
    first = batches[0]
    h3 = next(item for item in engine.ledger.receipt_rows(first["batch_id"]) if item.get("h") == "h3")
    results: list[dict[str, Any]] = []

    corrupt_digest = copy.deepcopy(h3)
    corrupt_digest["output_digest"] = "sha256:damaged"
    results.append(_caught("receipt_digest_corruption", "E_FORMAT", lambda: decode_receipt(corrupt_digest)))

    missing_parameter = copy.deepcopy(h3)
    missing_parameter.pop("backend_param_digest", None)
    results.append(
        _caught("missing_backend_parameter_field", "E_FORMAT", lambda: decode_receipt(missing_parameter))
    )

    unsupported_version = copy.deepcopy(h3)
    unsupported_version["h"] = "h9"
    results.append(
        _caught("unsupported_receipt_version", "E_FORMAT", lambda: decode_receipt(unsupported_version))
    )

    unknown_field = copy.deepcopy(h3)
    unknown_field["legacy_controller_hint"] = "repair-by-v1"
    decoded = decode_receipt(unknown_field)
    results.append(
        {
            "label": "unknown_field_preservation",
            "expected": "preserved",
            "ok": decoded.extras.get("legacy_controller_hint") == "repair-by-v1",
            "extras": decoded.extras,
        }
    )

    payload = engine.payload()
    preview_tampered = copy.deepcopy(payload)
    preview_tampered["preview_batches"][0]["destination"] = "tampered"
    results.append(
        _caught(
            "preview_payload_mismatch",
            "E_PREVIEW",
            lambda: authoritative_recompute(engine.ledger, preview_tampered),
        )
    )

    bad_record = copy.deepcopy(first)
    bad_record["duration_q"] = int(bad_record["duration_q"]) + 1
    results.append(_caught("replay_fixed_point_mismatch", "E_REPLAY", lambda: replay_batch(bad_record)))

    with tempfile.TemporaryDirectory(prefix="gray-merge-anomaly-") as tmp:
        temp_path = Path(tmp) / "snapshot.json"
        document = snapshot_document(engine.manifest(), payload)
        temp_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        tampered = copy.deepcopy(document)
        tampered["content_digest"] = "sha256:" + "0" * 64
        temp_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
        results.append(
            _caught("snapshot_digest_mismatch", "E_SNAPSHOT", lambda: SnapshotStore(temp_path).read_valid())
        )

        mismatch_path = Path(tmp) / "manifest.json"
        store = SnapshotStore(mismatch_path)
        store.write(engine.manifest(), payload)
        changed_manifest = copy.deepcopy(engine.manifest())
        changed_manifest["scale_exp"] = int(changed_manifest["scale_exp"]) + 1
        results.append(
            _caught(
                "snapshot_manifest_replacement",
                "E_SNAPSHOT",
                lambda: store.load_or_replace(changed_manifest, payload),
            )
        )
        replacement = json.loads(mismatch_path.read_text(encoding="utf-8"))
        results[-1]["replacement_verified"] = replacement["manifest"]["scale_exp"] == changed_manifest["scale_exp"]
    return results


def stable_json(value: Any) -> str:
    return canonical_json(value)

