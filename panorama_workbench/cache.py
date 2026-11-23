from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np


STATE_SCHEMA = 1


def encode_array(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    payload = zlib.compress(contiguous.tobytes(), level=9)
    return {
        "data": base64.b64encode(payload).decode("ascii"),
        "dtype": contiguous.dtype.str,
        "encoding": "zlib+base64",
        "shape": list(contiguous.shape),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    if payload.get("encoding") != "zlib+base64":
        raise ValueError("unsupported array encoding")
    data = zlib.decompress(base64.b64decode(payload["data"]))
    array = np.frombuffer(data, dtype=np.dtype(payload["dtype"]))
    return array.reshape(tuple(payload["shape"])).copy()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, payload: Any) -> None:
    serialized = json.dumps(
        payload, sort_keys=True, indent=2, ensure_ascii=False
    ).encode("utf-8")
    atomic_write_bytes(path, serialized + b"\n")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def payload_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def checksummed(payload: Any, schema: int = STATE_SCHEMA) -> dict[str, Any]:
    body = {"schema": schema, "payload": payload}
    body["checksum"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def verify_checksummed(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("document root must be an object")
    checksum = document.get("checksum")
    body = {"schema": document.get("schema"), "payload": document.get("payload")}
    expected = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if not isinstance(checksum, str) or checksum != expected:
        raise ValueError("document checksum mismatch")
    if not isinstance(body["payload"], dict):
        raise ValueError("document payload must be an object")
    return body["payload"]


class StateStore:
    def __init__(self, state_dir: str | Path) -> None:
        self.root = Path(state_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"

    def load(self) -> tuple[dict[str, Any] | None, str | None]:
        if not self.state_path.exists():
            return None, None
        try:
            document = json.loads(self.state_path.read_text(encoding="utf-8"))
            return verify_checksummed(document), None
        except Exception as exc:
            return None, str(exc)

    def save(self, state: dict[str, Any]) -> None:
        atomic_write_json(self.state_path, checksummed(state))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
