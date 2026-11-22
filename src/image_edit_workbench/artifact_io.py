from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .errors import E_SNAPSHOT_CORRUPT, WorkbenchError
from .util import canonical_json_bytes, sha256_bytes


ARTIFACT_MAGIC = b"IEWBART1\n"


def pack_artifacts(arrays: dict[str, np.ndarray], metadata: dict | None = None) -> bytes:
    entries: list[dict[str, object]] = []
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        dtype = array.dtype.newbyteorder("<")
        payload = array.tobytes(order="C")
        chunks.append(payload)
        entries.append(
            {
                "name": name,
                "dtype": dtype.str,
                "shape": list(array.shape),
                "offset": offset,
                "length": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
        offset += len(payload)
    header = {
        "format": "image-edit-workbench-artifact/v1",
        "arrays": entries,
        "metadata": metadata or {},
    }
    header_bytes = canonical_json_bytes(header)
    return ARTIFACT_MAGIC + len(header_bytes).to_bytes(8, "big") + header_bytes + b"".join(chunks)


def unpack_artifacts(data: bytes) -> tuple[dict[str, np.ndarray], dict]:
    if not data.startswith(ARTIFACT_MAGIC) or len(data) < len(ARTIFACT_MAGIC) + 8:
        raise WorkbenchError(E_SNAPSHOT_CORRUPT, "bad artifact magic or header", stage="artifact")
    start = len(ARTIFACT_MAGIC)
    header_len = int.from_bytes(data[start : start + 8], "big")
    header_start = start + 8
    header_end = header_start + header_len
    if header_end > len(data):
        raise WorkbenchError(E_SNAPSHOT_CORRUPT, "artifact header is truncated", stage="artifact")
    try:
        header = json.loads(data[header_start:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkbenchError(E_SNAPSHOT_CORRUPT, f"bad artifact header JSON: {exc}", stage="artifact") from exc
    payload = data[header_end:]
    arrays: dict[str, np.ndarray] = {}
    try:
        for entry in header["arrays"]:
            offset = int(entry["offset"])
            length = int(entry["length"])
            raw = payload[offset : offset + length]
            if len(raw) != length or sha256_bytes(raw) != entry["sha256"]:
                raise ValueError(f"payload digest mismatch for {entry['name']}")
            array = np.frombuffer(raw, dtype=np.dtype(entry["dtype"]).newbyteorder("<")).reshape(entry["shape"]).copy()
            arrays[str(entry["name"])] = array
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkbenchError(E_SNAPSHOT_CORRUPT, f"invalid artifact entry: {exc}", stage="artifact") from exc
    return arrays, header.get("metadata", {})


def write_artifact(path: Path, arrays: dict[str, np.ndarray], metadata: dict | None = None) -> tuple[str, int]:
    from .util import atomic_write_bytes

    data = pack_artifacts(arrays, metadata)
    atomic_write_bytes(path, data)
    return sha256_bytes(data), len(data)


def read_artifact(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    return unpack_artifacts(path.read_bytes())

