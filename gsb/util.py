from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FIXED_SCALE = 1_000_000
INT64_NULL = int(np.iinfo(np.int64).min)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def canonical_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=json_default,
        ) + "\n"
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=json_default,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = canonical_json(list(parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def canonical_row_order(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty(0, dtype=np.intp)
    if len(arrays[0]) == 0:
        return np.empty(0, dtype=np.intp)
    keys: list[np.ndarray] = []
    for array in arrays:
        if np.issubdtype(array.dtype, np.floating):
            keys.append(np.nan_to_num(array, nan=np.inf, posinf=np.inf, neginf=-np.inf))
        else:
            keys.append(array)
    return np.lexsort(tuple(reversed(keys)))


def _array_bytes(array: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    canonical = np.ascontiguousarray(array)
    if canonical.dtype.byteorder == ">" or (
        canonical.dtype.byteorder == "=" and not np.little_endian
    ):
        canonical = canonical.astype(canonical.dtype.newbyteorder("<"))
    return canonical.dtype.str, canonical.shape, canonical.tobytes(order="C")


def logical_content_digest(
    arrays: Mapping[str, np.ndarray],
    *,
    names: Sequence[str] | None = None,
    row_order: np.ndarray | None = None,
) -> str:
    ordered_names = list(names) if names is not None else sorted(arrays)
    if not ordered_names:
        return sha256_bytes(b"empty")
    count = int(len(arrays[ordered_names[0]]))
    order = canonical_row_order([arrays[name] for name in ordered_names])
    digest = hashlib.sha256()
    digest.update(b"GSB-CONTENT-v1\0")
    digest.update(str(count).encode("ascii"))
    for name in ordered_names:
        dtype, shape, payload = _array_bytes(arrays[name][order])
        digest.update(b"\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(dtype.encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(part) for part in shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def digest_arrays(arrays: Mapping[str, np.ndarray], *, names: Sequence[str] | None = None) -> str:
    return logical_content_digest(arrays, names=names)


def write_npz_deterministic(path: str | os.PathLike[str], arrays: Mapping[str, np.ndarray]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(
                array_buffer,
                np.ascontiguousarray(arrays[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 0
            info.external_attr = 0o100644 << 16
            archive.writestr(info, array_buffer.getvalue())
    data = buffer.getvalue()
    atomic_write_bytes(target, data)
    return sha256_bytes(data)


def read_npz(path: str | os.PathLike[str], expected_sha256: str | None = None) -> dict[str, np.ndarray]:
    source = Path(path)
    if not source.is_file():
        from .errors import GSBError

        raise GSBError("E_SAMPLE", "样本文件不存在", {"path": str(source)})
    actual = sha256_file(source)
    if expected_sha256 is not None and actual != expected_sha256:
        from .errors import GSBError

        raise GSBError(
            "E_SAMPLE",
            "样本文件摘要不匹配",
            {"path": str(source), "expected_sha256": expected_sha256, "actual_sha256": actual},
        )
    loaded: dict[str, np.ndarray] = {}
    try:
        with np.load(source, allow_pickle=False) as npz:
            for name in npz.files:
                loaded[name] = np.array(npz[name], copy=True)
    except Exception as exc:
        from .errors import GSBError

        raise GSBError(
            "E_SAMPLE",
            "样本文件损坏或不可读取",
            {"path": str(source), "reason": str(exc)},
        ) from exc
    return loaded


def finite_float(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite value: {value!r}")
    return float(value)


def json_number(value: Any) -> int | float:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    return value


def sort_key_value(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, np.integer)):
        return (0, int(value))
    if isinstance(value, (float, np.floating)):
        return (1, float(value))
    return (2, str(value))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def iter_json_records(path: str | os.PathLike[str]) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        yield from payload
    else:
        yield payload
